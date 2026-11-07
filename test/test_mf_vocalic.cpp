#include "PipelineRecorder.hpp"
#include "VocalicGrammar.hpp"
#include "VocalicShaders.hpp"

#include <MayaFlux/Portal/Graphics/ShaderFoundry.hpp>

#include <MayaFlux/Buffers/BufferManager.hpp>
#include <MayaFlux/Vruta/Scheduler.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <random>

using namespace MayaFlux;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

constexpr uint32_t k_rate = 48000;
constexpr uint32_t k_block = 512;

struct Rig {
    std::shared_ptr<Buffers::BufferManager> manager = std::make_shared<Buffers::BufferManager>(2, 1, k_rate, k_block);
    Vruta::TaskScheduler scheduler { k_rate };
    uint64_t phase = 0;

    // One audio cycle the way the engine runs it: input first, then pipeline tasks.
    void cycle(const std::vector<double>& in)
    {
        auto copy = in;
        manager->process_input(copy.data(), 1, k_block);
        scheduler.process_buffer_cycle_tasks();
    }

    std::vector<double> sine_block()
    {
        std::vector<double> b(k_block);
        for (auto& x : b) {
            x = 0.5 * std::sin(2.0 * std::numbers::pi * 440.0 * static_cast<double>(phase++) / k_rate);
        }
        return b;
    }
};

double sine_at(uint64_t n)
{
    return 0.5 * std::sin(2.0 * std::numbers::pi * 440.0 * static_cast<double>(n) / k_rate);
}

void test_take_length()
{
    Rig rig;
    Kriya::PipelineRecorder rec(rig.manager, rig.scheduler, { .max_seconds = 0.25 });

    check(rec.start(), "start a take");
    check(!rec.start(), "second start refused while recording");

    // Every other block is silent, as the engine delivers it.
    const std::vector<double> zeros(k_block, 0.0);
    for (int i = 0; i < 200 && rec.recording(); ++i) {
        rig.cycle(rig.sine_block());
        rig.cycle(zeros);
    }

    check(!rec.recording(), "take ends at max_seconds");
    check(rec.takes() == 1, "one take finished");

    auto take = rec.last_take();
    check(take != nullptr, "take is available");
    if (!take) {
        return;
    }

    const auto frames = take->get_num_frames();
    std::printf("      frames %llu (expected %u)\n", static_cast<unsigned long long>(frames), k_rate / 4);
    check(frames == k_rate / 4, "take is 0.25 s long, silent blocks skipped");

    std::vector<double> out(frames);
    take->read_frames(out, frames);
    double err = 0.0;
    for (uint64_t n = 0; n < frames; ++n) {
        err = std::max(err, std::abs(out[n] - sine_at(n)));
    }
    std::printf("      max error %.2e\n", err);
    check(err < 1e-9, "take holds the sine without gaps");
}

void test_stop_and_repeat()
{
    Rig rig;
    Kriya::PipelineRecorder rec(rig.manager, rig.scheduler, { .max_seconds = 2.0 });

    rec.start();
    for (int i = 0; i < 20; ++i) {
        auto b = rig.sine_block();
        rig.cycle(b);
        rig.cycle(b); // a stale listener repeats the block
    }
    const double before = rec.seconds();
    rec.stop();
    rig.cycle(rig.sine_block());

    check(!rec.recording(), "stop ends the take at the next block");
    auto take = rec.last_take();
    check(take && take->get_num_frames() > 0, "stopped take kept");
    if (take) {
        std::printf("      %.3f s, %llu frames\n", before, static_cast<unsigned long long>(take->get_num_frames()));
        check(take->get_num_frames() <= 20 * k_block, "repeated blocks skipped");
    }

    check(rec.start(), "a new take can start");
    for (int i = 0; i < 10; ++i) {
        rig.cycle(rig.sine_block());
    }
    rec.stop();
    rig.cycle(rig.sine_block());
    check(rec.takes() == 2, "two takes finished");
}


// ---- vowel / consonant split ------------------------------------------------------------------

// silence 0.3 s, vowel 0.4 s, noise 0.2 s, vowel 0.4 s, silence 0.2 s
struct Speech {
    std::vector<double> x;
    std::vector<std::pair<double, int>> parts { { 0.3, 0 }, { 0.4, 1 }, { 0.2, 2 }, { 0.4, 1 }, { 0.2, 0 } };

    Speech()
    {
        std::mt19937 rng(7);
        std::normal_distribution<double> noise(0.0, 0.15);
        uint64_t n = 0;
        for (auto [seconds, kind] : parts) {
            const auto count = static_cast<uint64_t>(seconds * k_rate);
            for (uint64_t k = 0; k < count; ++k, ++n) {
                double v = 0.0;
                if (kind == 1) {
                    // 130 Hz with ten harmonics, louder near 700 Hz as an open vowel is.
                    for (int h = 1; h <= 10; ++h) {
                        const double f = 130.0 * h;
                        const double w = 1.0 / (1.0 + std::pow((f - 700.0) / 400.0, 2.0));
                        v += 0.12 * w * std::sin(2.0 * std::numbers::pi * f * static_cast<double>(n) / k_rate);
                    }
                } else if (kind == 2) {
                    v = noise(rng);
                }
                x.push_back(v);
            }
        }
    }

    /// What a frame centred at sample s should be, or -1 near a boundary.
    int expected(double s, double margin) const
    {
        double t0 = 0.0;
        for (auto [seconds, kind] : parts) {
            const double a = t0 * k_rate;
            const double b = (t0 + seconds) * k_rate;
            if (s >= a + margin && s < b - margin) {
                return kind;
            }
            t0 += seconds;
        }
        return -1;
    }
};

void test_labels_cpu()
{
    Speech speech;
    const std::vector<float> x(speech.x.begin(), speech.x.end());
    Yantra::VocalicConfig cfg;

    const auto frames = Kinesis::Vocalic::features(x, cfg);
    const auto labels = Kinesis::Vocalic::labels(frames, cfg);

    int checked = 0;
    int wrong = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        const double centre = static_cast<double>(i * cfg.hop) + 0.5 * cfg.frame;
        const int want = speech.expected(centre, cfg.frame);
        if (want < 0) {
            continue;
        }
        ++checked;
        if (static_cast<int>(labels[i]) != want) {
            ++wrong;
        }
    }
    std::printf("      %d frames checked, %d wrong\n", checked, wrong);
    check(checked > 100 && wrong == 0, "frames away from boundaries get the right label");

    // A vowel frame and a noise frame, side by side.
    const auto& v = frames[static_cast<size_t>(0.5 * k_rate / cfg.hop)];
    const auto& c = frames[static_cast<size_t>(0.8 * k_rate / cfg.hop)];
    std::printf("      vowel rms %.3f zcr %.3f per %.2f | noise rms %.3f zcr %.3f per %.2f\n",
        v.rms, v.zcr, v.periodicity, c.rms, c.zcr, c.periodicity);
    check(v.periodicity > 0.8F && c.periodicity < 0.5F, "periodicity separates vowel from noise");
}

void test_grammar_split()
{
    Speech speech;
    auto take = std::make_shared<Kakshya::DynamicSoundStream>(k_rate, 1);
    take->ensure_capacity(speech.x.size());
    take->append_frames(speech.x, 0);
    take = take->snapshot(speech.x.size());

    Yantra::VocalicOperators ops;
    auto grammar = Yantra::vocalic_grammar(ops);
    check(grammar != nullptr, "grammar builds from the registry");

    const auto split = Yantra::split_take(*grammar, take, {});
    check(!split.gpu, "no GPU offline, so the CPU path ran");
    check(split.vowels && split.consonants, "two new streams");
    if (!split.vowels || !split.consonants) {
        return;
    }

    const double v = static_cast<double>(split.vowels->get_num_frames()) / k_rate;
    const double c = static_cast<double>(split.consonants->get_num_frames()) / k_rate;
    std::printf("      vowels %.3f s (0.8 expected), consonants %.3f s (0.2 expected)\n", v, c);
    check(std::abs(v - 0.8) < 0.06, "vowel stream holds both vowels");
    check(std::abs(c - 0.2) < 0.06, "consonant stream holds the noise");

    auto aligned = Yantra::VocalicConfig {};
    aligned.compact = false;
    const auto full = Yantra::split_take(*grammar, take, aligned);
    check(full.vowels && full.vowels->get_num_frames() == speech.x.size(), "compact off keeps the take's length");
}

void test_shader_specs()
{
    namespace G = Portal::Graphics;
    const std::pair<const char*, G::ShaderSpec> specs[] {
        { "vocalic_features", Yantra::VocalicShaders::features() },
        { "vocalic_labels", Yantra::VocalicShaders::labels() },
        { "vocalic_split", Yantra::VocalicShaders::split() },
    };
    for (const auto& [name, spec] : specs) {
        const auto glsl = G::detail::emit_glsl_kernel(spec);
        const std::string path = std::string(name) + ".comp";
        std::ofstream(path) << glsl;
        const std::string cmd = "glslangValidator -V --target-env vulkan1.2 -o /dev/null " + path + " > " + path + ".log 2>&1";
        const int rc = std::system(cmd.c_str());
        std::printf("      %s: %zu bytes of GLSL, %u bytes of push constants\n", name, glsl.size(), spec.push_constant_bytes);
        check(rc == 0, (std::string(name) + " compiles (see " + path + ".log)").c_str());
    }
}

} // namespace

int main()
{
    test_take_length();
    test_stop_and_repeat();
    test_labels_cpu();
    test_grammar_split();
    test_shader_specs();

    std::printf("\n%s (%d failed)\n", failures == 0 ? "all passed" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
