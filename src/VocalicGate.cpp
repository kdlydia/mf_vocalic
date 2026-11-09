#include "VocalicGate.hpp"

#include <MayaFlux/Buffers/BufferProcessingChain.hpp>

#include <bit>
#include <cmath>

namespace MayaFlux::Buffers {

struct VocalicGate::State {
    // Written by the primary processor on the audio thread, read behind its write head elsewhere.
    std::vector<float> ring;
    uint64_t ring_mask { 0 };
    std::atomic<uint64_t> written { 0 };

    // Per frame, (frame + 1) << 2 | label, so an empty slot or an old frame is never mistaken for this one.
    std::vector<std::atomic<uint64_t>> labels;
    uint64_t label_mask { 0 };
    std::atomic<uint64_t> latest { 0 };

    std::atomic<uint8_t> mode { static_cast<uint8_t>(VocalicGateMode::Off) };
    std::atomic<float> vowel_level { 1.F };
    std::atomic<float> consonant_level { 1.F };
    std::atomic<float> level { 0.F };

    std::shared_ptr<AudioBuffer> input;
    Vruta::TaskScheduler* scheduler { nullptr };
    uint32_t delay { 0 };
    uint32_t frame { 1536 };
    uint32_t hop { 256 };
    float glide { 1.F };

    // Audio thread only.
    uint64_t cycle { UINT64_MAX };
    uint8_t held { 0 };
    float vowel_gain { 0.F };
    float consonant_gain { 0.F };
    std::vector<double> block;

    [[nodiscard]] uint8_t label_at(uint64_t position)
    {
        const uint64_t centre = frame / 2;
        const uint64_t k = position < centre ? 0 : (position - centre) / hop;
        const uint64_t v = labels[k & label_mask].load(std::memory_order_acquire);
        if ((v >> 2) == k + 1) {
            held = static_cast<uint8_t>(v & 3U);
        }
        return held;
    }
};

namespace {

    uint64_t power_of_two_at_least(uint64_t n)
    {
        return std::bit_ceil(std::max<uint64_t>(n, 2));
    }

}

std::shared_ptr<VocalicGate> VocalicGate::attach(BufferManager& manager, Vruta::TaskScheduler& scheduler,
    VocalicGateConfig config)
{
    auto gate = std::shared_ptr<VocalicGate>(new VocalicGate());
    const auto rate = static_cast<uint32_t>(s_registered_sample_rate);
    config.analysis.sample_rate = rate;
    gate->m_config = config;

    auto block = manager.get_buffer_size(ProcessingToken::AUDIO_BACKEND);
    if (block == 0) {
        block = 512;
    }

    auto state = std::make_shared<State>();
    state->scheduler = &scheduler;
    state->frame = config.analysis.frame;
    state->hop = config.analysis.hop;
    state->delay = static_cast<uint32_t>(config.delay_seconds * rate);
    state->glide = config.glide_seconds > 0.0
        ? static_cast<float>(1.0 - std::exp(-1.0 / (config.glide_seconds * rate)))
        : 1.F;

    // Room for the delay, the longest window and a few blocks, so nothing is read while it is overwritten.
    const auto window = static_cast<uint64_t>(config.max_window_seconds * rate) + config.analysis.frame;
    const auto ring = power_of_two_at_least(state->delay + window + 8ULL * block);
    state->ring.assign(ring, 0.F);
    state->ring_mask = ring - 1;
    const auto label_slots = power_of_two_at_least(ring / state->hop + 4);
    state->labels = std::vector<std::atomic<uint64_t>>(label_slots);
    state->label_mask = label_slots - 1;
    state->block.assign(block, 0.0);

    state->input = std::make_shared<AudioBuffer>(config.input_channel, block);
    manager.register_input_listener(state->input, config.input_channel);

    for (size_t i = 0; i < config.output_channels.size(); ++i) {
        const auto channel = config.output_channels[i];
        auto out = std::make_shared<AudioBuffer>(channel, block);
        manager.add_buffer(out, ProcessingToken::AUDIO_BACKEND, channel);
        out->get_processing_chain()->add_processor(std::make_shared<VocalicGateProcessor>(state, i == 0), out);
        gate->m_outputs.push_back(std::move(out));
    }

    gate->m_state = std::move(state);
    gate->m_analyzer = std::make_shared<Yantra::VocalicAnalyzer>(config.analysis);
    return gate;
}

void VocalicGate::set_mode(VocalicGateMode mode)
{
    m_state->mode.store(static_cast<uint8_t>(mode), std::memory_order_relaxed);
}

VocalicGateMode VocalicGate::mode() const
{
    return static_cast<VocalicGateMode>(m_state->mode.load(std::memory_order_relaxed));
}

void VocalicGate::set_levels(double vowels, double consonants)
{
    m_state->vowel_level.store(static_cast<float>(vowels), std::memory_order_relaxed);
    m_state->consonant_level.store(static_cast<float>(consonants), std::memory_order_relaxed);
}

void VocalicGate::publish(uint64_t frame, Yantra::VocalicLabel label)
{
    const uint64_t packed = ((frame + 1) << 2) | (static_cast<uint64_t>(label) & 3U);
    m_state->labels[frame & m_state->label_mask].store(packed, std::memory_order_release);
    if (packed > m_state->latest.load(std::memory_order_relaxed)) {
        m_state->latest.store(packed, std::memory_order_relaxed);
    }
}

Yantra::VocalicLabel VocalicGate::latest() const
{
    return static_cast<Yantra::VocalicLabel>(m_state->latest.load(std::memory_order_relaxed) & 3U);
}

float VocalicGate::level() const
{
    return m_state->level.load(std::memory_order_relaxed);
}

uint64_t VocalicGate::samples_in() const
{
    return m_state->written.load(std::memory_order_acquire);
}

uint32_t VocalicGate::delay_samples() const
{
    return m_state->delay;
}

size_t VocalicGate::analyse()
{
    auto& s = *m_state;
    const uint64_t written = s.written.load(std::memory_order_acquire);
    if (written < s.frame) {
        return 0;
    }

    // The newest frame that lies entirely in what has arrived.
    const uint64_t last = (written - s.frame) / s.hop;
    if (last < m_next_frame) {
        return 0;
    }

    const auto max_frames = std::max<uint64_t>(1, static_cast<uint64_t>(m_config.max_window_seconds * m_config.analysis.sample_rate) / s.hop);
    if (last + 1 - m_next_frame > max_frames) {
        m_next_frame = last + 1 - max_frames;
    }

    // A few frames before the first new one, so the majority vote sees its neighbours.
    const uint64_t smooth = m_config.analysis.smooth;
    const uint64_t from = m_next_frame > smooth ? m_next_frame - smooth : 0;
    const uint64_t begin = from * s.hop;
    const uint64_t end = last * s.hop + s.frame;

    std::vector<double> x(end - begin);
    for (uint64_t i = 0; i < x.size(); ++i) {
        x[i] = s.ring[(begin + i) & s.ring_mask];
    }

    auto cfg = m_config.analysis;
    cfg.reference_rms = m_reference;
    m_analyzer->set_config(cfg);

    Yantra::Datum<std::vector<Kakshya::DataVariant>> input;
    input.data.emplace_back(std::move(x));
    input.modality = Kakshya::DataModality::AUDIO_1D;
    const auto out = m_analyzer->apply_operation(input);

    const auto* labels = out.data.size() > 1 ? std::get_if<std::vector<double>>(&out.data[1]) : nullptr;
    const auto* rms = out.data.size() > 2 ? std::get_if<std::vector<double>>(&out.data[2]) : nullptr;
    if (!labels) {
        return 0;
    }

    size_t published = 0;
    float peak = 0.F;
    for (size_t j = 0; j < labels->size(); ++j) {
        const uint64_t k = from + j;
        if (k > last) {
            break;
        }
        if (rms && j < rms->size()) {
            peak = std::max(peak, static_cast<float>((*rms)[j]));
        }
        if (k >= m_next_frame) {
            publish(k, static_cast<Yantra::VocalicLabel>(static_cast<uint8_t>((*labels)[j])));
            ++published;
        }
    }
    m_next_frame = last + 1;

    // The vowel floor follows the voice: the loudest recent frame, falling back slowly.
    const double seconds = static_cast<double>(published * s.hop) / m_config.analysis.sample_rate;
    m_reference = std::max(peak, static_cast<float>(m_reference * std::pow(m_config.reference_decay, seconds)));
    return published;
}

VocalicGateProcessor::VocalicGateProcessor(std::shared_ptr<VocalicGate::State> state, bool primary)
    : m_state(std::move(state))
    , m_primary(primary)
{
}

bool VocalicGateProcessor::is_compatible_with(const std::shared_ptr<Buffer>& buffer) const
{
    return std::dynamic_pointer_cast<AudioBuffer>(buffer) != nullptr;
}

// Audio thread.
void VocalicGateProcessor::processing_function(const std::shared_ptr<Buffer>& buffer)
{
    auto audio = std::dynamic_pointer_cast<AudioBuffer>(buffer);
    if (!audio) {
        return;
    }
    auto& out = audio->get_data();
    auto& s = *m_state;

    const uint64_t cycle = s.scheduler->get_current_buffer_cycle();
    if (!m_primary || cycle == s.cycle) {
        // Other channels, and a second run in the same cycle, play the block already made.
        std::copy_n(s.block.begin(), std::min(out.size(), s.block.size()), out.begin());
        return;
    }
    s.cycle = cycle;

    const auto& in = s.input->get_data();
    const uint64_t w = s.written.load(std::memory_order_relaxed);
    double sum = 0.0;
    for (size_t i = 0; i < in.size(); ++i) {
        s.ring[(w + i) & s.ring_mask] = static_cast<float>(in[i]);
        sum += in[i] * in[i];
    }
    s.written.store(w + in.size(), std::memory_order_release);
    s.level.store(in.empty() ? 0.F : static_cast<float>(std::sqrt(sum / static_cast<double>(in.size()))),
        std::memory_order_relaxed);

    const auto mode = static_cast<VocalicGateMode>(s.mode.load(std::memory_order_relaxed));
    const bool vowels = mode == VocalicGateMode::Vowels || mode == VocalicGateMode::Both;
    const bool consonants = mode == VocalicGateMode::Consonants || mode == VocalicGateMode::Both;
    const float vowel_level = s.vowel_level.load(std::memory_order_relaxed);
    const float consonant_level = s.consonant_level.load(std::memory_order_relaxed);

    s.block.resize(out.size());
    for (size_t j = 0; j < out.size(); ++j) {
        const uint64_t now = w + j;
        float y = 0.F;
        float tv = 0.F;
        float tc = 0.F;
        if (now >= s.delay) {
            const uint64_t p = now - s.delay;
            const uint8_t label = s.label_at(p);
            tv = vowels && label == 1 ? vowel_level : 0.F;
            tc = consonants && label == 2 ? consonant_level : 0.F;
            y = s.ring[p & s.ring_mask];
        }
        s.vowel_gain += (tv - s.vowel_gain) * s.glide;
        s.consonant_gain += (tc - s.consonant_gain) * s.glide;
        const double sample = static_cast<double>(y * (s.vowel_gain + s.consonant_gain));
        out[j] = sample;
        s.block[j] = sample;
    }
}

} // namespace MayaFlux::Buffers
