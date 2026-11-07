#include "VocalicAnalyzer.hpp"
#include "VocalicShaders.hpp"

#include <MayaFlux/Portal/Graphics/ShaderFoundry.hpp>
#include <MayaFlux/Yantra/Executors/ShaderSpecBinding.hpp>

#include <cmath>

namespace MayaFlux::Yantra {

namespace {

    struct FeaturePush {
        uint32_t n;
        uint32_t frame;
        uint32_t hop;
        uint32_t frames;
        uint32_t min_lag;
        uint32_t max_lag;
        uint32_t chunks;
    };

    struct LabelPush {
        uint32_t frames;
        uint32_t smooth;
        float silence;
        float voicing;
        float zcr_max;
        float vowel_rms;
    };

    std::vector<double> widen(const std::vector<float>& v)
    {
        return { v.begin(), v.end() };
    }

}

std::vector<float> vocalic_samples(const Kakshya::DataVariant& channel)
{
    return std::visit([](const auto& v) -> std::vector<float> {
        using T = typename std::decay_t<decltype(v)>::value_type;
        if constexpr (std::is_arithmetic_v<T>) {
            std::vector<float> out(v.size());
            std::ranges::transform(v, out.begin(), [](T s) { return static_cast<float>(s); });
            return out;
        } else {
            return {};
        }
    },
        channel);
}

std::shared_ptr<ShaderExecutionContext<>> vocalic_shader(
    const Portal::Graphics::ShaderSpec& spec, size_t push_bytes, std::string name)
{
    if (!Portal::Graphics::get_shader_foundry().is_initialized()) {
        return nullptr;
    }
    if (spec.push_constant_bytes != push_bytes) {
        MF_ERROR(Journal::Component::Yantra, Journal::Context::Runtime,
            "{}: push constants are {} bytes in the spec, {} pushed", name, spec.push_constant_bytes, push_bytes);
        return nullptr;
    }
    auto config = config_from_spec(spec);
    if (config.shader_id == Portal::Graphics::INVALID_SHADER) {
        return nullptr;
    }
    return std::make_shared<ShaderExecutionContext<>>(std::move(config), bindings_from_spec(spec), std::move(name));
}

VocalicAnalyzer::VocalicAnalyzer(VocalicConfig config)
{
    this->set_config(std::move(config));
}

bool VocalicAnalyzer::validate_analysis_input(const input_type& input) const
{
    return !input.data.empty() && !vocalic_samples(input.data[0]).empty();
}

bool VocalicAnalyzer::run_gpu(std::span<const float> x, const VocalicConfig& cfg,
    std::vector<Kinesis::Vocalic::Frame>& frames, std::vector<float>& labels)
{
    // One invocation per sample is dispatched, so frames * chunks must fit: hop >= chunks.
    if (m_cpu_only || m_gpu_failed || cfg.hop < VocalicShaders::k_lag_chunks) {
        return false;
    }
    if (!m_features) {
        m_features = vocalic_shader(VocalicShaders::features(), sizeof(FeaturePush), "vocalic_features");
        m_labels = vocalic_shader(VocalicShaders::labels(), sizeof(LabelPush), "vocalic_labels");
        if (!m_features || !m_labels) {
            m_features.reset();
            m_labels.reset();
            return false;
        }
    }

    try {
        const auto count = Kinesis::Vocalic::frame_count(x.size(), cfg.hop);
        const auto lag = [&](float hz) {
            return static_cast<uint32_t>(std::lround(static_cast<double>(cfg.sample_rate) / std::max(1.0F, hz)));
        };

        const uint32_t chunks = VocalicShaders::k_lag_chunks;
        m_features->set_binding_data(0, x);
        m_features->set_output_size(1, static_cast<size_t>(count) * chunks * 4 * sizeof(float));
        m_features->push(FeaturePush { static_cast<uint32_t>(x.size()), cfg.frame, cfg.hop, count, lag(cfg.max_hz), lag(cfg.min_hz), chunks });
        const auto fout = m_features->execute(input_type {}, ExecutionContext {});
        const auto raw = ShaderExecutionContext<>::read_output<float>(fout, 1);
        if (raw.size() < static_cast<size_t>(count) * chunks * 4) {
            throw std::runtime_error("short feature readback");
        }

        // Each frame's lag search came back in chunks; keep the best.
        frames.resize(count);
        std::vector<float> reduced(static_cast<size_t>(count) * 4, 0.F);
        for (uint32_t f = 0; f < count; ++f) {
            const float* row = raw.data() + static_cast<size_t>(f) * chunks * 4;
            float best = 0.F;
            for (uint32_t c = 0; c < chunks; ++c) {
                best = std::max(best, row[c * 4 + 2]);
            }
            frames[f] = { row[0], row[1], best };
            reduced[f * 4] = row[0];
            reduced[f * 4 + 1] = row[1];
            reduced[f * 4 + 2] = best;
        }

        // The loudest frame is a reduction over all frames; it is cheap here.
        const float vowel_rms = cfg.vowel_floor * Kinesis::Vocalic::loudest(frames);

        m_labels->set_binding_data(0, reduced);
        m_labels->set_output_size(1, count * sizeof(float));
        m_labels->push(LabelPush { count, cfg.smooth, cfg.silence, cfg.voicing, cfg.zcr_max, vowel_rms });
        const auto lout = m_labels->execute(input_type {}, ExecutionContext {});
        labels = ShaderExecutionContext<>::read_output<float>(lout, 1);
        labels.resize(count);
        return true;
    } catch (const std::exception& e) {
        MF_WARN(Journal::Component::Yantra, Journal::Context::Runtime,
            "VocalicAnalyzer: GPU path failed ({}), using the CPU", e.what());
        m_gpu_failed = true;
        return false;
    }
}

VocalicAnalyzer::output_type VocalicAnalyzer::analyze_implementation(const input_type& input)
{
    const auto& cfg = this->config<VocalicConfig>();
    const auto x = vocalic_samples(input.data[0]);

    std::vector<Kinesis::Vocalic::Frame> frames;
    std::vector<float> labels;
    const bool gpu = run_gpu(x, cfg, frames, labels);
    if (!gpu) {
        frames = Kinesis::Vocalic::features(x, cfg);
        labels = Kinesis::Vocalic::labels(frames, cfg);
    }

    std::vector<double> rms(frames.size());
    std::vector<double> zcr(frames.size());
    std::vector<double> per(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        rms[i] = frames[i].rms;
        zcr[i] = frames[i].zcr;
        per[i] = frames[i].periodicity;
    }

    std::vector<Kakshya::DataVariant> channels {
        input.data[0],
        widen(labels),
        std::move(rms),
        std::move(zcr),
        std::move(per),
    };

    // Set directly: inferring a structure from mixed-length channels only warns.
    output_type result;
    result.data = std::move(channels);
    result.metadata = input.metadata;
    result.container = input.container;
    result.metadata["frame"] = cfg.frame;
    result.metadata["hop"] = cfg.hop;
    result.metadata["sample_rate"] = cfg.sample_rate;
    result.metadata["gpu"] = gpu;

    store_current_analysis(labels);
    return result;
}

} // namespace MayaFlux::Yantra
