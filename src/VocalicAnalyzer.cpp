#include "VocalicAnalyzer.hpp"
#include "VocalicShaders.hpp"

#include <MayaFlux/Portal/Graphics/ShaderFoundry.hpp>
#include <MayaFlux/Yantra/Executors/GpuResourceManager.hpp>
#include <MayaFlux/Yantra/Executors/ShaderSpecBinding.hpp>

#include <bit>
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

    struct ReducePush {
        uint32_t frames;
        uint32_t chunks;
    };

    struct LabelPush {
        uint32_t frames;
        uint32_t smooth;
        float silence;
        float voicing;
        float zcr_max;
        float vowel_floor;
    };

    std::array<uint32_t, 3> groups_for(size_t invocations)
    {
        return { static_cast<uint32_t>((invocations + 255) / 256), 1U, 1U };
    }

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

bool VocalicAnalyzer::prepare_gpu()
{
    if (m_chain) {
        return true;
    }
    if (!Portal::Graphics::get_shader_foundry().is_initialized()) {
        return false;
    }

    const auto features = VocalicShaders::features();
    const auto reduce = VocalicShaders::reduce();
    const auto labels = VocalicShaders::labels();
    if (features.push_constant_bytes != sizeof(FeaturePush) || reduce.push_constant_bytes != sizeof(ReducePush)
        || labels.push_constant_bytes != sizeof(LabelPush)) {
        MF_ERROR(Journal::Component::Yantra, Journal::Context::Runtime,
            "VocalicAnalyzer: push constant sizes do not match the shader specs");
        m_gpu_failed = true;
        return false;
    }

    m_features_config = config_from_spec(features);
    m_reduce_config = config_from_spec(reduce);
    m_labels_config = config_from_spec(labels);
    for (const auto* config : { &m_features_config, &m_reduce_config, &m_labels_config }) {
        if (config->shader_id == Portal::Graphics::INVALID_SHADER) {
            return false;
        }
    }

    // Every slot is a shared buffer, so the context only declares where they sit.
    namespace S = VocalicShaders::Slot;
    using B = GpuBufferBinding;
    std::vector<B> bindings;
    for (const uint32_t slot : { S::x, S::partial, S::frame_features, S::peak, S::labels }) {
        bindings.push_back({ .set = 0,
            .binding = slot,
            .direction = B::Direction::OUTPUT,
            .element_type = slot == S::peak ? B::ElementType::UINT32 : B::ElementType::FLOAT32 });
    }

    m_chain = std::make_shared<ShaderExecutionContext<>>(m_features_config, std::move(bindings), "vocalic_classify");
    m_recorder = std::make_shared<GpuDispatchRecorder>(true);
    m_chain->set_recording(m_recorder);
    return true;
}

bool VocalicAnalyzer::run_gpu(std::span<const float> x, const VocalicConfig& cfg,
    std::vector<Kinesis::Vocalic::Frame>& frames, std::vector<float>& labels)
{
    if (m_cpu_only || m_gpu_failed || x.empty() || !prepare_gpu()) {
        return false;
    }

    try {
        namespace S = VocalicShaders::Slot;
        using Element = GpuBufferBinding::ElementType;
        constexpr auto usage = Portal::Graphics::BufferUsageHint::COMPUTE;

        const auto n = static_cast<uint32_t>(x.size());
        const uint32_t count = Kinesis::Vocalic::frame_count(x.size(), cfg.hop);
        const uint32_t chunks = VocalicShaders::k_lag_chunks;
        const auto lag = [&](float hz) {
            return static_cast<uint32_t>(std::lround(static_cast<double>(cfg.sample_rate) / std::max(1.0F, hz)));
        };

        auto& chain = *m_chain;
        chain.ensure_shared_buffer(0, S::x, n, Element::FLOAT32, usage);
        chain.ensure_shared_buffer(0, S::partial, static_cast<size_t>(count) * chunks * 4, Element::FLOAT32, usage);
        chain.ensure_shared_buffer(0, S::frame_features, static_cast<size_t>(count) * 4, Element::FLOAT32, usage);
        chain.ensure_shared_buffer(0, S::peak, 1, Element::UINT32, usage);
        chain.ensure_shared_buffer(0, S::labels, count, Element::FLOAT32, usage);

        chain.upload_shared_raw(0, S::x, reinterpret_cast<const uint8_t*>(x.data()), x.size_bytes());
        // The loudest frame starts at the reference, so a short window of a stream is judged against it.
        chain.fill_shared(0, S::peak, std::bit_cast<uint32_t>(std::max(0.F, cfg.reference_rms)));

        const FeaturePush feature_push { n, cfg.frame, cfg.hop, count, lag(cfg.max_hz), lag(cfg.min_hz), chunks };
        const ReducePush reduce_push { count, chunks };
        const LabelPush label_push { count, cfg.smooth, cfg.silence, cfg.voicing, cfg.zcr_max, cfg.vowel_floor };

        chain.record_stages({
            { .config = m_features_config,
                .stage_fn = [feature_push](GpuDispatchCore& c) { c.set_push_constants(feature_push); },
                .explicit_groups = groups_for(static_cast<size_t>(count) * chunks) },
            { .config = m_reduce_config,
                .stage_fn = [reduce_push](GpuDispatchCore& c) { c.set_push_constants(reduce_push); },
                .explicit_groups = groups_for(count) },
            { .config = m_labels_config,
                .stage_fn = [label_push](GpuDispatchCore& c) { c.set_push_constants(label_push); },
                .explicit_groups = groups_for(count) },
        });

        // The first read submits the three stages and waits for them.
        std::vector<float> per_frame(static_cast<size_t>(count) * 4);
        chain.download_shared(0, S::frame_features, per_frame.data(), per_frame.size() * sizeof(float));
        labels.resize(count);
        chain.download_shared(0, S::labels, labels.data(), labels.size() * sizeof(float));

        frames.resize(count);
        for (uint32_t f = 0; f < count; ++f) {
            frames[f] = { per_frame[f * 4], per_frame[f * 4 + 1], per_frame[f * 4 + 2] };
        }
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
