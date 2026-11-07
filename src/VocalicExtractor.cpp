#include "VocalicExtractor.hpp"
#include "VocalicAnalyzer.hpp"
#include "VocalicShaders.hpp"

namespace MayaFlux::Yantra {

namespace {

    struct SplitPush {
        uint32_t n;
        uint32_t frame;
        uint32_t hop;
        uint32_t frames;
        uint32_t fade;
    };

    template <typename T>
    T meta_or(const std::unordered_map<std::string, std::any>& meta, const std::string& key, T fallback)
    {
        if (auto it = meta.find(key); it != meta.end()) {
            if (const auto* v = std::any_cast<T>(&it->second)) {
                return *v;
            }
        }
        return fallback;
    }

}

VocalicExtractor::VocalicExtractor(VocalicConfig config)
{
    this->set_config(std::move(config));
}

bool VocalicExtractor::validate_input(const input_type& input) const
{
    return input.data.size() >= 2 && !vocalic_samples(input.data[0]).empty();
}

bool VocalicExtractor::run_gpu(std::span<const float> x, std::span<const float> labels, const VocalicConfig& cfg,
    std::vector<float>& parts)
{
    if (m_cpu_only || m_gpu_failed) {
        return false;
    }
    if (!m_split) {
        m_split = vocalic_shader(VocalicShaders::split(), sizeof(SplitPush), "vocalic_split");
        if (!m_split) {
            return false;
        }
    }

    try {
        const auto n = static_cast<uint32_t>(x.size());
        m_split->set_binding_data(0, x);
        m_split->set_binding_data(1, labels);
        m_split->set_output_size(2, static_cast<size_t>(n) * 4 * sizeof(float));
        m_split->push(SplitPush { n, cfg.frame, cfg.hop, static_cast<uint32_t>(labels.size()), cfg.fade });
        const auto result = m_split->execute(input_type {}, ExecutionContext {});
        parts = ShaderExecutionContext<>::read_output<float>(result, 2);
        if (parts.size() < static_cast<size_t>(n) * 4) {
            throw std::runtime_error("short split readback");
        }
        return true;
    } catch (const std::exception& e) {
        MF_WARN(Journal::Component::Yantra, Journal::Context::Runtime,
            "VocalicExtractor: GPU path failed ({}), using the CPU", e.what());
        m_gpu_failed = true;
        return false;
    }
}

VocalicExtractor::output_type VocalicExtractor::extract_implementation(const input_type& input)
{
    auto cfg = this->config<VocalicConfig>();
    // Frames are described by whoever labelled them.
    cfg.frame = meta_or<uint32_t>(input.metadata, "frame", cfg.frame);
    cfg.hop = meta_or<uint32_t>(input.metadata, "hop", cfg.hop);

    const auto x = vocalic_samples(input.data[0]);
    const auto labels = vocalic_samples(input.data[1]);
    const size_t n = x.size();

    std::vector<float> vowels(n);
    std::vector<float> consonants(n);
    std::vector<float> vowel_gain(n);
    std::vector<float> consonant_gain(n);

    std::vector<float> parts;
    const bool gpu = run_gpu(x, labels, cfg, parts);
    if (gpu) {
        for (size_t s = 0; s < n; ++s) {
            vowels[s] = parts[s * 4];
            consonants[s] = parts[s * 4 + 1];
            vowel_gain[s] = parts[s * 4 + 2];
            consonant_gain[s] = parts[s * 4 + 3];
        }
    } else {
        for (size_t s = 0; s < n; ++s) {
            const auto [v, c] = Kinesis::Vocalic::gains(labels, s, cfg);
            vowels[s] = x[s] * v;
            consonants[s] = x[s] * c;
            vowel_gain[s] = v;
            consonant_gain[s] = c;
        }
    }

    std::vector<double> v_out;
    std::vector<double> c_out;
    if (cfg.compact) {
        v_out = Kinesis::Vocalic::compact(vowels, vowel_gain);
        c_out = Kinesis::Vocalic::compact(consonants, consonant_gain);
    } else {
        v_out.assign(vowels.begin(), vowels.end());
        c_out.assign(consonants.begin(), consonants.end());
    }

    std::vector<Kakshya::DataVariant> channels {
        std::move(v_out),
        std::move(c_out),
        std::vector<double>(labels.begin(), labels.end()),
    };

    // Set directly: inferring a structure from mixed-length channels only warns.
    output_type result;
    result.data = std::move(channels);
    result.metadata = input.metadata;
    result.metadata["compact"] = cfg.compact;
    result.metadata["gpu_split"] = gpu;
    return result;
}

} // namespace MayaFlux::Yantra
