#pragma once

#include "Vocalic.hpp"

#include <MayaFlux/Portal/Graphics/ShaderSpec.hpp>
#include <MayaFlux/Yantra/Analyzers/Analyzer.hpp>
#include <MayaFlux/Yantra/Executors/ShaderExecutionContext.hpp>

namespace MayaFlux::Yantra {

/**
 * @brief Labels each frame of a voice as silence, vowel or consonant.
 *
 * Grammar selection: Vocalic::Classify.
 *
 * Input: channel 0 is mono audio. Output channels:
 *   0  the audio, unchanged
 *   1  label per frame (0 silence, 1 vowel, 2 consonant)
 *   2  RMS per frame
 *   3  zero crossings per sample, per frame
 *   4  periodicity (best normalised autocorrelation) per frame
 * Metadata "frame", "hop" and "sample_rate" describe the frames; "gpu" says
 * whether the shaders ran.
 *
 * A vowel frame is loud, periodic and has few zero crossings. Consonants are
 * the rest that is not silent. Two compute shaders do the work: one for the
 * frame features, one for the labels and their majority vote. Without a GPU
 * the same arithmetic runs on the CPU (Kinesis::Vocalic).
 */
class VocalicAnalyzer : public Analyzer<std::vector<Kakshya::DataVariant>> {
public:
    using supports = Supports<VocalicConfig>;
    static constexpr Target target = targets::vulkan;

    explicit VocalicAnalyzer(VocalicConfig config = {});

    [[nodiscard]] AnalysisType get_analysis_type() const override { return AnalysisType::FEATURE; }
    [[nodiscard]] std::string get_name() const override { return "VocalicAnalyzer"; }
    [[nodiscard]] ConfigRefs configs() const override { return this->stored_configs(supports {}); }

    /// Run on the CPU even when a GPU is there.
    void set_cpu_only(bool cpu) { m_cpu_only = cpu; }

protected:
    output_type analyze_implementation(const input_type& input) override;
    bool validate_analysis_input(const input_type& input) const override;

private:
    bool run_gpu(std::span<const float> x, const VocalicConfig& cfg,
        std::vector<Kinesis::Vocalic::Frame>& frames, std::vector<float>& labels);

    std::shared_ptr<ShaderExecutionContext<>> m_features;
    std::shared_ptr<ShaderExecutionContext<>> m_labels;
    bool m_cpu_only { false };
    bool m_gpu_failed { false };
};

/// Mono samples of channel 0 as floats, from any numeric channel type.
[[nodiscard]] std::vector<float> vocalic_samples(const Kakshya::DataVariant& channel);

/**
 * A compute context for a ShaderSpec, or null when there is no GPU yet or it
 * does not compile. @p push_bytes is the size of the struct the caller pushes;
 * it must match the spec's push constant block.
 */
[[nodiscard]] std::shared_ptr<ShaderExecutionContext<>> vocalic_shader(
    const Portal::Graphics::ShaderSpec& spec, size_t push_bytes, std::string name);

} // namespace MayaFlux::Yantra
