#pragma once

#include "Vocalic.hpp"

#include <MayaFlux/Yantra/Executors/ShaderExecutionContext.hpp>
#include <MayaFlux/Yantra/Extractors/Extractor.hpp>

namespace MayaFlux::Yantra {

/**
 * @brief Splits audio into its vowel and consonant parts by frame labels.
 *
 * Grammar selection: Vocalic::Separate. Takes VocalicAnalyzer's output
 * (audio in channel 0, labels in channel 1).
 *
 * Output channels:
 *   0  vowels
 *   1  consonants
 *   2  the frame labels, passed on
 * With compact on (the default) the silent parts of each are dropped, so the
 * two are shorter than the input; with it off both stay aligned with it.
 *
 * One compute shader writes both parts per sample, crossfading over the
 * label changes. Without a GPU the CPU does the same.
 */
class VocalicExtractor : public Extractor<std::vector<Kakshya::DataVariant>> {
public:
    using supports = Supports<VocalicConfig>;
    static constexpr Target target = targets::vulkan;

    explicit VocalicExtractor(VocalicConfig config = {});

    [[nodiscard]] ExtractionType get_extraction_type() const override { return ExtractionType::FEATURE_GUIDED; }
    [[nodiscard]] ConfigRefs configs() const override { return this->stored_configs(supports {}); }

    bool validate_input(const input_type& input) const override;

    void set_cpu_only(bool cpu) { m_cpu_only = cpu; }

protected:
    output_type extract_implementation(const input_type& input) override;
    [[nodiscard]] std::string get_extractor_name() const override { return "VocalicExtractor"; }

private:
    bool run_gpu(std::span<const float> x, std::span<const float> labels, const VocalicConfig& cfg, std::vector<float>& parts);

    std::shared_ptr<ShaderExecutionContext<>> m_split;
    bool m_cpu_only { false };
    bool m_gpu_failed { false };
};

} // namespace MayaFlux::Yantra
