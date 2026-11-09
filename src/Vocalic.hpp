#pragma once

#include <MayaFlux/Transitive/Reflect/EnumReflect.hpp>
#include <MayaFlux/Yantra/ComputeOperation.hpp>

#include <span>
#include <vector>

namespace MayaFlux::Yantra {

/**
 * @brief Grammar selections for splitting speech into vowels and consonants.
 *
 * Classify labels frames as silence, vowel or consonant (VocalicAnalyzer).
 * Separate splits the audio by those labels (VocalicExtractor).
 * Compose them in order: Vocalic::Classify | Vocalic::Separate.
 */
enum class Vocalic : uint8_t {
    Classify,
    Separate,
};

/// Frame labels written by Classify.
enum class VocalicLabel : uint8_t {
    Silence = 0,
    Vowel = 1,
    Consonant = 2,
};

/**
 * @brief Settings shared by both stages. Pass per call through
 * ExecutionContext::facets, or per rule in the grammar.
 */
struct VocalicConfig : OperationConfig {
    static constexpr auto describe_facets()
    {
        return std::tuple {
            FacetMember<&VocalicConfig::method> {},
            FacetMember<&VocalicConfig::sample_rate> {},
            FacetMember<&VocalicConfig::frame> {},
            FacetMember<&VocalicConfig::hop> {},
            FacetMember<&VocalicConfig::min_hz> {},
            FacetMember<&VocalicConfig::max_hz> {},
            FacetMember<&VocalicConfig::silence> {},
            FacetMember<&VocalicConfig::voicing> {},
            FacetMember<&VocalicConfig::zcr_max> {},
            FacetMember<&VocalicConfig::vowel_floor> {},
            FacetMember<&VocalicConfig::reference_rms> {},
            FacetMember<&VocalicConfig::smooth> {},
            FacetMember<&VocalicConfig::fade> {},
            FacetMember<&VocalicConfig::compact> {},
        };
    }

    Reflect::EnumByte<Vocalic> method { Vocalic::Classify };
    uint32_t sample_rate { 48000 };
    /// Analysis frame and hop, in samples.
    uint32_t frame { 1536 };
    uint32_t hop { 256 };
    /// Pitch range searched for periodicity.
    float min_hz { 70.F };
    float max_hz { 400.F };
    /// Frames quieter than this RMS are silence.
    float silence { 0.01F };
    /// Normalised autocorrelation a vowel frame must reach.
    float voicing { 0.5F };
    /// Zero crossings per sample above which a frame is a consonant.
    float zcr_max { 0.2F };
    /// A vowel frame is at least this fraction of the loudest frame's RMS.
    float vowel_floor { 0.12F };
    /// Loudness the vowel floor also measures against, for short windows of a
    /// longer stream. The floor uses whichever is louder, this or the loudest frame.
    float reference_rms { 0.F };
    /// Frames on each side in the majority vote that cleans the labels.
    uint32_t smooth { 2 };
    /// Frames on each side of a label change that are crossfaded.
    uint32_t fade { 1 };
    /// Separate drops the silent parts of each output.
    bool compact { true };
};

} // namespace MayaFlux::Yantra

/**
 * The same arithmetic as the shaders, on the CPU. Used by the tests, and by the
 * operators when no GPU is available.
 */
namespace MayaFlux::Kinesis::Vocalic {

/// Per frame: RMS, zero crossings per sample, best normalised autocorrelation.
struct Frame {
    float rms { 0.F };
    float zcr { 0.F };
    float periodicity { 0.F };
};

[[nodiscard]] uint32_t frame_count(size_t samples, uint32_t hop);

[[nodiscard]] std::vector<Frame> features(std::span<const float> x, const Yantra::VocalicConfig& cfg);

[[nodiscard]] uint8_t classify(const Frame& f, const Yantra::VocalicConfig& cfg, float vowel_rms);

/// Labels after the majority vote. A vowel reaches vowel_floor times the louder of
/// reference_rms and the loudest frame.
[[nodiscard]] std::vector<float> labels(std::span<const Frame> frames, const Yantra::VocalicConfig& cfg);

[[nodiscard]] float loudest(std::span<const Frame> frames);

/// Vowel and consonant gain at sample s, from frame labels.
[[nodiscard]] std::pair<float, float> gains(std::span<const float> labels, size_t s, const Yantra::VocalicConfig& cfg);

/// The audio masked into vowel and consonant parts, both as long as x.
void split(std::span<const float> x, std::span<const float> labels, const Yantra::VocalicConfig& cfg,
    std::vector<float>& vowels, std::vector<float>& consonants);

/// Keep only samples where the mask is open.
[[nodiscard]] std::vector<double> compact(std::span<const float> masked, std::span<const float> gain);

} // namespace MayaFlux::Kinesis::Vocalic
