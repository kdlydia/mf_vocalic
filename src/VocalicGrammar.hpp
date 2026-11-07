#pragma once

#include "VocalicAnalyzer.hpp"
#include "VocalicExtractor.hpp"

#include <MayaFlux/Kakshya/Source/DynamicSoundStream.hpp>
#include <MayaFlux/Yantra/ComputeGrammar.hpp>
#include <MayaFlux/Yantra/OperationSpec/OperationRegistry.hpp>

namespace MayaFlux::Yantra {

/// The two operator instances a vocalic grammar runs. Shared so settings can be changed later.
struct VocalicOperators {
    std::shared_ptr<VocalicAnalyzer> analyzer { std::make_shared<VocalicAnalyzer>() };
    std::shared_ptr<VocalicExtractor> extractor { std::make_shared<VocalicExtractor>() };
};

/**
 * @brief The default registry with Vocalic::Classify and Vocalic::Separate added.
 *
 * Any other registered operation can follow them in a signature, as long as it
 * takes the vector of channels Separate returns.
 */
[[nodiscard]] OperationRegistry vocalic_registry(const VocalicOperators& ops);

/// A grammar for Vocalic::Classify | Vocalic::Separate.
[[nodiscard]] std::shared_ptr<ComputationGrammar> vocalic_grammar(const VocalicOperators& ops);

/// What splitting a take gives back.
struct VocalicSplit {
    std::shared_ptr<Kakshya::DynamicSoundStream> vowels;
    std::shared_ptr<Kakshya::DynamicSoundStream> consonants;
    /// Label per frame of the original take (0 silence, 1 vowel, 2 consonant).
    std::vector<float> labels;
    uint32_t hop { 256 };
    uint32_t frame { 1536 };
    bool gpu { false };
};

/**
 * @brief Run a take through the grammar and put each part in its own stream.
 *
 * Runs the shaders synchronously, so call it from the graphics thread (a frame
 * metro), not the audio thread. Settings go in through the call's
 * ExecutionContext, so @p config applies to this call only.
 */
[[nodiscard]] VocalicSplit split_take(const ComputationGrammar& grammar,
    const std::shared_ptr<Kakshya::DynamicSoundStream>& take, VocalicConfig config = {});

} // namespace MayaFlux::Yantra
