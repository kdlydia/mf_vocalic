#pragma once

#include "VocalicAnalyzer.hpp"

#include <MayaFlux/Buffers/AudioBuffer.hpp>
#include <MayaFlux/Buffers/BufferManager.hpp>
#include <MayaFlux/Buffers/BufferProcessor.hpp>
#include <MayaFlux/Vruta/Scheduler.hpp>

#include <atomic>

namespace MayaFlux::Buffers {

enum class VocalicGateMode : uint8_t {
    Off = 0,
    Vowels = 1,
    Consonants = 2,
    Both = 3,
};

struct VocalicGateConfig {
    uint32_t input_channel { 0 };
    std::vector<uint32_t> output_channels { 0, 1 };
    /// How late the gated voice plays. The labels must be ready by then: half a
    /// frame, the time to the next analyse() call, and the analysis itself.
    double delay_seconds { 0.12 };
    /// Most audio one analyse() call labels. Anything older is skipped.
    double max_window_seconds { 0.5 };
    /// The loudness the vowel floor is measured against falls to this fraction each second.
    double reference_decay { 0.5 };
    /// Gain glide, in seconds, so label changes do not click.
    double glide_seconds { 0.004 };
    Yantra::VocalicConfig analysis {};
};

/**
 * @brief Plays the live microphone through a vowel and consonant gate.
 *
 * The audio stays on the audio thread. A processor on each output channel
 * writes the mic into a ring and plays it back delay_seconds late, with a gain
 * per part. analyse(), called from the graphics thread, runs the
 * Vocalic::Classify operator (the GPU chain) over the frames that arrived
 * since the last call and publishes their labels. Because the playback is
 * late, each sample is gated by the label of its own frame.
 *
 * AUDIO_PARALLEL is not used: it runs once per graphics frame, which is too
 * slow to carry the live voice.
 *
 * The mic is heard through the outputs, so use headphones.
 */
class VocalicGate {
public:
    struct State;

    static std::shared_ptr<VocalicGate> attach(BufferManager& manager, Vruta::TaskScheduler& scheduler,
        VocalicGateConfig config = {});

    void set_mode(VocalicGateMode mode);
    [[nodiscard]] VocalicGateMode mode() const;

    /// Linear gain of each part while it is let through.
    void set_levels(double vowels, double consonants);

    /// Label the frames that arrived since the last call. Graphics thread.
    /// @return Frames labelled.
    size_t analyse();

    /// Mark one frame, as analyse() does. Any thread but the audio thread.
    void publish(uint64_t frame, Yantra::VocalicLabel label);

    /// Label of the newest analysed frame.
    [[nodiscard]] Yantra::VocalicLabel latest() const;

    /// RMS of the last input block.
    [[nodiscard]] float level() const;

    /// Samples taken from the mic so far.
    [[nodiscard]] uint64_t samples_in() const;

    [[nodiscard]] uint32_t delay_samples() const;
    [[nodiscard]] const std::vector<std::shared_ptr<AudioBuffer>>& outputs() const { return m_outputs; }
    [[nodiscard]] const std::shared_ptr<Yantra::VocalicAnalyzer>& analyzer() const { return m_analyzer; }

private:
    VocalicGate() = default;

    VocalicGateConfig m_config;
    std::shared_ptr<State> m_state;
    std::vector<std::shared_ptr<AudioBuffer>> m_outputs;
    std::shared_ptr<Yantra::VocalicAnalyzer> m_analyzer;
    uint64_t m_next_frame { 0 };
    float m_reference { 0.F };
};

/// The processor behind each output of a VocalicGate. The first output's one also writes the ring.
class VocalicGateProcessor : public BufferProcessor {
public:
    VocalicGateProcessor(std::shared_ptr<VocalicGate::State> state, bool primary);

    void processing_function(const std::shared_ptr<Buffer>& buffer) override;
    [[nodiscard]] bool is_compatible_with(const std::shared_ptr<Buffer>& buffer) const override;

private:
    std::shared_ptr<VocalicGate::State> m_state;
    bool m_primary;
};

} // namespace MayaFlux::Buffers
