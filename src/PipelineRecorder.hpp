#pragma once

#include <MayaFlux/Buffers/AudioBuffer.hpp>
#include <MayaFlux/Buffers/BufferManager.hpp>
#include <MayaFlux/Kakshya/Source/DynamicSoundStream.hpp>
#include <MayaFlux/Kriya/BufferPipeline.hpp>
#include <MayaFlux/Vruta/Scheduler.hpp>

#include <atomic>

namespace MayaFlux::Kriya {

struct RecorderConfig {
    uint32_t input_channel { 0 };
    double max_seconds { 20.0 };
    // Put the listener on an output channel so the mic is heard while recording.
    bool monitor { false };
    // Output channel used when monitor is on.
    uint32_t monitor_channel { 0 };
};

/**
 * @brief Records takes from the microphone with a BufferPipeline.
 *
 * One input listener is made when the recorder is made. Each take is a new
 * pipeline with one capture of that listener, run with execute_for_cycles for
 * at most max_seconds. Blocks are appended to a fresh mono DynamicSoundStream.
 * All-zero blocks and repeats of the previous block are skipped.
 *
 * The listener is not on an output channel unless monitor is set, so the mic
 * does not reach the speakers.
 *
 * Call everything from one thread (the UI). The capture runs on the audio
 * thread; a take is handed over only after the audio thread has finished it.
 */
class PipelineRecorder {
public:
    PipelineRecorder(std::shared_ptr<Buffers::BufferManager> manager,
        Vruta::TaskScheduler& scheduler,
        RecorderConfig config = {});

    /// Start a take. Returns false while one is still running.
    bool start();

    /// Ask the running take to end. It ends at the next block.
    void stop();

    [[nodiscard]] bool recording() const;

    /// Length of the running or last take.
    [[nodiscard]] double seconds() const;

    /// RMS of the last captured block, 0 when not recording.
    [[nodiscard]] float level() const;

    /// Number of finished takes. Changes when a new take is ready.
    [[nodiscard]] uint32_t takes();

    /// The last finished take, trimmed to its length, or nullptr.
    [[nodiscard]] std::shared_ptr<Kakshya::DynamicSoundStream> last_take();

    [[nodiscard]] const std::shared_ptr<Buffers::AudioBuffer>& input() const { return m_input; }
    [[nodiscard]] uint32_t sample_rate() const { return m_rate; }

private:
    struct Take {
        std::shared_ptr<Kakshya::DynamicSoundStream> stream;
        uint64_t max_frames { 0 };
        std::atomic<bool> stop { false };
        std::atomic<bool> done { false };
        std::atomic<uint64_t> frames { 0 };
        std::atomic<float> level { 0.0F };
        std::vector<double> previous;
    };

    static void capture(Take& take, const std::vector<double>& block);
    static void finish(Take& take);
    void settle();

    std::shared_ptr<Buffers::BufferManager> m_manager;
    Vruta::TaskScheduler& m_scheduler;
    RecorderConfig m_config;
    uint32_t m_rate { 48000 };
    uint32_t m_block { 512 };

    std::shared_ptr<Buffers::AudioBuffer> m_input;
    std::shared_ptr<Take> m_current;
    std::shared_ptr<BufferPipeline> m_pipeline;
    // Ended pipelines are kept: a cancelled routine may still run one step.
    std::vector<std::shared_ptr<BufferPipeline>> m_retired;

    std::shared_ptr<Kakshya::DynamicSoundStream> m_last;
    uint32_t m_finished { 0 };
};

} // namespace MayaFlux::Kriya
