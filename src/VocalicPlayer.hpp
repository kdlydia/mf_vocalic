#pragma once

#include <MayaFlux/Buffers/BufferManager.hpp>
#include <MayaFlux/Kakshya/Source/DynamicSoundStream.hpp>
#include <MayaFlux/Kriya/TapSet.hpp>
#include <MayaFlux/Vruta/Scheduler.hpp>

namespace MayaFlux::Kriya {

enum class VocalicPart : uint8_t {
    Vowels = 0,
    Consonants = 1,
};

/**
 * @brief Loops the vowel and consonant streams of a take, each with its own TapSet.
 *
 * play() replaces what was playing. Level and speed are kept per part and
 * carry over to the next take. Each stream is scaled so its peak reaches
 * peak() (0.9 by default), so a quiet recording still plays loud; level()
 * applies on top of that. Stopped tap sets are kept, not freed, since a
 * sampler freed while the engine runs can crash the audio thread.
 *
 * Call from one thread, not the audio thread.
 */
class VocalicPlayer {
public:
    VocalicPlayer(std::shared_ptr<Buffers::BufferManager> manager, Vruta::TaskScheduler& scheduler,
        std::vector<uint32_t> channels = { 0, 1 });

    void play(const std::shared_ptr<Kakshya::DynamicSoundStream>& vowels,
        const std::shared_ptr<Kakshya::DynamicSoundStream>& consonants);

    void stop();

    void level(VocalicPart part, double gain);
    /// Relative to the recording. Negative reads backwards, zero holds.
    void speed(VocalicPart part, double ratio);

    /// Scale each stream to this peak when it starts playing; zero or less turns it off.
    void normalize(double peak) { m_peak = peak; }
    [[nodiscard]] double peak() const { return m_peak; }

    [[nodiscard]] double level(VocalicPart part) const { return m_voices[index(part)].level; }
    [[nodiscard]] double speed(VocalicPart part) const { return m_voices[index(part)].speed; }
    [[nodiscard]] bool playing() const { return m_playing; }

private:
    struct Voice {
        std::optional<TapSet> taps;
        double level { 0.8 };
        double speed { 1.0 };
        double scale { 1.0 }; ///< Normalisation for the stream now playing.
    };

    static size_t index(VocalicPart part) { return static_cast<size_t>(part); }
    void start(Voice& voice, const std::shared_ptr<Kakshya::DynamicSoundStream>& stream);
    void retire(Voice& voice);

    std::shared_ptr<Buffers::BufferManager> m_manager;
    Vruta::TaskScheduler& m_scheduler;
    std::vector<uint32_t> m_channels;
    uint32_t m_block { 512 };

    Voice m_voices[2];
    std::vector<TapSet> m_retired;
    bool m_playing { false };
    double m_peak { 0.9 };
};

} // namespace MayaFlux::Kriya
