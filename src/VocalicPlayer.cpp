#include "VocalicPlayer.hpp"

namespace MayaFlux::Kriya {

namespace {

    double peak_of(const std::shared_ptr<Kakshya::DynamicSoundStream>& stream)
    {
        const auto& data = stream->get_data();
        const auto* x = data.empty() ? nullptr : std::get_if<std::vector<double>>(&data[0]);
        if (!x) {
            return 0.0;
        }
        const auto n = std::min<size_t>(x->size(), stream->get_num_frames());
        double peak = 0.0;
        for (size_t i = 0; i < n; ++i) {
            peak = std::max(peak, std::abs((*x)[i]));
        }
        return peak;
    }

}

VocalicPlayer::VocalicPlayer(std::shared_ptr<Buffers::BufferManager> manager, Vruta::TaskScheduler& scheduler,
    std::vector<uint32_t> channels)
    : m_manager(std::move(manager))
    , m_scheduler(scheduler)
    , m_channels(std::move(channels))
{
    m_block = m_manager->get_buffer_size(Buffers::ProcessingToken::AUDIO_BACKEND);
    if (m_block == 0) {
        m_block = 512;
    }
}

void VocalicPlayer::retire(Voice& voice)
{
    if (voice.taps) {
        voice.taps->stop();
        m_retired.push_back(std::move(*voice.taps));
        voice.taps.reset();
    }
}

void VocalicPlayer::start(Voice& voice, const std::shared_ptr<Kakshya::DynamicSoundStream>& stream)
{
    retire(voice);
    if (!stream || stream->get_num_frames() == 0) {
        return;
    }
    const double peak = m_peak > 0.0 ? peak_of(stream) : 0.0;
    // Up to 40 dB of make-up gain, so a take of near silence is not blown up to noise.
    voice.scale = peak > 0.0 ? std::min(m_peak / peak, 100.0) : 1.0;

    TapSetBuilder builder(stream, *m_manager, m_scheduler, m_block);
    builder.tap().speed(voice.speed).level(voice.level * voice.scale).on_channels(m_channels);
    builder.loop(true);
    voice.taps = builder.start();
}

void VocalicPlayer::play(const std::shared_ptr<Kakshya::DynamicSoundStream>& vowels,
    const std::shared_ptr<Kakshya::DynamicSoundStream>& consonants)
{
    start(m_voices[index(VocalicPart::Vowels)], vowels);
    start(m_voices[index(VocalicPart::Consonants)], consonants);
    m_playing = true;
}

void VocalicPlayer::stop()
{
    for (auto& voice : m_voices) {
        retire(voice);
    }
    m_playing = false;
}

void VocalicPlayer::level(VocalicPart part, double gain)
{
    auto& voice = m_voices[index(part)];
    voice.level = gain;
    if (voice.taps) {
        voice.taps->level(0, gain * voice.scale);
    }
}

void VocalicPlayer::speed(VocalicPart part, double ratio)
{
    auto& voice = m_voices[index(part)];
    voice.speed = ratio;
    if (voice.taps) {
        voice.taps->speed(0, ratio);
    }
}

} // namespace MayaFlux::Kriya
