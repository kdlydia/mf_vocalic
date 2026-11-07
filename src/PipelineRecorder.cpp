#include "PipelineRecorder.hpp"

#include <MayaFlux/Kriya/Capture.hpp>

#include <cmath>

namespace MayaFlux::Kriya {

PipelineRecorder::PipelineRecorder(std::shared_ptr<Buffers::BufferManager> manager,
    Vruta::TaskScheduler& scheduler,
    RecorderConfig config)
    : m_manager(std::move(manager))
    , m_scheduler(scheduler)
    , m_config(config)
    , m_rate(static_cast<uint32_t>(Buffers::s_registered_sample_rate))
{
    m_block = m_manager->get_buffer_size(Buffers::ProcessingToken::AUDIO_BACKEND);
    if (m_block == 0) {
        m_block = 512;
    }

    m_input = std::make_shared<Buffers::AudioBuffer>(m_config.input_channel, m_block);
    m_manager->register_input_listener(m_input, m_config.input_channel);

    if (m_config.monitor) {
        m_manager->add_buffer(m_input, Buffers::ProcessingToken::AUDIO_BACKEND, m_config.monitor_channel);
    }
}

bool PipelineRecorder::start()
{
    if (recording()) {
        return false;
    }
    settle();

    if (m_pipeline) {
        m_pipeline->end();
        m_retired.push_back(std::move(m_pipeline));
    }

    auto take = std::make_shared<Take>();
    take->stream = std::make_shared<Kakshya::DynamicSoundStream>(m_rate, 1);
    take->max_frames = static_cast<uint64_t>(m_config.max_seconds * m_rate);
    take->stream->ensure_capacity(take->max_frames);
    take->previous.assign(m_block, 0.0);

    // The frame count ends a take. The cycle count is only a cap, with room
    // for the silent and repeated blocks that are skipped.
    const auto cycles = 4 * static_cast<uint64_t>(std::ceil(m_config.max_seconds * m_rate / m_block)) + 4;

    auto pipeline = BufferPipeline::create(m_scheduler, m_manager);
    *pipeline >> CaptureBuilder(m_input)
                     .on_data_ready([take](Kakshya::DataVariant& data, uint32_t) {
                         if (const auto* block = std::get_if<std::vector<double>>(&data)) {
                             capture(*take, *block);
                         }
                     });
    pipeline->on_complete([take] { finish(*take); });

    m_current = take;
    m_pipeline = pipeline;

    pipeline->execute_for_cycles(cycles);
    return true;
}

void PipelineRecorder::stop()
{
    if (m_current) {
        m_current->stop.store(true, std::memory_order_release);
    }
}

bool PipelineRecorder::recording() const
{
    return m_current && !m_current->done.load(std::memory_order_acquire);
}

double PipelineRecorder::seconds() const
{
    if (m_current) {
        return static_cast<double>(m_current->frames.load(std::memory_order_relaxed)) / m_rate;
    }
    return m_last ? static_cast<double>(m_last->get_num_frames()) / m_rate : 0.0;
}

float PipelineRecorder::level() const
{
    return recording() ? m_current->level.load(std::memory_order_relaxed) : 0.0F;
}

uint32_t PipelineRecorder::takes()
{
    settle();
    return m_finished;
}

std::shared_ptr<Kakshya::DynamicSoundStream> PipelineRecorder::last_take()
{
    settle();
    return m_last;
}

// Once the audio thread has finished a take, copy out the frames written.
void PipelineRecorder::settle()
{
    if (!m_current || !m_current->done.load(std::memory_order_acquire)) {
        return;
    }
    const uint64_t frames = m_current->frames.load(std::memory_order_relaxed);
    if (frames > 0) {
        m_last = m_current->stream->snapshot(frames);
        ++m_finished;
    }
    m_current.reset();
}

// Audio thread.
void PipelineRecorder::capture(Take& take, const std::vector<double>& block)
{
    if (take.done.load(std::memory_order_relaxed)) {
        return;
    }

    if (take.stop.load(std::memory_order_acquire)) {
        finish(take);
        return;
    }

    if (block.empty()) {
        return;
    }

    double sum = 0.0;
    bool silent = true;
    for (const double x : block) {
        sum += x * x;
        silent = silent && x == 0.0;
    }
    if (silent) {
        return;
    }

    // A listener that was not refreshed still holds the last block.
    if (block.size() == take.previous.size() && std::equal(block.begin(), block.end(), take.previous.begin())) {
        return;
    }
    take.previous.assign(block.begin(), block.end());

    take.level.store(static_cast<float>(std::sqrt(sum / static_cast<double>(block.size()))), std::memory_order_relaxed);

    const uint64_t have = take.frames.load(std::memory_order_relaxed);
    const uint64_t room = take.max_frames > have ? take.max_frames - have : 0;
    const size_t count = std::min<size_t>(block.size(), room);

    if (count > 0) {
        take.stream->append_frames(std::span<const double>(block.data(), count), 0);
        take.frames.store(have + count, std::memory_order_relaxed);
    }

    if (count < block.size() || have + count >= take.max_frames) {
        finish(take);
    }
}

void PipelineRecorder::finish(Take& take)
{
    take.level.store(0.0F, std::memory_order_relaxed);
    take.done.store(true, std::memory_order_release);
}

} // namespace MayaFlux::Kriya
