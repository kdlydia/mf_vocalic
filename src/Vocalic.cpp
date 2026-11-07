#include "Vocalic.hpp"

#include <algorithm>
#include <cmath>

namespace MayaFlux::Kinesis::Vocalic {

namespace {

    uint32_t lag_of(float hz, uint32_t rate)
    {
        return static_cast<uint32_t>(std::lround(static_cast<double>(rate) / std::max(1.0F, hz)));
    }

}

uint32_t frame_count(size_t samples, uint32_t hop)
{
    return hop == 0 ? 0 : static_cast<uint32_t>((samples + hop - 1) / hop);
}

// Mirrors k_features_glsl.
std::vector<Frame> features(std::span<const float> x, const Yantra::VocalicConfig& cfg)
{
    const uint32_t frames = frame_count(x.size(), cfg.hop);
    const uint32_t min_lag = lag_of(cfg.max_hz, cfg.sample_rate);
    const uint32_t max_lag = lag_of(cfg.min_hz, cfg.sample_rate);

    std::vector<Frame> out(frames);
    for (uint32_t i = 0; i < frames; ++i) {
        const size_t start = static_cast<size_t>(i) * cfg.hop;
        const size_t len = std::min<size_t>(cfg.frame, x.size() - start);
        const float* w = x.data() + start;

        float e = 0.F;
        float z = 0.F;
        for (size_t k = 0; k < len; ++k) {
            e += w[k] * w[k];
            if (k > 0 && (w[k] >= 0.F) != (w[k - 1] >= 0.F)) {
                z += 1.F;
            }
        }
        out[i].rms = len > 0 ? std::sqrt(e / static_cast<float>(len)) : 0.F;
        out[i].zcr = len > 1 ? z / static_cast<float>(len - 1) : 0.F;

        float best = 0.F;
        if (out[i].rms > 1e-5F) {
            for (uint32_t lag = min_lag; lag <= max_lag && 2 * lag <= len; lag += 2) {
                float c = 0.F;
                float a = 0.F;
                float b = 0.F;
                for (size_t k = 0; k + lag < len; ++k) {
                    c += w[k] * w[k + lag];
                    a += w[k] * w[k];
                    b += w[k + lag] * w[k + lag];
                }
                best = std::max(best, c / std::sqrt(std::max(a * b, 1e-12F)));
            }
        }
        out[i].periodicity = best;
    }
    return out;
}

uint8_t classify(const Frame& f, const Yantra::VocalicConfig& cfg, float vowel_rms)
{
    if (f.rms < cfg.silence) {
        return 0;
    }
    if (f.periodicity >= cfg.voicing && f.zcr <= cfg.zcr_max && f.rms >= vowel_rms) {
        return 1;
    }
    return 2;
}

float loudest(std::span<const Frame> frames)
{
    float peak = 0.F;
    for (const auto& f : frames) {
        peak = std::max(peak, f.rms);
    }
    return peak;
}

// Mirrors k_labels_glsl.
std::vector<float> labels(std::span<const Frame> frames, const Yantra::VocalicConfig& cfg)
{
    const float vowel_rms = cfg.vowel_floor * loudest(frames);
    const auto n = static_cast<int64_t>(frames.size());
    const auto r = static_cast<int64_t>(cfg.smooth);

    std::vector<float> out(frames.size());
    for (int64_t i = 0; i < n; ++i) {
        int votes[3] = { 0, 0, 0 };
        for (int64_t j = i - r; j <= i + r; ++j) {
            const auto k = std::clamp<int64_t>(j, 0, n - 1);
            ++votes[classify(frames[static_cast<size_t>(k)], cfg, vowel_rms)];
        }
        const uint8_t own = classify(frames[static_cast<size_t>(i)], cfg, vowel_rms);
        uint8_t best = own;
        for (uint8_t c = 0; c < 3; ++c) {
            if (votes[c] > votes[best]) {
                best = c;
            }
        }
        out[static_cast<size_t>(i)] = static_cast<float>(best);
    }
    return out;
}

// Mirrors k_split_glsl.
std::pair<float, float> gains(std::span<const float> labels, size_t s, const Yantra::VocalicConfig& cfg)
{
    if (labels.empty()) {
        return { 0.F, 0.F };
    }
    const auto last = static_cast<int64_t>(labels.size()) - 1;
    const auto at = [&](int64_t k, float which) {
        return labels[static_cast<size_t>(std::clamp<int64_t>(k, 0, last))] == which ? 1.F : 0.F;
    };

    // Label k describes the frame centred at k * hop + frame / 2.
    const float pos = (static_cast<float>(s) - 0.5F * static_cast<float>(cfg.frame)) / static_cast<float>(cfg.hop);
    const auto r = static_cast<int64_t>(cfg.fade);

    float v = 0.F;
    float c = 0.F;
    for (int64_t j = -r; j <= r; ++j) {
        const float p = pos + static_cast<float>(j);
        const auto k = static_cast<int64_t>(std::floor(p));
        const float t = p - static_cast<float>(k);
        v += (1.F - t) * at(k, 1.F) + t * at(k + 1, 1.F);
        c += (1.F - t) * at(k, 2.F) + t * at(k + 1, 2.F);
    }
    const auto count = static_cast<float>(2 * r + 1);
    return { v / count, c / count };
}

void split(std::span<const float> x, std::span<const float> labels, const Yantra::VocalicConfig& cfg,
    std::vector<float>& vowels, std::vector<float>& consonants)
{
    vowels.assign(x.size(), 0.F);
    consonants.assign(x.size(), 0.F);
    for (size_t s = 0; s < x.size(); ++s) {
        const auto [v, c] = gains(labels, s, cfg);
        vowels[s] = x[s] * v;
        consonants[s] = x[s] * c;
    }
}

std::vector<double> compact(std::span<const float> masked, std::span<const float> gain)
{
    std::vector<double> out;
    out.reserve(masked.size());
    for (size_t s = 0; s < masked.size() && s < gain.size(); ++s) {
        if (gain[s] > 0.F) {
            out.push_back(masked[s]);
        }
    }
    return out;
}

} // namespace MayaFlux::Kinesis::Vocalic
