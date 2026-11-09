#pragma once

#include <MayaFlux/Portal/Graphics/ShaderSpec.hpp>

/**
 * Compute shaders for the vocalic split, declared as ShaderSpecs. ShaderFoundry
 * writes the bindings and push constant block, and compiles them at run time.
 * Each kernel mirrors a function in Vocalic.cpp.
 *
 * Push constant fields are declared in the same order as the structs the
 * operators push, all 4-byte scalars.
 */
namespace MayaFlux::Yantra::VocalicShaders {

namespace G = Portal::Graphics;
using Format = Kakshya::GpuDataFormat;

/// Lag search is split over this many invocations per frame; the reduce stage takes the max.
inline constexpr uint32_t k_lag_chunks = 16;

/**
 * The classify chain runs three kernels in one submission. They share five
 * buffers, at the same binding in every kernel:
 *   0 x               the samples
 *   1 partial         per frame and lag chunk: rms, zero crossings, best periodicity
 *   2 frame_features  per frame: rms, zero crossings, periodicity
 *   3 peak            the loudest frame's rms, as float bits (rms is never negative)
 *   4 labels          per frame
 */
namespace Slot {
    inline constexpr uint32_t x = 0;
    inline constexpr uint32_t partial = 1;
    inline constexpr uint32_t frame_features = 2;
    inline constexpr uint32_t peak = 3;
    inline constexpr uint32_t labels = 4;
}

/// One invocation per frame and lag chunk: RMS, zero crossings per sample, and
/// the best normalised autocorrelation over every chunks-th lag.
inline G::ShaderSpec features()
{
    return G::ShaderSpec::Assemble {}
        .start_binding(Slot::x)
        .ssbo("x", G::BindingDirection::Input, Format::FLOAT32)
        .ssbo("partial", G::BindingDirection::Output, Format::VEC4_F32)
        .pc("n", Format::UINT32)
        .pc("frame", Format::UINT32)
        .pc("hop", Format::UINT32)
        .pc("frames", Format::UINT32)
        .pc("min_lag", Format::UINT32)
        .pc("max_lag", Format::UINT32)
        .pc("chunks", Format::UINT32)
        .kernel(MF_KERNEL([](float* x, vec4* partial, uint n, uint frame, uint hop, uint frames, uint min_lag, uint max_lag, uint chunks, uint i) {
            if (i >= frames * chunks)
                return;

            uint f = i / chunks;
            uint part = i % chunks;
            uint start = f * hop;
            uint len = min(frame, n - start);

            float e = 0.0;
            float z = 0.0;
            for (uint k = 0u; k < len; ++k) {
                float v = x[start + k];
                e += v * v;
                if (k > 0u && ((v >= 0.0) != (x[start + k - 1u] >= 0.0)))
                    z += 1.0;
            }
            float rms = len > 0u ? sqrt(e / float(len)) : 0.0;
            float zcr = len > 1u ? z / float(len - 1u) : 0.0;

            float best = 0.0;
            if (rms > 1e-5) {
                for (uint lag = min_lag + 2u * part; lag <= max_lag && 2u * lag <= len; lag += 2u * chunks) {
                    float c = 0.0;
                    float a = 0.0;
                    float b = 0.0;
                    for (uint k = 0u; k + lag < len; ++k) {
                        float u = x[start + k];
                        float w = x[start + k + lag];
                        c += u * w;
                        a += u * u;
                        b += w * w;
                    }
                    best = max(best, c / sqrt(max(a * b, 1e-12)));
                }
            }

            partial[i] = vec4(rms, zcr, best, 0.0);
        }))
        .build();
}

/// One invocation per frame: the best periodicity over its lag chunks, and the loudest frame.
inline G::ShaderSpec reduce()
{
    return G::ShaderSpec::Assemble {}
        .start_binding(Slot::partial)
        .ssbo("partial", G::BindingDirection::Input, Format::VEC4_F32)
        .ssbo("frame_features", G::BindingDirection::Output, Format::VEC4_F32)
        .ssbo("peak", G::BindingDirection::InOut, Format::UINT32)
        .pc("frames", Format::UINT32)
        .pc("chunks", Format::UINT32)
        .kernel(MF_KERNEL([](vec4* partial, vec4* frame_features, uint* peak, uint frames, uint chunks, uint i) {
            if (i >= frames)
                return;

            uint base = i * chunks;
            float best = 0.0;
            for (uint c = 0u; c < chunks; ++c)
                best = max(best, partial[base + c].z);

            vec4 head = partial[base];
            frame_features[i] = vec4(head.x, head.y, best, 0.0);
            atomicMax(peak[0], floatBitsToUint(head.x));
        }))
        .build();
}

/// One invocation per frame: label, then a majority vote over neighbours.
inline G::ShaderSpec labels()
{
    return G::ShaderSpec::Assemble {}
        .start_binding(Slot::frame_features)
        .ssbo("frame_features", G::BindingDirection::Input, Format::VEC4_F32)
        .ssbo("peak", G::BindingDirection::Input, Format::UINT32)
        .ssbo("labels", G::BindingDirection::Output, Format::FLOAT32)
        .pc("frames", Format::UINT32)
        .pc("smooth_frames", Format::UINT32)
        .pc("silence", Format::FLOAT32)
        .pc("voicing", Format::FLOAT32)
        .pc("zcr_max", Format::FLOAT32)
        .pc("vowel_floor", Format::FLOAT32)
        .function("uint", "classify", "vec4 v, float vowel_rms", R"(
    if (v.x < pc.silence)
        return 0u;
    if (v.z >= pc.voicing && v.y <= pc.zcr_max && v.x >= vowel_rms)
        return 1u;
    return 2u;)")
        .kernel(MF_KERNEL([](vec4* frame_features, uint* peak, float* labels, uint frames, uint smooth_frames, float silence, float voicing, float zcr_max, float vowel_floor, uint i) {
            if (i >= frames)
                return;

            float vowel_rms = vowel_floor * uintBitsToFloat(peak[0]);
            int votes[3] = int[3](0, 0, 0);
            int r = int(smooth_frames);
            int last = int(frames) - 1;
            for (int j = int(i) - r; j <= int(i) + r; ++j)
                votes[classify(frame_features[clamp(j, 0, last)], vowel_rms)] += 1;

            uint best = classify(frame_features[i], vowel_rms);
            for (uint c = 0u; c < 3u; ++c) {
                if (votes[c] > votes[best])
                    best = c;
            }
            labels[i] = float(best);
        }))
        .build();
}

/// One invocation per sample: vowel and consonant parts, with crossfades.
/// Writes (vowel, consonant, vowel gain, consonant gain).
inline G::ShaderSpec split()
{
    return G::ShaderSpec::Assemble {}
        .ssbo("x", G::BindingDirection::Input, Format::FLOAT32)
        .ssbo("labels", G::BindingDirection::Input, Format::FLOAT32)
        .ssbo("parts", G::BindingDirection::Output, Format::VEC4_F32)
        .pc("n", Format::UINT32)
        .pc("frame", Format::UINT32)
        .pc("hop", Format::UINT32)
        .pc("frames", Format::UINT32)
        .pc("fade", Format::UINT32)
        .function("float", "label_is", "int k, float which", R"(
    return labels[clamp(k, 0, int(pc.frames) - 1)] == which ? 1.0 : 0.0;)")
        .kernel(MF_KERNEL([](float* x, float* labels, vec4* parts, uint n, uint frame, uint hop, uint frames, uint fade, uint i) {
            if (i >= n)
                return;

            float pos = (float(i) - 0.5 * float(frame)) / float(hop);
            int r = int(fade);

            float v = 0.0;
            float c = 0.0;
            for (int j = -r; j <= r; ++j) {
                float q = pos + float(j);
                int k = int(floor(q));
                float t = q - float(k);
                v += (1.0 - t) * label_is(k, 1.0) + t * label_is(k + 1, 1.0);
                c += (1.0 - t) * label_is(k, 2.0) + t * label_is(k + 1, 2.0);
            }
            float count = float(2 * r + 1);
            v /= count;
            c /= count;

            parts[i] = vec4(x[i] * v, x[i] * c, v, c);
        }))
        .build();
}

} // namespace MayaFlux::Yantra::VocalicShaders
