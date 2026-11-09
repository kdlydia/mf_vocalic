# Log

Working notes for mf_vocalic and R261107. Newest at the bottom.

## 2026-11-07

Recorder: BufferPipeline can record if it is wrapped. `capture_input_from` makes a new listener on an output channel each call, so the mic reaches the speakers. `PipelineRecorder` registers one listener and never puts it on an output. The input listener is rewritten on every input call, so a capture sees fresh data without the chain running. Each take is a pipeline run with `execute_for_cycles`; the frame count ends the take and the cycle count is only a cap (half the blocks are silent and skipped). Stopping sets a flag the capture reads on the audio thread. The take is copied out with `snapshot` once the audio thread has marked it done.

Split: `Vocalic::Classify` (an Analyzer) and `Vocalic::Separate` (an Extractor), catalogued in an `OperationRegistry` and composed in a `ComputationGrammar`. Settings go per call through `ExecutionContext::facets`.

Shaders: a GLSL string in `GpuComputeConfig::shader_path` fails, since the foundry cannot tell its stage. `ShaderSpec::Assemble` with an `MF_KERNEL` body, then `config_from_spec` and `bindings_from_spec`, is the way. The emitter adds no bounds check and starts bindings at 0.

First features kernel ran one thread per frame and was slower than the CPU (539 ms against 857 ms for 10 s). Splitting the lag search over 16 threads per frame: 126 ms against 1.66 s for 20 s. GPU and CPU labels agree on every frame.

An exception inside a `schedule_sequence` task ends the program (the coroutine terminates). The app catches around the split.

Next: try it on real speech and tune voicing and the vowel floor; nasals count as vowels.

## 2026-11-09

New nightly (main a8aeb5f5). Install it from the AUR clone after deleting the cached tarball: every nightly has the same file name, so paru and makepkg reused the old one.

Classifier on the new dispatch API: one context, five shared buffers, and features, reduce and labels recorded into one submission with `record_stages`. The loudest frame is found on the GPU with `atomicMax` on its float bits. 12 ms for 1.25 s, against about 25 ms with two contexts; labels still match the CPU on every frame. ShaderSpec kernels still use set 0 bindings (pushed now); ShaderSpec has no pointer arguments yet.

Live mode: `VocalicGate`. AUDIO_PARALLEL and `AudioContextProcessor` run on the graphics thread once per frame, which cannot carry 48 kHz audio, and there is no mic path into that unit yet. So the voice stays on the audio thread: a ring, played back 0.12 s late, gated by labels that `analyse()` computes on the GPU from the graphics thread. Offline test: vowel mode passes all of the vowel and none of the noise, consonant mode the reverse.

The player now normalises each stream to a 0.9 peak (takes were too quiet).

Next: try live mode and the split on real speech; tune voicing and the vowel floor.

