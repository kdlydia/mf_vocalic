# mf_vocalic

Records a voice with a buffer pipeline and splits it into a vowel stream and a consonant stream on the GPU, through the Yantra grammar.

## Pieces

| File | Namespace | Does |
| ---- | --------- | ---- |
| `PipelineRecorder` | `Kriya` | Records takes from the mic. One input listener; each take is a new `BufferPipeline` with one capture, run with `execute_for_cycles`. |
| `Vocalic` | `Yantra`, `Kinesis::Vocalic` | The `Vocalic` grammar enum, `VocalicConfig`, and the CPU version of the arithmetic. |
| `VocalicShaders` | `Yantra` | Three compute shaders as `ShaderSpec`s with `MF_KERNEL` bodies: frame features, labels, split. |
| `VocalicAnalyzer` | `Yantra` | `Vocalic::Classify`. Labels frames silence, vowel or consonant. |
| `VocalicExtractor` | `Yantra` | `Vocalic::Separate`. Splits the audio by the labels, with crossfades. |
| `VocalicGrammar` | `Yantra` | Registry and grammar for `Vocalic::Classify \| Vocalic::Separate`; `split_take` gives two new streams. |
| `VocalicPlayer` | `Kriya` | Loops the two streams, one `TapSet` each, with level and speed per part. |

A vowel frame is loud, periodic (normalised autocorrelation over 70 to 400 Hz) and has few zero crossings. Anything else above the silence threshold is a consonant. Nasals and liquids are periodic too, so they count as vowels.

Without a GPU (or in the offline tests) the operators run the same arithmetic on the CPU.

## Use

The app owns the managers and passes them in:

```cpp
auto recorder = std::make_shared<Kriya::PipelineRecorder>(get_buffer_manager(), *get_scheduler());
auto player = std::make_shared<Kriya::VocalicPlayer>(get_buffer_manager(), *get_scheduler());
Yantra::VocalicOperators ops;
auto grammar = Yantra::vocalic_grammar(ops);

recorder->start();
// ...
recorder->stop();

// later, on the graphics thread, once recorder->takes() has changed:
auto split = Yantra::split_take(*grammar, recorder->last_take(), {});
player->play(split.vowels, split.consonants);
```

The grammar can also be called directly:

```cpp
Yantra::VocalicConfig config;
config.voicing = 0.6F;
ExecutionContext ctx;
ctx.facets.set(config);
auto out = grammar->process<std::vector<Kakshya::DataVariant>>(input, Vocalic::Classify | Vocalic::Separate, ctx);
// out.data: vowels, consonants, frame labels
```

Settings in `VocalicConfig`: frame and hop, pitch range, silence, voicing, zero-crossing limit, vowel floor, label smoothing, crossfade, and `compact` (drop the silent parts, on by default).

## Tests

```sh
cmake -G Ninja -B test/build -S test -DCMAKE_BUILD_TYPE=Release
cmake --build test/build
cd test/build && ./test_mf_vocalic
```

Recorder takes against a sine, labels against a made-up vowel/noise/silence signal, the grammar split, and the three shaders compiled with `glslangValidator`.

R261107 checks the GPU against the CPU inside the engine: `VOCALIC_CHECK=16 ./build/R261107`.
