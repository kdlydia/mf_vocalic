#include "VocalicGrammar.hpp"

namespace MayaFlux::Yantra {

namespace {

    std::shared_ptr<Kakshya::DynamicSoundStream> stream_of(const Kakshya::DataVariant& channel, uint32_t rate)
    {
        auto stream = std::make_shared<Kakshya::DynamicSoundStream>(rate, 1);
        const auto* samples = std::get_if<std::vector<double>>(&channel);
        if (samples && !samples->empty()) {
            stream->ensure_capacity(samples->size());
            stream->append_frames(std::span<const double>(*samples), 0);
            return stream->snapshot(samples->size());
        }
        return stream;
    }

    template <typename T>
    T meta_or(const std::unordered_map<std::string, std::any>& meta, const std::string& key, T fallback)
    {
        if (auto it = meta.find(key); it != meta.end()) {
            if (const auto* v = std::any_cast<T>(&it->second)) {
                return *v;
            }
        }
        return fallback;
    }

}

OperationRegistry vocalic_registry(const VocalicOperators& ops)
{
    auto registry = make_default_registry();
    registry.catalog<VocalicAnalyzer>(Vocalic::Classify,
        [analyzer = ops.analyzer](const VocalicConfig&) { return analyzer; });
    registry.catalog<VocalicExtractor>(Vocalic::Separate,
        [extractor = ops.extractor](const VocalicConfig&) { return extractor; });
    return registry;
}

std::shared_ptr<ComputationGrammar> vocalic_grammar(const VocalicOperators& ops)
{
    auto registry = std::make_shared<const OperationRegistry>(vocalic_registry(ops));
    return ComputationGrammar::Builder(registry)
        .rule(Vocalic::Classify)
        .rule(Vocalic::Separate)
        .build();
}

VocalicSplit split_take(const ComputationGrammar& grammar,
    const std::shared_ptr<Kakshya::DynamicSoundStream>& take, VocalicConfig config)
{
    VocalicSplit split;
    split.hop = config.hop;
    split.frame = config.frame;
    if (!take || take->get_num_frames() == 0) {
        return split;
    }

    const uint64_t frames = take->get_num_frames();
    std::vector<double> samples(frames);
    const auto& data = take->get_data();
    if (!data.empty()) {
        if (const auto* ch = std::get_if<std::vector<double>>(&data[0])) {
            std::copy_n(ch->begin(), std::min<size_t>(frames, ch->size()), samples.begin());
        }
    }

    ExecutionContext ctx;
    ctx.facets.set(config);

    Datum<std::vector<Kakshya::DataVariant>> input;
    input.data.emplace_back(std::move(samples));
    input.modality = Kakshya::DataModality::AUDIO_1D;

    const auto result = grammar.process<std::vector<Kakshya::DataVariant>>(
        input, Vocalic::Classify | Vocalic::Separate, ctx);

    if (result.data.size() < 3) {
        return split;
    }

    split.vowels = stream_of(result.data[0], config.sample_rate);
    split.consonants = stream_of(result.data[1], config.sample_rate);
    if (const auto* labels = std::get_if<std::vector<double>>(&result.data[2])) {
        split.labels.assign(labels->begin(), labels->end());
    }
    split.gpu = meta_or<bool>(result.metadata, "gpu", false) && meta_or<bool>(result.metadata, "gpu_split", false);
    return split;
}

} // namespace MayaFlux::Yantra
