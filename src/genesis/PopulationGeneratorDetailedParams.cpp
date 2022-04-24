#include <params/ParamsFactories.hpp>
#include "PopulationGeneratorDetailedParams.hpp"
#include "NeuroComponentsFactory.hpp"
#include "TrivialNeuroComponentsFactory.hpp"
#include <neuro/ChannelProjectorFactory.hpp>

namespace soft_npu {

PopulationGeneratorDetailedParams::PopulationGeneratorDetailedParams(const ParamsType& params,
     RandomEngineType& randomEngine) :
     params(params),
     randomEngine(randomEngine) {

}

void PopulationGeneratorDetailedParams::makeAndSetNeurons(
        const ParamsType& details,
        NeuroComponentsFactory& factory,
        Population& population) const {

    std::unordered_map<SizeType, Neuron*> neuronsById;
    std::vector<std::pair<SizeType, SizeType>> inhibitionSourcePairings;
    std::unordered_map<std::string, std::shared_ptr<const NeuronParams>> neuronParamsByName;

    for (const auto& neuronJson : details["neurons"]) {
        SizeType neuronId = neuronJson["neuronId"];
        std::string neuronParamsName = neuronJson["neuronParamsName"];

        auto it = neuronParamsByName.find(neuronParamsName);
        if (it == neuronParamsByName.end()) {
            it = neuronParamsByName.emplace(neuronParamsName, ParamsFactories::extractNeuronParams(params, neuronParamsName)).first;
        }

        auto neuronParams = it->second;

        auto neuron = factory.makeNeuron(neuronId, neuronParams);

        auto sourcesIt = neuronJson.find("continuousInhibitionSourceNeuronIds");
        if (sourcesIt != neuronJson.end()) {
            const auto& continuousInhibitionSourceNeuronIds = *sourcesIt;

            std::transform(
                    continuousInhibitionSourceNeuronIds.cbegin(),
                    continuousInhibitionSourceNeuronIds.cend(),
                    std::back_inserter(inhibitionSourcePairings),
                    [neuronId](SizeType sourceNeuronId) {
                        return std::make_pair(sourceNeuronId, neuronId);
                    });
        }

        neuronsById[neuronId] = neuron.get();
        population.addNeuron(std::move(neuron), Population::defaultLocation);
    }

    for (const auto& inhibitionSourcePairing : inhibitionSourcePairings) {
        auto sourceNeuronId = inhibitionSourcePairing.first;
        auto targetNeuronId = inhibitionSourcePairing.second;

        neuronsById[targetNeuronId]->addContinuousInhibitionSource(neuronsById[sourceNeuronId]);
    }
}

void makeAndSetSynapses(
        const ParamsType& details,
        NeuroComponentsFactory& factory,
        Population& population) {

    auto synapseJsons = details["synapses"];
    for (const auto& synapseJson : synapseJsons) {

        SizeType preSynapticNeuronId = synapseJson["preSynapticNeuronId"];
        SizeType postSynapticNeuronId = synapseJson["postSynapticNeuronId"];
        ValueType initialWeight = synapseJson["initialWeight"];
        TimeType conductionDelay = synapseJson["conductionDelay"];

        auto& preSynapticNeuron = population.getNeuronById(preSynapticNeuronId);
        auto& postSynapticNeuron = population.getNeuronById(postSynapticNeuronId);

        bool isInhibitory =  preSynapticNeuron.getNeuronParams()->isInhibitory;

        auto synapse = factory.makeSynapse(
                &preSynapticNeuron,
                &postSynapticNeuron,
                conductionDelay,
                initialWeight);

        preSynapticNeuron.addOutboundSynapse(synapse.get());

        if (isInhibitory) {
            population.addInhibitorySynapse(std::move(synapse));
        } else {
            population.addExcitatorySynapse(std::move(synapse));
        }
    }
}

std::unique_ptr<Population> PopulationGeneratorDetailedParams::generatePopulation() {

    TrivialNeuroComponentsFactory factory;

    auto population = std::make_unique<Population>(params);
    const auto& detailedParams = params["populationGenerators"]["pDetailedParams"];

    makeAndSetNeurons(detailedParams, factory, *population);
    makeAndSetSynapses(detailedParams, factory, *population);

    population->setChannelProjector(ChannelProjectorFactory::createFromParams(params, randomEngine, *population));

    return population;
}

}
