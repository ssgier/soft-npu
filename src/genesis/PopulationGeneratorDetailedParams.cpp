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

std::unique_ptr<Population> PopulationGeneratorDetailedParams::generatePopulation() {

    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(params);
    auto inhibitoryNeuronParams = ParamsFactories::extractInhibitoryNeuronParams(params);

    TrivialNeuroComponentsFactory factory;

    std::unordered_map<SizeType, Neuron*> neuronsById;

    auto population = std::make_unique<Population>(params);
    const auto& detailedParams = params["populationGenerators"]["pDetailedParams"];

    for (const auto& neuronJson : detailedParams["neurons"]) {
        SizeType neuronId = neuronJson["neuronId"];
        bool isInhibitory = neuronJson["isInhibitory"];

        auto neuron = factory.makeNeuron(neuronId, isInhibitory ? inhibitoryNeuronParams : excitatoryNeuronParams);
        neuronsById[neuronId] = neuron.get();
        population->addNeuron(std::move(neuron), Population::defaultLocation);
    }

    auto synapseJsons = detailedParams["synapses"];
    for (const auto& synapseJson : synapseJsons) {

        SizeType preSynapticNeuronId = synapseJson["preSynapticNeuronId"];
        SizeType postSynapticNeuronId = synapseJson["postSynapticNeuronId"];
        ValueType initialWeight = synapseJson["initialWeight"];
        TimeType conductionDelay = synapseJson["conductionDelay"];

        auto preSynapticNeuron = neuronsById[preSynapticNeuronId];
        auto postSynapticNeuron = neuronsById[postSynapticNeuronId];

        bool isInhibitory =  preSynapticNeuron->getNeuronParams()->isInhibitory;

        auto synapse = factory.makeSynapse(
                preSynapticNeuron,
                postSynapticNeuron,
                conductionDelay,
                initialWeight);

        preSynapticNeuron->addOutboundSynapse(synapse.get());

        if (isInhibitory) {
            population->addInhibitorySynapse(std::move(synapse));
        } else {
            population->addExcitatorySynapse(std::move(synapse));
        }
    }

    population->setChannelProjector(ChannelProjectorFactory::createFromParams(params, randomEngine, *population));

    return population;
}

}
