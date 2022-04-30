
#include <params/ParamsFactories.hpp>
#include "PopulationGeneratorP1000.hpp"
#include "NeuroComponentsFactory.hpp"
#include "TrivialNeuroComponentsFactory.hpp"
#include <neuro/ChannelProjectorFactory.hpp>
#include <algorithm>

namespace soft_npu {


PopulationGeneratorP1000::PopulationGeneratorP1000(
        const ParamsType& params,
        RandomEngineType& randomEngine) :
    params(params),
    randomEngine(randomEngine) {
}

std::unique_ptr<Population> soft_npu::PopulationGeneratorP1000::generatePopulation() {

    constexpr SizeType numNeurons = 1000;
    constexpr double connectionProbability = 0.1; // TODO: move to config 
    constexpr TimeType maxConductionDelay = 20e-3; // move to config 
    const auto p1000Params = params["populationGenerators"]["p1000"];
    const TimeType inhibitoryConductionDelayDeterministicPart = p1000Params["inhibitoryConductionDelayDeterministicPart"];
    const TimeType inhibitoryConductionDelayRandomPart = p1000Params["inhibitoryConductionDelayRandomPart"];
    const ValueType inhibitorySynapseWeight = p1000Params["inhibitorySynapseWeight"];
    const ValueType excitatorySynapseInitialWeight = p1000Params["excitatorySynapseInitialWeight"];

    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(params);
    auto inhibitoryNeuronParams = ParamsFactories::extractInhibitoryNeuronParams(params);
    auto synapseParams = ParamsFactories::extractSynapseParams(params);

    TrivialNeuroComponentsFactory factory;

    std::unordered_map<SizeType, Neuron*> neuronsById;

    auto population = std::make_unique<Population>();

    for (SizeType neuronId = 0; neuronId < numNeurons; ++neuronId) {
        bool isInhibitory = neuronId >= 800;

        auto neuron = factory.makeNeuron(neuronId, isInhibitory ? inhibitoryNeuronParams : excitatoryNeuronParams);
        neuronsById[neuronId] = neuron.get();
        population->addNeuron(std::move(neuron), Population::defaultLocation);
    }

    std::uniform_real_distribution<double> uniformDistribution;

    for (SizeType preSynapticNeuronId = 0; preSynapticNeuronId < numNeurons; ++ preSynapticNeuronId) {

        for (SizeType postSynapticNeuronId = 0; postSynapticNeuronId < numNeurons; ++postSynapticNeuronId) {
            if (preSynapticNeuronId != postSynapticNeuronId && uniformDistribution(randomEngine) < connectionProbability) {

                auto preSynapticNeuron = neuronsById[preSynapticNeuronId];
                auto postSynapticNeuron = neuronsById[postSynapticNeuronId];

                bool isInhibitory = preSynapticNeuron->getNeuronParams()->isInhibitory;
                ValueType initialWeight = isInhibitory ? inhibitorySynapseWeight : excitatorySynapseInitialWeight;

                TimeType conductionDelay = isInhibitory ?
                                           inhibitoryConductionDelayDeterministicPart + uniformDistribution(randomEngine) * inhibitoryConductionDelayRandomPart :
                                           std::max(1e-3, uniformDistribution(randomEngine) * maxConductionDelay);

                auto synapse = factory.makeSynapse(
                        synapseParams,
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
        }
    }

    population->setChannelProjector(ChannelProjectorFactory::createFromParams(params, randomEngine, *population));

    return population;
}


}
