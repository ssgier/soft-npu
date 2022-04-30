#include <params/ParamsFactories.hpp>
#include "PopulationGeneratorR2DSheet.hpp"
#include "TrivialNeuroComponentsFactory.hpp"
#include "SynapseInjectionR2DSheet.hpp"
#include <neuro/ChannelProjectorFactory.hpp>

namespace soft_npu {

PopulationGeneratorR2DSheet::PopulationGeneratorR2DSheet(const ParamsType& params,
                                                         RandomEngineType& randomEngine) :
                                                         params(params), randomEngine(randomEngine) {

}

std::unique_ptr<Population> PopulationGeneratorR2DSheet::generatePopulation() {

    auto generatorParams = params["populationGenerators"]["r2dSheet"];
    SizeType numNeurons = generatorParams["numNeurons"];
    ValueType pctInhibitoryNeurons = generatorParams["pctInhibitoryNeurons"];

    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(params);
    auto inhibitoryNeuronParams = ParamsFactories::extractInhibitoryNeuronParams(params);

    TrivialNeuroComponentsFactory factory;

    std::uniform_real_distribution<ValueType> uniformDistribution;

    auto population = std::make_unique<Population>();

    for (SizeType neuronId = 0; neuronId < numNeurons; ++ neuronId) {
        bool isInhibitory = neuronId >= (1 - pctInhibitoryNeurons) * numNeurons;

        Population::Location location = {
                uniformDistribution(randomEngine),
                uniformDistribution(randomEngine)
        };

        auto neuron = factory.makeNeuron(neuronId, isInhibitory ? inhibitoryNeuronParams : excitatoryNeuronParams);

        population->addNeuron(std::move(neuron), location);
    }

    SynapseInjectionR2DSheet::injectSynapses(params, randomEngine, factory, *population);

    population->setChannelProjector(ChannelProjectorFactory::createFromParams(params, randomEngine, *population));

    return population;
}

}
