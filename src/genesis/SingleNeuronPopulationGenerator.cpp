#include "SingleNeuronPopulationGenerator.hpp"
#include <params/ParamsFactories.hpp>
#include <genesis/NeuroComponentsFactory.hpp>
#include <genesis/TrivialNeuroComponentsFactory.hpp>
#include <neuro/ChannelProjectorFactory.hpp>

namespace soft_npu {

SingleNeuronPopulationGenerator::SingleNeuronPopulationGenerator(
        const ParamsType& params, RandomEngineType& randomEngine) :
    params(params), randomEngine(randomEngine) {}

std::unique_ptr<Population> SingleNeuronPopulationGenerator::generatePopulation() {
    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(params);

    std::unique_ptr<NeuroComponentsFactory> factory = std::make_unique<TrivialNeuroComponentsFactory>();

    auto population = std::make_unique<Population>(params);
    population->addNeuron(factory->makeNeuron(0, excitatoryNeuronParams), Population::defaultLocation);

    population->setChannelProjector(ChannelProjectorFactory::createFromParams(params, randomEngine, *population));

    return population;
}

}
