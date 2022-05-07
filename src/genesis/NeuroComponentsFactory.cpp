#include "NeuroComponentsFactory.hpp"
#include <neuro/Population.hpp>

namespace soft_npu {

std::unique_ptr<Neuron> NeuroComponentsFactory::makeNeuron(std::shared_ptr<const NeuronParams> neuronParams,
        const Population& population) {
    auto nextNeuronId = population.getPopulationSize();
    return makeNeuron(nextNeuronId, neuronParams);
}

}
