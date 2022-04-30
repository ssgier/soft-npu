#include "TrivialNeuroComponentsFactory.hpp"
#include <neuro/Neuron.hpp>

namespace soft_npu{

std::unique_ptr<Neuron> TrivialNeuroComponentsFactory::makeNeuron(SizeType neuronId, std::shared_ptr<const NeuronParams> neuronParams) {
    return std::make_unique<Neuron>(neuronId, neuronParams);
}

std::unique_ptr<Synapse> TrivialNeuroComponentsFactory::makeSynapse(
        std::shared_ptr<const SynapseParams> synapseParams,
        const Neuron* preSynapticNeuron,
        Neuron* postSynapticNeuron,
        TimeType conductionDelay,
        ValueType initialWeight) {
    return std::make_unique<Synapse>(synapseParams, preSynapticNeuron, postSynapticNeuron, conductionDelay, initialWeight);
}

}
