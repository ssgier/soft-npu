#pragma once

#include <Aliases.hpp>
#include <neuro/Neuron.hpp>
#include <neuro/Synapse.hpp>

namespace soft_npu {

struct NeuroComponentsFactory {
    virtual ~NeuroComponentsFactory() {};
    virtual std::unique_ptr<Neuron> makeNeuron(SizeType neuronId, std::shared_ptr<const NeuronParams> neuronParams) = 0;
    virtual std::unique_ptr<Synapse> makeSynapse(
            const Neuron* preSynapticNeuron,
            Neuron* postSynapticNeuron,
            TimeType conductionDelay,
            ValueType initialWeight) = 0;
};

}
