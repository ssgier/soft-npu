#pragma once

#include "NeuroComponentsFactory.hpp"

namespace soft_npu {

struct TrivialNeuroComponentsFactory : public NeuroComponentsFactory {
public:
    std::unique_ptr<Neuron> makeNeuron(SizeType neuronId, std::shared_ptr<const NeuronParams> neuronParams) override;
    std::unique_ptr<Synapse> makeSynapse(
            const Neuron* preSynapticNeuron,
            Neuron* postSynapticNeuron,
            TimeType conductionDelay,
            ValueType initialWeight) override;
};

}




