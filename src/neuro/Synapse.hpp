#pragma once

#include <Aliases.hpp>
#include "STDPRule.hpp"
#include <core/CycleContext.hpp>
#include <core/StaticContext.hpp>
#include <boost/core/noncopyable.hpp>
#include "ShortTermPlasticityState.hpp"
#include <neuro/SynapseParams.hpp>

namespace soft_npu {

struct CycleContext;
class Neuron;

struct Synapse : private boost::noncopyable {
    Synapse(
            std::shared_ptr<const SynapseParams> synapseParams,
            const Neuron* preSynapticNeuron,
            Neuron* postSynapticNeuron,
            TimeType conductionDelay,
            ValueType initialWeight);

    Synapse(const Synapse& other) = delete;

    ShortTermPlasticityState shortTermPlasticityState;
    std::shared_ptr<const SynapseParams> synapseParams;
    const Neuron* preSynapticNeuron;
    Neuron* postSynapticNeuron;
    TimeType conductionDelay;
    ValueType weight;

    void handleSTDP(const CycleContext&, TimeType postSynSpikeTime, TimeType transmissionTime);
};

}





