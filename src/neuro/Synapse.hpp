#pragma once

#include <Aliases.hpp>
#include "STDPRule.hpp"
#include <core/CycleContext.hpp>
#include <core/StaticContext.hpp>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

struct CycleContext;
class Neuron;

struct Synapse : private boost::noncopyable {
    Synapse(const Neuron* preSynapticNeuron,
            Neuron* postSynapticNeuron,
            TimeType conductionDelay,
            ValueType initialWeight);

    Synapse(const Synapse& other) = delete;

    const Neuron* preSynapticNeuron;
    Neuron* postSynapticNeuron;
    TimeType conductionDelay;
    ValueType weight;

    void handleSTDP(const CycleContext&, TimeType timePostMinusPre);
};

}





