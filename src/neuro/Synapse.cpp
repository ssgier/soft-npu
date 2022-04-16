#include "Synapse.hpp"
#include "STDPRule.hpp"
#include <core/EventProcessor.hpp>
#include <core/DAergicModulator.hpp>

namespace soft_npu {

Synapse::Synapse(
        const Neuron* preSynapticNeuron,
        Neuron* postSynapticNeuron,
        TimeType conductionDelay,
        ValueType initialWeight) :
        preSynapticNeuron(preSynapticNeuron),
        postSynapticNeuron(postSynapticNeuron),
        conductionDelay(conductionDelay),
        weight(initialWeight) {
    if (preSynapticNeuron->getNeuronId() == postSynapticNeuron->getNeuronId()) {
        throw std::runtime_error("Circular synapses are not allowed");
    }
}

void Synapse::handleSTDP(const CycleContext& ctx, TimeType timePostMinusPre) {
    ValueType stdpValue = STDPRule::evaluateSTDPRule(ctx.staticContext.synapseParams, timePostMinusPre);
    ctx.staticContext.dopaminergicModulator.createEligibilityTrace(ctx, this, stdpValue);
}

}
