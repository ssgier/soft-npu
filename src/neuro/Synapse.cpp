#include "Synapse.hpp"
#include "STDPRule.hpp"
#include <core/EventProcessor.hpp>
#include <core/DAergicModulator.hpp>

namespace soft_npu {

Synapse::Synapse(
        std::shared_ptr<const SynapseParams> synapseParams,
        const Neuron* preSynapticNeuron,
        Neuron* postSynapticNeuron,
        TimeType conductionDelay,
        ValueType initialWeight) :
        synapseParams(synapseParams),
        preSynapticNeuron(preSynapticNeuron),
        postSynapticNeuron(postSynapticNeuron),
        conductionDelay(conductionDelay),
        weight(initialWeight) {
    if (preSynapticNeuron->getNeuronId() == postSynapticNeuron->getNeuronId()) {
        throw std::runtime_error("Circular synapses are not allowed");
    }
}

void Synapse::handleSTDP(const CycleContext& ctx, TimeType postSynSpikeTime, TimeType transmissionTime) {
    auto timePostMinusPre = postSynSpikeTime == transmissionTime ? 0.0 : postSynSpikeTime - transmissionTime;
    if (std::abs(timePostMinusPre) < synapseParams->stdpCutOffTime) {
        ValueType stdpValue = STDPRule::evaluateSTDPRule(*synapseParams, timePostMinusPre);
        ctx.staticContext.dopaminergicModulator.createEligibilityTrace(ctx, this, stdpValue);
    }
}

}
