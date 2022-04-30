#pragma once
#include "CycleContext.hpp"
#include "StaticContext.hpp"
#include <neuro/SynapseParams.hpp>
#include <neuro/Neuron.hpp>
#include <neuro/Synapse.hpp>

namespace soft_npu {

class TransmissionEvent {
public:
    TransmissionEvent(ValueType unscaledEpsp, Neuron &targetNeuron) :
        targetNeuron(targetNeuron), synapse(nullptr), unscaledEpsp(unscaledEpsp) {}

    TransmissionEvent(ValueType unscaledEpsp, Synapse *synapse, Neuron &targetNeuron) :
        targetNeuron(targetNeuron), synapse(synapse), unscaledEpsp(unscaledEpsp) {}

    void process(const CycleContext &cycleContext) const {

        ValueType scaledEpsp = unscaledEpsp;

        // note: avoiding direct check of inhibitory flag via pre-synaptic neuron pointer on the synapse,
        // because this would lead to a premature load of the synapse in the hot loop, causing a
        // significant performance penalty.
        bool isExcitatorySynapse = synapse != nullptr && unscaledEpsp >= 0.0;

        if (isExcitatorySynapse) {
            if (synapse->synapseParams->shortTermPlasticityParams) {
                synapse->shortTermPlasticityState.update(cycleContext, *synapse->synapseParams);
                scaledEpsp *= synapse->shortTermPlasticityState.lastValue;
                synapse->shortTermPlasticityState.onTransmission(cycleContext, *synapse->synapseParams);
            }

            targetNeuron.registerInboundSynapticTransmission(cycleContext, synapse);
            synapse->handleSTDP(cycleContext, targetNeuron.getLastSpikeTime(), cycleContext.time);
        }

        targetNeuron.produceEPSP(cycleContext, cycleContext.time, scaledEpsp);
    }

private:
    Neuron& targetNeuron;
    Synapse* synapse;
    ValueType unscaledEpsp;
};

static_assert(std::is_trivially_destructible<TransmissionEvent>::value, "Must be trivially destructible");

}

