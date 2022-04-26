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

        bool isExcitatorySynapse = synapse != nullptr && unscaledEpsp >= 0.0;

        if (isExcitatorySynapse && cycleContext.staticContext.synapseParams.shortTermPlasticityParams) {
            synapse->shortTermPlasticityState.update(cycleContext);
            scaledEpsp *= synapse->shortTermPlasticityState.lastValue;
        }

        targetNeuron.produceEPSP(cycleContext, cycleContext.time, scaledEpsp);

        if (isExcitatorySynapse) {
            targetNeuron.registerInboundSynapticTransmission(cycleContext, synapse);

            TimeType timePostMinusPre = targetNeuron.getLastSpikeTime() - cycleContext.time;

            if (cycleContext.staticContext.synapseParams.shortTermPlasticityParams) {
                synapse->shortTermPlasticityState.onTransmission(cycleContext);
            }

            if (- timePostMinusPre < cycleContext.staticContext.synapseParams.stdpCutOffTime) {
                synapse->handleSTDP(cycleContext, timePostMinusPre);
            }
        }
    }

private:
    Neuron& targetNeuron;
    Synapse* synapse;
    ValueType unscaledEpsp;
};

static_assert(std::is_trivially_destructible<TransmissionEvent>::value, "Must be trivially destructible");

}

