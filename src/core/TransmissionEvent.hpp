#pragma once
#include "CycleContext.hpp"
#include <neuro/Neuron.hpp>
#include <neuro/Synapse.hpp>

namespace soft_npu {

class TransmissionEvent {
public:
    TransmissionEvent(ValueType epsp, Neuron &targetNeuron) :
        targetNeuron(targetNeuron), synapse(nullptr), epsp(epsp) {}

    TransmissionEvent(ValueType epsp, Synapse *synapse, Neuron &targetNeuron) :
        targetNeuron(targetNeuron), synapse(synapse), epsp(epsp) {}

    void process(const CycleContext &cycleContext) const {
        targetNeuron.produceEPSP(cycleContext, cycleContext.time, epsp);

        if (synapse != nullptr && epsp >= 0) {
            targetNeuron.registerInboundSynapticTransmission(cycleContext, synapse);

            TimeType timePostMinusPre = targetNeuron.getLastSpikeTime() - cycleContext.time;

            if (- timePostMinusPre < cycleContext.staticContext.synapseParams.stdpCutOffTime) {
                synapse->handleSTDP(cycleContext, timePostMinusPre);
            }
        }
    }

private:
    Neuron& targetNeuron;
    Synapse* synapse;
    ValueType epsp;
};

static_assert(std::is_trivially_destructible<TransmissionEvent>::value, "Must be trivially destructible");

}

