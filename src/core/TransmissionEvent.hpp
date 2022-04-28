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

            if (cycleContext.staticContext.synapseParams.shortTermPlasticityParams) {
                synapse->shortTermPlasticityState.onTransmission(cycleContext);
            }

            TimeType lastPostSynSpikeTime = targetNeuron.getLastSpikeTime();

            if (lastPostSynSpikeTime == cycleContext.time) {
                handleSTDP(cycleContext, targetNeuron.getNextToLastSpikeTime());
            } else {
                targetNeuron.registerInboundSynapticTransmission(cycleContext, synapse);
            }

            handleSTDP(cycleContext, lastPostSynSpikeTime);
        }
    }

private:
    Neuron& targetNeuron;
    Synapse* synapse;
    ValueType unscaledEpsp;

    void handleSTDP(const CycleContext& ctx, TimeType postSynSpikeTime) const {
        TimeType timePostMinusPre = ctx.time == postSynSpikeTime ? 0.0 : postSynSpikeTime - ctx.time;

        if (- timePostMinusPre < ctx.staticContext.synapseParams.stdpCutOffTime) {
            synapse->handleSTDP(ctx, timePostMinusPre);
        }
    }
};

static_assert(std::is_trivially_destructible<TransmissionEvent>::value, "Must be trivially destructible");

}

