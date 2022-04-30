#include "Neuron.hpp"
#include "Synapse.hpp"
#include <cmath>
#include <neuro/Population.hpp>
#include <core/EventProcessor.hpp>
#include <core/TransmissionEvent.hpp>
#include <core/SynapticTransmissionStats.hpp>

namespace soft_npu {

Neuron::Neuron(SizeType neuronId, std::shared_ptr<const NeuronParams> neuronParams) noexcept :
        neuronParams(neuronParams), neuronId(neuronId), lastTime(0), lastVoltage(0),
        lastSpikeTime(std::numeric_limits<TimeType>::lowest()) {
}

std::shared_ptr<const NeuronParams> Neuron::getNeuronParams() const noexcept {
    return neuronParams;
}

SizeType Neuron::getNeuronId() const noexcept {
    return neuronId;
}

void Neuron::addOutboundSynapse(Synapse* synapse) {
    outboundSynapses.push_back(synapse);
}

void Neuron::addContinuousInhibitionSource(Neuron * source) {
    continuousInhibitionSources.push_back(source);
}

void Neuron::pushThresholdEvalEvent(const CycleContext& ctx) {
    ctx.staticContext.eventProcessor.pushFiringThresholdEvalEvent(*this);
}

void Neuron::processInboundOnSpike(const CycleContext& cycleContext) {
    for (const auto& synapticTransmissionInfo : synapticTransmissionSTDPBuffer) {
        synapticTransmissionInfo.synapse->handleSTDP(cycleContext, cycleContext.time, synapticTransmissionInfo.transmissionTime);
    }

    synapticTransmissionSTDPBuffer.clear();
}

void Neuron::registerInboundSynapticTransmission(const CycleContext& cycleContext, Synapse* synapse) {
    synapticTransmissionSTDPBuffer.emplace_back(synapse, cycleContext.time);
}

void Neuron::processOutboundOnSpike(const CycleContext& cycleContext) {

    ValueType signum = neuronParams->isInhibitory ? -1.0 : 1.0;

    for (Synapse* synapse : outboundSynapses) {

        cycleContext.staticContext.eventProcessor.pushSynapticTransmissionEvent(
                synapse->conductionDelay,
                synapse->weight * signum,
                synapse,
                *synapse->postSynapticNeuron);
    }
}

void Neuron::fire(const CycleContext& cycleContext) noexcept {

    lastVoltage = neuronParams->resetVoltage;
    lastTime = cycleContext.time + neuronParams->refractoryPeriod;
    lastSpikeTime = cycleContext.time;

    processInboundOnSpike(cycleContext);
    processOutboundOnSpike(cycleContext);

    cycleContext.staticContext.population.onSpike(cycleContext, *this);

    cycleContext.staticContext.synapticTransmissionStats.increaseTransmissionCount(outboundSynapses.size());
}

Neuron::SynapticTransmissionInfo::SynapticTransmissionInfo(Synapse *synapse, TimeType transmissionTime) : synapse(
        synapse), transmissionTime(transmissionTime) {}
}
