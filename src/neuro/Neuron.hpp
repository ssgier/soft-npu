#pragma once

#include <core/CycleContext.hpp>
#include "NeuronParams.hpp"
#include <memory>
#include <vector>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

struct Synapse;

class Neuron : private boost::noncopyable {
public:
    Neuron(SizeType neuronId, std::shared_ptr<const NeuronParams> neuronParams) noexcept;

    void produceEPSP(const CycleContext &cycleContext, TimeType time, ValueType epsp) noexcept {
        if (!isRefractoryPeriod(time)) {
            update(time);

            lastVoltage = std::max(lastVoltage + epsp, neuronParams->voltageFloor);

            if (lastVoltage >= neuronParams->thresholdVoltage) {
                fire(cycleContext);
            }
        }
    }

    void addOutboundSynapse(Synapse* synapse);

    ValueType getMembraneVoltage(TimeType time) const noexcept {
        TimeType timeSinceLastEvaluation = time - lastTime;
        return lastVoltage * exp(- timeSinceLastEvaluation * neuronParams->timeConstantInverse);
    }

    SizeType getNeuronId() const noexcept;

    std::shared_ptr<const NeuronParams> getNeuronParams() const noexcept;

    TimeType getLastSpikeTime() const noexcept;

    void registerInboundSynapticTransmission(const CycleContext& cycleContext, Synapse* synapse);

    auto cbeginOutboundSynapses() const noexcept {
        return outboundSynapses.cbegin();
    }

    auto cendOutboundSynapses() const noexcept {
        return outboundSynapses.cend();
    }

private:

    struct SynapticTransmissionInfo {

        SynapticTransmissionInfo(Synapse *synapse, TimeType transmissionTime);

        Synapse* synapse;
        TimeType transmissionTime;
    };

    std::vector<SynapticTransmissionInfo> synapticTransmissionSTDPBuffer;
    std::vector<Synapse*> outboundSynapses;
    std::shared_ptr<const NeuronParams> neuronParams;

    const SizeType neuronId;
    TimeType lastTime;
    ValueType lastVoltage;
    TimeType lastSpikeTime;

    bool isRefractoryPeriod(TimeType time) const noexcept {
        return time < lastTime;
    }

    void update(TimeType time) noexcept {
        lastVoltage = getMembraneVoltage(time);
        lastTime = time;
    }

    void fire(const CycleContext& cycleContext) noexcept;
    void processInboundOnSpike(const CycleContext& cycleContext);
    void processOutboundOnSpike(const CycleContext& cycleContext);
};

}




