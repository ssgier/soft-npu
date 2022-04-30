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

            // epsp override scaling is experimental. May be refined later.
            lastVoltage = std::max(lastVoltage + epsp * neuronParams->epspOverrideScaleFactor, neuronParams->voltageFloor);

            if (lastVoltage >= neuronParams->thresholdVoltage) {
                pushThresholdEvalEvent(cycleContext);
            }
        }
    }

    void fireIfAboveThreshold(const CycleContext& ctx, TimeType time) {

        auto compareVoltage = lastVoltage;
        if (!continuousInhibitionSources.empty()) {
            ValueType continuousInhibition = 0;

            for (auto source : continuousInhibitionSources) {
                source->update(time);
                continuousInhibition += source->getMembraneVoltage(time);
            }

            compareVoltage -= continuousInhibition;
        }

        if (compareVoltage >= neuronParams->thresholdVoltage) {
            fire(ctx);
        }
    }

    void addOutboundSynapse(Synapse* synapse);
    void addContinuousInhibitionSource(Neuron * source);

    ValueType getMembraneVoltage(TimeType time) const noexcept {

        if (time == lastTime) {
            return lastVoltage;
        } else {

            auto lastSourceSpikeTime = std::accumulate(
                    continuousInhibitionSources.cbegin(),
                    continuousInhibitionSources.cend(),
                    std::numeric_limits<TimeType>::lowest(),
                    [](TimeType time, const Neuron* source) {
                        return std::max(source->lastSpikeTime, time);
                    });

            if (lastSourceSpikeTime < lastTime) {
                TimeType timeSinceLastEvaluation = time - lastTime;
                return lastVoltage * exp(- timeSinceLastEvaluation * neuronParams->timeConstantInverse);
            } else {
                return neuronParams->resetVoltage;
            }

        }
    }

    SizeType getNeuronId() const noexcept;

    std::shared_ptr<const NeuronParams> getNeuronParams() const noexcept;

    TimeType getLastSpikeTime() const noexcept {
        return lastSpikeTime;
    }

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
    std::vector<Neuron*> continuousInhibitionSources;
    std::shared_ptr<const NeuronParams> neuronParams;

    const SizeType neuronId;
    TimeType lastTime;
    ValueType lastVoltage;
    TimeType lastSpikeTime;

    void pushThresholdEvalEvent(const CycleContext&);

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

