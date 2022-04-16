#pragma once

#include <core/EventProcessor.hpp>
#include <core/CycleOutputBuffer.hpp>
#include <unordered_set>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

struct ChannelProjector : private boost::noncopyable {

    virtual ~ChannelProjector() = default;
    void projectChannelSpike(const CycleContext& ctx, SizeType channelId) const;
    virtual void projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const = 0;
    virtual std::vector<std::pair<ValueType, Neuron*>> getEPSPsWithTargetNeurons(SizeType channelId) const = 0;
    virtual std::unordered_set<SizeType> getMotorNeuronIds() const = 0;

protected:
    using ChannelSpikeProjectionResult = std::vector<std::pair<ValueType, Neuron*>>;
};

}
