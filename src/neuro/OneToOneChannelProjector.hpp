#pragma once

#include "ChannelProjector.hpp"

namespace soft_npu {

class OneToOneChannelProjector : public ChannelProjector {
public:
    OneToOneChannelProjector(const ParamsType& params, const Population& population);
    void projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const override;
    std::unordered_set<SizeType> getMotorNeuronIds() const override;

protected:
    ChannelSpikeProjectionResult getEPSPsWithTargetNeurons(SizeType channelId) const override;
private:
    std::unordered_map<SizeType, ChannelSpikeProjectionResult> channelIdToResult;
};

}
