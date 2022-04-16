#pragma once

#include "ChannelProjector.hpp"

namespace soft_npu {

class TopographicChannelProjector : public ChannelProjector {
public:
    TopographicChannelProjector(
            const ParamsType& params,
            const Population& population);

    void projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const override;
    ChannelSpikeProjectionResult getEPSPsWithTargetNeurons(SizeType channelId) const override;
    std::unordered_set<SizeType> getMotorNeuronIds() const override;

private:
    std::unordered_map<SizeType, ChannelSpikeProjectionResult> channelIdToResult;
    std::unordered_map<SizeType, SizeType> motorNeuronIdToOutputChannelId;
};

}
