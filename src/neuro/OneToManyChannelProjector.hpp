#pragma once

#include "ChannelProjector.hpp"

namespace soft_npu {

class OneToManyChannelProjector : public ChannelProjector {
public:
    OneToManyChannelProjector(
            const ParamsType& params,
            RandomEngineType& randomEngine,
            const Population& population);

    void projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const override;
    std::unordered_set<SizeType> getMotorNeuronIds() const override;

protected:
    ChannelSpikeProjectionResult getEPSPsWithTargetNeurons(SizeType channelId) const override;

private:
    std::unordered_map<SizeType, ChannelSpikeProjectionResult> channelIdToResult;
    SizeType fromMotorNeuronId;
    SizeType toMotorNeuronId;
    SizeType fromOutChannelId;
    SizeType toOutChannelId;
};

}




