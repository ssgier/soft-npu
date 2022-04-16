#pragma once

#include "ChannelProjector.hpp"

namespace soft_npu {

class ExplicitChannelProjector : public ChannelProjector {
public:
    ExplicitChannelProjector() = default;

    void addSensoryNeuron(Neuron& neuron, std::vector<SizeType> channelIds, ValueType epsp);
    void addMotorNeuron(const Neuron& motorNeuron, SizeType channelId);

    void projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const override;
    ChannelSpikeProjectionResult getEPSPsWithTargetNeurons(SizeType channelId) const override;
    std::unordered_set<SizeType> getMotorNeuronIds() const override;

private:
    std::unordered_map<SizeType, ChannelSpikeProjectionResult> channelIdToResult;
    std::unordered_map<SizeType, SizeType> motorNeuronIdToOutputChannelId;
};

}
