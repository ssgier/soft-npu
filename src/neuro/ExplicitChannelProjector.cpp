#include "ExplicitChannelProjector.hpp"

namespace soft_npu {

void ExplicitChannelProjector::addSensoryNeuron(
        Neuron& neuron,
        std::vector<SizeType> channelIds,
        ValueType epsp) {
    for (auto channelId : channelIds) {
        channelIdToResult.emplace(channelId, ChannelSpikeProjectionResult()).first->second.push_back({epsp, &neuron});
    }
}

void ExplicitChannelProjector::addMotorNeuron(const Neuron &motorNeuron, SizeType channelId) {
    motorNeuronIdToOutputChannelId.emplace(motorNeuron.getNeuronId(), channelId);
}

void
ExplicitChannelProjector::projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const {
    auto it = motorNeuronIdToOutputChannelId.find(spikingNeuron.getNeuronId());

    if (it != motorNeuronIdToOutputChannelId.cend()) {
        SizeType outputChannelId = it->second;
        cycleOutputBuffer.addSpike(outputChannelId);
    }
}

ChannelProjector::ChannelSpikeProjectionResult
ExplicitChannelProjector::getEPSPsWithTargetNeurons(SizeType channelId) const {
    auto it = channelIdToResult.find(channelId);
    return it == channelIdToResult.cend() ? ChannelProjector::ChannelSpikeProjectionResult() : it->second;
}

std::unordered_set<SizeType> ExplicitChannelProjector::getMotorNeuronIds() const {
    std::unordered_set<SizeType> rv;

    for (const auto& entry : motorNeuronIdToOutputChannelId) {
        SizeType motorNeuronId = entry.first;
        rv.insert(motorNeuronId);
    }

    return rv;
}

}
