#include "OneToOneChannelProjector.hpp"
#include <neuro/Population.hpp>

namespace soft_npu {

OneToOneChannelProjector::OneToOneChannelProjector(
        const ParamsType& params,
        const Population& population) {
    ValueType epsp = params["channelProjectors"]["OneToOne"]["epsp"];

    for (auto it = population.cbeginNeurons(); it != population.cendNeurons(); ++ it) {
        auto neuron = (*it).get();
        ChannelSpikeProjectionResult projectionResult({{epsp, neuron}});
        channelIdToResult.emplace(neuron->getNeuronId(), projectionResult);
    }
}

ChannelProjector::ChannelSpikeProjectionResult OneToOneChannelProjector::getEPSPsWithTargetNeurons(SizeType channelId) const {
    return channelIdToResult.find(channelId)->second;
}

void OneToOneChannelProjector::projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const {
    if (!spikingNeuron.getNeuronParams()->isInhibitory) {

        SizeType channelId = spikingNeuron.getNeuronId();
        cycleOutputBuffer.addSpike(channelId);
    }
}

std::unordered_set<SizeType> OneToOneChannelProjector::getMotorNeuronIds() const {
    std::unordered_set<SizeType> rv;

    for (const auto& entry : channelIdToResult) {
        rv.insert(entry.first);
    }

    return rv;
}
}
