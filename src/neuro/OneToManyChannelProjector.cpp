#include <unordered_set>
#include "OneToManyChannelProjector.hpp"
#include <neuro/Population.hpp>

namespace soft_npu {

OneToManyChannelProjector::OneToManyChannelProjector(const ParamsType& params,
                                                     RandomEngineType& randomEngine,
                                                     const Population& population)
                                                     {

    auto oneToManyParams = params["channelProjectors"]["OneToMany"];

    SizeType fromInChannelId = oneToManyParams["fromInChannelId"];
    SizeType toInChannelId = oneToManyParams["toInChannelId"];
    SizeType fromSensoryNeuronId = oneToManyParams["fromSensoryNeuronId"];
    SizeType toSensoryNeuronId = oneToManyParams["toSensoryNeuronId"];

    fromOutChannelId = oneToManyParams["fromOutChannelId"];
    toOutChannelId = oneToManyParams["toOutChannelId"];
    fromMotorNeuronId = oneToManyParams["fromMotorNeuronId"];
    toMotorNeuronId = oneToManyParams["toMotorNeuronId"];

    SizeType divergence = oneToManyParams["divergence"];
    ValueType epsp = oneToManyParams["epsp"];

    if (toSensoryNeuronId - fromSensoryNeuronId < divergence) {
        throw std::runtime_error("Not enough sensory neurons for specified divergence");
    }

    std::uniform_int_distribution<SizeType> distribution(fromSensoryNeuronId, toSensoryNeuronId - 1);

    for (SizeType channelId = fromInChannelId; channelId < toInChannelId; ++ channelId) {

        auto& projectionResult = channelIdToResult.emplace(
                channelId,
                ChannelSpikeProjectionResult()).first->second;

        std::unordered_set<SizeType> neuronIds;

        while (neuronIds.size() < divergence) {
            neuronIds.insert(distribution(randomEngine));
        }

        std::transform(
                neuronIds.cbegin(), neuronIds.cend(),
                std::back_inserter(projectionResult), [&population, epsp](SizeType neuronId) {
            return std::make_pair(epsp, &population.getNeuronById(neuronId));
        });
    }
}

ChannelProjector::ChannelSpikeProjectionResult OneToManyChannelProjector::getEPSPsWithTargetNeurons(
        SizeType channelId) const {

    assert(channelIdToResult.find(channelId) != channelIdToResult.end());

    return channelIdToResult.find(channelId)->second;
}

void OneToManyChannelProjector::projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const {
    auto spikingNeuronId = spikingNeuron.getNeuronId();

    if (spikingNeuronId >= fromMotorNeuronId && spikingNeuronId < toMotorNeuronId) {
        auto outChannelId = fromOutChannelId + (spikingNeuronId - fromMotorNeuronId) % (toOutChannelId - fromOutChannelId);
        cycleOutputBuffer.addSpike(outChannelId);
    }
}

std::unordered_set<SizeType> OneToManyChannelProjector::getMotorNeuronIds() const {
    std::unordered_set<SizeType> rv;

    for (SizeType motorNeuronId = fromMotorNeuronId; motorNeuronId < toMotorNeuronId; ++motorNeuronId) {
        rv.insert(motorNeuronId);
    }

    return rv;
}

}
