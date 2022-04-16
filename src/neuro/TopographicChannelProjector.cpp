#include "TopographicChannelProjector.hpp"
#include <neuro/Population.hpp>

namespace soft_npu {


TopographicChannelProjector::TopographicChannelProjector(const ParamsType& params,
                                                         const Population& population) {

    auto projectorParams = params["channelProjectors"]["Topographic"];

    ValueType startXSensory = projectorParams["startXSensory"];
    ValueType startYSensory = projectorParams["startYSensory"];
    ValueType startXMotor = projectorParams["startXMotor"];
    ValueType startYMotor = projectorParams["startYMotor"];

    SizeType numInputChannels = projectorParams["numInputChannels"];
    ValueType projectionRadius = projectorParams["projectionRadius"];
    ValueType inputInterChannelDistance = projectorParams["inputInterChannelDistance"];
    SizeType numOutputChannels = projectorParams["numOutputChannels"];
    ValueType convergenceRadius = projectorParams["convergenceRadius"];
    ValueType outputInterChannelDistance = projectorParams["outputInterChannelDistance"];
    ValueType epsp = projectorParams["epsp"];

    if (convergenceRadius * 2 > outputInterChannelDistance) {
        throw std::runtime_error("Output channel convergence fields must not overlap");
    }

    for (auto it = population.cbeginNeurons(); it != population.cendNeurons(); ++it) {
        auto& neuron = *it;

        if (!neuron->getNeuronParams()->isInhibitory) {
            auto location = population.getCellLocation(neuron->getNeuronId());

            for (SizeType inputChannelId = 0; inputChannelId < numInputChannels; ++inputChannelId) {
                ValueType xTarget = startXSensory;
                ValueType yTarget = startYSensory + inputChannelId * inputInterChannelDistance;

                Population::Location targetLocation({xTarget, yTarget});

                if (PopulationUtils::isDistanceShorterThan(location, targetLocation, projectionRadius)) {
                    auto& projectionResult = channelIdToResult.emplace(inputChannelId, ChannelSpikeProjectionResult()).first->second;
                    projectionResult.push_back(std::make_pair(epsp, neuron.get()));
                }
            }

            for (SizeType outputChannelId = 0; outputChannelId < numOutputChannels; ++outputChannelId) {
                ValueType xTarget = startXMotor;
                ValueType yTarget = startYMotor + outputChannelId * outputInterChannelDistance;

                Population::Location targetLocation({xTarget, yTarget});

                if (PopulationUtils::isDistanceShorterThan(location, targetLocation, convergenceRadius)) {
                    motorNeuronIdToOutputChannelId.emplace(neuron->getNeuronId(), outputChannelId);
                }
            }
        }
    }
}

void TopographicChannelProjector::projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer,
                                                     const Neuron& neuron) const {
    auto it = motorNeuronIdToOutputChannelId.find(neuron.getNeuronId());

    if (it != motorNeuronIdToOutputChannelId.cend()) {
        SizeType outputChannelId = it->second;
        cycleOutputBuffer.addSpike(outputChannelId);
    }
}

ChannelProjector::ChannelSpikeProjectionResult
TopographicChannelProjector::getEPSPsWithTargetNeurons(SizeType channelId) const {

    auto it = channelIdToResult.find(channelId);
    return it == channelIdToResult.cend() ? ChannelProjector::ChannelSpikeProjectionResult() : it->second;
}

std::unordered_set<SizeType> TopographicChannelProjector::getMotorNeuronIds() const {
    std::unordered_set<SizeType> rv;

    for (const auto& entry : motorNeuronIdToOutputChannelId) {
        SizeType motorNeuronId = entry.first;
        rv.insert(motorNeuronId);
    }

    return rv;
}

}
