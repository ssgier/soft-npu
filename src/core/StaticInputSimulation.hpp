#pragma once

#include <Aliases.hpp>
#include <neuro/Neuron.hpp>
#include <genesis/PopulationGeneratorFactory.hpp>
#include "RewardDoseInfo.hpp"
#include "AbstractSimulation.hpp"
#include <unordered_set>
#include <deque>
#include <core/ChannelSpikeInfo.hpp>

namespace soft_npu {

class StaticInputSimulation : public AbstractSimulation {
public:
    explicit StaticInputSimulation(std::shared_ptr<const ParamsType> params);

    void setSpikeTrains(std::deque<ChannelSpikeInfo> spikeTrains);
    void setRewardDoses(std::deque<RewardDoseInfo> rewardDoses);

    void recordOutputChannel(SizeType channelId);
    std::vector<ChannelSpikeInfo> getRecordedOutputChannelSpikes() const;

    void runController(
            CycleController& controller,
            Population& population,
            TimeType simulationTime,
            SynapticTransmissionStats&) override;

private:
    std::deque<ChannelSpikeInfo> spikeTrains;
    std::deque<RewardDoseInfo> rewardDoses;
    std::unordered_set<SizeType> outChannelIdsToRecord;
    std::vector<ChannelSpikeInfo> recordedOutputChannelSpikes;
};

}
