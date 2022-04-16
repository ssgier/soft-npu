#include "StaticInputSimulation.hpp"
#include <memory>
#include "CommonEvent.hpp"
#include "CycleController.hpp"

namespace soft_npu {

StaticInputSimulation::StaticInputSimulation(
        std::shared_ptr<const ParamsType> params) :
            AbstractSimulation(params) {
}

void StaticInputSimulation::runController(
        CycleController& controller,
        Population&,
        TimeType simulationTime,
        SynapticTransmissionStats&) {
    TimeType currentTime;

    auto& cycleInputBuffer = controller.getCycleInputBuffer();
    const auto& cycleOutputBuffer = controller.getCycleOutputBuffer();

    while ((currentTime = controller.getTime()) <  simulationTime) {

        cycleInputBuffer.reset();
        for (; !spikeTrains.empty() && spikeTrains.front().time <= currentTime; spikeTrains.pop_front()) {
            cycleInputBuffer.addSpike(spikeTrains.front().channelId);
        }

        for (; !rewardDoses.empty() && rewardDoses.front().time <= currentTime; rewardDoses.pop_front()) {
            cycleInputBuffer.addReward(rewardDoses.front().dosage);
        }

        controller.runCycle();

        std::for_each(
                cycleOutputBuffer.cbeginSpikingChannelIds(),
                cycleOutputBuffer.cendSpikingChannelIds(),
                [currentTime, this](SizeType channelId) {

                    if (outChannelIdsToRecord.find(channelId) != outChannelIdsToRecord.end()) {
                        recordedOutputChannelSpikes.emplace_back(currentTime, channelId);
                    }
                });
    }
}

void StaticInputSimulation::recordOutputChannel(SizeType channelId) {
    outChannelIdsToRecord.insert(channelId);
}

void StaticInputSimulation::setSpikeTrains(std::deque<ChannelSpikeInfo> spikeTrains) {
    assert(std::is_sorted(spikeTrains.cbegin(), spikeTrains.cend()));
    this->spikeTrains = std::move(spikeTrains);
}

void StaticInputSimulation::setRewardDoses(std::deque<RewardDoseInfo> rewardDoses) {
    assert(std::is_sorted(rewardDoses.cbegin(), rewardDoses.cend()));
    this->rewardDoses = std::move(rewardDoses);
}

std::vector<ChannelSpikeInfo> StaticInputSimulation::getRecordedOutputChannelSpikes() const {
    return recordedOutputChannelSpikes;
}

}
