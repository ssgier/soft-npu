#include "EnvEventFactories.hpp"
#include <core/CycleInputBuffer.hpp>
#include "EnvEventQueue.hpp"
#include "DetectionCorrectnessStats.hpp"
#include "EnvEvent.hpp"

namespace soft_npu::EnvEventFactories {


ValueType getRewardDosage(
        SizeType targetPopulationFiringCount,
        SizeType opponentPopulationFiringCount,
        const StimulusLearningInfo& slInfo) {
    if (targetPopulationFiringCount == opponentPopulationFiringCount) {
        return 0;
    } else if (targetPopulationFiringCount > opponentPopulationFiringCount) {
        return slInfo.rewardDosage;
    } else {
        return - slInfo.rewardDosage;
    }
}

SizeType getChanRangeIntegrationSum(const std::vector<SizeType>& spikeCounts, SizeType chanIdFrom, SizeType chanIdTo) {

    assert(chanIdTo <= spikeCounts.size());

    SizeType rv = 0;
    for (SizeType i = chanIdFrom; i != chanIdTo; ++i) {
        rv += spikeCounts[i];
    }
    return rv;
}

std::unique_ptr<EnvEvent> makeRewardEvent(
        StimulusLearningInfo&,
        TimeType rewardTime, ValueType dosage) {
    return std::make_unique<EnvEvent>(rewardTime, [dosage](const EnvContext& ctx, TimeType){
        ctx.cycleInputBuffer.addReward(dosage);
    });
}

std::unique_ptr<EnvEvent> makeDetectionValidatorEvent(
        StimulusLearningInfo& slInfo,
        TimeType validationWindowEnd,
        SizeType chanFromTarget,
        SizeType chanToTarget,
        SizeType chanFromOpponent,
        SizeType chanToOpponent,
        SizeType integrationSumStartTarget,
        SizeType integrationSumStartOpponent
        ) {

    return std::make_unique<EnvEvent>(validationWindowEnd, [
                                                            &slInfo,
                                                            chanFromTarget,
                                                            chanToTarget,
                                                            chanFromOpponent,
                                                            chanToOpponent,
                                                            integrationSumStartTarget,
                                                            integrationSumStartOpponent] (const EnvContext& ctx, TimeType time){
        SizeType totalSpikeCountTarget = getChanRangeIntegrationSum(
                ctx.spikeCountsIndexedByChannelId,
                chanFromTarget,
                chanToTarget) - integrationSumStartTarget;

        SizeType totalSpikeCountOpponent = getChanRangeIntegrationSum(
                ctx.spikeCountsIndexedByChannelId,
                chanFromOpponent,
                chanToOpponent) - integrationSumStartOpponent;

        auto rewardDosage = getRewardDosage(totalSpikeCountTarget, totalSpikeCountOpponent, slInfo);

        if (std::abs(rewardDosage) > 0) {
            ctx.envEventQueue.push(makeRewardEvent(slInfo, time + slInfo.rewardDelay, rewardDosage));
        }

        if (slInfo.statsStartTime <= time) {

            if (totalSpikeCountTarget > totalSpikeCountOpponent) {
                ++ ctx.detectionCorrectnessStats.numCorrectDetections;
            } else if (totalSpikeCountTarget < totalSpikeCountOpponent) {
                ++ ctx.detectionCorrectnessStats.numWrongDetections;
            } else {
                ++ ctx.detectionCorrectnessStats.numAbstinences;
            }
        }
    });
}

std::unique_ptr<EnvEvent> makeStimulusLearningEvent(
        StimulusLearningInfo& slInfo,
        TimeType startTime) {

    return std::make_unique<EnvEvent>(
            startTime,
            [&slInfo, startTime](const EnvContext& ctx, TimeType time) {

        if (startTime < slInfo.endTime) {

            bool chooseB = std::bernoulli_distribution()(ctx.randomEngine);
            const StimulusType& chosenStimulus = chooseB ? slInfo.stimulusB : slInfo.stimulusA;

            auto stimulusBuffer = std::make_unique<StimulusBufferType>();
            ExpUtils::shiftStimulus(*stimulusBuffer, chosenStimulus, startTime);
            ctx.stimulusBuffers.push_back(std::move(stimulusBuffer));

            TimeType maxIntervalDuration = slInfo.intervalTo - slInfo.intervalFrom;

            TimeType nextStartTime = time +
                                     slInfo.intervalFrom +
                                     std::uniform_real_distribution<TimeType>()(ctx.randomEngine) * maxIntervalDuration;

            auto recurringEvent = makeStimulusLearningEvent(
                    slInfo,
                    nextStartTime);

            ctx.envEventQueue.push(std::move(recurringEvent));

            SizeType chanFromTarget = slInfo.detectorAChannelIdFrom;
            SizeType chanToTarget = slInfo.detectorAChannelIdTo;
            SizeType chanFromOpponent = slInfo.detectorBChannelIdFrom;
            SizeType chanToOpponent = slInfo.detectorBChannelIdTo;

            if (chooseB) {
                std::swap(chanFromTarget, chanFromOpponent);
                std::swap(chanToTarget, chanToOpponent);
            }

            SizeType opponentChansIntegrationSum = getChanRangeIntegrationSum(ctx.spikeCountsIndexedByChannelId, chanFromOpponent, chanToOpponent);
            SizeType targetChansIntegrationSum = getChanRangeIntegrationSum(ctx.spikeCountsIndexedByChannelId, chanFromTarget, chanToTarget);

            auto detectionValidatorEvent = makeDetectionValidatorEvent(
                    slInfo,
                    time + slInfo.readOutTime,
                    chanFromTarget,
                    chanToTarget,
                    chanFromOpponent,
                    chanToOpponent,
                    targetChansIntegrationSum,
                    opponentChansIntegrationSum
                    );

            ctx.envEventQueue.push(std::move(detectionValidatorEvent));
        }
    });
}
}
