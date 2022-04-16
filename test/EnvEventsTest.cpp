#include <experiments/EnvEventQueue.hpp>
#include <gtest/gtest.h>
#include <experiments/StimulusLearningInfo.hpp>
#include <experiments/ExpUtils.hpp>
#include <core/CycleInputBuffer.hpp>
#include <core/CycleOutputBuffer.hpp>
#include <experiments/EnvContext.hpp>
#include <experiments/EnvEventFactories.hpp>
#include <experiments/EnvEvent.hpp>
#include <experiments/DetectionCorrectnessStats.hpp>
#include <TestUtil.hpp>

using namespace soft_npu;

void executeNextEventWithAsserts(
        const EnvContext& ctx,
        EnvEventQueue& envEventQueue,
        TimeType time,
        SizeType expectedQueueSizeAfter) {
    ASSERT_FLOAT_EQ(envEventQueue.top()->scheduleTime, time);
    envEventQueue.top()->task(ctx, time);
    envEventQueue.pop();
    ASSERT_EQ(envEventQueue.size(), expectedQueueSizeAfter);
}

void prepareRandomEngineForBernoulliDistVal(
        RandomEngineType& randomEngine,
        RandomEngineType& trackerRandomEngine,
        bool targetValue) {
    while(std::bernoulli_distribution()(trackerRandomEngine) != targetValue) {
        std::bernoulli_distribution()(randomEngine);
    }
}

void assertContainsShiftedStimulus(const StimulusBufferType& stimulusBuffer, const StimulusType& stimulus, TimeType shift) {

    ASSERT_EQ(stimulusBuffer.size(), stimulus.size());

    for (SizeType i = 0; i < stimulus.size(); ++i) {
        ASSERT_EQ(stimulusBuffer[i].channelId, stimulus[i].channelId);
        ASSERT_FLOAT_EQ(stimulusBuffer[i].time, stimulus[i].time + shift);
    }
}

auto makeParams() {
    auto params = getTemplateParams();
    (*params)["simulation"]["populationGenerator"] = "p1000";
    return params;
}

class EnvEventsTest : public ::testing::Test {
public:
    EnvEventsTest() :
        params(makeParams()),
        slInfo(std::make_unique<StimulusLearningInfo>()),
        spikeCountsIndexedByChannelId(20),
        randomEngine(0),
        envContext(std::make_unique<EnvContext>(
            envEventQueue,
            cycleInputBuffer,
            cycleOutputBuffer,
            randomEngine,
            stimulusBuffers,
            spikeCountsIndexedByChannelId,
            detectionCorrectnessStats,
            dt)),
        trackerRandomEngine(0){
        slInfo->stimulusA = ExpUtils::makeStrideStimulus(0, 3, 1, false);
        slInfo->stimulusB = ExpUtils::makeStrideStimulus(3, 6, 1, false);
        slInfo->endTime = 100;
        slInfo->intervalFrom = 2;
        slInfo->intervalTo = 2;
        slInfo->rewardDosage = 1.3;
        slInfo->detectorAChannelIdFrom = 10;
        slInfo->detectorAChannelIdTo = 15;
        slInfo->detectorBChannelIdFrom = 15;
        slInfo->detectorBChannelIdTo = 20;
        slInfo->readOutTime = 10e-3;
        slInfo->rewardDelay = 0.1;
        slInfo->statsStartTime = 0;
    }

protected:
    std::shared_ptr<ParamsType> params;
    std::unique_ptr<StimulusLearningInfo> slInfo;
    std::vector<std::unique_ptr<StimulusBufferType>> stimulusBuffers;
    CycleInputBuffer cycleInputBuffer;
    CycleOutputBuffer cycleOutputBuffer;
    std::vector<SizeType> spikeCountsIndexedByChannelId;
    EnvEventQueue envEventQueue;
    TimeType dt = 1e-4;
    RandomEngineType randomEngine;
    DetectionCorrectnessStats detectionCorrectnessStats;
    std::unique_ptr<EnvContext> envContext;
    RandomEngineType trackerRandomEngine;
};

TEST_F(EnvEventsTest, SingleStimulusRealizationIncorrectDetection) {

    slInfo->endTime = 3;

    envEventQueue.push(EnvEventFactories::makeStimulusLearningEvent(*slInfo, 2));
    ASSERT_EQ(envEventQueue.size(), 1);

    prepareRandomEngineForBernoulliDistVal(randomEngine, trackerRandomEngine, true); // will lead to stimulus B, assumes implementation details

    spikeCountsIndexedByChannelId[14] = 5;
    spikeCountsIndexedByChannelId[15] = 4;

    executeNextEventWithAsserts(*envContext, envEventQueue, 2, 2);

    ASSERT_EQ(stimulusBuffers.size(), 1);
    assertContainsShiftedStimulus(*stimulusBuffers[0], slInfo->stimulusB, 2);

    spikeCountsIndexedByChannelId[14] += 2;
    spikeCountsIndexedByChannelId[15] += 1;

    executeNextEventWithAsserts(*envContext, envEventQueue, 2 + slInfo->readOutTime, 2);
    executeNextEventWithAsserts(*envContext, envEventQueue, 2 + slInfo->readOutTime + slInfo->rewardDelay, 1);

    ASSERT_FLOAT_EQ(cycleInputBuffer.getReward(), - slInfo->rewardDosage);

    executeNextEventWithAsserts(*envContext, envEventQueue, 4, 0);

    ASSERT_EQ(detectionCorrectnessStats.numCorrectDetections, 0);
    ASSERT_EQ(detectionCorrectnessStats.numWrongDetections, 1);
}

TEST_F(EnvEventsTest, SingleStimulusRealizationCorrectDetection) {

    slInfo->endTime = 3;

    envEventQueue.push(EnvEventFactories::makeStimulusLearningEvent(*slInfo, 2));
    ASSERT_EQ(envEventQueue.size(), 1);

    prepareRandomEngineForBernoulliDistVal(randomEngine, trackerRandomEngine, true);

    spikeCountsIndexedByChannelId[14] = 5;
    spikeCountsIndexedByChannelId[15] = 4;

    executeNextEventWithAsserts(*envContext, envEventQueue, 2, 2);

    ASSERT_EQ(stimulusBuffers.size(), 1);
    assertContainsShiftedStimulus(*stimulusBuffers[0], slInfo->stimulusB, 2);

    spikeCountsIndexedByChannelId[14];
    spikeCountsIndexedByChannelId[15] += 2;

    executeNextEventWithAsserts(*envContext, envEventQueue, 2 + slInfo->readOutTime, 2);
    executeNextEventWithAsserts(*envContext, envEventQueue, 2 + slInfo->readOutTime + slInfo->rewardDelay, 1);

    ASSERT_FLOAT_EQ(cycleInputBuffer.getReward(), slInfo->rewardDosage);

    executeNextEventWithAsserts(*envContext, envEventQueue, 4, 0);

    ASSERT_EQ(detectionCorrectnessStats.numCorrectDetections, 1);
    ASSERT_EQ(detectionCorrectnessStats.numWrongDetections, 0);
}

TEST_F(EnvEventsTest, LateStatsStartTime) {

    slInfo->endTime = 3;
    slInfo->statsStartTime = 2.5;

    envEventQueue.push(EnvEventFactories::makeStimulusLearningEvent(*slInfo, 2));
    ASSERT_EQ(envEventQueue.size(), 1);

    prepareRandomEngineForBernoulliDistVal(randomEngine, trackerRandomEngine, true);

    while(!envEventQueue.empty()) {
        envEventQueue.top()->task(*envContext, envEventQueue.top()->scheduleTime);
        envEventQueue.pop();
    }

    ASSERT_EQ(detectionCorrectnessStats.numWrongDetections, 0);
}

TEST_F(EnvEventsTest, RewardDosage) {

    slInfo->endTime = 3;

    envEventQueue.push(EnvEventFactories::makeStimulusLearningEvent(*slInfo, 2));

    prepareRandomEngineForBernoulliDistVal(randomEngine, trackerRandomEngine, true);

    executeNextEventWithAsserts(*envContext, envEventQueue, 2, 2);

    spikeCountsIndexedByChannelId[14] += 2;
    spikeCountsIndexedByChannelId[15] += 3;

    executeNextEventWithAsserts(*envContext, envEventQueue, 2 + slInfo->readOutTime, 2);
    executeNextEventWithAsserts(*envContext, envEventQueue, 2 + slInfo->readOutTime + slInfo->rewardDelay, 1);

    ASSERT_FLOAT_EQ(cycleInputBuffer.getReward(), slInfo->rewardDosage);
}
