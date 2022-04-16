#include <plog/Log.h>
#include <core/CycleInputBuffer.hpp>
#include "POCDynamicSimulation.hpp"
#include "ExpUtils.hpp"
#include "EnvEventQueue.hpp"
#include "EnvContext.hpp"
#include "StimulusLearningInfo.hpp"
#include "EnvEventFactories.hpp"
#include <core/CycleController.hpp>
#include "DetectionCorrectnessStats.hpp"
#include "EnvEvent.hpp"
#include <core/SynapticTransmissionStats.hpp>

namespace soft_npu {

POCDynamicSimulation::POCDynamicSimulation(std::shared_ptr<const ParamsType> params) :
    AbstractSimulation(params),
    optimResultHolder(),
    rewardDosage((*params)["pocDynamicSimulation"]["rewardDosage"]),
    abortAfterNumSynapticTransmissions((*params)["pocDynamicSimulation"]["abortAfterNumSynapticTransmissions"]),
    flipDetectorChannels((*params)["pocDynamicSimulation"]["flipDetectorChannels"]) {
}

bool timeCrossed(TimeType threshold, TimeType currentTime, TimeType dt) {
    return currentTime >= threshold && currentTime - dt < threshold;
}

void drainStimulusBuffers(
        std::vector<std::unique_ptr<StimulusBufferType>>& stimulusBuffers,
        CycleInputBuffer& cycleInputBuffer,
        TimeType currentTime
        ) {

    for (const auto& sb : stimulusBuffers) {
        for (; !sb->empty() && sb->front().time <= currentTime; sb->pop_front()) {
            cycleInputBuffer.addSpike(sb->front().channelId);
        }
    }

    auto isEmpty = [](const std::unique_ptr<StimulusBufferType>& sb) {
        return sb->empty();
    };

    stimulusBuffers.erase(
            std::remove_if(
                    stimulusBuffers.begin(),
                    stimulusBuffers.end(),
                    isEmpty),
            stimulusBuffers.end());
}

void updateCount(std::vector<SizeType>& counts, SizeType channelId) {
    if (channelId >= counts.size()) {
        counts.resize(channelId + 1);
    }

    ++ counts[channelId];
}

std::unique_ptr<StimulusLearningInfo> makeStimulusLearningInfo(
        TimeType simulationTime,
        RandomEngineType&,
        ValueType rewardDosage,
        bool flipDetectorChannels) {
    auto slInfo = std::make_unique<StimulusLearningInfo>();

    // TODO: move into config

    slInfo->stimulusA = {{0, 0}, {10e-3, 1}};
    slInfo->stimulusB = {{0, 1}, {10e-3, 0}};
    slInfo->endTime = simulationTime;
    slInfo->intervalFrom = 1000e-3;
    slInfo->intervalTo = 1025e-3;
    slInfo->rewardDosage = rewardDosage;
    slInfo->detectorAChannelIdFrom = 0;
    slInfo->detectorAChannelIdTo = 1;
    slInfo->detectorBChannelIdFrom = 1;
    slInfo->detectorBChannelIdTo = 2;
    slInfo->readOutTime = 31e-3;
    slInfo->rewardDelay = 10e-3;
    slInfo->statsStartTime = 0;

    if (flipDetectorChannels) {
        std::swap(slInfo->detectorAChannelIdFrom, slInfo->detectorBChannelIdFrom);
        std::swap(slInfo->detectorAChannelIdTo, slInfo->detectorBChannelIdTo);
    }

    return slInfo;
}

void POCDynamicSimulation::runController(
        CycleController &controller,
        Population&,
        TimeType simulationTime,
        SynapticTransmissionStats& synapticTransmissionStats
        ) {

    TimeType currentTime;

    std::vector<std::unique_ptr<StimulusBufferType>> stimulusBuffers;
    auto& cycleInputBuffer = controller.getCycleInputBuffer();
    auto& cycleOutputBuffer = controller.getCycleOutputBuffer();
    std::vector<SizeType> spikeCountsIndexedByChannelId(1000);
    DetectionCorrectnessStats detectionCorrectnessStats;

    EnvEventQueue envEventQueue;
    TimeType dt = controller.getTimeIncrement();

    auto slInfo = makeStimulusLearningInfo(simulationTime, randomEngine, rewardDosage, flipDetectorChannels);

    EnvContext ctx(
            envEventQueue,
            cycleInputBuffer,
            cycleOutputBuffer,
            randomEngine,
            stimulusBuffers,
            spikeCountsIndexedByChannelId,
            detectionCorrectnessStats,
            dt);

    TimeType stimulationStartTime = 0;

    envEventQueue.push(EnvEventFactories::makeStimulusLearningEvent(*slInfo, stimulationStartTime));

    PLOG_DEBUG << "Starting main loop";

    for (SizeType iterCount = 0; (currentTime = controller.getTime()) <  simulationTime; ++ iterCount) {

        cycleInputBuffer.reset();

        for (; !envEventQueue.empty() && envEventQueue.top()->scheduleTime <= currentTime; envEventQueue.pop()) {
            envEventQueue.top()->task(ctx, currentTime);
        }

        drainStimulusBuffers(stimulusBuffers, cycleInputBuffer, currentTime);

        controller.runCycle();

        if (synapticTransmissionStats.getTransmissionCount() > abortAfterNumSynapticTransmissions) {
            PLOG_INFO << "Max num transmission events of " << abortAfterNumSynapticTransmissions
                << " breached. Aborting.";
            optimResultHolder.objFuncVal = std::numeric_limits<double>::max();
            return;
        }

        for (
            auto it = cycleOutputBuffer.cbeginSpikingChannelIds();
            it != cycleOutputBuffer.cendSpikingChannelIds();
            ++it) {

            updateCount(spikeCountsIndexedByChannelId, *it);
        }
    }

    auto totalNumDetectionTrials =
            detectionCorrectnessStats.numCorrectDetections + detectionCorrectnessStats.numWrongDetections + detectionCorrectnessStats.numAbstinences;

    auto partCorrect = static_cast<double>(detectionCorrectnessStats.numCorrectDetections) / totalNumDetectionTrials;
    auto partWrong = static_cast<double>(detectionCorrectnessStats.numWrongDetections) / totalNumDetectionTrials;
    auto partAbstained = static_cast<double>(detectionCorrectnessStats.numAbstinences) / totalNumDetectionTrials;

    PLOG_DEBUG << "Total num detection trials: " << totalNumDetectionTrials << std::endl
        << "correct: " << 100.0 * partCorrect << " %" << std::endl
        << "wrong: " << 100.0 * partWrong << " %" << std::endl
        << "abstained: " << 100.0 * partAbstained << " %" << std::endl;

    double objCandidate = 1 - partCorrect + partWrong;

    optimResultHolder.objFuncVal = objCandidate;
}

}
