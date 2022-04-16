#include "EnvContext.hpp"

namespace soft_npu {

EnvContext::EnvContext(
        EnvEventQueue& envEventQueue,
        CycleInputBuffer &cycleInputBuffer,
        const CycleOutputBuffer& cycleOutputBuffer,
        RandomEngineType& randomEngine,
        std::vector<std::unique_ptr<StimulusBufferType>>& stimulusBuffers,
        const std::vector<SizeType>& spikeCountsIndexedByChannelId,
        DetectionCorrectnessStats& detectionCorrectnessStats,
        const TimeType dt) :
            envEventQueue(envEventQueue),
            cycleInputBuffer(cycleInputBuffer),
            cycleOutputBuffer(cycleOutputBuffer),
            randomEngine(randomEngine),
            stimulusBuffers(stimulusBuffers),
            spikeCountsIndexedByChannelId(
            spikeCountsIndexedByChannelId),
            detectionCorrectnessStats(detectionCorrectnessStats),
            dt(dt) {}

}
