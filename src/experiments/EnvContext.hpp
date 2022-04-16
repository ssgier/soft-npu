#pragma once

#include <Aliases.hpp>
#include <vector>
#include "ExpUtils.hpp"
#include "EnvEventQueue.hpp"

namespace soft_npu {

class CycleInputBuffer;
class CycleOutputBuffer;
struct DetectionCorrectnessStats;

struct EnvContext {
    EnvContext(
            EnvEventQueue& envEventQueue,
            CycleInputBuffer &cycleInputBuffer,
            const CycleOutputBuffer &cycleOutputBuffer,
            RandomEngineType& randomEngine,
            std::vector<std::unique_ptr<StimulusBufferType>>& stimulusBuffers,
            const std::vector<SizeType> &spikeCountsIndexedByChannelId,
            DetectionCorrectnessStats& detectionCorrectnessStats,
            const TimeType dt);

    EnvEventQueue& envEventQueue;
    CycleInputBuffer& cycleInputBuffer;
    const CycleOutputBuffer& cycleOutputBuffer;
    RandomEngineType& randomEngine;
    std::vector<std::unique_ptr<StimulusBufferType>>& stimulusBuffers;
    const std::vector<SizeType>& spikeCountsIndexedByChannelId;
    DetectionCorrectnessStats& detectionCorrectnessStats;
    const TimeType dt;
};

}

