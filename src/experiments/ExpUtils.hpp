#pragma once

#include <vector>
#include <core/ChannelSpikeInfo.hpp>
#include <deque>
#include <unordered_set>

namespace soft_npu {

using StimulusType = std::vector<ChannelSpikeInfo>;
using StimulusBufferType = std::deque<ChannelSpikeInfo>;

namespace ExpUtils {

StimulusType makeStrideStimulus(
        SizeType fromChannelId,
        SizeType toChannelId,
        TimeType timeDist,
        bool reverse
);

StimulusType makePoissonStimulus(
        SizeType fromChannelId,
        SizeType toChannelId,
        ValueType rate,
        TimeType duration,
        RandomEngineType &randomEngine);

void shiftStimulus(
        StimulusBufferType& buffer,
        const StimulusType& stimulus,
        TimeType shiftBy
);

void repeatStimulus(
        StimulusBufferType& buffer,
        const StimulusType& stimulus,
        TimeType offset,
        TimeType repetitionPeriod,
        TimeType until
);

}
}
