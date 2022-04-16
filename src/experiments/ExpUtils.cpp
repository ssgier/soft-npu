#include "ExpUtils.hpp"

namespace soft_npu::ExpUtils {

StimulusType makeStrideStimulus(SizeType fromChannelId, SizeType toChannelId, TimeType timeDist, bool reverse) {

    StimulusType rv;

    TimeType currentTime = 0;
    if (reverse) {
        for (SizeType channelId = toChannelId - 1; channelId >= fromChannelId && channelId != std::numeric_limits<SizeType>::max(); --channelId) {
            rv.emplace_back(currentTime, channelId);
            currentTime += timeDist;
        }
    } else {
        for (SizeType channelId = fromChannelId; channelId < toChannelId; ++channelId) {
            rv.emplace_back(currentTime, channelId);
            currentTime += timeDist;
        }
    }

    return rv;
}

StimulusType makePoissonStimulus(
        SizeType fromChannelId,
        SizeType toChannelId,
        ValueType rate,
        TimeType duration,
        RandomEngineType& randomEngine) {

    auto expDist = std::exponential_distribution<TimeType>(rate);

    std::vector<ChannelSpikeInfo> rv;

    for (SizeType channelId = fromChannelId; channelId < toChannelId; ++ channelId) {
        TimeType nextSpikeTime = expDist(randomEngine);

        while (nextSpikeTime < duration) {
            rv.emplace_back(nextSpikeTime, channelId);
            nextSpikeTime += expDist(randomEngine);
        }
    }

    std::sort(rv.begin(), rv.end());

    return rv;
}

void shiftStimulus(
        StimulusBufferType& buffer,
        const StimulusType& stimulus,
        TimeType shiftBy) {

    for (const auto& spikeInfo : stimulus) {
        auto shiftedSpike = spikeInfo;
        shiftedSpike.time += shiftBy;
        buffer.push_back(shiftedSpike);
    }
}

void repeatStimulus(
        StimulusBufferType& buffer,
        const StimulusType& stimulus,
        TimeType offset,
        TimeType repetitionPeriod,
        TimeType until) {

    TimeType currentTime;

    for (SizeType i = 0; (currentTime = offset + i * repetitionPeriod) < until; ++i) {
        shiftStimulus(buffer, stimulus, currentTime);
    }
}

}


