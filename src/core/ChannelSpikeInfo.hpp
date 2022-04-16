#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct ChannelSpikeInfo {
    ChannelSpikeInfo(TimeType time, SizeType channelId);
    ChannelSpikeInfo(const ChannelSpikeInfo& rhs) = default;
    ChannelSpikeInfo& operator=(const ChannelSpikeInfo& rhs) = default;

    TimeType time;
    SizeType channelId;
};

bool operator<(const ChannelSpikeInfo& lhs, const ChannelSpikeInfo& rhs);

}
