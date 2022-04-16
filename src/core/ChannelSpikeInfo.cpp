#include "ChannelSpikeInfo.hpp"

namespace soft_npu {

ChannelSpikeInfo::ChannelSpikeInfo(TimeType time, SizeType channelId): time(time), channelId(channelId) {}

bool operator<(const ChannelSpikeInfo& lhs, const ChannelSpikeInfo& rhs) {
    return lhs.time < rhs.time;
}

}
