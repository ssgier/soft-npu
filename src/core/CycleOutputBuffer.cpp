#include "CycleOutputBuffer.hpp"

namespace soft_npu {

void CycleOutputBuffer::addSpike(SizeType channelId) {
    spikingChannelIds.push_back(channelId);
}

void CycleOutputBuffer::reset() noexcept {
    spikingChannelIds.clear();
}

}
