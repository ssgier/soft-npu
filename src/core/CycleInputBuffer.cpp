#include "CycleInputBuffer.hpp"

namespace soft_npu {

void CycleInputBuffer::addReward(ValueType reward) noexcept {
    this->reward += reward;
}

void CycleInputBuffer::addSpike(SizeType channelId) {
    spikingChannelIds.push_back(channelId);
}

void CycleInputBuffer::reset() noexcept {
    reward = 0;
    spikingChannelIds.clear();
}

CycleInputBuffer::const_iterator CycleInputBuffer::cendSpikingChannelIds() const noexcept {
    return spikingChannelIds.cend();
}

CycleInputBuffer::const_iterator CycleInputBuffer::cbeginSpikingChannelIds() const noexcept {
    return spikingChannelIds.cbegin();
}

ValueType CycleInputBuffer::getReward() const noexcept {
    return reward;
}

}