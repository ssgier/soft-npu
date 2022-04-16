#pragma once

#include <Aliases.hpp>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

class CycleInputBuffer : private boost::noncopyable {
public:
    using const_iterator = std::vector<SizeType>::const_iterator;

    void addReward(ValueType reward) noexcept;
    void addSpike(SizeType channelId);
    void reset() noexcept;

    const_iterator cbeginSpikingChannelIds() const noexcept;
    const_iterator cendSpikingChannelIds() const noexcept;
    ValueType getReward() const noexcept;

private:
    ValueType reward = 0;
    std::vector<SizeType> spikingChannelIds;
};

}
