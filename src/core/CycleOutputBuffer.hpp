#pragma once

#include <Aliases.hpp>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

class CycleOutputBuffer : private boost::noncopyable {
public:
    void addSpike(SizeType channelId);
    void reset() noexcept;

    auto cbeginSpikingChannelIds() const noexcept {
        return spikingChannelIds.cbegin();
    }

    auto cendSpikingChannelIds() const noexcept {
        return spikingChannelIds.cend();
    }

private:
    std::vector<SizeType> spikingChannelIds;
};

}
