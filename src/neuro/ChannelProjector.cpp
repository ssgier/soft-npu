#include "ChannelProjector.hpp"

namespace soft_npu {

void ChannelProjector::projectChannelSpike(const CycleContext& ctx, SizeType channelId) const {
    auto epspsWithTargetNeurons = getEPSPsWithTargetNeurons(channelId);

    for (auto& epspWithTargetNeuron : epspsWithTargetNeurons) {
        ValueType epsp = epspWithTargetNeuron.first;
        auto targetNeuron = epspWithTargetNeuron.second;

        ctx.staticContext.eventProcessor.pushImmediateTransmissionEvent(epsp, *targetNeuron);
    }
}

}