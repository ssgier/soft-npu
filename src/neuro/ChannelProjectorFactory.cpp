#include "ChannelProjectorFactory.hpp"
#include "OneToOneChannelProjector.hpp"
#include "OneToManyChannelProjector.hpp"
#include "TopographicChannelProjector.hpp"

namespace soft_npu::ChannelProjectorFactory {

std::unique_ptr<ChannelProjector> createFromParams(
        const ParamsType& params,
        RandomEngineType& randomEngine,
        const Population& population) {
    std::string channelProjectorName = params["simulation"]["channelProjector"];

    if (channelProjectorName == "OneToOne") {
        return std::make_unique<OneToOneChannelProjector>(params, population);
    } else if (channelProjectorName == "OneToMany") {
        return std::make_unique<OneToManyChannelProjector>(params, randomEngine, population);
    } else if (channelProjectorName == "Topographic") {
        return std::make_unique<TopographicChannelProjector>(params, population);
    } else {
        throw std::runtime_error("Invalid channel projector name: " + channelProjectorName);
    }
}

}
