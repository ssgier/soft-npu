#pragma once

#include "ChannelProjector.hpp"

namespace soft_npu::ChannelProjectorFactory {

std::unique_ptr<ChannelProjector> createFromParams(
        const ParamsType& params,
        RandomEngineType& randomEngine,
        const Population& population);

}
