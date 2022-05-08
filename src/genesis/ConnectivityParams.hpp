#pragma once

#include <Aliases.hpp>

namespace soft_npu {
struct ConnectivityParams {
    ValueType connectDensity;
    ValueType minInitialWeight;
    ValueType maxInitialWeight;
    TimeType minConductionDelay;
    TimeType maxConductionDelay;
};
}
