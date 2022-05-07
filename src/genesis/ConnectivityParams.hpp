#pragma once

#include <Aliases.hpp>

namespace soft_npu {
struct ConnectivityParams {
    ValueType connectDensity;
    ValueType initialWeight;
    TimeType minConductionDelay;
    TimeType maxConductionDelay;
};
}
