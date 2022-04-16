#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct SynapseParams {
    TimeType stdpTimeConstantInverse;
    TimeType stdpCutOffTime;
    ValueType stdpScaleFactorPotentiation;
    ValueType stdpScaleFactorDepression;
    ValueType maxWeight;
    TimeType eligibilityTraceTimeConstantInverse;
    ValueType eligibilityTraceCutOffTime;
};

}