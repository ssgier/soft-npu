#pragma once

#include <Aliases.hpp>
#include <optional>
#include "ShortTermPlasticityParams.hpp"

namespace soft_npu {

struct SynapseParams {
    TimeType tauInversePotentiation;
    TimeType tauInverseDepression;
    TimeType stdpCutOffTime;
    ValueType stdpScaleFactorPotentiation;
    ValueType stdpScaleFactorDepression;
    ValueType maxWeight;
    TimeType eligibilityTraceTimeConstantInverse;
    ValueType eligibilityTraceCutOffTime;
    std::optional<ShortTermPlasticityParams> shortTermPlasticityParams;
};

}
