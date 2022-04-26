#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct ShortTermPlasticityParams {
    bool isDepression;
    ValueType restingValue;
    ValueType changeParameter;
    TimeType tauInverse;
};
}
