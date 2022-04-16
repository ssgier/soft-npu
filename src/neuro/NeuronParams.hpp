#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct NeuronParams {
    TimeType timeConstantInverse = 1;
    TimeType refractoryPeriod = 0;
    ValueType thresholdVoltage = 1;
    ValueType resetVoltage = 0;
    ValueType voltageFloor = 0;
    bool isInhibitory = false;
};

}

