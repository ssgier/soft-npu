#pragma once

#include <ostream>

namespace soft_npu {

enum class TerminationReason {
    targetFitnessValueReached,
    maxNumIterationsReached,
    timeLimitPassed
};

std::string toString(TerminationReason terminationReason);
std::ostream& operator<<(std::ostream& os, soft_npu::TerminationReason terminationReason);

}


