#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct FitnessFunction {
    virtual double evaluate(const ParamsType&) const = 0;
    virtual ~FitnessFunction() = default;
};

}
