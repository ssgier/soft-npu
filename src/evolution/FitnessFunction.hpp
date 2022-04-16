#pragma once

#include <nlohmann/json.hpp>

namespace soft_npu {

struct FitnessFunction {
    virtual double evaluate(const nlohmann::json&) const = 0;
    virtual ~FitnessFunction() = default;
};

}
