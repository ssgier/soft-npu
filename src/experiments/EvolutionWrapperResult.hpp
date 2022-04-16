#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct EvolutionWrapperResult {
    std::shared_ptr<const ParamsType> evolvedParams;
    double fitnessValue;
};

}
