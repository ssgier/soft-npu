#pragma once

#include "TerminationReason.hpp"
#include <Aliases.hpp>

namespace soft_npu {

struct EvolutionResult {
    TerminationReason terminationReason;
    int numberOfIterations;
    double timePassedSeconds;
    double topFitnessValue;
    std::shared_ptr<const nlohmann::json> topGeneJson;
};

}
