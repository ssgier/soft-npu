#pragma once

#include <limits>
#include <plog/Record.h>

namespace soft_npu {

struct EvolutionParams {

    double abortAfterSeconds = 6000;
    int maxNumIterations = -1;
    double targetFitnessValue = std::numeric_limits<double>::lowest();

    int proxyPopulationSize = 101;
    int mainPopulationSize = 100;
    int elitePopulationSize = 5;
    double minMutationProbability = 0.0;
    double maxMutationProbability = 0.9;
    double minMutationStrength = 0.0;
    double maxMutationStrength = 0.45;
    double crossoverProbability = 0.5;
};

plog::Record& operator<<(plog::Record&, const EvolutionParams&);

}
