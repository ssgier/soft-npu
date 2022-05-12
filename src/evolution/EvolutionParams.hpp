#pragma once

#include <limits>
#include <plog/Record.h>

namespace soft_npu {

struct EvolutionParams {

    double abortAfterSeconds = 6000;
    int maxNumIterations = -1;
    double targetFitnessValue = std::numeric_limits<double>::lowest();

    unsigned int populationSize = 100;
    unsigned int eliteSize = 5;
    double minMutationProbability = 0.0;
    double maxMutationProbability = 0.9;
    double minMutationStrength = 0.0;
    double maxMutationStrength = 0.45;
    double crossoverProbability = 0.5;
    double tournamentSelectionProbability = 0.75;
};

plog::Record& operator<<(plog::Record&, const EvolutionParams&);

}
