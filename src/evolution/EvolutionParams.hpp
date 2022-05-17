#pragma once

#include <limits>
#include <Aliases.hpp>
#include <plog/Record.h>

namespace soft_npu {

struct EvolutionParams {

    double abortAfterSeconds = 6000;
    int maxNumIterations = -1;
    double targetFitnessValue = std::numeric_limits<double>::lowest();

    unsigned int populationSize = 100;
    unsigned int eliteSize = 5;
    ValueType minMutationProbability = 0.0;
    ValueType maxMutationProbability = 0.9;
    ValueType minMutationStrength = 0.0;
    ValueType maxMutationStrength = 0.45;
    ValueType crossoverProbability = 0.5;
    ValueType tournamentSelectionProbability = 0.75;
    SizeType resultExtractionNumEvalSeeds = 5;
    SizeType resultExtractionNumCandidates = 5;
};

plog::Record& operator<<(plog::Record&, const EvolutionParams&);

}
