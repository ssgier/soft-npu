#include "EvolutionParams.hpp"

namespace soft_npu {

plog::Record& operator<<(plog::Record& record, const EvolutionParams& evolutionParams) {
    record  << "Abort after seconds: " << evolutionParams.abortAfterSeconds << std::endl
        << "Max num iterations: " << evolutionParams.maxNumIterations << std::endl
        << "Target fitness value: " << evolutionParams.targetFitnessValue << std::endl
        << "Population size: " << evolutionParams.populationSize << std::endl
        << "Elite size: " << evolutionParams.eliteSize << std::endl
        << "Min mutation probability: " << evolutionParams.minMutationProbability << std::endl
        << "Max mutation probability: " << evolutionParams.maxMutationProbability << std::endl
        << "Min mutation strength: " << evolutionParams.minMutationStrength << std::endl
        << "Max mutation strength: " << evolutionParams.maxMutationStrength << std::endl
        << "Crossover probability: " << evolutionParams.crossoverProbability << std::endl
        << "Tournament selection probability: " << evolutionParams.tournamentSelectionProbability;

    return record;
}

}
