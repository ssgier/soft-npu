#include "Evolution.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include "Candidate.hpp"
#include "CandidateWithFitness.hpp"
#include "SelectionUtils.hpp"
#include "GeneOperationUtils.hpp"
#include <execution>
#include <plog/Log.h>
#include <Aliases.hpp>
#include "Gene.hpp"
#include "MutationParams.hpp"

namespace soft_npu {

using Candidates = std::vector<std::shared_ptr<const Candidate>>;
using EvaluatedCandidates = std::vector<CandidateWithFitness>;

void validateEvolutionParams(const EvolutionParams& params) {
    if (params.populationSize <= params.eliteSize) {
        throw std::runtime_error("Population size must be larger than elite size");
    } else if (params.maxMutationProbability < params.minMutationProbability || params.minMutationProbability < 0 ||
                params.maxMutationProbability > 1) {
        throw std::runtime_error("Following inequality must hold: 0 <= min mutation probability <= max mutation probability <= 1");
    } else if (params.minMutationStrength > params.maxMutationStrength) {
        throw std::runtime_error("Min mutation strength must not be greater than max mutation strength");
    } else if (params.minMutationStrength < 0) {
        throw std::runtime_error("Mutation strength must not be negative");
    } else if (params.crossoverProbability < 0 || params.crossoverProbability > 1) {
        throw std::runtime_error("Crossover probability must not be less than zero or greater than 1");
    } else if (params.tournamentSelectionProbability < 0 || params.tournamentSelectionProbability > 1) {
        throw std::runtime_error("Tournament selection probability must be in [0, 1]");
    }
}

MutationParams generateMutationParams(
    RandomEngineType& randomEngine,
    const EvolutionParams& evolutionParams
) {
    std::uniform_real_distribution<double> mutationProbProbDist(evolutionParams.minMutationProbability, 
        evolutionParams.maxMutationProbability);

    std::uniform_real_distribution<double> mutationStrengthProbDist(evolutionParams.minMutationStrength,
        evolutionParams.maxMutationStrength);

    MutationParams rv;
    rv.mutationProbability = mutationProbProbDist(randomEngine);
    rv.mutationStrength = mutationStrengthProbDist(randomEngine);
    return rv;
}

template <typename T>
bool hasTimeLimitPassed(const std::chrono::time_point<T> & startTs, const EvolutionParams& evolutionParams) {
    if (evolutionParams.abortAfterSeconds < 0) {
        return false;
    }
    auto nowTs = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(nowTs - startTs);

    return duration.count() > evolutionParams.abortAfterSeconds;
}

void sortCandidates(EvaluatedCandidates& population) {
    std::sort(population.begin(), population.end(), [](const auto& candidate0, const auto& candidate1) {
        return candidate0.fitnessValue <  candidate1.fitnessValue;
    });
}

std::shared_ptr<Candidate> createOffspring(
    const EvolutionParams& evolutionParams,
    const EvaluatedCandidates& evaluatedCandidatesSorted,
    RandomEngineType& randomEngine
    ) {

    GeneVector mutatedGenesSorted;
   
    std::transform(
            evaluatedCandidatesSorted.cbegin(),
            evaluatedCandidatesSorted.cend(),
            std::back_inserter(mutatedGenesSorted),
            [&evolutionParams, &randomEngine] (const auto& candidateWithFitness) {
                return candidateWithFitness.candidate->getGene()->mutate(
                        generateMutationParams(randomEngine, evolutionParams), randomEngine);
            });

    SizeType numParents = 2;
    GeneVector parentGenes;
    for (SizeType i = 0; i < numParents; ++i) {
        auto parentIndex = SelectionUtils::selectParentIndex(evolutionParams, randomEngine);
        parentGenes.push_back(mutatedGenesSorted[parentIndex]);
    }

    auto offspringGene = GeneOperationUtils::crossover(evolutionParams, parentGenes, randomEngine);

    return std::make_shared<Candidate>(offspringGene);
}

EvaluatedCandidates evaluateFitness(
        const Candidates& candidates,
        const FitnessFunction& fitnessFunction,
        RandomEngineType& randomEngine) {

    EvaluatedCandidates result;

    std::transform(
            candidates.cbegin(),
            candidates.cend(),
            std::back_inserter(result),
            [](auto candidate) {
                CandidateWithFitness candidateWithFitness;
                candidateWithFitness.candidate = candidate;
                candidateWithFitness.fitnessValue = std::numeric_limits<ValueType>::quiet_NaN();
                return candidateWithFitness;
            });

    std::for_each(
            std::execution::par,
            result.begin(),
            result.end(),
            [&fitnessFunction, &randomEngine](auto& candidateWithFitness) {
                candidateWithFitness.fitnessValue = fitnessFunction.evaluate(
                        *candidateWithFitness.candidate->getGeneValue(), randomEngine());
            });

    return result;
}

void iterate(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& fitnessFunction,
    EvaluatedCandidates& evaluatedCandidatesSorted,
    RandomEngineType& randomEngine
) {
    Candidates newGenerationCandidates;

    for (SizeType i = 0; i < evolutionParams.eliteSize; ++i) {
        newGenerationCandidates.push_back(evaluatedCandidatesSorted[i].candidate);
    }

    while (newGenerationCandidates.size() < evolutionParams.populationSize) {
        newGenerationCandidates.push_back(createOffspring(evolutionParams, evaluatedCandidatesSorted, randomEngine));
    }

    PLOG_DEBUG << "Evaluating main population fitness";

    EvaluatedCandidates evaluatedNewCandidates = evaluateFitness(
            newGenerationCandidates,
            fitnessFunction,
            randomEngine);

    evaluatedCandidatesSorted = evaluatedNewCandidates;
}

EvolutionResult Evolution::runImpl(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& fitnessFunction,
    const ParamsType& geneInfoJson) {

    PLOG_INFO << "Running with evo params: " << std::endl << evolutionParams;

    auto prototypeGene = GeneOperationUtils::assembleFromInfo(geneInfoJson);

    auto startTs = std::chrono::high_resolution_clock::now();

    validateEvolutionParams(evolutionParams);

    Candidates population;

    RandomEngineType randomEngine(0);

    auto originCandidate = std::make_shared<Candidate>(prototypeGene->clone());

    population.push_back(originCandidate);

    for (SizeType i = 1; i < evolutionParams.populationSize; ++i) {
        population.push_back(std::make_shared<Candidate>(
                originCandidate->getGene()->mutate(generateMutationParams(randomEngine, evolutionParams), randomEngine)));
    }

    auto evaluatedCandidates = evaluateFitness(population, fitnessFunction, randomEngine);

    TerminationReason terminationReason;
    int iteration = 0;

    while (true) {

        sortCandidates(evaluatedCandidates);

        if (evaluatedCandidates.front().fitnessValue <= evolutionParams.targetFitnessValue) {
            terminationReason = TerminationReason::targetFitnessValueReached;
            break;
        }

        if (evolutionParams.maxNumIterations >= 0 && iteration > evolutionParams.maxNumIterations) {
            terminationReason = TerminationReason::maxNumIterationsReached;
            break;
        }

        if (hasTimeLimitPassed(startTs, evolutionParams)) {
            terminationReason = TerminationReason::timeLimitPassed;
            break;
        }

        PLOG_DEBUG << "Starting iteration " << iteration;

        iterate(
            evolutionParams,
            fitnessFunction,
            evaluatedCandidates,
            randomEngine);

        PLOG_INFO << "Iteration completed. Best fitness value: " << evaluatedCandidates.front().fitnessValue
            << ", worst fitness value: " << evaluatedCandidates.back().fitnessValue;

        ++ iteration;
    }

    auto nowTs = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(nowTs - startTs);

    EvolutionResult result;
    result.terminationReason = terminationReason;
    result.topFitnessValue = evaluatedCandidates.front().fitnessValue;
    result.timePassedSeconds = duration.count() * 1e-3;
    result.numberOfIterations = iteration;
    result.topGeneValue = evaluatedCandidates.front().candidate->getGeneValue();

    PLOG_INFO << "Evolution terminated, reason: " << toString(terminationReason)
        << ", achieved fitness value: " << result.topFitnessValue;

    return result;
}

}
