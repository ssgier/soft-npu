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
#include <util/InterruptSignalChecker.hpp>

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
    } else if (params.resultExtractionNumCandidates < 1) {
        throw std::runtime_error("Result extraction num candidates must be strictly positive");
    } else if (params.resultExtractionNumCandidates > params.populationSize) {
        throw std::runtime_error("Result extraction num candidates must not be greater than population size");
    } else if (params.resultExtractionNumEvalSeeds < 1) {
        throw std::runtime_error("Result extraction num evaluation seeds must be strictly positive");
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

struct CandidateEvalJob {
    std::shared_ptr<const Candidate> candidate;
    SizeType seed;
    ValueType resultFitnessValue;
};

EvaluatedCandidates evaluateFitness(
        const Candidates& candidates,
        const FitnessFunction& fitnessFunction,
        RandomEngineType& randomEngine) {

    std::vector<CandidateEvalJob> jobs;

    std::transform(
            candidates.cbegin(),
            candidates.cend(),
            std::back_inserter(jobs),
            [&randomEngine](auto candidate) {
                CandidateEvalJob job;
                job.candidate = candidate;
                job.seed = randomEngine();
                job.resultFitnessValue = std::numeric_limits<ValueType>::quiet_NaN();
                return job;
            });

    std::for_each(
            std::execution::par,
            jobs.begin(),
            jobs.end(),
            [&fitnessFunction](auto& job) {
                job.resultFitnessValue = fitnessFunction.evaluate(
                        *job.candidate->getGeneValue(), job.seed);
            });

    EvaluatedCandidates result;

    std::transform(
            jobs.cbegin(),
            jobs.cend(),
            std::back_inserter(result),
            [] (const auto& job) {
                CandidateWithFitness candidateWithFitness;
                candidateWithFitness.candidate = job.candidate;
                candidateWithFitness.fitnessValue = job.resultFitnessValue;
                return candidateWithFitness;
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

struct CandidateExtractionEvalJob {
    SizeType candidateIndex;
    SizeType runIndex;
    SizeType seed;
};

CandidateWithFitness extractBestCandidate(
        const EvolutionParams& evolutionParams,
        const EvaluatedCandidates& sortedCandidates,
        const FitnessFunction& fitnessFunction,
        RandomEngineType& randomEngine) {

    std::vector<std::vector<ValueType>> fitnessValues(evolutionParams.resultExtractionNumCandidates);

    std::vector<ValueType> fillValue(evolutionParams.resultExtractionNumEvalSeeds);
    std::fill(
            fitnessValues.begin(),
            fitnessValues.end(),
            fillValue);

    std::vector<CandidateExtractionEvalJob> jobs;

    for (SizeType candidateIdx = 0; candidateIdx < evolutionParams.resultExtractionNumCandidates; ++ candidateIdx) {

        fitnessValues[candidateIdx][0] = sortedCandidates[candidateIdx].fitnessValue;

        for (SizeType runIdx = 1; runIdx < evolutionParams.resultExtractionNumEvalSeeds; ++ runIdx) {
            CandidateExtractionEvalJob job;
            job.candidateIndex = candidateIdx;
            job.runIndex = runIdx;
            job.seed = randomEngine();
            jobs.push_back(job);
        }
    }

    std::for_each(
            std::execution::par,
            jobs.cbegin(),
            jobs.cend(),
            [&sortedCandidates, &fitnessValues, &fitnessFunction](const auto& job) {
                auto fitnessValue = fitnessFunction.evaluate(
                        *sortedCandidates[job.candidateIndex].candidate->getGeneValue(),
                        job.seed);

                fitnessValues[job.candidateIndex][job.runIndex] = fitnessValue;
            });

    std::vector<ValueType> meanFitnessValues;

    std::transform(
            fitnessValues.cbegin(),
            fitnessValues.cend(),
            std::back_inserter(meanFitnessValues),
            [](const auto& fitnessValuesForCandidate) {
                return std::reduce(
                        fitnessValuesForCandidate.cbegin(),
                        fitnessValuesForCandidate.cend()) / fitnessValuesForCandidate.size();
            });

    SizeType bestCandidateIndex = std::distance(
            meanFitnessValues.cbegin(), std::min_element(meanFitnessValues.cbegin(), meanFitnessValues.cend()));

    CandidateWithFitness bestCandidate;
    bestCandidate.candidate = sortedCandidates[bestCandidateIndex].candidate;
    bestCandidate.fitnessValue = meanFitnessValues[bestCandidateIndex];
    return bestCandidate;
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

        PLOG_INFO << "Iteration completed. Best fitness value: " << evaluatedCandidates.front().fitnessValue
            << ", worst fitness value: " << evaluatedCandidates.back().fitnessValue;

        if (InterruptSignalChecker::wasSent()) {
            terminationReason = TerminationReason::interruptSignalReceived;
            break;
        }

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

        ++ iteration;
    }

    auto nowTs = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(nowTs - startTs);

    PLOG_INFO << "Extracting best candidate from population based on mean fitness value...";

    auto bestCandidate = extractBestCandidate(
            evolutionParams,
            evaluatedCandidates,
            fitnessFunction,
            randomEngine);

    EvolutionResult result;
    result.terminationReason = terminationReason;
    result.topFitnessValue = bestCandidate.fitnessValue;
    result.timePassedSeconds = duration.count() * 1e-3;
    result.numberOfIterations = iteration;
    result.topGeneValue = bestCandidate.candidate->getGeneValue();

    PLOG_INFO << "Evolution terminated, reason: " << toString(terminationReason)
        << ", achieved fitness value: " << result.topFitnessValue;

    return result;
}

}
