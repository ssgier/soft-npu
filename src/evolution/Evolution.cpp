#include "Evolution.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include "Candidate.hpp"
#include "GeneOperationUtils.hpp"
#include <execution>
#include <plog/Log.h>
#include <Aliases.hpp>
#include "Gene.hpp"
#include "MutationParams.hpp"

namespace soft_npu {

using CandidateVector = std::vector<std::shared_ptr<Candidate>>;

void validateEvolutionParams(const EvolutionParams& params) {
    if (params.proxyPopulationSize <= params.mainPopulationSize) {
        throw std::runtime_error("Proxy population size must be larger than main population size");
    } else if (params.mainPopulationSize <= params.elitePopulationSize) {
        throw std::runtime_error("Main population size must be larger than elite population size");
    } else if (params.maxMutationProbability < params.minMutationProbability || params.minMutationProbability < 0 ||
                params.maxMutationProbability > 1) {
        throw std::runtime_error("Following inequality must hold: 0 <= min mutation probability <= max mutation probability <= 1");
    } else if (params.minMutationStrength > params.maxMutationStrength) {
        throw std::runtime_error("Min mutation strength must not be greater than max mutation strength");
    } else if (params.minMutationStrength < 0) {
        throw std::runtime_error("Mutation strength must not be negative");
    } else if (params.crossoverProbability < 0 || params.crossoverProbability > 1) {
        throw std::runtime_error("Crossover probability must not be less than zero or greater than 1");
    } else if (params.elitePopulationSize <= 0) {
        throw std::runtime_error("Population sizes must be strictly positive");
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

void sortPopulation(std::vector<std::shared_ptr<Candidate>>& population) {
    std::sort(population.begin(), population.end(), [](auto candidate0, auto candidate1) {
        return candidate0->getFitness() <  candidate1->getFitness();
    });
}

std::shared_ptr<Candidate> createOffspring(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& proxyFitnessFunction,
    const FitnessFunction& mainFitnessFunction,
    const CandidateVector& elitePopulation,
    RandomEngineType& randomEngine
    ) {

    GeneVector genes;
    std::transform(elitePopulation.cbegin(), elitePopulation.cend(), std::back_inserter(genes),
                   [&randomEngine, &evolutionParams](auto candidate) {
        return candidate->getGene()->mutate(generateMutationParams(randomEngine, evolutionParams), randomEngine);
    });

    auto offspringGene = GeneOperationUtils::crossover(evolutionParams, genes, randomEngine);

    return std::make_shared<Candidate>(offspringGene, proxyFitnessFunction, mainFitnessFunction);
}

void evaluateFitness(std::vector<std::shared_ptr<Candidate>>& candidates) {
    std::for_each(
        std::execution::par,
        candidates.cbegin(),
        candidates.cend(),
        [](auto candidate) {
            candidate->evaluateFitness();
        });
}

void iterate(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& proxyFitnessFunction,
    const FitnessFunction& mainFitnessFunction,
    std::vector<std::shared_ptr<Candidate>>& elitePopulation,
    RandomEngineType& randomEngine
) {
    std::vector<std::shared_ptr<Candidate>> proxyPopulation;
    proxyPopulation.reserve(evolutionParams.proxyPopulationSize);

    for (int i = 0; i < evolutionParams.proxyPopulationSize; ++ i) {
        proxyPopulation.push_back(createOffspring(evolutionParams, proxyFitnessFunction, mainFitnessFunction,
                                                  elitePopulation, randomEngine));
    }

    PLOG_DEBUG << "Evaluating proxy population fitness";

    std::for_each(
        std::execution::par,
        proxyPopulation.cbegin(),
        proxyPopulation.cend(),
        [](auto candidate) {
            candidate->evaluateFitnessProxy();
        });

    std::nth_element(
        proxyPopulation.begin(),
        proxyPopulation.begin() + evolutionParams.mainPopulationSize,
        proxyPopulation.end(),
        [](auto candidate0, auto candidate1) {
            return candidate0->getFitnessProxy() < candidate1->getFitnessProxy();
        });

    std::vector<std::shared_ptr<Candidate>> mainPopulation;

    std::copy_n(
        proxyPopulation.cbegin(),
        evolutionParams.mainPopulationSize,
        std::back_inserter(mainPopulation));

    auto cmp = [](auto c0, auto c1) {
        return c0->getFitnessProxy() < c1->getFitnessProxy();
    };

    PLOG_DEBUG << "Best proxy fitness in main population: "
        << (*std::min_element(mainPopulation.cbegin(), mainPopulation.cend(), cmp))->getFitnessProxy()
        << ", worst proxy fitness in main population: "
        << (*std::max_element(mainPopulation.cbegin(), mainPopulation.cend(), cmp))->getFitnessProxy();

    PLOG_DEBUG << "Evaluating main population fitness";

    evaluateFitness(mainPopulation);

    std::copy(
        elitePopulation.cbegin(),
        elitePopulation.cend(),
        std::back_inserter(mainPopulation));

    std::partial_sort(
        mainPopulation.begin(),
        mainPopulation.begin() + evolutionParams.elitePopulationSize,
        mainPopulation.end(),
        [](auto candidate0, auto candidate1) {
            return candidate0->getFitness() < candidate1->getFitness();
        });

    elitePopulation.clear();
    std::copy_n(mainPopulation.cbegin(), evolutionParams.elitePopulationSize, std::back_inserter(elitePopulation));
}

EvolutionResult Evolution::runImpl(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& proxyFitnessFunction,
    const FitnessFunction& mainFitnessFunction,
    const nlohmann::json& geneInfoJson) {

    PLOG_INFO << "Running with evo params: " << std::endl << evolutionParams;

    auto prototypeGene = GeneOperationUtils::assembleFromJson(geneInfoJson);

    auto startTs = std::chrono::high_resolution_clock::now();

    validateEvolutionParams(evolutionParams);

    std::vector<std::shared_ptr<Candidate>> elitePopulation;

    RandomEngineType randomEngine(0);

    auto originCandidate = std::make_shared<Candidate>(
            prototypeGene->clone(),
            proxyFitnessFunction,
            mainFitnessFunction);

    elitePopulation.push_back(originCandidate);

    for (auto i = 1; i < evolutionParams.elitePopulationSize; ++i) {

        elitePopulation.push_back(std::make_shared<Candidate>(
                originCandidate->getGene()->mutate(generateMutationParams(randomEngine, evolutionParams), randomEngine),
                proxyFitnessFunction,
                mainFitnessFunction));
    }

    PLOG_DEBUG << "Evaluating fitness of initial elite population";

    evaluateFitness(elitePopulation);
    sortPopulation(elitePopulation);

    TerminationReason terminationReason;
    int iteration = 0;

    while (true) {

        if (elitePopulation[0]->getFitness() <= evolutionParams.targetFitnessValue) {
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
            proxyFitnessFunction,
            mainFitnessFunction,
            elitePopulation,
            randomEngine);

        PLOG_INFO << "Iteration completed. Best elite fitness: " << elitePopulation.front()->getFitness()
            << ", worst elite fitness: " << elitePopulation.back()->getFitness();

        ++ iteration;
    }

    auto nowTs = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(nowTs - startTs);

    EvolutionResult result;
    result.terminationReason = terminationReason;
    result.topFitnessValue = elitePopulation[0]->getFitness();
    result.timePassedSeconds = duration.count() * 1e-3;
    result.numberOfIterations = iteration;
    result.topGeneJson = elitePopulation[0]->getGeneValueJson();

    PLOG_INFO << "Evolution terminated, reason: " << toString(terminationReason)
        << ", achieved fitness value: " << result.topFitnessValue;

    return result;
}

}
