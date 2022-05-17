#include <gtest/gtest.h>
#include <evolution/Evolution.hpp>

using namespace soft_npu;

TEST(EvolutionTest, SimpleOptimizationProblem) {
    EvolutionParams evolutionParams;

    evolutionParams.abortAfterSeconds = -1;
    evolutionParams.maxNumIterations = 10000;
    evolutionParams.targetFitnessValue = 1e-6;
    evolutionParams.populationSize = 100;
    evolutionParams.eliteSize = 5;
    evolutionParams.minMutationProbability = 0.0;
    evolutionParams.maxMutationProbability = 0.9;
    evolutionParams.minMutationStrength = 0;
    evolutionParams.maxMutationStrength = 0.45;
    evolutionParams.crossoverProbability = 0.5;
    evolutionParams.tournamentSelectionProbability = 0.75;
    evolutionParams.resultExtractionNumEvalSeeds = 5;
    evolutionParams.resultExtractionNumCandidates = 5;

    auto geneInfoJson = R"(

[
    {
        "id": "x",
        "prototypeValue": 0.0,
        "minValue": 0.0,
        "maxValue": 20.0
    },
    {
        "id": "y",
        "prototypeValue": 10.0,
        "minValue": 0.0,
        "maxValue": 20.0
    }
]

)"_json;

    auto fitnessFunction = [](const nlohmann::json& json, SizeType) {
        double x = json["x"];
        double y = json["y"];

        return - x * (5-x) + 2.5 * 2.5 + y;
    };

    auto result = Evolution::run(evolutionParams, fitnessFunction, geneInfoJson);
    ASSERT_EQ(result.terminationReason, TerminationReason::targetFitnessValueReached);
    ASSERT_NEAR((*result.topGeneValue)["x"], 2.5, 1e-1);
    ASSERT_DOUBLE_EQ((*result.topGeneValue)["y"], 0);
}
