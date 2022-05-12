#include <gtest/gtest.h>
#include <Aliases.hpp>
#include <evolution/EvolutionParams.hpp>
#include <evolution/SelectionUtils.hpp>

using namespace soft_npu;

TEST (SelectionUtilsTest, TournamentProbabilityZero) {
    RandomEngineType randomEngine;
    EvolutionParams evolutionParams;
    evolutionParams.populationSize = 2;
    evolutionParams.tournamentSelectionProbability = 0;
    auto parentIndex = SelectionUtils::selectParentIndex(evolutionParams, randomEngine);
    ASSERT_EQ(parentIndex, 1);
}

TEST (SelectionUtilsTest, TournamentProbabilityOne) {
    RandomEngineType randomEngine;
    EvolutionParams evolutionParams;
    evolutionParams.populationSize = 2;
    evolutionParams.tournamentSelectionProbability = 1;
    auto parentIndex = SelectionUtils::selectParentIndex(evolutionParams, randomEngine);
    ASSERT_EQ(parentIndex, 0);
}
