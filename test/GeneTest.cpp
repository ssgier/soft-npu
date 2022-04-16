#include <gtest/gtest.h>
#include <evolution/Gene.hpp>
#include <evolution/RealNumberGeneElement.hpp>
#include <evolution/BooleanGeneElement.hpp>
#include <evolution/MutationParams.hpp>

using namespace soft_npu;

auto createRealNumberLeafGene(double minValue, double maxValue, double value) {
    return std::make_shared<Gene>(std::make_shared<RealNumberGeneElement>("", minValue, maxValue, value));
}

auto createBooleanLeafGene(bool value) {
    return std::make_shared<Gene>(std::make_shared<BooleanGeneElement>("", value));
}

TEST(GeneTest, MutationRate0) {

    auto subGene = std::make_shared<Gene>(GeneVector({
        createRealNumberLeafGene(0.0, 2.0, 1.0),
        createRealNumberLeafGene(2.0, 2.0, 2.0)}));

    auto gene = std::make_shared<Gene>(GeneVector({
        createRealNumberLeafGene(0.0, 20.0, 10.0),
        subGene,
        createBooleanLeafGene(false)}));

    MutationParams mutationParams;
    mutationParams.mutationProbability = 0;

    RandomEngineType randomEngine;

    auto zeroMutationResultGene = gene->mutate(mutationParams, randomEngine);

    auto depth1It = zeroMutationResultGene->cbeginSubGenes();
    auto subGeneAtPos0 = *depth1It;

    ASSERT_TRUE(subGeneAtPos0->isLeaf());
    ASSERT_DOUBLE_EQ(std::stod(subGeneAtPos0->getElementValueAsString()), 10.0);

    ++ depth1It;

    auto subGeneAtPos1 = *depth1It;

    ASSERT_FALSE(subGeneAtPos1->isLeaf());
    auto leafGenesIt = subGeneAtPos1->cbeginSubGenes();
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 1.0);
    ++leafGenesIt;
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 2.0);

    ++ depth1It;

    auto subGeneAtPos2 = *depth1It;
    ASSERT_TRUE(subGeneAtPos2->isLeaf());
    ASSERT_FALSE(std::stoi(subGeneAtPos2->getElementValueAsString()));
}

void sideEffectBernoulli(const MutationParams& mutationParams, RandomEngineType& randomEngine) {
    std::bernoulli_distribution(mutationParams.mutationProbability)(randomEngine);
}

TEST(GeneTest, MutationRate1) {

    auto subGene = std::make_shared<Gene>(GeneVector({
                                                         createRealNumberLeafGene(0.0, 2.0, 1.0),
                                                         createRealNumberLeafGene(2.0, 2.0, 2.0)}));

    auto gene = std::make_shared<Gene>(GeneVector({
                                                      createRealNumberLeafGene(0.0, 20.0, 10.0),
                                                      subGene,
                                                      createBooleanLeafGene(false)}));

    double fpTol = 1e-6;

    MutationParams mutationParams;
    mutationParams.mutationProbability = 1;
    mutationParams.mutationStrength = 1.5;

    RandomEngineType randomEngine(0);
    RandomEngineType trackingRandomEngine(0);

    auto resultGene = gene->mutate(mutationParams, randomEngine);

    auto depth1It = resultGene->cbeginSubGenes();
    auto subGeneAtPos0 = *depth1It;

    ASSERT_TRUE(subGeneAtPos0->isLeaf());

    for (int i = 0; i < 2; ++i) {
        sideEffectBernoulli(mutationParams, trackingRandomEngine);
    }

    double expectedValue = std::max(0.0, std::min(20.0,
        std::normal_distribution<double>(10, mutationParams.mutationStrength * 20)(trackingRandomEngine)));
    ASSERT_NEAR(std::stod(subGeneAtPos0->getElementValueAsString()), expectedValue, fpTol);

    ++ depth1It;

    auto subGeneAtPos1 = *depth1It;

    ASSERT_FALSE(subGeneAtPos1->isLeaf());

    auto leafGenesIt = subGeneAtPos1->cbeginSubGenes();

    for (int i = 0; i < 2; ++i){
        sideEffectBernoulli(mutationParams, trackingRandomEngine);
    }

    expectedValue = std::min(2.0, std::max(0.0, std::normal_distribution<double>(
        1.0, mutationParams.mutationStrength * 2)(trackingRandomEngine)));

    ASSERT_NEAR(std::stod((*leafGenesIt)->getElementValueAsString()), expectedValue, fpTol);
    ++leafGenesIt;
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 2.0);
    ++ depth1It;
    auto subGeneAtPos2 = *depth1It;
    ASSERT_TRUE(subGeneAtPos2->isLeaf());
    ASSERT_TRUE(std::stoi(subGeneAtPos2->getElementValueAsString()));
}
