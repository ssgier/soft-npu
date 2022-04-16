#include <gtest/gtest.h>
#include <evolution/GeneOperationUtilsImpl.hpp>
#include <evolution/GeneOperationUtils.hpp>
#include "evolution/RealNumberGeneElement.hpp"

using namespace soft_npu;

struct CountingGenerator {
    int operator()(int typeId) {
        auto it = typeIdToNextNumber.find(typeId);
        if (it == typeIdToNextNumber.end()) {
            typeIdToNextNumber[typeId] = 1;
            return 0;
        }

        return it->second ++;
    }

    std::unordered_map<int, int> typeIdToNextNumber;
};

template<int id, typename T, T... vals>
struct ProbabilityDistributionTypeMock {

    constexpr static int typeId = id;

    template<typename ...Args>
    explicit ProbabilityDistributionTypeMock(Args...) {}

    template<typename Generator>
    T operator()(Generator& generator) {

        if constexpr(sizeof...(vals) == 0) {
            return throwOutOfBoundsError();
        } else {
            return getNextValue<vals...>(generator(typeId));
        }
    }

    T throwOutOfBoundsError() {
        throw std::runtime_error("Index out of bounds");
    }

    template<T val1>
    T getNextValue(int index) {
        if (index == 0) {
            return val1;
        } else {
            return throwOutOfBoundsError();
        }
    }

    template<T val1, T val2, T... remaining>
    T getNextValue(int index) {
        if (index == 0) {
            return val1;
        } else {
            return getNextValue<val2, remaining...>(index - 1);
        }
    }
};

auto createRealNumberLeafGene(double value) {
    return std::make_shared<Gene>(std::make_shared<RealNumberGeneElement>("", value, value, value));
}

TEST(GeneOperationUtilsTest, SingleElementGene) {

    using UniformIntDistType = ProbabilityDistributionTypeMock<0, int, 1>;
    using BernoulliDistType = ProbabilityDistributionTypeMock<1, bool>;

    EvolutionParams evolutionParams;
    CountingGenerator randomEngine;

    GeneVector genes = {createRealNumberLeafGene(1.0), createRealNumberLeafGene(2.0)};

    auto resultGene = GeneOperationUtilsImpl::crossover<UniformIntDistType, BernoulliDistType>(
        evolutionParams,
        genes,
        randomEngine
        );

    ASSERT_TRUE(resultGene->isLeaf());
    ASSERT_DOUBLE_EQ(std::stod(resultGene->getElementValueAsString()), 2.0);
}

TEST(GeneOperationUtilsTest, NoCrossoverAtDepth0) {
    using UniformIntDistType = ProbabilityDistributionTypeMock<0, int, 1>;
    using BernoulliDistType = ProbabilityDistributionTypeMock<1, bool, false>;

    EvolutionParams evolutionParams;
    CountingGenerator randomEngine;

    auto gene0 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(1.0), createRealNumberLeafGene(2.0)}));
    auto gene1 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(3.0), createRealNumberLeafGene(4.0)}));

    GeneVector genes = {gene0, gene1};

    auto resultGene = GeneOperationUtilsImpl::crossover<UniformIntDistType, BernoulliDistType>(
        evolutionParams,
        genes,
        randomEngine
    );

    ASSERT_FALSE(resultGene->isLeaf());
    ASSERT_EQ(std::distance(resultGene->cbeginSubGenes(), resultGene->cendSubGenes()), 2);

    auto subGeneIt = resultGene->cbeginSubGenes();
    ASSERT_TRUE((*subGeneIt)->isLeaf());
    ASSERT_DOUBLE_EQ(std::stod((*subGeneIt)->getElementValueAsString()), 3.0);
    ++subGeneIt;
    ASSERT_DOUBLE_EQ(std::stod((*subGeneIt)->getElementValueAsString()), 4.0);
}

TEST(GeneOperationUtilsTest, CrossoverAtDepth1) {
    using UniformIntDistType = ProbabilityDistributionTypeMock<0, int, 1, 0>;
    using BernoulliDistType = ProbabilityDistributionTypeMock<1, bool, true>;

    EvolutionParams evolutionParams;
    CountingGenerator randomEngine;

    auto gene0 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(1.0), createRealNumberLeafGene(2.0)}));
    auto gene1 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(3.0), createRealNumberLeafGene(4.0)}));

    GeneVector genes = {gene0, gene1};

    auto resultGene = GeneOperationUtilsImpl::crossover<UniformIntDistType, BernoulliDistType>(
        evolutionParams,
        genes,
        randomEngine
    );

    ASSERT_FALSE(resultGene->isLeaf());
    ASSERT_EQ(std::distance(resultGene->cbeginSubGenes(), resultGene->cendSubGenes()), 2);

    auto subGeneIt = resultGene->cbeginSubGenes();
    ASSERT_TRUE((*subGeneIt)->isLeaf());
    ASSERT_DOUBLE_EQ(std::stod((*subGeneIt)->getElementValueAsString()), 3.0);
    ++subGeneIt;
    ASSERT_DOUBLE_EQ(std::stod((*subGeneIt)->getElementValueAsString()), 2.0);
}

TEST(GeneOperationUtilsTest, NoCrossoverAtDepth1) {
    using UniformIntDistType = ProbabilityDistributionTypeMock<0, int, 1, 0>;
    using BernoulliDistType = ProbabilityDistributionTypeMock<1, bool, true, false>;

    EvolutionParams evolutionParams;
    CountingGenerator randomEngine;

    auto subGeneForGene0 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(1.0), createRealNumberLeafGene(2.0)}));
    auto subGeneForGene1 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(3.0), createRealNumberLeafGene(4.0)}));

    auto gene0 = std::make_shared<Gene>(GeneVector({subGeneForGene0}));
    auto gene1 = std::make_shared<Gene>(GeneVector({subGeneForGene1}));

    GeneVector genes = {gene0, gene1};

    auto resultGene = GeneOperationUtilsImpl::crossover<UniformIntDistType, BernoulliDistType>(
        evolutionParams,
        genes,
        randomEngine
    );

    auto leafGenesIt = (*resultGene->cbeginSubGenes())->cbeginSubGenes();
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 3.0);
    ++leafGenesIt;
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 4.0);
}

TEST(GeneOperationUtilsTest, CrossoverAtDepth1WithDepth2Tree) {
    using UniformIntDistType = ProbabilityDistributionTypeMock<0, int, 1, 0, 1>;
    using BernoulliDistType = ProbabilityDistributionTypeMock<1, bool, true, true>;

    EvolutionParams evolutionParams;
    CountingGenerator randomEngine;

    auto subGeneForGene0 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(1.0), createRealNumberLeafGene(2.0)}));
    auto subGeneForGene1 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(3.0), createRealNumberLeafGene(4.0)}));

    auto gene0 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(10), subGeneForGene0}));
    auto gene1 = std::make_shared<Gene>(GeneVector({createRealNumberLeafGene(11), subGeneForGene1}));

    GeneVector genes = {gene0, gene1};

    auto resultGene = GeneOperationUtilsImpl::crossover<UniformIntDistType, BernoulliDistType>(
        evolutionParams,
        genes,
        randomEngine
    );

    auto depth1It = resultGene->cbeginSubGenes();
    auto subGeneAtPos0 = *depth1It;
    ASSERT_TRUE(subGeneAtPos0->isLeaf());
    ASSERT_DOUBLE_EQ(std::stod(subGeneAtPos0->getElementValueAsString()), 11);

    ++ depth1It;

    auto subGeneAtPos1 = *depth1It;

    ASSERT_FALSE(subGeneAtPos1->isLeaf());
    auto leafGenesIt = subGeneAtPos1->cbeginSubGenes();
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 1.0);
    ++leafGenesIt;
    ASSERT_DOUBLE_EQ(std::stod((*leafGenesIt)->getElementValueAsString()), 4.0);
}

TEST(GeneOperationUtilsTest, JsonTransformations) {

    auto geneInfoJson = R"(
[
    {
        "id": "a",
        "prototypeValue": 1.0,
        "minValue": 0.0,
        "maxValue": 20.0
    },
    [
        {
            "id": "b",
            "prototypeValue": true
        },
        {
            "id": "c",
            "prototypeValue": 2.0,
            "minValue": 0.0,
            "maxValue": 20.0
        }
    ]
]

)"_json;

    auto expectedFlatValueJson = R"(
{
    "a": 1.0,
    "b": true,
    "c": 2.0
}
)"_json;

    auto gene = GeneOperationUtils::assembleFromJson(geneInfoJson);
    auto actualFlatValueJson = GeneOperationUtils::extractFlatGeneValueJson(*gene);

    ASSERT_EQ(actualFlatValueJson, expectedFlatValueJson);
}
