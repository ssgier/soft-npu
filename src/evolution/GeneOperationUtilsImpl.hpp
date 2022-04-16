#pragma once

#include "Gene.hpp"
#include "EvolutionParams.hpp"

namespace soft_npu::GeneOperationUtilsImpl {

template<
    typename UniFormIntDistType,
    typename BernoulliDistType,
    typename RndEngineType
    >
std::shared_ptr<const Gene> crossover(
    const EvolutionParams& evolutionParams,
    const GeneVector& genes,
    RndEngineType& randomEngine
) {
    auto exampleGene = genes[0];

    if (exampleGene->isLeaf() || !BernoulliDistType(evolutionParams.crossoverProbability)(randomEngine)) {

        UniFormIntDistType indexSamplingDistribution(
            0,  genes.size() - 1);
        return genes[indexSamplingDistribution(randomEngine)]->clone();
    } else {

        auto numberOfSubGenes = std::distance(exampleGene->cbeginSubGenes(), exampleGene->cendSubGenes());

        GeneVector crossedSubGenes;
        crossedSubGenes.reserve(numberOfSubGenes);

        for (int subGenePosition = 0; subGenePosition < numberOfSubGenes; ++ subGenePosition) {
            GeneVector subGenesAtPosition;
            subGenesAtPosition.reserve(genes.size());

            std::transform(genes.cbegin(), genes.cend(), std::back_inserter(subGenesAtPosition), [subGenePosition](auto gene){
                auto it = gene->cbeginSubGenes();
                std::advance(it, subGenePosition);
                return *it;
            });

            crossedSubGenes.push_back(
                crossover<UniFormIntDistType, BernoulliDistType>(evolutionParams, subGenesAtPosition, randomEngine));
        }

        return std::make_shared<Gene>(std::move(crossedSubGenes));
    }
}

}
