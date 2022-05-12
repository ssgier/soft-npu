#pragma once

#include <Aliases.hpp>
#include <vector>

namespace soft_npu {

class Gene;
struct EvolutionParams;

namespace GeneOperationUtils {

std::shared_ptr<const Gene> crossover(
    const EvolutionParams &evolutionParams,
    const std::vector<std::shared_ptr<const Gene>> &genes,
    RandomEngineType &randomEngine
);

std::shared_ptr<const Gene> assembleFromInfo(const ParamsType& geneInfoJson);

ParamsType extractFlatGeneValue(const Gene& gene);

}
}
