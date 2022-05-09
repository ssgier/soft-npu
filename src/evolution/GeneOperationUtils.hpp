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

std::shared_ptr<const Gene> assembleFromJson(const ParamsType& geneInfoJson);

ParamsType extractFlatGeneValueJson(const Gene& gene);

}
}
