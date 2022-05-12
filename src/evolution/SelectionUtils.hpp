#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct EvolutionParams;

namespace SelectionUtils {

SizeType selectParentIndex(
        const EvolutionParams& evolutionParams,
        RandomEngineType& randomEngine);

}
}
