#pragma once

#include <Aliases.hpp>
#include <evolution/EvolutionParams.hpp>
#include "EvolutionWrapperResult.hpp"

namespace soft_npu {

namespace EvolutionWrapper {

EvolutionWrapperResult run(
    const ParamsType& paramsTemplate,
    const ParamsType& geneInfoTemplate,
    const EvolutionParams& evolutionParams);

EvolutionParams calibrate(const ParamsType &paramsTemplate, const ParamsType &geneInfoTemplate,
          double abortTimeSeconds);

}
}
