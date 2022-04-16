#pragma once

#include <genesis/SingleNeuronPopulationGenerator.hpp>
#include "neuro/Population.hpp"

namespace soft_npu::PopulationGeneratorFactory {

std::unique_ptr<PopulationGenerator> createFromParams(
        const ParamsType& params,
        RandomEngineType& randomEngine);

}






