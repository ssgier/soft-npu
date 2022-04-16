#pragma once

#include <genesis/PopulationGenerator.hpp>
#include <neuro/Population.hpp>

namespace soft_npu {

class PopulationGeneratorDetailedParams : public PopulationGenerator {
public:
    PopulationGeneratorDetailedParams(
            const ParamsType& params,
            RandomEngineType& randomEngine
            );

    std::unique_ptr<Population> generatePopulation() override;

private:
    const ParamsType& params;
    RandomEngineType& randomEngine;
};

}



