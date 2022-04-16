#pragma once

#include "PopulationGenerator.hpp"

namespace soft_npu {

class PopulationGeneratorR2DSheet : public PopulationGenerator {
public:
    PopulationGeneratorR2DSheet(
            const ParamsType& params,
            RandomEngineType& randomEngine
            );

    std::unique_ptr<Population> generatePopulation() override;

private:
    const ParamsType& params;
    RandomEngineType& randomEngine;
};

}

