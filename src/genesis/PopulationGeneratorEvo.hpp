#pragma once

#include "PopulationGenerator.hpp"

namespace soft_npu {

class PopulationGeneratorEvo : public PopulationGenerator {
public:
    PopulationGeneratorEvo(
            const ParamsType& params,
            RandomEngineType& randomEngine
            );

    std::unique_ptr<Population> generatePopulation() override;

private:
    const ParamsType& params;
    RandomEngineType& randomEngine;
};

}
