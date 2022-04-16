#pragma once

#include <random>
#include "PopulationGenerator.hpp"

namespace soft_npu {

class PopulationGeneratorP1000 : public PopulationGenerator {
public:
    PopulationGeneratorP1000(
            const ParamsType& params,
            RandomEngineType& randomEngine);

    std::unique_ptr<Population> generatePopulation() override;

private:
    const ParamsType& params;
    RandomEngineType& randomEngine;
};

}
