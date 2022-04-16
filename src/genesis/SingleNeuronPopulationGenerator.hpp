#pragma once

#include <memory>
#include <neuro/Population.hpp>
#include <genesis/PopulationGenerator.hpp>

namespace soft_npu {

class SingleNeuronPopulationGenerator : public PopulationGenerator {
public:
    SingleNeuronPopulationGenerator(
            const ParamsType& params,
            RandomEngineType& randomEngine);

    std::unique_ptr<Population> generatePopulation() override;

private:
    const ParamsType& params;
    RandomEngineType& randomEngine;
};

}




