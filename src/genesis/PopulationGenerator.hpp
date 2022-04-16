#pragma once

#include <neuro/Population.hpp>

namespace soft_npu {

struct PopulationGenerator {
public:
    virtual ~PopulationGenerator() {};
    virtual std::unique_ptr<Population> generatePopulation() = 0;
};

}
