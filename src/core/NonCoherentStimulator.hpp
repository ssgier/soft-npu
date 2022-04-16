#pragma once

#include <Aliases.hpp>
#include <vector>
#include <random>

namespace soft_npu {

class Population;
struct CycleContext;
class Neuron;

class NonCoherentStimulator {
public:

    NonCoherentStimulator(
            const ParamsType& params,
            RandomEngineType& randomEngine,
            Population& population,
            TimeType dt);

    void processCycle(const CycleContext&);
    void setRate(ValueType rate) noexcept;

private:
    RandomEngineType& randomEngine;
    std::vector<std::reference_wrapper<Neuron>> neuronsToStimulate;
    std::poisson_distribution<SizeType> poissonDistribution;
    TimeType dt;
    ValueType epsp;
};

}


