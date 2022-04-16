#pragma once

#include <Aliases.hpp>
#include <genesis/PopulationGenerator.hpp>
#include <core/SimulationResult.hpp>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

class CycleController;
class Population;
class SynapticTransmissionStats;

class AbstractSimulation : private boost::noncopyable {
public:
    AbstractSimulation(std::shared_ptr<const ParamsType> params);

    void recordVoltage(SizeType neuronId, TimeType time);
    SimulationResult run();
    virtual void runController(
            CycleController& controller,
            Population& population,
            TimeType simulationTime,
            SynapticTransmissionStats& synapticTransmissionStats) = 0;
    virtual ~AbstractSimulation() = default;
protected:
    RandomEngineType randomEngine;
    std::shared_ptr<const ParamsType> params;
    std::unique_ptr<PopulationGenerator> populationGenerator;
private:
    std::vector<std::pair<SizeType, TimeType>> neuronIdTimePairsToRecordVoltageAt;
};

}
