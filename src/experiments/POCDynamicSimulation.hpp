#pragma once

#include <core/AbstractSimulation.hpp>
#include "OptimResultHolder.hpp"

namespace soft_npu {

class POCDynamicSimulation : public AbstractSimulation {
public:
    explicit POCDynamicSimulation(std::shared_ptr<const ParamsType> params);
    void runController(
            CycleController& controller,
            Population& population,
            TimeType simulationTime,
            SynapticTransmissionStats& synapticTransmissionStats
            ) override;

    OptimResultHolder optimResultHolder;
    ValueType rewardDosage;
    ValueType abortAfterWallSeconds;
    ValueType costAfterWallSeconds;
    bool flipDetectorChannels;
};

}
