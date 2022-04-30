#pragma once

#include <neuro/Neuron.hpp>
#include <core/CycleContext.hpp>

namespace soft_npu {
class FiringThresholdEvalEvent {
public:
    explicit FiringThresholdEvalEvent(Neuron& neuron) : neuron(neuron) {
    }

    void process(const CycleContext& ctx) const {
        neuron.fireIfAboveThreshold(ctx, ctx.time);
    }
private:
    Neuron& neuron;
};
}
