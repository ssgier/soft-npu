#pragma once

#include <core/CycleContext.hpp>

namespace soft_npu {

class Neuron;

struct SpikeListener {
    virtual ~SpikeListener() {};
    virtual void onSpike(const CycleContext& cycleContext, const Neuron& neuron) const = 0;
};

template<typename F>
class SpikeListenerImpl : public SpikeListener {
public:
    explicit SpikeListenerImpl(F&& processingFunction) : processingFunction(std::forward<F>(processingFunction)) {}

    void onSpike(const CycleContext& cycleContext, const Neuron& neuron) const override {
        processingFunction(cycleContext, neuron);
    }

private:
    const F processingFunction;
};

}
