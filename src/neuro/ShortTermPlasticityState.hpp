#pragma once

#include <Aliases.hpp>
#include <limits>
#include <core/CycleContext.hpp>
#include <core/StaticContext.hpp>
#include <neuro/SynapseParams.hpp>


namespace soft_npu {

struct ShortTermPlasticityState {

    void update(const CycleContext& ctx, const SynapseParams& synapseParams) {
        if (lastTime < ctx.time) {
            const auto& stpParams = *synapseParams.shortTermPlasticityParams;
            lastValue = stpParams.restingValue + exp(- (ctx.time - lastTime) * stpParams.tauInverse) * (lastValue - stpParams.restingValue);
            lastTime = ctx.time;
        }
    }

    void onTransmission(const CycleContext& ctx, const SynapseParams& synapseParams) {
        const auto& stpParams = *synapseParams.shortTermPlasticityParams;

        update(ctx, synapseParams);

        if (stpParams.isDepression) {
            lastValue -= stpParams.changeParameter * lastValue;
        } else {
            lastValue += stpParams.changeParameter * (1.0 - lastValue);
        }
    }

    TimeType lastTime = std::numeric_limits<TimeType>::lowest();
    ValueType lastValue = 0;
};

}
