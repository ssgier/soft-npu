#pragma once

#include <core/CycleContext.hpp>
#include <core/StaticContext.hpp>
#include <neuro/SynapseParams.hpp>
#include <neuro/Synapse.hpp>

namespace soft_npu {

struct Synapse;

struct EligibilityTrace {

    EligibilityTrace(const CycleContext& ctx, Synapse* synapse, ValueType stdpValue) noexcept :
        synapse(synapse),
        lastTime(ctx.time),
        expiryTime(ctx.time + synapse->synapseParams->eligibilityTraceCutOffTime),
        lastValue(stdpValue * synapse->synapseParams->eligibilityTraceTimeConstantInverse) {

    }

    ValueType updateAndGetIntegralValue(const CycleContext& ctx) noexcept {

        TimeType stopTime = std::min(ctx.time, expiryTime);

        auto tauInverse = synapse->synapseParams->eligibilityTraceTimeConstantInverse;
        auto timeDiff = stopTime - lastTime;
        ValueType integralValue = (1 - exp(- timeDiff * tauInverse)) * lastValue / tauInverse;

        lastValue *= exp(- timeDiff * tauInverse);
        lastTime = ctx.time;

        return integralValue;
    }

    Synapse* synapse;
    TimeType lastTime;
    TimeType expiryTime;
    ValueType lastValue;
};

}
