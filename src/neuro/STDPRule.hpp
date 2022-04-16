#pragma once

#include <Aliases.hpp>
#include <cmath>
#include "SynapseParams.hpp"

namespace soft_npu::STDPRule {

inline static ValueType evaluateSTDPRule(const SynapseParams& synapseParams, TimeType timeDiffPostVsPre) noexcept {
    if (timeDiffPostVsPre < 0) {
        return -synapseParams.stdpScaleFactorDepression * exp(timeDiffPostVsPre * synapseParams.stdpTimeConstantInverse);
    } else {
        return synapseParams.stdpScaleFactorPotentiation * exp(- timeDiffPostVsPre * synapseParams.stdpTimeConstantInverse);
    }
}

}