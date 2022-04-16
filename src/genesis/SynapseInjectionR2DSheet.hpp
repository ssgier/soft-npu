#pragma once

#include <memory>
#include <neuro/Population.hpp>
#include "NeuroComponentsFactory.hpp"

namespace soft_npu::SynapseInjectionR2DSheet {

void injectSynapses(
        const ParamsType& params,
        RandomEngineType& randomEngine,
        NeuroComponentsFactory& factory,
        Population& population
        );
}

