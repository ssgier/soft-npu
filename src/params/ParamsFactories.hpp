#pragma once

#include <neuro/SynapseParams.hpp>
#include <neuro/NeuronParams.hpp>
#include <Aliases.hpp>

namespace soft_npu::ParamsFactories {

SynapseParams extractSynapseParams(const ParamsType& params);

std::shared_ptr<NeuronParams> extractExcitatoryNeuronParams(const ParamsType& params);

std::shared_ptr<NeuronParams> extractInhibitoryNeuronParams(const ParamsType& params);

}




