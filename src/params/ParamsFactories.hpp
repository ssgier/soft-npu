#pragma once

#include <neuro/SynapseParams.hpp>
#include <neuro/NeuronParams.hpp>
#include <Aliases.hpp>

namespace soft_npu::ParamsFactories {

std::shared_ptr<SynapseParams> extractSynapseParams(const ParamsType& params);
std::shared_ptr<NeuronParams> extractNeuronParams(const ParamsType& params, const std::string& neuronParamsName);
std::shared_ptr<NeuronParams> extractExcitatoryNeuronParams(const ParamsType& params);
std::shared_ptr<NeuronParams> extractInhibitoryNeuronParams(const ParamsType& params);

}




