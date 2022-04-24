#include "ParamsFactories.hpp"
namespace soft_npu::ParamsFactories {

SynapseParams extractSynapseParams(const ParamsType& params) {
    SynapseParams synapseParams;
    synapseParams.stdpTimeConstantInverse = 1.0 / static_cast<TimeType>(params["synapseParams"]["stdpTimeConstant"]);
    synapseParams.stdpCutOffTime = params["synapseParams"]["stdpCutOffTime"];
    synapseParams.stdpScaleFactorPotentiation = params["synapseParams"]["stdpScaleFactorPotentiation"];
    synapseParams.stdpScaleFactorDepression = synapseParams.stdpScaleFactorPotentiation * static_cast<double>(params["synapseParams"]["stdpDepressionVsPotentiationRatio"]);
    synapseParams.maxWeight = params["synapseParams"]["maxWeight"];
    synapseParams.eligibilityTraceTimeConstantInverse = 1.0 / static_cast<TimeType>(params["synapseParams"]["eligibilityTraceTimeConstant"]);
    synapseParams.eligibilityTraceCutOffTime =
            static_cast<TimeType>(params["synapseParams"]["eligibilityTraceCutOffTimeFactor"]) /
                synapseParams.eligibilityTraceTimeConstantInverse;
    return synapseParams;
}

std::shared_ptr<NeuronParams> extractNeuronParams(const ParamsType& params, const std::string& neuronParamsName) {

    auto neuronParams = std::make_shared<NeuronParams>();
    const auto& neuronParamsDetails = params["neuronParams"][neuronParamsName];
    neuronParams->timeConstantInverse = 1 / static_cast<ValueType>(neuronParamsDetails["timeConstant"]);
    neuronParams->refractoryPeriod = neuronParamsDetails["refractoryPeriod"];
    neuronParams->thresholdVoltage = neuronParamsDetails["thresholdVoltage"];
    neuronParams->resetVoltage = neuronParamsDetails["resetVoltage"];
    neuronParams->voltageFloor = neuronParamsDetails["voltageFloor"];
    neuronParams->isInhibitory = neuronParamsDetails["isInhibitory"];
    return neuronParams;
}

std::shared_ptr<NeuronParams> extractExcitatoryNeuronParams(const ParamsType& params) {
    return extractNeuronParams(params, "excitatory");
}

std::shared_ptr<NeuronParams> extractInhibitoryNeuronParams(const ParamsType& params) {
    return extractNeuronParams(params, "inhibitory");
}



}
