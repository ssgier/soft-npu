#include "ParamsFactories.hpp"
namespace soft_npu::ParamsFactories {

SynapseParams extractSynapseParams(const ParamsType& params) {

    const auto& synapseParamsDetails = params["synapseParams"];

    SynapseParams synapseParams;
    synapseParams.stdpTimeConstantInverse = 1.0 / static_cast<TimeType>(synapseParamsDetails["stdpTimeConstant"]);
    synapseParams.stdpCutOffTime = synapseParamsDetails["stdpCutOffTime"];
    synapseParams.stdpScaleFactorPotentiation = synapseParamsDetails["stdpScaleFactorPotentiation"];
    synapseParams.stdpScaleFactorDepression = synapseParams.stdpScaleFactorPotentiation * static_cast<ValueType>(synapseParamsDetails["stdpDepressionVsPotentiationRatio"]);
    synapseParams.maxWeight = synapseParamsDetails["maxWeight"];
    synapseParams.eligibilityTraceTimeConstantInverse = 1.0 / static_cast<TimeType>(synapseParamsDetails["eligibilityTraceTimeConstant"]);
    synapseParams.eligibilityTraceCutOffTime =
            static_cast<TimeType>(synapseParamsDetails["eligibilityTraceCutOffTimeFactor"]) /
                synapseParams.eligibilityTraceTimeConstantInverse;

    auto it = synapseParamsDetails.find("shortTermPlasticityParams");
    if (it != synapseParamsDetails.end()) {
        synapseParams.shortTermPlasticityParams.emplace();
        synapseParams.shortTermPlasticityParams->isDepression = (*it)["isDepression"];
        synapseParams.shortTermPlasticityParams->restingValue = (*it)["restingValue"];
        synapseParams.shortTermPlasticityParams->changeParameter = (*it)["changeParameter"];
        synapseParams.shortTermPlasticityParams->tauInverse = 1.0 / static_cast<TimeType>((*it)["timeConstant"]);
    }

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

    auto it = neuronParamsDetails.find("epspOverrideScaleFactor");
    if (it != neuronParamsDetails.end()) {
        neuronParams->epspOverrideScaleFactor = *it;
    }

    return neuronParams;
}

std::shared_ptr<NeuronParams> extractExcitatoryNeuronParams(const ParamsType& params) {
    return extractNeuronParams(params, "excitatory");
}

std::shared_ptr<NeuronParams> extractInhibitoryNeuronParams(const ParamsType& params) {
    return extractNeuronParams(params, "inhibitory");
}



}
