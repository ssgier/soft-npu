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

std::shared_ptr<NeuronParams> extractNeuronParams(const ParamsType& params, bool inhibitory) {

    auto innerKey = inhibitory ? "inhibitory" : "excitatory";

    auto neuronParams = std::make_shared<NeuronParams>();
    neuronParams->timeConstantInverse = 1 / static_cast<ValueType>(params["neuronParams"][innerKey]["timeConstant"]);
    neuronParams->refractoryPeriod = params["neuronParams"][innerKey]["refractoryPeriod"];
    neuronParams->thresholdVoltage = params["neuronParams"][innerKey]["thresholdVoltage"];
    neuronParams->resetVoltage = params["neuronParams"][innerKey]["resetVoltage"];
    neuronParams->voltageFloor = params["neuronParams"][innerKey]["voltageFloor"];
    neuronParams->isInhibitory = params["neuronParams"][innerKey]["isInhibitory"];
    return neuronParams;
}

std::shared_ptr<NeuronParams> extractExcitatoryNeuronParams(const ParamsType& params) {
    return extractNeuronParams(params, false);
}

std::shared_ptr<NeuronParams> extractInhibitoryNeuronParams(const ParamsType& params) {
    return extractNeuronParams(params, true);
}



}
