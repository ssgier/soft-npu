#include <sstream>
#include "OptimParamsMapper.hpp"

namespace soft_npu {

OptimParamsMapper::OptimParamsDimInfo::OptimParamsDimInfo(std::string path, double floorValue, double ceilValue) :
    path(std::move(path)), floorValue(floorValue), ceilValue(ceilValue) {

}

OptimParamsMapper::OptimParamsMapper(std::shared_ptr<const ParamsType> templateParams)
    : templateParams(templateParams) {

    dimInfos.emplace_back("nonCoherentStimulator.rate", 0.5, 5);
    dimInfos.emplace_back("nonCoherentStimulator.epsp", 0.3, 5);

    dimInfos.emplace_back("dopaminergicModulator.releaseBaseRate", 0.01, 2.0);

    dimInfos.emplace_back("synapseParams.eligibilityTraceTimeConstant", 0.1, 1.0);

    dimInfos.emplace_back("neuronParams.excitatory.timeConstant", 1e-3, 30e-3);
    dimInfos.emplace_back("neuronParams.excitatory.refractoryPeriod", 1e-3, 15e-3);
    dimInfos.emplace_back("neuronParams.excitatory.voltageFloor", -1.5, 0.0);

    dimInfos.emplace_back("neuronParams.inhibitory.timeConstant", 1e-3, 10e-3);
    dimInfos.emplace_back("neuronParams.inhibitory.refractoryPeriod", 1e-3, 10e-3);

    dimInfos.emplace_back("pocDynamicSimulation.rewardDosage", 0.05, 2.0);

    dimInfos.emplace_back("synapseParams.maxWeight", 0.5, 1.0);
    dimInfos.emplace_back("synapseParams.stdpTimeConstantPotentiation", 1e-3, 30e-3);

    dimInfos.emplace_back("populationGenerators.pEvo.inhibitorySynapseWeight", 0.1, 2.0);

    dimInfos.emplace_back("synapseParams.stdpDepressionVsPotentiationRatio", 1.0, 2.0);
    dimInfos.emplace_back("synapseParams.stdpCutOffTime", 20e-3, 50e-3);

    dimInfos.emplace_back("populationGenerators.pEvo.targetNumMotorNeurons", 8, 200);
}

SizeType OptimParamsMapper::getNumDimensions() const {
    return dimInfos.size();
}

std::vector<std::string> splitString(const std::string& string) {
    std::vector<std::string> rv;
    std::stringstream ss(string);
    std::string intermediate;

    while(std::getline(ss, intermediate, '.')) {
        rv.push_back(intermediate);
    }

    return rv;
}

template <typename T>
T& getRef(T& params, const std::vector<std::string>& path, SizeType pathPos) {
    if (pathPos == path.size()) {
        return params;
    } else {
        return getRef(params[path[pathPos]], path, pathPos + 1);
    }
}

double OptimParamsMapper::extractInitialValue(const OptimParamsDimInfo &dimInfo) const {
    const double& templateValue = getRef(*templateParams, splitString(dimInfo.path), 0);

    double initialValueCandidate = (templateValue - dimInfo.floorValue) / (dimInfo.ceilValue - dimInfo.floorValue);

    return std::max(0.0, std::min(1.0, initialValueCandidate));
}

std::vector<double> OptimParamsMapper::getInitialValues() const {
    std::vector<double> rv;

    auto that = this;
    std::transform(dimInfos.cbegin(), dimInfos.cend(), std::back_inserter(rv), [that](const auto& dimInfo) {
        return that->extractInitialValue(dimInfo);
    });

    return rv;
}

void OptimParamsMapper::applyValue(ParamsType &params, const OptimParamsDimInfo &dimInfo, double value) const {
    auto& paramValue = getRef(params, splitString(dimInfo.path), 0);
    paramValue = dimInfo.floorValue + (dimInfo.ceilValue - dimInfo.floorValue) * value;
}

std::shared_ptr<ParamsType> OptimParamsMapper::getParams(const double* values) const {

    auto params = std::make_shared<ParamsType>(*templateParams);

    SizeType numDims = getNumDimensions();

    for (SizeType i = 0; i < numDims; ++i) {
        applyValue(*params, dimInfos[i], values[i]);
    }

    return params;
}

}
