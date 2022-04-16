#pragma once

#include "EvolutionParams.hpp"
#include "FitnessFunction.hpp"
#include "EvolutionResult.hpp"

namespace soft_npu {

namespace Evolution {

template<typename F>
struct FitnessEval : public FitnessFunction {
    explicit FitnessEval(F&& evalFunction) : evalFunction(std::forward<F>(evalFunction)) {}

    double evaluate(const nlohmann::json& geneValueJson) const override {
        return evalFunction(geneValueJson);
    }

    const F evalFunction;
};

template<typename ProxyFitnessEvalType, typename MainFitnessEvalType>
EvolutionResult run(
    const EvolutionParams& evolutionParams,
    ProxyFitnessEvalType&& proxyFitnessEval,
    MainFitnessEvalType&& mainFitnessEval,
    const nlohmann::json& geneInfoJson) {

    auto proxyFitnessFunction = std::make_unique<FitnessEval<ProxyFitnessEvalType>>(
        std::forward<ProxyFitnessEvalType>(proxyFitnessEval));

    auto mainFitnessFunction = std::make_unique<FitnessEval<MainFitnessEvalType>>(
        std::forward<MainFitnessEvalType>(mainFitnessEval));

    return runImpl(
        evolutionParams,
        *proxyFitnessFunction,
        *mainFitnessFunction,
        geneInfoJson);
}

EvolutionResult runImpl(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& proxyFitnessFunction,
    const FitnessFunction& mainFitnessFunction,
    const nlohmann::json& geneInfoJson);
}

}
