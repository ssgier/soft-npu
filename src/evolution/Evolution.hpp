#pragma once

#include "EvolutionParams.hpp"
#include "FitnessFunction.hpp"
#include "EvolutionResult.hpp"

namespace soft_npu {

namespace Evolution {

template<typename F>
struct FitnessEval : public FitnessFunction {
    explicit FitnessEval(F&& evalFunction) : evalFunction(std::forward<F>(evalFunction)) {}

    double evaluate(const ParamsType& geneValue, SizeType seed) const override {
        return evalFunction(geneValue, seed);
    }

    const F evalFunction;
};

template<typename FitnessEvalType>
EvolutionResult run(
    const EvolutionParams& evolutionParams,
    FitnessEvalType&& fitnessEval,
    const ParamsType& geneInfoJson) {

    auto fitnessFunction = std::make_unique<FitnessEval<FitnessEvalType>>(
        std::forward<FitnessEvalType>(fitnessEval));

    return runImpl(
        evolutionParams,
        *fitnessFunction,
        geneInfoJson);
}

EvolutionResult runImpl(
    const EvolutionParams& evolutionParams,
    const FitnessFunction& fitnessFunction,
    const ParamsType& geneInfoJson);
}

}
