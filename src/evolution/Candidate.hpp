#pragma once

#include <Aliases.hpp>
#include <memory>
#include <limits>

namespace soft_npu {

struct FitnessFunction;
class Gene;

class Candidate {
public:
    Candidate(
        std::shared_ptr<const Gene> gene,
        const FitnessFunction& proxyFitnessFunction,
        const FitnessFunction& mainFitnessFunction);

    void evaluateFitnessProxy();
    void evaluateFitness();

    double getFitness() const;
    double getFitnessProxy() const;
    std::shared_ptr<const Gene> getGene() const;
    std::shared_ptr<const ParamsType> getGeneValueJson() const;

private:
    std::shared_ptr<const Gene> gene;
    std::shared_ptr<const ParamsType> geneValueJson;

    const FitnessFunction& proxyFitnessFunction;
    const FitnessFunction& mainFitnessFunction;

    bool isProxyFitnessValueCached = false;
    double cachedProxyFitnessValue = std::numeric_limits<double>::quiet_NaN();

    bool isMainFitnessValueCached = false;
    double cachedMainFitnessValue = std::numeric_limits<double>::quiet_NaN();

};

}
