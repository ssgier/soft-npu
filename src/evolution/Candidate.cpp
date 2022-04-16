#include "Candidate.hpp"
#include "GeneOperationUtils.hpp"
#include "Gene.hpp"
#include "FitnessFunction.hpp"

namespace soft_npu {

Candidate::Candidate(std::shared_ptr<const Gene> gene, const FitnessFunction& proxyFitnessFunction,
                     const FitnessFunction& mainFitnessFunction):
                     gene(gene),
                     geneValueJson(std::make_shared<nlohmann::json>(GeneOperationUtils::extractFlatGeneValueJson(*gene))),
                     proxyFitnessFunction(proxyFitnessFunction),
                     mainFitnessFunction(mainFitnessFunction) {

}

void Candidate::evaluateFitnessProxy() {
    if (!isProxyFitnessValueCached) {
        cachedProxyFitnessValue = proxyFitnessFunction.evaluate(*geneValueJson);
        isProxyFitnessValueCached = true;
    }
}

void Candidate::evaluateFitness() {
    if (!isMainFitnessValueCached) {
        cachedMainFitnessValue = mainFitnessFunction.evaluate(*geneValueJson);
        isMainFitnessValueCached = true;
    }
}

double Candidate::getFitness() const {

    if (isMainFitnessValueCached) {
        return cachedMainFitnessValue;
    } else {
        throw std::runtime_error("Fitness has not been evaluated");
    }
}

double Candidate::getFitnessProxy() const {

    if (isProxyFitnessValueCached) {
        return cachedProxyFitnessValue;
    } else {
        throw std::runtime_error("Fitness proxy has not been evaluated");
    }
}

std::shared_ptr<const Gene> Candidate::getGene() const {
    return gene;
}

std::shared_ptr<const nlohmann::json> Candidate::getGeneValueJson() const {
    return geneValueJson;
}
}
