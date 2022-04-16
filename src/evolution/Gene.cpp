#include "Gene.hpp"
#include "GeneElement.hpp"
#include "MutationParams.hpp"

namespace soft_npu {

Gene::Gene(std::vector<std::shared_ptr<const Gene>>&& subGenes): subGenes(std::move(subGenes)) {
}

Gene::Gene(std::shared_ptr<const GeneElement> geneElement): geneElement(geneElement) {
}

GeneVector mutateGeneVector(const GeneVector& genes, const MutationParams& mutationParams,
                                   RandomEngineType& randomEngine) {
    GeneVector rv;
    std::transform(genes.cbegin(), genes.cend(), std::back_inserter(rv), [&mutationParams, &randomEngine](auto gene) {
        return gene->mutate(mutationParams, randomEngine);
    });

    return rv;
}

std::shared_ptr<Gene> Gene::mutate(const MutationParams &mutationParams, RandomEngineType& randomEngine) const {

    if (std::bernoulli_distribution(mutationParams.mutationProbability)(randomEngine)) {
        return isLeaf() ?
            std::make_shared<Gene>(geneElement->mutate(mutationParams.mutationStrength, randomEngine)) :
            std::make_shared<Gene>(mutateGeneVector(subGenes, mutationParams, randomEngine));
    } else {
        return clone();
    }
}

std::shared_ptr<Gene> Gene::clone() const {
    return isLeaf() ?
        std::make_shared<Gene>(geneElement->clone()) :
        std::make_shared<Gene>(GeneVector (subGenes));
}

bool Gene::isLeaf() const {
    return geneElement != nullptr;
}

Gene::SubGeneConstIterator Gene::cbeginSubGenes() const {
    return subGenes.cbegin();
}

Gene::SubGeneConstIterator Gene::cendSubGenes() const {
    return subGenes.cend();
}

std::string Gene::getElementId() const {
    return geneElement->getId();
}

std::string Gene::getElementValueAsString() const {
    return geneElement->getValueAsString();
}

std::shared_ptr<const GeneElement> Gene::getGeneElement() const {
    return geneElement;
}
}
