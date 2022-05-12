#include "Candidate.hpp"
#include "GeneOperationUtils.hpp"
#include "Gene.hpp"
#include "FitnessFunction.hpp"

namespace soft_npu {

Candidate::Candidate(std::shared_ptr<const Gene> gene):
                     gene(gene),
                     geneValue(std::make_shared<ParamsType>(GeneOperationUtils::extractFlatGeneValue(*gene))) {
}

std::shared_ptr<const Gene> Candidate::getGene() const {
    return gene;
}

std::shared_ptr<const ParamsType> Candidate::getGeneValue() const {
    return geneValue;
}

}
