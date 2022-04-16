#pragma once

#include <Aliases.hpp>
#include <string>
#include <vector>
#include <memory>

namespace soft_npu {

struct MutationParams;
class GeneElement;

class Gene;

using GeneVector = std::vector<std::shared_ptr<const Gene>>;

class Gene {
public:
    explicit Gene(GeneVector&& subGenes);
    explicit Gene(std::shared_ptr<const GeneElement> geneElement);

    using SubGeneConstIterator = std::vector<std::shared_ptr<const Gene>>::const_iterator;

    std::shared_ptr<Gene> mutate(const MutationParams& mutationParams, RandomEngineType& randomEngine) const;
    std::shared_ptr<Gene> clone() const;
    bool isLeaf() const;

    SubGeneConstIterator cbeginSubGenes() const;
    SubGeneConstIterator cendSubGenes() const;

    std::shared_ptr<const GeneElement> getGeneElement() const;
    std::string getElementId() const;
    std::string getElementValueAsString() const;

private:
    std::vector<std::shared_ptr<const Gene>> subGenes;
    std::shared_ptr<const GeneElement> geneElement;
};

}
