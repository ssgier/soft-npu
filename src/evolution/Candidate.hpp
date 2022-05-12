#pragma once

#include <Aliases.hpp>
#include <memory>
#include <limits>

namespace soft_npu {

struct FitnessFunction;
class Gene;

class Candidate {
public:
    explicit Candidate(std::shared_ptr<const Gene> gene);

    std::shared_ptr<const Gene> getGene() const;
    std::shared_ptr<const ParamsType> getGeneValue() const;

private:
    std::shared_ptr<const Gene> gene;
    std::shared_ptr<const ParamsType> geneValue;
};

}
