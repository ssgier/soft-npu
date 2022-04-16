#pragma once

#include <memory>
#include <Aliases.hpp>

namespace soft_npu {

class Optimizer {
public:
    Optimizer(
            std::shared_ptr<const ParamsType> templateParams,
            SizeType populationSize,
            SizeType numSeedsPerCandidate,
            ValueType targetObjValue
            );

    std::shared_ptr<const ParamsType> optimize() const;
private:
    std::shared_ptr<const ParamsType> templateParams;
    SizeType populationSize;
    SizeType numSeedsPerCandidate;
    ValueType targetObjValue;
};

}




