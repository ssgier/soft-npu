#pragma once

#include "GeneElement.hpp"

namespace soft_npu {

class IntegerGeneElement : public GeneElement {
public:
    IntegerGeneElement(
        std::string id,
        int minValue,
        int maxValue,
        int value);

    std::string getValueAsString() const override;
    std::shared_ptr<GeneElement> mutate(double mutationStrength, RandomEngineType&) const override;
    std::shared_ptr<GeneElement> clone() const override;

    ParamsType valueAsJson() const override;

private:
    int minValue;
    int maxValue;
    int value;
};

}
