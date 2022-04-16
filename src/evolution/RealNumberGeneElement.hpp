#pragma once

#include "GeneElement.hpp"

namespace soft_npu {

class RealNumberGeneElement : public GeneElement {
public:
    RealNumberGeneElement(
        std::string id,
        double minValue,
        double maxValue,
        double value);

    std::string getValueAsString() const override;
    std::shared_ptr<GeneElement> mutate(double mutationStrength, RandomEngineType&) const override;
    std::shared_ptr<GeneElement> clone() const override;

    nlohmann::json valueAsJson() const override;

private:
    double minValue;
    double maxValue;
    double value;
};

}
