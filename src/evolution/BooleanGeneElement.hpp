#pragma once

#include "GeneElement.hpp"

namespace soft_npu {

class BooleanGeneElement : public GeneElement {
public:
    BooleanGeneElement(
        std::string id,
        bool value);

    std::string getValueAsString() const override;
    std::shared_ptr<GeneElement> mutate(double mutationStrength, RandomEngineType&) const override;
    std::shared_ptr<GeneElement> clone() const override;

    nlohmann::json valueAsJson() const override;
private:
    bool value;
};

}
