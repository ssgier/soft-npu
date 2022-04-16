#include "RealNumberGeneElement.hpp"

namespace soft_npu {

RealNumberGeneElement::RealNumberGeneElement(
    std::string id,
    double minValue,
    double maxValue,
    double value) :
    GeneElement(std::move(id)),
    minValue(minValue),
    maxValue(maxValue),
    value(value) {
}

std::string RealNumberGeneElement::getValueAsString() const {
    return std::to_string(value);
}

std::shared_ptr<GeneElement> RealNumberGeneElement::mutate(
    double mutationStrength,
    RandomEngineType& randomEngine) const {

    double mutatedValue = std::max(
        minValue,
        std::min(maxValue,
                 std::normal_distribution<double>(value, mutationStrength * (maxValue - minValue))(randomEngine)));

    return std::make_shared<RealNumberGeneElement>(
        getId(),
        minValue,
        maxValue,
        mutatedValue
        );
}

std::shared_ptr<GeneElement> RealNumberGeneElement::clone() const {
    return std::make_shared<RealNumberGeneElement>(getId(), minValue, maxValue, value);
}

nlohmann::json RealNumberGeneElement::valueAsJson() const {
    return value;
}


}