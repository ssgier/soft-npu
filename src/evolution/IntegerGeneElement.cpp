#include "IntegerGeneElement.hpp"

namespace soft_npu {


IntegerGeneElement::IntegerGeneElement(
    std::string id,
    int minValue,
    int maxValue,
    int value) :
    GeneElement(std::move(id)),
    minValue(minValue),
    maxValue(maxValue),
    value(value)
    {
}

std::string IntegerGeneElement::getValueAsString() const {
    return std::to_string(value);
}

std::shared_ptr<GeneElement> IntegerGeneElement::mutate(double mutationStrength, RandomEngineType & randomEngine) const {

    double target = std::normal_distribution<double>(
        value, mutationStrength * (maxValue - minValue))(randomEngine);

    int roundedTarget = std::round(target);

    int mutatedValue = std::max(minValue, std::min(maxValue, roundedTarget));

    return std::make_shared<IntegerGeneElement>(
        getId(),
        minValue,
        maxValue,
        mutatedValue
    );
}

std::shared_ptr<GeneElement> IntegerGeneElement::clone() const {
    return std::make_shared<IntegerGeneElement>(getId(), minValue, maxValue, value);
}

ParamsType IntegerGeneElement::valueAsJson() const {
    return value;
}
}
