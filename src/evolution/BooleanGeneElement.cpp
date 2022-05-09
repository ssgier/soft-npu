#include "BooleanGeneElement.hpp"

namespace soft_npu {


BooleanGeneElement::BooleanGeneElement(std::string id, bool value):
    GeneElement(std::move(id)), value(value) {

}

std::string BooleanGeneElement::getValueAsString() const {
    return std::to_string(value);
}

std::shared_ptr<GeneElement> BooleanGeneElement::mutate(double, RandomEngineType &) const {
    return std::make_shared<BooleanGeneElement>(getId(), !value);
}

std::shared_ptr<GeneElement> BooleanGeneElement::clone() const {
    return std::make_shared<BooleanGeneElement>(getId(), value);
}

ParamsType BooleanGeneElement::valueAsJson() const {
    return value;
}
}
