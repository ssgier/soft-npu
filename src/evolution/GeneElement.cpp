#include "GeneElement.hpp"

namespace soft_npu {
    soft_npu::GeneElement::GeneElement(std::string id) : id(std::move(id)) {
    }

    std::string soft_npu::GeneElement::getId() const {
        return id;
    }
}
