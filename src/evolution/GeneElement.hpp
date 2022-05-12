#pragma once

#include <string>
#include <memory>
#include <Aliases.hpp>

namespace soft_npu {

class GeneElement {
public:
    GeneElement(std::string id);

    std::string getId() const;

    virtual std::string getValueAsString() const = 0;
    virtual std::shared_ptr<GeneElement> mutate(
        double mutationStrength, RandomEngineType&) const = 0;
    virtual std::shared_ptr<GeneElement> clone() const = 0;
    virtual ~GeneElement() = default;

    virtual ParamsType value() const = 0;

private:
    std::string id;
};

}
