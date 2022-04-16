#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct NeuronInfo {
    NeuronInfo(SizeType neuronId, bool isInhibitory) noexcept : neuronId(neuronId), isInhibitory(isInhibitory) {}

    const SizeType neuronId;
    const bool isInhibitory;
};

}