#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct NeuronSpikeInfo {
    NeuronSpikeInfo(TimeType time, SizeType neuronId) noexcept : neuronId(neuronId), time(time) {}

    const SizeType neuronId;
    const TimeType time;
};

}