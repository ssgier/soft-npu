#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct VoltageRecording {

    VoltageRecording(const SizeType neuronId, const TimeType time, const ValueType voltage) : neuronId(neuronId),
                                                                                              time(time),
                                                                                              voltage(voltage) {}

    const SizeType neuronId;
    const TimeType time;
    const ValueType voltage;
};

}
