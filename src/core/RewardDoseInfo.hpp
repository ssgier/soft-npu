#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct RewardDoseInfo {
    RewardDoseInfo(TimeType time, ValueType dosage) noexcept : time(time), dosage(dosage) {}

    const TimeType time;
    const ValueType dosage;
};

bool operator< (const RewardDoseInfo& lhs, const RewardDoseInfo& rhs);

}