#include "RewardDoseInfo.hpp"

namespace soft_npu {

bool operator< (const RewardDoseInfo& lhs, const RewardDoseInfo& rhs) {
    return lhs.time < rhs.time;
}

}