#pragma once

#include "EnvContext.hpp"
#include "ExpUtils.hpp"
#include "StimulusLearningInfo.hpp"
#include <memory>

namespace soft_npu {

struct EnvEvent;

namespace EnvEventFactories {

std::unique_ptr<EnvEvent> makeStimulusLearningEvent(
        StimulusLearningInfo& slInfo,
        TimeType startTime);

std::unique_ptr<EnvEvent> makeRewardEvent(
        std::shared_ptr<StimulusLearningInfo> slInfo,
        TimeType rewardTime, ValueType dosage);

}
}

