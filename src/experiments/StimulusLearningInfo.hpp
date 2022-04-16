#pragma once

#include "ExpUtils.hpp"

namespace soft_npu {

struct StimulusLearningInfo {
    StimulusType stimulusA;
    StimulusType stimulusB;
    TimeType endTime;
    TimeType intervalFrom;
    TimeType intervalTo;
    ValueType rewardDosage;
    SizeType detectorAChannelIdFrom;
    SizeType detectorAChannelIdTo;
    SizeType detectorBChannelIdFrom;
    SizeType detectorBChannelIdTo;
    TimeType readOutTime;
    TimeType rewardDelay;
    TimeType statsStartTime;
};

}
