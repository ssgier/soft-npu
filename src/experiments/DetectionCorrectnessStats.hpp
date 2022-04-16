#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct DetectionCorrectnessStats {
    SizeType numCorrectDetections = 0;
    SizeType numWrongDetections = 0;
    SizeType numAbstinences = 0;
};

}
