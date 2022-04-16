#pragma once

#include "VoltageRecording.hpp"
#include "NeuronSpikeInfo.hpp"

namespace soft_npu {

struct Recordings {
    std::vector<VoltageRecording> voltageRecordings;
    std::vector<NeuronSpikeInfo> neuronSpikeRecordings;
    SizeType numExcitatorySpikes = 0;
    SizeType numInhibitorySpikes = 0;
};

}
