#pragma once

#include <Aliases.hpp>

namespace soft_npu {

class SynapticTransmissionStats {
public:
    void incrementTransmissionCount() {
        ++ transmissionCount;
    }

    void increaseTransmissionCount(SizeType by) {
        transmissionCount += by;
    }

    long getTransmissionCount() const {
        return transmissionCount;
    }
private:
    long transmissionCount = 0;
};

}
