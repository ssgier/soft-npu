#include <sstream>
#include "TerminationReason.hpp"

using namespace soft_npu;

namespace soft_npu {

std::ostream& operator<<(std::ostream& os, TerminationReason terminationReason) {

    os << soft_npu::toString(terminationReason);

    return os;
}

std::string toString(TerminationReason terminationReason) {
    switch (terminationReason) {
        case TerminationReason::targetFitnessValueReached: return "target fitness value reached";
        case TerminationReason::maxNumIterationsReached:   return "max number of iterations reached";
        case TerminationReason::timeLimitPassed:           return "time limit passed";
        default:
            std::stringstream ss;
            ss << "Unsupported enum value: " << terminationReason;
            throw std::runtime_error(ss.str());
    }
}

}


