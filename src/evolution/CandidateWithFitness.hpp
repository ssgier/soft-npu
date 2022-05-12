#pragma once

#include <Aliases.hpp>

namespace soft_npu {

class Candidate;

struct CandidateWithFitness {
    std::shared_ptr<const Candidate> candidate;
    ValueType fitnessValue;
};

}
