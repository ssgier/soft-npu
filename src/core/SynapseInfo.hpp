#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct SynapseInfo {

    SynapseInfo(
            SizeType preSynapticNeuronId,
            SizeType postSynapticNeuronId,
            ValueType weight,
            bool isInhibitory)
            :
            preSynapticNeuronId(preSynapticNeuronId),
            postSynapticNeuronId(postSynapticNeuronId),
            weight(weight),
            isInhibitory(isInhibitory) {}

    SizeType preSynapticNeuronId;
    SizeType postSynapticNeuronId;
    ValueType weight;
    bool isInhibitory;
};

inline bool operator<(const SynapseInfo& lhs, const SynapseInfo& rhs) {
    if (lhs.preSynapticNeuronId == rhs.preSynapticNeuronId) {
        return lhs.postSynapticNeuronId < rhs.postSynapticNeuronId;
    } else {
        return lhs.preSynapticNeuronId < rhs.preSynapticNeuronId;
    }
}

}