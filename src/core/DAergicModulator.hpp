#pragma once

#include <Aliases.hpp>
#include <neuro/EligibilityTrace.hpp>
#include <deque>
#include <unordered_set>

namespace soft_npu {

class DAergicModulator {
public:
    DAergicModulator(const ParamsType& params, const Population& population);

    void createEligibilityTrace(const CycleContext& ctx, Synapse* synapse, ValueType stdpValue);
    void processReward(const CycleContext& ctx, ValueType amount);
    void processCycle(const CycleContext&);
    void setDopamineReleaseBaseRate(ValueType rate) noexcept;

private:
    std::deque<EligibilityTrace> eligibilityTraceBuffer;
    std::unordered_set<SizeType> motorNeuronIds;
    const TimeType dopamineReleasePeriod;
    ValueType dopamineReleaseBaseRate;
    TimeType nextDAReleaseTime;
    ValueType dopamineReleaseAdjustmentFactor;
    ValueType accruedRewardAmount;

    void processDopamineRelease(const CycleContext& ctx, ValueType dopamineRateToReleaseAt);
};

}
