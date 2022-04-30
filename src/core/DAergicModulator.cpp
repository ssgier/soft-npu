#include "DAergicModulator.hpp"
#include <neuro/Synapse.hpp>
#include <neuro/Population.hpp>

namespace soft_npu {

ValueType getDAReleaseAdjustmentFactor(const ParamsType& params) {
    ValueType traceCutOffTimeFactor =
            params["synapseParams"]["eligibilityTraceCutOffTimeFactor"];

    return 1 / (1 - exp(- traceCutOffTimeFactor));
}

DAergicModulator::DAergicModulator(const ParamsType& params, const Population& population) :
    eligibilityTraceBuffer(),
    motorNeuronIds(population.getMotorNeuronIds()),
    dopamineReleasePeriod(1.0 / static_cast<ValueType>(params["dopaminergicModulator"]["releaseFrequency"])),
    dopamineReleaseBaseRate(params["dopaminergicModulator"]["releaseBaseRate"]),
    nextDAReleaseTime(dopamineReleasePeriod),
    dopamineReleaseAdjustmentFactor(getDAReleaseAdjustmentFactor(params)),
    accruedRewardAmount(0) {
}

void DAergicModulator::processReward(const CycleContext&, ValueType amount) {
    accruedRewardAmount += amount;
}

void DAergicModulator::processCycle(const CycleContext& ctx) {
    if (nextDAReleaseTime <= ctx.time) {

        ValueType dopamineRateToReleaseAt =
            dopamineReleaseAdjustmentFactor * (dopamineReleaseBaseRate + accruedRewardAmount / dopamineReleasePeriod);
        processDopamineRelease(ctx, dopamineRateToReleaseAt);
        nextDAReleaseTime += dopamineReleasePeriod;
        accruedRewardAmount = 0;
    }
}

inline void updateSynapticWeight(
        Synapse* synapse,
        ValueType proposedWeightChange) {
    const auto& synapseParams = *synapse->synapseParams;

    ValueType weightCandidate = synapse->weight + proposedWeightChange;
    ValueType weight = std::max(
        static_cast<ValueType>(0.0),
        std::min(synapseParams.maxWeight, weightCandidate)
    );

    synapse->weight = weight;
}

void
DAergicModulator::processDopamineRelease(const CycleContext& ctx, ValueType dopamineRateToReleaseAt) {

    for (auto& eligibiliyTrace : eligibilityTraceBuffer) {

        // there is a problem here in that the tonic DA release and the punishment (negative DA) neutralize each other.
        // Should not be the case as the patterns should still be learned. Perhaps two modulator dimensions are necessary.
        // However, tonic release should be much smaller in amplitude than reward, so perhaps not a significant problem...
        if (dopamineRateToReleaseAt < 0) {
// TODO: find a way to include punishment mechanism selection in meta heuristics
//            if (motorNeuronIds.find(eligibiliyTrace.synapse->postSynapticNeuron->getNeuronId()) != motorNeuronIds.end()) {
                if (eligibiliyTrace.lastValue > 0) {
                    // NOP: synapse will be depressed
                } else {
                    dopamineRateToReleaseAt = 0; // prevent the synapse from getting reinforced on negative reward
                }
//            } else {
//                dopamineRateToReleaseAt = - dopamineRateToReleaseAt; // for non-motor neurons, reinforce synapses
//            }
        }

        ValueType proposedWeightChange = eligibiliyTrace.updateAndGetIntegralValue(ctx) * dopamineRateToReleaseAt;
        updateSynapticWeight(eligibiliyTrace.synapse, proposedWeightChange);
    }

    while(!eligibilityTraceBuffer.empty() && eligibilityTraceBuffer.front().expiryTime < ctx.time) {
        eligibilityTraceBuffer.pop_front();
    }
}

void DAergicModulator::createEligibilityTrace(const CycleContext& ctx, Synapse* synapse, ValueType stdpValue) {
    eligibilityTraceBuffer.emplace_back(ctx, synapse, stdpValue);
}

void DAergicModulator::setDopamineReleaseBaseRate(ValueType targetRate) noexcept {
    dopamineReleaseBaseRate = targetRate;
}

}
