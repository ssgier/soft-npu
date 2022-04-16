#pragma once

namespace soft_npu {

class Population;
class DAergicModulator;
class EventProcessor;
class SynapticTransmissionStats;
class CycleOutputBuffer;
struct SynapseParams;

struct StaticContext {
    StaticContext(
            EventProcessor& eventProcessor,
            DAergicModulator& dopaminergicModulator,
            const SynapseParams& synapseParams,
            const Population& population,
            CycleOutputBuffer& cycleOutputBuffer,
            SynapticTransmissionStats& synapticTransmissionStats) noexcept :
            eventProcessor(eventProcessor),
            dopaminergicModulator(dopaminergicModulator),
            synapseParams(synapseParams),
            population(population),
            cycleOutputBuffer(cycleOutputBuffer),
            synapticTransmissionStats(synapticTransmissionStats) {
    }

    EventProcessor& eventProcessor;
    DAergicModulator& dopaminergicModulator;
    const SynapseParams& synapseParams;
    const Population& population;
    CycleOutputBuffer& cycleOutputBuffer;
    SynapticTransmissionStats& synapticTransmissionStats;
};

}
