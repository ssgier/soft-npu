#pragma once

namespace soft_npu {

class Population;
class DAergicModulator;
class EventProcessor;
class SynapticTransmissionStats;
class CycleOutputBuffer;

struct StaticContext {
    StaticContext(
            EventProcessor& eventProcessor,
            DAergicModulator& dopaminergicModulator,
            const Population& population,
            CycleOutputBuffer& cycleOutputBuffer,
            SynapticTransmissionStats& synapticTransmissionStats) noexcept :
            eventProcessor(eventProcessor),
            dopaminergicModulator(dopaminergicModulator),
            population(population),
            cycleOutputBuffer(cycleOutputBuffer),
            synapticTransmissionStats(synapticTransmissionStats) {
    }

    EventProcessor& eventProcessor;
    DAergicModulator& dopaminergicModulator;
    const Population& population;
    CycleOutputBuffer& cycleOutputBuffer;
    SynapticTransmissionStats& synapticTransmissionStats;
};

}
