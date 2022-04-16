#pragma once

#include <neuro/ChannelProjector.hpp>
#include "DAergicModulator.hpp"
#include "EventProcessor.hpp"
#include "CycleInputBuffer.hpp"
#include "CycleOutputBuffer.hpp"
#include "NonCoherentStimulator.hpp"
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

struct Recordings;

class CycleController : private boost::noncopyable {
public:
    CycleController(
            const ParamsType& params,
            RandomEngineType& randomEngine,
            Population& population,
            bool recordSpikes,
            const std::vector<std::pair<SizeType, TimeType>>& neuronIdTimePairsToRecordVoltageAt,
            SynapticTransmissionStats& synapticTransmissionStats
            );

    CycleInputBuffer& getCycleInputBuffer();
    CycleOutputBuffer& getCycleOutputBuffer();
    void runCycle();
    TimeType getTime() const noexcept;
    std::shared_ptr<const Recordings> getRecordings() const noexcept;
    uint64_t getNumEventsProcessed() const noexcept;

    void setNonCoherentStimulationRate(ValueType rate) noexcept;
    void setDopamineReleaseBaseRate(ValueType rate) noexcept;
    TimeType getTimeIncrement() const noexcept;

private:

    TimeType dt;
    SizeType currentCycle;
    TimeType currentTime;

    CycleInputBuffer cycleInputBuffer;
    CycleOutputBuffer cycleOutputBuffer;
    NonCoherentStimulator nonCoherentStimulator;
    EventProcessor eventProcessor;
    DAergicModulator dopaminergicModulator;
    const StaticContext staticContext;
    std::shared_ptr<Recordings> recordings;
};

}
