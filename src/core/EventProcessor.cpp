#include <cmath>
#include "EventProcessor.hpp"
#include "TransmissionEvent.hpp"
#include "CommonEvent.hpp"

namespace soft_npu {

template<typename T>
static BatchedRingBuffer<T> makeBuffer(
        TimeType dt, TimeType lookAheadWindow, SizeType subBufferReserveSlots) {
    auto bufferSize = static_cast<SizeType>(ceil(lookAheadWindow / dt) + 2);

    return {bufferSize, subBufferReserveSlots};
}

EventProcessor::EventProcessor(const ParamsType& params, TimeType dt,
                               SynapticTransmissionStats& synapticTransmissionStats)
: EventProcessor(
        dt,
        params["eventProcessor"]["lookAheadWindow"],
        params["eventProcessor"]["subBufferReserveSlots"],
        synapticTransmissionStats
        ) {
}

EventProcessor::EventProcessor(TimeType dt, TimeType lookAheadWindow, SizeType subBufferReserveSlots,
                               SynapticTransmissionStats& synapticTransmissionStats) :
        transmissionEventBuffer(makeBuffer<TransmissionEvent>(dt, lookAheadWindow, subBufferReserveSlots)),
        frequency(1 / dt),
        numEventsProcessed(0),
        synapticTransmissionStats(synapticTransmissionStats) {
}

void EventProcessor::pushImmediateTransmissionEvent(ValueType epsp, Neuron &targetNeuron) {
    transmissionEventBuffer.emplaceAtOffset(0, epsp, targetNeuron);
}

void EventProcessor::pushCommonEvent(TimeType targetTime, std::unique_ptr<CommonEvent>&& commonEvent) {
    commonEventsQueue.push(CommonEventWithTargetTime(targetTime, std::move(commonEvent)));
}


void EventProcessor::processCycle(const CycleContext & cycleContext) {

    processBatch(cycleContext, transmissionEventBuffer);

    for (;
            !commonEventsQueue.empty() &&
            commonEventsQueue.top().targetTime <= cycleContext.time;
            commonEventsQueue.pop()) {

        commonEventsQueue.top().commonEvent->process(cycleContext);
        ++ numEventsProcessed;
    }

    transmissionEventBuffer.clearAndAdvance();
}

SizeType EventProcessor::getNumEventsProcessed() const noexcept {
    return numEventsProcessed;
}

inline void EventProcessor::processBatch(const CycleContext &) {}

EventProcessor::CommonEventWithTargetTime::CommonEventWithTargetTime(TimeType targetTime,
                                                                     std::unique_ptr<CommonEvent> &&commonEvent) :
        targetTime(targetTime), commonEvent(std::move(commonEvent)) {

}

bool EventProcessor::CommonEventWithTargetTime::operator>(const EventProcessor::CommonEventWithTargetTime &other) const {
    return targetTime > other.targetTime;
}

}
