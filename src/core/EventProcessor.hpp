#pragma once

#include <memory>
#include <queue>
#include "BatchedRingBuffer.hpp"
#include <Aliases.hpp>
#include "TransmissionEvent.hpp"
#include "FiringThresholdEvalEvent.hpp"
#include <neuro/Neuron.hpp>
#include <neuro/Synapse.hpp>
#include "CommonEvent.hpp"
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

struct StaticContext;
class SynapticTransmissionStats;

class EventProcessor : private boost::noncopyable {
public:
    explicit EventProcessor(
            const ParamsType& params,
            TimeType dt,
            SynapticTransmissionStats& synapticTransmissionStats
            );

    void processCycle(const CycleContext&);

    void pushCommonEvent(TimeType targetTime, std::unique_ptr<CommonEvent>&& commonEvent);

    void pushImmediateTransmissionEvent(ValueType epsp, Neuron& targetNeuron);

    void pushFiringThresholdEvalEvent(Neuron& neuron) {
        firingThresholdEvalBuffer.emplace_back(neuron);
    }

    void pushSynapticTransmissionEvent(TimeType delay, ValueType epsp, Synapse* synapse, Neuron &targetNeuron) {
        assert(delay > 0);
        pushBufferedEvent(delay, transmissionEventBuffer, epsp, synapse, targetNeuron);
    }

    SizeType getNumEventsProcessed() const noexcept;

private:

    explicit EventProcessor(
            TimeType dt,
            TimeType lookAheadWindow,
            SizeType subBufferReserveSlots,
            SynapticTransmissionStats& synapticTransmissionStats);

    SizeType getTargetOffset(TimeType delay) const noexcept {

        assert(delay >= 0);
        return std::max(static_cast<SizeType>(1), static_cast<SizeType>(ceil(delay * frequency)));
    }

    void processBatch(const CycleContext&);

    template<typename ElementType, typename... BufferTypes>
    void processBatch(const CycleContext& cycleContext, BatchedRingBuffer<ElementType>& buffer, BufferTypes&... buffers) {

        for (auto cit = buffer.cBeginElementsAtCurrentLocation(); cit != buffer.cEndElementsAtCurrentLocation(); ++ cit) {
            cit->process(cycleContext);
            ++ numEventsProcessed;
        }

        processBatch(cycleContext, buffers...);
    }

    template<typename T, typename... Args>
    void pushBufferedEvent(TimeType delay, BatchedRingBuffer<T>& buffer, Args&&... args) noexcept {
        buffer.emplaceAtOffset(getTargetOffset(delay), std::forward<Args>(args)...);
    }

    struct CommonEventWithTargetTime {
        CommonEventWithTargetTime(TimeType targetTime, std::unique_ptr<CommonEvent>&& commonEvent);

        bool operator>(const CommonEventWithTargetTime& other) const;

        TimeType targetTime;
        std::unique_ptr<CommonEvent> commonEvent;
    };

    BatchedRingBuffer<TransmissionEvent> transmissionEventBuffer;
    std::vector<FiringThresholdEvalEvent> firingThresholdEvalBuffer;
    using QueueType = std::priority_queue<CommonEventWithTargetTime, std::vector<CommonEventWithTargetTime>, std::greater<CommonEventWithTargetTime>>;
    QueueType commonEventsQueue;
    ValueType frequency;
    uint64_t numEventsProcessed;
    SynapticTransmissionStats& synapticTransmissionStats;
};


}
