#pragma once

#include <Aliases.hpp>
#include "EventProcessor.hpp"
#include "CycleContext.hpp"

namespace soft_npu {

template<typename T>
void pushAsCommonEvent(EventProcessor& eventProcessor, TimeType targetTime, T&& processingFunction) {
    eventProcessor.pushCommonEvent(targetTime, makeCommonEvent(std::forward<T>(processingFunction)));
}

template<typename T>
void pushAsRecurringCommonEvent(EventProcessor& eventProcessor, TimeType firstTime, TimeType interval, T&& processingFunction) {
    pushAsCommonEvent(eventProcessor, firstTime, [interval, processingFunction, &eventProcessor](const CycleContext& cycleContext) {
        processingFunction(cycleContext);
        pushAsRecurringCommonEvent(eventProcessor, cycleContext.time + interval, interval, processingFunction);
    });
}

}
