#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct EnvContext;

struct EnvEvent {

    template<typename F>
    EnvEvent(TimeType scheduleTime, F&& task) :
    scheduleTime(scheduleTime),
    task(std::forward<F>(task))
    {}

    TimeType scheduleTime;
    std::function<void(const EnvContext&, TimeType)> task;
};

}
