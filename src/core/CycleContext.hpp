#pragma once

#include <Aliases.hpp>

namespace soft_npu {

struct StaticContext;

struct CycleContext {

    CycleContext(
        TimeType time,
        const StaticContext& staticContext,
        SizeType cycleId
        ) noexcept :
            time(time),
            staticContext(staticContext),
            cycleId(cycleId) {

    }

    CycleContext(const CycleContext& rhs) = delete;

    const TimeType time;
    const StaticContext& staticContext;
    SizeType cycleId;
};

}

