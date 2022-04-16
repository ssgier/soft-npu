#pragma once

#include <queue>
#include <memory>
#include <vector>

namespace soft_npu {

struct EnvEvent;

struct EnvEventPtrGreater {
    bool operator()(const std::unique_ptr<EnvEvent>& lhs, const std::unique_ptr<EnvEvent>& rhs) const noexcept;
};

using EnvEventQueue = std::priority_queue<
        std::unique_ptr<EnvEvent>,
                std::vector<std::unique_ptr<EnvEvent>>,
                EnvEventPtrGreater>;

}
