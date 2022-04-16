#include "EnvEventQueue.hpp"
#include "EnvEvent.hpp"

namespace soft_npu {

bool EnvEventPtrGreater::operator()(const std::unique_ptr<EnvEvent>& lhs, const std::unique_ptr<EnvEvent>& rhs) const noexcept {
    return lhs->scheduleTime > rhs->scheduleTime;
}

}
