#pragma once
#include <atomic>

namespace soft_npu {
class InterruptSignalChecker {
public:
    static bool wasSent();
private:
    static void firstInterruptHandler(int);
    static void secondInterruptHandler(int);
    struct State {
        State();
        std::atomic<bool> interruptSignalReceived;
        static_assert(std::atomic<bool>::is_always_lock_free);
    };

    static State state;
};
}
