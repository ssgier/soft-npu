#pragma once

namespace soft_npu {
class InterruptSignalChecker {
public:
    static bool wasSent();
private:
    static void firstInterruptHandler(int);
    static void secondInterruptHandler(int);
    struct State {
        State();
        bool interruptSignalReceived;
    };

    static State state;
};
}
