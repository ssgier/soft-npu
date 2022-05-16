#include "InterruptSignalChecker.hpp"
#include <csignal>
#include <plog/Log.h>

namespace soft_npu {

InterruptSignalChecker::State InterruptSignalChecker::state;

bool InterruptSignalChecker::wasSent() {
    return state.interruptSignalReceived;
}

void InterruptSignalChecker::secondInterruptHandler(int sig) {
    if (sig == SIGINT) {
        exit(-1);
    }    
}

void InterruptSignalChecker::firstInterruptHandler(int sig) {
    if (sig == SIGINT) {
        PLOG_INFO << "Interrupt signal received";
        state.interruptSignalReceived = true;
        signal(SIGINT, InterruptSignalChecker::secondInterruptHandler);
    }    
}

InterruptSignalChecker::State::State(): interruptSignalReceived(false) {
    signal(SIGINT, InterruptSignalChecker::firstInterruptHandler);
}

}
