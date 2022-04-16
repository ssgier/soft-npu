#pragma once

#include "CycleContext.hpp"

namespace soft_npu {

class CommonEvent {
public:
    virtual ~CommonEvent() = default;
    virtual void process(const CycleContext&) = 0;
};

template<typename T>
class CommonEventImpl : public CommonEvent {
public:
    explicit CommonEventImpl(T&& processingFunction) : processingFunction(std::forward<T>(processingFunction)) {}

    void process(const CycleContext& cycleContext) override {
        processingFunction(cycleContext);
    }

private:
    const T processingFunction;
};

template<typename T>
std::unique_ptr<CommonEvent> makeCommonEvent(T&& processingFunction) {
    return std::make_unique<CommonEventImpl<T> >(std::forward<T>(processingFunction));
}

}




