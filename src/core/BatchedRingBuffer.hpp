#pragma once

#include <vector>
#include <algorithm>
#include <Aliases.hpp>
#include <stdexcept>
#include <cassert>

namespace soft_npu {

template<typename T>
class BatchedRingBuffer {

public:

    using const_iterator = typename std::vector<T>::const_iterator;

    BatchedRingBuffer(SizeType bufferSize, SizeType subBufferInitialCapacity):
        buffer(bufferSize), currentPosition(0) {
        std::for_each(buffer.begin(), buffer.end(), [subBufferInitialCapacity](auto &subBuffer) {
            subBuffer.reserve(subBufferInitialCapacity);
        });
    }

    BatchedRingBuffer(const BatchedRingBuffer<T>& other) = delete;

    bool isOffsetWithinHorizon(SizeType offset) const noexcept {
        return offset < buffer.size();
    }

    template<typename... Args>
    void emplaceAtOffset(SizeType offset, Args&&... args) noexcept {

        assert(offset >= 0 && isOffsetWithinHorizon(offset));

        buffer[getTargetPosition(offset)].emplace_back(std::forward<Args>(args)...);
    }

    const_iterator cBeginElementsAtCurrentLocation() const noexcept {
        return buffer[currentPosition].cbegin();
    }

    const_iterator cEndElementsAtCurrentLocation() const noexcept {
        return buffer[currentPosition].cend();
    }

    void clearAndAdvance() noexcept {

        buffer[currentPosition].clear();

        ++ currentPosition;

        if (currentPosition == buffer.size()) {
            currentPosition -= buffer.size();
        }
    }

private:
    std::vector<std::vector<T>> buffer;
    SizeType currentPosition;

    SizeType getTargetPosition(SizeType offset) {
        SizeType targetPosition = currentPosition + offset;

        if (targetPosition >= buffer.size()) {
            targetPosition -= buffer.size();
        }

        return targetPosition;
    }
};

}




