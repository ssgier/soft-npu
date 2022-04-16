#include <Aliases.hpp>
#include <core/BatchedRingBuffer.hpp>
#include "gtest/gtest.h"
#include <array>
#include <vector>
#include <random>

using namespace soft_npu;

struct Element {
    explicit Element(int value) : value(value), allowCopyAndMoveConstruction(false) {}
    explicit Element(int value, bool allowCopyAndMoveConstruction) :
            value(value), allowCopyAndMoveConstruction(allowCopyAndMoveConstruction){}

    Element(const Element& rhs) : value(rhs.value), allowCopyAndMoveConstruction(rhs.allowCopyAndMoveConstruction) {
        if (!allowCopyAndMoveConstruction) {
            throw std::runtime_error("Must not be invoked");
        }
    }

    Element(Element&& rhs) : value(rhs.value), allowCopyAndMoveConstruction(rhs.allowCopyAndMoveConstruction) {
        if (!allowCopyAndMoveConstruction) {
            throw std::runtime_error("Must not be invoked");
        }
    }

    Element& operator=(Element&& rhs) = delete;
    Element& operator=(const Element& rhs) = delete;

    bool operator==(const Element &rhs) const {
        return value == rhs.value && allowCopyAndMoveConstruction == rhs.allowCopyAndMoveConstruction;
    }

    bool operator!=(const Element &rhs) const {
        return !(rhs == *this);
    }

    int value;
    bool allowCopyAndMoveConstruction;
};

class BatchedRingBufferTest : public ::testing::Test {
public:
    BatchedRingBufferTest(): batchedRingBuffer(10, 10) {}
protected:
    BatchedRingBuffer<Element> batchedRingBuffer;
};

TEST_F(BatchedRingBufferTest, Empty) {
    ASSERT_EQ(batchedRingBuffer.cBeginElementsAtCurrentLocation(), batchedRingBuffer.cEndElementsAtCurrentLocation());
}

TEST_F(BatchedRingBufferTest, SingleElement) {
    int value = 2;

    batchedRingBuffer.emplaceAtOffset(1, value);
    batchedRingBuffer.clearAndAdvance();

    ASSERT_EQ(std::distance(batchedRingBuffer.cBeginElementsAtCurrentLocation(),
                            batchedRingBuffer.cEndElementsAtCurrentLocation()), 1);

    ASSERT_EQ(batchedRingBuffer.cBeginElementsAtCurrentLocation()->value, value);
    batchedRingBuffer.clearAndAdvance();
    ASSERT_EQ(batchedRingBuffer.cBeginElementsAtCurrentLocation(), batchedRingBuffer.cEndElementsAtCurrentLocation());
}

TEST_F(BatchedRingBufferTest, RoundTrip) {
    int firstPassValue = 2;
    int secondPassValue = 3;

    batchedRingBuffer.emplaceAtOffset(1, firstPassValue);

    for (auto i = 0; i < 2; ++i) {
        batchedRingBuffer.clearAndAdvance();
    }

    batchedRingBuffer.emplaceAtOffset(9, secondPassValue);

    for (auto i = 0; i < 9; ++i) {
        batchedRingBuffer.clearAndAdvance();
    }

    ASSERT_EQ(std::distance(batchedRingBuffer.cBeginElementsAtCurrentLocation(),
                            batchedRingBuffer.cEndElementsAtCurrentLocation()), 1);

    ASSERT_EQ(batchedRingBuffer.cBeginElementsAtCurrentLocation()->value, secondPassValue);
}

TEST_F(BatchedRingBufferTest, RandomizedInput) {
    constexpr auto numTimeSlots = 101;

    std::array<std::vector<Element>, numTimeSlots> flatExpectedData;
    std::for_each(flatExpectedData.begin(), flatExpectedData.end(), [](auto &subBuffer) {
        subBuffer.reserve(10);
    });

    std::default_random_engine generator(0);
    std::uniform_int_distribution<SizeType> amountDistribution(0, 10);
    std::uniform_int_distribution<SizeType> offsetDistribution(1, 9);
    std::uniform_int_distribution<int> valueDistribution(-1000, 1000);

    for (auto flatLocation = 0; flatLocation < numTimeSlots; ++flatLocation) {
        auto amount = amountDistribution(generator);

        for (SizeType i = 0; i < amount; ++i) {
            auto offset = offsetDistribution(generator);
            auto value = valueDistribution(generator);

            batchedRingBuffer.emplaceAtOffset(offset, value, true);

            auto targetLocationFlatData = flatLocation + offset;
            if (targetLocationFlatData < numTimeSlots) {
                flatExpectedData[targetLocationFlatData].emplace_back(value, true);
            }
        }

        ASSERT_TRUE(std::equal(
                batchedRingBuffer.cBeginElementsAtCurrentLocation(),
                batchedRingBuffer.cEndElementsAtCurrentLocation(),
                flatExpectedData[flatLocation].cbegin(),
                flatExpectedData[flatLocation].cend()));

        batchedRingBuffer.clearAndAdvance();
    }
}