#include <gtest/gtest.h>
#include <Aliases.hpp>
#include <neuro/Population.hpp>

using namespace soft_npu;

TEST(PopulationTest, ZeroDistance) {
    Population::Location location = {0.5, 0.5};

    auto distance = PopulationUtils::getDistance(location, location);

    ASSERT_FLOAT_EQ(distance, 0);

    ASSERT_FALSE(PopulationUtils::isDistanceShorterThan(location, location, 0));
    ASSERT_TRUE(PopulationUtils::isDistanceShorterThan(location, location, 0.1));
}

TEST(PopulationTest, PointsNearCenter) {
    Population::Location location0 = {0.5, 0.5};
    Population::Location location1 = {0.4, 0.6};

    auto distance = PopulationUtils::getDistance(location0, location1);
    auto expectedDistance = std::sqrt(0.1 * 0.1 + 0.1 * 0.1);

    ASSERT_FLOAT_EQ(distance, expectedDistance);
    ASSERT_FALSE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance - 0.0001));
    ASSERT_TRUE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance + 0.0001));
}

TEST(PopulationTest, PointsAcrossXBorder) {
    Population::Location location0 = {0.9, 0.5};
    Population::Location location1 = {0.1, 0.6};

    auto distance = PopulationUtils::getDistance(location0, location1);
    auto expectedDistance = std::sqrt(0.2 * 0.2 + 0.1 * 0.1);

    ASSERT_FLOAT_EQ(distance, expectedDistance);
    ASSERT_FALSE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance - 0.0001));
    ASSERT_TRUE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance + 0.0001));
}

TEST(PopulationTest, PointsAcrossYBorder) {
    Population::Location location0 = {0.5, 0.1};
    Population::Location location1 = {0.4, 0.9};

    auto distance = PopulationUtils::getDistance(location0, location1);
    auto expectedDistance = std::sqrt(0.2 * 0.2 + 0.1 * 0.1);

    ASSERT_FLOAT_EQ(distance, expectedDistance);
    ASSERT_FALSE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance - 0.0001));
    ASSERT_TRUE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance + 0.0001));
}

TEST(PopulationTest, PointsAcrossCorner) {
    Population::Location location0 = {0.1, 0.9};
    Population::Location location1 = {0.9, 0.1};

    auto distance = PopulationUtils::getDistance(location0, location1);
    auto expectedDistance = std::sqrt(0.2 * 0.2 + 0.2 * 0.2);

    ASSERT_FLOAT_EQ(distance, expectedDistance);
    ASSERT_FALSE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance - 0.0001));
    ASSERT_TRUE(PopulationUtils::isDistanceShorterThan(location0, location1, expectedDistance + 0.0001));
}