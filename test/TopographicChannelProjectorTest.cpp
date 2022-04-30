#include <gtest/gtest.h>
#include <genesis/TrivialNeuroComponentsFactory.hpp>
#include "TestUtil.hpp"
#include <neuro/Population.hpp>
#include <params/ParamsFactories.hpp>
#include <neuro/ChannelProjectorFactory.hpp>
#include <neuro/TopographicChannelProjector.hpp>

using namespace soft_npu;

TEST(TopographicChannelProjectorTest, TwoInputChannelsWithIntersection) {

    auto params = getTemplateParams();

    TrivialNeuroComponentsFactory factory;
    auto population = std::make_shared<Population>();

    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(*params);
    auto inhibitoryNeuronParams = ParamsFactories::extractInhibitoryNeuronParams(*params);

    constexpr SizeType numNeurons = 5;

    std::array<Population::Location, numNeurons> locations = {{
        {0.1, 0.11},
        {0.1, 0.125},
        {0.1, 0.14},
        {0.129, 0.125},
        {0.1, 0.125},
    }};

    for (SizeType neuronId = 0; neuronId < numNeurons; ++ neuronId) {

        auto neuronParams = neuronId == 4 ? inhibitoryNeuronParams : excitatoryNeuronParams;
        population->addNeuron(factory.makeNeuron(neuronId, neuronParams), locations[neuronId]);
    }

    TopographicChannelProjector channelProjector(*params, *population);

    auto channel0Result = channelProjector.getEPSPsWithTargetNeurons(0);
    ASSERT_TRUE(channel0Result.size() == 2);
    ASSERT_EQ(channel0Result[0].second->getNeuronId(), 0);
    ASSERT_EQ(channel0Result[1].second->getNeuronId(), 1);

    auto channel1Result = channelProjector.getEPSPsWithTargetNeurons(1);
    ASSERT_TRUE(channel1Result.size() == 2);
    ASSERT_EQ(channel1Result[0].second->getNeuronId(), 1);
    ASSERT_EQ(channel1Result[1].second->getNeuronId(), 2);

    auto channel2Result = channelProjector.getEPSPsWithTargetNeurons(2);
    ASSERT_TRUE(channel2Result.empty());
}

TEST(TopographicChannelProjectorTest, TwoOutputChannels) {
    auto params = getTemplateParams();

    TrivialNeuroComponentsFactory factory;
    auto population = std::make_shared<Population>();

    auto neuronParams = ParamsFactories::extractExcitatoryNeuronParams(*params);

    constexpr SizeType numNeurons = 3;

    std::array<Population::Location, numNeurons> locations = {{
                                                                      {0.5, 0.53},
                                                                      {0.5, 0.56},
                                                                      {0.5, 0.71},
                                                              }};

    for (SizeType neuronId = 0; neuronId < numNeurons; ++ neuronId) {
        population->addNeuron(factory.makeNeuron(neuronId, neuronParams), locations[neuronId]);
    }

    TopographicChannelProjector channelProjector(*params, *population);

    auto motorNeuronIds = channelProjector.getMotorNeuronIds();
    ASSERT_EQ(motorNeuronIds.size(), 2);
    ASSERT_TRUE(motorNeuronIds.find(0) != motorNeuronIds.end());
    ASSERT_TRUE(motorNeuronIds.find(1) != motorNeuronIds.end());

    CycleOutputBuffer cycleOutputBuffer;
    channelProjector.projectNeuronSpike(cycleOutputBuffer, population->getNeuronById(0));
    ASSERT_EQ(std::distance(cycleOutputBuffer.cbeginSpikingChannelIds(), cycleOutputBuffer.cendSpikingChannelIds()), 1);
    ASSERT_EQ(*cycleOutputBuffer.cbeginSpikingChannelIds(), 0);

    cycleOutputBuffer.reset();
    channelProjector.projectNeuronSpike(cycleOutputBuffer, population->getNeuronById(1));
    ASSERT_EQ(std::distance(cycleOutputBuffer.cbeginSpikingChannelIds(), cycleOutputBuffer.cendSpikingChannelIds()), 1);
    ASSERT_EQ(*cycleOutputBuffer.cbeginSpikingChannelIds(), 1);

    cycleOutputBuffer.reset();
    channelProjector.projectNeuronSpike(cycleOutputBuffer, population->getNeuronById(2));
    ASSERT_EQ(cycleOutputBuffer.cbeginSpikingChannelIds(), cycleOutputBuffer.cendSpikingChannelIds());

}
