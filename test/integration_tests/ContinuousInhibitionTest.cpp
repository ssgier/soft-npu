#include <gtest/gtest.h>
#include <Aliases.hpp>
#include <core/StaticInputSimulation.hpp>
#include <TestUtil.hpp>

using namespace soft_npu;

auto makeContinuousInhibitionTestTemplateParams() {
    auto params = getTemplateParams();

    auto populationJson = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "neuronParamsName": "excitatory",
            "continuousInhibitionSourceNeuronIds": [1]
        },
        {
            "neuronId": 1,
            "neuronParamsName": "continuousInhibitionSource"
        }
    ],
    "synapses": []
}
)"_json;

    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 0.5;
    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = populationJson;

    return params;
}

TEST (ContinuousInhibitionTest, SourceSpikeBeforeLastTime) {

    StaticInputSimulation simulation(makeContinuousInhibitionTestTemplateParams());

    simulation.setSpikeTrains({
        {1e-3, 1},
        {1e-3, 1},
        {2e-3, 0}
    });
 
    simulation.recordVoltage(0, 3e-3);
    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0.5 * exp(-1e-3 / 20e-3));
}

TEST (ContinuousInhibitionTest, SourceSpikeAfterLastTime) {

    StaticInputSimulation simulation(makeContinuousInhibitionTestTemplateParams());

    simulation.setSpikeTrains({
        {1e-3, 0},
        {2e-3, 1},
        {2e-3, 1}
    });
 
    simulation.recordVoltage(0, 3e-3);
    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0);
}

TEST (ContinuousInhibitionTest, ContinuousInhibitionPreventsSpike) {
    StaticInputSimulation simulation(makeContinuousInhibitionTestTemplateParams());

    simulation.setSpikeTrains({
        {2e-3, 1},
        {10e-3, 0},
        {10e-3, 0}
    });
 
    auto simulationResult = simulation.run();

    ASSERT_TRUE(simulationResult.recordedSpikes.empty());
}

TEST (ContinuousInhibitionTest, SpikeDespiteContinuousInhibition) {
    StaticInputSimulation simulation(makeContinuousInhibitionTestTemplateParams());

    simulation.setSpikeTrains({
        {9e-3, 1},
        {10e-3, 0},
        {10e-3, 0},
        {10e-3, 0}
    });
 
    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 1);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 10e-3);
}
