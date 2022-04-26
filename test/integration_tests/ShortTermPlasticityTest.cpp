#include <gtest/gtest.h>
#include <Aliases.hpp>
#include <core/StaticInputSimulation.hpp>
#include <TestUtil.hpp>

using namespace soft_npu;

auto makeStpTestTemplateParams() {
    auto params = getTemplateParams();

    auto populationJson = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "neuronParamsName": "excitatory"
        },
        {
            "neuronId": 1,
            "neuronParamsName": "excitatory"
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 1.1,
            "conductionDelay": 10e-3
        }
    ]
}
)"_json;

    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = populationJson;

    return params;
}

auto getParamsDepressionScenario() {
    auto stpParamsJson = R"(
{
    "isDepression": true,
    "restingValue": 0.8,
    "changeParameter":  0.5,
    "timeConstant": 100e-3
}
)"_json;

    auto params = makeStpTestTemplateParams();
    (*params)["synapseParams"]["shortTermPlasticityParams"] = stpParamsJson;
    return params;
}

auto getParamsFacilitationScenario() {
    auto stpParamsJson = R"(
{
    "isDepression": false,
    "restingValue": 0.6,
    "changeParameter":  0.5,
    "timeConstant": 100e-3
}
)"_json;

    auto params = makeStpTestTemplateParams();
    (*params)["synapseParams"]["shortTermPlasticityParams"] = stpParamsJson;
    return params;
}

TEST (ShortTermPlasticityTest, DepressionPreventsSpike) {
    StaticInputSimulation simulation(getParamsDepressionScenario());

    TimeType timeBetweenPreSynSpikes = 11e-3;

    simulation.setSpikeTrains({
        {10e-3, 0},
        {10e-3 + timeBetweenPreSynSpikes, 0}
    });

    simulation.recordVoltage(1, 20e-3 + timeBetweenPreSynSpikes);
    auto simulationResult = simulation.run();
    ASSERT_EQ(simulationResult.recordedSpikes.size(), 2);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage,
            1.1 * 0.8 * exp(- timeBetweenPreSynSpikes / 20e-3) + 1.1 * 0.8 * (1 - 0.5 * exp(- timeBetweenPreSynSpikes / 100e-3)));
}

TEST (ShortTermPlasticityTest, SpikeDespikeDepression) {
    StaticInputSimulation simulation(getParamsDepressionScenario());

    TimeType timeBetweenPreSynSpikes = 10.5e-3;

    simulation.setSpikeTrains({
        {10e-3, 0},
        {10e-3 + timeBetweenPreSynSpikes, 0}
    });

    simulation.recordVoltage(1, 20e-3 + timeBetweenPreSynSpikes);
    auto simulationResult = simulation.run();
    ASSERT_EQ(simulationResult.recordedSpikes.size(), 3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 20e-3 + timeBetweenPreSynSpikes);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0);
}

TEST (ShortTermPlasticityTest, NoSpikeDespiteFacilitation) {
    StaticInputSimulation simulation(getParamsFacilitationScenario());

    TimeType timeBetweenPreSynSpikes = 27.0e-3;

    simulation.setSpikeTrains({
        {10e-3, 0},
        {10e-3 + timeBetweenPreSynSpikes, 0}
    });

    simulation.recordVoltage(1, 20e-3 + timeBetweenPreSynSpikes);
    auto simulationResult = simulation.run();
    ASSERT_EQ(simulationResult.recordedSpikes.size(), 2);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage,
            1.1 * 0.6 * exp(- timeBetweenPreSynSpikes / 20e-3) + (1.1 * (0.6 + 0.5 * (1 - 0.6) * exp(- timeBetweenPreSynSpikes / 100e-3))));
}

TEST (ShortTermPlasticityTest, FacilitationCausesSpike) {
    StaticInputSimulation simulation(getParamsFacilitationScenario());

    TimeType timeBetweenPreSynSpikes = 26.5e-3;

    simulation.setSpikeTrains({
        {10e-3, 0},
        {10e-3 + timeBetweenPreSynSpikes, 0}
    });

    simulation.recordVoltage(1, 20e-3 + timeBetweenPreSynSpikes);
    auto simulationResult = simulation.run();
    ASSERT_EQ(simulationResult.recordedSpikes.size(), 3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 20e-3 + timeBetweenPreSynSpikes);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0);
}
