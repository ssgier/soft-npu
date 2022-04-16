#include <gtest/gtest.h>
#include <core/StaticInputSimulation.hpp>
#include <vector>
#include <params/ParamsFactories.hpp>
#include <TestUtil.hpp>

using namespace soft_npu;

TEST(STDPIntegrationTests, SimplePotentiation) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;
    (*params)["synapseParams"]["maxWeight"] = 1.5;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 1.0,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {11e-3, 0}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 2);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 12e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 1);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 2);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 2);

    ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].postSynapticNeuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.1);
}

TEST(STDPIntegrationTests, PostSynapticSpikeFollowedByTwoPreSynapticSpikes) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;
    (*params)["synapseParams"]["maxWeight"] = 1.5;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.5,
            "conductionDelay": 15e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {11e-3, 1},
                                      {12e-3, 0},
                                      {23e-3, 0},
                              });

    simulation.recordVoltage(1, 27e-3);
    simulation.recordVoltage(1, 38e-3);

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 12e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 23e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 0);

    ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].postSynapticNeuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.5 - 0.12 * exp(-16e-3 / 20e-3) - 0.12 * exp(-27e-3 / 20e-3));

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0.5);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[1].voltage, 0.5 * exp(-11e-3 / 20e-3) + 0.5);
}

TEST(STDPIntegrationTests, PotentiationTwoEPSPs) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;
    (*params)["synapseParams"]["maxWeight"] = 1.5;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        },
        {
            "neuronId": 2,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 2,
            "initialWeight": 0.8,
            "conductionDelay": 1e-3
        },
        {
            "preSynapticNeuronId": 1,
            "postSynapticNeuronId": 2,
            "initialWeight": 0.8,
            "conductionDelay": 2e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {11e-3, 0},
                                      {12e-3, 1}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 12e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 14e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 2);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 3);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 4);

    auto finalSynapseInfos = simulationResult.finalSynapseInfos;
    std::sort(finalSynapseInfos.begin(), finalSynapseInfos.end());

    ASSERT_EQ(finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(finalSynapseInfos[0].postSynapticNeuronId, 2);
    ASSERT_FLOAT_EQ(finalSynapseInfos[0].weight, 0.8 + 0.1 * exp(-2e-3 / 20e-3));

    ASSERT_EQ(finalSynapseInfos[1].preSynapticNeuronId, 1);
    ASSERT_EQ(finalSynapseInfos[1].postSynapticNeuronId, 2);
    ASSERT_FLOAT_EQ(finalSynapseInfos[1].weight, 0.8 + 0.1);
}

TEST(STDPIntegrationTests, PotentiationTwoEPSPOneWouldBeEnough) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;
    (*params)["synapseParams"]["maxWeight"] = 1.5;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        },
        {
            "neuronId": 2,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 2,
            "initialWeight": 1.0,
            "conductionDelay": 1e-3
        },
        {
            "preSynapticNeuronId": 1,
            "postSynapticNeuronId": 2,
            "initialWeight": 1.0,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {11e-3, 0},
                                      {11e-3, 1}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 12e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 2);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 3);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 4);

    auto finalSynapseInfos = simulationResult.finalSynapseInfos;
    std::sort(finalSynapseInfos.begin(), finalSynapseInfos.end());

    ASSERT_EQ(finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(finalSynapseInfos[0].postSynapticNeuronId, 2);
    ASSERT_FLOAT_EQ(finalSynapseInfos[0].weight, 1.0 + 0.1);

    ASSERT_EQ(finalSynapseInfos[1].preSynapticNeuronId, 1);
    ASSERT_EQ(finalSynapseInfos[1].postSynapticNeuronId, 2);
    ASSERT_FLOAT_EQ(finalSynapseInfos[1].weight, 1.0 + 0.1);
}

TEST(STDPIntegrationTests, PotentiationUpperWeightBoundary) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 0.8;
    (*params)["synapseParams"]["maxWeight"] = 1.0;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.99,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {10e-3, 1},
                                      {11e-3, 0},
                                      {11.1e-3, 0},
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].postSynapticNeuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0);
}

TEST(STDPIntegrationTests, SimpleDepression) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["synapseParams"]["maxWeight"] = 1.0;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.5,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {9e-3, 1},
                                      {11e-3, 0}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 2);

    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 9e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 1);

    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 0);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 2);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 3);

    ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].postSynapticNeuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.5 - 0.12 * exp(-3e-3 / 20e-3));
}

TEST(STDPIntegrationTests, DepressionLowerWeightBound) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["synapseParams"]["maxWeight"] = 1.0;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.05,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {9e-3, 1},
                                      {11e-3, 0}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(simulationResult.finalSynapseInfos[0].postSynapticNeuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.0);
}

TEST(STDPIntegrationTests, NoSTDPInhibitorySynapse) {
auto params = getTemplateParams();
(*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;

auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": true
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.1,
            "conductionDelay": 1e-3
        }
    ]
}
)";

(*params)["simulation"]["populationGenerator"] = "pDetailedParams";
(*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

StaticInputSimulation simulation(params);

simulation.setSpikeTrains({
{11e-3, 0},
{13e-3, 1},
});

simulation.recordVoltage(1, 12e-3);

auto simulationResult = simulation.run();

ASSERT_EQ(simulationResult.recordedSpikes.size(), 2);
ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11e-3);
ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 13e-3);
ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 1);

ASSERT_EQ(simulationResult.voltageRecordings[0].neuronId, 1);
ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, -0.1);
ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].time, 12e-3);

ASSERT_EQ(simulationResult.numExcitatorySpikes, 1);
ASSERT_EQ(simulationResult.numInhibitorySpikes, 1);
ASSERT_FLOAT_EQ(simulationResult.meanExcitatoryFiringRate, 1);
ASSERT_FLOAT_EQ(simulationResult.meanInhibitoryFiringRate, 1);
ASSERT_EQ(simulationResult.numEventsProcessed, 4);

ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
ASSERT_EQ(simulationResult.finalSynapseInfos[0].preSynapticNeuronId, 0);
ASSERT_EQ(simulationResult.finalSynapseInfos[0].postSynapticNeuronId, 1);
ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.1);
}

TEST(STDPIntegrationTests, STDPJustBeforeCutOffTime) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["synapseParams"]["stdpCutOffTime"] = 5e-3;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.1,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {10e-3, 0},
                                      {15.9e-3, 1},
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.numEventsProcessed, 3);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.1 + 0.1 * exp(- 4.9e-3 / 20e-3));
}

TEST(STDPIntegrationTests, NoSTDPAfterCutOffTime) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["synapseParams"]["stdpCutOffTime"] = 5e-3;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.1,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {10e-3, 0},
                                      {16.1e-3, 1},
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.numEventsProcessed, 3);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.1);
}

TEST(STDPIntegrationTests, STDPComplexScenario) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;
    (*params)["synapseParams"]["maxWeight"] = 1.5;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        },
        {
            "neuronId": 2,
            "isInhibitory": false
        },
        {
            "neuronId": 3,
            "isInhibitory": false
        },
        {
            "neuronId": 4,
            "isInhibitory": true
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 4,
            "initialWeight": 0.1,
            "conductionDelay": 1e-3
        },
        {
            "preSynapticNeuronId": 1,
            "postSynapticNeuronId": 4,
            "initialWeight": 0.99,
            "conductionDelay": 1e-3
        },
        {
            "preSynapticNeuronId": 2,
            "postSynapticNeuronId": 4,
            "initialWeight": 0.5,
            "conductionDelay": 1e-3
        },
        {
            "preSynapticNeuronId": 3,
            "postSynapticNeuronId": 4,
            "initialWeight": 0.5,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {0.5e-3, 0},
                                      {11e-3, 0},
                                      {11e-3, 1},
                                      {11e-3, 2},
                                      {20e-3, 4},
                                      {22e-3, 3},
                                      {35e-3, 3}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 6);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 2);
    ASSERT_EQ(simulationResult.numEventsProcessed, 13);

    auto finalSynapseInfos = simulationResult.finalSynapseInfos;
    std::sort(finalSynapseInfos.begin(), finalSynapseInfos.end());

    ASSERT_EQ(finalSynapseInfos[0].preSynapticNeuronId, 0);
    ASSERT_EQ(finalSynapseInfos[0].postSynapticNeuronId, 4);
    ASSERT_FLOAT_EQ(finalSynapseInfos[0].weight, 0.1 + 0.1 + 0.1 * exp(-10.5e-3 / 20e-3));

    ASSERT_EQ(finalSynapseInfos[1].preSynapticNeuronId, 1);
    ASSERT_EQ(finalSynapseInfos[1].postSynapticNeuronId, 4);
    ASSERT_FLOAT_EQ(finalSynapseInfos[1].weight, 0.99 + 0.1);

    ASSERT_EQ(finalSynapseInfos[2].preSynapticNeuronId, 2);
    ASSERT_EQ(finalSynapseInfos[2].postSynapticNeuronId, 4);
    ASSERT_FLOAT_EQ(finalSynapseInfos[2].weight, 0.5 + 0.1);

    ASSERT_EQ(finalSynapseInfos[3].preSynapticNeuronId, 3);
    ASSERT_EQ(finalSynapseInfos[3].postSynapticNeuronId, 4);
    ASSERT_FLOAT_EQ(finalSynapseInfos[3].weight, 0.5 - 0.12 * exp(-3e-3 / 20e-3) - 0.12 * exp(-16e-3 / 20e-3));
}

TEST(STDPIntegrationTests, PotentiationFromZeroEfficacy) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.1;
    (*params)["synapseParams"]["maxWeight"] = 1.5;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 1,
            "initialWeight": 0.0,
            "conductionDelay": 1e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                      {11e-3, 0},
                                      {15e-3, 1}
                              });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.finalSynapseInfos.size(), 1);
    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 0.1 * exp(- 3e-3/20e-3));
}
