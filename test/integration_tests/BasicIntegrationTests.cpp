#include <gtest/gtest.h>
#include <core/StaticInputSimulation.hpp>
#include <vector>
#include <cmath>
#include <TestUtil.hpp>

using namespace soft_npu;

TEST(BasicIntegrationTests, NoActivitySimulation) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    auto simulationResult = simulation.run();
    ASSERT_FLOAT_EQ(simulationResult.simulationTime, 1.0);
    ASSERT_TRUE(simulationResult.recordedSpikes.empty());
    ASSERT_EQ(simulationResult.numExcitatorySpikes, 0);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_FLOAT_EQ(simulationResult.meanExcitatoryFiringRate, 0);
    ASSERT_FLOAT_EQ(simulationResult.meanInhibitoryFiringRate, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 0);
    ASSERT_DOUBLE_EQ(simulationResult.eventThroughput, 0);
}

TEST(BasicIntegrationTests, SimpleExcitatorySpike) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {11e-3, 0},
        {12e-3, 0},
        {13e-3, 0}
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 13e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 1);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_FLOAT_EQ(simulationResult.meanExcitatoryFiringRate, 1);
    ASSERT_FLOAT_EQ(simulationResult.meanInhibitoryFiringRate, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 3);
}


TEST(BasicIntegrationTests, LeakyIntegrationMissedSpike) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
            {11e-3, 0},
            {12e-3, 0},
            {17.3e-3, 0}
    });

    auto simulationResult = simulation.run();

    ASSERT_TRUE(simulationResult.recordedSpikes.empty());
}

TEST(BasicIntegrationTests, LeakyIntegrationInTimeForSpike) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {11e-3, 0},
        {12e-3, 0},
        {17.2e-3, 0}
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 17.2e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
}

TEST(BasicIntegrationTests, LeakyIntegrationVoltageTrajectory) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {11e-3, 0},
    });

    simulation.recordVoltage(0, 11e-3);
    simulation.recordVoltage(0, 15e-3);

    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0.4);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[1].voltage, 0.4 * exp(- 4e-3 / 20e-3));
}

TEST(BasicIntegrationTests, LeakyIntegrationVoltageTrajectoryInhibitoryNeuron) {
    auto params = getTemplateParams();

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": true
        }
    ],
    "synapses": []
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                                            {11e-3, 0},
                                                    });

    simulation.recordVoltage(0, 11e-3);
    simulation.recordVoltage(0, 15e-3);

    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0.4);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[1].voltage, 0.4 * exp(- 4e-3 / 5e-3));
}

TEST(BasicIntegrationTests, NoEPSPDuringRefractoryPeriod) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                                            {11e-3, 0},
                                                            {11.1e-3, 0},
                                                            {11.2e-3, 0}, // spike at this point
                                                            {21.1e-3, 0}, // during refractory period
                                                            {21.4e-3, 0}, // after refractory period
                                                            {21.5e-3, 0}
                                                    });

    simulation.recordVoltage(0, 21.1e-3);
    simulation.recordVoltage(0, 21.4e-3);

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11.2e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0.0);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[1].voltage, 0.4);
}

TEST(BasicIntegrationTests, EPSPAfterRefractoryPeriod) {
    auto params = getTemplateParams();
    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
                                                            {11e-3, 0},
                                                            {11.1e-3, 0},
                                                            {11.2e-3, 0}, // spike at this point
                                                            {21.3e-3, 0}, // after refractory period
                                                            {21.4e-3, 0},
                                                            {21.5e-3, 0}
                                                    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 2);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11.2e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 21.5e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 0);
}

TEST(BasicIntegrationTests, EventProcessorDiscreteTimeSnapping) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {11.31e-3, 0}, // dt is 1e-4, so should snap to 11.4
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11.4e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
}

TEST(BasicIntegrationTests, SimpleSynapticTransmission) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": true
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

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 1);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 1);
    ASSERT_FLOAT_EQ(simulationResult.meanExcitatoryFiringRate, 1);
    ASSERT_FLOAT_EQ(simulationResult.meanInhibitoryFiringRate, 1);
    ASSERT_EQ(simulationResult.numEventsProcessed, 2);
}

TEST(BasicIntegrationTests, TwoCoincidingEPSPs) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;

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
            "initialWeight": 0.5,
            "conductionDelay": 10e-3
        },
        {
            "preSynapticNeuronId": 1,
            "postSynapticNeuronId": 2,
            "initialWeight": 0.5,
            "conductionDelay": 2e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {1e-3, 0},
        {9e-3, 1},
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 1e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 9e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 11e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 2);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 3);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 0);
    ASSERT_FLOAT_EQ(simulationResult.meanExcitatoryFiringRate, 1);
    ASSERT_FLOAT_EQ(simulationResult.meanInhibitoryFiringRate, 0);
    ASSERT_EQ(simulationResult.numEventsProcessed, 4);
}

TEST(BasicIntegrationTests, TwoEPSPsAndOneIPSP) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;

    auto populationJsonText = R"(
{
    "neurons": [
        {
            "neuronId": 0,
            "isInhibitory": false
        },
        {
            "neuronId": 1,
            "isInhibitory": true
        },
        {
            "neuronId": 2,
            "isInhibitory": false
        },
        {
            "neuronId": 3,
            "isInhibitory": false
        }
    ],
    "synapses": [
        {
            "preSynapticNeuronId": 0,
            "postSynapticNeuronId": 3,
            "initialWeight": 1.0,
            "conductionDelay": 10.1e-3
        },
        {
            "preSynapticNeuronId": 1,
            "postSynapticNeuronId": 3,
            "initialWeight": 0.5,
            "conductionDelay": 2e-3
        },
        {
            "preSynapticNeuronId": 2,
            "postSynapticNeuronId": 3,
            "initialWeight": 0.6,
            "conductionDelay": 2.3e-3
        }
    ]
}
)";

    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {1e-3, 0},
        {9e-3, 1},
        {9e-3, 2}
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 4);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 1e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 9e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[1].neuronId, 1);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 9e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[2].neuronId, 2);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[3].time, 11.3e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[3].neuronId, 3);

    ASSERT_EQ(simulationResult.numExcitatorySpikes, 3);
    ASSERT_EQ(simulationResult.numInhibitorySpikes, 1);
    ASSERT_FLOAT_EQ(simulationResult.meanExcitatoryFiringRate, 1);
    ASSERT_FLOAT_EQ(simulationResult.meanInhibitoryFiringRate, 1);
    ASSERT_EQ(simulationResult.numEventsProcessed, 6);
}

TEST(BasicIntegrationTests, VoltageFloor) {
    auto params = getTemplateParams();
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["neuronParams"]["inhibitory"]["refractoryPeriod"] = 0.5e-3;

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
            "initialWeight": 0.8,
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
        {11e-3, 0},
        {12e-3, 0},
    });

    simulation.recordVoltage(1, 10.9e-3);
    simulation.recordVoltage(1, 11e-3);
    simulation.recordVoltage(1, 11.9e-3);
    simulation.recordVoltage(1, 12e-3);
    simulation.recordVoltage(1, 13e-3);
    simulation.recordVoltage(1, 14e-3);

    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[0].voltage, 0);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[1].voltage, -0.8);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[2].voltage, -0.8 * exp(- 0.9e-3 / 20e-3));
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[3].voltage, -1);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[4].voltage, -1);
    ASSERT_FLOAT_EQ(simulationResult.voltageRecordings[5].voltage, -1 * exp(-  1e-3 / 20e-3));
}

TEST(BasicIntegrationTests, PoissonSpikeTrainsFuzzyTest) {
    auto params = getTemplateParams();

    (*params)["simulation"]["populationGenerator"] = "p1000";
    (*params)["simulation"]["untilTime"] = 1.0;
    (*params)["nonCoherentStimulator"]["rate"] = 8;
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["populationGenerators"]["p1000"]["inhibitorySynapseWeight"] = 0.0;
    (*params)["populationGenerators"]["p1000"]["excitatorySynapseInitialWeight"] = 0.0;
    (*params)["synapseParams"]["stdpScaleFactorPotentiation"] = 0.0;
    (*params)["neuronParams"]["excitatory"]["refractoryPeriod"] = 0.0;
    (*params)["neuronParams"]["inhibitory"]["refractoryPeriod"] = 0.0;

    StaticInputSimulation simulation(params);

    auto simulationResult = simulation.run();

    ASSERT_TRUE(std::abs(simulationResult.meanExcitatoryFiringRate - 8) / 8 < 0.1);
    ASSERT_TRUE(std::abs(simulationResult.meanInhibitoryFiringRate - 8) / 8 < 0.1);
}

TEST(BasicIntegrationTests, DifferentBufferSizes) {
    auto params = getTemplateParams();
    auto channelEmulatorParamsJsonString = R"(
{
    "poissonSpikeTrains": [
        {
            "fromChannelId" : 0,
            "toChannelId" : 800,
            "rate": 8
        }
    ]
}
)";
    (*params)["simulation"]["populationGenerator"] = "p1000";
    (*params)["simulation"]["untilTime"] = 1.0;
    (*params)["channelEmulator"] = nlohmann::json::parse(channelEmulatorParamsJsonString);
    (*params)["channelProjectors"]["OneToOne"]["epsp"] = 1.0;
    (*params)["populationGenerators"]["p1000"]["inhibitorySynapseWeight"] = 0.4;
    (*params)["populationGenerators"]["p1000"]["excitatorySynapseInitialWeight"] = 0.1;

    StaticInputSimulation simulation1(params);
    auto simulation1Result = simulation1.run();

    (*params)["eventProcessor"]["lookAheadWindow"] = 51.3e-3;

    StaticInputSimulation simulation2(params);
    auto simulation2Result = simulation2.run();

    ASSERT_EQ(simulation1Result.recordedSpikes.size(), simulation2Result.recordedSpikes.size());

    for (SizeType i = 0; i < simulation1Result.recordedSpikes.size(); ++i) {
        const auto& spikeInfo1 = simulation1Result.recordedSpikes[i];
        const auto& spikeInfo2 = simulation2Result.recordedSpikes[i];

        ASSERT_FLOAT_EQ(spikeInfo1.time, spikeInfo2.time);
        ASSERT_EQ(spikeInfo1.neuronId, spikeInfo2.neuronId);
    }

    ASSERT_EQ(simulation1Result.finalSynapseInfos.size(), simulation2Result.finalSynapseInfos.size());

    for (SizeType i = 0; i < simulation1Result.finalSynapseInfos.size(); ++i) {
        const auto& synapseInfo1 = simulation1Result.finalSynapseInfos[i];
        const auto& synapseInfo2 = simulation2Result.finalSynapseInfos[i];

        ASSERT_EQ(synapseInfo1.preSynapticNeuronId, synapseInfo2.preSynapticNeuronId);
        ASSERT_EQ(synapseInfo1.postSynapticNeuronId, synapseInfo2.postSynapticNeuronId);
        ASSERT_FLOAT_EQ(synapseInfo1.weight, synapseInfo2.weight);
    }

    ASSERT_EQ(simulation1Result.numExcitatorySpikes, simulation2Result.numExcitatorySpikes);
    ASSERT_EQ(simulation1Result.numInhibitorySpikes, simulation2Result.numInhibitorySpikes);
    ASSERT_EQ(simulation1Result.numEventsProcessed, simulation2Result.numEventsProcessed);
}

TEST(BasicIntegrationTests, OneToManyChannelProjectorInput) {
    auto params = getTemplateParams();

    (*params)["simulation"]["populationGenerator"] = "p1000";
    (*params)["populationGenerators"]["p1000"]["excitatorySynapseInitialWeight"] = 0.0;

    (*params)["simulation"]["channelProjector"] = "OneToMany";
    (*params)["channelProjectors"]["OneToMany"]["fromInChannelId"] = 10;
    (*params)["channelProjectors"]["OneToMany"]["toInChannelId"] = 12;
    (*params)["channelProjectors"]["OneToMany"]["fromSensoryNeuronId"] = 100;
    (*params)["channelProjectors"]["OneToMany"]["toSensoryNeuronId"] = 110;
    (*params)["channelProjectors"]["OneToMany"]["divergence"] = 2;
    (*params)["channelProjectors"]["OneToMany"]["epsp"] = 1.0;

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {11e-3, 10},
        {101e-3, 11},
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 4);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 11e-3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[1].time, 11e-3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[2].time, 101e-3);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[3].time, 101e-3);

    for (auto i = 0; i < 4; ++i) {
        SizeType neuronId = simulationResult.recordedSpikes[i].neuronId;
        ASSERT_TRUE(neuronId >= 100 && neuronId < 110);
    }
}

TEST(BasicIntegrationTests, OneToManyChannelProjectorOutput) {
    auto params = getTemplateParams();

    (*params)["simulation"]["populationGenerator"] = "p1000";
    (*params)["populationGenerators"]["p1000"]["excitatorySynapseInitialWeight"] = 0.0;

    (*params)["simulation"]["channelProjector"] = "OneToMany";
    (*params)["channelProjectors"]["OneToMany"]["fromInChannelId"] = 10;
    (*params)["channelProjectors"]["OneToMany"]["toInChannelId"] = 11;
    (*params)["channelProjectors"]["OneToMany"]["fromSensoryNeuronId"] = 100;
    (*params)["channelProjectors"]["OneToMany"]["toSensoryNeuronId"] = 105;

    (*params)["channelProjectors"]["OneToMany"]["fromOutChannelId"] = 6;
    (*params)["channelProjectors"]["OneToMany"]["toOutChannelId"] = 8;
    (*params)["channelProjectors"]["OneToMany"]["fromMotorNeuronId"] = 100;
    (*params)["channelProjectors"]["OneToMany"]["toMotorNeuronId"] = 105;

    (*params)["channelProjectors"]["OneToMany"]["divergence"] = 5;
    (*params)["channelProjectors"]["OneToMany"]["epsp"] = 1.0;

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {11e-3, 10},
    });

    simulation.recordOutputChannel(6);
    simulation.recordOutputChannel(7);

    simulation.run();

    auto recordedOutputChannelSpikes = simulation.getRecordedOutputChannelSpikes();

    ASSERT_EQ(recordedOutputChannelSpikes.size(), 5);

    SizeType spikeCountChannel6 = 0;
    SizeType spikeCountChannel7 = 0;

    for (auto& channelSpikeInfo : recordedOutputChannelSpikes) {
        ASSERT_FLOAT_EQ(channelSpikeInfo.time, 11e-3);

        if (channelSpikeInfo.channelId == 6) {
            ++ spikeCountChannel6;
        } else if (channelSpikeInfo.channelId == 7) {
            ++ spikeCountChannel7;
        }
    }

    ASSERT_EQ(spikeCountChannel6, 3);
    ASSERT_EQ(spikeCountChannel7, 2);
}