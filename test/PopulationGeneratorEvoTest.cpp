#include <gtest/gtest.h>
#include <TestUtil.hpp>
#include <Aliases.hpp>
#include <genesis/PopulationGeneratorEvo.hpp>
#include <vector>
#include <unordered_set>
#include <core/StaticInputSimulation.hpp>

using namespace soft_npu;

auto makeParamsTemplate() {
    auto params = getTemplateParams();

    auto pEvoParams = R"(
{
    "channelProjectedEpsp": 1.5,
    "inChannelDivergence": 10,
    "outChannelConvergence": 20,
    "minConductionDelay": 1e-3,
    "maxConductionDelay": 20e-3,
    "initialWeight": 0.1,
    "intraCircuitConnectDensity": 1,
    "interCircuitConnectDensity": 1
}
)"_json;

    auto autoInhibitionNeuronParams = R"(
{
    "timeConstant": 10e-3,
    "refractoryPeriod": 0e-3,
    "thresholdVoltage": 1.0,
    "resetVoltage": 0.0,
    "voltageFloor": 0.0,
    "isInhibitory": true,
    "epspOverrideScaleFactor": 0.1

}
)"_json;

    auto crossInhibitionNeuronParams = autoInhibitionNeuronParams;
    crossInhibitionNeuronParams["epspOverrideScaleFactor"] = 0.2;

    (*params)["simulation"]["populationGenerator"] = "pEvo";
    (*params)["populationGenerators"]["pEvo"] = pEvoParams;
    (*params)["neuronParams"]["autoInhibition"] = autoInhibitionNeuronParams;
    (*params)["neuronParams"]["crossInhibition"] = crossInhibitionNeuronParams;
    return params;
}

auto generatePopulation(const ParamsType& params) {
    RandomEngineType randomEngine(0);
    PopulationGeneratorEvo populationGenerator(params, randomEngine);
    return populationGenerator.generatePopulation();
}

void checkInhibitionSources(const Neuron& neuron) {
    std::vector<SizeType> expectedInhibitionSourceNeuronIds;
    if (neuron.getNeuronId() >= 20 && neuron.getNeuronId() < 40) {
        expectedInhibitionSourceNeuronIds = {60, 62};
    } else if (neuron.getNeuronId() >= 40 && neuron.getNeuronId() < 60) {
        expectedInhibitionSourceNeuronIds = {61, 63};
    }

    std::vector<SizeType> actualInhibitionSourceNeuronIds;

    std::transform(
            neuron.cbeginInhibitionSources(),
            neuron.cendInhibitionSources(),
            std::back_inserter(actualInhibitionSourceNeuronIds),
            [] (auto inhibitionSource) {
                return inhibitionSource->getNeuronId();
            });

    ASSERT_EQ(expectedInhibitionSourceNeuronIds, actualInhibitionSourceNeuronIds);

    if (neuron.getNeuronId() == 60 || neuron.getNeuronId() == 63) {
        ASSERT_FLOAT_EQ(neuron.getNeuronParams()->epspOverrideScaleFactor, 0.1);
    } else if (neuron.getNeuronId() == 61 || neuron.getNeuronId() == 62) {
        ASSERT_FLOAT_EQ(neuron.getNeuronParams()->epspOverrideScaleFactor, 0.2);
    }
}

void checkSynapseParams(const Population& population) {
    std::for_each(population.cbeginNeurons(), population.cendNeurons(), [](const auto& neuron) {
        std::for_each(neuron->cbeginOutboundSynapses(), neuron->cendOutboundSynapses(), [](const auto synapse) {
            ASSERT_GE(synapse->conductionDelay, 1e-3);
            ASSERT_LT(synapse->conductionDelay, 20e-3);
            ASSERT_FLOAT_EQ(synapse->weight, 0.1);
        });
    });
}

void checkHasExpectedPostSynNeurons(const Neuron& neuron, std::unordered_set<SizeType> expectedPostSynNeuronIds) {
    std::unordered_set<SizeType> actualPostSynNeuronIds;

    std::transform(
            neuron.cbeginOutboundSynapses(),
            neuron.cendOutboundSynapses(),
            std::inserter(actualPostSynNeuronIds, actualPostSynNeuronIds.begin()),
            [] (auto synapse) {
                return synapse->postSynapticNeuron->getNeuronId();
            });

    ASSERT_EQ(expectedPostSynNeuronIds, actualPostSynNeuronIds);
}

TEST (PopulationGeneratorEvoTest, NoIntraNoInterConnect) {
    auto params = makeParamsTemplate();
    (*params)["populationGenerators"]["pEvo"]["intraCircuitConnectDensity"] = 0;
    (*params)["populationGenerators"]["pEvo"]["interCircuitConnectDensity"] = 0;
    auto population = generatePopulation(*params);

    ASSERT_EQ(population->getPopulationSize(), 64); 
    std::for_each(population->cbeginNeurons(), population->cendNeurons(), [](const auto& neuron) {
        std::unordered_set<SizeType> expectedPostSynNeuronIds;

        if (neuron->getNeuronId() >= 20 && neuron->getNeuronId() < 40) {
            expectedPostSynNeuronIds.insert(60);
            expectedPostSynNeuronIds.insert(61);
        } else if (neuron->getNeuronId() >= 40 && neuron->getNeuronId() < 60) {
            expectedPostSynNeuronIds.insert(62);
            expectedPostSynNeuronIds.insert(63);
        }

        checkHasExpectedPostSynNeurons(*neuron, expectedPostSynNeuronIds);
        checkInhibitionSources(*neuron);
    });

    checkSynapseParams(*population);
}

TEST (PopulationGeneratorEvoTest, FullIntraNoInterConnect) {
    auto params = makeParamsTemplate();
    (*params)["populationGenerators"]["pEvo"]["intraCircuitConnectDensity"] = 1;
    (*params)["populationGenerators"]["pEvo"]["interCircuitConnectDensity"] = 0;
    auto population = generatePopulation(*params);

    ASSERT_EQ(population->getPopulationSize(), 64); 
    std::for_each(population->cbeginNeurons(), population->cendNeurons(), [](const auto& neuron) {
        std::unordered_set<SizeType> expectedPostSynNeuronIds;

        if (neuron->getNeuronId() >= 20 && neuron->getNeuronId() < 40) {
            for (SizeType postSynNeuronId = 20; postSynNeuronId < 40; ++ postSynNeuronId) {
                if (postSynNeuronId != neuron->getNeuronId()) {
                    expectedPostSynNeuronIds.insert(postSynNeuronId);
                }
            }

            expectedPostSynNeuronIds.insert(60);
            expectedPostSynNeuronIds.insert(61);
        } else if (neuron->getNeuronId() >= 40 && neuron->getNeuronId() < 60) {
            for (SizeType postSynNeuronId = 40; postSynNeuronId < 60; ++ postSynNeuronId) {
                if (postSynNeuronId != neuron->getNeuronId()) {
                    expectedPostSynNeuronIds.insert(postSynNeuronId);
                }
            }

            expectedPostSynNeuronIds.insert(62);
            expectedPostSynNeuronIds.insert(63);
        }

        checkHasExpectedPostSynNeurons(*neuron, expectedPostSynNeuronIds);
        checkInhibitionSources(*neuron);
    });

    checkSynapseParams(*population);
}

TEST (PopulationGeneratorEvoTest, NoIntraFullInterConnect) {
    auto params = makeParamsTemplate();
    (*params)["populationGenerators"]["pEvo"]["intraCircuitConnectDensity"] = 0;
    (*params)["populationGenerators"]["pEvo"]["interCircuitConnectDensity"] = 1;
    auto population = generatePopulation(*params);

    ASSERT_EQ(population->getPopulationSize(), 64); 
    std::for_each(population->cbeginNeurons(), population->cendNeurons(), [](const auto& neuron) {
        std::unordered_set<SizeType> expectedPostSynNeuronIds;

        if (neuron->getNeuronId() < 20) {
            for (SizeType postSynNeuronId = 20; postSynNeuronId < 60; ++postSynNeuronId) {
                expectedPostSynNeuronIds.insert(postSynNeuronId);
            }
        } else if (neuron->getNeuronId() >= 20 && neuron->getNeuronId() < 40) {
            expectedPostSynNeuronIds.insert(60);
            expectedPostSynNeuronIds.insert(61);
        } else if (neuron->getNeuronId() >= 40 && neuron->getNeuronId() < 60) {
            expectedPostSynNeuronIds.insert(62);
            expectedPostSynNeuronIds.insert(63);
        }

        checkHasExpectedPostSynNeurons(*neuron, expectedPostSynNeuronIds);
        checkInhibitionSources(*neuron);
    });

    checkSynapseParams(*population);
}

TEST (PopulationGeneratorEvoTest, FullIntraFullInterConnect) {
    auto params = makeParamsTemplate();
    (*params)["populationGenerators"]["pEvo"]["intraCircuitConnectDensity"] = 1;
    (*params)["populationGenerators"]["pEvo"]["interCircuitConnectDensity"] = 1;
    auto population = generatePopulation(*params);

    ASSERT_EQ(population->getPopulationSize(), 64); 
    std::for_each(population->cbeginNeurons(), population->cendNeurons(), [](const auto& neuron) {
        std::unordered_set<SizeType> expectedPostSynNeuronIds;

        if (neuron->getNeuronId() < 20) {
            for (SizeType postSynNeuronId = 20; postSynNeuronId < 60; ++postSynNeuronId) {
                expectedPostSynNeuronIds.insert(postSynNeuronId);
            }
        } else if (neuron->getNeuronId() >= 20 && neuron->getNeuronId() < 40) {
            for (SizeType postSynNeuronId = 20; postSynNeuronId < 40; ++ postSynNeuronId) {
                if (postSynNeuronId != neuron->getNeuronId()) {
                    expectedPostSynNeuronIds.insert(postSynNeuronId);
                }
            }

            expectedPostSynNeuronIds.insert(60);
            expectedPostSynNeuronIds.insert(61);
        } else if (neuron->getNeuronId() >= 40 && neuron->getNeuronId() < 60) {
            for (SizeType postSynNeuronId = 40; postSynNeuronId < 60; ++ postSynNeuronId) {
                if (postSynNeuronId != neuron->getNeuronId()) {
                    expectedPostSynNeuronIds.insert(postSynNeuronId);
                }
            }

            expectedPostSynNeuronIds.insert(62);
            expectedPostSynNeuronIds.insert(63);
        }

        checkHasExpectedPostSynNeurons(*neuron, expectedPostSynNeuronIds);
        checkInhibitionSources(*neuron);
    });

    checkSynapseParams(*population);
}

TEST (PopulationGeneratorEvoTest, InputChannelWiring) {
    auto params = makeParamsTemplate();
    (*params)["populationGenerators"]["pEvo"]["intraCircuitConnectDensity"] = 0;
    (*params)["populationGenerators"]["pEvo"]["interCircuitConnectDensity"] = 0;

    StaticInputSimulation simulation(params);
    simulation.setSpikeTrains({
        {10e-3, 0},
        {15e-3, 1}
    });

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 20);

    for (SizeType i = 0; i < 20; ++i) {
        TimeType expectedTime = i < 10 ? 10e-3 : 15e-3;
        ASSERT_EQ(simulationResult.recordedSpikes[i].neuronId, i);
        ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[i].time, expectedTime);
    }
}

TEST (PopulationGeneratorEvoTest, OutputChannelWiring) {
    auto params = makeParamsTemplate();
    (*params)["populationGenerators"]["pEvo"]["inChannelDivergence"] = 1;
    (*params)["populationGenerators"]["pEvo"]["initialWeight"] = 1;
    (*params)["neuronParams"]["autoInhibition"]["epspOverrideScaleFactor"] = 0;
    (*params)["neuronParams"]["crossInhibition"]["epspOverrideScaleFactor"] = 0;
    (*params)["populationGenerators"]["pEvo"]["intraCircuitConnectDensity"] = 0;
    (*params)["populationGenerators"]["pEvo"]["interCircuitConnectDensity"] = 1;

    StaticInputSimulation simulation(params);
    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    for (SizeType outChannelId = 0; outChannelId < 2; ++ outChannelId) {
        simulation.recordOutputChannel(outChannelId);
    }

    auto simulationResult = simulation.run();

    ASSERT_EQ(simulationResult.recordedSpikes.size(), 41);
    ASSERT_FLOAT_EQ(simulationResult.recordedSpikes[0].time, 10e-3);
    ASSERT_EQ(simulationResult.recordedSpikes[0].neuronId, 0);

    std::unordered_set<SizeType> distinctNeuronIds;
    std::array<std::vector<TimeType>, 2> expectedSpikeTimesPerOutChannelId;
    std::array<std::vector<TimeType>, 2> actualSpikeTimesPerOutChannelId;

    for (int i = 1; i < 41; ++i) {

        auto time = simulationResult.recordedSpikes[i].time;
        auto neuronId = simulationResult.recordedSpikes[i].neuronId;

        ASSERT_GE(time, 11e-3);
        ASSERT_LT(time, 30e-3);
        ASSERT_GE(neuronId, 2);
        ASSERT_LT(neuronId, 42);

        distinctNeuronIds.insert(neuronId);

        SizeType outChannelId = neuronId < 22 ? 0 : 1;
        expectedSpikeTimesPerOutChannelId[outChannelId].push_back(time);
    }

    for (const auto& outChannelSpike : simulation.getRecordedOutputChannelSpikes()) {
        actualSpikeTimesPerOutChannelId[outChannelSpike.channelId].push_back(outChannelSpike.time);
    }

    ASSERT_EQ(actualSpikeTimesPerOutChannelId, expectedSpikeTimesPerOutChannelId);
    ASSERT_EQ(distinctNeuronIds.size(), 40);
}
