#include <gtest/gtest.h>
#include <TestUtil.hpp>
#include <Aliases.hpp>
#include <genesis/PopulationGeneratorFactory.hpp>
#include <genesis/CloningPopulationGenerator.hpp>
#include <neuro/Synapse.hpp>

using namespace soft_npu;

void checkPopulationForCloningTest(
        const Population& population,
        ValueType expectedSynapticWeight) {
    ASSERT_EQ(population.getPopulationSize(), 2);

    const Neuron& neuron0 = population.getNeuronById(0);
    const Neuron& neuron1 = population.getNeuronById(1);

    ASSERT_EQ(neuron0.getNeuronParams()->isInhibitory, false);
    ASSERT_EQ(neuron1.getNeuronParams()->isInhibitory, true);

    ASSERT_EQ(std::distance(neuron0.cbeginOutboundSynapses(), neuron0.cendOutboundSynapses()), 1);
    ASSERT_EQ(std::distance(neuron1.cbeginOutboundSynapses(), neuron1.cendOutboundSynapses()), 0);

    const Synapse& synapse = **neuron0.cbeginOutboundSynapses();
    ASSERT_EQ(synapse.preSynapticNeuron->getNeuronId(), 0);
    ASSERT_EQ(synapse.postSynapticNeuron->getNeuronId(), 1);
    ASSERT_FLOAT_EQ(synapse.conductionDelay, 1e-3);
    ASSERT_FLOAT_EQ(synapse.weight, expectedSynapticWeight);
}

TEST(PopulationGeneratorTests, CloningPopulationGeneratorTest) {
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

    auto params = getTemplateParams();
    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);

    RandomEngineType randomEngine;
    auto cloneePopulation = PopulationGeneratorFactory::
            createFromParams(*params, randomEngine)->generatePopulation();

    CloningPopulationGenerator cloningPopulationGenerator(*cloneePopulation);

    auto clonedPopulation = cloningPopulationGenerator.generatePopulation();
    Synapse& clonedSynapse = **clonedPopulation->getNeuronById(0).cbeginOutboundSynapses();
    clonedSynapse.weight = 0.8;


    checkPopulationForCloningTest(*cloneePopulation, 1.0);
    checkPopulationForCloningTest(*clonedPopulation, 0.8);
}

TEST(PopulationGeneratorTests, R2DSheetTest) {
    auto params = getTemplateParams();
    (*params)["simulation"]["populationGenerator"] = "r2dSheet";
    (*params)["populationGenerators"]["r2dSheet"]["pctInhibitoryNeurons"] = 0.1;
    (*params)["populationGenerators"]["r2dSheet"]["maxExcConductionDelay"] = 10e-3;
    (*params)["populationGenerators"]["r2dSheet"]["inhibitoryConductionDelayRandomPart"] = 5e-3;
    RandomEngineType randomEngine;
    auto population = PopulationGeneratorFactory::createFromParams(*params, randomEngine)->generatePopulation();

    auto countIncompleteTargetProjections = 0;

    for (auto it = population->cbeginNeurons(); it != population->cendNeurons(); ++ it) {
        auto& sourceNeuron = *it;
        auto sourceLocation = population->getCellLocation(sourceNeuron->getNeuronId());
        bool isInhibitory = sourceNeuron->getNeuronId() >= 9000;
        if (isInhibitory) {
            for (auto synIt = sourceNeuron->cbeginOutboundSynapses(); synIt != sourceNeuron->cendOutboundSynapses(); ++ synIt) {
                auto& synapse = *synIt;

                ASSERT_TRUE(synapse->conductionDelay >= 3e-3);
                ASSERT_TRUE(synapse->conductionDelay <= 8e-3);

                auto distance = PopulationUtils::getDistance(sourceLocation, population->getCellLocation(synapse->postSynapticNeuron->getNeuronId()));

                ASSERT_TRUE(distance < 0.01);
            }
        } else {
            for (auto synIt = sourceNeuron->cbeginOutboundSynapses(); synIt != sourceNeuron->cendOutboundSynapses(); ++ synIt) {
                auto& synapse = *synIt;

                ASSERT_TRUE(synapse->conductionDelay >= std::numeric_limits<ValueType>::epsilon());
                ASSERT_TRUE(synapse->conductionDelay <= 10e-3 - std::numeric_limits<ValueType>::epsilon());
            }

            std::vector<SizeType> distantNeuronIds;
            std::vector<SizeType> nearbyNeuronIds;


            for (auto synIt = sourceNeuron->cbeginOutboundSynapses(); synIt != sourceNeuron->cendOutboundSynapses(); ++ synIt) {
                auto& synapse = *synIt;
                auto postSynNeuronId = synapse->postSynapticNeuron->getNeuronId();

                auto distance = PopulationUtils::getDistance(sourceLocation, population->getCellLocation(postSynNeuronId));

                if (distance >= 0.03) {
                    distantNeuronIds.push_back(postSynNeuronId);
                } else {
                    nearbyNeuronIds.push_back(postSynNeuronId);
                }
            }

            if (nearbyNeuronIds.size() < 8) {
                SizeType eligibleNeuronCount = 0;

                for (auto testNeuronIt = population->cbeginNeurons(); testNeuronIt != population->cendNeurons(); ++ testNeuronIt) {
                    auto& testNeuron = *testNeuronIt;

                    if (PopulationUtils::getDistance(sourceLocation, population->getCellLocation(testNeuron->getNeuronId())) < 0.03 &&
                        testNeuron->getNeuronId() != sourceNeuron->getNeuronId()) {
                        ++ eligibleNeuronCount;
                    }
                }

                ASSERT_EQ(nearbyNeuronIds.size(), eligibleNeuronCount);
            }

            if (std::distance(sourceNeuron->cbeginOutboundSynapses(), sourceNeuron->cendOutboundSynapses()) == 10) {
                ASSERT_EQ(distantNeuronIds.size(), 2);

                auto distantNeuronIdsDistance = PopulationUtils::getDistance(
                        population->getCellLocation(distantNeuronIds[0]),
                        population->getCellLocation(distantNeuronIds[1])
                );

                ASSERT_TRUE(distantNeuronIdsDistance <= 0.01 * 2);

            } else {
                ++ countIncompleteTargetProjections;
            }
        }
    }

    ASSERT_TRUE(countIncompleteTargetProjections < 2500); // just a ballpark
}
