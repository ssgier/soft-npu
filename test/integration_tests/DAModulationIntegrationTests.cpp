#include <gtest/gtest.h>
#include <core/StaticInputSimulation.hpp>
#include <vector>
#include <params/ParamsFactories.hpp>
#include <TestUtil.hpp>

using namespace soft_npu;

auto getCustomizedTemplateParams() {
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
            "conductionDelay": 10e-3
        }
    ]
}
)";

    (*params)["simulation"]["untilTime"] = 2;
    (*params)["simulation"]["populationGenerator"] = "pDetailedParams";
    (*params)["populationGenerators"]["pDetailedParams"] = nlohmann::json::parse(populationJsonText);
    (*params)["dopaminergicModulator"]["releaseBaseRate"] = 0;
    (*params)["synapseParams"]["eligibilityTraceTimeConstant"] = 0.5;
    (*params)["synapseParams"]["eligibilityTraceCutOffTimeFactor"] = 2;

    return params;
}

TEST(DAModulationIntegrationTests, NoPotentiationWithoutDA) {

    StaticInputSimulation simulation(getCustomizedTemplateParams());

    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0);
}

TEST(DAModulationIntegrationTests, SimplePotentiationOnReward) {

    StaticInputSimulation simulation(getCustomizedTemplateParams());

    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    simulation.setRewardDoses({
        {400e-3, 1.0}
    });

    auto simulationResult = simulation.run();

    ValueType expectedDopamineReleaseRate = 1.0 / 250e-3 / (1 - exp(-2));

    ValueType expectedPotentiation =
            0.1 / 500e-3 * exp(-230e-3 / 500e-3)
            * 500e-3 * (1 - exp(-250e-3 / 500e-3)) * expectedDopamineReleaseRate;

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0 + expectedPotentiation);
}

TEST(DAModulationIntegrationTests, RewardVsBaseRateOrthogonality) {

    auto params = getCustomizedTemplateParams();
    (*params)["dopaminergicModulator"]["releaseBaseRate"] = 1.0;

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    simulation.setRewardDoses({
        {400e-3, 1.0}
    });

    auto simulationResult = simulation.run();

    ValueType expectedDopamineReleaseRateForReward = 1.0 / 250e-3 / (1 - exp(-2));
    ValueType expectedPotentiationForReward =
            0.1 / 500e-3 * exp(-230e-3 / 500e-3)
            * 500e-3 * (1 - exp(-250e-3 / 500e-3)) * expectedDopamineReleaseRateForReward;

    ValueType expectedPotentiationForBaseRate = 0.1;

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0 + expectedPotentiationForReward + expectedPotentiationForBaseRate);
}

TEST(DAModulationIntegrationTests, NoReinforcementAfterCutOffTime) {

    auto params = getCustomizedTemplateParams();

    (*params)["synapseParams"]["eligibilityTraceTimeConstant"] = 750e-3 - 20e-3;
    (*params)["synapseParams"]["eligibilityTraceCutOffTimeFactor"] = 1;

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    simulation.setRewardDoses({
        {751e-3, 1.0}
    });

    auto simulationResult = simulation.run();

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0);
}

TEST(DAModulationIntegrationTests, MultipleRewardDosesWithinTraceLifeTime) {

    StaticInputSimulation simulation(getCustomizedTemplateParams());

    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    simulation.setRewardDoses({
        {400e-3, 1.0},
        {800e-3, 0.5}
    });

    auto simulationResult = simulation.run();

    ValueType expectedDopamineReleaseRateFirstDose = 1.0 / 250e-3 / (1 - exp(-2));
    ValueType expectedPotentiationFirstDose =
            0.1 / 500e-3 * exp(-230e-3 / 500e-3)
            * 500e-3 * (1 - exp(-250e-3 / 500e-3)) * expectedDopamineReleaseRateFirstDose;

    ValueType expectedDopamineReleaseRateSecondDose = 0.5 / 250e-3 / (1 - exp(-2));
    ValueType expectedPotentiationSecondDose =
            0.1 / 500e-3 * exp(-730e-3 / 500e-3)
            * 500e-3 * (1 - exp(-250e-3 / 500e-3)) * expectedDopamineReleaseRateSecondDose;

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight,
                    1.0 + expectedPotentiationFirstDose + expectedPotentiationSecondDose);
}

TEST(DAModulationIntegrationTests, SimpleDepressionOnReward) {

    StaticInputSimulation simulation(getCustomizedTemplateParams());

    simulation.setSpikeTrains({
        {10e-3, 0},
        {15e-3, 1}
    });

    simulation.setRewardDoses({
        {400e-3, 1.0}
    });

    auto simulationResult = simulation.run();

    ValueType depressionSTDPValue = - 0.12 * exp(-5e-3 / 20e-3);

    ValueType expectedDopamineReleaseRate = 1.0 / 250e-3 / (1 - exp(-2));
    ValueType expectedDepression =
            - depressionSTDPValue / 500e-3 * exp(-230e-3 / 500e-3)
            * 500e-3 * (1 - exp(-250e-3 / 500e-3)) * expectedDopamineReleaseRate;

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0 - expectedDepression);
}

TEST(DAModulationIntegrationTests, TraceInstantiationDuringDARelease) {

    StaticInputSimulation simulation(getCustomizedTemplateParams());

    simulation.setSpikeTrains({
        {410e-3, 0}
    });

    simulation.setRewardDoses({
        {400e-3, 1.0}
    });

    auto simulationResult = simulation.run();

    ValueType expectedDopamineReleaseRate = 1.0 / 250e-3 / (1 - exp(-2));

    ValueType expectedPotentiation =
            0.1 / 500e-3 * 500e-3 * (1 - exp(-80e-3 / 500e-3)) * expectedDopamineReleaseRate;

    ASSERT_FLOAT_EQ(simulationResult.finalSynapseInfos[0].weight, 1.0 + expectedPotentiation);
}

ValueType getFinalWeightCustomScenario(
        const std::deque<RewardDoseInfo>& rewardDoses,
        ValueType dopamineReleaseBaseRate) {

    auto params = getCustomizedTemplateParams();

    (*params)["dopaminergicModulator"]["releaseBaseRate"] = dopamineReleaseBaseRate;

    StaticInputSimulation simulation(params);

    simulation.setSpikeTrains({
        {10e-3, 0}
    });

    simulation.setRewardDoses(rewardDoses);

    auto simulationResult = simulation.run();

    return simulationResult.finalSynapseInfos[0].weight;
}

TEST(DAModulationIntegrationTests, InequalitiesDifferentDosages) {
    ValueType finalWeightEarlierDose = getFinalWeightCustomScenario({{400e-3, 1.0}}, 0);
    ValueType finalWeightLaterDose = getFinalWeightCustomScenario({{600e-3, 1.0}}, 0);
    ValueType finalWeightDividedDose = getFinalWeightCustomScenario({
        {400e-3, 0.5},
        {600e-3, 0.5}
    }, 0);

    ASSERT_LT(finalWeightDividedDose, finalWeightEarlierDose);
    ASSERT_LT(finalWeightLaterDose, finalWeightDividedDose);

    ValueType potentiationRatioLaterVsEarlier = (finalWeightLaterDose - 1.0) / (finalWeightEarlierDose - 1.0);
    ValueType potentiationRatioDividedVsEarlier = (finalWeightDividedDose - 1.0) / (finalWeightEarlierDose - 1.0);

    ASSERT_FLOAT_EQ(potentiationRatioLaterVsEarlier, exp(-250e-3 / 500e-3));
    ASSERT_FLOAT_EQ(potentiationRatioDividedVsEarlier, 0.5 + 0.5 * exp(-250e-3 / 500e-3));
}

TEST(DAModulationIntegrationTests, EquivalenceRewardVsDABaseRate) {
    ValueType finalWeightReward = getFinalWeightCustomScenario({
        {250e-3, 0.25},
        {500e-3, 0.25},
        {750e-3, 0.25},
        {1000e-3, 0.25},
        {1250e-3, 0.25},
    }, 0);

    ValueType finalWeightDABaseRate = getFinalWeightCustomScenario({}, 1.0);

    ASSERT_FLOAT_EQ(finalWeightReward, finalWeightDABaseRate);
}
