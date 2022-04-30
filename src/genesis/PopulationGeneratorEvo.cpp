#include "PopulationGeneratorEvo.hpp"
#include <params/ParamsFactories.hpp>
#include "TrivialNeuroComponentsFactory.hpp"
#include <neuro/ExplicitChannelProjector.hpp>
#include <neuro/ChannelProjectorFactory.hpp>

namespace soft_npu {

PopulationGeneratorEvo::PopulationGeneratorEvo(const ParamsType& params,
                                               RandomEngineType& randomEngine) :
                                               params(params),
                                               randomEngine(randomEngine) {
}

std::unique_ptr<Population> PopulationGeneratorEvo::generatePopulation() {

    auto population = std::make_unique<Population>();
    auto channelProjector = std::make_unique<ExplicitChannelProjector>();

    ValueType channelProjectedEpsp = params["populationGenerators"]["pEvo"]["channelProjectedEpsp"];
    SizeType targetNumMotorNeurons = params["populationGenerators"]["pEvo"]["targetNumMotorNeurons"];

    SizeType numMotorNeurons = targetNumMotorNeurons;
    if (numMotorNeurons % 2 != 0) {
        -- numMotorNeurons;
    }

    TimeType minCondutionDelay = params["populationGenerators"]["pEvo"]["minConductionDelay"];
    TimeType maxCondutionDelay = params["populationGenerators"]["pEvo"]["maxConductionDelay"];
    ValueType inhibitorySynapseWeight = params["populationGenerators"]["pEvo"]["inhibitorySynapseWeight"];
    ValueType maxSynapticWeight = params["synapseParams"]["maxWeight"];

    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(params);
    auto inhibitoryNeuronParams = ParamsFactories::extractInhibitoryNeuronParams(params);
    auto synapseParams = ParamsFactories::extractSynapseParams(params);

    TrivialNeuroComponentsFactory factory;

    SizeType numNeurons = 4 + numMotorNeurons;

    for (SizeType neuronId = 0; neuronId < numNeurons; ++ neuronId) {
        bool isInhibitory = neuronId >= numNeurons - 2;

        auto neuronParams = isInhibitory ? inhibitoryNeuronParams :
        excitatoryNeuronParams;

        population->addNeuron(factory.makeNeuron(neuronId, neuronParams),
                              Population::defaultLocation);
    }

    auto& sensoryNeuron0 = population->getNeuronById(0);
    auto& sensoryNeuron1 = population->getNeuronById(1);

    channelProjector->addSensoryNeuron(sensoryNeuron0, {0}, channelProjectedEpsp);
    channelProjector->addSensoryNeuron(sensoryNeuron1, {1}, channelProjectedEpsp);

    auto fromMotorNeuronId = 2;
    auto toMotorNeuronId = fromMotorNeuronId + numMotorNeurons;

    // full connectivity between sensory and motor neurons

    std::uniform_real_distribution<TimeType>
    conductionDelayDistribution(minCondutionDelay, maxCondutionDelay);

    for (SizeType preSynNeuronId = 0; preSynNeuronId < 2; ++ preSynNeuronId) {
        for (SizeType postSynNeuronId = fromMotorNeuronId; postSynNeuronId <
            toMotorNeuronId; ++ postSynNeuronId) {
            auto& preSynNeuron = population->getNeuronById(preSynNeuronId);
            auto& postSynNeuron = population->getNeuronById(postSynNeuronId);

            ValueType initialWeight = 0.0;
            TimeType conductionDelay = conductionDelayDistribution(randomEngine);

            auto synapse = factory.makeSynapse(
                    synapseParams,
                    &preSynNeuron,
                    &postSynNeuron,
                    conductionDelay,
                    initialWeight);

            preSynNeuron.addOutboundSynapse(synapse.get());
            population->addExcitatorySynapse(std::move(synapse));
        }
    }


    // two mutually inhibiting groups of motor neurons

    auto& inhibitoryNeuron0 = population->getNeuronById(numNeurons - 2);
    auto& inhibitoryNeuron1 = population->getNeuronById(numNeurons - 1);

    for (SizeType motorNeuronId = fromMotorNeuronId; motorNeuronId < toMotorNeuronId;  ++ motorNeuronId) {

        auto& motorNeuron = population->getNeuronById(motorNeuronId);

        bool group = motorNeuronId < fromMotorNeuronId + numMotorNeurons / 2;
        auto& outInhibitoryNeuron = group ? inhibitoryNeuron1 : inhibitoryNeuron0;
        auto& inInhibitoryNeuron = group ? inhibitoryNeuron0 : inhibitoryNeuron1;

        SizeType channelId = group ? 1 : 0;
        channelProjector->addMotorNeuron(motorNeuron, channelId);

        TimeType conductionDelay = 0.1e-3;

        auto outSynapse = factory.makeSynapse(
                synapseParams,
                &motorNeuron,
                &outInhibitoryNeuron,
                conductionDelay,
                maxSynapticWeight
                );

        auto inSynapse = factory.makeSynapse(
                synapseParams,
                &inInhibitoryNeuron,
                &motorNeuron,
                conductionDelay,
                inhibitorySynapseWeight
                );

        motorNeuron.addOutboundSynapse(outSynapse.get());
        inInhibitoryNeuron.addOutboundSynapse(inSynapse.get());

        population->addExcitatorySynapse(std::move(outSynapse));
        population->addInhibitorySynapse(std::move(inSynapse));
    }

    population->setChannelProjector(std::move(channelProjector));

    return population;
}


}
