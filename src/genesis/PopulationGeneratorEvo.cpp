#include "PopulationGeneratorEvo.hpp"
#include <params/ParamsFactories.hpp>
#include "TrivialNeuroComponentsFactory.hpp"
#include <neuro/ExplicitChannelProjector.hpp>
#include <neuro/ChannelProjectorFactory.hpp>
#include "SubCircuitAdapter.hpp"
#include "ConnectivityParams.hpp"

namespace soft_npu {

PopulationGeneratorEvo::PopulationGeneratorEvo(const ParamsType& params,
                                               RandomEngineType& randomEngine) :
                                               params(params),
                                               randomEngine(randomEngine) {
}

std::unique_ptr<Population> PopulationGeneratorEvo::generatePopulation() {
    TrivialNeuroComponentsFactory factory;
    auto population = std::make_unique<Population>();
    auto channelProjector = std::make_unique<ExplicitChannelProjector>();
    const auto& evoParams = params["populationGenerators"]["pEvo"];

    std::array<std::unique_ptr<SubCircuitAdapter>, 2> sensoryCircuits;
    std::array<std::unique_ptr<SubCircuitAdapter>, 2> motorCircuits;

    for (SizeType i = 0; i < 2; ++i) {
        sensoryCircuits[i] = std::make_unique<SubCircuitAdapter>(
            factory, *population, *channelProjector, randomEngine);

        motorCircuits[i] = std::make_unique<SubCircuitAdapter>(
            factory, *population, *channelProjector, randomEngine);
    }

    SizeType inChannelDivergence = evoParams["inChannelDivergence"];
    ValueType channelProjectedEpsp = evoParams["channelProjectedEpsp"];
    SizeType outChannelConvergence = evoParams["outChannelConvergence"];

    ConnectivityParams connParamsIntra;
    connParamsIntra.minConductionDelay = evoParams["minConductionDelay"];
    connParamsIntra.maxConductionDelay = evoParams["maxConductionDelay"];
    connParamsIntra.minInitialWeight = evoParams["minInitialWeight"];
    connParamsIntra.maxInitialWeight = evoParams["maxInitialWeight"];
    connParamsIntra.connectDensity = evoParams["intraCircuitConnectDensity"];

    ConnectivityParams connParamsInter = connParamsIntra;
    connParamsInter.connectDensity = evoParams["interCircuitConnectDensity"];

    ConnectivityParams connParamsContinuousInhibition = connParamsIntra;
    connParamsContinuousInhibition.connectDensity = 1.0;

    auto excitatoryNeuronParams = ParamsFactories::extractExcitatoryNeuronParams(params);
    auto synapseParams = ParamsFactories::extractSynapseParams(params);

    for (SizeType inChannelId = 0; inChannelId < 2; ++ inChannelId) {
        auto& circuit = *sensoryCircuits[inChannelId];
        circuit.addNeurons(inChannelDivergence, excitatoryNeuronParams);
        circuit.connectAllToInput(inChannelId, channelProjectedEpsp);
    }

    for (SizeType outChannelId = 0; outChannelId < 2; ++ outChannelId) {
        auto& circuit = *motorCircuits[outChannelId];
        circuit.addNeurons(outChannelConvergence, excitatoryNeuronParams);
        circuit.buildInternalConnnectivity(synapseParams, connParamsIntra);
        circuit.connectAllToOutput(outChannelId);
    }

    for (SizeType i = 0; i < 2; ++i) {
        for (SizeType j = 0; j < 2; ++j) {
            sensoryCircuits[i]->createProjectionOnto(*motorCircuits[j], connParamsInter, synapseParams);
        }
    }

    auto crossInhibitionNeuronParams = ParamsFactories::extractNeuronParams(params, "crossInhibition");
    auto autoInhibitionNeuronParams = ParamsFactories::extractNeuronParams(params, "autoInhibition");

    auto inhibitionSource00 = factory.makeNeuron(autoInhibitionNeuronParams, *population);
    motorCircuits[0]->addInhibitionSource(*inhibitionSource00);
    motorCircuits[0]->addInhibitionSink(*inhibitionSource00, connParamsContinuousInhibition, synapseParams);
    population->addNeuron(std::move(inhibitionSource00), Population::defaultLocation);
    
    auto inhibitionSource01 = factory.makeNeuron(crossInhibitionNeuronParams, *population);
    motorCircuits[1]->addInhibitionSource(*inhibitionSource01);
    motorCircuits[0]->addInhibitionSink(*inhibitionSource01, connParamsContinuousInhibition, synapseParams);
    population->addNeuron(std::move(inhibitionSource01), Population::defaultLocation);

    auto inhibitionSource10 = factory.makeNeuron(crossInhibitionNeuronParams, *population);
    motorCircuits[0]->addInhibitionSource(*inhibitionSource10);
    motorCircuits[1]->addInhibitionSink(*inhibitionSource10, connParamsContinuousInhibition, synapseParams);
    population->addNeuron(std::move(inhibitionSource10), Population::defaultLocation);

    auto inhibitionSource11 = factory.makeNeuron(autoInhibitionNeuronParams, *population);
    motorCircuits[1]->addInhibitionSource(*inhibitionSource11);
    motorCircuits[1]->addInhibitionSink(*inhibitionSource11, connParamsContinuousInhibition, synapseParams);
    population->addNeuron(std::move(inhibitionSource11), Population::defaultLocation);

    population->setChannelProjector(std::move(channelProjector));

    return population;
}

}
