#include <genesis/PopulationGeneratorFactory.hpp>
#include <chrono>
#include <plog/Log.h>
#include "AbstractSimulation.hpp"
#include "CycleController.hpp"
#include <core/SynapseInfo.hpp>
#include <core/SimulationResult.hpp>
#include "Recordings.hpp"
#include "SynapticTransmissionStats.hpp"

namespace soft_npu {

AbstractSimulation::AbstractSimulation(std::shared_ptr<const ParamsType> params) :
    randomEngine((*params)["simulation"]["seed"]),
    params(params),
    populationGenerator(PopulationGeneratorFactory::createFromParams(*params, randomEngine)) {
}

void AbstractSimulation::recordVoltage(SizeType neuronId, TimeType time) {
    neuronIdTimePairsToRecordVoltageAt.emplace_back(neuronId, time);
}

template<typename T>
double convertToSecondsTime(const T& val) {
    return std::chrono::duration_cast<std::chrono::microseconds>(val).count() * 1e-6;
}

std::vector<SynapseInfo> extractSynapseInfos(std::shared_ptr<Population> population) {
    std::vector<SynapseInfo> synapseInfos;

    for (auto itNeuron = population->cbeginNeurons(); itNeuron != population->cendNeurons(); ++itNeuron) {
        auto& preSynapticNeuron = **itNeuron;

        for (auto itSynapse = preSynapticNeuron.cbeginOutboundSynapses(); itSynapse != preSynapticNeuron.cendOutboundSynapses(); ++itSynapse) {
            synapseInfos.emplace_back(
                    preSynapticNeuron.getNeuronId(),
                    (*itSynapse)->postSynapticNeuron->getNeuronId(),
                    (*itSynapse)->weight, preSynapticNeuron.getNeuronParams()->isInhibitory);
        }
    }

    return synapseInfos;
}

std::vector<Population::Location> extractLocationsIndexedByNeuronId(const Population& population) {
    std::vector<Population::Location> locationsIndexedByNeuronId;

    std::transform(
            population.cbeginNeurons(),
            population.cendNeurons(),
            std::back_inserter(locationsIndexedByNeuronId),
            [&population](auto& neuron) {
                return population.getCellLocation(neuron->getNeuronId());
            });

    return locationsIndexedByNeuronId;
}

auto extractNeuronInfos(const Population& population) {
    std::vector<NeuronInfo> neuronInfos;

    std::transform(
            population.cbeginNeurons(),
            population.cendNeurons(),
            std::back_inserter(neuronInfos),
            [](auto& neuron) {
                return NeuronInfo(neuron->getNeuronId(), neuron->getNeuronParams()->isInhibitory);
            });

    return neuronInfos;
}

SimulationResult AbstractSimulation::run() {

    auto startTs = std::chrono::high_resolution_clock::now();

    TimeType simulationTime = (*params)["simulation"]["untilTime"];

    PLOG_DEBUG << "Generating population";

    std::shared_ptr<Population> population = populationGenerator->generatePopulation();

    auto startTsEventProcessor = std::chrono::high_resolution_clock::now();

    auto synapticTransmissionStats = std::make_shared<SynapticTransmissionStats>();

    PLOG_DEBUG << "Creating cycle controller";

    CycleController controller(
            *params,
            randomEngine,
            *population,
            true,
            neuronIdTimePairsToRecordVoltageAt,
            *synapticTransmissionStats
    );

    runController(controller, *population, simulationTime, *synapticTransmissionStats);
    auto endTs = std::chrono::high_resolution_clock::now();

    auto wallTimeTotal = convertToSecondsTime(endTs - startTs);
    double wallTimeEventProcessor = convertToSecondsTime(endTs - startTsEventProcessor);

    auto recordings = controller.getRecordings();

    auto numInhibitoryNeurons = PopulationUtils::getNumInhibitoryNeurons(*population);
    auto numExcitatoryNeurons = population->getPopulationSize() - numInhibitoryNeurons;

    auto meanExcitatoryFiringRate = numExcitatoryNeurons == 0 ? 0 : recordings->numExcitatorySpikes / simulationTime / numExcitatoryNeurons;
    auto meanInhibitoryFiringRate = numInhibitoryNeurons == 0 ? 0 : recordings->numInhibitorySpikes / simulationTime / numInhibitoryNeurons;

    return SimulationResult(
            simulationTime,
            std::move(recordings->neuronSpikeRecordings),
            std::move(recordings->voltageRecordings),
            extractNeuronInfos(*population),
            extractSynapseInfos(population),
            extractLocationsIndexedByNeuronId(*population),
            recordings->numExcitatorySpikes,
            recordings->numInhibitorySpikes,
            synapticTransmissionStats->getTransmissionCount(),
            meanExcitatoryFiringRate,
            meanInhibitoryFiringRate,
            wallTimeTotal,
            wallTimeEventProcessor,
            controller.getNumEventsProcessed());
}

}
