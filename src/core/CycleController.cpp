#include "CycleController.hpp"
#include "Recordings.hpp"
#include <neuro/ChannelProjectorFactory.hpp>
#include <neuro/Population.hpp>
#include "CommonEvent.hpp"

namespace soft_npu {

void setupSpikeRecording(
        Population& population,
        Recordings& recordings) {
    population.addSpikeListener([&recordings](const CycleContext& cycleContext, const Neuron& neuron) {
        recordings.neuronSpikeRecordings.emplace_back(cycleContext.time, neuron.getNeuronId());

        if (neuron.getNeuronParams()->isInhibitory) {
            ++ recordings.numInhibitorySpikes;
        } else {
            ++ recordings.numExcitatorySpikes;
        }
    });
}

void pushVoltageRecordingEvents(Recordings& recordings,
                                       EventProcessor& eventProcessor,
                                       Population& population,
                                       const std::vector<std::pair<SizeType, TimeType>>& neuronIdTimePairsToRecordVoltageAt) {
    for (const auto& neuronIdAndTime : neuronIdTimePairsToRecordVoltageAt) {
        auto neuronId = neuronIdAndTime.first;

        eventProcessor.pushCommonEvent(neuronIdAndTime.second, makeCommonEvent(
        [&recordings, neuronId, &population](const CycleContext& cycleContext) {
            auto voltage = population.getNeuronById(neuronId).getMembraneVoltage(cycleContext.time);
            recordings.voltageRecordings.emplace_back(neuronId, cycleContext.time, voltage);
        }));
    }
}

CycleController::CycleController(const ParamsType& params,
                                 RandomEngineType& randomEngine,
                                 Population& population,
                                 bool recordSpikes,
                                 const std::vector<std::pair<SizeType, TimeType>>& neuronIdTimePairsToRecordVoltageAt,
                                 SynapticTransmissionStats& synapticTransmissionStats) :
        dt(params["cycleController"]["dt"]),
        currentCycle(0),
        currentTime(0),
        nonCoherentStimulator(params, randomEngine, population, dt),
        eventProcessor(params, dt, synapticTransmissionStats),
        dopaminergicModulator(params, population),
        staticContext(
                eventProcessor,
                dopaminergicModulator,
                population,
                cycleOutputBuffer,
                synapticTransmissionStats),
        recordings(std::make_shared<Recordings>())
                                 {
    if (recordSpikes) {
        setupSpikeRecording(population, *recordings);
    }

    pushVoltageRecordingEvents(*recordings, eventProcessor, population, neuronIdTimePairsToRecordVoltageAt);
}

void CycleController::runCycle() {

    cycleOutputBuffer.reset();

    const CycleContext ctx(dt * currentCycle, staticContext, currentCycle);

    for (auto it = cycleInputBuffer.cbeginSpikingChannelIds(); it != cycleInputBuffer.cendSpikingChannelIds(); ++it) {
        staticContext.population.projectChannelSpike(ctx, *it);
    }

    dopaminergicModulator.processReward(ctx, cycleInputBuffer.getReward());

    nonCoherentStimulator.processCycle(ctx);
    eventProcessor.processCycle(ctx);
    dopaminergicModulator.processCycle(ctx);

    ++ currentCycle;
    currentTime = currentCycle * dt;
}

std::shared_ptr<const Recordings> CycleController::getRecordings() const noexcept {
    return recordings;
}

TimeType CycleController::getTime() const noexcept {
    return currentTime;
}

uint64_t CycleController::getNumEventsProcessed() const noexcept {
    return eventProcessor.getNumEventsProcessed();
}

CycleInputBuffer& CycleController::getCycleInputBuffer() {
    return cycleInputBuffer;
}

CycleOutputBuffer& CycleController::getCycleOutputBuffer() {
    return cycleOutputBuffer;
}

void CycleController::setNonCoherentStimulationRate(ValueType rate) noexcept {
    nonCoherentStimulator.setRate(rate);
}

void CycleController::setDopamineReleaseBaseRate(ValueType rate) noexcept {
    dopaminergicModulator.setDopamineReleaseBaseRate(rate);
}

TimeType CycleController::getTimeIncrement() const noexcept {
    return dt;
}

}
