#pragma once

#include <vector>
#include <neuro/Population.hpp>
#include "NeuronSpikeInfo.hpp"
#include "SynapseInfo.hpp"
#include "NeuronInfo.hpp"
#include "VoltageRecording.hpp"

namespace soft_npu {

struct SimulationResult {
    SimulationResult(
            TimeType simulationTime,
            std::vector<NeuronSpikeInfo> recordedSpikes,
            std::vector<VoltageRecording> voltageRecordings,
            std::vector<NeuronInfo> neuronInfos,
            std::vector<SynapseInfo> finalSynapseInfos,
            std::vector<Population::Location> locationsIndexedByNeuronId,
            SizeType numExcitatorySpikes,
            SizeType numInhibitorySpikes,
            SizeType numSynapticTransmissions,
            double meanExcitatoryFiringRate,
            double meanInhibitoryFiringRate,
            double wallTimeTotal,
            double wallTimeEventProcessor,
            uint64_t numEventsProcessed);

    TimeType simulationTime;
    const std::vector<NeuronSpikeInfo> recordedSpikes;
    const std::vector<VoltageRecording> voltageRecordings;
    const std::vector<NeuronInfo> neuronInfos;
    const std::vector<SynapseInfo> finalSynapseInfos;
    const std::vector<Population::Location> locationsIndexedByNeuronId;
    const SizeType numExcitatorySpikes;
    const SizeType numInhibitorySpikes;
    const double meanExcitatoryFiringRate;
    const double meanInhibitoryFiringRate;

    const TimeType wallTimeTotal;
    const TimeType wallTimeEventProcessor;

    const uint64_t numEventsProcessed;
    const double eventThroughput;
    const double spikeProcessingThroughput;
    const double synapticTransmissionProcessingThroughput;
};

std::ostream& operator<<(std::ostream& os, const SimulationResult&);

}