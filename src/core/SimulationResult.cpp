
#include <ostream>
#include "SimulationResult.hpp"

namespace soft_npu {

std::ostream& operator<<(std::ostream& os, const SimulationResult& simulationResult) {

    os << "Simulation result:" << std::endl << std::endl
       << "Simulation time: " << simulationResult.simulationTime << " s" << std::endl
       << "Number of excitatory spikes: " << simulationResult.numExcitatorySpikes << std::endl
       << "Number of inhibitory spikes: " << simulationResult.numInhibitorySpikes << std::endl
       << "Mean excitatory firing rate: " << simulationResult.meanExcitatoryFiringRate << " Hz" << std::endl
       << "Mean inhibitory firing rate: " << simulationResult.meanInhibitoryFiringRate << " Hz" << std::endl
       << "Total wall time: " << simulationResult.wallTimeTotal << " s" << std::endl
       << "Wall time spent processing events: " << simulationResult.wallTimeEventProcessor << " s" << std::endl
       << "Number of events processed: " << simulationResult.numEventsProcessed << std::endl
       << "Event throughput: " << simulationResult.eventThroughput << " per second" << std::endl
        << "Spike processing throughput: " << simulationResult.spikeProcessingThroughput << " per second" << std::endl
        << "Synaptic transmission processing throughput: " << simulationResult.synapticTransmissionProcessingThroughput << " per second" << std::endl;

    return os;
}

SimulationResult::SimulationResult(TimeType simulationTime, std::vector<NeuronSpikeInfo> recordedSpikes,
                                   std::vector<VoltageRecording> voltageRecordings,
                                   std::vector<NeuronInfo> neuronInfos,
                                   std::vector<SynapseInfo> finalSynapseInfos,
                                   std::vector<Population::Location> locationsIndexedByNeuronId,
                                   SizeType numExcitatorySpikes,
                                   SizeType numInhibitorySpikes,
                                   SizeType numSynapticTransmissions,
                                   double meanExcitatoryFiringRate,
                                   double meanInhibitoryFiringRate, double wallTimeTotal, double wallTimeEventProcessor,
                                   uint64_t numEventsProcessed) :

        simulationTime(simulationTime),
        recordedSpikes(std::move(recordedSpikes)),
        voltageRecordings(std::move(voltageRecordings)),
        neuronInfos(std::move(neuronInfos)),
        finalSynapseInfos(std::move(finalSynapseInfos)),
        locationsIndexedByNeuronId(std::move(locationsIndexedByNeuronId)),
        numExcitatorySpikes(numExcitatorySpikes),
        numInhibitorySpikes(numInhibitorySpikes),
        meanExcitatoryFiringRate(meanExcitatoryFiringRate),
        meanInhibitoryFiringRate(meanInhibitoryFiringRate),

        wallTimeTotal(wallTimeTotal),
        wallTimeEventProcessor(wallTimeEventProcessor),
        numEventsProcessed(numEventsProcessed),
        eventThroughput(numEventsProcessed / wallTimeEventProcessor),
        spikeProcessingThroughput((numExcitatorySpikes + numInhibitorySpikes) / wallTimeEventProcessor),
        synapticTransmissionProcessingThroughput(numSynapticTransmissions / wallTimeEventProcessor)
{}
}