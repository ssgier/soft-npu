
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <util/FileUtil.hpp>
#include <core/StaticInputSimulation.hpp>

using namespace plog;
using namespace soft_npu;

int main()
{
    ConsoleAppender<plog::TxtFormatter> consoleAppender;

    plog::init(plog::debug, &consoleAppender);

    PLOG_INFO << "Application starting";

    auto params = std::make_shared<ParamsType>(ParamsType::parse(FileUtil::getFileContent(
            "../resources/benchmarkParams.json")));

    StaticInputSimulation simulation(params);

    auto simulationResult = simulation.run();
    PLOG_INFO << simulationResult;

    PLOG_INFO << "Writing spike trains to csv";
    FileUtil::writeSpikeTrainsToCSV("spikeTrains.csv", simulationResult.recordedSpikes);
    FileUtil::writeSynapseInfosToCSV("synapseInfos.csv", simulationResult.finalSynapseInfos);
    FileUtil::writeLocationsToCSV("locations.csv", simulationResult.locationsIndexedByNeuronId);
    FileUtil::writeNeuronInfosToCSV("neuronInfos.csv", simulationResult.neuronInfos);

    PLOG_INFO << "Terminating";

    return 0;
}
