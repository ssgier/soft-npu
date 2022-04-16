#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <memory>
#include <util/FileUtil.hpp>
#include <genesis/SingleNeuronPopulationGenerator.hpp>
#include "core/StaticInputSimulation.hpp"

using namespace plog;
using namespace soft_npu;

int main()
{
    ConsoleAppender<plog::TxtFormatter> consoleAppender;

    plog::init(plog::debug, &consoleAppender);

    PLOG_INFO << "Benchmark starting";

    auto params = std::make_shared<ParamsType>(ParamsType::parse(FileUtil::getFileContent(
            "../resources/benchmarkParams.json")));

    int numRuns = 10;
    double aggSynTransmissionProcThroughput = 0;

    for (int i = 0; i < numRuns; ++i) {
        StaticInputSimulation simulation(params);
        auto simulationResult = simulation.run();
        PLOG_INFO << simulationResult;
        aggSynTransmissionProcThroughput += simulationResult.synapticTransmissionProcessingThroughput;
    }

    PLOG_INFO << "Mean synaptic transmission processing throughput: " << aggSynTransmissionProcThroughput / numRuns;
    PLOG_INFO << "Terminating";
}
