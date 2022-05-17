#include <Aliases.hpp>
#include <util/FileUtil.hpp>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <experiments/EvolutionWrapper.hpp>
#include <experiments/POCDynamicSimulation.hpp>

using namespace plog;
using namespace soft_npu;

void runEvolution(const std::string& paramsTemplatePath) {
    ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    PLOG_INFO << "Running evolution, params template file path: " << paramsTemplatePath;

    auto paramsTemplate = ParamsType::parse(FileUtil::getFileContent(
        "../resources/paramsTemplate.json"));

    auto geneInfoTemplate = ParamsType::parse(FileUtil::getFileContent(
        "../resources/geneInfoTemplate.json"));

    EvolutionParams evolutionParams;

    auto outParams = EvolutionWrapper::run(paramsTemplate, geneInfoTemplate, evolutionParams);

    std::ofstream fout("../resources/outParams.json");
    fout << outParams.evolvedParams->dump(4);
    fout.close();
}

void runRepro() {
    ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::debug, &consoleAppender);

    PLOG_INFO << "Running reproduction";

    auto params = std::make_shared<ParamsType>(ParamsType::parse(FileUtil::getFileContent(
            "../resources/outParams.json")));

    POCDynamicSimulation simulation(params);

    auto simulationResult = simulation.run();
    PLOG_INFO << simulationResult;
}

int main(int argc, char * argv[]) {

    if (argc == 1) {
        runEvolution("../resources/paramsTemplate.json");
    } else if (argc == 2 && strcmp("--resume", argv[1]) == 0) {
        runEvolution("../resources/outParams.json");
    } else if (argc == 2 && strcmp("--repro", argv[1]) == 0) {
        runRepro();
    } else {
        throw std::runtime_error("Invalid program arguments");
    }
}

