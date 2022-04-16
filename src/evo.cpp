#include <Aliases.hpp>
#include <util/FileUtil.hpp>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <experiments/EvolutionWrapper.hpp>

using namespace plog;
using namespace soft_npu;

int main() {
    ConsoleAppender<plog::TxtFormatter> consoleAppender;

    plog::init(plog::info, &consoleAppender);

    PLOG_INFO << "Application starting";

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

