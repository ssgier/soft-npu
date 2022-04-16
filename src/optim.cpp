
#include <Aliases.hpp>
#include <util/FileUtil.hpp>
#include <omp.h>

#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <experiments/Optimizer.hpp>

using namespace plog;
using namespace soft_npu;

int main() {
    ConsoleAppender<plog::TxtFormatter> consoleAppender;

    plog::init(plog::info, &consoleAppender);
    omp_set_nested(2);

    PLOG_INFO << "Application starting";

    SizeType populationSize = 20;
    SizeType numSeedsPerCandidate = 10;
    ValueType targetObjValue = 0.27;

    auto templateParams = std::make_shared<ParamsType>(ParamsType::parse(FileUtil::getFileContent(
            "../resources/paramsTemplate.json")));

    auto outParams = Optimizer(templateParams, populationSize, numSeedsPerCandidate, targetObjValue).optimize();

    std::ofstream fout("../resources/outParamsCmaes.json");
    fout << outParams->dump();
    fout.close();
}

