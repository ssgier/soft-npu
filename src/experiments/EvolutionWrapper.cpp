#include <libcmaes/pwq_bound_strategy.h>
#include <libcmaes/genopheno.h>
#include <libcmaes/cmaparameters.h>
#include <libcmaes/cmasolutions.h>
#include <libcmaes/esostrategy.h>
#include <libcmaes/cmaes.h>
#include <libcmaes/pwq_bound_strategy.h>
#include <plog/Log.h>
#include "EvolutionWrapper.hpp"
#include "EvolutionWrapperResult.hpp"
#include "POCDynamicSimulation.hpp"
#include "evolution/Evolution.hpp"
#include <core/SimulationResult.hpp>
#include <string>

using namespace libcmaes;

namespace soft_npu {

std::vector<std::string> splitString(const std::string& string) {
    std::vector<std::string> rv;
    std::stringstream ss(string);
    std::string intermediate;

    while(std::getline(ss, intermediate, '.')) {
        rv.push_back(intermediate);
    }

    return rv;
}

template <typename T>
T& getRef(T& params, const std::vector<std::string>& path, SizeType pathPos) {
    if (pathPos == path.size()) {
        return params;
    } else {
        return getRef(params[path[pathPos]], path, pathPos + 1);
    }
}

ParamsType extractAtPath(const ParamsType& paramsTemplate, const std::string& path) {
    return getRef(paramsTemplate, splitString(path), 0);
}

ParamsType enrichGeneInfoTemplate(const ParamsType &paramsTemplate, const ParamsType &geneInfoTemplate) {

    if (geneInfoTemplate.is_object()) {

        PLOG_DEBUG << "Extracting from params template at path: " << geneInfoTemplate["id"];
        ParamsType prototypeValue = extractAtPath(paramsTemplate, geneInfoTemplate["id"]);

        ParamsType enrichedTemplate(geneInfoTemplate);

        if (prototypeValue.is_number_float()) {

            double minValue = geneInfoTemplate["minValue"];
            double maxValue = geneInfoTemplate["maxValue"];
            double prototypeValueToUse = std::min(maxValue, std::max(minValue, prototypeValue.get<double>()));

            enrichedTemplate["prototypeValue"] = prototypeValueToUse;

        } else if (prototypeValue.is_number_integer()) {
            int minValue = geneInfoTemplate["minValue"];
            int maxValue = geneInfoTemplate["maxValue"];
            int prototypeValueToUse = std::min(maxValue, std::max(minValue, prototypeValue.get<int>()));

            enrichedTemplate["prototypeValue"] = prototypeValueToUse;

        } else if (prototypeValue.is_boolean()) {
            enrichedTemplate["prototypeValue"] = prototypeValue;
        } else {
            std::stringstream ss;
            ss << "Unsupported type of prototype value, value: " << prototypeValue;
            throw std::runtime_error(ss.str());
        }

        return enrichedTemplate;
    } else if (geneInfoTemplate.is_array()) {

        ParamsType enrichedTemplate;

        std::transform(geneInfoTemplate.cbegin(), geneInfoTemplate.cend(), std::back_inserter(enrichedTemplate),
                       [&paramsTemplate](auto subObject) {
            return enrichGeneInfoTemplate(paramsTemplate, subObject);
        });

        return enrichedTemplate;
    } else {
        std::stringstream ss;
        ss << "Unsupported type of template, value: " << geneInfoTemplate;
        throw std::runtime_error(ss.str());
    }
}

ParamsType extractParamsFromFlatValueJson(const ParamsType& paramsTemplate, const ParamsType& flatValueJson) {

    ParamsType mergedParams(paramsTemplate);
    for (auto& [id, value] : flatValueJson.items()) {
        auto& paramValue = getRef(mergedParams, splitString(id), 0);
        paramValue = value;
    }

    return mergedParams;
}

double evaluateSingle(const ParamsType& params_, int seed, double simulationTime) {
    auto params = std::make_shared<ParamsType>(params_);
    (*params)["simulation"]["untilTime"] = simulationTime;
    (*params)["simulation"]["seed"] = seed;
    (*params)["pocDynamicSimulation"]["flipDetectorChannels"] = seed % 2 == 0;

    POCDynamicSimulation simulation(params);

    simulation.run();

    return simulation.optimResultHolder.objFuncVal;
}

double evaluateFitnessFunction(const ParamsType& params, double simulationTime) {
    int numSeedsPerCandidate = 10;

    std::vector<SizeType> seeds(numSeedsPerCandidate);
    std::iota(seeds.begin(), seeds.end(), 10);

    std::vector<double> individualObjVals(seeds.size());

    for (int i = 0; i < numSeedsPerCandidate; ++i) {
        individualObjVals[i] = evaluateSingle(params, seeds[i], simulationTime);
    }

    double avgObjValue = std::accumulate(individualObjVals.cbegin(), individualObjVals.cend(), 0.0) / individualObjVals.size();

    return avgObjValue;
}

EvolutionWrapperResult EvolutionWrapper::run(const ParamsType &paramsTemplate, const ParamsType &geneInfoTemplate,
                                             const EvolutionParams &evolutionParams) {
    auto geneInfoJson = enrichGeneInfoTemplate(paramsTemplate, geneInfoTemplate);

    double simulationTime = paramsTemplate["simulation"]["untilTime"].get<double>();

    auto proxyFitnessFunction = [](auto) {
        return -1;
    };

    auto mainFitnessFunction = [&paramsTemplate, simulationTime](auto flatValueJson) {
        return evaluateFitnessFunction(extractParamsFromFlatValueJson(paramsTemplate, flatValueJson), simulationTime);
    };

    auto evolutionResult = Evolution::run(
        evolutionParams,
        proxyFitnessFunction,
        mainFitnessFunction,
        geneInfoJson
        );

    EvolutionWrapperResult result;
    result.evolvedParams = std::make_shared<ParamsType>(
        extractParamsFromFlatValueJson(paramsTemplate, *evolutionResult.topGeneJson));
    result.fitnessValue = evolutionResult.topFitnessValue;

    return result;
}

EvolutionParams extractFromBuffer(const double* buf) {
    EvolutionParams evolutionParams;

    evolutionParams.elitePopulationSize = std::max(1, static_cast<int>(buf[0]));
    evolutionParams.mainPopulationSize = evolutionParams.elitePopulationSize * buf[1];
    evolutionParams.proxyPopulationSize = evolutionParams.mainPopulationSize + 1;
    evolutionParams.minMutationProbability = buf[2];
    evolutionParams.maxMutationProbability = buf[2] + (1 - buf[2]) * buf[3];
    evolutionParams.minMutationStrength = buf[4];
    evolutionParams.maxMutationStrength = buf[4] + (1 - buf[4]) * buf[5];
    evolutionParams.crossoverProbability = buf[6];

    return evolutionParams;
}

EvolutionParams
EvolutionWrapper::calibrate(const ParamsType &paramsTemplate, const ParamsType &geneInfoTemplate,
                            double abortTimeSeconds) {

    // TODO: move numbers to config

    std::vector<double> x0 = {5, 10, 0.0, 0.9, 0.0, 0.45, 0.5};
    std::vector<double> lbounds = {1, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> ubounds = {20, 20, 1.0, 1.0, 1.0, 1.0, 1.0};

    GenoPheno<pwqBoundStrategy> gp(lbounds.data(), ubounds.data(), x0.size());
    CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(x0, 0.1, -1, 1, gp);

    cmaparams.set_max_fevals(45);
    cmaparams.set_seed(0);
    cmaparams.set_quiet(false);
    cmaparams.set_algo(aCMAES);

    FitFunc objFunc = [&paramsTemplate, &geneInfoTemplate, abortTimeSeconds](const double* x, const int) {
        EvolutionParams evolutionParams = extractFromBuffer(x);
        evolutionParams.abortAfterSeconds = abortTimeSeconds;
        evolutionParams.targetFitnessValue = std::numeric_limits<double>::lowest();
        evolutionParams.maxNumIterations = -1;

        double objVal = run(paramsTemplate, geneInfoTemplate, evolutionParams).fitnessValue;

        PLOG_INFO << "Function eval completed. Objective function value: " << objVal;

        return objVal;
    };

    CMASolutions cmasols = cmaes<>(objFunc,cmaparams);
    auto bestCandidate = cmasols.get_best_seen_candidate();

    EvolutionParams evolutionParams = extractFromBuffer(gp.pheno(bestCandidate.get_x_dvec()).transpose().data());

    return evolutionParams;
}


}
