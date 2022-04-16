#include <libcmaes/pwq_bound_strategy.h>
#include <libcmaes/genopheno.h>
#include <libcmaes/cmaparameters.h>
#include <libcmaes/cmasolutions.h>
#include <libcmaes/esostrategy.h>
#include <libcmaes/cmaes.h>
#include <plog/Log.h>
#include "Optimizer.hpp"
#include "OptimParamsMapper.hpp"
#include "POCDynamicSimulation.hpp"
#include <core/SimulationResult.hpp>

using namespace libcmaes;

namespace soft_npu {

Optimizer::Optimizer(std::shared_ptr<const ParamsType> templateParams, SizeType populationSize, SizeType numSeedsPerCandidate,
                     ValueType targetObjValue) :
        templateParams(templateParams), populationSize(populationSize),
        numSeedsPerCandidate(numSeedsPerCandidate), targetObjValue(targetObjValue) {

}

std::shared_ptr<const ParamsType> Optimizer::optimize() const {
    OptimParamsMapper optimParamsMapper(templateParams);

    auto numSeedsPerCandidateCopy = numSeedsPerCandidate;

    FitFunc objFunc = [&optimParamsMapper, numSeedsPerCandidateCopy](const double *x, const int)
    {

        std::vector<SizeType> seeds(numSeedsPerCandidateCopy);
        std::iota(seeds.begin(), seeds.end(), 10);
        std::vector<double> individualObjVals(seeds.size());

        for (SizeType i = 0; i < numSeedsPerCandidateCopy; ++i) {
            auto params = optimParamsMapper.getParams(x);
            (*params)["simulation"]["seed"] = seeds[i];
            (*params)["pocDynamicSimulation"]["flipDetectorChannels"] = i % 2 == 0;

            POCDynamicSimulation simulation(params);

            auto startTs = std::chrono::high_resolution_clock::now();
            simulation.run();
            auto stopTs = std::chrono::high_resolution_clock::now();

            if (simulation.optimResultHolder.objFuncVal == std::numeric_limits<double>::max()) {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTs - startTs);
                PLOG_INFO << "High obj val simulation took " << duration.count() << " ms";
            }

            individualObjVals[i] = simulation.optimResultHolder.objFuncVal;
        }

        double avgObjValue = std::accumulate(individualObjVals.cbegin(), individualObjVals.cend(), 0.0) / individualObjVals.size();

        PLOG_INFO << "Mean of obj function value: " << avgObjValue;

        return avgObjValue;
    };

    std::vector<double> x0 = optimParamsMapper.getInitialValues();



    double sigma = 0.1;
    std::vector<double> lbounds(x0.size(), 0.0);
    std::vector<double> ubounds(x0.size(), 1.0);
    GenoPheno<pwqBoundStrategy> gp(lbounds.data(), ubounds.data(), x0.size());
    CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(x0, sigma, populationSize, 1, gp);

    cmaparams.set_mt_feval(true);
    cmaparams.set_ftarget(targetObjValue);
    cmaparams.set_fplot("fplot.dat");
    cmaparams.set_algo(aCMAES);

    CMASolutions cmasols = cmaes<>(objFunc,cmaparams);
    Candidate bestCandidate = cmasols.get_best_seen_candidate();

    PLOG_INFO << "Optimization completed, status = " << cmasols.run_status() << ", objective function value = " << bestCandidate.get_fvalue();

    auto outParams = optimParamsMapper.getParams(gp.pheno(bestCandidate.get_x_dvec()).transpose().data());
    return outParams;
}
}
