#include "PopulationGeneratorFactory.hpp"
#include "PopulationGeneratorP1000.hpp"
#include "PopulationGeneratorDetailedParams.hpp"
#include "PopulationGeneratorR2DSheet.hpp"
#include "PopulationGeneratorEvo.hpp"

namespace soft_npu::PopulationGeneratorFactory {

std::unique_ptr<PopulationGenerator> createFromParams(
        const ParamsType& params,
        RandomEngineType& randomEngine) {
    std::string populationGeneratorName = params["simulation"]["populationGenerator"];

    if (populationGeneratorName == "SingleNeuron") {
        return std::make_unique<SingleNeuronPopulationGenerator>(params, randomEngine);
    } else if (populationGeneratorName == "p1000") {
        return std::make_unique<PopulationGeneratorP1000>(params, randomEngine);
    } else if (populationGeneratorName == "pEvo") {
        return std::make_unique<PopulationGeneratorEvo>(params, randomEngine);
    } else if (populationGeneratorName == "r2dSheet") {
        return std::make_unique<PopulationGeneratorR2DSheet>(params, randomEngine);
    } else if (populationGeneratorName == "pDetailedParams") {
        return std::make_unique<PopulationGeneratorDetailedParams>(params, randomEngine);
    } else {
        throw std::runtime_error("Invalid population generator name: " + populationGeneratorName);
    }
}

}
