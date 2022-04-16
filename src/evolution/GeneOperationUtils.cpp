#include <sstream>
#include "GeneOperationUtils.hpp"
#include "GeneOperationUtilsImpl.hpp"
#include "BooleanGeneElement.hpp"
#include "RealNumberGeneElement.hpp"
#include "IntegerGeneElement.hpp"
#include <random>
#include "EvolutionParams.hpp"

namespace soft_npu::GeneOperationUtils {

std::shared_ptr<const Gene> crossover(
    const EvolutionParams& evolutionParams,
    const GeneVector& genes,
    RandomEngineType& randomEngine
) {
    return GeneOperationUtilsImpl::crossover<std::uniform_int_distribution<size_t>, std::uniform_real_distribution<double>>(
        evolutionParams, genes, randomEngine);
}

std::string getUnsupportedTypeErrorMsg(const nlohmann::json& json) {
    std::stringstream ss;
    ss << "Unsupported json type: " << json.type_name() << " for json: " << json;
    throw std::runtime_error(ss.str());
}

std::shared_ptr<GeneElement> extractGeneElement(const nlohmann::json& leafGeneInfoJson) {
    std::string id = leafGeneInfoJson["id"];
    const auto& prototypeValueJson = leafGeneInfoJson["prototypeValue"];

    if (prototypeValueJson.is_boolean()) {
        return std::make_shared<BooleanGeneElement>(id, prototypeValueJson.get<bool>());
    } else if (prototypeValueJson.is_number_float()) {

        double minValue = leafGeneInfoJson["minValue"].get<double>();
        double maxValue = leafGeneInfoJson["maxValue"].get<double>();
        double prototypeValue = prototypeValueJson.get<double>();

        return std::make_shared<RealNumberGeneElement>(id, minValue, maxValue, prototypeValue);
    } else if (prototypeValueJson.is_number_integer()) {
        int minValue = leafGeneInfoJson["minValue"].get<int>();
        int maxValue = leafGeneInfoJson["maxValue"].get<int>();
        int prototypeValue = prototypeValueJson.get<int>();

        return std::make_shared<IntegerGeneElement>(id, minValue, maxValue, prototypeValue);
    }
    else {
        throw std::runtime_error(getUnsupportedTypeErrorMsg(prototypeValueJson));
    }
}

std::shared_ptr<const Gene> assembleFromJson(const nlohmann::json& geneInfoJson) {
    if (geneInfoJson.is_object()) {
        return std::make_shared<Gene>(extractGeneElement(geneInfoJson));
    } else if (geneInfoJson.is_array()) {
        GeneVector geneVector;

        std::transform(geneInfoJson.cbegin(), geneInfoJson.cend(), std::back_inserter(geneVector), [](auto subJson) {
            return assembleFromJson(subJson);
        });

        return std::make_shared<Gene>(std::move(geneVector));
    } else {
        throw std::runtime_error(getUnsupportedTypeErrorMsg(geneInfoJson));
    }
}

void extractFlatGeneValueJsonRecursionHelper(const Gene& gene, nlohmann::json& workingData) {
    if (gene.isLeaf()) {
        auto element = gene.getGeneElement();
        workingData[element->getId()] = element->valueAsJson();
    } else {
        std::for_each(gene.cbeginSubGenes(), gene.cendSubGenes(), [&workingData](auto subGene) {
            extractFlatGeneValueJsonRecursionHelper(*subGene, workingData);
        });
    }
}

nlohmann::json extractFlatGeneValueJson(const Gene &gene) {
    nlohmann::json rv;
    extractFlatGeneValueJsonRecursionHelper(gene, rv);
    return rv;
}

}
