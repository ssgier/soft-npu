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

std::string getUnsupportedTypeErrorMsg(const ParamsType& value) {
    std::stringstream ss;
    ss << "Unsupported type: " << value.type_name() << " for value: " << value;
    throw std::runtime_error(ss.str());
}

std::shared_ptr<GeneElement> extractGeneElement(const ParamsType& leafGeneInfoValue) {
    std::string id = leafGeneInfoValue["id"];
    const auto& prototypeValue = leafGeneInfoValue["prototypeValue"];

    if (prototypeValue.is_boolean()) {
        return std::make_shared<BooleanGeneElement>(id, prototypeValue.get<bool>());
    } else if (prototypeValue.is_number_float()) {

        double minValue = leafGeneInfoValue["minValue"].get<double>();
        double maxValue = leafGeneInfoValue["maxValue"].get<double>();
        double prototypeValueDouble = prototypeValue.get<double>();

        return std::make_shared<RealNumberGeneElement>(id, minValue, maxValue, prototypeValueDouble);
    } else if (prototypeValue.is_number_integer()) {
        int minValue = leafGeneInfoValue["minValue"].get<int>();
        int maxValue = leafGeneInfoValue["maxValue"].get<int>();
        int prototypeValueInt = prototypeValue.get<int>();

        return std::make_shared<IntegerGeneElement>(id, minValue, maxValue, prototypeValueInt);
    }
    else {
        throw std::runtime_error(getUnsupportedTypeErrorMsg(prototypeValue));
    }
}

std::shared_ptr<const Gene> assembleFromInfo(const ParamsType& geneInfoValue) {
    if (geneInfoValue.is_object()) {
        return std::make_shared<Gene>(extractGeneElement(geneInfoValue));
    } else if (geneInfoValue.is_array()) {
        GeneVector geneVector;

        std::transform(geneInfoValue.cbegin(), geneInfoValue.cend(), std::back_inserter(geneVector), [](auto subInfo) {
            return assembleFromInfo(subInfo);
        });

        return std::make_shared<Gene>(std::move(geneVector));
    } else {
        throw std::runtime_error(getUnsupportedTypeErrorMsg(geneInfoValue));
    }
}

void extractFlatGeneValueRecursionHelper(const Gene& gene, ParamsType& workingData) {
    if (gene.isLeaf()) {
        auto element = gene.getGeneElement();
        workingData[element->getId()] = element->value();
    } else {
        std::for_each(gene.cbeginSubGenes(), gene.cendSubGenes(), [&workingData](auto subGene) {
            extractFlatGeneValueRecursionHelper(*subGene, workingData);
        });
    }
}

ParamsType extractFlatGeneValue(const Gene &gene) {
    ParamsType rv;
    extractFlatGeneValueRecursionHelper(gene, rv);
    return rv;
}

}
