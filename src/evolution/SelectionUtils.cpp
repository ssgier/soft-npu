#include "SelectionUtils.hpp"
#include "EvolutionParams.hpp"

namespace soft_npu {

SizeType SelectionUtils::selectParentIndex(
        const EvolutionParams& evolutionParams,
        RandomEngineType& randomEngine) {

    SizeType candidateIdx0;
    SizeType candidateIdx1;

    std::uniform_int_distribution<SizeType> indexDistribution(0, evolutionParams.populationSize - 1);
    std::bernoulli_distribution bernoulliDistribution(evolutionParams.tournamentSelectionProbability);

    do {
        candidateIdx0 = indexDistribution(randomEngine);
        candidateIdx1 = indexDistribution(randomEngine);
    } while (candidateIdx0 == candidateIdx1);

    if (candidateIdx1 < candidateIdx0) {
        std::swap(candidateIdx0, candidateIdx1);
    }

    return bernoulliDistribution(randomEngine) ? candidateIdx0 : candidateIdx1; 
}

}
