#include "SynapseInjectionR2DSheet.hpp"
#include "NeuroComponentsFactory.hpp"
#include <boost/functional/hash.hpp>
#include <unordered_set>

namespace soft_npu::SynapseInjectionR2DSheet {

using GridLocation = std::pair<SizeType, SizeType>;

Population::Location getTargetLocation(
        RandomEngineType& randomEngine,
        const Population::Location& sourceLocation,
        bool distantLocation,
        ValueType minDistance,
        ValueType projectionRadius) {
    if (distantLocation) {

        std::uniform_real_distribution<ValueType> distribution;

        auto requiredDistance = minDistance + projectionRadius;

        for (auto i = 0; i < 1000; ++i) {
            auto x = distribution(randomEngine);
            auto y = distribution(randomEngine);
            Population::Location candidateLocation = {x, y};

            if (!PopulationUtils::isDistanceShorterThan(sourceLocation, candidateLocation, requiredDistance)) {
                return candidateLocation;
            }
        }

        throw std::runtime_error("Unable to get target location");
    } else {
        return sourceLocation;
    }
}

void doInjectSynapses(
        RandomEngineType& randomEngine,
        NeuroComponentsFactory& factory,
        Population& population,
        bool inhibitorySource,
        bool distantLocation,
        ValueType minDistance,
        ValueType projectionRadius,
        SizeType numTargets,
        TimeType conductionDelayDeterministicPart,
        TimeType conductionDelayRandomPart,
        ValueType initialWeight
        ) {


    std::uniform_real_distribution<TimeType> uniformDistribution;
    std::unordered_map<GridLocation, std::vector<Neuron*>, boost::hash<GridLocation>> gridLocationToNeurons;

    const static ValueType gridSpacingFactor = std::sqrt(2.0) / (std::sqrt(2.0) - 1);

    SizeType gridDim = std::floor(1.0 / (gridSpacingFactor * projectionRadius));
    ValueType gridSpacing = 1.0 / gridDim;

    for (auto neuronIt = population.cbeginNeurons(); neuronIt != population.cendNeurons(); ++ neuronIt) {

        auto sourceLocation = population.getCellLocation((*neuronIt)->getNeuronId());

        SizeType xGridLocation = (static_cast<SizeType>(std::floor(sourceLocation[0] / gridSpacing - 0.5)) + gridDim) % gridDim;
        SizeType yGridLocation = (static_cast<SizeType>(std::floor(sourceLocation[1] / gridSpacing - 0.5)) + gridDim) % gridDim;

        SizeType xGridNeighborLocation = (xGridLocation + 1) % gridDim;
        SizeType yGridNeighborLocation = (yGridLocation + 1) % gridDim;

        std::vector<GridLocation> relevantGridLocations = {
                {xGridLocation, yGridLocation},
                {xGridLocation, yGridNeighborLocation},
                {xGridNeighborLocation, yGridLocation},
                {xGridNeighborLocation, yGridNeighborLocation}
        };

        for (const auto& gridLocation : relevantGridLocations) {
            auto gridIt = gridLocationToNeurons.find(gridLocation);

            if (gridIt == gridLocationToNeurons.end()) {
                gridIt = gridLocationToNeurons.emplace_hint(gridIt, gridLocation, std::vector<Neuron*>());
            }

            gridIt->second.push_back((*neuronIt).get());
        }
    }

    for (auto neuronIt = population.cbeginNeurons(); neuronIt != population.cendNeurons(); ++ neuronIt) {

        auto& sourceNeuron = **neuronIt;

        if (inhibitorySource ^ sourceNeuron.getNeuronParams()->isInhibitory) {
            continue;
        }

        auto sourceLocation = population.getCellLocation(sourceNeuron.getNeuronId());
        auto targetLocation = getTargetLocation(randomEngine, sourceLocation, distantLocation, minDistance, projectionRadius);

        SizeType xGridLocation = static_cast<SizeType>(targetLocation[0] / gridSpacing) % gridDim;
        SizeType yGridLocation = static_cast<SizeType>(targetLocation[1] / gridSpacing) % gridDim;

        GridLocation targetGridLocation(xGridLocation, yGridLocation);

        auto gridIt = gridLocationToNeurons.find(targetGridLocation);

        if (gridIt != gridLocationToNeurons.end()) {
            auto& candidateNeurons = gridIt->second;

            std::vector<SizeType> indices(candidateNeurons.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), randomEngine);

            SizeType synapseCount = 0;
            for (SizeType index : indices) {
                if (synapseCount >= numTargets) {
                    break;
                }

                auto candidateLocation = population.getCellLocation(candidateNeurons[index]->getNeuronId());

                bool withinRange = PopulationUtils::isDistanceShorterThan(targetLocation, candidateLocation, projectionRadius);

                bool isEligible = withinRange && candidateNeurons[index]->getNeuronId() != sourceNeuron.getNeuronId();

                if (isEligible) {

                    auto targetNeuron = candidateNeurons[index];

                    TimeType conductionDelay = conductionDelayDeterministicPart +
                            uniformDistribution(randomEngine) * conductionDelayRandomPart;

                    auto synapse = factory.makeSynapse(&sourceNeuron, targetNeuron, conductionDelay, initialWeight);

                    sourceNeuron.addOutboundSynapse(synapse.get());

                    if (sourceNeuron.getNeuronParams()->isInhibitory) {
                        population.addInhibitorySynapse(std::move(synapse));
                    } else {
                        population.addExcitatorySynapse(std::move(synapse));
                    }

                    ++ synapseCount;
                }
            }
        }
    }
}

void injectSynapses(
        const soft_npu::ParamsType &params,
        RandomEngineType &randomEngine,
        NeuroComponentsFactory& factory,
        Population &population) {
    auto generatorParams = params["populationGenerators"]["r2dSheet"];
    ValueType pctExcLongDistanceTargets = generatorParams["pctExcLongDistanceTargets"];
    ValueType radiusExcShort = generatorParams["radiusExcShort"];
    ValueType radiusExcLong = generatorParams["radiusExcLong"];
    ValueType radiusInh = generatorParams["radiusInh"];

    SizeType numTargetsExc = generatorParams["numTargetsExc"];
    SizeType numTargetsInh = generatorParams["numTargetsInh"];

    TimeType maxExcConductionDelay = generatorParams["maxExcConductionDelay"];
    TimeType inhibitoryConductionDelayDeterministicPart = generatorParams["inhibitoryConductionDelayDeterministicPart"];
    TimeType inhibitoryConductionDelayRandomPart = generatorParams["inhibitoryConductionDelayRandomPart"];

    ValueType inhibitorySynapseWeight = generatorParams["inhibitorySynapseWeight"];
    ValueType excitatorySynapseInitialWeight = generatorParams["excitatorySynapseInitialWeight"];

    SizeType numDistantExcTargets = numTargetsExc * pctExcLongDistanceTargets;
    SizeType numNearbyExcTargets = numTargetsExc - numDistantExcTargets;

    doInjectSynapses(
            randomEngine,
            factory,
            population,
            false,
            false,
            0,
            radiusExcShort,
            numNearbyExcTargets,
            std::numeric_limits<ValueType>::epsilon(),
            maxExcConductionDelay - std::numeric_limits<ValueType>::epsilon(),
            excitatorySynapseInitialWeight);

    doInjectSynapses(
            randomEngine,
            factory,
            population,
            false,
            true,
            radiusExcShort,
            radiusExcLong,
            numDistantExcTargets,
            std::numeric_limits<ValueType>::epsilon(),
            maxExcConductionDelay - std::numeric_limits<ValueType>::epsilon(),
            excitatorySynapseInitialWeight);

    doInjectSynapses(
            randomEngine,
            factory,
            population,
            true,
            false,
            0,
            radiusInh,
            numTargetsInh,
            inhibitoryConductionDelayDeterministicPart,
            inhibitoryConductionDelayRandomPart,
            inhibitorySynapseWeight);
}

}
