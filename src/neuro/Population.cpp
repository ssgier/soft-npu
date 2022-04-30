#include "Population.hpp"
#include "Synapse.hpp"
#include <params/ParamsFactories.hpp>
#include "neuro/ChannelProjector.hpp"

namespace soft_npu {

void Population::onSpike(const CycleContext& cycleContext, const Neuron& neuron) const {

    if (channelProjector != nullptr) {
        channelProjector->projectNeuronSpike(cycleContext.staticContext.cycleOutputBuffer, neuron);
    }

    for (auto& entry : spikeListenersById) {
        entry.second->onSpike(cycleContext, neuron);
    }
}

void Population::addExcitatorySynapse(std::unique_ptr<Synapse> synapse) {
    excitatorySynapses.push_back(std::move(synapse));
}

void Population::addInhibitorySynapse(std::unique_ptr<Synapse> synapse) {
    inhibitorySynapses.push_back(std::move(synapse));
}

void Population::addNeuron(std::unique_ptr<Neuron> neuron, Location location) {

    if (neuron->getNeuronId() != neuronsIndexedById.size()) {
        throw std::runtime_error(
                "Neurons must be added to population with contiguously increasing neuron ids");
    }

    neuronsIndexedById.push_back(std::move(neuron));
    locationsIndexedByNeuronId.push_back(location);
}

SizeType Population::getPopulationSize() const {
    return neuronsIndexedById.size();
}

Neuron& Population::getNeuronById(SizeType neuronId) const {
    return *neuronsIndexedById[neuronId];
}

Population::Location Population::getCellLocation(SizeType neuronId) const {
    return locationsIndexedByNeuronId[neuronId];
}

void Population::removeSpikeListener(SizeType spikeListenerId) {
    spikeListenersById.erase(spikeListenerId);
}

void Population::projectChannelSpike(const CycleContext &ctx, SizeType channelId) const {
    channelProjector->projectChannelSpike(ctx, channelId);
}

std::unordered_set<SizeType> Population::getMotorNeuronIds() const {
    return channelProjector->getMotorNeuronIds();
}

void Population::setChannelProjector(std::unique_ptr<const ChannelProjector> channelProjector) {
    this->channelProjector = std::move(channelProjector);
}


namespace PopulationUtils {

SizeType getNumInhibitoryNeurons(const Population& population) {
    SizeType count = 0;
    for (auto it = population.cbeginNeurons(); it != population.cendNeurons(); ++it) {
        if ((*it)->getNeuronParams()->isInhibitory) {
            ++ count;
        }
    }

    return count;
}

inline ValueType get1DSquaredDistance(ValueType a, ValueType b) {
    if (b < a) {
        std::swap(a, b);
    }

    auto diff = std::min(b - a, a + 1 - b);

    return diff * diff;
}

inline ValueType getSquaredDistance(const Population::Location& l0, const Population::Location & l1) {
    return get1DSquaredDistance(l0[0], l1[0]) + get1DSquaredDistance(l0[1], l1[1]);
}

ValueType getDistance(const Population::Location& l0, const Population::Location & l1) {
    return std::sqrt(getSquaredDistance(l0, l1));
}

bool isDistanceShorterThan(const Population::Location& l0, const Population::Location& l1, ValueType distance) {
    return getSquaredDistance(l0, l1) < distance * distance;
}

}

}
