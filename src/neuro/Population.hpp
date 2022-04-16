#pragma once

#include "SpikeListener.hpp"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <neuro/SynapseParams.hpp>
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

struct ChannelProjector;
class Synapse;

class Population : private boost::noncopyable {
public:

    using Location = std::array<ValueType, 2>;
    static constexpr Location defaultLocation = {0, 0};
    using neuron_ptr_const_iterator = std::vector<std::unique_ptr<Neuron>>::const_iterator;

    explicit Population(const ParamsType&);
    explicit Population(const SynapseParams&);

    void onSpike(const CycleContext& cycleContext, const Neuron& neuron) const;

    template<typename T>
    SizeType addSpikeListener(T&& processingFunction) {
        auto spikeListenerId = nextSpikeListenerId ++;
        spikeListenersById[spikeListenerId] = std::make_unique<SpikeListenerImpl<T>>(std::forward<T>(processingFunction));
        return spikeListenerId;
    }

    std::unordered_set<SizeType> getMotorNeuronIds() const;
    void projectChannelSpike(const CycleContext& ctx, SizeType channelId) const;

    void removeSpikeListener(SizeType spikeListenerId);
    void addNeuron(std::unique_ptr<Neuron> neuron, Location location);
    void addExcitatorySynapse(std::unique_ptr<Synapse>);
    void addInhibitorySynapse(std::unique_ptr<Synapse>);
    void setChannelProjector(std::unique_ptr<const ChannelProjector>);

    neuron_ptr_const_iterator cbeginNeurons() const noexcept {
        return neuronsIndexedById.cbegin();
    }

    neuron_ptr_const_iterator cendNeurons() const noexcept {
        return neuronsIndexedById.cend();
    }

    Neuron& getNeuronById(SizeType neuronId) const;

    SizeType getPopulationSize() const;
    Location getCellLocation(SizeType neuronId) const;

    const SynapseParams& getSynapseParams() const;

private:
    SynapseParams synapseParams;
    std::vector<std::unique_ptr<Neuron>> neuronsIndexedById;
    std::vector<Location> locationsIndexedByNeuronId;
    std::vector<std::unique_ptr<Synapse>> excitatorySynapses;
    std::vector<std::unique_ptr<Synapse>> inhibitorySynapses;
    std::unique_ptr<const ChannelProjector> channelProjector;

    std::unordered_map<SizeType, std::unique_ptr<SpikeListener>> spikeListenersById;
    SizeType nextSpikeListenerId = 0;
};

namespace PopulationUtils {

SizeType getNumInhibitoryNeurons(const Population& population);
ValueType getDistance(const Population::Location&, const Population::Location&);
bool isDistanceShorterThan(const Population::Location&, const Population::Location&, ValueType distance);

}

}
