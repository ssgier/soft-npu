#pragma once

#include "Aliases.hpp"
#include <boost/core/noncopyable.hpp>

namespace soft_npu {

class Population;
class ExplicitChannelProjector;
class Neuron;
class NeuroComponentsFactory;
struct NeuronParams;
struct SynapseParams;
struct ConnectivityParams;

class SubCircuitAdapter : private boost::noncopyable {

public:
    using NeuronConstIterator = std::vector<Neuron*>::const_iterator;

    SubCircuitAdapter(
            NeuroComponentsFactory&,
            Population&,
            ExplicitChannelProjector&,
            RandomEngineType&);

    void addNeurons(SizeType numberOfNeurons, std::shared_ptr<const NeuronParams> neuronParams);

    void createProjectionOnto(
            const SubCircuitAdapter& target,
            const ConnectivityParams& connectivityParams,
            std::shared_ptr<const SynapseParams>) const;

    void buildInternalConnnectivity(
            std::shared_ptr<const SynapseParams> synapseParams,
            const ConnectivityParams& connectivityParams) const;

    void connectAllToInput(SizeType channelId, ValueType channelProjectedEpsp) const;
    void connectAllToOutput(SizeType channelId) const;
    void addInhibitionSource(Neuron& inhibitionSource) const;
    void addInhibitionSink(
        Neuron& inhibitionSink,
        const ConnectivityParams&,
        std::shared_ptr<const SynapseParams>) const;

    NeuronConstIterator cbeginNeurons() const;
    NeuronConstIterator cendNeurons() const;
private:
    NeuroComponentsFactory& factory;
    Population& population;
    ExplicitChannelProjector& channelProjector;
    RandomEngineType& randomEngine;
    std::vector<Neuron*> neurons;
};
}



