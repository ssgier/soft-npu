#include "SubCircuitAdapter.hpp"
#include <neuro/Population.hpp>
#include <neuro/ExplicitChannelProjector.hpp>
#include <genesis/NeuroComponentsFactory.hpp>
#include "ConnectivityParams.hpp"


namespace soft_npu {

SubCircuitAdapter::SubCircuitAdapter(
        NeuroComponentsFactory& factory,
        Population& population,
        ExplicitChannelProjector& channelProjector,
        RandomEngineType& randomEngine) :
    factory(factory),
    population(population),
    channelProjector(channelProjector),
    randomEngine(randomEngine) {
}

void SubCircuitAdapter::addNeurons(SizeType numNeurons, std::shared_ptr<const NeuronParams> neuronParams) {
    for (SizeType i = 0; i < numNeurons; ++i) {
        auto neuron = factory.makeNeuron(neuronParams, population);
        neurons.push_back(neuron.get());
        population.addNeuron(std::move(neuron), Population::defaultLocation);
    }
}

void SubCircuitAdapter::connectAllToInput(SizeType channelId, ValueType channelProjectedEpsp) const {
    for (auto neuron : neurons) {
        channelProjector.addSensoryNeuron(*neuron, {channelId}, channelProjectedEpsp);
    }
}

void SubCircuitAdapter::connectAllToOutput(SizeType channelId) const {
    for (auto neuron : neurons) {
        channelProjector.addMotorNeuron(*neuron, channelId);
    }
}

void SubCircuitAdapter::buildInternalConnnectivity(
        std::shared_ptr<const SynapseParams> synapseParams,
        const ConnectivityParams &connectivityParams) const {
    createProjectionOnto(*this, connectivityParams, synapseParams);
}

void SubCircuitAdapter::createProjectionOnto(
        const SubCircuitAdapter& target,
        const ConnectivityParams& connectivityParams,
        std::shared_ptr<const SynapseParams> synapseParams) const {

    SizeType numProjectionsPerCell = connectivityParams.connectDensity * target.neurons.size();
    auto targetNeuronsCopy = target.neurons;
    std::uniform_real_distribution<TimeType> conductionDelayDist(
            connectivityParams.minConductionDelay,
            connectivityParams.maxConductionDelay);

    for (auto preSynNeuron : neurons) {
        std::shuffle(targetNeuronsCopy.begin(), targetNeuronsCopy.end(), randomEngine);
        SizeType numPostSynNeuronsAdded = 0;
        for (auto postSynNeuron : targetNeuronsCopy) {

            if (numPostSynNeuronsAdded >= numProjectionsPerCell) {
                break;
            }

            if (preSynNeuron->getNeuronId() == postSynNeuron->getNeuronId()) {
                continue;
            }

            auto synapse = factory.makeSynapse(
                    synapseParams,
                    preSynNeuron,
                    postSynNeuron,
                    conductionDelayDist(randomEngine),
                    connectivityParams.initialWeight);

            preSynNeuron->addOutboundSynapse(synapse.get());
            if (preSynNeuron->getNeuronParams()->isInhibitory) {
                population.addInhibitorySynapse(std::move(synapse));
            } else {
                population.addExcitatorySynapse(std::move(synapse));
            }

            ++ numPostSynNeuronsAdded;
        }
    }
}

void SubCircuitAdapter::addInhibitionSource(Neuron &inhibitionSource) const {
    for (auto neuron : neurons) {
        neuron->addContinuousInhibitionSource(&inhibitionSource);
    }
}

void SubCircuitAdapter::addInhibitionSink(
    Neuron &inhibitionSink,
    const ConnectivityParams& connectivityParams,
    std::shared_ptr<const SynapseParams> synapseParams) const {

    if (connectivityParams.connectDensity < 1.0) {
        throw std::runtime_error("Adding inhibition sink with connect density < 1 is not defined");
    }

    std::uniform_real_distribution<TimeType> conductionDelayDist(
        connectivityParams.minConductionDelay,
        connectivityParams.maxConductionDelay);

    for (auto neuron : neurons) {

        auto synapse = factory.makeSynapse(
            synapseParams,
            neuron,
            &inhibitionSink,
            conductionDelayDist(randomEngine),
            connectivityParams.initialWeight);

        neuron->addOutboundSynapse(synapse.get());
        if (neuron->getNeuronParams()->isInhibitory) {
            population.addInhibitorySynapse(std::move(synapse));
        } else {
            population.addExcitatorySynapse(std::move(synapse));
        }
    }
}

SubCircuitAdapter::NeuronConstIterator SubCircuitAdapter::cbeginNeurons() const {
    return neurons.cbegin();
}

SubCircuitAdapter::NeuronConstIterator SubCircuitAdapter::cendNeurons() const {
    return neurons.cend();
}

}
