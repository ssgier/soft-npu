#include "CloningPopulationGenerator.hpp"
#include "NeuroComponentsFactory.hpp"
#include "TrivialNeuroComponentsFactory.hpp"

namespace soft_npu {


CloningPopulationGenerator::CloningPopulationGenerator(const Population& clonee) noexcept :
    clonee(clonee) {
}

std::unique_ptr<Population> CloningPopulationGenerator::generatePopulation() {

    TrivialNeuroComponentsFactory factory;

    auto population = std::make_unique<Population>(clonee.getSynapseParams());

    for (auto cit = clonee.cbeginNeurons(); cit != clonee.cendNeurons(); ++ cit) {
        const Neuron& cloneeNeuron = **cit;
        auto clonedNeuron = factory.makeNeuron(
                cloneeNeuron.getNeuronId(),
                cloneeNeuron.getNeuronParams());

        population->addNeuron(std::move(clonedNeuron), clonee.getCellLocation(cloneeNeuron.getNeuronId()));
    }

    for (auto cit = clonee.cbeginNeurons(); cit != clonee.cendNeurons(); ++ cit) {
        const Neuron& cloneeNeuron = **cit;
        for (
                auto csynIt = cloneeNeuron.cbeginOutboundSynapses();
                csynIt != cloneeNeuron.cendOutboundSynapses();
                ++ csynIt) {
            const Synapse& cloneeSynapse = **csynIt;
            auto& preSynapticNeuron = population->getNeuronById(cloneeSynapse.preSynapticNeuron->getNeuronId());
            auto& postSynapticNeuron = population->getNeuronById(cloneeSynapse.postSynapticNeuron->getNeuronId());

            auto clonedSynapse = factory.makeSynapse(
                    &preSynapticNeuron,
                    &postSynapticNeuron,
                    cloneeSynapse.conductionDelay,
                    cloneeSynapse.weight
                    );

            preSynapticNeuron.addOutboundSynapse(clonedSynapse.get());

            if (cloneeNeuron.getNeuronParams()->isInhibitory) {
                population->addInhibitorySynapse(std::move(clonedSynapse));
            } else {
                population->addExcitatorySynapse(std::move(clonedSynapse));
            }
        }
    }

    population->setChannelProjector(std::make_unique<ThrowingChannelProjector>());

    return population;
}

void CloningPopulationGenerator::ThrowingChannelProjector::projectNeuronSpike(CycleOutputBuffer&,
                                                                              const Neuron&) const {
    throw std::runtime_error(msg);
}

std::unordered_set<SizeType> CloningPopulationGenerator::ThrowingChannelProjector::getMotorNeuronIds() const {
    throw std::runtime_error(msg);
}

ChannelProjector::ChannelSpikeProjectionResult
CloningPopulationGenerator::ThrowingChannelProjector::getEPSPsWithTargetNeurons(SizeType) const {
    throw std::runtime_error(msg);
}

}
