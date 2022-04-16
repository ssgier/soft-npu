#pragma once

#include "PopulationGenerator.hpp"
#include <neuro/ChannelProjector.hpp>

namespace soft_npu {

class CloningPopulationGenerator : public PopulationGenerator {
public:
    explicit CloningPopulationGenerator(const Population& clonee) noexcept;
    std::unique_ptr<Population> generatePopulation() override;
private:

    struct ThrowingChannelProjector : public ChannelProjector {
        constexpr static auto msg = "CloningPopulationGenerator implementation is incomplete. Channel projector cloning needs to be implemented first";

        void projectNeuronSpike(CycleOutputBuffer& cycleOutputBuffer, const Neuron& spikingNeuron) const override;
        std::unordered_set<SizeType> getMotorNeuronIds() const override;

    protected:
        ChannelSpikeProjectionResult getEPSPsWithTargetNeurons(SizeType channelId) const override;
    };

    const Population& clonee;
};

}
