#include <algorithm>
#include <unordered_set>
#include <neuro/Population.hpp>
#include <neuro/Neuron.hpp>
#include "NonCoherentStimulator.hpp"
#include "EventProcessor.hpp"

namespace soft_npu {

std::vector<std::reference_wrapper<Neuron>> getNeuronsToStimulate(Population& population) {

    std::vector<std::reference_wrapper<Neuron>> rv;
    std::transform(
            population.cbeginNeurons(),
            population.cendNeurons(),
            std::back_inserter(rv), [](const auto& neuron) {
                return std::ref(*neuron);
            });
    return rv;
}

ValueType getPoissonDistLambda(SizeType numNeuronsToStimulate, TimeType dt, ValueType rate) {
    return numNeuronsToStimulate * dt * rate;
}

NonCoherentStimulator::NonCoherentStimulator(
        const ParamsType& params,
        RandomEngineType& randomEngine,
        Population& population,
        TimeType dt) :
        randomEngine(randomEngine),
        neuronsToStimulate(getNeuronsToStimulate(population)),
        poissonDistribution(getPoissonDistLambda(neuronsToStimulate.size(), dt,
       static_cast<ValueType>(params["nonCoherentStimulator"]["rate"]))),
        dt(dt),

        epsp(params["nonCoherentStimulator"]["epsp"])
        {
}

void NonCoherentStimulator::processCycle(const CycleContext& ctx) {

    if (!neuronsToStimulate.empty()) {
        auto numSpikingNeurons = std::min(
                static_cast<SizeType>(poissonDistribution(randomEngine)),
                static_cast<SizeType>(neuronsToStimulate.size()));

        std::vector<std::reference_wrapper<Neuron>> selectedNeurons;
        selectedNeurons.reserve(numSpikingNeurons);

        for (SizeType i = 0; i < numSpikingNeurons; ++i) {
            SizeType neuronIdx = std::uniform_int_distribution<SizeType>(0, neuronsToStimulate.size() - 1)(randomEngine);
            selectedNeurons.push_back(neuronsToStimulate[neuronIdx]);
        }

        for (auto neuron : selectedNeurons) {
            ctx.staticContext.eventProcessor.pushImmediateTransmissionEvent(epsp, neuron);
        }
    }
}

void NonCoherentStimulator::setRate(ValueType rate) noexcept {
    typename std::poisson_distribution<SizeType>::param_type param(
        getPoissonDistLambda(neuronsToStimulate.size(), dt, rate));
    poissonDistribution.param(param);
}

}
