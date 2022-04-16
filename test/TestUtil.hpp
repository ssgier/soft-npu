#pragma once

#include <Aliases.hpp>

namespace soft_npu {

auto getTemplateParams() {
    std::string jsonString = R"(
{
    "simulation": {
        "seed": 0,
        "untilTime": 1.0,
        "populationGenerator": "SingleNeuron",
        "channelProjector": "OneToOne"
    },
    "nonCoherentStimulator": {
        "rate": 0,
        "epsp": 1.0
    },
    "channelProjectors": {
        "OneToOne": {
            "epsp": 0.4
        },
        "OneToMany": {
          "fromInChannelId": 0,
          "toInChannelId": 200,
          "fromSensoryNeuronId": 0,
          "toSensoryNeuronId": 800,
          "fromOutChannelId": 0,
          "toOutChannelId": 2,
          "fromMotorNeuronId": 200,
          "toMotorNeuronId": 800,
          "divergence": 5,
          "epsp": 1.5
        },
        "Topographic": {
          "startXSensory": 0.1,
          "startYSensory": 0.1,
          "startXMotor": 0.5,
          "startYMotor": 0.5,
          "numInputChannels": 2,
          "projectionRadius": 0.03,
          "inputInterChannelDistance": 0.05,
          "numOutputChannels": 2,
          "convergenceRadius": 0.05,
          "outputInterChannelDistance": 0.1,
          "epsp": 1.5
        }
    },
    "populationGenerators": {
        "p1000": {
            "inhibitoryConductionDelayDeterministicPart": 3e-3,
            "inhibitoryConductionDelayRandomPart" : 1e-3,
            "inhibitorySynapseWeight": 0.3,
            "excitatorySynapseInitialWeight": 0.1
        },
        "r2dSheet": {
          "numNeurons": 10000,
          "pctInhibitoryNeurons": 0.1,
          "pctExcLongDistanceTargets": 0.25,
          "radiusExcShort": 0.03,
          "radiusExcLong": 0.01,
          "radiusInh": 0.01,
          "numTargetsExc": 10,
          "numTargetsInh": 3,
          "maxExcConductionDelay": 20e-3,
          "inhibitoryConductionDelayDeterministicPart": 3e-3,
          "inhibitoryConductionDelayRandomPart" : 1e-3,
          "inhibitorySynapseWeight": 0.3,
          "excitatorySynapseInitialWeight": 0.1
        }
    },
    "dopaminergicModulator": {
        "releaseBaseRate": 1.0,
        "releaseFrequency": 4.0
    },
	"cycleController": {
		"dt": 1e-4
	},
	"eventProcessor": {
		"lookAheadWindow": 20e-3,
		"subBufferReserveSlots": 1000
	},
    "synapseParams": {
        "stdpTimeConstant": 20e-3,
        "stdpCutOffTime": 1.0,
        "stdpScaleFactorPotentiation": 0.1,
        "stdpDepressionVsPotentiationRatio": 1.2,
        "maxWeight": 0.7,
        "eligibilityTraceTimeConstant": 10e-3,
        "eligibilityTraceCutOffTimeFactor": 2
    },
    "neuronParams": {
        "excitatory": {
            "timeConstant": 20e-3,
            "refractoryPeriod": 10e-3,
            "thresholdVoltage": 1.0,
            "resetVoltage": 0.0,
            "voltageFloor": -1.0,
            "isInhibitory": false
        },
        "inhibitory": {
            "timeConstant": 5e-3,
            "refractoryPeriod": 5e-3,
            "thresholdVoltage": 1.0,
            "resetVoltage": 0.0,
            "voltageFloor": 0.0,
            "isInhibitory": true
        }
    },
    "expMetricsManager": {
        "reportingInterval": 10
    }
})";
    return std::make_shared<ParamsType>(nlohmann::json::parse(jsonString));
}

}