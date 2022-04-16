#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <core/NeuronSpikeInfo.hpp>
#include <fstream>
#include <core/SynapseInfo.hpp>
#include <neuro/Population.hpp>
#include <core/NeuronInfo.hpp>
#include <iomanip>

namespace soft_npu::FileUtil {

void writeSpikeTrainsToCSV(const std::string& filePath, const std::vector<NeuronSpikeInfo>& spikeInfos) {
    std::ofstream fs;
    fs.open(filePath);
    fs << std::fixed << std::setprecision(4);
    fs << "Time,NeuronId" << std::endl;
    for (const auto& spikeInfo : spikeInfos) {
        fs << spikeInfo.time << ',' << spikeInfo.neuronId << std::endl;
    }
    fs.close();
}

void writeSynapseInfosToCSV(const std::string& filePath, const std::vector<SynapseInfo>& synapseInfos) {
    std::ofstream fs;
    fs.open(filePath);
    fs << "PreSynapticNeuronId,PostSynapticNeuronId,Weight,IsInhibitory" << std::endl;
    for (const auto& synapseInfo : synapseInfos) {
        fs << synapseInfo.preSynapticNeuronId << ',' << synapseInfo.postSynapticNeuronId
            << ',' << synapseInfo.weight << ',' << (synapseInfo.isInhibitory ? "true" : "false") << std::endl;
    }
    fs.close();
}

void writeLocationsToCSV(const std::string& filePath, const std::vector<Population::Location>& locationsIndexedByNeuronId) {
    std::ofstream fs;
    fs.open(filePath);
    fs << "NeuronId,LocationX,LocationY" << std::endl;
    SizeType nextNeuronId = 0;
    for (const auto& location : locationsIndexedByNeuronId) {
        fs << nextNeuronId << ',' << location[0] << ',' << location[1] << std::endl;
        ++nextNeuronId;
    }
    fs.close();
}

void writeNeuronInfosToCSV(const std::string& filePath, const std::vector<NeuronInfo>& neuronInfos) {
    std::ofstream fs;
    fs.open(filePath);
    fs << "NeuronId,IsInhibitory" << std::endl;
    for (const auto& neuronInfo : neuronInfos) {
        fs << neuronInfo.neuronId << ',' << neuronInfo.isInhibitory << std::endl;
    }
    fs.close();
}

template<typename T>
std::string getFileContent(const T& path) {
    std::ifstream is(path);

    if (!is) {
        std::stringstream ss;
        ss << "Unable to open file: " << path;
        throw std::runtime_error(ss.str());
    }

    std::stringstream buffer;
    buffer << is.rdbuf();
    return buffer.str();
}

}

