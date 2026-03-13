#pragma once
#include <vector>
#include "neuron.h"

class Layer {
public:
    std::vector<Neuron> neurons;
    ActivationType activationType;

    Layer(int numNeurons, int inputsPerNeuron, ActivationType type = ActivationType::SIGMOID);
    std::vector<double> forward(const std::vector<double>& inputs);
};