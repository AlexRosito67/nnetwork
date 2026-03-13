#pragma once
#include <vector>

enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR,
    SOFTMAX
};

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;
    ActivationType activationType;

    Neuron(int num_inputs, ActivationType type = ActivationType::SIGMOID);

    double forward(const std::vector<double>& inputs);
    double activate(double x);
    double derivative();
};