#include "neuron.h"
#include <cmath>
#include <cstdlib>
#include <random>
Neuron::Neuron(int num_inputs, ActivationType type)
    : activationType(type) {
    
    static std::mt19937 rng(std::random_device{}());

    // Distribution — range
    std::uniform_real_distribution<double> dist(-1.0, 1.0);    
    for (int i = 0; i < num_inputs; ++i)
        weights.push_back(static_cast<double>(dist(rng)));
    bias = static_cast<double>(dist(rng));
}

double Neuron::activate(double x) {
    switch (activationType) {
        case ActivationType::SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ActivationType::TANH:
            return tanh(x);
        case ActivationType::RELU:
            return x > 0.0 ? x : 0.0;
        case ActivationType::LEAKY_RELU:
            return x > 0.0 ? x : 0.01 * x;
        case ActivationType::LINEAR:
            return x;
        case ActivationType::SOFTMAX:
            return exp(x); // se normaliza en Layer::forward
        default:
            return x;
    }
}

double Neuron::derivative() {
    switch (activationType) {
        case ActivationType::SIGMOID:
            return output * (1.0 - output);
        case ActivationType::TANH:
            return 1.0 - output * output;
        case ActivationType::RELU:
            return output > 0.0 ? 1.0 : 0.0;
        case ActivationType::LEAKY_RELU:
            return output > 0.0 ? 1.0 : 0.01;
        case ActivationType::LINEAR:
            return 1.0;
        case ActivationType::SOFTMAX:
            return output * (1.0 - output);
        default:
            return 1.0;
    }
}

double Neuron::forward(const std::vector<double>& inputs) {
    double sum = bias;
    for (size_t i = 0; i < weights.size(); i++)
        sum += weights[i] * inputs[i];
    output = activate(sum);
    return output;
}