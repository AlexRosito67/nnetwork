#include "layer.h"
#include <cmath>
#include <algorithm>

Layer::Layer(int numNeurons, int inputsPerNeuron, ActivationType type)
    : activationType(type) {
    for (int i = 0; i < numNeurons; i++)
        neurons.emplace_back(inputsPerNeuron, type);
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    for (auto& n : neurons)
        outputs.push_back(n.forward(inputs));

    // normalización especial para softmax
    if (activationType == ActivationType::SOFTMAX) {
        double maxVal = *std::max_element(outputs.begin(), outputs.end());
        double sum = 0.0;
        for (auto& o : outputs) {
            o = exp(o - maxVal); // estabilidad numérica
            sum += o;
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            outputs[i] /= sum;
            neurons[i].output = outputs[i];
        }
    }

    return outputs;
}