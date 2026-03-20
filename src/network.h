#pragma once
#include <vector>
#include <string>
#include "layer.h"

class Network {
public:
    std::vector<Layer> layers;

    // sizes = {inputs, hidden1, hidden2, ..., outputs}
    // activations = one per layer, e.g. {SIGMOID, SIGMOID, LINEAR}
    // if activations is empty, defaults to SIGMOID for all layers
    Network(std::vector<int> sizes,
            std::vector<ActivationType> activations = {});

    std::vector<double> forward(const std::vector<double>& input);
    void train(const std::vector<double>& inputs,
               const std::vector<double>& targets,
               double lr);
    void save(const std::string& filename);
    int load(const std::string& filename);
};