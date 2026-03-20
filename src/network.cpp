#include "network.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

Network::Network(std::vector<int> sizes, std::vector<ActivationType> activations) {
    if (sizes.size() < 2)
        throw std::invalid_argument("Network needs at least 2 sizes (input and output)");

    // si no se especifican activaciones, sigmoid para todas
    if (activations.empty())
        activations.resize(sizes.size() - 1, ActivationType::SIGMOID);

    if (activations.size() != sizes.size() - 1)
        throw std::invalid_argument("Number of activations must match number of layers");

    for (size_t i = 0; i < sizes.size() - 1; i++)
        layers.emplace_back(sizes[i + 1], sizes[i], activations[i]);
}

std::vector<double> Network::forward(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (auto& layer : layers)
        current = layer.forward(current);
    return current;
}

void Network::train(const std::vector<double>& inputs,
                    const std::vector<double>& targets,
                    double lr) {
    // forward pass
    std::vector<std::vector<double>> layerInputs;
    layerInputs.push_back(inputs);
    std::vector<double> current = inputs;
    for (auto& layer : layers) {
        current = layer.forward(current);
        layerInputs.push_back(current);
    }

    // backprop - output layer
    Layer& outputLayer = layers.back();
    for (size_t i = 0; i < outputLayer.neurons.size(); i++) {
        double error = targets[i] - layerInputs.back()[i];
        outputLayer.neurons[i].delta = error * outputLayer.neurons[i].derivative();
    }

    // backprop - hidden layers
    for (int l = (int)layers.size() - 2; l >= 0; l--) {
        Layer& curr = layers[l];
        Layer& next = layers[l + 1];
        for (size_t i = 0; i < curr.neurons.size(); i++) {
            double error = 0.0;
            for (auto& n : next.neurons)
                error += n.weights[i] * n.delta;
            curr.neurons[i].delta = error * curr.neurons[i].derivative();
        }
    }

    // update weights
    for (size_t l = 0; l < layers.size(); l++) {
        const std::vector<double>& inp = layerInputs[l];
        for (auto& n : layers[l].neurons) {
            for (size_t w = 0; w < n.weights.size(); w++)
                n.weights[w] += lr * n.delta * inp[w];
            n.bias += lr * n.delta;
        }
    }
}

void Network::save(const std::string& filename) {
    std::ofstream file(filename);
    file << layers.size() << "\n";
    for (auto& layer : layers) {
        file << (int)layer.activationType << "\n";
        file << layer.neurons.size() << "\n";
        file << layer.neurons[0].weights.size() << "\n"; // aquí
        for (auto& n : layer.neurons) {
            for (double w : n.weights)
                file << w << " ";
            file << n.bias << "\n";
        }
    }
}

int Network::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open model file: " << filename << std::endl;
        return -1;
    }
    size_t numLayers;
    file >> numLayers;
    if (numLayers != layers.size()) {
        std::cerr << "Error: Model has " << numLayers << " layers but network has " << layers.size() << std::endl;
        return -1;
    }

    for (size_t l = 0; l < numLayers; l++) {
        
        int actType;
        file >> actType;
        layers[l].activationType = (ActivationType)actType;
        for (auto& n : layers[l].neurons)
            n.activationType = (ActivationType)actType;

        size_t numNeurons;
        file >> numNeurons;
        if (numNeurons != layers[l].neurons.size()) {
            std::cerr << "Error: Layer " << l << " mismatch." << std::endl;
            return -1;
        }

        // ******** aquí
        size_t numInputs;
        file >> numInputs;
        if (numInputs != layers[l].neurons[0].weights.size()) {
            std::cerr << "Error: The model you are defining doesn't coincide with the model saved." << std::endl;
            return -1;
        }
        //*********

        for (size_t i = 0; i < numNeurons; i++) {
            for (size_t w = 0; w < layers[l].neurons[i].weights.size(); w++)
                file >> layers[l].neurons[i].weights[w];
            file >> layers[l].neurons[i].bias;
        }
    }
    return 0;
}