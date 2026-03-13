#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <map>
#include "parser.h"
#include "network.h"

std::string trainingFileName = "";
std::string weightsFileName  = "";

void printHelp() {
    std::cout << "\nUsage: nnetwork [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -d <dims>     Network dimensions, e.g. 1,6,1 or 2,8,4,1\n";
    std::cout << "  -a <action>   Action: train | predict | validate | export\n";
    std::cout << "  -act <funcs>  Activation per layer, e.g. sigmoid,sigmoid,linear\n";
    std::cout << "  -f <file>     Training data file (CSV)\n";
    std::cout << "  -m <file>     Model file to load\n";
    std::cout << "  -o <file>     Output file (model or header)\n";
    std::cout << "  -e <epochs>   Number of epochs (default: 50000)\n";
    std::cout << "  -l <rate>     Learning rate (default: 0.05)\n";
    std::cout << "  -v <values>   Input values for predict, e.g. 0.172 or 0.5,1.2\n";
    std::cout << "  -h            Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  nnetwork -d 1,6,1 -a train -f data.csv -o model.txt\n";
    std::cout << "  nnetwork -d 1,6,6,1 -act sigmoid,sigmoid,linear -a train -f data.csv -o model.txt\n";
    std::cout << "  nnetwork -d 1,6,1 -a predict -v 0.172 -m model.txt\n";
    std::cout << "  nnetwork -d 1,6,1 -a validate -f data.csv -m model.txt\n";
    std::cout << "  nnetwork -d 1,6,1 -a export -m model.txt -o nnetwork.h\n\n";
}

void trainNetWork(Network& network, const std::vector<std::vector<double>>& data, int numEpochs, double learningRate) {
    std::vector<double> targetVec(1);
    for (int i = 0; i < numEpochs; i++) {
        for (const auto& item : data) {
            std::vector<double> inputs(item.begin(), item.end() - 1);
            targetVec[0] = item.back();
            network.train(inputs, targetVec, learningRate);
        }
        if (i % 1000 == 0) {
            double error = 0.0;
            for (const auto& item : data) {
                std::vector<double> inputs(item.begin(), item.end() - 1);
                targetVec[0] = item.back();
                auto output = network.forward(inputs);
                error += (targetVec[0] - output[0]) * (targetVec[0] - output[0]);
            }
            std::cout << "Epoch: " << i << " Error: " << error / data.size() << std::endl;
        }
    }
}

std::vector<int> parseDims(const std::string& argumentValue) {
    std::vector<int> dims;
    if (argumentValue.empty()) {
        std::cerr << "Error: No dimensions provided." << std::endl;
        return dims;
    }
    std::stringstream ss(argumentValue);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            dims.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << token << " is not a number." << std::endl;
            return {};
        }
    }
    return dims;
}

std::vector<double> parseValues(const std::string& argumentValue) {
    std::vector<double> values;
    std::stringstream ss(argumentValue);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stod(token));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid value: " << token << " is not a number." << std::endl;
            return {};
        }
    }
    return values;
}

std::vector<ActivationType> parseActivations(const std::string& value) {
    std::map<std::string, ActivationType> map = {
        {"sigmoid",    ActivationType::SIGMOID},
        {"tanh",       ActivationType::TANH},
        {"relu",       ActivationType::RELU},
        {"leaky_relu", ActivationType::LEAKY_RELU},
        {"linear",     ActivationType::LINEAR},
        {"softmax",    ActivationType::SOFTMAX}
    };

    std::vector<ActivationType> activations;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto it = map.find(token);
        if (it == map.end()) {
            std::cerr << "Error: Unknown activation '" << token << "'" << std::endl;
            return {};
        }
        activations.push_back(it->second);
    }
    return activations;
}

void validateNetwork(Network& network, const std::vector<std::vector<double>>& data) {
    double maxError   = 0.0;
    double totalError = 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n";
    std::cout << std::left
              << std::setw(14) << "Input"
              << std::setw(14) << "Expected"
              << std::setw(14) << "Output"
              << std::setw(14) << "Error"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (const auto& item : data) {
        std::vector<double> inputs(item.begin(), item.end() - 1);
        double expected = item.back();
        auto output = network.forward(inputs);
        double error = std::abs(expected - output[0]);

        std::cout << std::setw(14) << inputs[0]
                  << std::setw(14) << expected
                  << std::setw(14) << output[0]
                  << std::setw(14) << error
                  << "\n";

        totalError += error;
        maxError = std::max(maxError, error);
    }

    std::cout << std::string(56, '-') << "\n";
    std::cout << "Max error:  " << maxError             << "\n";
    std::cout << "Mean error: " << totalError / data.size() << "\n\n";
}

void exportHeader(Network& network, const std::string& filename) {
    std::ofstream file(filename);

    std::string arch = std::to_string(network.layers[0].neurons[0].weights.size());
    for (auto& layer : network.layers)
        arch += "-" + std::to_string(layer.neurons.size());

    file << "#ifndef NNETWORK_H\n";
    file << "#define NNETWORK_H\n\n";
    file << "#include <math.h>\n\n";
    file << "// Generated by nnetwork\n";
    file << "// Architecture: " << arch << "\n\n";

    // activation functions needed
    std::map<ActivationType, std::string> actNames = {
        {ActivationType::SIGMOID,    "sigmoid"},
        {ActivationType::TANH,       "tanh_act"},
        {ActivationType::RELU,       "relu"},
        {ActivationType::LEAKY_RELU, "leaky_relu"},
        {ActivationType::LINEAR,     "linear"},
        {ActivationType::SOFTMAX,    "softmax"}
    };

    file << "static inline float sigmoid(float x)    { return 1.0f / (1.0f + expf(-x)); }\n";
    file << "static inline float tanh_act(float x)   { return tanhf(x); }\n";
    file << "static inline float relu(float x)        { return x > 0.0f ? x : 0.0f; }\n";
    file << "static inline float leaky_relu(float x)  { return x > 0.0f ? x : 0.01f * x; }\n";
    file << "static inline float linear(float x)      { return x; }\n\n";

    for (size_t l = 0; l < network.layers.size(); l++) {
        auto& layer    = network.layers[l];
        int numNeurons = layer.neurons.size();
        int numWeights = layer.neurons[0].weights.size();

        file << "const float w" << l+1 << "[" << numNeurons << "][" << numWeights << "] = {\n";
        for (int i = 0; i < numNeurons; i++) {
            file << "    {";
            for (int w = 0; w < numWeights; w++) {
                file << layer.neurons[i].weights[w];
                if (w < numWeights - 1) file << ", ";
            }
            file << "}";
            if (i < numNeurons - 1) file << ",";
            file << "\n";
        }
        file << "};\n";

        file << "const float b" << l+1 << "[" << numNeurons << "] = {";
        for (int i = 0; i < numNeurons; i++) {
            file << layer.neurons[i].bias;
            if (i < numNeurons - 1) file << ", ";
        }
        file << "};\n\n";
    }

    int maxNeurons = 0;
    for (auto& layer : network.layers)
        maxNeurons = std::max(maxNeurons, (int)layer.neurons.size());

    file << "static inline void predict(const float* input, float* output) {\n";
    file << "    float a[" << maxNeurons << "], b[" << maxNeurons << "];\n";
    file << "    const float* in = input;\n";
    file << "    float* out = a;\n\n";

    for (size_t l = 0; l < network.layers.size(); l++) {
        int numNeurons = network.layers[l].neurons.size();
        int numInputs  = network.layers[l].neurons[0].weights.size();
        std::string actName = actNames[network.layers[l].activationType];
        file << "    // layer " << l+1 << " (" << actName << ")\n";
        file << "    for (int i = 0; i < " << numNeurons << "; i++) {\n";
        file << "        float sum = b" << l+1 << "[i];\n";
        file << "        for (int j = 0; j < " << numInputs << "; j++)\n";
        file << "            sum += w" << l+1 << "[i][j] * in[j];\n";
        file << "        out[i] = " << actName << "(sum);\n";
        file << "    }\n";
        if (l < network.layers.size() - 1)
            file << "    in = out; out = (out == a) ? b : a;\n\n";
    }

    file << "\n    for (int i = 0; i < " << network.layers.back().neurons.size() << "; i++)\n";
    file << "        output[i] = out[i];\n";
    file << "}\n\n";
    file << "#endif // NNETWORK_H\n";

    std::cout << "Header exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    srand(std::time(0));

    if (argc == 1) {
        printHelp();
        return 0;
    }

    std::vector<int>         dims;
    int                      numEpochs    = 50000;
    double                   learningRate = 0.05;
    std::string              modelFileName = "";
    std::string              action        = "train";
    std::vector<double>      predictValues;
    std::vector<ActivationType> activations;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-d") {
            dims = parseDims(argv[++i]);
        } else if (arg == "-f") {
            trainingFileName = argv[++i];
        } else if (arg == "-o") {
            weightsFileName = argv[++i];
        } else if (arg == "-e") {
            numEpochs = std::stoi(argv[++i]);
        } else if (arg == "-l") {
            learningRate = std::stod(argv[++i]);
        } else if (arg == "-m") {
            modelFileName = argv[++i];
        } else if (arg == "-a") {
            action = argv[++i];
        } else if (arg == "-v") {
            predictValues = parseValues(argv[++i]);
        } else if (arg == "-act") {
            activations = parseActivations(argv[++i]);
        } else if (arg == "-h") {
            printHelp();
            return 0;
        } else {
            std::cerr << "Error: Unknown option '" << arg << "'. Use -h for help." << std::endl;
            return 1;
        }
    }

    // validaciones por accion
    if (action == "train") {
        if (dims.empty()) {
            std::cerr << "Error: No dimensions provided. Use -d <dims>" << std::endl;
            return 1;
        }
        if (trainingFileName.empty()) {
            std::cerr << "Error: No training file provided. Use -f <file>" << std::endl;
            return 1;
        }
        if (weightsFileName.empty()) {
            std::cerr << "Error: No output file provided. Use -o <file>" << std::endl;
            return 1;
        }
    } else if (action == "predict") {
        if (dims.empty()) {
            std::cerr << "Error: No dimensions provided. Use -d <dims>" << std::endl;
            return 1;
        }
        if (modelFileName.empty()) {
            std::cerr << "Error: No model file provided. Use -m <file>" << std::endl;
            return 1;
        }
        if (predictValues.empty()) {
            std::cerr << "Error: No input values provided. Use -v <values>" << std::endl;
            return 1;
        }
    } else if (action == "validate") {
        if (dims.empty()) {
            std::cerr << "Error: No dimensions provided. Use -d <dims>" << std::endl;
            return 1;
        }
        if (modelFileName.empty()) {
            std::cerr << "Error: No model file provided. Use -m <file>" << std::endl;
            return 1;
        }
        if (trainingFileName.empty()) {
            std::cerr << "Error: No training file provided. Use -f <file>" << std::endl;
            return 1;
        }
    } else if (action == "export") {
        if (dims.empty()) {
            std::cerr << "Error: No dimensions provided. Use -d <dims>" << std::endl;
            return 1;
        }
        if (modelFileName.empty()) {
            std::cerr << "Error: No model file provided. Use -m <file>" << std::endl;
            return 1;
        }
        if (weightsFileName.empty()) {
            std::cerr << "Error: No output file provided. Use -o <file>" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: Unknown action '" << action << "'. Use train, predict, validate or export." << std::endl;
        return 1;
    }

    // construir red despues de validar
    Network network(dims, activations);

    if (!modelFileName.empty()) {
        network.load(modelFileName);
        std::cout << "Model loaded from: " << modelFileName << std::endl;
    }

    if (action == "train") {
        Parser p = Parser(trainingFileName);
        auto data = p.read_data();
        trainNetWork(network, data, numEpochs, learningRate);
        network.save(weightsFileName);
        std::cout << "Model saved to: " << weightsFileName << std::endl;

    } else if (action == "predict") {
        auto output = network.forward(predictValues);
        std::cout << "Input:  ";
        for (size_t i = 0; i < predictValues.size(); i++) {
            std::cout << predictValues[i];
            if (i < predictValues.size() - 1) std::cout << ", ";
        }
        std::cout << "\nOutput: ";
        for (size_t i = 0; i < output.size(); i++) {
            std::cout << output[i];
            if (i < output.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;

    } else if (action == "validate") {
        Parser p = Parser(trainingFileName);
        auto data = p.read_data();
        validateNetwork(network, data);

    } else if (action == "export") {
        exportHeader(network, weightsFileName);
    }

    return 0;
}