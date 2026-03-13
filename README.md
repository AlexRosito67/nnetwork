# nnetwork

A lightweight neural network CLI tool written in C++ from scratch.
Designed for regression tasks and embedded systems deployment.

## Features

- Configurable architecture: any number of layers and neurons
- Per-layer activation functions: `sigmoid`, `tanh`, `relu`, `leaky_relu`, `linear`, `softmax`
- Train, validate, predict and export in one tool
- Exports trained model as a C++ header (`nnetwork.h`) ready for microcontrollers
- Continue training from a saved model
- No external dependencies — pure C++17

## Build
```bash
mkdir build
cd build
cmake ..
make
```

## Usage
```bash
# Train from scratch
nnetwork -d 1,6,6,1 -act tanh,tanh,tanh -a train -f data.csv -e 200000 -l 0.01 -o model.txt

# Continue training
nnetwork -d 1,6,6,1 -act tanh,tanh,tanh -a train -f data.csv -e 200000 -l 0.005 -m model.txt -o model.txt

# Validate
nnetwork -d 1,6,6,1 -act tanh,tanh,tanh -a validate -f data.csv -m model.txt

# Predict
nnetwork -d 1,6,6,1 -act tanh,tanh,tanh -a predict -v 0.65 -m model.txt

# Export C++ header for microcontroller
nnetwork -d 1,6,6,1 -act tanh,tanh,tanh -a export -m model.txt -o nnetwork.h
```

## Arguments

| Flag | Description |
|------|-------------|
| `-d` | Network dimensions, e.g. `1,6,6,1` |
| `-a` | Action: `train`, `predict`, `validate`, `export` |
| `-act` | Activation per layer, e.g. `tanh,tanh,linear` |
| `-f` | Training CSV file |
| `-m` | Model file to load |
| `-o` | Output file (model or header) |
| `-e` | Number of epochs (default: 50000) |
| `-l` | Learning rate (default: 0.05) |
| `-v` | Input values for prediction |

## CSV Format
```
input1,input2,...,expected_output
0.65,0.872
0.72,0.901
```

## Embedded Deployment

After exporting the header, use it directly on ESP32 or any microcontroller:
```cpp
#include "nnetwork.h"

float input[1] = { ratio };
float output[1];
predict(input, output);
```

## Application

Originally developed to calibrate a thermistor-based thermometer using a ratiometric
circuit with an AD620N instrumentation amplifier. The exported header runs directly
on an ATtiny85 or ESP32 with no external dependencies.

## Author

Alex Rosito with the valuable assistance of Vera — Valley Glen, California# C++ Project

