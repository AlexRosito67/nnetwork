# nnetwork

A CLI tool that trains neural networks and deploys them to microcontrollers.
Written in pure C++17 — no frameworks, no dependencies, no runtime.
Train on your desktop. Export a single C header. Run on any microcontroller 
with a C compiler.

## Why nnetwork?

Most neural network frameworks require installing Python, pip packages, 
and megabytes of dependencies just to get started.

nnetwork is a single binary. Build it once with CMake and you're done.
No pip, no conda, no virtual environments, no CUDA drivers.

And when your model is trained, it exports to a single C header file 
that compiles on any microcontroller with a C compiler — 
no runtime, no interpreter, no overhead.

Open source. No license restrictions.

## Features

- Configurable architecture: any number of layers and neurons
- Per-layer activation functions: `sigmoid`, `tanh`, `relu`, `leaky_relu`, `linear`, `softmax`
- Train, validate, predict and export in one tool
- Exports trained model as a C header (`nnetwork.h`) ready for microcontrollers
- Continue training from a saved model


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

Alex Rosito  — Valley Glen, California


