cmake_minimum_required(VERSION 3.10)
project(MyProject)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nnetwork
    src/main.cpp
    src/neuron.cpp
    src/layer.cpp
    src/network.cpp
    ../CommonLib/parser.cpp
)

target_include_directories(nnetwork PRIVATE ../CommonLib)
