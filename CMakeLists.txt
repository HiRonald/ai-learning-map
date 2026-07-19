cmake_minimum_required(VERSION 3.16)
project(ai_learning_map LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MLP_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/mlp-from-scratch/code)

add_executable(mlp_demo
    ${MLP_CODE_DIR}/main.cpp
    ${MLP_CODE_DIR}/nn/matrix.cpp
    ${MLP_CODE_DIR}/nn/layer.cpp
    ${MLP_CODE_DIR}/nn/activator.cpp
    ${MLP_CODE_DIR}/nn/mlp.cpp
    ${MLP_CODE_DIR}/demos/common.cpp
    ${MLP_CODE_DIR}/demos/download.cpp
    ${MLP_CODE_DIR}/demos/idx_io.cpp
    ${MLP_CODE_DIR}/demos/xor.cpp
    ${MLP_CODE_DIR}/demos/sine.cpp
    ${MLP_CODE_DIR}/demos/fashion.cpp
)

target_include_directories(mlp_demo PRIVATE ${MLP_CODE_DIR})

target_compile_options(mlp_demo PRIVATE
    -Wall
    -Wextra
)

add_custom_target(run
    COMMAND $<TARGET_FILE:mlp_demo> xor
    DEPENDS mlp_demo
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
