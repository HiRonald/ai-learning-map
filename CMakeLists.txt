cmake_minimum_required(VERSION 3.16)
project(ai_learning_map LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MLP_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/mlp-from-scratch/code)

add_executable(mlp_xor
    ${MLP_CODE_DIR}/main.cpp
    ${MLP_CODE_DIR}/matrix.cpp
    ${MLP_CODE_DIR}/layer.cpp
    ${MLP_CODE_DIR}/activator.cpp
    ${MLP_CODE_DIR}/mlp.cpp
)

target_include_directories(mlp_xor PRIVATE ${MLP_CODE_DIR})

target_compile_options(mlp_xor PRIVATE
    -Wall
    -Wextra
)

add_custom_target(run
    COMMAND $<TARGET_FILE:mlp_xor>
    DEPENDS mlp_xor
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
