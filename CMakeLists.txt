cmake_minimum_required(VERSION 3.16)
project(ai_learning_map LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MLP_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/mlp-from-scratch/code)
set(TRAIN_EVAL_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/train-eval-basics/code)
set(DATA_REPR_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/data-and-representation/code)
set(CNN_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/01-classic-dl/cnn-basics/code)

# 共享：手写 nn + IDX/下载工具（供多个主题可执行文件复用）
add_library(mlp_core STATIC
    ${MLP_CODE_DIR}/nn/matrix.cpp
    ${MLP_CODE_DIR}/nn/layer.cpp
    ${MLP_CODE_DIR}/nn/activator.cpp
    ${MLP_CODE_DIR}/nn/mlp.cpp
    ${MLP_CODE_DIR}/demos/common.cpp
    ${MLP_CODE_DIR}/demos/download.cpp
    ${MLP_CODE_DIR}/demos/idx_io.cpp
)
target_include_directories(mlp_core PUBLIC ${MLP_CODE_DIR})
target_compile_options(mlp_core PRIVATE -Wall -Wextra)

add_executable(mlp_demo
    ${MLP_CODE_DIR}/main.cpp
    ${MLP_CODE_DIR}/demos/xor.cpp
    ${MLP_CODE_DIR}/demos/sine.cpp
    ${MLP_CODE_DIR}/demos/fashion.cpp
)
target_link_libraries(mlp_demo PRIVATE mlp_core)
target_compile_options(mlp_demo PRIVATE -Wall -Wextra)

add_executable(train_eval_demo
    ${TRAIN_EVAL_CODE_DIR}/main.cpp
)
target_link_libraries(train_eval_demo PRIVATE mlp_core)
target_compile_options(train_eval_demo PRIVATE -Wall -Wextra)

add_executable(data_repr_demo
    ${DATA_REPR_CODE_DIR}/main.cpp
)
target_link_libraries(data_repr_demo PRIVATE mlp_core)
target_compile_options(data_repr_demo PRIVATE -Wall -Wextra)

add_library(cnn_core STATIC
    ${CNN_CODE_DIR}/cnn/tensor.cpp
    ${CNN_CODE_DIR}/cnn/conv2d.cpp
    ${CNN_CODE_DIR}/cnn/pool2d.cpp
    ${CNN_CODE_DIR}/cnn/ops.cpp
    ${CNN_CODE_DIR}/cnn/net.cpp
)
target_include_directories(cnn_core PUBLIC ${CNN_CODE_DIR})
target_link_libraries(cnn_core PUBLIC mlp_core)
target_compile_options(cnn_core PRIVATE -Wall -Wextra)

add_executable(cnn_demo
    ${CNN_CODE_DIR}/main.cpp
    ${CNN_CODE_DIR}/demos/filter_demo.cpp
    ${CNN_CODE_DIR}/demos/param_demo.cpp
    ${CNN_CODE_DIR}/demos/fashion_demo.cpp
)
target_link_libraries(cnn_demo PRIVATE cnn_core)
target_compile_options(cnn_demo PRIVATE -Wall -Wextra)

add_custom_target(run
    COMMAND $<TARGET_FILE:mlp_demo> xor
    DEPENDS mlp_demo
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
