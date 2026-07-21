cmake_minimum_required(VERSION 3.16)
project(ai_learning_map LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MLP_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/mlp-from-scratch/code)
set(TRAIN_EVAL_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topics/00-foundations/train-eval-basics/code)

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

add_custom_target(run
    COMMAND $<TARGET_FILE:mlp_demo> xor
    DEPENDS mlp_demo
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
