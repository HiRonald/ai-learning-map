# 根目录薄封装：C++ 主题走 CMake；文档站走 npm（site/）
BUILD_DIR := build
DEMO ?= xor
MODE ?= normal
MLP_DATA_DIR ?= $(CURDIR)/topics/00-foundations/mlp-from-scratch/data
TRAIN_EVAL_DATA_DIR ?= $(CURDIR)/topics/00-foundations/train-eval-basics/data

.PHONY: all configure build run run-eval clean rebuild docs docs-dev docs-build

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@ln -sfn $(BUILD_DIR)/compile_commands.json compile_commands.json

build: configure
	cmake --build $(BUILD_DIR)

# DEMO=xor|sine|fashion|list
run: build
	MLP_DATA_DIR="$(MLP_DATA_DIR)" $(BUILD_DIR)/mlp_demo $(DEMO)

# MODE=normal|overfit|earlystop|list  （train-eval-basics）
run-eval: build
	MLP_DATA_DIR="$(MLP_DATA_DIR)" TRAIN_EVAL_DATA_DIR="$(TRAIN_EVAL_DATA_DIR)" \
		$(BUILD_DIR)/train_eval_demo $(MODE)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

docs: docs-dev

docs-dev:
	cd site && npm install && npm run docs:dev

docs-build:
	cd site && npm install && npm run docs:build
