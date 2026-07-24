# 根目录薄封装：C++ 主题走 CMake；文档站走 npm（site/）
BUILD_DIR := build
DEMO ?= xor
MODE ?= normal
MLP_DATA_DIR ?= $(CURDIR)/topics/00-foundations/mlp-from-scratch/data
TRAIN_EVAL_DATA_DIR ?= $(CURDIR)/topics/00-foundations/train-eval-basics/data

.PHONY: all configure build run run-eval run-repr run-cnn run-residual clean rebuild docs docs-dev docs-build

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@ln -sfn $(BUILD_DIR)/compile_commands.json compile_commands.json

build: configure
	cmake --build $(BUILD_DIR)

# DEMO=xor|sine|fashion|list          （mlp-from-scratch）
# DEMO=filter|param|fashion|list      （cnn-basics；未改 DEMO 时 run-cnn 默认 filter）
run: build
	MLP_DATA_DIR="$(MLP_DATA_DIR)" $(BUILD_DIR)/mlp_demo $(DEMO)

# MODE=normal|overfit|earlystop|list  （train-eval-basics）
run-eval: build
	MLP_DATA_DIR="$(MLP_DATA_DIR)" TRAIN_EVAL_DATA_DIR="$(TRAIN_EVAL_DATA_DIR)" \
		$(BUILD_DIR)/train_eval_demo $(MODE)

# data-and-representation：tokenizer + embedding 查表（不训练）
run-repr: build
	$(BUILD_DIR)/data_repr_demo

# cnn-basics；fashion 复用 MLP_DATA_DIR。DEMO 仍是 xor（mlp 默认）时改走 filter。
run-cnn: build
	@demo="$(DEMO)"; \
	if [ "$$demo" = "xor" ]; then demo=filter; fi; \
	MLP_DATA_DIR="$(MLP_DATA_DIR)" $(BUILD_DIR)/cnn_demo $$demo

# residual-basics：DEMO=identity|fashion|list（默认 identity；xor 时改走 identity）
run-residual: build
	@demo="$(DEMO)"; \
	if [ "$$demo" = "xor" ]; then demo=identity; fi; \
	MLP_DATA_DIR="$(MLP_DATA_DIR)" $(BUILD_DIR)/residual_demo $$demo

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

docs: docs-dev

docs-dev:
	cd site && npm install && npm run docs:dev

docs-build:
	cd site && npm install && npm run docs:build
