# 根目录薄封装：C++ 主题走 CMake；文档站走 npm（site/）
BUILD_DIR := build
DEMO ?= xor
MODE ?= normal
MLP_DATA_DIR ?= $(CURDIR)/topics/00-foundations/mlp-from-scratch/data
TRAIN_EVAL_DATA_DIR ?= $(CURDIR)/topics/00-foundations/train-eval-basics/data
RNN_DATA_DIR ?= $(CURDIR)/topics/01-classic-dl/rnn-seq/data
ATTN_DATA_DIR ?= $(CURDIR)/topics/02-transformers/attention-basics/data
DDP_DATA_DIR ?= $(CURDIR)/topics/05-ai-infra/distributed-training-101/data
STEPS ?= 1000
SAMPLES ?= 20
BATCH ?= 32
PRESET ?= demo
NPROC ?= 2

.PHONY: all configure build run run-eval run-repr run-cnn run-residual run-rnn run-lstm run-attention run-attention-torch run-nanogpt run-ddp run-nanogpt-sample clean rebuild docs docs-dev docs-build

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

# rnn-seq：DEMO=unroll|temp|list（默认 unroll；xor 时改走 unroll）
run-rnn: build
	@demo="$(DEMO)"; \
	if [ "$$demo" = "xor" ]; then demo=unroll; fi; \
	RNN_DATA_DIR="$(RNN_DATA_DIR)" $(BUILD_DIR)/rnn_demo $$demo

# lstm-seq：DEMO=gates|recall|list（默认 gates；xor 时改走 gates）
run-lstm: build
	@demo="$(DEMO)"; \
	if [ "$$demo" = "xor" ]; then demo=gates; fi; \
	$(BUILD_DIR)/lstm_demo $$demo

# attention-basics：人名上训最小 GPT。STEPS= / BATCH= / SAMPLES=；首次下载 names.txt → ATTN_DATA_DIR
run-attention:
	ATTN_DATA_DIR="$(ATTN_DATA_DIR)" STEPS="$(STEPS)" BATCH="$(BATCH)" SAMPLES="$(SAMPLES)" \
		python3 topics/02-transformers/attention-basics/code/micro_gpt.py

run-attention-torch:
	ATTN_DATA_DIR="$(ATTN_DATA_DIR)" STEPS="$(STEPS)" BATCH="$(BATCH)" SAMPLES="$(SAMPLES)" \
		python3 topics/02-transformers/attention-basics/code/micro_gpt_torch.py

# distributed-training-101：nanoGPT Shakespeare。PRESET=demo|shakespeare；STEPS= / BATCH= 仅命令行覆盖时传入
run-nanogpt:
	DDP_DATA_DIR="$(DDP_DATA_DIR)" PRESET="$(PRESET)" \
		$(if $(filter command line,$(origin STEPS)),STEPS="$(STEPS)") \
		$(if $(filter command line,$(origin BATCH)),BATCH="$(BATCH)") \
		python3 topics/05-ai-infra/distributed-training-101/code/train.py

run-ddp:
	DDP_DATA_DIR="$(DDP_DATA_DIR)" PRESET="$(PRESET)" \
		$(if $(filter command line,$(origin STEPS)),STEPS="$(STEPS)") \
		$(if $(filter command line,$(origin BATCH)),BATCH="$(BATCH)") \
		python3 topics/05-ai-infra/distributed-training-101/code/train.py --nproc $(NPROC)

run-nanogpt-sample:
	DDP_DATA_DIR="$(DDP_DATA_DIR)" \
		python3 topics/05-ai-infra/distributed-training-101/code/train.py --sample-only

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

docs: docs-dev

docs-dev:
	cd site && npm install && npm run docs:dev

docs-build:
	cd site && npm install && npm run docs:build
