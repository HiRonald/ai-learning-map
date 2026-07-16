# 根目录薄封装：C++ 主题走 CMake；文档站走 npm（site/）
BUILD_DIR := build

.PHONY: all configure build run clean rebuild docs docs-dev docs-build

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build $(BUILD_DIR)

run: build
	cmake --build $(BUILD_DIR) --target run

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

docs: docs-dev

docs-dev:
	cd site && npm install && npm run docs:dev

docs-build:
	cd site && npm install && npm run docs:build
