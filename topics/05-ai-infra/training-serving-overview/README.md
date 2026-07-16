# 训练与推理鸟瞰

> 状态：draft · 轨道：AI Infra · 难度：L2

## 要解决什么问题

从「能在笔记本上 train 一个小网络」到「别人能稳定调用你的模型」，中间缺了哪些环节？

## 直觉

一条流水线：数据管线 → 训练（单机/分布式）→ 导出/转换 → 推理优化 → Serving → 观测与迭代。

## 结构 / 公式

（写作中）一张端到端架构简图；每个框只写职责与常见关键词（DataLoader、checkpoint、ONNX/TensorRT、gRPC/HTTP）。

## 实验

代码目录：`code/`（可选：最小 FastAPI 加载玩具权重的示意）

## 局限与下一步

- 不覆盖完整 K8s 生产方案
- 下一站：[分布式训练 101](../distributed-training-101/) / [推理优化 101](../inference-optimization-101/) → [具身最小闭环](../../06-embodied/sense-plan-act/)
