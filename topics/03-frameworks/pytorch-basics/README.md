# PyTorch 基础：对照重写 MLP / XOR

> 状态：draft · 轨道：Frameworks · 难度：L2

## 要解决什么问题

把 [从零 MLP](../../00-foundations/mlp-from-scratch/) 里手写的前向、损失、反向，映射到 PyTorch 的 `nn.Module` / `autograd` / `optimizer`。

## 直觉

框架替你做的是：张量加速、自动求导、常用层与优化器；**训练闭环的形状没有变**。

## 结构 / 公式

（写作中）一张对照表：手写 `Layer` ↔ `nn.Linear`，手写梯度更新 ↔ `loss.backward(); opt.step()`。

## 实验

代码目录：`code/`（待添加）

## 局限与下一步

- 不覆盖分布式 DataParallel 细节
- 下一站：[提示与上下文](../../04-llm-apps/prompt-and-context/)（或先去 [训练与推理鸟瞰](../../05-ai-infra/training-serving-overview/)）
