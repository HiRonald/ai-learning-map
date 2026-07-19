# 视频大纲：从零实现 MLP

目标时长：10–14 分钟

## 钩子（15–30s）

「XOR 只有 4 个样本，但单层感知机学不会——我们用纯 C++ 手写小网络，再下载 Fashion-MNIST，让终端用 ASCII 认出鞋子和包。」

## 第一段：直觉

- 为什么需要非线性隐藏层
- 训练闭环：前向 → 损失 → 反向 → 更新

## 第二段：结构

对照仓库：

- `code/nn/`：`Matrix` / `Layer` / `Mlp`
- `code/demos/`：`xor` → `sine` → `fashion`

易错点：

1. MSE 梯度是 `(ŷ − y) / N`
2. 回传输入梯度要用 **更新前** 的 `W`

## 第三段：实验演示

```bash
make run                 # xor
make run DEMO=sine       # 回归
make run DEMO=fashion    # 真数据 + ASCII 画廊
```

- `xor` / `sine`：玩具数据跑通闭环（分类 vs 回归）
- `fashion`：下载 IDX、`softmax_ce`、test acc、hit/miss ASCII

## 结尾引导

- 收获：可运行闭环 + 三条递进 demo
- 下一集预告：CNN——卷积到底在共享什么
- 路径：`topics/00-foundations/mlp-from-scratch/`
