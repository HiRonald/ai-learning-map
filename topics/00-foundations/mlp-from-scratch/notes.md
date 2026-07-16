# 从零实现的 MLP（多层感知机）设计文档

本项目是一个用纯 C++ 手写的小型神经网络，目标是把 **前向传播、反向传播、损失与参数更新** 串成一条可运行的学习闭环，并用经典的 **XOR** 问题验证。

## 1. 目标与范围

- 实现可训练的全连接多层感知机（MLP）
- 支持 mini-batch 训练与推理
- 矩阵运算、激活函数、损失、层、网络分层清晰
- 训练与推理过程打印关键指标（loss / accuracy / 逐样本预测）
- 不依赖第三方深度学习框架

非目标：GPU、自动微分框架、大规模数据集、生产级性能优化。

## 2. 目录与模块划分

```
topics/00-foundations/mlp-from-scratch/
├── README.md          # 对外讲解（网站正文）
├── notes.md           # 本设计笔记
├── video.md
├── meta.yaml
└── code/              # 可运行源码
    ├── main.cpp
    ├── mlp.h/cpp
    ├── layer.h/cpp
    ├── activator.h/cpp
    └── matrix.h/cpp
```

仓库根目录用 CMake/Makefile 构建本主题的 `code/`（产物在根目录 `build/`）。

依赖关系：

```
main -> Mlp -> Layer -> Activator
                 |         |
                 +-----> Matrix <----+
```

## 3. 数据与形状约定

全项目统一采用 **行样本** 约定：

| 张量 | 形状 | 含义 |
|------|------|------|
| 输入 `X` | `(batch, in)` | 每一行是一个样本 |
| 权重 `W` | `(in, out)` | 全连接参数 |
| 偏置 `b` | `(1, out)` | 对 batch 维广播 |
| 线性输出 `Z` | `(batch, out)` | `Z = XW + b` |
| 激活输出 `A` | `(batch, out)` | `A = f(Z)` |
| 标签 `Y` | `(batch, out)` | XOR 中 `out=1` |

这个约定和常见教学代码（NumPy 风格）一致，方便对照推导。

## 4. 前向传播

对每一层：

\[
Z = XW + b,\quad A = f(Z)
\]

网络把上一层的 `A` 作为下一层的 `X`。最后一层的 `A` 即网络输出 `\hat{Y}`。

### 输出层的特殊处理

隐藏层与输出层在代码里都是 `Layer`，但职责不同：

- **隐藏层**：学习非线性特征，本 demo 使用 `sigmoid`
- **输出层**：单独构造，可换成：
  - `sigmoid`：二分类 / XOR（输出落在 `(0,1)`）
  - `identity` / `linear`：回归
  - 后续可扩展 `softmax`：多分类

也就是说：**输出层不是“没有激活”，而是“按任务选择激活”**。

## 5. 损失函数

当前支持：

### MSE（本 demo 使用）

\[
L = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
\]

\[
\frac{\partial L}{\partial \hat{Y}} = \frac{\hat{Y} - Y}{N}
\]

### BCE（二分类交叉熵）

适用于输出已经是概率的场景；实现里对 `log` 做了数值裁剪，避免 `log(0)`。

## 6. 反向传播（核心）

从损失开始，梯度沿层逆序回传。对单个全连接层，已知上游梯度 \(\frac{\partial L}{\partial A}\)：

1. \(\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot f'(Z)\)
2. \(\frac{\partial L}{\partial W} = X^{T}\frac{\partial L}{\partial Z}\)
3. \(\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial Z}\)
4. \(\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^{T}\)

然后用梯度下降更新：

\[
W \leftarrow W - \eta \frac{\partial L}{\partial W},\quad
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

注意：计算 \(\frac{\partial L}{\partial X}\) 时必须使用 **更新前** 的 `W`。

## 7. 激活函数导数约定

`Activator::backward(z)` 一律接收预激活值 `z`，返回 `f'(z)`：

| 名称 | \(f(z)\) | \(f'(z)\) |
|------|----------|-----------|
| relu | \(\max(0,z)\) | \(1_{z>0}\) |
| sigmoid | \(1/(1+e^{-z})\) | \(s(1-s)\) |
| tanh | \(\tanh z\) | \(1-\tanh^2 z\) |
| identity | \(z\) | \(1\) |

早期草稿里曾把 `sigmoid` 导数误写成 `z*(1-z)`，那只在输入已是 `s(z)` 时成立；当前实现已改正。

## 8. 参数初始化

权重使用简易 Xavier/Glorot 均匀初始化：

\[
W_{ij} \sim U\left(-\sqrt{\frac{6}{in+out}},\ \sqrt{\frac{6}{in+out}}\right)
\]

偏置初始化为 0。随机数种子固定为 `42`，便于复现实验。

## 9. XOR Demo 配置

- 结构：`2 -> 4 -> 1`
- 激活：隐藏层 / 输出层均为 `sigmoid`
- 损失：`mse`
- 学习率：`1.0`
- epoch：`5000`，batch size：`4`（全批量）

XOR 不是线性可分问题，至少需要一个非线性隐藏层；本配置用于验证整条训练链路是否正确。

## 10. 构建与运行

在**仓库根目录**用 CMake 构建本主题 `code/`（产物进根目录 `build/`）。

```bash
make          # 配置 + 编译
make run      # 编译并运行
make clean    # 删除 build/
```

## 11. 训练/推理日志含义

训练阶段每隔若干 epoch 打印：

- `loss`：当前全体样本上的损失
- `acc`：把预测按 `0.5` 阈值化后的分类准确率

推理阶段打印每个样本的：

- 原始输入
- 真实标签
- 网络原始输出（连续值）
- 阈值化后的 0/1 预测

## 12. 后续可扩展方向

- 学习率衰减 / Adam
- Softmax + 多类交叉熵
- 权重保存与加载（`save` / `load`）
- Dropout、BatchNorm
- 用更大的数据集（如 MNIST 子集）做验证

## 13. 关键修复回顾（相对初稿）

1. 补全 `Matrix` 实现（原先只有头文件声明）
2. 修正权重形状与矩阵乘顺序，支持 `(batch, feature)` 输入
3. 修正 MSE 梯度（原先错误地做了同形状 `dot`）
4. 修正 sigmoid 导数对 `z` 的计算
5. 反向传播中先算输入梯度再更新权重
6. 输出层单独配置激活函数
7. 权重随机初始化与学习率落地
8. 训练/推理日志与设计文档补齐
