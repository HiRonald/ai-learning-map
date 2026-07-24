# 从零实现的 MLP（多层感知机）设计文档

纯 C++ 手写小型神经网络，目标是把 **前向 → 损失 → 反向 → 更新** 串成可运行闭环，用三个 demo 验证：`xor` → `sine` → `fashion`。

## 1. 目标与范围

- 可训练的全连接 MLP + mini-batch
- 矩阵 / 激活 / 损失 / 层 / 网络分层清晰
- 不依赖第三方深度学习框架

非目标：GPU、自动微分、大规模刷分、生产级优化。

## 2. 目录

```
topics/00-foundations/mlp-from-scratch/
├── README.md / notes.md / video.md / meta.yaml
├── data/                    # 下载缓存（gitignore）
└── code/
    ├── main.cpp             # CLI
    ├── nn/                  # 核心实现
    │   ├── matrix.*
    │   ├── activator.*
    │   ├── layer.*
    │   └── mlp.*
    └── demos/
        ├── common.*         # 拼网、指标、训练小工具
        ├── download.*       # curl + gunzip
        ├── idx_io.*         # Fashion/MNIST IDX
        ├── xor.*
        ├── sine.*
        └── fashion.*
```

仓库根目录 CMake/Makefile 构建（产物在 `build/`）。

```
main -> demos/* -> nn::Mlp -> Layer -> Activator
                              |         |
                              +-----> Matrix <----+
```

## 3. 数据与形状约定

行样本：

| 张量 | 形状 | 含义 |
|------|------|------|
| 输入 `X` | `(batch, in)` | 每一行是一个样本 |
| 权重 `W` | `(in, out)` | 全连接参数 |
| 偏置 `b` | `(1, out)` | 对 batch 维广播 |
| 线性输出 `Z` | `(batch, out)` | `Z = XW + b` |
| 激活输出 `A` | `(batch, out)` | `A = f(Z)` |
| 标签 `Y` | `(batch, out)` | 二分类/回归 `out=1`；Fashion one-hot `out=10` |

## 4. 前向传播

\[
Z = XW + b,\quad A = f(Z)
\]

- **隐藏层**：非线性特征（sigmoid / tanh / relu）
- **输出层**：按任务选激活——`sigmoid`（二分类）、`identity`（回归或多类 logits）

多类时 softmax 做在 `softmax_ce` 损失里，不是单独一层激活。

## 5. 损失函数

### MSE

\[
L = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2,\quad
\frac{\partial L}{\partial \hat{Y}} = \frac{\hat{Y} - Y}{N}
\]

### BCE

输出已是概率；`log` 有数值裁剪。

### Softmax CE

输出为 logits，标签 one-hot；对 logits 的梯度：

\[
\frac{\partial L}{\partial Z} = \frac{\mathrm{softmax}(Z) - Y}{N}
\]

## 6. 反向传播（核心）

已知 \(\frac{\partial L}{\partial A}\)：

1. \(\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot f'(Z)\)
2. \(\frac{\partial L}{\partial W} = X^{T}\frac{\partial L}{\partial Z}\)
3. \(\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial Z}\)
4. \(\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^{T}\)（必须用 **更新前** 的 `W`）

\[
W \leftarrow W - \eta \frac{\partial L}{\partial W},\quad
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

## 7. 激活导数约定

`Activator::backward(z)` 一律对预激活 `z` 求 `f'(z)`：

| 名称 | \(f(z)\) | \(f'(z)\) |
|------|----------|-----------|
| relu | \(\max(0,z)\) | \(1_{z>0}\) |
| sigmoid | \(1/(1+e^{-z})\) | \(s(1-s)\) |
| tanh | \(\tanh z\) | \(1-\tanh^2 z\) |
| identity | \(z\) | \(1\) |

## 8. 参数初始化

Xavier/Glorot 均匀：\(W_{ij} \sim U(-\sqrt{6/(in+out)},\ \sqrt{6/(in+out)})\)；偏置 0；RNG 种子 `42`。

## 9. Demo 配置

| Demo | 结构 | 激活 | 损失 | lr | 数据 |
|------|------|------|------|----|------|
| `xor` | `2→4→1` | sigmoid / sigmoid | mse | 1.0 | 4 样本，全批量 |
| `sine` | `1→32→32→1` | tanh / identity | mse | 0.05 | 64 点 `sin(x)` |
| `fashion` | `784→128→64→10` | relu / identity | softmax_ce | 0.15 | Fashion-MNIST 子集 4k/800 |

## 10. 构建与运行

```bash
make
make run                 # xor
make run DEMO=sine
make run DEMO=fashion    # 首次下载
make clean
```

## 11. 日志含义

- 分类：`loss` + `acc`（二分类阈值 0.5；多类 argmax）
- 回归：`loss` + `mae`
- `fashion` 另打印 per-class recall 与 ASCII hit/miss 画廊

## 12. 后续可扩展

- Adam / 学习率衰减、权重存取、Dropout / BatchNorm
- 与 [CNN 基础](../../01-classic-dl/cnn-basics/) 对照同一 Fashion 数据（`make run-cnn DEMO=fashion`）
