# 从零实现 MLP

> 状态：published · 轨道：Foundations · 难度：L1 · 语言：C++

## 要解决什么问题

不依赖任何深度学习框架，手写一个可训练的多层感知机，完整走通：

**前向传播 → 损失 → 反向传播 → 参数更新**

三个递进例子：冒烟（XOR）→ 回归（sine）→ 真数据（Fashion-MNIST）。

## 直觉

线性模型画不出 XOR 的分界线；加一层非线性隐藏层后，网络可以先把输入「扭」到更好分隔的空间，再做分类或回归。

本主题关心的不是刷准确率，而是你能指着代码说出：梯度从损失回到每一层权重时经过了哪些步骤。

## 结构

```
mlp-from-scratch/
├── README.md / notes.md / video.md / meta.yaml
├── code/
│   ├── main.cpp          # CLI：选择 demo
│   ├── nn/               # 核心：Matrix / Activator / Layer / Mlp
│   └── demos/            # xor · sine · fashion
└── data/                 # 下载缓存（gitignore）
```

形状约定（行样本）：`X (batch, in)`，`W (in, out)`，`b (1, out)`。一层前向：`Z = XW + b`，`A = f(Z)`。

输出层按任务选激活：二分类 `sigmoid`，回归 / 多类 logits 用 `identity`（多类配合 `softmax_ce`）。

细节见 [notes.md](./notes.md)。

## 例子

| Demo | 任务 | 网络 | 说明 |
|------|------|------|------|
| `xor`（默认） | 4 样本 XOR | `2→4→1` | 冒烟：非线性隐藏层必要 |
| `sine` | 拟合 `sin(x)` | `1→32→32→1` | 回归：`identity` 输出 |
| `fashion` | Fashion-MNIST 10 类 | `784→128→64→10` | 需下载；ASCII 画廊 |

```bash
# 仓库根目录
make run                 # xor
make run DEMO=sine
make run DEMO=fashion    # 首次 curl ~30MB → data/
```

预期：`xor` → 100% acc；`sine` → MAE ≲ 0.08；`fashion` 子集 test acc ≳ 70%，终端能认出鞋子/包，也会把 Shirt 和 T-shirt 搞混。

## 局限与下一步

- 故意没做：GPU、自动微分、Adam、全量 60k 刷分
- `fashion` 用子集是为了纯 C++ 能在笔记本跑完；MLP 也没有卷积的空间归纳偏置
- 下一站：[训练与评测](../train-eval-basics/) → [数据与表征](../data-and-representation/) → [CNN](../../01-classic-dl/cnn-basics/) / [RNN](../../01-classic-dl/rnn-seq/)
