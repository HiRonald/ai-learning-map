# 残差连接：为什么深了还能训

> 状态：published · 轨道：Classic DL · 难度：L2 · 语言：C++

## 要解决什么问题

网络一层层叠深之后，训练容易崩或收益变差。残差连接（\(y = x + F(x)\)）到底押了什么注，让深度重新变得可训？

## 直觉

默认走一条 **恒等捷径**：块只需学习「相对 \(x\) 的增量」\(F(x)\)。CNN 管空间结构；残差管深度与优化。

## 结构

```
residual-basics/
├── README.md / notes.md / video.md / meta.yaml
└── code/
    ├── residual/   # MLP ResidualBlock · DeepMap · ResidualConvBlock · DeepCnn
    └── demos/      # identity · fashion
```

## 例子

| Demo | 任务 | 说明 |
|------|------|------|
| `identity`（默认） | 深 MLP 学 \(y=x\) | 概念课：残差默认恒等 |
| `fashion` | 同深度 **plain CNN vs residual CNN** | 接到 `cnn-basics`；对齐 trunk Conv 数 |

```bash
make run-residual                 # identity
make run-residual DEMO=fashion    # 朴素卷积，约数分钟
```

## 局限与下一步

- 不覆盖：ResNet-50、BN、ImageNet
- 下一站：[RNN](../rnn-seq/) → [Diffusion](../diffusion-basics/) 或 [Attention](../../02-transformers/attention-basics/)
