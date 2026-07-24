# 残差连接设计文档

纯 C++。`identity` 用 `mlp_core`；`fashion` 复用 `cnn_core` 的 `Conv2d`。

## 1. 范围

| 组件 | 用途 |
|------|------|
| `ResidualBlock` | MLP：\(y=x+F(x)\)，供概念 |
| `DeepMap` | identity toy |
| `ResidualConvBlock` | 卷积：\(y=x+\mathrm{Conv}(\mathrm{ReLU}(\mathrm{Conv}(x)))\) |
| `DeepCnn` | Fashion：plain trunk vs residual trunk |

非目标：完整 ResNet、BN、ImageNet。

## 2. 残差块

MLP / Conv 两种实现同一式子；支路末端 **近零初始化**。

## 3. Fashion CNN 结构

```
28×28 → Conv 1→C k=3 p=1 → ReLU → Pool       # 14×14
     → trunk: plain(2N Conv+ReLU) 或 N×ResBlock  # 同 #Conv
     → Pool → Flatten(C·7·7) → Dense → 10
```

默认：C=8，N=3（trunk 6 Conv），子集 2k/400，5 epoch。

## 4. Demo

| Demo | 对照 |
|------|------|
| `identity` | plain 16 Linear vs residual 8 blocks |
| `fashion` | plain-cnn vs residual-cnn（同 Conv 预算） |

## 5. 运行

```bash
make run-residual
make run-residual DEMO=fashion
```
