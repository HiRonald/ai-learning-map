# LSTM 设计文档

纯 C++ 手写 **LSTM**，用合成 **delayed recall** 验证门控能跨噪声保留早期信号。

## 1. 一句话对照

| 问题 | 答案 |
|------|------|
| 用什么网络？ | LSTM + Dense(1)；同超参打印 Vanilla RNN |
| 什么数据？ | 程序生成：记二进制 \(x_0\)，噪声，末步 query |
| 训练目标？ | \(\hat y \approx x_0\)；MSE；acc@0.5 |

## 2. 目录

```
topics/01-classic-dl/lstm-seq/
├── README.md / notes.md / video.md / meta.yaml
└── code/
    ├── main.cpp
    ├── lstm/
    │   ├── cell.*   # 门控 + c/h；BPTT
    │   └── net.*    # 扫完窗口 → Dense(1)
    └── demos/
        ├── gates.*  # 固定权重，打印门与 c
        └── recall.* # 合成任务
```

## 3. Cell 约定

权重打包列顺序 \([f \mid i \mid g \mid o]\)：

- \(W_x\): \((I,\ 4H)\)，\(W_h\): \((H,\ 4H)\)，\(b\): \((1,\ 4H)\)
- 遗忘门偏置初始化为 \(+1\)

BPTT：从 \(\partial L/\partial h_T\) 回传，\(c\) 与 \(h\) 两条链；累加 \(dW\) 后 `apply_gradients()` 一次。

## 4. Delayed recall

| 项 | 值 |
|----|-----|
| \(T\) | 12 |
| \(x_0\) | \(\{0,1\}\) 均匀 |
| \(x_1..x_{T-2}\) | 噪声 \(U(0,1)\) |
| \(x_{T-1}\) | value=0，query=1 |
| 输入维 | 2：`[value, query]` |
| H | 16 |
| 优化 | SGD，lr=0.1，batch=64，40 epoch |
| 样本 | train 2048 / test 512 |

成功判据（`recall` exit code）：LSTM 测试 **accuracy ≥ 0.95**（相对 ~0.5 基线）。

短 \(T\) 上 Vanilla RNN 也可能学会 latch，因此对照 RNN 只作参考，不以「必赢 RNN」为门槛；机制直觉以 `gates` 为准。

## 5. Demo

### `gates`

手设 H=1：\(f\) 常开、早期写入、后续 \(x=0\) 时 \(c\) 不掉。

### `recall`

训 LSTM；同超参跑 Vanilla RNN 打印；对照「永远猜 0.5」。

## 6. 刻意不做

GRU、UCR/气温刷分、字符 LM、Attention。

## 7. 运行

```bash
make run-lstm
make run-lstm DEMO=recall
```
