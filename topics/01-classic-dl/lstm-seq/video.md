# 视频大纲：LSTM

目标时长：8–12 分钟

## 钩子（15–30s）

「开头一个数，中间全是噪声，末尾再问——RNN 为什么容易忘？LSTM 多了一条什么路？」

## 第一段：直觉

\(c\) 慢通道 + forget / input / output。对照 `gates` 表：写入 → 保持。

## 第二段：结构

公式一行带过；强调与 Vanilla RNN「只有 \(h\)」的差别。

## 第三段：实验演示

```bash
make run-lstm
make run-lstm DEMO=recall
```

指出 LSTM vs RNN 的测试 MAE。

## 结尾引导

- 收获：门控让长依赖可学
- 下一站：encoder–decoder（Seq2Seq）
- 路径：`topics/01-classic-dl/lstm-seq/`
