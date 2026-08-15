# Seq2Seq：编码器–解码器如何映射序列

> 状态：draft · 轨道：Classic DL · 难度：L2 · 语言：C++ 或 PyTorch

## 要解决什么问题

输入一条序列、输出另一条（长度可以不同）时，怎样先压成上下文、再逐步生成？

## 直觉

Encoder 读完得到 \(h\)（或 \(c,h\)）；Decoder 以之为起点，逐步吐出目标序列——经典「翻译机」骨架。

## 结构 / 公式

（写作中）RNN/LSTM encoder–decoder；收尾可点到 attention，再接到 Transformers 轨。

## 实验

代码目录：`code/`（待添加：最小字符级或短序列拷贝/翻译玩具）

## 局限与下一步

- 不覆盖：完整 NMT、Transformer encoder–decoder 训练
- 前置：[LSTM](../lstm-seq/)（或至少 [RNN](../rnn-seq/)）
- 下一站：[Attention](../../02-transformers/attention-basics/)
