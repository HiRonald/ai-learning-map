# 视频大纲：Attention 基础

目标时长：8–12 分钟

## 钩子（15–30s）

「RNN 一步步看，Transformer 为什么能按需看全句——还不许偷看未来？」

## 第一段：直觉

Q / K / V：我在找什么、你有什么、取走什么。路由，不是固定只看上一步。

## 第二段：结构

公式一行带过。点一句：逐步 + KV cache 与下三角因果 mask 等价。

## 第三段：实验演示

```bash
make run-attention
```

看 loss 在掉、能采样即可。

## 结尾引导

- 收获：Attention 是按相似度路由；因果 mask / KV cache 是同一约束的两种写法
- 下一站：LLM 生命周期（预训练 / SFT / 对齐）
- 路径：`topics/02-transformers/attention-basics/`
