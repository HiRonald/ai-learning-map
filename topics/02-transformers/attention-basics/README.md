# Attention 基础：信息如何被路由

> 状态：draft · 轨道：Transformers · 难度：L2

## 要解决什么问题

不靠逐步压缩成单一隐状态，序列中每个位置如何按需去「看」其他位置？

## 直觉

Query / Key / Value：用相似度决定从哪里取信息，而不是固定只看上一步。

## 结构 / 公式

（写作中）`Attention(Q, K, V) = softmax(QK^T / √d) V` 的形状级讲解。

## 实验

代码目录：`code/`（待添加：最小自注意力数值例子）

## 局限与下一步

- 不覆盖完整 GPT 训练
- 前置建议先看 [残差连接](../../01-classic-dl/residual-basics/)（Transformer block 默认 \(x + \mathrm{Attn}(x)\)）
- 下一站：[LLM 生命周期](../llm-lifecycle/) → [多模态](../multimodal-basics/) / [高效微调](../efficient-finetune/) → [LLM Apps](../../04-llm-apps/)
