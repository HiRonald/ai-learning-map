# Attention 基础设计文档

numpy 手写 **Tensor 自动微分** + 最小 decoder GPT；同目录 `micro_gpt_torch.py` 用 `nn` / autograd 对照同一套超参。

灵感：Karpathy micrograd（标量自动微分）+ nanoGPT / makemore（最小 GPT、人名 LM）。

## 1. 一句话对照

| 问题 | 答案 |
|------|------|
| 用什么网络？ | 1 层、4 头、\(d=16\) pre-norm GPT |
| 什么数据？ | 字符级人名；`<EOS>` 作 BOS/EOS |
| 训练目标？ | 逐步 next-token NLL |

## 2. 目录

```
topics/02-transformers/attention-basics/
├── README.md / notes.md / video.md / meta.yaml
├── data/                 # names.txt 缓存（gitignore）
└── code/
    ├── micro_gpt.py         # Tensor → 算子 → GPT
    └── micro_gpt_torch.py   # nn.Module 对照
```

```
make run-attention
make run-attention-torch
```

## 3. Tensor 自动微分

相对 `mlp-from-scratch` 的手写矩阵反向：这里每个算子是图上一个节点，`backward()` 拓扑序回传。节点从标量换成了 ndarray。

实现要点：

- 广播反向用 `_unbroadcast` 把梯度求和收回原始 shape
- 1D/2D 乘法用 `np.dot`（macOS Accelerate 的 `np.matmul` 会误报 overflow）
- Attention 的 \(QK^\top\) 在 GPT 路径里写成逐元素乘再 `sum`，避开 3D `dot` 语义

`micro_gpt_torch.py` 把这一层换成框架：`nn.Linear` / `nn.Embedding` / `loss.backward()` / `AdamW`。多头仍手写 `q @ k.transpose(-2,-1)`，不用 `nn.MultiheadAttention`。

## 4. 变长 batch：右 padding + 因果 mask

名字长短不一，**不能**把不同长度的 1D 序列直接叠成一个张量——矩阵运算要矩形 `(B, T)`。

实际做法（decoder-only 训练的常规写法）：

1. 本 batch 里按最长的那条 **右 padding**（短的右边补 `eos`）
2. 加下三角因果 mask：位置 \(t\) 只看 \(1..t\)
3. loss 只在真实 token 上平均（`mask` 丢掉 pad）

右 padding + 因果 mask 有个方便之处：真 token 全在左边，**看不到**右边的 pad，所以不必再单独做 key padding mask。HuggingFace / 多数因果 LM 都这么干。

另一路是 nanoGPT 那种：把语料拼成一条长流，切固定 `block_size` 块——永远对齐，零 padding。人名这种「一条样本一个名字」用 pad 更自然。

训练一次前向整段 `(B, T)`；推理逐步生成（本玩具每次把已生成前缀再跑一遍。生产上会用 KV cache 避免重复算过去的 K/V）。

## 5. 多头

不是 \(H\) 套独立的小矩阵，而是 **一次** \(W_Q,W_K,W_V\in\mathbb{R}^{C\times C}\)，再 `reshape` 成 `(B, H, T, D)`（`C = H·D`）。每个头自己一份 `(T, T)` 权重、自己的 softmax，最后 `concat` 回 `(B, T, C)`，经 \(W_O\) 混个头之间的信息。

单头只有一张路由表；多头是 \(H\) 张，可以同时学「看前一个字母 / 看词首 / …」这类不同模式。形状：

```text
x, Q, K, V     (B, T, 16)
split_heads    (B, 4, T, 4)
scores/weights (B, 4, T, T)   ← 4 张因果注意力表
merge_heads    (B, T, 16)
```

## 6. Block 约定

Pre-norm（LLaMA 风格 RMSNorm，无均值中心、无 gain）：

```
x = wte[tok] + wpe[pos]
for layer:
    x = x + attn(rmsnorm(x))
    x = x + mlp(rmsnorm(x))   # ReLU，隐层 4d
x = rmsnorm(x)
logits = lm_head(x)
```

线性层无 bias；AdamW（\(\beta_1=0.9,\beta_2=0.999\), wd=0.01）。

| 项 | 值 |
|----|-----|
| `emb_dim` | 16 |
| `layer_num` | 1 |
| `head_num` | 4 |
| 优化 | AdamW，lr=0.001 |
| batch | 32（右 padding 到本 batch 最长） |
| 默认步数 | 1000 |

## 7. Demo

下载 `names.txt` → `topics/02-transformers/attention-basics/data/`。采样温度默认 0.1。人名质量不是门槛；看 loss 下降、能吐出字母串即可。

## 8. 刻意不做

PyTorch `nn.Transformer`、BPE、OpenWebText、生产级 KV cache。Shakespeare 长流切块 + GELU / dropout / 多卡 DDP 见 [分布式训练 101](../../05-ai-infra/distributed-training-101/)。

## 9. 运行

```bash
make run-attention
make run-attention-torch
make run-attention STEPS=200
make run-attention BATCH=8
```
