# Attention 基础设计文档

numpy 手写 **Tensor 自动微分** + 最小 decoder GPT。在 makemore 人名上走通训练/采样。

灵感：Karpathy micrograd（标量自动微分）+ nanoGPT / makemore（最小 GPT、人名 LM）。无 PyTorch。

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
    └── micro_gpt.py      # Tensor → 算子 → GPT → 训练 / 采样
```

```
make run-attention
```

## 3. Tensor 自动微分

相对 `mlp-from-scratch` 的手写矩阵反向：这里每个算子是图上一个节点，`backward()` 拓扑序回传。节点从标量换成了 ndarray。

实现要点：

- 广播反向用 `_unbroadcast` 把梯度求和收回原始 shape
- 1D/2D 乘法用 `np.dot`（macOS Accelerate 的 `np.matmul` 会误报 overflow）
- Attention 的 \(QK^\top\) 在 GPT 路径里写成逐元素乘再 `sum`，避开 3D `dot` 语义

## 4. 因果 Attention 与 KV cache

每次只进 1 个 token，把该步的 K/V append 进 cache，再对历史做点积。这和整段一次算、再加下三角 mask 等价。训练时 cache 留在图上，好让后续步的 loss 回传到更早的 K/V；推理时 `detach()`，不建图。

## 5. Block 约定

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
| 默认步数 | 1000（每步 1 个名字，无 batch） |

## 6. Demo

下载 `names.txt` → `topics/02-transformers/attention-basics/data/`。采样温度默认 0.1。人名质量不是门槛；看 loss 下降、能吐出字母串即可。

## 7. 刻意不做

PyTorch `nn.Transformer`、BPE、dropout、GELU、多卡、刷 Shakespeare / OpenWebText、生产级 KV cache。

## 8. 运行

```bash
make run-attention
make run-attention STEPS=200
```
