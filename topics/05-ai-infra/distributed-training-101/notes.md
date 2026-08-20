# 分布式训练 101 设计文档

PyTorch **DDP 数据并行**；载体是 Karpathy nanoGPT 的字符级 Tiny Shakespeare（不是 GPT-2 预训练权重，也不是 OpenWebText）。

## 1. 一句话对照

| 问题 | 答案 |
|------|------|
| 用什么网络？ | nanoGPT / GPT-2 decoder：LayerNorm、GELU、权重共享 |
| 什么数据？ | Tiny Shakespeare 字符级；拼成一条长流再切块 |
| 训练目标？ | 逐步 next-token cross-entropy |
| 本课要摸的系统点？ | 模型复制 + 数据分片 + backward 里 all-reduce |

## 2. 目录

```
topics/05-ai-infra/distributed-training-101/
├── README.md / notes.md / video.md / meta.yaml
├── data/                 # input.txt / ckpt.pt（gitignore）
└── code/
    ├── model.py          # GPT
    └── train.py          # 数据 + DDP 训练 + 采样
```

```
make run-nanogpt
make run-ddp
make run-nanogpt PRESET=shakespeare
```

## 3. 和 attention-basics 的差异

[attention-basics](../../02-transformers/attention-basics/notes.md) 刻意做成最小可读 GPT。本主题换成 nanoGPT 原结构，好直接对照 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的 `model.py` 与 `config/train_shakespeare_char.py`。

| | 人名玩具 GPT | 本主题 nanoGPT |
|--|--------------|----------------|
| 数据 | 一条样本一个名字，右 padding | 全文一条流，随机 `(B,T)` 窗口 |
| Norm | RMSNorm（无 gain） | `nn.LayerNorm` |
| MLP | ReLU | GELU |
| QKV | 三个 `Linear` | 一个 `c_attn` 再 `split` |
| 输出 | 独立 `lm_head` | 与 `wte` 绑同一份权重 |
| 规模 | 1 层 \(d=16\) | demo 4×128；shakespeare 6×384 |

长流切块永远是矩形，所以 **零 padding**。代价是窗口可以跨过剧本场景边界——对「学会莎士比亚的用词」够用。

## 4. 数据并行在算什么

每个 rank：

1. 持有 **完整** 模型副本（参数相同）
2. 抽自己的 batch（本实现用 `seed = 1337 + rank`，和 nanoGPT 一样近似无重叠）
3. 前向、算 loss、`backward()`
4. DDP hook 对每份梯度做 **all-reduce 平均**
5. 各 rank 用同一份平均梯度 `optimizer.step()` → 副本继续保持一致

所以：

\[
\text{tokens/iter} = B_{\text{per rank}} \times T \times \text{grad\_accum} \times \text{world\_size}
\]

两卡不是「把模型劈开」，是 **全局 batch 翻倍**（或墙钟减半，取决于你是否把 per-rank batch 砍半以保持全局 batch 不变）。本脚本保持 per-rank batch 不变，所以 `make run-ddp` 的全局 batch 是单进程的 `NPROC` 倍——loss 曲线不会和单进程逐点重合，这是特征不是 bug。

梯度累积：`grad_accum > 1` 时，前 `n-1` 个 micro-batch 包在 `model.no_sync()` 里，避免每步都通信；最后一次 backward 才 all-reduce。等效于更大 batch、更少通信次数。

## 5. 进程组与设备

| 场景 | backend | device |
|------|---------|--------|
| `python3 train.py` | 无 | `cuda` > `mps` > `cpu` |
| `python3 train.py --nproc N` + NVIDIA | NCCL | `cuda:LOCAL_RANK` |
| `python3 train.py --nproc N` + Mac / 无 GPU | gloo | CPU（MPS 不进进程组） |

`--nproc N` 时父进程 `mp.spawn` 出 N 个 worker，并写上 `RANK` / `WORLD_SIZE` / `MASTER_ADDR`。若你已经用 `torchrun` 启动，这些变量已在环境里，脚本不会再套一层 spawn。只 rank 0 打日志、存 `ckpt.pt`、采样。

## 6. 超参

### demo（默认，笔记本）

| 项 | 值 |
|----|-----|
| 层 / 头 / \(d\) | 4 / 4 / 128 |
| `block_size` | 128 |
| batch | 32 |
| 步数 | 400 |
| dropout | 0 |
| 优化 | AdamW，warmup 20 + cosine 到 \(10^{-4}\) |

约 0.8M 参（不含位置 embed）。看 loss 下降、能吐出英文碎片即可。

### shakespeare（Karpathy 原配置）

`config/train_shakespeare_char.py`：

| 项 | 值 |
|----|-----|
| 层 / 头 / \(d\) | 6 / 6 / 384 |
| `block_size` | 256 |
| batch | 64 |
| 步数 | 5000 |
| dropout | 0.2 |
| lr / min_lr / warmup | \(10^{-3}\) / \(10^{-4}\) / 100 |
| \(\beta_2\) | 0.99（token/iter 少，把二阶动量放慢） |

约 10.6M 参。语料很小，**会过拟合**；`always_save_checkpoint = False` 在原仓库是「只在 val 变好时存」，本课每次跑完存一份方便 `--sample-only`。

## 7. 刻意不做

FSDP / ZeRO-3、张量并行、流水线并行、多机 IP、`torch.compile`、FlashAttention 手写 kernel、BPE / GPT-2 权重、wandb。推理侧 KV cache 见 [推理优化 101](../inference-optimization-101/)。

## 8. 运行

```bash
make run-nanogpt
make run-ddp NPROC=2
make run-nanogpt PRESET=shakespeare
make run-nanogpt STEPS=50 BATCH=8
make run-nanogpt-sample
```
