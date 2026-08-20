# distributed-training-101 实验代码

Karpathy nanoGPT 结构的字符级莎士比亚 GPT；`train.py` 单进程和 `--nproc` DDP 共用。

| 命令 | 做什么 |
|------|--------|
| `make run-nanogpt` | 单进程 demo 预设 |
| `make run-ddp` | `--nproc 2`（可用 `NPROC=`） |
| `make run-nanogpt PRESET=shakespeare` | 原版 6/6/384、5000 步 |
| `make run-nanogpt STEPS=50` | 缩短步数 |
| `make run-nanogpt-sample` | 加载 `data/ckpt.pt` 采样 |

数据目录：`DDP_DATA_DIR`（默认 `topics/05-ai-infra/distributed-training-101/data/`）。结构说明见 [notes.md](../notes.md) / [README](../README.md)。
