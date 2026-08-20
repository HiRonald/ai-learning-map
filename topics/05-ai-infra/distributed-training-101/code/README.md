# distributed-training-101 实验代码

Karpathy nanoGPT 结构的字符级莎士比亚 GPT。改超参优先打开 `train.sh` / `sample.sh` 顶部的变量。

| 命令 | 做什么 |
|------|--------|
| `./train.sh` | 训练（默认 shakespeare 预设） |
| `./train.sh --steps 200 --batch 32` | 命令行覆盖脚本里的值 |
| `NPROC=2 ./train.sh` 或脚本里 `NPROC=2` | DDP |
| `RESUME=1` 写进脚本，或 `./train.sh --resume` | 从 `ckpt.pt` 续训 |
| `./sample.sh` | 推理 |
| `./sample.sh --prompt "HAMLET:" --samples 5` | 换前缀、多采几段 |
| `make run-nanogpt` | 仓库根目录的薄封装（默认 demo） |

数据目录：`DDP_DATA_DIR`（默认 `../data/`）。结构说明见 [notes.md](../notes.md) / [README](../README.md)。
