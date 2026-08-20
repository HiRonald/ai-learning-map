# attention-basics 实验代码

同一套人名 GPT：`micro_gpt.py` 手写 Tensor；`micro_gpt_torch.py` 用 PyTorch `nn` / autograd 对照。

| 命令 | 做什么 |
|------|--------|
| `make run-attention` | numpy 手写自动微分 |
| `make run-attention-torch` | PyTorch 对照版 |
| `make run-attention STEPS=200` | 缩短步数（torch 目标同样认 `STEPS=` / `BATCH=`） |

数据目录：`ATTN_DATA_DIR`（默认 `topics/02-transformers/attention-basics/data/`）。结构说明见 [notes.md](../notes.md) / [README](../README.md)。
