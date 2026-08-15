# attention-basics 实验代码

numpy 手写 Tensor 自动微分 + 最小 GPT。需已安装 `numpy`。

| 命令 | 做什么 |
|------|--------|
| `make run-attention` | 人名 LM；首次下载 `names.txt` |
| `make run-attention STEPS=200` | 缩短步数 |

数据目录：`ATTN_DATA_DIR`（默认 `topics/02-transformers/attention-basics/data/`）。结构说明见 [notes.md](../notes.md) / [README](../README.md)。
