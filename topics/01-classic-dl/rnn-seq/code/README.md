# rnn-seq 实验代码

| 命令 | 做什么 |
|------|--------|
| `make run-rnn` | 不训练：打印短序列上每一步的隐藏状态 |
| `make run-rnn DEMO=temp` | 日最低气温：过去 14 天 → 预测明天 |

数据缓存目录由 `RNN_DATA_DIR` 指定（Makefile 默认指向本主题 `data/`）。结构说明见 [notes.md](../notes.md) / [README](../README.md)。
