# cnn-basics 实验代码

纯 C++ 手写最小 CNN，三个 demo：

| 命令 | Demo |
|------|------|
| `make run-cnn` | `filter` — 固定核可视化 |
| `make run-cnn DEMO=param` | Dense vs Conv 参数量 |
| `make run-cnn DEMO=fashion` | Fashion-MNIST 小网络 |

Fashion 数据与 `mlp-from-scratch` 共用（`MLP_DATA_DIR`）。实现说明见上级 [notes.md](../notes.md)。
