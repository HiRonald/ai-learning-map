# 实验代码：训练与评测

可执行文件：`train_eval_demo`（仓库根目录 `make run-eval`）。

复用 `mlp-from-scratch` 的 `nn/` 与 IDX/下载工具，本目录只保留：

- `main.cpp` — Fashion-MNIST 的 train/val/test 切分、学习曲线、majority 基线、每类召回

```bash
make run-eval                 # normal
make run-eval MODE=overfit
make run-eval MODE=earlystop  # patience=5 + restore best
```
