# 训练与评测基础

> 状态：published · 轨道：Foundations · 难度：L1 · 语言：C++

## 要解决什么问题

会跑 `loss` 下降，不等于模型可用。如何用 **train / val / test**、过拟合信号和基础指标判断「训对了」？

## 直觉

| 集合 | 角色 |
|------|------|
| train | 更新权重（背题） |
| val | 每个 epoch 盯着看：是否还能泛化；用来选 epoch / 调参 |
| test | **只在最后看一次**；训练过程偷看它 = 把最终考试题练进习惯里 |

训练误差看「背题能力」，验证误差看「迁移能力」；两者拉开时，多半在过拟合。  
准确率好看也不够：Shirt / T-shirt / Coat 的 per-class recall 会揭穿「平均分掩盖偏科」。

## 结构

```
官方 Fashion train 60k ──切分──► train + val
官方 Fashion test  10k ───────► test（最终一次）
```

每个 epoch 同时打 `train_loss/acc` 与 `val_loss/acc`，并写出 CSV 学习曲线。  
网络复用 [mlp-from-scratch](../mlp-from-scratch/) 的 `784→128→64→10`。

## 实验

```bash
# 仓库根目录
make run-eval                   # MODE=normal：20k/5k/5k，10 epoch
make run-eval MODE=overfit      # 3k train + 40 epoch：逼出过拟合
make run-eval MODE=earlystop    # 同 overfit，但 patience=5 早停并恢复最优权重
```

| Mode | 设置 | 预期现象 |
|------|------|----------|
| `normal` | train=20k，10 epoch | train 与 val 大致一起涨 |
| `overfit` | train=3k，40 epoch | train acc 继续爬，val 早早停住甚至掉 |
| `earlystop` | 同 overfit + `patience=5` | val 平台期停训；最终用 best-val 权重评 test |

早停在做什么：盯 **val_acc**；连续 `patience` 个 epoch 无提升 → 停；把权重 **restore** 回最好那一拍（日志里 `*` 标记）。  
这是对抗过拟合里改动最小的一招；weight decay / dropout 留给框架课。

曲线 CSV：`topics/00-foundations/train-eval-basics/data/curve_{mode}.csv`  
Fashion 数据与 MLP 主题共用：`mlp-from-scratch/data/fashion-mnist/`（首次自动下载）。

过关自检：

1. 能说出为什么调参看 val、不看 test  
2. 在 `overfit` 日志里指出 train↑ val 停住的区间  
3. 对比 `overfit` vs `earlystop`：峰值接近，但后者少跑了无效 epoch  
4. 知道为什么还要看 per-class，不能只报一个准确率  

## 局限与下一步

- 不覆盖完整实验设计、weight decay / dropout、统计显著性  
- 下一站：[数据与表征](../data-and-representation/)；或提前插 [PyTorch 基础](../../03-frameworks/pytorch-basics/)
