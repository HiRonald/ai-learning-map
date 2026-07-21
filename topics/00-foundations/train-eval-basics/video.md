# 视频大纲：训练与评测基础

目标时长：6–10 分钟

## 钩子（15–30s）

「训练集准确率 99%，上线却翻车——缺的不是模型，是评测习惯。」

## 第一段：直觉

train / val / test 各管什么；为什么训练日志不该盯 test。

## 第二段：结构

学习曲线：正常一起涨 vs 过拟合分叉；准确率 vs per-class recall。

## 第三段：实验演示

```bash
make run-eval
make run-eval MODE=overfit
make run-eval MODE=earlystop
```

指着表格说：哪一列是 val、哪一行开始分叉、`*` 是 best、早停如何砍掉无效 epoch、最后才碰 test。

## 结尾引导

- 下一站：数据与表征——模型真正吃进去的是什么
- 路径：`topics/00-foundations/train-eval-basics/`
