# 视频大纲：分布式训练 101

目标时长：8–12 分钟

## 钩子（15–30s）

「训练日志里的 all-reduce 是在同步什么？多买几张卡，模型是被切开了，还是每张卡都有一份完整副本？」

## 第一段：直觉

数据并行：复制模型、拆样本、平均梯度。对照一句模型并行（层/矩阵切开）就够，不展开。

## 第二段：载体

nanoGPT 字符级莎士比亚：长流切 `(B, T)`，GPT-2 块。点明和人名玩具 GPT 的差别（padding vs 切块）。

## 第三段：实验演示

```bash
make run-nanogpt          # 单进程，loss 在掉
make run-ddp              # 两进程；强调 tokens/iter 乘了 world_size
```

看 rank 0 的 loss、一段 `ROMEO:` 采样。Mac 上说明走的是 gloo+CPU。

## 结尾引导

- 收获：DDP 同步的是梯度；全局 batch = 每卡 × 累积 × 卡数
- 下一站：推理侧延迟 / 吞吐 / KV cache
- 路径：`topics/05-ai-infra/distributed-training-101/`
