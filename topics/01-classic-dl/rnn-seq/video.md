# 视频大纲：RNN 序列

目标时长：8–12 分钟

## 钩子（15–30s）

「给你过去两周每天的最低气温——明天大概几度？」

## 第一段：直觉

隐藏状态 = 读到现在为止的小结；每天共用同一套权重。

## 第二段：结构

画展开图（14 步可缩成 3 步示意）：\(x_t, h_{t-1} \rightarrow h_t\)，最后 \(h_T \rightarrow \hat{y}\)。  
点一句：训练时误差要沿时间往回传，因为权重是共享的。

## 第三段：实验演示

```bash
make run-rnn              # 盯 h_t 表
make run-rnn DEMO=temp    # 墨尔本日气温，对比「用昨天当预测」
```

讲清：真实 CSV、滑窗 14→1、Vanilla RNN 回归。

## 结尾引导

- 更长依赖 → Attention  
- 路径：`topics/01-classic-dl/rnn-seq/`
