# 视频大纲：CNN 基础

目标时长：8–12 分钟

## 钩子（15–30s）

「上次把 28×28 拍成 784 维向量；为什么识别衣服不用把每个像素都连到所有权重上？」

## 第一段：直觉

局部性、平移相似模式、权值共享。对照 `filter`：同一 3×3 核滑过整图。

## 第二段：结构

卷积滑动示意图 + `param` 里 Dense vs Conv 参数量。点到：stride / padding / 通道。

## 第三段：实验演示

```bash
make run-cnn                 # filter：Sobel ASCII
make run-cnn DEMO=param      # 参数量
make run-cnn DEMO=fashion    # 小 CNN 训 Fashion 子集
```

强调：同子集对照 MLP；结构是 LeNet-5 风格。看参数与墙钟，别只盯着 Fashion 准确率。

## 结尾引导

- 下一站：RNN，或直接跳 Attention / PyTorch 对照 `nn.Conv2d`（那里墙钟才会正常）
- 路径：`topics/01-classic-dl/cnn-basics/`
