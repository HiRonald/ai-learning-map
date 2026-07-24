# CNN 基础：卷积在做什么

> 状态：published · 轨道：Classic DL · 难度：L2 · 语言：C++

## 要解决什么问题

全连接把空间结构拍扁了。卷积如何用 **局部连接 + 权值共享** 更高效地处理图像这类网格数据？

## 直觉

同一套小滤波器在整张图上滑动：在不同位置检测同类局部模式（边缘、纹理等），参数量远小于同等「全连接大层」。

## 结构

```
cnn-basics/
├── README.md / notes.md / video.md / meta.yaml
└── code/
    ├── main.cpp
    ├── cnn/          # Tensor · Conv2d · MaxPool2d · LeNet
    └── demos/        # filter · param · fashion
```

NCHW 张量；`Conv2d` / `MaxPool2d` 手写前向+反向；`fashion` 用 LeNet-5 风格网络。细节见 [notes.md](./notes.md)。

## 例子

| Demo | 作用 | 说明 |
|------|------|------|
| `filter`（默认） | 固定 Sobel / 锐化核 | ASCII 看边缘响应 |
| `param` | Dense vs Conv 参数量 | 同尺寸特征图差几个数量级 |
| `fashion` | 同子集 MLP vs LeNet | 打 params / 墙钟 / acc；复用 MLP 数据 |

```bash
make run-cnn                 # filter
make run-cnn DEMO=param
make run-cnn DEMO=fashion    # 首次若无缓存会下载 Fashion-MNIST
```

预期：`filter` 能看出横/竖边缘；`param` 打印 Dense/Conv 比例；`fashion` 里 LeNet 参数少于 MLP，准确率通常接近或略好，墙钟更慢（朴素卷积）。

## 局限与下一步

- 故意没做：im2col、GPU、BatchNorm、检测/分割；LeNet 用 ReLU+MaxPool 而非原版 tanh+AvgPool
- Fashion 轮廓对 MLP 太友好，**不是**刷分擂台；看参数量与归纳偏置
- 墙钟慢是教学实现（嵌套循环）的代价，不是卷积「本质更慢」
- 下一站：[残差连接](../residual-basics/) → [RNN 序列](../rnn-seq/) → [Diffusion 基础](../diffusion-basics/) 或 [Attention](../../02-transformers/attention-basics/)；框架对照见 [PyTorch 基础](../../03-frameworks/pytorch-basics/)
