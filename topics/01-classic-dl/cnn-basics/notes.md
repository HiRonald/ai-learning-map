# CNN 基础设计文档

纯 C++ 手写最小 CNN，目标是把 **局部连接 + 权值共享** 讲清楚，并与 MLP 主题的 Fashion-MNIST 对照。

三个 demo：`filter` → `param` → `fashion`。

## 1. 目标与范围

- 手写 `Conv2d` / `MaxPool2d`（前向 + 反向 + SGD 更新）
- 分类头复用已有全连接 `Layer`（`mlp_core`）
- 同一 Fashion 数据上对比参数量与准确率直觉

非目标：im2col、GPU、BatchNorm、现代检测/分割、刷全量 60k。

## 2. 目录

```
topics/01-classic-dl/cnn-basics/
├── README.md / notes.md / video.md / meta.yaml
└── code/
    ├── main.cpp
    ├── cnn/
    │   ├── tensor.*     # NCHW
    │   ├── conv2d.*
    │   ├── pool2d.*
    │   ├── ops.*        # relu / flatten / reshape 图像
    │   └── net.*        # Conv→Pool→Conv→Pool→FC
    └── demos/
        ├── filter.*     # 固定核滑动可视化
        ├── param.*      # FC vs Conv 参数量
        └── fashion.*    # Fashion-MNIST 小 CNN
```

数据复用：`mlp-from-scratch/data/fashion-mnist/`（`MLP_DATA_DIR`）。

## 3. 张量约定

NCHW，行优先平坦存储：

\[
\text{index}(n,c,h,w) = ((n \cdot C + c) \cdot H + h) \cdot W + w
\]

全连接层仍用已有 `Matrix`：形状 `(batch, features)`。`flatten` 把 `(N,C,H,W)` 收成 `(N, C·H·W)`。

## 4. 卷积

输出空间尺寸（本主题只用正方形核、等步幅）：

\[
H_\text{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1
\]

前向（每个输出位置）：

\[
Y_{n,o,i,j} = b_o + \sum_{c,u,v} W_{o,c,u,v}\, X_{n,c,\, iS+u-P,\, jS+v-P}
\]

（越界当 0。）

反向要点：

- \(\partial L/\partial W\)：输入补丁与上游梯度的相关
- \(\partial L/\partial b\)：对 batch / 空间求和
- \(\partial L/\partial X\)：把上游梯度按核「摊」回输入（等价于转置卷积直觉）

参数量：\(C_\text{out} \cdot (C_\text{in} \cdot K \cdot K + 1)\)（含偏置）。同尺寸全连接是 \(H W C_\text{in} \cdot H W C_\text{out}\)，差几个数量级。

## 5. MaxPool

默认 `2×2`、stride 2、无填充。前向每个窗口取 max，并缓存 argmax，反向只把梯度送回获胜位置。

## 6. 网络（`fashion`）：LeNet-5 风格

28×28 零填到 32×32（经典做法）：

```
(N,1,28,28) → pad → (N,1,32,32)
  → Conv2d(1→6,  k=5) → ReLU → MaxPool(2)   # → (N,6,14,14)
  → Conv2d(6→16, k=5) → ReLU → MaxPool(2)   # → (N,16,5,5)
  → Flatten                                 # → (N,400)
  → Dense(400→120, relu) → Dense(120→84, relu) → Dense(84→10, identity)
  → softmax_ce
```

原版用 tanh + 平均池化；这里用 ReLU + MaxPool。

约 **61k** 参数（卷积 ~2.6k，FC 头占大头），对照 MLP ~**109k**。

算力：和之前的小 CNN 同量级——5×5 更贵，但通道少、很快收到 5×5。4k 子集大约一分钟内能跑完。

准确率预期：同子集上常略超或接近大 MLP；不保证碾压（Fashion 对 MLP 太友好）。

## 7. Demo 配置

| Demo | 作用 | 数据 |
|------|------|------|
| `filter`（默认） | 固定 Sobel / 锐化核扫一张图，ASCII 前后对比 | 合成十字 |
| `param` | 打印同输入下 FC vs Conv 参数量 | 无训练 |
| `fashion` | 同子集 MLP vs LeNet + ASCII 画廊 | Fashion 子集 |

## 8. 构建与运行

```bash
make run-cnn                 # filter
make run-cnn DEMO=param
make run-cnn DEMO=fashion    # 复用 MLP 的 Fashion 缓存
```

## 9. 后续可扩展

- im2col / BLAS 加速后放大子集
- 接到 `pytorch-basics` 用 `nn.Conv2d` 对照同一结构（墙钟才会「像论文里那样」）
