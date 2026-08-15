# 视频大纲：残差连接

目标时长：8–12 分钟

## 钩子（15–30s）

「层数加多，按理说更强——为什么学一个恒等映射反而更难？」

## 第一段：直觉

恒等捷径；块学增量 \(F(x)\)。

## 第二段：结构

\(y = x + F(x)\)；点到与 U-Net concat 的区别。

## 第三段：实验演示

```bash
make run-residual                 # identity：plain vs residual MLP
make run-residual DEMO=fashion    # plain CNN vs residual CNN
```

盯 identity 表；fashion 看同 Conv 预算下 residual 是否更好训。

## 结尾引导

- 下一站：RNN，或 Attention block 里的残差
- 路径：`topics/01-classic-dl/residual-basics/`
