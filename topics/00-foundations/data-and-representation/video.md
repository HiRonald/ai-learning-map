# 视频大纲：数据与表征

目标时长：6–10 分钟

## 钩子（15–30s）

「模型从没见过『猫』这个字的含义——它见过的是一串数字。」

## 第一段：直觉

表征 = 约定好的数值化世界；对照 Fashion 像素向量。

## 第二段：结构

管道：原文 → token → id → embedding 查表；`<unk>` 处理 OOV。

## 第三段：实验演示

```bash
make run-repr
```

Part A：词表 / id / 查表。  
Part B：bigram 训练、探针、贪心续写；点明第一层 = embedding。

## 结尾引导

- 下一站：PyTorch（`nn.Embedding`）或 CNN / RNN——结构如何匹配这些张量
- 路径：`topics/00-foundations/data-and-representation/`
