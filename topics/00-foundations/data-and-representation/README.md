# 数据与表征

> 状态：published · 轨道：Foundations · 难度：L1 · 语言：C++

## 要解决什么问题

模型并不直接理解「图片/句子」，它吃的是数值张量。数据从原始形态变成可学习表示，中间发生了什么？  
进一步：embedding 表能不能被训练？和 `nn.Embedding` 是什么关系？

## 直觉

**表征**是把世界压成向量的约定。本主题分两段：

| 部分 | 做什么 |
|------|--------|
| **A. 查表** | 原文 → token → id → 取 `E` 的行（不训练） |
| **B. 训练** | 上一词 one-hot → MLP 预测下一词；**第一层 `W` 就是 embedding** |

关键等式：`one_hot(id) @ W == E[id]`。所以不必另写查表层——MLP 第一层在学这张表。

## 结构

```text
Part A:  text → tokenize → ids → lookup(E) → (seq, dim)
Part B:  prev_id → one-hot → Linear(V→dim) → ReLU → Linear(dim→V) → next_id
                              ↑
                         embedding E
```

任务：**bigram 语言模型**（给定上一词，预测下一词）。  
数据：几句硬编码玩具英文（cat/mat、dog/log 成对出现）。

## 实验

```bash
make run-repr
```

预期：

- A：词表、OOV→`<unk>`、查表形状  
- B：train acc 升高；探针如 `sat→on`；从 `the` / `cat` 贪心续写；打印学到的余弦（玩具信号，别当 Word2Vec）

过关自检：

1. 分清 token id 与 embedding 向量  
2. 能说出「one-hot @ W = 查表」  
3. 知道 bigram 很短视，小语料会循环/歧义  

## 局限与下一步

- 不是真 LM；不做 BPE、不做上下文窗口 >1  
- 下一站：[PyTorch 基础](../../03-frameworks/pytorch-basics/)（`nn.Embedding`），或 [CNN](../../01-classic-dl/cnn-basics/) / [RNN](../../01-classic-dl/rnn-seq/)
