# 数据与表征

> 状态：draft · 轨道：Foundations · 难度：L1

## 要解决什么问题

模型并不直接理解「图片/句子」，它吃的是数值张量。数据从原始形态变成可学习表示，中间发生了什么？

## 直觉

**表征**是把世界压成向量的约定：像素、token id、embedding。数据质量与分布，往往比换一个时髦结构更影响上限。

## 结构 / 公式

（写作中）原始数据 → 清洗/标注 → 特征或 token → embedding；tokenizer 与词表的最小图示。

## 实验

代码目录：`code/`（待添加：极简 tokenizer 或 embedding 查表示意）

## 局限与下一步

- 不覆盖大规模数据工程与标注平台
- 下一站：[CNN 基础](../../01-classic-dl/cnn-basics/) 或 [RNN 序列](../../01-classic-dl/rnn-seq/)
