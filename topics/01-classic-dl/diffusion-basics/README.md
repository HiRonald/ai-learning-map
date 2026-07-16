# Diffusion 基础：生成式另一条主线

> 状态：draft · 轨道：Classic DL · 难度：L2

## 要解决什么问题

除了「分类/预测下一个 token」，如何从噪声一步步还原出图像（或其它数据）？

## 直觉

训练时学会「去噪」；生成时从纯噪声反着走，得到新样本。与判别式模型互补，是现代图像/视频生成的骨干直觉。

## 结构 / 公式

（写作中）前向加噪 / 反向去噪示意；不展开完整 SDE 推导。

## 实验

代码目录：`code/`（待添加：一维或小图 toy denoising 示意）

## 局限与下一步

- 不覆盖 Stable Diffusion 全家桶与微调生态
- 下一站：[Attention 基础](../../02-transformers/attention-basics/)
