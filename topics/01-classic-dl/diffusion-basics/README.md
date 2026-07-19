# Diffusion 基础：生成式另一条主线

> 状态：draft · 轨道：Classic DL · 难度：L2

## 要解决什么问题

除了「分类/预测下一个 token」，如何从噪声一步步还原出图像（或其它数据）？

## 直觉

训练时学会「去噪」；生成时从纯噪声反着走，得到新样本。与判别式模型互补，是现代图像/视频生成的骨干直觉。

挂在 Classic DL：先建立「加噪/去噪」结构偏置；当代缩放路径已与 Transformer 合流（见下），学完可衔接到 [Attention](../../02-transformers/attention-basics/) / [多模态](../../02-transformers/multimodal-basics/)。

## 结构 / 公式

1. 前向加噪 / 反向去噪示意；不展开完整 SDE 推导。
2. **现代化衔接（点到即可）**
   - 早期常见 U-Net 骨架做去噪网络
   - 规模化路径转向 **DiT**（Diffusion Transformer）等：去噪器本身是 Transformer
   - **Flow matching** / rectified flow：另一条「从噪声到数据」的连续路径表述，工业生成里越来越常见
3. 与自回归 LLM：都是生成，但归纳偏置与采样过程不同。

## 实验

代码目录：`code/`（待添加：一维或小图 toy denoising 示意）

## 局限与下一步

- 不覆盖 Stable Diffusion 全家桶、微调生态与视频扩散工程细节
- 下一站：[Attention 基础](../../02-transformers/attention-basics/)
