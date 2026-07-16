# AI 学习路线图（Roadmap）

本仓库的目标：建立 AI 行业主要技术栈的 **框架性理解**，再选择一点深入。

状态说明：`published` = 可学可讲 · `draft` = 占位/写作中 · `planned` = 仅标题

目录编号与学习顺序一致：`00 → 06` 即推荐主线。

## Track 总览

| # | Track | 目录 | 一层抽象 | 一句话 |
|---|-------|------|----------|--------|
| 00 | Foundations | [`topics/00-foundations/`](topics/00-foundations/) | 学习基本功 | 训练闭环、评测、数据表征 |
| 01 | Classic DL | [`topics/01-classic-dl/`](topics/01-classic-dl/) | 结构归纳偏置 | CNN / RNN / Diffusion |
| 02 | Transformers | [`topics/02-transformers/`](topics/02-transformers/) | 模型层 | Attention、LLM 炼成、多模态、高效微调 |
| 03 | Frameworks | [`topics/03-frameworks/`](topics/03-frameworks/) | 工程接口 | PyTorch 对照 from-scratch |
| 04 | LLM Apps | [`topics/04-llm-apps/`](topics/04-llm-apps/) | 应用层 | Prompt / RAG / Agent（软件工具循环） |
| 05 | AI Infra | [`topics/05-ai-infra/`](topics/05-ai-infra/) | 系统层 | 训练/推理/分布式/可观测 |
| 06 | Embodied | [`topics/06-embodied/`](topics/06-embodied/) | 物理闭环 | 感知-决策-执行 |

## 主题进度

### 00 · Foundations

| 主题 | 状态 | 说明 |
|------|------|------|
| [mlp-from-scratch](topics/00-foundations/mlp-from-scratch/) | published | 纯 C++ MLP 解 XOR，打通前向/反向/更新 |
| [train-eval-basics](topics/00-foundations/train-eval-basics/) | draft | 训练/验证、过拟合、指标 |
| [data-and-representation](topics/00-foundations/data-and-representation/) | draft | 数据、embedding、tokenizer |

### 01 · Classic DL

| 主题 | 状态 | 说明 |
|------|------|------|
| [cnn-basics](topics/01-classic-dl/cnn-basics/) | draft | 局部连接与权值共享 |
| [rnn-seq](topics/01-classic-dl/rnn-seq/) | draft | 状态沿时间传递 |
| [diffusion-basics](topics/01-classic-dl/diffusion-basics/) | draft | 生成式：噪声 ↔ 数据 |

### 02 · Transformers

| 主题 | 状态 | 说明 |
|------|------|------|
| [attention-basics](topics/02-transformers/attention-basics/) | draft | Self-Attention 信息路由 |
| [llm-lifecycle](topics/02-transformers/llm-lifecycle/) | draft | 预训练 / SFT / 对齐 |
| [multimodal-basics](topics/02-transformers/multimodal-basics/) | draft | 视觉等模态如何进同一套模型 |
| [efficient-finetune](topics/02-transformers/efficient-finetune/) | draft | LoRA / PEFT 直觉 |

### 03 · Frameworks

| 主题 | 状态 | 说明 |
|------|------|------|
| [pytorch-basics](topics/03-frameworks/pytorch-basics/) | draft | PyTorch 重写 XOR/MLP |

### 04 · LLM Apps

| 主题 | 状态 | 说明 |
|------|------|------|
| [prompt-and-context](topics/04-llm-apps/prompt-and-context/) | draft | 提示与上下文工程 |
| [rag-basics](topics/04-llm-apps/rag-basics/) | draft | 先查再答 |
| [agents-basics](topics/04-llm-apps/agents-basics/) | draft | 工具循环（软件 Agent） |

### 05 · AI Infra

| 主题 | 状态 | 说明 |
|------|------|------|
| [training-serving-overview](topics/05-ai-infra/training-serving-overview/) | draft | 数据 → 训练 → 导出 → Serving |
| [distributed-training-101](topics/05-ai-infra/distributed-training-101/) | draft | 数据并行直觉 |
| [inference-optimization-101](topics/05-ai-infra/inference-optimization-101/) | draft | 量化、批处理、KV Cache |
| [llmops-observability](topics/05-ai-infra/llmops-observability/) | draft | 线上日志、追踪与反馈飞轮 |

### 06 · Embodied

| 主题 | 状态 | 说明 |
|------|------|------|
| [sense-plan-act](topics/06-embodied/sense-plan-act/) | draft | 物理/仿真世界最小闭环 |

## 建议学习顺序

与目录编号一致：

```text
00 Foundations:  mlp-from-scratch → train-eval-basics → data-and-representation
01 Classic DL:   cnn-basics / rnn-seq → diffusion-basics
02 Transformers: attention-basics → llm-lifecycle → multimodal-basics / efficient-finetune
03 Frameworks:   pytorch-basics（也可插在 Transformers 中段对照）
04 LLM Apps:     prompt-and-context → rag-basics → agents-basics
05 AI Infra:     training-serving-overview → distributed-training-101 / inference-optimization-101 → llmops-observability
06 Embodied:     sense-plan-act（与 agents-basics 对照着看）
```

每个主题尽量走完同一流水线：

```text
学概念 → 写 README → 最小 code → 跑通 → video.md → 录制 → 填链接 → published
```

## 架构审视（2026-07）

### 分层（= 目录编号）

```text
00 Foundations     数据/评测/训练闭环
01 Classic DL      结构（CNN/RNN/扩散）
02 Transformers    模型（Transformer/LLM/多模态/PEFT）
03 Frameworks      框架 API
04 LLM Apps        软件应用（Prompt/RAG/Agent）
05 AI Infra        系统（训推/观测）
06 Embodied        物理闭环
```

RAG/Agent 已从 Transformers 拆到 `04-llm-apps`，并与具身 Agent（`06-embodied`）明确分工。

### 关键拼图

| 拼图 | 落点 | 为何需要 |
|------|------|----------|
| 多模态 | `multimodal-basics` | 业界默认语境已含 VLM |
| 高效微调 | `efficient-finetune` | 落地微调以 PEFT 为主 |
| 提示/上下文 | `prompt-and-context` | 应用层最低成本杠杆 |
| RAG / Agent 分拆 | `rag-basics` / `agents-basics` | 检索 ≠ 工具循环 |
| 线上可观测 | `llmops-observability` | 没有观测就没有迭代 |

### 有意不单独开轨

- 经典统计 ML 全家桶、完整强化学习课、CUDA/编译器细节
- 对齐算法深水区、向量数据库运维手册、多智能体框架评测
- 第二框架（TensorFlow 等）：需要时并进 `03-frameworks` 子主题即可

### 仍可后续加

- 语音（ASR/TTS）并进多模态或 Classic
- 推荐系统 / 搜广（职业方向需要再开 Track）
- 安全与红队（可挂在 `llmops` 或 Apps 下）
