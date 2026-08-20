# AI 学习路线图（Roadmap）

本仓库的目标：建立 AI 行业主要技术栈的 **框架性理解**，再选择一点深入。

状态说明：`published` = 可学可讲 · `draft` = 占位/写作中 · `planned` = 仅标题

目录编号与学习顺序大体一致：`00 → 06` 即推荐主线；`pytorch-basics` 建议提前插入（见下）。

## Track 总览

| # | Track | 目录 | 一层抽象 | 一句话 |
|---|-------|------|----------|--------|
| 00 | Foundations | [`topics/00-foundations/`](topics/00-foundations/) | 学习基本功 | 训练闭环、评测、数据表征 |
| 01 | Classic DL | [`topics/01-classic-dl/`](topics/01-classic-dl/) | 结构归纳偏置 | CNN / 残差 / RNN / LSTM / Seq2Seq |
| 02 | Transformers | [`topics/02-transformers/`](topics/02-transformers/) | 模型层 | Attention、LLM 炼成、多模态、高效微调 |
| 03 | Frameworks | [`topics/03-frameworks/`](topics/03-frameworks/) | 工程接口 | PyTorch 对照 from-scratch |
| 04 | LLM Apps | [`topics/04-llm-apps/`](topics/04-llm-apps/) | 应用层 | Prompt / RAG / Agent（软件工具循环） |
| 05 | AI Infra | [`topics/05-ai-infra/`](topics/05-ai-infra/) | 系统层 | 训练/推理/分布式/可观测 |
| 06 | Embodied | [`topics/06-embodied/`](topics/06-embodied/) | 物理闭环 | 感知-决策-执行 |

## 主题进度

### 00 · Foundations

| 主题 | 状态 | 说明 |
|------|------|------|
| [mlp-from-scratch](topics/00-foundations/mlp-from-scratch/) | published | 纯 C++ MLP：xor → sine → Fashion-MNIST，打通前向/反向/更新 |
| [train-eval-basics](topics/00-foundations/train-eval-basics/) | published | Fashion 切 train/val/test；normal vs overfit 学习曲线 |
| [data-and-representation](topics/00-foundations/data-and-representation/) | published | 查表 + bigram MLP（第一层 = embedding） |

### 01 · Classic DL

| 主题 | 状态 | 说明 |
|------|------|------|
| [cnn-basics](topics/01-classic-dl/cnn-basics/) | published | 局部连接与权值共享；filter → param → Fashion LeNet |
| [residual-basics](topics/01-classic-dl/residual-basics/) | published | \(y=x+F(x)\)：identity + Fashion plain/residual CNN |
| [rnn-seq](topics/01-classic-dl/rnn-seq/) | published | Vanilla RNN；日最低气温下一天预测 |
| [lstm-seq](topics/01-classic-dl/lstm-seq/) | published | 门控 + 二值 delayed recall；对照基线 |
| [seq2seq-basics](topics/01-classic-dl/seq2seq-basics/) | draft | Encoder–Decoder；收尾接到 Attention |

### 02 · Transformers

| 主题 | 状态 | 说明 |
|------|------|------|
| [attention-basics](topics/02-transformers/attention-basics/) | published | numpy Tensor 自动微分；因果 Attention + 最小 GPT 人名 |
| [llm-lifecycle](topics/02-transformers/llm-lifecycle/) | draft | 预训练 / SFT / 对齐 / test-time compute |
| [multimodal-basics](topics/02-transformers/multimodal-basics/) | draft | 视觉等模态如何进同一套模型 |
| [efficient-finetune](topics/02-transformers/efficient-finetune/) | draft | LoRA / PEFT 直觉 |

### 03 · Frameworks

| 主题 | 状态 | 说明 |
|------|------|------|
| [pytorch-basics](topics/03-frameworks/pytorch-basics/) | draft | PyTorch 重写 XOR/MLP（建议 Foundations 后即学） |

### 04 · LLM Apps

| 主题 | 状态 | 说明 |
|------|------|------|
| [prompt-and-context](topics/04-llm-apps/prompt-and-context/) | draft | 提示与上下文工程 |
| [rag-basics](topics/04-llm-apps/rag-basics/) | draft | 先查再答（与长期 Memory 区分） |
| [agents-basics](topics/04-llm-apps/agents-basics/) | draft | 工具循环、MCP、状态与终止 |

### 05 · AI Infra

| 主题 | 状态 | 说明 |
|------|------|------|
| [training-serving-overview](topics/05-ai-infra/training-serving-overview/) | draft | 数据 → 训练 → 导出 → Serving |
| [distributed-training-101](topics/05-ai-infra/distributed-training-101/) | published | nanoGPT 莎士比亚 + PyTorch DDP |
| [inference-optimization-101](topics/05-ai-infra/inference-optimization-101/) | draft | 量化、批处理、KV Cache；vLLM 等引擎落点 |
| [llmops-observability](topics/05-ai-infra/llmops-observability/) | draft | 追踪、Eval、护栏与反馈飞轮 |

### 06 · Embodied

| 主题 | 状态 | 说明 |
|------|------|------|
| [sense-plan-act](topics/06-embodied/sense-plan-act/) | draft | 物理/仿真世界最小闭环 |

## 建议学习顺序

目录编号是主线；**PyTorch 提前**，其余基本按轨走：

```text
00 Foundations:  mlp-from-scratch → train-eval-basics → data-and-representation
03 Frameworks:   pytorch-basics   ← 插在 Foundations 后，后续「配方/系统」实验不悬空
01 Classic DL:   cnn-basics → residual-basics → rnn-seq → lstm-seq → seq2seq-basics
02 Transformers: attention-basics → llm-lifecycle → multimodal-basics / efficient-finetune
04 LLM Apps:     prompt-and-context → rag-basics → agents-basics
05 AI Infra:     training-serving-overview → distributed-training-101 / inference-optimization-101 → llmops-observability
                 （旁路：做 Agent 若要自建 serving，可与 agents 并行看 inference-optimization）
06 Embodied:     sense-plan-act（与 agents-basics 对照着看）
```

每个主题尽量走完同一流水线：

```text
学概念 → 写 README → 最小 code → 跑通 → video.md → 录制 → 填链接 → published
```

## 语言约定（C++ vs PyTorch）

判据只有一条：**这个主题要你亲手摸的是「算子/闭环」，还是「结构配方 / 系统接口」？**

| 用 C++（from-scratch） | 用 PyTorch（或脚本） |
|------------------------|----------------------|
| 必须自己写出前向/反向或核心循环，才能建立直觉 | 核心直觉是结构/流程；再手写核收益低、墙钟差 |
| 复用已有 `mlp_core` / `cnn_core` 成本低 | 需要更深网、对照曲线、生态调用（tokenizer、API） |
| 主题目标 = 「这个东西在算什么」 | 主题目标 = 「工业上怎么搭 / 怎么跑」 |

### 主题落点

| 主题 | 语言 | 说明 |
|------|------|------|
| `mlp-from-scratch` | **C++** | 训练闭环本身 |
| `train-eval-basics` | **C++** | 复用 MLP；切分与曲线 |
| `data-and-representation` | **C++** | embedding = 查表 |
| `cnn-basics` | **C++** | 卷积滑动 / 共享核 |
| `residual-basics` | **C++ 优先** | 概念是 \(x+F(x)\)；用已有 `Layer` 做残差 MLP 即可，**不必**再写深 CNN |
| `rnn-seq` | **C++ 优先** | 小序列手写状态更新；要上长序列再 PyTorch |
| `lstm-seq` | **C++ 优先** | 门控步进；与 Vanilla RNN 同任务对照即可 |
| `seq2seq-basics` | **C++ 或 PyTorch** | 短序列 encoder–decoder；带 attention 时再 PyTorch |
| `attention-basics` | **Python / numpy** | 手写 Tensor 自动微分 + 最小 GPT。PyTorch 对照留给 `pytorch-basics` |
| `pytorch-basics` | **Python** | 桥：手写 ↔ `nn` / autograd |
| `llm-lifecycle` / `multimodal` / `efficient-finetune` | **Python** | 配方与接口 |
| `prompt` / `rag` / `agents` | **Python** | 应用层 |
| Infra / Embodied | **Python** 或概念示意 | 系统与闭环，不手写算子 |

学习顺序上：**Foundations（C++）→ 插入 `pytorch-basics` → 之后「配方类」默认 Python**；Classic DL 里仍在「摸算子」的（CNN / 残差 / 小 RNN）可以继续 C++，与是否已学 PyTorch 不冲突。

不要为了 residual 再手写一遍深卷积——那是 CNN 主题的课，不是残差的课。

## 架构审视（2026-07）

### 分层（= 目录编号）

```text
00 Foundations     数据/评测/训练闭环
01 Classic DL      结构（CNN/残差/RNN/LSTM/Seq2Seq）
02 Transformers    模型（Transformer/LLM/多模态/PEFT）03 Frameworks      框架 API（轨薄，学习顺序上提前插）
04 LLM Apps        软件应用（Prompt/RAG/Agent + 协议/记忆要点）
05 AI Infra        系统（训推/推理引擎/观测·Eval·护栏）
06 Embodied        物理闭环
```

RAG/Agent 已从 Transformers 拆到 `04-llm-apps`，并与具身 Agent（`06-embodied`）明确分工。

### 关键拼图

| 拼图 | 落点 | 为何需要 |
|------|------|----------|
| 残差 / 深度可训 | `residual-basics` | ResNet、Transformer block 的共同积木 |
| 多模态 | `multimodal-basics` | 业界默认语境已含 VLM |
| 高效微调 | `efficient-finetune` | 落地微调以 PEFT 为主 |
| 提示/上下文 | `prompt-and-context` | 应用层最低成本杠杆 |
| RAG / Agent 分拆 | `rag-basics` / `agents-basics` | 检索 ≠ 工具循环 |
| 工具协议（MCP 等） | `agents-basics` | 工具层已趋标准化 |
| 长期 Memory（≠ RAG） | `rag-basics` 对照 + `agents-basics` | 会话/跨会话状态 ≠ 文档检索 |
| Reasoning / test-time compute | `llm-lifecycle` | 单次深推理改变 Agent 形态 |
| 推理引擎（vLLM 等） | `inference-optimization-101` | 学杠杆与引擎职责，不单开产品课 |
| Eval（LLM/Agent） | `llmops-observability`（辅：`train-eval-basics`） | 经典指标 ≠ 生成/工具质量 |
| Guardrails / 安全 | `llmops-observability`（辅：`agents-basics`） | 生产 Agent 需要约束层 |
| 线上可观测 | `llmops-observability` | 没有观测就没有迭代 |

### 有意不单独开主题 / 开轨

- 经典统计 ML 全家桶、完整强化学习课
- **CUDA / 编译器 / GEMM·Tensor Core 调优**（算子层；地图只保留「大矩阵乘吃满 GPU」级直觉）
- 对齐算法深水区、向量数据库运维手册、多智能体框架横评
- **vLLM / SGLang / TGI 等引擎的调参手册**（概念落在 `inference-optimization-101`，不各开一课）
- 第二框架（TensorFlow 等）：需要时并进 `03-frameworks` 子主题即可

### 仍可后续加

- 语音（ASR/TTS）并进多模态
- 推荐系统 / 搜广（职业方向需要再开 Track）
- 推理/系统工程师旁路：再加深 serving 与算子（当前主线不膨胀）
