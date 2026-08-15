# ai-learning-map

用一个仓库建立 **AI 技术全景图**：边学边写最小可运行实验，边产出讲解文与视频。

不追求每个点最深，追求 **覆盖主要技术栈 + 主题之间能连成地图**。

## 仓库怎么读

| 路径 | 作用 |
|------|------|
| [`ROADMAP.md`](ROADMAP.md) | 学习地图、完成状态、架构审视 |
| [`topics/`](topics/) | 主题单元（讲解 + 代码 + 视频大纲） |
| [`shared/`](shared/) | 跨主题模板与公共约定 |
| [`site/`](site/) | 文档站（由 topics 生成） |
| [`media/`](media/) | 视频链接索引（成片不进 git） |

七条轨道（编号=学习顺序）：Foundations → Classic DL → Transformers → Frameworks → **LLM Apps** → AI Infra → Embodied。详见 [`ROADMAP.md`](ROADMAP.md)。

## 主题单元约定

每个主题目录固定包含：

- `README.md`：对外讲解
- `meta.yaml`：状态 / 标签 / 前置
- `video.md`：视频大纲
- `code/`：最小可运行实验（可选）

模板：[`shared/templates/topic/`](shared/templates/topic/)

## 快速开始（当前已发布主题）

从零 MLP（`xor` / `sine` / `fashion`）：

```bash
make run                 # 默认 xor
make run DEMO=fashion    # 首次自动下载 Fashion-MNIST
```

训练与评测（train / val / test + 过拟合对照）：

```bash
make run-eval                   # normal
make run-eval MODE=overfit      # 小训练集拉长 epoch
make run-eval MODE=earlystop    # 同设置 + 早停
```

数据与表征（查表 + bigram MLP）：

```bash
make run-repr
```

CNN 基础（`filter` / `param` / `fashion`）：

```bash
make run-cnn                 # 默认 filter
make run-cnn DEMO=fashion    # 复用 MLP 的 Fashion 缓存
```

残差连接（`identity` / `fashion`）：

```bash
make run-residual                 # 默认 identity：深网学 y=x
make run-residual DEMO=fashion
```

Attention 基础（numpy 最小 GPT）：

```bash
make run-attention                 # 首次自动下载人名
```

文档站：

```bash
cd site && npm install && npm run docs:dev
```

## Git 用法（简记）

- **主题内容用目录**，不要用长期分支分区
- 日常可短命分支 `topic/xxx`，合并进 `main` 后删除
- 大视频文件不进仓库，只在 `media/index.md` 留链接
