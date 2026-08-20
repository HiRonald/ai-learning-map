# ai-learning-map

用一个仓库建立 **AI 技术全景图**：边学边写最小可运行实验，边产出讲解文与视频。

不追求每个点最深，追求 **覆盖主要技术栈 + 主题之间能连成地图**。

## 从这里开始

- [学习路线图](/ROADMAP) — Track 与主题进度
- [MLP from scratch](/topics/00-foundations/mlp-from-scratch/) — 手写前向/反向
- [训练与评测基础](/topics/00-foundations/train-eval-basics/) — train/val/test 与过拟合
- [数据与表征](/topics/00-foundations/data-and-representation/) — tokenize → embedding 查表
- [CNN 基础](/topics/01-classic-dl/cnn-basics/) — 局部连接与权值共享
- [残差连接](/topics/01-classic-dl/residual-basics/) — 深度可训的恒等捷径
- [视频索引](/media/) — 成片外链

## 仓库结构（简）

| 路径 | 作用 |
|------|------|
| `topics/` | 主题单元（讲解 + 代码 + 视频大纲） |
| `shared/` | 跨主题模板与约定 |
| `site/` | 本站配置 |
| `media/` | 视频链接索引 |

## 本地运行实验

```bash
# 仓库根目录
make run                 # xor
make run DEMO=fashion    # 首次下载 Fashion-MNIST
make run-eval            # train/val/test 学习曲线
make run-repr            # tokenize → embedding 查表
make run-cnn             # filter；DEMO=param|fashion
make run-residual        # identity；DEMO=fashion
make run-attention       # 人名玩具 GPT
make run-nanogpt         # nanoGPT 莎士比亚（单进程）
make run-ddp             # 同上 + 2 进程 DDP
```

## 本地预览本站

```bash
cd site && npm install && npm run docs:dev
```
