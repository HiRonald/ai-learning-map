# ai-learning-map

用一个仓库建立 **AI 技术全景图**：边学边写最小可运行实验，边产出讲解文与视频。

不追求每个点最深，追求 **覆盖主要技术栈 + 主题之间能连成地图**。

## 从这里开始

- [学习路线图](/ROADMAP) — Track 与主题进度
- [MLP from scratch](/topics/00-foundations/mlp-from-scratch/) — 当前已发布主题
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
make run
```

## 本地预览本站

```bash
cd site && npm install && npm run docs:dev
```
