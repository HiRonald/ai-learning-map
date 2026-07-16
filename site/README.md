# 文档站（VitePress）

本目录放站点配置与 npm 依赖。正文通过符号链接读取仓库内容：

- `topics` → `../topics`
- `media` → `../media`
- `ROADMAP.md` → `../ROADMAP.md`

首页为本地 [`index.md`](./index.md)。

## 开发

```bash
cd site
npm install
npm run docs:dev
```

或在仓库根目录：`make docs-dev`

## 构建

```bash
cd site
npm run docs:build
```

静态产物在 `site/.vitepress/dist/`。

## 维护说明

- `code/` 被 `srcExclude` 排除，不会当成文档页
- 新增主题后，在 [`.vitepress/config.mts`](.vitepress/config.mts) 侧边栏补一项
- 若克隆后符号链接丢失，在 `site/` 下重新执行：

```bash
ln -sfn ../topics topics
ln -sfn ../media media
ln -sfn ../ROADMAP.md ROADMAP.md
```
