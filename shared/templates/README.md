# Topic 模板说明

新建主题时复制本目录：

```bash
cp -R shared/templates/topic topics/<track>/<topic-id>
```

然后填写：

| 文件 | 必填 | 说明 |
|------|------|------|
| `README.md` | 是 | 对外讲解（网站正文） |
| `meta.yaml` | 是 | 状态、标签、前置主题 |
| `video.md` | 建议 | 视频大纲 |
| `code/` | 有实验时 | 最小可运行代码 |
| `notes.md` | 可选 | 推导细节 / 踩坑 |

原则：一个主题只讲清一个核心思想。
