import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'ai-learning-map',
  description: 'AI 技术全景学习地图：主题讲解 + 最小可运行代码 + 视频大纲',
  lang: 'zh-CN',

  srcExclude: [
    '**/node_modules/**',
    '**/code/**',
    '**/meta.yaml',
  ],

  ignoreDeadLinks: true,

  vite: {
    resolve: {
      preserveSymlinks: true,
    },
    server: {
      fs: {
        allow: ['..'],
      },
    },
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '路线图', link: '/ROADMAP' },
      { text: '主题', link: '/topics/00-foundations/mlp-from-scratch/' },
      { text: '视频索引', link: '/media/' },
    ],

    sidebar: [
      {
        text: '总览',
        items: [
          { text: '项目说明', link: '/' },
          { text: '学习路线图', link: '/ROADMAP' },
          { text: '视频索引', link: '/media/' },
        ],
      },
      {
        text: '00 · Foundations',
        items: [
          { text: '轨道说明', link: '/topics/00-foundations/' },
          { text: 'MLP from scratch', link: '/topics/00-foundations/mlp-from-scratch/' },
          { text: '训练与评测基础', link: '/topics/00-foundations/train-eval-basics/' },
          { text: '数据与表征', link: '/topics/00-foundations/data-and-representation/' },
        ],
      },
      {
        text: '01 · Classic DL',
        items: [
          { text: '轨道说明', link: '/topics/01-classic-dl/' },
          { text: 'CNN 基础', link: '/topics/01-classic-dl/cnn-basics/' },
          { text: 'RNN 序列', link: '/topics/01-classic-dl/rnn-seq/' },
          { text: 'Diffusion 基础', link: '/topics/01-classic-dl/diffusion-basics/' },
        ],
      },
      {
        text: '02 · Transformers',
        items: [
          { text: '轨道说明', link: '/topics/02-transformers/' },
          { text: 'Attention 基础', link: '/topics/02-transformers/attention-basics/' },
          { text: 'LLM 生命周期', link: '/topics/02-transformers/llm-lifecycle/' },
          { text: '多模态基础', link: '/topics/02-transformers/multimodal-basics/' },
          { text: '高效微调', link: '/topics/02-transformers/efficient-finetune/' },
        ],
      },
      {
        text: '03 · Frameworks',
        items: [
          { text: '轨道说明', link: '/topics/03-frameworks/' },
          { text: 'PyTorch 基础', link: '/topics/03-frameworks/pytorch-basics/' },
        ],
      },
      {
        text: '04 · LLM Apps',
        items: [
          { text: '轨道说明', link: '/topics/04-llm-apps/' },
          { text: '提示与上下文', link: '/topics/04-llm-apps/prompt-and-context/' },
          { text: 'RAG 基础', link: '/topics/04-llm-apps/rag-basics/' },
          { text: 'Agent 基础', link: '/topics/04-llm-apps/agents-basics/' },
        ],
      },
      {
        text: '05 · AI Infra',
        items: [
          { text: '轨道说明', link: '/topics/05-ai-infra/' },
          { text: '训练与推理鸟瞰', link: '/topics/05-ai-infra/training-serving-overview/' },
          { text: '分布式训练 101', link: '/topics/05-ai-infra/distributed-training-101/' },
          { text: '推理优化 101', link: '/topics/05-ai-infra/inference-optimization-101/' },
          { text: 'LLMOps / 可观测性', link: '/topics/05-ai-infra/llmops-observability/' },
        ],
      },
      {
        text: '06 · Embodied',
        items: [
          { text: '轨道说明', link: '/topics/06-embodied/' },
          { text: 'Sense · Plan · Act', link: '/topics/06-embodied/sense-plan-act/' },
        ],
      },
    ],

    outline: { level: [2, 3], label: '本页目录' },
    search: { provider: 'local' },
  },
})
