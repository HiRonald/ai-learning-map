# 推理优化 101

> 状态：draft · 轨道：AI Infra · 难度：L2

## 要解决什么问题

训练追求效果；上线追求 **延迟、吞吐、成本**。量化、批处理、KV Cache 各自在优化什么？推理引擎（如 vLLM）又站在哪一层？

## 直觉

少算（量化/蒸馏）、多拼（连续批处理）、少重复（KV Cache / 前缀缓存复用历史状态）——三条常见杠杆。

引擎把这些杠杆产品化：把「能跑通的生成」变成「多人同时请求仍扛得住」。本主题学 **杠杆与职责**，不背某一家的配置手册。

## 结构 / 公式

1. **在线路径**：预填（prefill）vs 解码（decode）；延迟 / 吞吐 / 成本三角。
2. **三条杠杆**
   - 量化：用更低精度换显存与带宽（质量要可接受）
   - 批处理：把多请求的 decode 拼在一起吃满 GPU（含 continuous batching 直觉）
   - KV Cache：别为同一前缀反复算 Attention 状态；PagedAttention 一类思路解决「显存碎片 / 动态长度」
3. **引擎落点（点名即可）**
   - **vLLM**：高吞吐 serving 的常见默认；PagedAttention + continuous batching
   - **SGLang** 等：偏结构化输出 / 前缀缓存 / Agent 负载的另一条线
   - **TGI**、**llama.cpp** / **Ollama**、**MLX**：云端与本地的其它常见入口
   - 选型直觉：要的是吞吐、本地便携，还是严格 JSON / 工具调用——瓶颈不同，引擎不同

不展开：具体 flag、版本对比、压测报表。

## 实验

代码目录：`code/`（可选：对比假延迟数字的示意脚本；或本地用 Ollama/llama.cpp 感受「引擎层」存在即可）

## 局限与下一步

- 不覆盖具体推理引擎调参手册，也不开 GEMM / CUDA kernel 课（见 [ROADMAP 有意不开](../../../ROADMAP.md)）
- 下一站：[LLMOps / 可观测性](../llmops-observability/) → [Sense · Plan · Act](../../06-embodied/sense-plan-act/)
