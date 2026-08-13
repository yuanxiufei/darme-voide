# 本地模型评估方案

> **硬件配置**：NVIDIA RTX A5000 24GB GDDR6 | 96GB 系统内存 | FP16 ≈ 75 TFLOPS

---

## 1. 显存预算

| 占用项 | VRAM 消耗 | 说明 |
|---|---|---|
| CUDA 驱动 / 系统保留 | ~0.5 GB | 固定开销 |
| **可用显存** | **~23.5 GB** | 推理工作空间 |
| SD/视频推理峰值缓冲 | ~2–4 GB | 取决于分辨率和 batch |
| **安全加载上限** | **~20 GB** | 模型权重占用上限 |

**结论**：A5000 单次只能加载一个中大型模型，管线各环节必须串行、即用即卸。

---

## 2. 管线环节与模型选型

### 环节 1 | 剧本改写 → LLM (text)

**需求**：长文本（小说章节→分镜剧本）改写，中文输出，结构化格式。

| 模型 | 量化 | VRAM | 中文能力 | 推荐指数 |
|---|---|---|---|---|
| **qwen3:14b** | Q4_K_M | ~9 GB | ★★★★★ | ⭐⭐⭐ |
| **qwen3:32b** | Q4_K_M | ~19 GB | ★★★★★ | ⭐⭐ |
| glm4:9b-chat | Q4_K_M | ~6 GB | ★★★★ | ⭐⭐ |
| deepseek-r1:14b | Q4_K_M | ~9 GB | ★★★★★ | ⭐⭐⭐ |
| mistral-small:22b | Q4_K_M | ~13 GB | ★★★ | ⭐ |
| gemma3:12b | Q4_K_M | ~8 GB | ★★★ | ⭐ |

**推荐**：`qwen3:14b` Q4_K_M（~9 GB）— 中文最强 14B，剧本改写、结构化输出均能胜任。显存宽裕可升至 `qwen3:32b` IQ4_XS（~16 GB）提升深度理解。

> 注意：GLM4 9B 也够用但偶尔丢格式约束；DeepSeek-R1 推理强但中文长篇稳定性不如 Qwen。

---

### 环节 2 | 角色/场景提取 → LLM (text)

**需求**：从剧本中识别角色名、外貌、性格、地点、时间，同名去重。

- **模型复用**：与环节 1 用同一个 LLM，`qwen3:14b` 足够。
- **VRAM**：~9 GB（同环节 1）。

---

### 环节 3 | 音色分配 → LLM (text)

**需求**：为每个角色匹配最佳音色，从预设音色列表中选择。

- **模型复用**：环节 1/2/3/5 共享同一个 LLM 实例，不重复加载。
- **VRAM**：~9 GB。
- **推荐**：`qwen3:14b`，音色匹配任务简单，7B 也够。

---

### 环节 4 | 语音样本生成 → TTS

**需求**：中文字符级自然语音合成，需要多音色支持。

| 方案 | VRAM | 质量 | 多音色 | 部署复杂度 |
|---|---|---|---|---|
| **CosyVoice 2** | ~4 GB | ★★★★★ | 原生支持 | 中 |
| **GPT-SoVITS v2** | ~4 GB | ★★★★ | 克隆 | 中 |
| Fish-Speech 1.5 | ~4 GB | ★★★★ | 克隆 | 中 |
| ChatTTS | ~2 GB | ★★★ | 有限 | 低 |

**推荐**：`CosyVoice 2`（阿里最新，中文 TTS 天花板）。支持 0-shot 音色克隆，一条 3 秒参考音频即可复刻角色音色。部署为独立 HTTP 服务，不受 Ollama 生态约束。

---

### 环节 5 | 分镜拆解 → LLM (text)

**需求**：剧本→镜头序列，需输出 JSON 结构化（shot_type/angle/movement/action/dialogue/image_prompt/video_prompt 等复杂字段）。

| 模型 | 量化 | VRAM | JSON 稳定性 | 推荐 |
|---|---|---|---|---|
| **qwen3:32b** | IQ4_XS | ~16 GB | ★★★★★ | ⭐⭐⭐ |
| **qwen3:14b** | Q4_K_M | ~9 GB | ★★★★ | ⭐⭐ |
| deepseek-r1:32b | IQ4_XS | ~16 GB | ★★★★ | ⭐⭐ |

**推荐**：`qwen3:32b` IQ4_XS（~16 GB）。分镜拆解是管线中最复杂的推理任务（同时生成 10+ 维度），14B 偶有字段缺失/格式错误。32B 在 16 GB 量化下 24 GB 显存刚好容纳，建议此环节**单独切换**大模型。

> 如果环节 1-3 用 14B（~9 GB），环节 5 切换到 32B（~16 GB），Ollama 先 `keep_alive=0` 卸载 14B 再加载 32B。

---

### 环节 6 | 图片生成 → Stable Diffusion / Flux

**需求**：角色立绘、场景图、分镜图，要求跨图角色一致性。

| 方案 | VRAM | 质量 | 速度 | 一致性 | 推荐 |
|---|---|---|---|---|---|
| **SDXL + IP-Adapter + ControlNet** | 10–14 GB | ★★★★ | ★★★★ | ★★★★ | ⭐⭐⭐ |
| **Flux.1-dev Q4** | 10–12 GB | ★★★★★ | ★★ | ★★★★ | ⭐⭐ |
| Kolors (快手) | 8–10 GB | ★★★★ | ★★★★ | ★★★ | ⭐⭐ |
| SD 3.5 Medium | 8–10 GB | ★★★★ | ★★★★ | ★★★ | ⭐⭐ |

**推荐**：`SDXL` + `IP-Adapter FaceID` + `ControlNet Canny/Depth`。SDXL 生态最成熟，IP-Adapter 确保跨图角色一致性。显存占用约 12 GB，A5000 跑 1024×1024 约 8–15 秒/张。

**一致性方案**：
1. 首张角色图生成后 → IP-Adapter 提取 face/character embedding
2. 分镜图生成时注入 IP-Adapter embedding → 角色面貌保持一致
3. ControlNet Canny 锁定构图 → 镜头角度（中景/特写/全景）可控

**替代方案**：`Flux.1-dev GGUF Q4` 画质更高但慢 2–3 倍，批量生成不划算。

---

### 环节 7 | 视频生成 → Wan 2.6 / CogVideoX

**需求**：图生视频（分镜图→5–8 秒动态），角色与参考图保持一致。

| 方案 | VRAM | 质量 | 速度 | i2v | 推荐 |
|---|---|---|---|---|---|
| **Wan 2.6 i2v 14B GGUF Q4** | 14–16 GB | ★★★★★ | ★★ | ✅ | ⭐⭐⭐ |
| Wan 2.6 i2v 14B fp8 | 16–18 GB | ★★★★★ | ★★ | ✅ | ⭐⭐ |
| CogVideoX 5B | 10–12 GB | ★★★ | ★★★ | ✅ | ⭐⭐ |
| AnimateDiff (SDXL) | 8–10 GB | ★★★ | ★★★★ | ✅ | ⭐ |

**推荐**：`Wan 2.6 i2v 14B GGUF Q4_K_M`（~15 GB）。当前最强开源图生视频模型，支持 720×480 输出。A5000 跑一段 5s 视频约 3–8 分钟。

| 模型 | 5s 视频耗时 | VRAM 峰值 |
|---|---|---|
| Wan 2.6 GGUF Q4 | 3–6 min | ~15 GB |
| CogVideoX 5B | 1–2 min | ~11 GB |
| AnimateDiff SDXL | 30s–1min | ~10 GB |

**关键约束**：视频生成必须独占 GPU——先卸载 SD/LLM 模型，`keep_alive=0` 清空显存，再加载 Wan 2.6。

---

### 环节 8 | 配音 (TTS) → CosyVoice 2

**需求**：分行匹配角色音色，生成带情感的中文配音。

- **复用环节 4** 的 CosyVoice 2 服务。
- **VRAM**：~4 GB，可与 LLM 共存（9 + 4 = 13 < 24 GB）。
- **推荐**：TTS 独立进程，常驻不卸载。

---

## 3. 全管线显存调度表

```
Stage 1–3:  [LLM 14B 9GB]                                    = 9.5 GB
Stage 4:    [LLM 14B 9GB] + [CosyVoice ~4GB]                  = 13 GB
Stage 5:    [LLM 32B 16GB]                                     = 16.5 GB
Stage 6:    [SDXL+IP-Adapter 12GB]                             = 14 GB
Stage 7:    [Wan 2.6 GGUF Q4 15GB]                             = 17 GB
Stage 8:    [CosyVoice ~4GB]                                   = 4.5 GB
Stage 9–10: [CPU only, FFmpeg]                                 = 0 GB
```

所有阶段均在 24 GB 安全范围内（峰值 17 GB / 24 GB）。

---

## 4. 模型卸载策略

使用 Ollama 的 `keep_alive` 参数控制模型生命周期：

| 切换场景 | 操作 |
|---|---|
| 环节 5 前（14B→32B） | `ollama stop qwen3:14b` → `keep_alive=0` → 加载 `qwen3:32b` |
| 环节 6 前（LLM→SD） | `ollama stop qwen3:32b` → 启动 SD WebUI/ComfyUI |
| 环节 7 前（SD→Wan） | 关闭 SD → 加载 Wan 2.6 GGUF |
| 环节 8（TTS） | CosyVoice 独立进程，不受影响 |

**OllamaManager 伪代码**：

```typescript
class GPUScheduler {
  async runStage(stage: string, fn: () => Promise<void>) {
    if (stage === 'storyboard') await this.switchLLM('qwen3:32b')
    if (stage === 'image') {
      await ollama.stop('qwen3:32b')
      await this.startSDXL()
    }
    if (stage === 'video') {
      await this.stopSDXL()
      await this.loadWan2_6()
    }
    await fn()
    if (stage === 'video') await this.unloadWan2_6()
  }
}
```

---

## 5. 部署清单

| 环节 | 推荐模型 | 部署方式 | VRAM | 备注 |
|---|---|---|---|---|
| 1–3 文本 | qwen3:14b Q4_K_M | Ollama | 9 GB | 环节 1–3 共享实例 |
| 4 TTS 样本 | CosyVoice 2 | 独立 WebSocket/HTTP 服务 | 4 GB | 常驻，不卸载 |
| 5 分镜 | qwen3:32b IQ4_XS | Ollama | 16 GB | 切换卸载 14B |
| 6 图片 | SDXL + IP-Adapter | ComfyUI / diffusers | 12 GB | 端口 8188 |
| 7 视频 | Wan 2.6 14B GGUF Q4 | ComfyUI / 独立脚本 | 15 GB | 生成后立即卸载 |
| 8 配音 | CosyVoice 2 | 复用环节 4 | 4 GB | — |
| 9–10 合成 | FFmpeg | 系统安装 | CPU | — |

---

## 6. 实施优先级

| 优先级 | 任务 | 估计工作量 | 价值 |
|---|---|---|---|
| **P0** | Ollama 部署 qwen3:14b + qwen3:32b + GPU 调度器 | 0.5 天 | 文本管线全本地化 |
| **P0** | CosyVoice 2 部署 + 音色样本生成 | 1 天 | 配音本地化 |
| **P1** | SDXL + IP-Adapter ComfyUI 工作流 | 1.5 天 | 角色一致性图片 |
| **P1** | 后端 adapter：图片生成接 SD WebUI API | 1 天 | 图片链路贯通 |
| **P2** | Wan 2.6 GGUF 部署 + 后端 adapter | 1.5 天 | 视频本地化（最耗时） |
| **P3** | 显存监控 Dashboard + 自动卸载 | 1 天 | 可靠性 |
| **总计** | | **6.5 天** | 全链路本地化 |
