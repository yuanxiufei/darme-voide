"""**本项目自己的推理引擎**（`app/services/engine/`，2026-09-17 起）。

## 为什么有这一层

产品要的是**生成能力本身**：给定提示词/首帧，在我们后端里**加载模型 → 采样 → 解码 → 出片** ✓——
**不是**去调别人的推理服务 ✗。此前的 H3 路线是「薄封装 + 调用 ComfyUI(8188)」✗，
那是**调用**，不是**实现** ✗；本目录是把它换成**我们自己的实现** ✓。

## 两条硬约束（决定了实现方式，见各模块注释）

1. **许可证**：``ComfyUI`` 是 **GPL-3.0** ✗ ⇒ 它的源码**不能搬进本仓库**
   （会把整个项目传染成 GPL ✗）。所以这里的原则是：
   * 数学/算法 **自己实现**（公开论文里的公式 ✓ 例如 Karras 2022 的 sigma 调度 ✓）；
   * 深度学习脚手架用 **Apache/MIT 许可**的库（``torch`` / ``diffusers`` / ``safetensors`` ✓）；
   * **一行 ComfyUI 源码都不抄** ✗。
2. **依赖**：本机后端 venv **已装 torch** ✓（2026-09-17 是 CPU 轮子 ✓；**2026-09-26 更正** ✓：
   现在跑在 A5000 工作站上 ✓ ⇒ CUDA 轮子 ✓、`device=cuda` ✓ —— ⚠️ **别写死设备** ✗，
   一律实测 ✓，见 `torch_backend.py` 模块头 ✓ 与 `engine_readiness.device_facts()` ✓）⇒ 引擎分两层落地：
   * **纯算法层**（本目录的 ``schedules`` / ``geometry``）：**零依赖**、可立刻验证 ✓；
   * **张量层**（``sampler`` / ``latents`` / 模型加载）：需要 ``torch``（**已装** ✓ 懒加载 ✓ 装了才跑 ✓）。

## 分层（与项目约定一致）

``services/engine`` 只依赖 ``core`` ✓（不碰 routers ✓、不碰本地服务封装 ✓）。
模型文件路径来自 ``configs/model-paths.json`` / ``configs/models.json`` ✓（与 ``model_manager`` 同一份清单 ✓）。
"""
from __future__ import annotations

__all__ = ["accel_chain", "audio_mix", "audio_vae", "cache_guard", "cache_key", "chat_template", "checkpoint_meta", "clip_text", "conditioning", "dit", "dryrun", "geometry", "gguf", "gguf_dequant", "gguf_to_llm", "guidance",
           "h3_form", "h3_keys", "hybrid_load", "hybrid_merge", "image_ops", "inventory", "latent_container", "latent_formats", "llm", "llm_backend", "loader", "mappings", "media", "pipeline",
           "quant", "safetensors", "sampler", "schedules", "script_parse", "sdxl", "sdxl_backend", "sdxl_vae", "segments", "text_encoder",
           "tiers", "tokenizer_bpe", "tokenizer_hub", "tokenizer_own", "tokenizers_tuning",
           "torch_backend", "upscale", "upscale_net", "vae", "weights"]
