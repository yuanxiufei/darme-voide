"""**本项目自己的推理引擎**（`app/services/engine/`，2026-09-17 起）。

## 为什么有这一层

产品要的是**生成能力本身**：给定提示词/首帧，在我们后端里**加载模型 → 采样 → 解码 → 出片** ✓——
**不是**去调别人的推理服务 ✗。此前的 H3 路线是「薄封装 + 调用 ComfyUI(8188)」✗，
那是**调用**，不是**实现** ✗；本目录是把它换成**我们自己的实现** ✓。

## 两条硬约束（决定了实现方式，见各模块注释）

1. **许可证**：``reference/ComfyUI`` 是 **GPL-3.0** ✗ ⇒ 它的源码**不能搬进本仓库**
   （会把整个项目传染成 GPL ✗）。所以这里的原则是：
   * 数学/算法 **自己实现**（公开论文里的公式 ✓ 例如 Karras 2022 的 sigma 调度 ✓）；
   * 深度学习脚手架用 **Apache/MIT 许可**的库（``torch`` / ``diffusers`` / ``safetensors`` ✓）；
   * **一行 ComfyUI 源码都不抄** ✗。
2. **依赖**：本机后端 venv 目前**没有任何推理依赖**（无 torch / numpy ✗）⇒ 引擎分两层落地：
   * **纯算法层**（本目录的 ``schedules`` / ``geometry``）：**零依赖**、可立刻验证 ✓；
   * **张量层**（``sampler`` / ``latents`` / 模型加载）：需要 ``torch``（懒加载 ✓，装了才跑 ✓）。

## 分层（与项目约定一致）

``services/engine`` 只依赖 ``core`` ✓（不碰 routers ✓、不碰本地服务封装 ✓）。
模型文件路径来自 ``configs/model-paths.json`` / ``configs/models.json`` ✓（与 ``model_manager`` 同一份清单 ✓）。
"""
from __future__ import annotations

__all__ = ["audio_vae", "conditioning", "dit", "dryrun", "geometry", "gguf", "guidance",
           "h3_form", "h3_keys", "inventory", "loader", "mappings", "media", "pipeline",
           "quant", "safetensors", "sampler", "schedules", "segments", "text_encoder",
           "tokenizer_bpe", "tokenizer_hub", "tokenizer_own", "tokenizers_tuning",
           "torch_backend", "vae", "weights"]
