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

## 模块分组（6 组，2026-10-30 立）

目录是**平的** ✓（模块全部平铺、不分箱 ✓ —— ⚠️ **不写死条数** ✗✗：写死一个数每加一个模块就漂一次 ✓，
盘上真数由 ``tests/structure_test.py`` ㊀ 报 ✓；理由见 ``backend-py/README.md`` §目录结构 ④：这里是
「按能力切」而不是「按域切」，任何一次搬家都要同时改 300+ 处调用点与 6 套静态守卫 ✓✗）。
分工靠 :data:`MODULE_GROUPS` **登记** ✓：每组一个职责边界、组内模块不许跨界依赖 ✓，
**盘上多一个模块而没登记 ⇒ 报错** ✓（判据在 ``tests/structure_test.py``，别再靠人记 ✗）。

| 组 | 职责（一句话） | 组内不许做 |
|---|---|---|
| :data:`GROUP_NUMERIC` | 零依赖的**数学与规划**：调度 / 几何 / 引导 / 采样 / 分段 / 关键帧 / 掩码 / 混音 | 不 import torch ✗、不读文件 ✗ |
| :data:`GROUP_FORMAT` | **容器与像素**：safetensors / gguf / 内嵌元数据 / 潜空间口径 / 落盘 / 图像算子 | 不做模型数学 ✗ |
| :data:`GROUP_WEIGHT` | **量化与装载**：反量化 / 命名映射 / 混合加载 / LoRA / 档位表 | 不决定"跑哪一步" ✗ |
| :data:`GROUP_TOKEN` | **文本入口**：BPE / Unigram / WordPiece / GGUF 词表 / HF 改造 / chat 模板 | 不做图像数学 ✗ |
| :data:`GROUP_MODEL` | **网络本体**：DiT / H3 / SDXL / VAE / 文本塔 / LLM / 超清 | 不编排阶段 ✗、不管显存回收 ✗ |
| :data:`GROUP_ORCH` | **编排与可见面**：管线 / 运行时 / 门面 / 闸门 / 体检 / 干跑 / 桥 / 导演稿解析 | 不写模型数学 ✗ |
"""
from __future__ import annotations

#: ① 零依赖的数学与规划（**唯一允许「不装 torch 也能测」的一组** ✓）
GROUP_NUMERIC: tuple[str, ...] = (
    "schedules", "geometry", "guidance", "sampler", "segments", "keyframes",
    "conditioning", "audio_mix", "cache_key", "accel_chain",
    #: ⚠️ ``vae_tiling``（2026-10-31 补登记 ✓）挂这一组的理由：它算的是**切块起点 / 羽化权重 /
    #: 累加归一** ✓ —— 全是纯数学（只吃一个 ``Callable`` 做逐块算子 ✓），**不 import torch** ✓、
    #: 不读文件 ✓，与 :mod:`sampler` 同性质 ✓。它**不是** VAE 本体 ✗（``vae_wan``/``vae_h3`` 才是 ✓），
    #: 所以不该跟着进 ``model`` 组 ✗。
    "vae_tiling",
    #: ⚠️ ``scheduler``（2026-11-01 补登记 ✓）挂这里：它算的是**分配**（此刻谁干 / 余量够不够 / 该退到哪）
    #: ✓，只吃 `device_policy` 的读数与纯数学 ✓，**不 import torch** ✓、**不读文件** ✓ ——
    #: 与 `sampler` / `accel_chain` 同性质 ✓。它**不是**装载 ✗（真装权重在 `loader` / `torch_backend` ✓），
    #: 也不是设备策略本身 ✗（那管「这次让谁干」的判据 ✓，这管「多个活怎么排队」✓）。
    "scheduler",
)

#: ② 容器与像素（读得进、写得出的那一层）
GROUP_FORMAT: tuple[str, ...] = (
    "safetensors", "gguf", "checkpoint_meta", "latent_formats", "latent_container",
    "media", "image_ops", "cache_guard",
    #: ⚠️ ``qwen3vl_pixels``（2026-11-01 补登记 ✓）挂这里的理由：它只做**像素** ✓
    #: （resize 到 32 倍数 / `(v-0.5)/0.5` / merge 块优先的行序 ✓），产出的还是张量 ✓ ——
    #: 没有一层网络 ✓、没有任何可学的参数 ✓ ⇒ 与 `image_ops` 同性质 ✓，进 `model` 组反而会让
    #: "这一组都是网络本体"这句话失真 ✗。
    "qwen3vl_pixels",
    #: ⚠️ ``win_io``（2026-11-01 移植自 expert-stream ✓）挂这里：它是**文件字节这一层** ✓ ——
    #: 给偏移、回字节 ✓，**不认识张量** ✗（shape/dtype 由 `safetensors` 读 header 算 ✓）、
    #: **不做模型数学** ✗、**不 import torch** ✗。挂 `weight` 组会与"量化/命名映射"混为一谈 ✗：
    #: 那组都在"解释字节是什么意思" ✓，这层只负责"字节进来得够快、且不占页缓存" ✓。
    "win_io",
)

#: ③ 量化与权重装载（"文件里的字节 ⇒ 能算的张量"）
#: ⚠️ ``residency``（2026-10-31 ✓）挂这一组的理由：它决定的是**权重摆在哪台设备上** ✓
#: （卡 + CPU 联合驻留 ✓）—— 不是"模型怎么算"✗、也不是"谁来调度阶段"✗，
#: 和 `weights` / `hybrid_load` 同属"让权重能被算"的那一步 ✓。
#: ⚠️ ``device_policy``（2026-10-31 ✓）与 ``residency`` 是**一对** ✓✗：``residency`` 管「装不下时
#: **怎么分层**」✓（卡＋CPU 联合驻留的机制 ✓），``device_policy`` 管「**这次让谁干**」✓
#: （按当时显存读数在 cuda / cpu / 联合之间选 ✓）。两者都在回答"权重落在哪台设备上" ✓，
#: 所以同组 ✓ —— 拆到 orchestr 组会让"决定设备"的两份逻辑分居两处 ✓✗。
#: ⚠️ 它**不 import torch**（`probe()` 的参数是可注入的 ✓）⇒ 没有卡也能整份自检 ✓。
GROUP_WEIGHT: tuple[str, ...] = (
    "quant", "quant_nvfp4", "weights", "mappings", "hybrid_load", "hybrid_merge",
    "tiers", "lora", "lora_cache", "residency", "device_policy",
    #: ⚠️ ``prefetch``（2026-11-01 移植自 airllm ✓，**同日接进装载** ✓）挂这里：它决定的是
    #: **权重什么时候进内存** ✓（趁上一件在反量化/写卡，就把下一件读进内存 ✓），和 `residency` /
    #: `device_policy` 回答的是同一句话的另半句 ✓✗ —— 那两者管"落在**哪台设备**"✓，这管"落在
    #: **什么时候**"✓。接线点是 `weights._load_streaming` 的逐张量读 ✓（作用域一次装载即止 ✓，
    #: 跨阶段预取被否掉 ✗）。它**不决定装哪件** ✗（装配线给 key 与函数 ✓）、**不 import torch** ✗。
    "prefetch",
)

#: ④ 文本入口（prompt ⇒ token ⇒ 向量）
GROUP_TOKEN: tuple[str, ...] = (
    "tokenizer_bpe", "tokenizer_own", "tokenizer_gguf", "tokenizer_hub",
    "tokenizers_tuning", "chat_template",
    #: ⚠️ ``h3_prompt``（2026-11-01 补登记 ✓）挂这里的理由：它是**文本入口** ✓ —— 决定
    #: "<Picture 1>: " / "<Video k>" / "<T.T seconds>" 这些段怎么切、以及视觉块在文本序列里占哪几个位置 ✓
    #: （`token_tags` 也是它算 ✓）。它**不是** chat 模板 ✗（H3 一个 `<|im_start|>` 都不加 ✓），
    #: 更不是视觉塔 ✗（那是 `qwen3vl_vision` ✓）；放在这里与 `chat_template` 并列才说得通 ✓。
    "h3_prompt",
)

#: ⑤ 网络本体（"一族模型一个骨架"）
#: ⚠️ ``vae_flux``（2026-10-31 ✓）挂这里的理由：它是 **Flux 族 VAE 的本体口径** ✓
#: （16 通道 ✓ 无 quant_conv ✓ 先反缩放再解码 ✓）—— 和 `sdxl_vae` / `vae_h3` 同一层 ✓，
#: 都是"潜变量 ⇒ 像素"的那个网络 ✓；不是容器（那是 `latent_container` ✗）也不是量化（`weights` ✗）。
#: ⚠️ ``qwen3_te``（2026-10-31 ✓）同理：Qwen3 文本塔的**网络本体** ✓（36 层 / layer_idx=-2 ✓）。
#: ⚠️ ``qwen35_te``（2026-10-31 ✓）是 2.1 那座**混合塔**的本体 ✓（24 层 Gated DeltaNet ＋ 8 层
#: full attention ✓、`layer_idx=-2` ✓）—— 与 `qwen3_te` 并列而非替换 ✗：两族层型不同 ✓，
#: 混进一个文件就会让 `layer_types` 的两种真源打架 ✗。
#: ⚠️ ``dit_qwen_image``（2026-10-31 ✓）与 `dit_qwen21` **并列而非合并** ✓：两者族名不同
#: （`qwen_image` vs `qwen_image21` ✓）、时间口径相反（1000·σ 不翻转 vs lumina 式翻转 ✓）、
#: 调制组数不同（6 组 vs 2 行 ✓）—— 合并到一个文件会让两套口径互相污染 ✗。
#: ⚠️ ``qwen25_te``（2026-10-31 ✓）与 `qwen3_te` / `qwen35_te` 并列 ✓：Qwen2.5-VL 的塔
#: **q/k/v 带 bias** ✓、**无 QK-norm** ✓、取**最后一层**且不过 final norm ✓ —— 四格差异全在
#: `qwen25_te.Qwen25TeShape` 里，与另两座的 `layer_idx=-2` + QK-norm 口径不同 ✗。
#: ⚠️ ``vae_wan``（2026-10-31 ✓）与 `vae_flux` / `vae_h3` 同层 ✓：都是"潜变量 ⇒ 像素"的网络 ✓；
#: 它先接住的是 Qwen-Image 2.1 那个 64 通道 VAE ✓（Wan 2.2 布局 / 238 键 ✓），
#: **不是**容器（`latent_container` ✗）也不是规格表（`latent_formats` ✗）。
GROUP_MODEL: tuple[str, ...] = (
    "dit", "dit_lumina", "dit_qwen21", "dit_qwen_image", "dit_family", "dit_generic",
    "h3_form", "h3_keys", "h3_edit", "sdxl",
    "sdxl_backend",
    "sdxl_vae", "vae_flux", "vae_wan", "vae", "vae_h3", "audio_vae", "upscale", "upscale_net",
    "text_encoder", "clip_text", "qwen3_te", "qwen35_te", "qwen25_te", "llm", "llm_backend",
    #: ⚠️ ``qwen3vl_te`` / ``qwen3vl_vision``（2026-11-01 补登记 ✓）与 `qwen3_te` / `qwen25_te` 并列
    #: 而非合并 ✗：它们是 H3 那一座 **Qwen3-VL-32B** ✓ —— 视觉塔与语言塔在**同一个 safetensors** 里 ✓，
    #: 但两半边层型完全不同（27 层 ViT + DeepStack merger vs 语言塔 interleaved mrope ✓）、
    #: 键前缀一个 `visual.*` 一个 `model.*` ✓。拆两份是因为**装配时机不同** ✓：
    #: 视觉塔只在有图/帧时才要 ✓，语言塔每轮都要 ✓（放同一个文件就得整份常驻 ✓✗）。
    "qwen3vl_te", "qwen3vl_vision",
    "gguf_dequant", "gguf_to_llm", "torch_backend",
)

#: ⑥ 编排与可见面（"谁来调度、缺什么、退到哪"）
GROUP_ORCH: tuple[str, ...] = (
    "pipeline", "chain", "runtime", "facade", "loader", "registry", "inventory",
    "dryrun", "bridge", "script_parse",
    #: ⚠️ ``h3_encode``（2026-11-01 补登记 ✓）挂这里的理由：它**不写模型数学** ✓ ——
    #: 只做"先视觉塔、再语言塔"的串联 ✓（`plan_sequence` / `encode_condition` / `combine` ✓），
    #: 每一层的前向都在 `qwen3vl_vision` / `qwen3vl_te` 里 ✓。与 `pipeline` 同性质 ✓：
    #: 都是"把别人算好的东西按顺序接起来" ✓。放 `model` 组会把"一组都是网络本体"这句话冲淡 ✗。
    "h3_encode",
)

#: 组名 ⇒ 组内模块（**登记表是唯一真源** ✓：``__all__`` 由它拼出，结构守卫按它核盘 ✓）
MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "numeric": GROUP_NUMERIC,
    "format": GROUP_FORMAT,
    "weight": GROUP_WEIGHT,
    "token": GROUP_TOKEN,
    "model": GROUP_MODEL,
    "orchestration": GROUP_ORCH,
}

__all__ = [name for group in MODULE_GROUPS.values() for name in group]
