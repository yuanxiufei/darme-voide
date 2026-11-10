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
    #: ⚠️ ``profile``（2026-11-07 新增 ✓，G1 ✓）挂这里：它是**机器档那张表** ✓（`auto` / `gpu` /
    #: `l1` 8 GiB 卡·32 GB 内存 / `l2` 6 GiB·16 GB / `cpu` ✓）＋ 一句话展开成一组旋钮 ✓。
    #: 与 `tiers`（**精度**档位：步数/分辨率/加速件 ✓）**并列而不是合并** ✗✗ —— 两张表正交 ✓：
    #: 「这张卡多小」与「这一档跑几步」互不推导 ✓。它**不 import torch** ✓（读数由调用方给 ✓）、
    #: **不读盘** ✓、**不装任何东西** ✗ ⇒ 与 `device_policy` 同性质（都在回答「这台机器能怎么跑」✓），
    #: 放 `orchestration` 组会与「谁来编排阶段」混为一谈 ✗。
    "profile",
    #: ⚠️ ``prefetch``（2026-11-01 移植自 airllm ✓，**同日接进装载** ✓）挂这里：它决定的是
    #: **权重什么时候进内存** ✓（趁上一件在反量化/写卡，就把下一件读进内存 ✓），和 `residency` /
    #: `device_policy` 回答的是同一句话的另半句 ✓✗ —— 那两者管"落在**哪台设备**"✓，这管"落在
    #: **什么时候**"✓。接线点是 `weights._load_streaming` 的逐张量读 ✓（作用域一次装载即止 ✓，
    #: 跨阶段预取被否掉 ✗）。它**不决定装哪件** ✗（装配线给 key 与函数 ✓）、**不 import torch** ✗。
    "prefetch",
    #: ⚠️ ``host_cache``（2026-11-02 补登记 ✓）挂这里：三级内存里**内存那一级的配额账** ✓
    #: （SSD → 内存 → 显存 ✓ —— `weights` 管第一段、这件管第二段的上限、`residency` 管第三段 ✓）。
    #: 它只算"该卸谁、卸完还差多少"✓，**不碰张量** ✗（真换成 0 元素占位是 `residency` 干的 ✓）。
    "host_cache",
    #: ⚠️ ``foresight``（2026-11-02 补登记 ✓）挂这里：它是**显存那一侧的"什么时候"** ✓✗ ——
    #: `device_policy` 答"落哪台设备"、`prefetch` 答"什么时候进内存"、这件答"什么时候**上卡**"✓。
    #: 它只做判据（该不该发、落地没落地、能不能用 ✓），分配与拷贝在 `residency._SlotBook` ✓；
    #: **不 import torch** ✗（无卡机器上也能自检 ✓）。默认 **0 槽 ＝ 不存在这一级** ✓。
    "foresight",
    #: ⚠️ ``materialize``（2026-11-07 新增 ✓，G3 ✓）挂这里：它把「反量化算出来的字节」按
    #: 源指纹落到盘上 ✓，第二次装载只读盘 ✓ —— 与 `hybrid_load`（搬**原字节**）是**同一件事的
    #: 另一种输入** ✓：两者都在「权重的字节从哪来」这一层 ✓，都不决定跑哪一步 ✗。
    #: 它**不 import torch / numpy** ✗✗（数值转换在调用方 ✓，本模块只认 bytes ✓）⇒ 零依赖可测 ✓。
    "materialize",
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
    #: ⚠️ ``umt5_tokenizer``（2026-11-01 补登记 ✓）挂这里是**照它的性质** ✓：UMT5 的 sentencepiece
    #: **unigram** 分词器（`nmt_nfkc` 归一 ＋ `add_dummy_prefix` ✓，词表 256384 ✓）—— 与 Qwen 系 BPE
    #: 是两套东西 ✓✗，所以它与 `tokenizer_bpe` / `tokenizer_own` 并列 ✓，**不是**文本塔本体 ✗
    #: （塔在 `umt5_te` ✓）。
    "umt5_tokenizer",
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
#: ⚠️ ``dit_build``（2026-11-01 补登记 ✓）与 `dit_family` **成对而非合并** ✓：`dit_family`
#: 回答"这份权重是哪个族" ✓（只看键名 ✓、不建网 ✓）；`dit_build` 回答"这个族怎么建" ✓
#: （一族一行 `FamilyBuilder` ✓ + 一份公共流程 ✓）。合并会让"判族"被迫 import 全部骨架 ✗。
#: ⚠️ ``dit_flux``（2026-11-01 补登记 ✓）与 `dit_qwen_image` 同层 ✓：都是**网络本体** ✓，
#: 只是 Flux 那条是两段（19 双流 + 38 单流 ✓）、时间口径 `1000·σ` 不翻转 ✓。
GROUP_MODEL: tuple[str, ...] = (
    "dit", "dit_lumina", "dit_qwen21", "dit_qwen_image", "dit_family", "dit_build",
    #: ⚠️ ``dit_wan22``（2026-11-01 补登记 ✓）与 `dit_flux` / `dit_qwen_image` 同层 ✓：都是**网络本体** ✓。
    #: 它是本仓唯一**五维潜变量 + 视频**的建族 ✓（40 层 / `in_dim` 36 / 文本口 umt5 4096 ✓），
    #: 由 `dit_build` 静态 import 并登记成 `wan` 行 ✓ —— 只漏了这张登记表 ✓✗（结构守卫 ㊀ 报的就是它 ✓）。
    "dit_wan22",
    #: ⚠️ ``dit_seedvr2``（2026-11-01 补登记 ✓）与 `dit_wan22` / `dit_flux` 同层 ✓：都是**网络本体** ✓ ——
    #: SEEDVR2（NaDiT）是视频修复/超分主干 ✓（32 块 ✓、前 10 块双塔后 22 块共享 ✓、`in=33` ✓）。
    #: 它不是装配行 ✗（`dit_build` 里有没有登记成族是另一件事 ✓），也不做编排 ✗。
    "dit_seedvr2",
    "dit_flux", "dit_generic",
    #: ⚠️ ``sdpa``（2026-11-10 新增 ✓）挂这里的理由：它**不是**网络本体 ✗，但它是网络本体
    #: **每一次前向都必走的一格** ✓（`dit` / `dit_flux` / `dit_generic` / `dit_lumina` /
    #: `dit_music3` / `dit_qwen21` / `dit_qwen_image` / `dit_wan22` / `h3_form` 九处的 SDPA
    #: 全从这里出去 ✓）。它只做**核选择**：判 cuDNN fused attention 能不能用 ✓、能就套
    #: `sdpa_kernel([CUDNN])` ✓，判不过走原路 ✓ —— 不写模型数学 ✗（一行 softmax 都没有 ✓）、
    #: 不编排阶段 ✗（谁在什么时候调它是各族 forward 的事 ✓）、不碰显存 ✗（不 has 权重 ✓）。
    #: 放 `orch` 组会让人以为它在决定「跑哪一步」 ✗✗（那正是 `pipeline` / `runtime` 的活 ✓）；
    #: 放 `numeric` 组更不行 —— 它 import torch ✓（那一组硬约束是「不装 torch 也能测」✓✗）。
    "sdpa",
    #: ⚠️ ``dit_music3``（2026-11-02 补登记 ✓）与 `dit_flux` / `dit_generic` 同层 ✓：都是**网络本体** ✓
    #: （Music3 的音乐主干 ✓，`dit_build.BUILDERS` 里登记成 `music3` 行、契约 ``latent_dims=3`` ✓）。
    #: 它**不是**编排 ✗（谁调它、怎么排阶段是 `pipeline` / `runtime` 的事 ✓），也不做量化 ✗。
    "dit_music3",
    #: ⚠️ ``te_music3``（2026-11-02 补登记 ✓）与 `dit_music3` 同层 ✓：都是**网络本体** ✓ ——
    #: 它是 Music3 的 **语义塔** ✓（Qwen3-8B 形态 36 层 + RVQ 深度解码器 ✓，int8_convrot
    #: 常驻 160 层 ✓，词表内嵌在权重文件里 ✓）。产出 ``context (1,F,8×4096)`` ✓ 正好是
    #: `dit_music3` 契约的输入 ✓。它不是编排 ✗、不是文本编码器通用件 ✗（只服务这一族 ✓）。
    #: ⚠️ 同日我先写过一份 ``ar_music3.py`` 的等价实现 ✓，逐项对拍后**删掉收敛到这份** ✗ ——
    #: 两份同名实现并存是本仓最恨的那类（改一处、另一处静默漂 ✓✗）。
    "te_music3",
    "h3_form", "h3_keys", "h3_edit", "sdxl",
    "sdxl_backend",
    "sdxl_vae", "vae_flux", "vae_wan", "vae", "vae_h3", "audio_vae", "upscale", "upscale_net",
    #: ⚠️ ``vae_seedvr2``（2026-11-01 补登记 ✓）与 `vae_wan` / `vae_h3` 同层 ✓：都是"潜变量 ⇒ 像素"的
    #: 网络本体 ✓（SEEDVR2 的 ``VideoAutoencoderKL``：4 级下采样块 + 4 级上采样块 ✓，空间各 3 次 ⇒ ÷8 ✓、
    #: 时间各 2 次 ⇒ ÷4 ✓，编码器直出 32 通道 ⇒ 取前 16 均值即潜变量 ✓ 16ch ✓）。
    #: 它**不是**容器 ✗（``latent_container``）、**不是**规格表 ✗
    #: （``latent_formats`` 的 ``seedvr2`` 口径才是 0.9152 那个数 ✓），也不做编排 ✗。
    #: 之前漏的只是这张登记表 ✓✗（结构守卫 ㊀ 报的就是它 ✓）。
    "vae_seedvr2",
    #: ⚠️ ``vae_music3``（2026-11-02 补登记 ✓）与 `audio_vae` / `vae_seedvr2` 同层 ✓：都是「潜变量 ⇒ 波形/像素」
    #: 的网络本体 ✓（Music3 的音频 VAE ✓，3 维潜变量那一路 ✓）。它**不是**混音/编排 ✗
    #: （`audio_mix` 在本目录 ``numeric`` 组 ✓、排阶段是 `pipeline` 的事 ✓）。
    "vae_music3",
    "text_encoder", "clip_text", "qwen3_te", "qwen35_te", "qwen25_te", "llm", "llm_backend",
    #: ⚠️ ``umt5_te``（2026-11-01 补登记 ✓）与 `qwen3_te` / `qwen25_te` / `qwen35_te` 并列 ✓：
    #: Wan2.1/2.2 那座 **UMT5-XXL** 文本塔 ✓（4096 宽 / 24 层 / **双向**注意力 ✓、相对位置偏置
    #: **每层各一份** ✓、FFN 门控 ✓）—— 与本仓其余几座（因果 + `layer_idx=-2` ✓）口径相反 ✗，
    #: 合并会让 attention mask 方向混用 ✓✗（症状是「出图与提示词几乎无关」而不是报错 ✓）。
    "umt5_te",
    #: ⚠️ ``qwen3vl_te`` / ``qwen3vl_vision``（2026-11-01 补登记 ✓）与 `qwen3_te` / `qwen25_te` 并列
    #: 而非合并 ✗：它们是 H3 那一座 **Qwen3-VL-32B** ✓ —— 视觉塔与语言塔在**同一个 safetensors** 里 ✓，
    #: 但两半边层型完全不同（27 层 ViT + DeepStack merger vs 语言塔 interleaved mrope ✓）、
    #: 键前缀一个 `visual.*` 一个 `model.*` ✓。拆两份是因为**装配时机不同** ✓：
    #: 视觉塔只在有图/帧时才要 ✓，语言塔每轮都要 ✓（放同一个文件就得整份常驻 ✓✗）。
    "qwen3vl_te", "qwen3vl_vision",
    #: ⚠️ ``qwen25vl_vision``（2026-11-01 ✓）与 `qwen3vl_vision` **并列而非合并** ✓：两者都是视觉塔 ✓，
    #: 但**尺子完全不同** ✗ —— Qwen2.5-VL 是 patch **14** / CLIP 均值方差 / 窗口注意力（4 层走全图）✓，
    #: Qwen3-VL 是 patch **16** / 0.5 均值 / DeepStack ✓。「共用一份代码 + 一堆开关」会让口径混用
    #: **不报错**只让整图偏色 ✓✗（这正是 `qwen3vl_pixels.process_image_with` 存在的理由 ✓）。
    "qwen25vl_vision",
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
    #: ⚠️ ``canvas`` / ``profiles``（2026-11-01 新增 ✓）也挂这里，**不进** `model` 组 ✗：
    #: 一个只做"像素 ↔ 比例 ↔ 吸附"的算术 ✓、一个只是**数据表** ✓（用途 ⇒ 画布/步数/cfg ✓，
    #: 出处写工作流文件名 ✓）—— 两者都**不写模型数学** ✗，与 `pipeline` / `facade` 同性质 ✓。
    #: 放 `model` 组会让人以为"画布是网络的一部分" ✓✗（它恰恰是要**脱离**具体网络才搬得动 ✓）。
    "canvas", "profiles",
    #: ⚠️ ``tuning``（2026-11-06 新增 ✓）挂这里：它是**参数覆盖层** ✓ ——
    #: 读外部 JSON、白名单校验、把值换进 `GenerationRequest` ✓，不含任何模型数学 ✗、
    #: 不碰显存 ✗、不决定跑哪一族 ✗。与 `profiles` 同性质（都是"数据决定数值" ✓），
    #: 区别只在于它是**可写可回滚**的那一份 ✓✗（profiles 是盘上模型的说明，tuning 是人的裁决）。
    "tuning",
    #: ⚠️ ``feasibility``（2026-11-07 新增 ✓）挂这里的理由：它**只下判断、不装东西** ✓ ——
    #: 输入是装配线挑到的件 ＋ `device_policy` 的读数 ✓，输出是 ``verdict`` 四档 ✓，
    #: 全程**不装载** ✓、**不写模型数学** ✗（每件要多少 GiB 一律问 `torch_backend.stage_part_need` ✓）。
    #: 与 `inventory`（件齐不齐 ✓）/ `loader`（怎么排班 ✓）/ `dryrun`（能不能跑编排 ✓）同性质 ✓ ——
    #: 放 `weight` 组会让"量化与装载"这句话失真 ✗（它一件都不装 ✓）。
    "feasibility",
    #: ⚠️ ``qc_feedback``（2026-11-07 新增 ✓）挂这里：它是**读结论、给闸门** ✓ ——
    #: 读派生索引库里已归因的质检跑次 ✓、算出"降步数会不会把质检通过率打下去" ✓，
    #: 不写模型数学 ✗、不装东西 ✗、不改参数 ✗（改参数仍然只走 `tuning.set_field` ✓，
    #: 那是人的裁决 ✓）。与 `feasibility` 同性质：**只下判断** ✓。
    "qc_feedback",
    #: ⚠️ ``output_paths``（2026-11-07 新增 ✓）挂这里：它只是**落点口径** ✓ ——
    #: 「没给 `outputs_dir` 时落哪」的唯一真源 ✓（`<storageRoot>/engine/<kind>/` ✓、
    #: 干跑 `<dataRoot>/engine-dryrun/<kind>/` ✓）。**不写模型数学** ✗（不碰张量 ✗）、
    #: **不做编排** ✗（谁调 `write` 是 `pipeline` 的事 ✓），与 `bridge` 的落点口径同一棵树 ✓。
    #: 放 `model` 组会让"这里是网络本体"这句话失真 ✗。
    "output_paths",
    #: ⚠️ ``hooks``（2026-11-07 补登记 ✓）挂这里：它是**引擎 → 服务层的单向接线口** ✓ ——
    #: 引擎只认 :data:`hooks.SOURCES` 这张表 ✓，实现由服务层在 ``app/main.py`` 启动时登记 ✓；
    #: 引擎侧**不 import 服务层任何模块** ✗（独立性契约 ② 判的就是这个 ✓），
    #: 没登记时各调用点**如实说"没接入"** ✓（不是静默给 0 条 ✓✗）。
    #: 与 `bridge` 同为「跨层接线」✓ 故同组；它不写模型数学 ✗、不装东西 ✗（登记的是模块对象 ✓）。
    "hooks",
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
