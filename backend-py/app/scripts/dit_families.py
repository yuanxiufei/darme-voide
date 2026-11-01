"""**盘上有几族 DiT、每族差什么才能跑**（2026-10-30 新增，服务「所有模型都能跑」✓）。

为什么要有这个脚本：引擎此前只认 H3（``dit.py``）与 SDXL（``sdxl.py``）两族 ✓，
盘上却躺着十来族 ✓ —— 但"差什么"从来没人答得出来 ✗：是权重不对？是文本编码器没下？
是 VAE 没有？还是**骨架根本没写**？三种缺法修法完全不同 ✓✗，混在一起就只能靠猜 ✓✗。

本脚本只做**核对**（判族 + 读拓扑 + 看配套件在不在 ✓），不装载、不出图 ✓：
它答的是"能不能开始写这一族的前向"，不是"能不能出片" ✓✗。

用法::

    python -X utf8 -m app.scripts.dit_families            # 全盘扫（默认根 = 清单 models_dir）
    python -X utf8 -m app.scripts.dit_families --json
    python -X utf8 -m app.scripts.dit_families --root D:/某/处
    python -X utf8 -m app.scripts.dit_families --strict   # 有「可建但没通路」的族 ⇒ 退出 1

⚠️ 退出码语义 ✗：默认 **0**（这是体检，不是门禁 ✓）；``--strict`` 才是门禁 ✓。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.services.engine import dit_family, inventory

#: 扫哪些目录（ComfyUI 的惯例落点 ✓ —— ``diffusion_models`` 与 ``unet`` 是同一类东西的两代叫法 ✓）
SCAN_KINDS: tuple[str, ...] = ("diffusion_models", "unet", "checkpoints")

#: 小于这个体积的不当 DiT 看（LoRA / VAE / 零碎补丁都在这个量级 ✓）
MIN_BYTES = 300 * 1024 ** 3 // 1000        # ≈0.3 GiB

#: 各族**配套件**（文件名都实测过盘 ✓ —— 不按"猜想的命名"写 ✓✗）。
#: 语义：``text`` / ``vae`` 为空 ⇒ 这族**本来就不需要**（例：SEEDVR2 是视频超分，只要 VAE ✓）。
COMPANIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "h3": {"text": ("text_encoders/qwen3vl_32b_minimax_h3_*",),
           "vae": ("vae/minimax_h3_video_vae*", "vae/minimax_h3_audio_vae*")},
    "sdxl": {"text": (), "vae": ()},        # 单文件 checkpoint 自带 TE+VAE ✓
    "krea2": {"text": ("clip/Qwen3_VL_4B/Krea2-Engineer-V1-*",), "vae": ("vae/ae.safetensors",)},
    "zimage": {"text": ("text_encoders/qwen_3_4b.safetensors",), "vae": ("vae/ae.safetensors",)},
    "qwen_image": {"text": ("text_encoders/qwen_2.5_vl_7b*",), "vae": ("vae/qwen_image_vae*",)},
    "qwen_image21": {"text": ("text_encoders/qwen3.5_9b_qwen_image_2.1_pe_*",),
                     "vae": ("vae/qwen_image_2.1_vae*",)},
    "flux": {"text": ("text_encoders/clip_l*", "text_encoders/t5xxl*"),
             "vae": ("vae/ae.safetensors",)},
    "wan": {"text": ("text_encoders/umt5_xxl*",), "vae": ("vae/wan_2.1_vae*",)},
    "seedvr2": {"text": (), "vae": ("vae/SEEDVR2/ema_vae*",)},
    "music3": {"text": ("text_encoders/minimax_music3_text_encoder*",),
               "vae": ("vae/minimax_music3_dav*",)},
}

#: 引擎侧**现在**走到哪一步（每族一句，改代码时同步改这里 ✓ —— 它只描述"通路"，不描述"效果" ✓）
ENGINE_STATE: dict[str, str] = {
    "h3": "已实现（`dit.py` + `h3_form.py`，专用形态）",
    "sdxl": "已实现（`sdxl.py`，UNet 专用）",
    "krea2": "前向骨架已实现（`dit_generic.py`：维度全从权重形状反推、键集合与盘上 878 张量逐键对账；"
             "**未做**参考图分支与端到端 TE/VAE 接线）",
    "zimage": "**端到端已闭合 + 已进产品入口**（`dit_lumina.py` 骨架 + `attach_qwen3_te` / `attach_flux_vae`；"
              "`runtime.ensure_loaded(stage=\"zimage\", ditPath/textWeights/imageVaeWeights/tokenizerPath)` 单独一条装配线 ✓，"
              "2026-10-31 实测装配 40.5 秒、1024² / 8 步出图 12.9 秒、同 seed 逐字节可复现 ✓ 且与手动 attach 那条路**逐字节相同** ✓）；"
              "σ 走 `schedules.flow_sigmas_for_steps`（flow ✓ 恒在 [0,1] ✓ 不是 karras ✗）；"
              "四格缺任一 ⇒ 当场拒并**只点名真缺的那几格** ✓（不猜路径 ✗：猜错是「装得上、出噪声」✓✗）",
    "qwen_image21": "**前向骨架已实现 + 已接进后端装载/前向**（`dit_qwen21.py`：单流 + SwiGLU + 三轴 RoPE(16,56,56)"
                    " + ZeroCentered RMSNorm + **两行调制**[文本 t=0 / 图像采样 t]；键集合与盘上 649 张量逐键对账 ✓、"
                    "与独立参照实现双精度一致 ✓、四处改坏反验可辨 ✓；`torch_backend._build_qwen21_module` 按权重建骨架"
                    "（7.12 G 参数 / bf16 约 14 GiB，meta 建 + `to_empty` ⇒ 峰值只一份 ✓）并搬 `text_dim/in_channels/vae_scale`）；"
                    "**VAE 解码已通**（2026-10-31 ✓：`vae_wan.py` 逐层对齐 comfy `wan/vae2_2.py` ✓，"
                    "`torch_backend.attach_wan_vae` 238 键严格装载 + `decode` 走 `wan21-real` 分支 ✓，"
                    "`latent_formats.qwen_image21` 登记 64ch/2D/16×/meanstd ✓，反缩放 `x*std+mean` 的 64 个数"
                    "与 comfy `QwenImage21` 逐值相同 ✓；实测 512×512 latent→PNG 0.36 s、显存峰值 1.3 GiB ✓）；"
                    "**TE 已闭合**（2026-10-31 ✓：`qwen35_te.py` 专写那座 Qwen3.5-9B **混合塔** ✓ —— "
                    "24 层 Gated DeltaNet[`in_proj_qkv/z/a/b` + 因果 conv1d k=4 切尾 + `A_log`/`dt_bias`]"
                    " ＋ 每 4 层一个 full attention[head_dim 256 / 4 KV 头 / qk-norm]，取 `layer_idx=-2` ✓；"
                    "块级与整层对 comfy 参考实现**逐位对拍 maxDiff=0** ✓、chunked delta rule 与朴素逐 token "
                    "递推两档一致 ✓、八处易错口径改坏反验可辨 ✓；`torch_backend.attach_qwen35_te` 把同文件的"
                    "549 键视觉塔与 lm_head **单列**不装 ✓；`qwen3_te.py` 那套 dense 壳装不上它 ✗ 是结构事实 ✓）；"
                    "**装配线已通**（`runtime.ensure_loaded(stage=\"qwen_image21\", ditPath/textWeights/"
                    "imageVaeWeights/tokenizerPath)` 第四条线 ✓，四格缺任一当场拒且只点名真缺的 ✓，装配时**显式核族**"
                    "✓ —— 2512 两代 `text_dim`/VAE 通道全一样 ✓✗，只有族名与 649 vs 1933 键能分 ✓）；"
                    "**分时驻留已实现**（2026-10-31 ✓，这是 2.1 与前三族的**真正区别** ✗✗）：三件**反量化到 bf16** 后的"
                    "驻留量是 DiT **13.26 GiB** ＋ TE **14.79 GiB** ＋ VAE **0.63 GiB** ＝ **28.68 GiB** ✓ > 卡上 22.49 ✓"
                    " ⇒ 「四格齐全一次装完」**必然 OOM** ✓✗（⚠️ 盘上那三个 int8 文件相加只有 ≈16.2 GiB ✓ —— 那是**装进来的"
                    "字节数** ✗ 不是驻留量 ✓，按它判'装得下'是这条线最容易犯的错 ✓✗），且 OOM 落在最后一件 ⇒ 报错指向 VAE ✓✗；"
                    "现在装配线**只装 DiT** ＋ `TorchBackend.declare_chain` 登记另两件 ✓，管线在 encode / sample / decode "
                    "三个边界调 `enter_phase` 换装 ✓（峰值 ≈**14.8 GiB** ✓，实测装配 20.4 秒 / reserved **13.27 GiB** ✓）；"
                    "⚠️ 两个配套事实：`release_part` 必须 `gc.collect()` ＋ `empty_cache()` ✓✗（只置 None 显存不退 ✓）；"
                    "`latent_shape` / `_latent_mode` 的守门必须读 `_config` **不读** `_model` ✓✗（`init` 跑在 encode 之后、"
                    "那时 DiT 已退 ⇒ 读 `_model` 会静默造一维占位 ✓✗，症状是采样第 1 步 `latents 要 (B,C,H,W)，实际 (512,)` ✓）；"
                    "⭐ **设备与精度必须成对** ✓（`_compute_dtype()` ✓：`device=\"cuda\"` ⇒ bf16 ✓、`device=\"cpu\"` ⇒ **fp32** ✓✗"
                    "—— 旧写法四处按「机器有没有卡」选 bf16 ✓，在有卡而显式 `device=\"cpu\"` 时 CPU 上会撞 bf16 的 conv 未实现 ✓✗；"
                    "CPU 上三件变 fp32 ⇒ 同一本账乘 2，峰值 ≈**29.5 GiB 内存** ✓，96 GB 装得下 ✓）；"
                    "⭐⭐ **设备是每阶段现算的** ✓（`device_policy.decide()` ✓，`policy` 默认 `auto` ✓）："
                    "退干净 → **现测** `mem_get_info` → 定设备/驻留方式 → 再装 ✓，四步不能挪 ✗"
                    "（先装后退 ⇒ 测出来的余量是「假的紧」✓✗，会把自己越推越往 CPU 且**回不来** ✓✗）；"
                    "判据三档：余量 ≥ 该件＋**1 GiB 硬余量** ⇒ 上卡 ✓；差一点（≥0.5 GiB）⇒ `residency=split` **联合** ✓；"
                    "＜0.5 GiB ⇒ 整件**转 CPU** ✓；需求未知时按**已用率**（≥80% 联合、≥92% 转移）✓；"
                    "`POLICIES` = `auto`／`gpu`／`cpu`／`balanced` ✓，拼错**当场抛** ✓✗（不许静默回落 ✓）；"
                    "⚠️ 主判据是**显存占用率**✗ 不是 SM 利用率 ✗（本仓没有 nvml ✓，`utilPercent` 只是注入位 ✓，"
                    "`source=\"memory\"` 必须如实报 ✓）；⚠️ 降级只换设备/驻留方式 ✓，**不动步数与分辨率** ✗✗；"
                    "⚠️ `describe().gpuAvailable`（本机有没有卡 ✓）与 `cudaAvailable`（这台后端**现在**在不在卡上 ✓）"
                    "是两格 ✗✗ —— 拿后者判「本机无卡」会读出「集显 ⇒ 很慢」这种反向结论 ✓✗（`engine_readiness` 已改读前者 ✓）；"
                    "`pipeline.run_sync(stage=\"qwen_image21\")` 整链（σ shift=**0.69** ✓，尺寸只吸附 16 —— 没有 patchify ✗）",
    #: ⭐⭐ **2512 / Edit 2511** ✓（2026-10-31 ✓）—— 与 `qwen_image21` **族名只差一个 `21`** ✗✗，
    #: 但件件不同 ✓：族名比较一律 `==` ✓（`startswith` 会让 2.1 被这条线的 shift 接走 ✓✗）。
    "qwen_image": (
        "**有骨架 + 有前向 + 装配线已接** ✓，`stage=\"qwen_image\"` 一张图从提示词到 PNG ✓；"
        "⚠️ 但**还没有一次真权重实跑出图** ✗（无实测 PNG ✓ 证据 ⇒ 别当成「已闭合」✗）。"
        "结构：双流 MMDiT ✓（`dit_qwen_image` ✓，盘上 **1933 键** ✓；Edit 2511 是 **3614 键** ✓，"
        "多出来的正是参考图那一路 ✗）—— `img_in` 吃 16×2×2＝**64** ✓ ⇒ **像素必须吸到 16 的倍数** ✓✗"
        "（只吸 8 ⇒ 潜边 83 是奇数 ⇒ 拍平当场形状错 ✓）；潜通道 **16** ✓（2.1 是 64 ✓），"
        "`vae_scale=8` ✓，时间口径 **`1000·σ` 不翻转** ✓（2.1 是 lumina 式翻转 ✗），"
        "flow shift **3.1** ✓（2.1 是 0.69 ✗、Z-Image 是 3.0 ✗ —— 抄错不报错、图僵 ✓✗）。"
        "文本塔：**Qwen2.5-VL-7B** ✓（`qwen25_te` ✓，28 层 / 3584 / 28×128 / kh=4 / qkv 带 bias / **无** QK-norm ✓，"
        "取**最后一层**、**不过** final norm ✓；同文件的 `visual.*` 714 与 `lm_head` 有意不装 ✓）；"
        "ChatML 的 `system` ＋ `user` 两段按**第 2 个** `<|im_start|>` 切 ✓✗（切错 ⇒ 提示词整段丢 ✓）。"
        "装配口 **四格** ✓（`ditPath` / `textWeights` / `tokenizerPath` / `imageVaeWeights` ✓）："
        "缺一格**当场拒** ✓、首行只列真缺的 ✓、装配里显式核 `family == qwen_image` ✓✗"
        "（`dit_generic` 会把 2.1 与 Krea2 的权重也装进来 ✓）；`_load_key` 已含 `textWeights` ✗ + `imageVaeWeights` ✗"
        "两格 ✓（漏了就会拿上一套件复用 ✓）。"
        "**分时驻留的第三本账** ✓✗：bf16 驻留 DiT **38.05 GiB** ＋ TE **16.1** ＋ VAE **0.24** ＝ **54.4 GiB** ✓"
        "（盘上 int8/fp8 三件只有 ≈47.03 GiB ✓ —— 又是「装入字节 ≠ 驻留量」✗）⇒ **单是 DiT 就超 22.49 的卡** ✓✗，"
        "必然 `split` 或转 CPU ✓；拿 2.1 的 13.26 判 ⇒ 结论「装得下」⇒ `residency` 留 `off` ⇒ 装到一半 OOM ✓✗。"
        "三个**按族参数**是共用 `enter_phase` 的解法 ✓：`declare_chain(text_kind=\"qwen25\")` ✓"
        "（缺省 `qwen35` 是 2.1 的 ⇒ 报的是「缺 `linear_attn.*`」，读着像权重错了 ✓✗）、"
        "`vae_family=\"wan21\"` ✓（16 通道；反缩放**恒等** ✓ —— `latent_formats.Wan21` 没覆写 scale/shift ✓，"
        "这 16 个 0/1 是**族事实** ✓ 不是「取不到就回退」✗）、`part_need_gib=QWEN_IMAGE_PART_NEED_GIB` ✓。"
        "就绪档：catalog 三条（`dit_qwen_image_2512` / `te_qwen25_vl_7b` / `vae_qwen_image` ✓，tags `qwen2512` ✓）"
        "＋ `STAGE_FILTERS[\"qwen_image\"].requireAll` ✓ ⇒ `readiness` 只列三件、全算必需 ✓✓"
        "（Edit 2511 **有意不打 tag** ✓✗：与 t2i 互斥，进 tag 会让 `requireAll` 把两代都算必需 ⇒ 假红 ＋ 多报 17.78 GiB ✓）。"
        "⭐ **Edit 2511 = 能装不能跑** ✓✗（**显式拒跑** ✓ 不静默 ✓）：它靠 `__index_timestep_zero__` 哨兵把参考图 token"
        "与目标 token 拼一条序列、给参考行喂 σ=0 的调制行 ✓；`QwenImageDiT.forward` 已收 `ref_latents` ✓，"
        "但参考图通路（Qwen2.5-VL `visual.*` ＋ 参考图 VAE 编码）**没接** ✓✗ ⇒ 照跑会画出「像提示词」的图、"
        "参考图整段缺席、0 报错 ✓✗；`denoise` 故在**前向之前**抛 ✓（`reason=\"pending\"` ✓），给两条出路："
        "先接视觉塔与 `ref_latents`，或换 2512 的 t2i 权重 ✓。"
        "`pipeline.run_sync(stage=\"qwen_image\")` 口径：默认 0.65 MP（视频口径 ✗）换 **1.76 MP / 1328²** ✓"
        "并**如实告警** ✓、吸附 16 ✓、`frames=1` / `fps=0` ✓、`timesteps` 就是 σ ✓、cfg=1.0 只报不改"
        "（官方档 4.0 / 50 步 ✓）—— `engine_qwen_image_test`（50 项 ✓，不装真权重 ✓）钉着上面每一条 ✓"),
}

#: **不走通用骨架**的族 ✓ —— 它们有自己的实现（H3 的分块打包形态、SDXL 的 UNet ✓）。
#: 这些族"depth/hidden 读不出来"**不是缺口** ✗（``buildable`` 对它们没意义 ✓），
#: 拿通用口径去判就会把"已实现"报成"拓扑读不全" ✓✗（本脚本第一版就是这么误报的 ✓）。
DEDICATED_FAMILIES: frozenset[str] = frozenset({"h3", "sdxl"})


def blocked(report: dict[str, Any]) -> list[str]:
    """``--strict`` 要拦的族 ✓ —— **两条**：没有通路 ✓ / 配套件缺 ✓（拓扑读不全不算：那是"还没判" ✓✗）。"""
    return sorted(key for key, bucket in report["families"].items()
                  if key not in ENGINE_STATE or bucket["companions"]["missing"])


def _roots(explicit: str = "") -> list[Path]:
    if explicit:
        return [Path(explicit)]
    root = inventory.models_dir()
    if root is not None:
        return [root]
    return [path for path, _ in inventory.candidate_roots()]


def _candidates(root: Path) -> list[Path]:
    found: list[Path] = []
    for kind in SCAN_KINDS:
        base = root / kind
        if not base.is_dir():
            continue
        found.extend(p for p in base.rglob("*.safetensors") if p.is_file())
    return sorted(found)


def _size_ok(path: Path) -> bool:
    try:
        return path.stat().st_size >= MIN_BYTES
    except OSError:
        return False


def _has(root: Path, pattern: str) -> list[str]:
    """``pattern`` 在 ``root`` 下命中几个文件（``*`` 允许出现在目录段里 ✓）。"""
    return [p.name for p in root.glob(pattern) if p.is_file()]


def _companions(root: Path, family: str) -> dict[str, Any]:
    wanted = COMPANIONS.get(family)
    if wanted is None:
        return {"checked": False, "text": [], "vae": [],
                "missing": ["配套件清单里没有这一族（`COMPANIONS` 未登记 ⇒ **没查** ✗）"]}
    missing: list[str] = []
    hits: dict[str, list[str]] = {}
    for slot in ("text", "vae"):
        patterns = wanted.get(slot, ())
        names: list[str] = []
        for pattern in patterns:
            found = _has(root, pattern)
            if found:
                names.extend(found)
            else:
                missing.append(f"缺 {slot}：`{pattern}`")
        hits[slot] = names
    return {"checked": True, "text": hits["text"], "vae": hits["vae"], "missing": missing}


def survey(*, root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    for path in _candidates(root):
        if not _size_ok(path):
            continue
        try:
            spec = dit_family.detect_file(path)
        except Exception as err:  # noqa: BLE001 —— 单个文件读崩不该让整盘扫挂掉 ✓
            unknown.append({"path": str(path), "why": f"判族抛错：{err}"})
            continue
        if spec is None:
            unknown.append({"path": str(path), "why": "键名不在任何一族的指纹里"})
            continue
        files.append({"path": str(path), "name": path.name, "family": spec.key,
                      "spec": spec.to_dict()})

    return summarise(files, root=root, unrecognised=unknown)


def summarise(files: list[dict[str, Any]], *, root: Path,
              unrecognised: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """已判好族的清单 ⇒ 按族归并 + 逐族算「差什么」（**纯函数** ✓：不读盘，除配套件之外 ✓）。"""
    unknown = list(unrecognised or [])
    families: dict[str, dict[str, Any]] = {}
    for item in files:
        bucket = families.setdefault(item["family"], {
            "family": item["family"], "label": item["spec"]["label"], "files": [],
            "buildable": 0, "unknown": 0,
        })
        bucket["files"].append(item)
        if item["spec"]["buildable"]:
            bucket["buildable"] += 1
        bucket["unknown"] += len(item["spec"]["unknown"])
        bucket["dedicated"] = item["family"] in DEDICATED_FAMILIES

    for key, bucket in families.items():
        bucket["companions"] = _companions(root, key)
        bucket["engine"] = ENGINE_STATE.get(key, "**没有通路**（骨架未写 ✓✗）")
        # 「差什么才能跑」= 骨架 + 配套件 + 声明过的未知项（**缺一项就跑不起来** ✓）
        gaps: list[str] = []
        if key not in ENGINE_STATE:
            gaps.append("前向骨架未实现 ⇒ 只能判族、不能前向 ✗")
        # ⚠️ 专用族**不拿通用口径判** ✗：H3/SDXL 的 depth/hidden 本来就"读不出来" ✓，
        #    它们走各自的实现与键名核对（`h3_keys` / `sdxl.py`）✓ —— 报"拓扑读不全"是假警报 ✓✗。
        if not bucket["dedicated"] and bucket["buildable"] < len(bucket["files"]):
            gaps.append(f"{len(bucket['files']) - bucket['buildable']} 份拓扑读不全（hidden/depth 缺失）")
        gaps.extend(bucket["companions"]["missing"])
        if bucket["unknown"] and not bucket["dedicated"]:
            gaps.append(f"{bucket['unknown']} 条「权重里读不到的参数」待补（见 `unknown` ✓）")
        bucket["gaps"] = gaps

    return {"root": str(root), "families": families, "unrecognised": unknown,
            "fileCount": len(files)}


#: 各族要说的话（**人看的摘要** ✓；数据仍是 :func:`survey` 的结论 ✓）
def render(report: dict[str, Any]) -> str:
    lines = [f"模型根：{report['root']}", f"认出 {report['fileCount']} 份 DiT 权重", ""]
    header = ("族", "份数", "可建", "depth/hidden", "开关（norm/attn/mlp/cond）", "TE/VAE", "通路")
    rows: list[tuple[str, ...]] = []
    for key in sorted(report["families"]):
        bucket = report["families"][key]
        first = bucket["files"][0]["spec"]
        switches = "/".join(str(first[field]) for field in ("norm", "attn", "mlp", "cond"))
        if first.get("qkNorm"):
            switches += "/qknorm"
        if first.get("attnGate"):
            switches += "/gate"
        if first.get("dualStream"):
            switches += "/双流"
        comp = bucket["companions"]
        parts = (f"TE {'✓' if comp['text'] else ('—' if comp['checked'] else '?')}"
                 f" VAE {'✓' if comp['vae'] else ('—' if comp['checked'] else '?')}")
        buildable = "专用" if bucket["dedicated"] else f"{bucket['buildable']}/{len(bucket['files'])}"
        geometry = ("—" if bucket["dedicated"] else f"{first['depth']}/{first['hidden']}")
        rows.append((key, str(len(bucket["files"])), buildable, geometry, switches, parts,
                     "有" if key in ENGINE_STATE else "**无**"))
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    lines.append("")
    for key in sorted(report["families"]):
        bucket = report["families"][key]
        lines.append(f"■ {key}（{bucket['label']}）")
        for item in bucket["files"]:
            spec = item["spec"]
            if bucket["dedicated"]:
                lines.append(f"  · {item['name']}：{bucket['engine']}")
                continue
            head = f"  · {item['name']}：depth={spec['depth']} hidden={spec['hidden']}"
            if spec["strippedPrefix"]:
                head += f"（剥前缀 `{spec['strippedPrefix']}`）"
            lines.append(head + ("" if spec["buildable"] else "  ⚠️ 拓扑读不全"))
        if bucket["gaps"]:
            for gap in bucket["gaps"]:
                lines.append(f"  ⚠️ {gap}")
        else:
            lines.append("  ✓ 这一族该有的都齐了")
    if report["unrecognised"]:
        lines.append("")
        lines.append("■ 认不出的（**不是**「能跑」，是「还没判」✓）")
        for item in report["unrecognised"]:
            lines.append(f"  · {item['path']}：{item['why']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盘点盘上的 DiT 族，并答「差什么才能跑」")
    parser.add_argument("--root", default="", help="模型根（默认取清单 models_dir）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--strict", action="store_true",
                        help="有「可建但没通路」或「配套件缺」的族 ⇒ 退出 1（默认只报不拦 ✓）")
    args = parser.parse_args(argv)

    roots = _roots(args.root)
    root = next((item for item in roots if item.is_dir()), roots[0])
    report = survey(root=root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))

    if args.strict:
        stuck = blocked(report)
        if stuck:
            print(f"\n阻塞（--strict）：{', '.join(stuck)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
