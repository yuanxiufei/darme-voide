"""H3 检查点**键名核对器**（零依赖 ✓ torch-free ✓）—— 真权重到手那一刻的「插上就验」。

## 为什么需要它（2026-09-20）

`H3FormTrunk` 的模块名与参考实现**逐字对齐** ✓ ⇒ 装真权重**不需要改名映射表** ✓ ——
但「对齐」此前只是**代码侧的声明** ✗：19.53 GiB 的真 checkpoint 到手之前，没有任何东西
机械验证过「这份权重的**键全集 + 形状关系**」真的能装进去 ✗（`weights.load_module_weights`
要**先建好 50 层模型**才报 missing ✗ —— 那已经在 GPU 上了 ✓✗）。

本模块把这件事**前置** ✓：给定「键名 → 形状」清单（safetensors / GGUF 体检都能给 ✓），
**不建模型、不碰 torch** ✓，直接核对：

* **键全集**：按参考实现的结构生成期望键表（见下方事实来源 ✓）；
* **形状关系**：尺寸**从 checkpoint 自己推导** ✓（hidden / head_dim / ffn / t_dim …），
  再逐键核对形状 ⇒ 内部不一致（如 `qkv_proj` 行数 ≠ 3×`out_proj` 列数 ✗）**当场红** ✓；
* **PDD 头库**：`video_out` / `audio_out` 行数 ÷ 单头宽度 ⇒ 头库大小 ✓
  （非整数倍 ⇒ 报错 ✓ 不静默回落 1 ✗）；
* **curve 变体**：`adaln_t_table` **替换** `time_embedder.*`（参考实现的曲线检查点 ✓）；
* **RefinerBlock 没有 adaLN** ✓ —— 早年「近似实现」多出来的
  `token_refiner.blocks.N.adaln_proj.*` 会被当 unexpected 报出 ✗。

## 事实来源（`reference/ComfyUI/comfy/ldm/minimax/model.py` ✓ 2026-09-20 读全 ✓）

* 顶层模块（`MiniMaxH3Model.__init__` ✓）：`video_patch_proj` / `audio_patch_proj`
  （**都 fp32** ✓）/ `condition_proj` / `time_embedder{proj_in, proj_out}`（或 `adaln_t_table` ✓）/
  `rope.inv_freq`（**缓冲区** ✓ 长度自由 ✗）/ `token_refiner` / `blocks` / `final_layer` ✓；
* 每层 DiTBlock（10 键 ✓）：`norm1` / `norm2` / `attn.{qkv_proj, q_norm, k_norm, out_proj}` /
  `mlp.{fc1, fc2}` / `adaln_proj.linear{weight, bias}`（**18 路** = expand 6 × modalities 3 ✓）；
* 每层 RefinerBlock（**8 键 ✓ 没有 adaln ✗**）：`norm1` / `norm2` /
  `attn.{qkv_proj, q_norm, k_norm, out_proj}` / `mlp.{fc1, fc2}` + 收尾 `final_norm` ✓；
* `final_layer`（7 键 ✓）：`norm` / `adaln_proj.linear{weight, bias}`（**2 路** ✓）/
  `video_out` / `audio_out`（**都 fp32 ✓ 且可打包 PDD 头库** ✓：行数 = banks×单头宽度 ✓）；
* 可选：`attn.to_gate_compress`（VSA 门 ✓ —— 只有带稀疏注意力补丁的检查点有 ✗）。

⚠️ 本模块**不核 dtype** ✗（fp32 岛的分布是参考实现的行为 ✓，装载时可 cast ✓ ——
不在「能不能装」的判据里 ✗）。也不判形态 ✗ —— 用 `h3_form.looks_like_h3_form` ✓。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .h3_form import H3_TRUNK_DEFAULTS, H3_VIDEO_OUT_KEY, head_banks_from_shape, video_patch_dim

__all__ = ["H3KeyAudit", "H3TrunkConfig", "audit_h3_checkpoint", "expected_h3_keys",
           "infer_h3_trunk_config"]

_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")
_REFINER_KEY = re.compile(r"^token_refiner\.blocks\.(\d+)\.")

#: 期望表里**只要求存在、形状另核**的键 ✗（形状取决于检查点事实 ✓：头库大小 / 网格长度 ✗）
_FREE_SHAPE_KEYS = frozenset({
    "rope.inv_freq",          # 长度 = ropeInvFreqLen（H3 默认 16 ✓ 但这是检查点自由项 ✗）
    "adaln_t_table",          # (grid, t_dim)：grid 是曲线检查点的事实 ✓
    H3_VIDEO_OUT_KEY, "final_layer.video_out.bias",                  # 行数 = banks×单头宽度 ✓
    "final_layer.audio_out.weight", "final_layer.audio_out.bias",
})

#: 可选键的模式（有也不算 unexpected ✗ —— 参考实现里就是「部分检查点才有」✓）
_OPTIONAL_PATTERNS = tuple(re.compile(p) for p in (
    r"^blocks\.\d+\.attn\.to_gate_compress\.weight$",     # VSA 门 ✓
    r"^token_refiner\.blocks\.\d+\.attn\.to_gate_compress\.weight$",
))


def _shape(value: Any) -> tuple[int, ...] | None:
    """归一成 ``tuple[int, ...]`` ✓（``None`` / 空 ⇒ ``None`` = 只看键名 ✗）。"""
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


def _default_dims() -> dict[str, int]:
    """H3 出厂尺寸 → 核对器用的维度表 ✓（**唯一来源** `H3_TRUNK_DEFAULTS` ✓ 不另抄 ✗）。"""
    d = H3_TRUNK_DEFAULTS
    return {
        "hidden": int(d["hidden"]), "head_dim": int(d["head_dim"]),
        "inner": int(d["heads"]) * int(d["head_dim"]), "ffn": int(d["ffn"]),
        "t_dim": int(d["time_dim"]), "time_in": int(d["time_input_dim"]),
        "text_dim": int(d["text_dim"]),
        "video_dim": video_patch_dim(int(d["latents_dim"]), tuple(d["patch_size"])),
        "audio_dim": int(d["audio_latents_dim"]),
    }


def _block_layer_keys(prefix: str, dims: dict[str, int], *, with_adaln: bool) -> dict[str, tuple[int, ...]]:
    """一层块的期望键表 ✓（DiT 带 18 路 adaLN ✓ / Refiner **不带** ✗ —— 参考实现如此 ✓）。"""
    keys: dict[str, tuple[int, ...]] = {
        f"{prefix}.norm1.weight": (dims["hidden"],),
        f"{prefix}.norm2.weight": (dims["hidden"],),
        f"{prefix}.attn.qkv_proj.weight": (3 * dims["inner"], dims["hidden"]),
        f"{prefix}.attn.q_norm.weight": (dims["head_dim"],),
        f"{prefix}.attn.k_norm.weight": (dims["head_dim"],),
        f"{prefix}.attn.out_proj.weight": (dims["hidden"], dims["inner"]),
        f"{prefix}.mlp.fc1.weight": (2 * dims["ffn"], dims["hidden"]),
        f"{prefix}.mlp.fc2.weight": (dims["hidden"], dims["ffn"]),
    }
    if with_adaln:
        keys[f"{prefix}.adaln_proj.linear.weight"] = (6 * dims["hidden"] * 3, dims["t_dim"])
        keys[f"{prefix}.adaln_proj.linear.bias"] = (6 * dims["hidden"] * 3,)
    return keys


def expected_h3_keys(dims: Mapping[str, int], *, depth: int, refiner_layers: int,
                     variant: str = "standard") -> dict[str, tuple[int, ...] | None]:
    """生成 H3 检查点的**期望键表** ✓（键 → 形状；``None`` = 只要求存在 ✓）。

    ⚠️ ``dims`` 用 :func:`audit_h3_checkpoint` **推导出的尺寸** ✓（不是出厂值 ✗）——
    这样非出厂尺寸的检查点也能核「结构自洽」✓。
    """
    out: dict[str, tuple[int, ...] | None] = {}
    out[f"video_patch_proj.weight"] = (dims["hidden"], dims["video_dim"])
    out["video_patch_proj.bias"] = (dims["hidden"],)
    out["audio_patch_proj.weight"] = (dims["hidden"], dims["audio_dim"])
    out["audio_patch_proj.bias"] = (dims["hidden"],)
    out["condition_proj.weight"] = (dims["hidden"], dims["text_dim"])
    out["condition_proj.bias"] = (dims["hidden"],)
    if variant == "standard":
        out["time_embedder.proj_in.weight"] = (dims["hidden"], dims["time_in"])
        out["time_embedder.proj_in.bias"] = (dims["hidden"],)
        out["time_embedder.proj_out.weight"] = (dims["t_dim"], dims["hidden"])
        out["time_embedder.proj_out.bias"] = (dims["t_dim"],)
    elif variant == "curve":
        out["adaln_t_table"] = None
    else:
        raise ValueError(f"variant 只认 standard / curve（收到 {variant!r} ✗）")
    out["rope.inv_freq"] = None
    out["token_refiner.final_norm.weight"] = (dims["hidden"],)
    out["final_layer.norm.weight"] = (dims["hidden"],)
    out["final_layer.adaln_proj.linear.weight"] = (2 * dims["hidden"], dims["t_dim"])
    out["final_layer.adaln_proj.linear.bias"] = (2 * dims["hidden"],)
    out[H3_VIDEO_OUT_KEY] = None
    out["final_layer.video_out.bias"] = None
    out["final_layer.audio_out.weight"] = None
    out["final_layer.audio_out.bias"] = None
    for index in range(depth):
        out.update(_block_layer_keys(f"blocks.{index}", dims, with_adaln=True))
    for index in range(refiner_layers):
        out.update(_block_layer_keys(f"token_refiner.blocks.{index}", dims, with_adaln=False))
    return out


@dataclass
class H3KeyAudit:
    """一次核对的结论 ✓（``ok=False`` 时看 ``problems`` / ``missing`` / ``shapeMismatch`` ✓）。"""

    variant: str = "unknown"
    depth: int = 0
    refiner_layers: int = 0
    head_banks_video: int = 1
    head_banks_audio: int = 1
    dims: dict[str, int] = field(default_factory=dict)
    dims_from_checkpoint: bool = True
    is_default_sized: bool = False
    expected_count: int = 0
    present_count: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatch: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """**能装** = 键齐 + 形状一致 + 无结构问题 ✓（unexpected 单列 ✗ 不阻断 ✓）。"""
        return not (self.problems or self.missing or self.shape_mismatch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "variant": self.variant, "depth": self.depth,
            "refinerLayers": self.refiner_layers,
            "headBanksVideo": self.head_banks_video, "headBanksAudio": self.head_banks_audio,
            "dims": dict(self.dims), "dimsFromCheckpoint": self.dims_from_checkpoint,
            "isDefaultSized": self.is_default_sized,
            "expectedCount": self.expected_count, "presentCount": self.present_count,
            "missing": self.missing[:12], "missingCount": len(self.missing),
            "unexpected": self.unexpected[:12], "unexpectedCount": len(self.unexpected),
            "shapeMismatch": self.shape_mismatch[:6], "shapeMismatchCount": len(self.shape_mismatch),
            "problems": self.problems,
        }


def _pick(tensors: Mapping[str, Any], key: str, axis: int) -> int | None:
    shape = _shape(tensors.get(key))
    if shape is None or len(shape) <= axis:
        return None
    return shape[axis]


def _max_index(tensors: Mapping[str, Any], pattern: re.Pattern[str]) -> int:
    best = -1
    for key in tensors:
        match = pattern.match(key)
        if match:
            best = max(best, int(match.group(1)))
    return best


def audit_h3_checkpoint(tensors: Mapping[str, Sequence[int] | None]) -> H3KeyAudit:
    """核对一份「键名 → 形状」清单**能不能装进** `H3FormTrunk` ✓（纯函数 ✓ 零依赖 ✓）。

    ``tensors`` 的形状可以是 list / tuple / None ✓（``None`` = 只核键名 ✗）。
    输入来源：``safetensors.inspect().tensors`` / ``gguf.inspect().tensors`` ✓。
    """
    audit = H3KeyAudit()
    names = set(tensors)

    # ── 层数与变体 ✓（都是检查点自己的事实 ✓）
    audit.depth = _max_index(tensors, _BLOCK_KEY) + 1
    audit.refiner_layers = _max_index(tensors, _REFINER_KEY) + 1
    if audit.depth < 1:
        audit.problems.append("一个 `blocks.N.*` 键都没有 ⇒ 推不出层数 ✗（H3 形态必须有 ✓）")
    if audit.refiner_layers < 1:
        audit.problems.append("没有 `token_refiner.blocks.N.*` ⇒ H3 必须带文本 refiner ✗（参考默认 2 层 ✓）")

    has_time = "time_embedder.proj_in.weight" in names
    has_curve = "adaln_t_table" in names
    if has_time and has_curve:
        audit.problems.append("`time_embedder.*` 与 `adaln_t_table` 同时存在 ✗（两种时间嵌入互斥 ✓）")
    elif has_time:
        audit.variant = "standard"
    elif has_curve:
        audit.variant = "curve"
    else:
        audit.problems.append("既没有 `time_embedder.proj_in.weight` 也没有 `adaln_t_table` ✗")
        audit.variant = "unknown"

    # ── 尺寸推导 ✓（从检查点自己读 ✓；推不出的维度回落出厂值并在 problems 里点名 ✗）
    dims = _default_dims()
    sources = {
        "hidden": ("blocks.0.norm1.weight", 0),
        "head_dim": ("blocks.0.attn.q_norm.weight", 0),
        "inner": ("blocks.0.attn.out_proj.weight", 1),
        "ffn": ("blocks.0.mlp.fc2.weight", 1),
        "t_dim": ("blocks.0.adaln_proj.linear.weight", 1),
        "text_dim": ("condition_proj.weight", 1),
        "video_dim": ("video_patch_proj.weight", 1),
        "audio_dim": ("audio_patch_proj.weight", 1),
    }
    missing_sources: list[str] = []
    for name, (key, axis) in sources.items():
        value = _pick(tensors, key, axis)
        if value is None:
            missing_sources.append(key)
        else:
            dims[name] = value
    if has_time:
        value = _pick(tensors, "time_embedder.proj_in.weight", 1)
        if value is not None:
            dims["time_in"] = value
    if missing_sources:
        audit.dims_from_checkpoint = False
        audit.problems.append(
            f"尺寸推导缺源（{len(missing_sources)} 个键推不出 ⇒ 对应维度回落 H3 出厂值 ✗）："
            f"{missing_sources[:6]}")
    if dims["head_dim"] and dims["inner"] % dims["head_dim"]:
        audit.problems.append(
            f"`out_proj` 列数 {dims['inner']} 不整除 `q_norm` 维 {dims['head_dim']} ✗ ⇒ 推不出头数 ✓")
    audit.dims = dims
    defaults = _default_dims()
    audit.is_default_sized = (dims == defaults and audit.depth == int(H3_TRUNK_DEFAULTS["layers"])
                              and audit.refiner_layers == int(H3_TRUNK_DEFAULTS["refiner_layers"]))

    # ── 期望键表 + 三类核对 ✓
    if audit.variant == "unknown":
        # 变体都推不出 ⇒ 只能做「有 time_embedder 的标准形态」核对（problems 已点名 ✗）
        expected = expected_h3_keys(dims, depth=max(audit.depth, 1),
                                    refiner_layers=max(audit.refiner_layers, 1), variant="standard")
    else:
        expected = expected_h3_keys(dims, depth=max(audit.depth, 1),
                                    refiner_layers=max(audit.refiner_layers, 1),
                                    variant=audit.variant)
    audit.expected_count = len(expected)
    audit.present_count = len(names)

    audit.missing = sorted(set(expected) - names)
    optional_like = {key for key in names
                     if any(pattern.match(key) for pattern in _OPTIONAL_PATTERNS)}
    audit.unexpected = sorted(names - set(expected) - optional_like)

    # 形状核对 ✓（普通键按期望表 ✗；自由形状键按结构关系 ✗ —— 见 _FREE_SHAPE_KEYS ✓）
    for key, want in expected.items():
        if key not in names or want is None:
            continue
        got = _shape(tensors[key])
        if got is not None and got != want:
            audit.shape_mismatch.append({"key": key, "expected": list(want), "got": list(got)})
    for key in _FREE_SHAPE_KEYS & names:
        got = _shape(tensors[key])
        if got is None:
            continue
        if key == "rope.inv_freq":
            if len(got) != 1:
                audit.problems.append(f"`rope.inv_freq` 应是 1 维（收到 {list(got)} ✗）")
        elif key == "adaln_t_table":
            if len(got) != 2 or got[1] != dims["t_dim"]:
                audit.problems.append(
                    f"`adaln_t_table` 应是 (grid, {dims['t_dim']})（收到 {list(got)} ✗）")
        elif key == H3_VIDEO_OUT_KEY:
            if len(got) != 2 or got[1] != dims["hidden"]:
                audit.problems.append(
                    f"`{key}` 应是 (banks×{dims['video_dim']}, {dims['hidden']})（收到 {list(got)} ✗）")
            else:
                try:
                    audit.head_banks_video = head_banks_from_shape(got, dims["video_dim"])
                except ValueError as err:
                    audit.problems.append(str(err))
        elif key.endswith("audio_out.weight"):
            if len(got) != 2 or got[1] != dims["hidden"]:
                audit.problems.append(
                    f"`{key}` 应是 (banks×{dims['audio_dim']}, {dims['hidden']})（收到 {list(got)} ✗）")
            else:
                try:
                    audit.head_banks_audio = head_banks_from_shape(got, dims["audio_dim"])
                except ValueError as err:
                    audit.problems.append(str(err))
    # 头的 bias 必须与头的行数一致 ✓（PDD 打包时 bias 也打包 ✓）
    for kind in ("video", "audio"):
        weight = _shape(tensors.get(f"final_layer.{kind}_out.weight"))
        bias = _shape(tensors.get(f"final_layer.{kind}_out.bias"))
        if weight is not None and bias is not None and bias != (weight[0],):
            audit.shape_mismatch.append(
                {"key": f"final_layer.{kind}_out.bias", "expected": [weight[0]], "got": list(bias)})
    return audit


@dataclass
class H3TrunkConfig:
    """从检查点**推导出的** `H3FormTrunk` 构造参数 ✓（`config` 可直接 `H3FormTrunk(**config)` ✓）。

    ⚠️ **每个字段都带来源** ✓（``sources``）—— 「权重里读的 ✓」与「不可推 ⇒ 用参考默认 ✗」
    **必须分开** ✗：混在一起看，默认值会被读成"检查点的事实" ✓✗（本仓在 `H3_TRUNK_DEFAULTS`
    上正是靠这条纪律避免"名字对、形状错" ✓）。
    """

    config: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    audit: H3KeyAudit = field(default_factory=H3KeyAudit)
    problems: list[str] = field(default_factory=list)

    @property
    def derived_count(self) -> int:
        """**真从权重读出来**的字段数 ✓（剩下的是不可推项 ✗ —— 报出来，免得被当全绿 ✓）。"""
        return sum(1 for value in self.sources.values() if value.startswith("权重"))

    @property
    def ok(self) -> bool:
        return not self.problems and self.audit.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "config": dict(self.config), "sources": dict(self.sources),
            "derivedCount": self.derived_count, "fieldCount": len(self.config),
            "problems": self.problems, "audit": self.audit.to_dict(),
        }


def infer_h3_trunk_config(tensors: Mapping[str, Sequence[int] | None], *,
                          patch_size: Sequence[int] | None = None) -> H3TrunkConfig:
    """从「键名 → 形状」**推出** `H3FormTrunk` 的构造参数 ✓（纯函数 ✓ 零依赖 ✓）。

    为什么必须推 ✗：检查点的尺寸**不在文件里**（safetensors 只有张量 ✓、GGUF 也一样 ✓）——
    出厂常量（`H3_TRUNK_DEFAULTS` ✓）只对**出厂那一份**成立 ✓；社区重导出 / 蒸馏版 / 自检的
    缩小版都不成立 ✗ ⇒ 拿常量硬装 = 「名字对、形状错」✓✗（`load_module_weights` 会报一堆
    missing ✓，而人容易读成"权重没下全" ✗）。

    ``patch_size`` **推不出来** ✗（视频 patch 尺寸在权重里没有形状痕迹 ✓）⇒ 由调用方显式给 ✓，
    不给则用参考默认 ``(1, 2, 2)`` ✓（并在 ``sources`` 里标注 ✗ —— 别让默认值冒充事实 ✓）。
    """
    audit = audit_h3_checkpoint(tensors)
    dims = audit.dims
    patch = tuple(int(value) for value in (patch_size or H3_TRUNK_DEFAULTS["patch_size"]))
    scale = math.prod(patch) or 1

    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    problems: list[str] = list(audit.problems)

    def put(key: str, value: Any, source: str) -> None:
        config[key] = value
        sources[key] = source

    put("hidden", dims["hidden"], "权重：`blocks.0.norm1.weight` 长度 ✓")
    put("layers", audit.depth, "权重：`blocks.N` 的最大 N+1 ✓")
    put("head_dim", dims["head_dim"], "权重：`blocks.0.attn.q_norm.weight` 长度 ✓")
    if dims["head_dim"] and dims["inner"] % dims["head_dim"] == 0:
        put("heads", dims["inner"] // dims["head_dim"],
            "权重：`attn.out_proj.weight` 列数 ÷ `q_norm` 长度 ✓")
    else:
        problems.append(f"推不出头数（inner {dims['inner']} ÷ head_dim {dims['head_dim']} 不是整数 ✗）")
        put("heads", int(H3_TRUNK_DEFAULTS["heads"]), "默认回落 ✗（推不出头数 ✓）")
    put("ffn", dims["ffn"], "权重：`blocks.0.mlp.fc2.weight` 列数 ✓")
    put("text_dim", dims["text_dim"], "权重：`condition_proj.weight` 列数 ✓")
    put("audio_latents_dim", dims["audio_dim"], "权重：`audio_patch_proj.weight` 列数 ✓")
    put("time_dim", dims["t_dim"], "权重：`blocks.0.adaln_proj.linear.weight` 列数 ✓")
    put("refiner_layers", audit.refiner_layers, "权重：`token_refiner.blocks.N` 的最大 N+1 ✓")

    # ── patch 尺寸与潜变量通道 ✓（`video_patch_proj` 列数 = latents_dim × pT·pH·pW ✓）
    if dims["video_dim"] % scale:
        problems.append(
            f"`video_patch_proj` 列数 {dims['video_dim']} 不是 patch 乘积 {scale} 的整数倍 ✗"
            f"（patch_size={patch} ⇒ 推不出 latents_dim ✓）")
        put("latents_dim", int(H3_TRUNK_DEFAULTS["latents_dim"]), "默认回落 ✗（推不出 ✓）")
    else:
        put("latents_dim", dims["video_dim"] // scale,
            f"权重：`video_patch_proj.weight` 列数 ÷ {scale}（patch {patch}）✓")
    put("patch_size", patch,
        "显式传入 ✓" if patch_size is not None
        else "参考默认 (1,2,2) ✗（**权重里没有 patch 事实** ⇒ 供方必须自述 ✓）")

    # ── rope 的 `inv_freq` 是**权重里的缓冲区** ✓ ⇒ 长度是真事实 ✓（不自己算 ✗）
    inv = _shape(tensors.get("rope.inv_freq"))
    if inv is not None and len(inv) == 1:
        put("inv_freq_len", int(inv[0]), "权重：`rope.inv_freq` 缓冲区长度 ✓")
    else:
        problems.append("`rope.inv_freq` 不在 / 不是 1 维 ✗ ⇒ inv_freq_len 推不出 ✓")
        put("inv_freq_len", int(H3_TRUNK_DEFAULTS["inv_freq_len"]), "默认回落 ✗")

    # ── 时间嵌入 ✓（standard 有 `time_embedder`；curve 检查点没有 ⇒ 回落 ✓ 但**点名** ✗）
    time_in = _pick(tensors, "time_embedder.proj_in.weight", 1)
    time_hidden = _pick(tensors, "time_embedder.proj_in.weight", 0)
    if time_hidden is None:                      # `proj_out` 列数同样等于 time_hidden ✓（两条来源 ✓）
        time_hidden = _pick(tensors, "time_embedder.proj_out.weight", 1)
    if time_in is not None:
        put("time_input_dim", time_in, "权重：`time_embedder.proj_in.weight` 列数 ✓")
    else:
        put("time_input_dim", int(H3_TRUNK_DEFAULTS["time_input_dim"]),
            "默认回落 ✗（curve 检查点无 `time_embedder` ✓）")
    if time_hidden is not None:
        put("time_hidden", time_hidden, "权重：`time_embedder.proj_in.weight` 行数 ✓")
    else:
        put("time_hidden", config["hidden"], "回落 hidden ✗（无 `time_embedder` 可读 ✓）")

    # ── adaLN 路数：行数 = 6 × hidden × modalities ✓ ⇒ modalities 可推 ✓（H3 = 3 ✓）
    rows = _pick(tensors, "blocks.0.adaln_proj.linear.weight", 0)
    span = 6 * dims["hidden"]
    if rows and span and rows % span == 0:
        put("modalities", rows // span, "权重：adaLN 行数 ÷ (6×hidden) ✓")
    else:
        problems.append(f"推不出 modalities（adaLN 行数 {rows} ÷ (6×hidden {span}) 不是整数 ✗）")
        put("modalities", 3, "参考默认 3 ✗（推不出 ✓）")

    # ── PDD 头库：两个头**必须同库** ✓（主干只有一个 `head_banks` ✓ —— 参考实现也是两头同库 ✓）
    put("head_banks", max(audit.head_banks_video, audit.head_banks_audio),
        "权重：`final_layer.video_out`/`audio_out` 行数 ÷ 单头宽度 ✓")
    if audit.head_banks_video != audit.head_banks_audio:
        problems.append(
            f"两处头库不一致（video {audit.head_banks_video} / audio {audit.head_banks_audio} ✗）"
            f"⇒ 主干只有一个 `head_banks` ✓（参考实现两头同库 ✓）")

    # ── eps 类：**形状里验不出来** ✗ ⇒ 参考默认 ✓ 但必须标注（别说成"权重事实" ✗）
    for key in ("norm_eps", "qk_norm_eps", "final_norm_eps"):
        put(key, H3_TRUNK_DEFAULTS[key], "参考默认 1e-5 ✗（**形状验不出 eps** ⇒ 不可推 ✓）")
    return H3TrunkConfig(config=config, sources=sources, audit=audit, problems=problems)
