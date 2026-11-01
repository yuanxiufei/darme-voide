"""按**权重键名**判族 + 读拓扑 —— 「所有模型都能跑」的第一层。

盘上十来族（Krea2 / Z-Image / Flux.1 / Qwen-Image / Wan2.2 / SEEDVR2 / Music3 / H3 / SDXL）
的数学骨架是同一套：``pre-norm → 注意力 → 残差 → pre-norm → MLP → 残差``，条件由 adaLN
调制或交叉注意力注入。差别只在几处开关：qkv 分几块、qk 是否归一、MLP 是 SwiGLU 还是 GELU、
条件是 adaLN 还是 cross-attn、单流还是双流。

⇒ 不必一族一份前向：先判族、把开关读出来，再交给通用骨架。

**纪律**：只报告权重里写着的事实（键名 + 形状）。读不到的（头数 / eps / 采样 shift /
latent 通道）一律进 ``unknown``，**不猜** —— 猜错不会报错，只会让出片与提示词无关。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import h3_form, quant
from .safetensors import SafetensorsInfo, TensorEntry, TensorReader
from .safetensors import inspect as _inspect_file

__all__ = ["COMPANION_SUFFIXES", "FAMILY_KEYS", "FamilySpec", "detect", "detect_file",
           "detect_from_keys", "readout"]

#: 量化**伴生**键后缀 —— 不是网络结构，判族时要当噪声滤掉
COMPANION_SUFFIXES: tuple[str, ...] = (
    ".weight_scale", ".comfy_quant", ".scale_input", ".scale_weight",
    ".input_scale", ".weight_scale_2",
)

#: 已知族（认不出就是 ``None``，不硬塞）
FAMILY_KEYS: tuple[str, ...] = (
    "h3", "flux", "qwen_image", "qwen_image21", "zimage", "krea2", "wan", "seedvr2",
    "music3", "sdxl",
)

#: 同一份权重会被不同工具包成不同前缀（实测 Krea2 的 int4 版多一层
#: ``model.diffusion_model.``，int8 版没有）⇒ 判族前先归一
_COMMON_PREFIXES: tuple[str, ...] = ("model.diffusion_model.", "diffusion_model.", "model.")

def _logical_keys(keys: list[str] | tuple[str, ...] | set[str]) -> tuple[set[str], str]:
    """滤掉量化伴生键 + 去公共前缀 ⇒ ``(逻辑键集, 被去掉的前缀)``。"""
    cleaned = [k for k in keys if not any(k.endswith(s) for s in COMPANION_SUFFIXES)]
    for prefix in _COMMON_PREFIXES:
        hit = sum(1 for k in cleaned if k.startswith(prefix))
        # 按**比例**判（个别键叫 ``model.xxx`` 不代表整份套了前缀）
        if cleaned and hit >= max(3, int(len(cleaned) * 0.6)):
            return {k[len(prefix):] for k in cleaned}, prefix
    return set(cleaned), ""


def _logical_tensors(info: SafetensorsInfo, stripped: str) -> dict[str, TensorEntry]:
    """逻辑键名 → 张量元信息（**读形状必须走这张表**）。

    为什么不能直接拿 ``info.tensors`` 查：判族用的是剥掉公共前缀后的键名
    （实测 Krea2 的 int4 版多一层 ``model.diffusion_model.``），拿逻辑名去查
    原始表永远查不到 ⇒ ``hidden=None``、判成"读不到形状"。
    """
    if not stripped:
        return dict(info.tensors)
    return {k[len(stripped):]: v for k, v in info.tensors.items() if k.startswith(stripped)}


@dataclass(frozen=True)
class FamilySpec:
    """一族的拓扑结论（``hidden is None`` ⇒ 形状没读到，别当 0 用）。"""

    key: str
    label: str
    block_prefix: str
    depth: int
    hidden: int | None = None
    #: ``rms_scale`` / ``layer_weight`` / ``layer_gamma_beta``
    norm: str = ""
    #: ``separate_qkv`` / ``fused_qkv`` / ``split_qkv`` / ``mmdit_split_qkv``
    attn: str = ""
    #: 注意力输出后有无门控（Krea2 的 ``attn.gate``）
    attn_gate: bool = False
    qk_norm: bool = False
    #: ``swiglu_gate_up_down`` / ``swiglu_w1w2w3`` / ``gelu_linear``
    mlp: str = ""
    #: ``adaln_lin`` / ``adaln_seq`` / ``cross_attn`` / ``dual_ada`` / ``mmdit_joint`` / ``none``
    cond: str = ""
    dual_stream: bool = False
    text_dim: int | None = None
    mlp_hidden: int | None = None
    stripped_prefix: str = ""
    #: 权重里读不到、建前向时必须显式给的参数
    unknown: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    @property
    def buildable(self) -> bool:
        """拓扑读全了没（不判「前向实现了没」）。"""
        return self.hidden is not None and self.depth > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "blockPrefix": self.block_prefix,
            "depth": self.depth, "hidden": self.hidden, "norm": self.norm,
            "attn": self.attn, "attnGate": self.attn_gate, "qkNorm": self.qk_norm,
            "mlp": self.mlp, "mlpHidden": self.mlp_hidden, "cond": self.cond,
            "dualStream": self.dual_stream, "textDim": self.text_dim,
            "strippedPrefix": self.stripped_prefix, "buildable": self.buildable,
            "unknown": list(self.unknown), "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class _Rule:
    """一条判族规则（``sentinels`` **全中**才算，不是中一个就算）。"""

    key: str
    label: str
    sentinels: tuple[str, ...]
    block_prefix: str
    norm: str
    attn: str
    mlp: str
    cond: str
    attn_gate: bool = False
    qk_norm: bool = False
    dual_stream: bool = False
    #: 从哪个键的 ``shape[1]`` 读 ``hidden``（⚠️ 融合的 ``qkv`` 也是 **[3h, h]**，
    #: ``shape[1]`` 就是 h ✓ —— 别拿 out 去除 3，那是把 h 又除了一遍 ✗）
    hidden_from: str | None = None
    mlp_from: str | None = None
    text_from: str | None = None
    unknown: tuple[str, ...] = ()


#: 判族表 —— 按特异性从高到低排，第一条命中的就是答案。
#: 每条开关都是 2026-10-30 从盘上真文件读出来的（不是照论文抄的）。
_RULES: tuple[_Rule, ...] = (
    _Rule(
        key="flux", label="Flux.1（DoubleStream + SingleStream）",
        sentinels=("double_blocks.0.img_attn.qkv.weight",
                   "double_blocks.0.img_mod.lin.weight",
                   "single_blocks.0.linear1.weight"),
        block_prefix="double_blocks.", norm="layer_weight", attn="fused_qkv",
        mlp="gelu_linear", cond="adaln_lin", qk_norm=True, dual_stream=True,
        hidden_from="double_blocks.0.img_attn.qkv.weight",
        mlp_from="double_blocks.0.img_mlp.0.weight", text_from="txt_in.weight",
        unknown=("heads（qkv 融合，头数权重里没有，需显式给）",
                 "single_blocks 层数要与 double_blocks 分开数（本 spec 只报 double）"),
    ),
    _Rule(
        key="qwen_image", label="Qwen-Image / 2512 / Edit（MMDiT 联合注意力）",
        sentinels=("transformer_blocks.0.attn.to_q.weight",
                   "transformer_blocks.0.attn.add_q_proj.weight",
                   "transformer_blocks.0.attn.to_add_out.weight"),
        block_prefix="transformer_blocks.", norm="layer_weight", attn="mmdit_split_qkv",
        mlp="gelu_linear", cond="mmdit_joint", qk_norm=True, dual_stream=True,
        hidden_from="transformer_blocks.0.attn.to_q.weight",
        mlp_from="transformer_blocks.0.img_mlp.net.0.proj.weight",
        text_from="txt_in.weight",
        unknown=("三轴 rope 分维（16/56/56）与 patch 2×2 在权重里没有，取值来自参考实现",
                 "参考图口径由 `__index_timestep_zero__` 哨兵自述，没有该键的按 `index` 档"),
    ),
    _Rule(
        # ⚠️ 2.1 与 2512 **结构不同**（实测 2026-10-30）：2.1 没有 ``add_q/add_k/add_v``
        #    （不是 MMDiT 双流），文本走 ``txt_in.in_layer/out_layer`` 投影后与图像**同流**，
        #    调制是全局 ``modulation.1``；MLP 换成 SwiGLU（``img_mlp.gate_up``）✓
        key="qwen_image21", label="Qwen-Image 2.1（单流 + SwiGLU）",
        sentinels=("transformer_blocks.0.attn.to_q.weight",
                   "transformer_blocks.0.img_mlp.gate_up.weight",
                   "txt_in.in_layer.weight", "modulation.1.weight"),
        block_prefix="transformer_blocks.", norm="layer_weight", attn="separate_qkv",
        mlp="swiglu_gate_up_down", cond="adaln_lin", qk_norm=True, dual_stream=False,
        hidden_from="transformer_blocks.0.attn.to_q.weight",
        mlp_from="transformer_blocks.0.img_mlp.gate_up.weight",
        unknown=("heads（to_q 是 [hidden,hidden]，权重里没有）",
                 "文本原始宽度（``txt_in.in_layer`` 是 [4096,4096] 方阵 ⇒ 读不到，需从 TE 配置给）",
                 "``modulation.1`` 输出 16384 = 4×4096 的分组语义"),
    ),
    _Rule(
        key="zimage", label="Z-Image（NextDiT：refiner + 融合层）",
        sentinels=("x_embedder.weight", "cap_embedder.0.weight",
                   "layers.0.attention.qkv.weight",
                   "layers.0.adaLN_modulation.0.weight"),
        block_prefix="layers.", norm="layer_weight", attn="fused_qkv",
        mlp="swiglu_w1w2w3", cond="adaln_seq", qk_norm=True, dual_stream=False,
        hidden_from="layers.0.attention.qkv.weight",
        mlp_from="layers.0.feed_forward.w1.weight",
        unknown=("heads（qkv 融合，权重里没有）",
                 "noise_refiner / context_refiner 层数与 layers 分开数（本 spec 只报 layers）",
                 "cap_embedder 的 pad token 语义（建前向时照抄）"),
    ),
    _Rule(
        key="krea2", label="Krea2（S3-DiT：Lumina-2 同构，单流 + attn 门控）",
        # ⚠️ ``mod.lin`` 是**裸参数**（键名就到 ``.lin`` 为止，没有 ``.weight`` ✓ 实测）
        sentinels=("blocks.0.attn.wq.weight", "blocks.0.mod.lin",
                   "blocks.0.attn.qknorm.qnorm.scale", "blocks.0.mlp.gate.weight"),
        block_prefix="blocks.", norm="rms_scale", attn="separate_qkv",
        mlp="swiglu_gate_up_down", cond="adaln_lin", attn_gate=True, qk_norm=True,
        hidden_from="blocks.0.attn.wq.weight", mlp_from="blocks.0.mlp.up.weight",
        unknown=("heads（wq 是 [hidden,hidden]，权重里没有）",
                 "txtfusion.* 的融合方案（layerwise/refiner，建前向时照键名读）",
                 "timestep 编码（t_embedder 一族，建前向时再读）"),
    ),
    _Rule(
        key="wan", label="Wan2.2 / Wan-Animate（self + cross 注意力 DiT）",
        sentinels=("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.q.weight",
                   "blocks.0.ffn.0.weight", "text_embedding.0.weight"),
        block_prefix="blocks.", norm="layer_weight", attn="split_qkv", mlp="gelu_linear",
        cond="cross_attn", qk_norm=True, hidden_from="blocks.0.self_attn.q.weight",
        mlp_from="blocks.0.ffn.0.weight", text_from="text_embedding.0.weight",
        unknown=("heads（q 是 [hidden,hidden]，权重里没有）",
                 "3D patch（patch_embedding 是 Conv3d，帧/高/宽三轴参数权重里没有）",
                 "Wan-Animate 多出的 k_img/v_img 参考图通路（本 spec 不区分）"),
    ),
    _Rule(
        key="seedvr2", label="SEEDVR2（双流 txt/vid + 各自 ada 调制）",
        sentinels=("blocks.0.ada.vid.attn_shift", "blocks.0.ada.txt.attn_shift",
                   "blocks.0.attn.norm_q.vid.weight", "vid_in.proj.weight"),
        block_prefix="blocks.", norm="layer_weight", attn="split_qkv", mlp="gelu_linear",
        cond="dual_ada", qk_norm=True, dual_stream=True,
        # ⚠️ ``norm_q.vid`` 是 **per-head** 宽度（实测 [128]，不是 hidden ✗）⇒ 用 ``proj_out`` 读
        hidden_from="blocks.0.attn.proj_out.vid.weight",
        text_from="txt_in.proj.weight",
        unknown=("heads（norm 是 [hidden]，但头数权重里没有）",
                 "上采样倍率（3b/7b 与 sharp 变体不同，权重里没有）"),
    ),
    _Rule(
        key="music3", label="MiniMax Music3（LLM 风格 DiT）",
        sentinels=("diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight",
                   "diffusion_transformer.transformer.layers.0.pre_norm.gamma"),
        block_prefix="diffusion_transformer.transformer.layers.",
        norm="layer_gamma_beta", attn="fused_qkv", mlp="gelu_linear", cond="none",
        hidden_from="diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight",
        mlp_from="diffusion_transformer.transformer.layers.0.ff.ff.0.proj.weight",
        unknown=("heads（qkv 融合，权重里没有）",
                 "音频 latent 的通道/帧率语义（权重里没有）"),
    ),
)

#: SDXL 是 UNet（不是 DiT）—— 已有 ``sdxl.py`` 专用实现，这里只负责认出来
_SDXL_SENTINELS: tuple[str, ...] = ("input_blocks.0.0.weight", "middle_block.1.proj_in.weight")


def _depth_of(keys: set[str], prefix: str) -> int:
    """数 ``<prefix>N.`` 里出现过多少个**不同**的 ``N``（缺层要看得出来，不取 max+1）。"""
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.")
    return len({int(m.group(1)) for k in keys if (m := pattern.match(k))})


def _stem_of(name: str) -> str:
    """``…wq.weight`` ⇒ ``…wq``（``comfy_quant`` 的说明按**模块名**登记 ✓）。"""
    return name[: -len(".weight")] if name.endswith(".weight") else name


def _packed_stems(specs: Mapping[str, Mapping[str, Any]] | None,
                  stripped: str) -> frozenset[str]:
    """自称 **4bit 打包**的层名（**逻辑键名**）✓；没有说明 ⇒ 空集（**不按形状猜** ✗）。"""
    if specs is None:
        return frozenset()
    stems = quant.packed_weight_stems_of(specs)
    if not stripped:
        return stems
    return frozenset(s[len(stripped):] for s in stems if s.startswith(stripped))


def _logical_dims(entry: TensorEntry, packed: frozenset[str], name: str) -> tuple[int, ...]:
    """**逻辑**形状 ✓ —— 4bit 打包的列数只有声明值的一半 ✓（不还原就会把 hidden 读成半值 ✗）。"""
    return quant.logical_weight_shape(entry.shape, packed=_stem_of(name) in packed)


def _in_features(entries: Mapping[str, TensorEntry], name: str,
                 packed: frozenset[str] = frozenset()) -> int | None:
    """输入宽度 ✓（2D 权重读 ``shape[1]``；4D conv 也读 ``shape[1]`` = 输入通道 ✓）。"""
    entry = entries.get(name)
    if entry is None or len(entry.shape) < 2:
        return None
    return int(_logical_dims(entry, packed, name)[1])


def _out_features(entries: Mapping[str, TensorEntry], name: str,
                  packed: frozenset[str] = frozenset()) -> int | None:
    entry = entries.get(name)
    if entry is None or len(entry.shape) < 2:
        return None
    return int(_logical_dims(entry, packed, name)[0])


def _hidden_of(entries: Mapping[str, TensorEntry], rule: _Rule,
               packed: frozenset[str]) -> int | None:
    if not rule.hidden_from:
        return None
    return _in_features(entries, rule.hidden_from, packed)


def _assemble(entries: Mapping[str, TensorEntry], rule: _Rule, keys: set[str],
              stripped: str, packed: frozenset[str],
              extra_unknown: tuple[str, ...] = (),
              extra_evidence: tuple[str, ...] = ()) -> FamilySpec:
    evidence = [f"命中：`{sentinel}`" for sentinel in rule.sentinels]
    if stripped:
        evidence.append(f"剥离公共前缀 `{stripped}`")
    if packed:
        evidence.append(f"{len(packed)} 层自称 4bit 打包 ⇒ 列数已按逻辑值还原")
    evidence.extend(extra_evidence)
    return FamilySpec(
        key=rule.key, label=rule.label, block_prefix=rule.block_prefix,
        depth=_depth_of(keys, rule.block_prefix), hidden=_hidden_of(entries, rule, packed),
        norm=rule.norm, attn=rule.attn, attn_gate=rule.attn_gate, qk_norm=rule.qk_norm,
        mlp=rule.mlp, cond=rule.cond, dual_stream=rule.dual_stream,
        text_dim=_in_features(entries, rule.text_from, packed) if rule.text_from else None,
        mlp_hidden=_out_features(entries, rule.mlp_from, packed) if rule.mlp_from else None,
        stripped_prefix=stripped, unknown=rule.unknown + extra_unknown,
        evidence=tuple(evidence),
    )


def _views(info: SafetensorsInfo, raw: set[str]) -> list[tuple[set[str], str, dict[str, TensorEntry]]]:
    """候选视图 ``[(逻辑键集, 被剥掉的前缀, 逻辑键→张量)]``。

    为什么不止一个视图：实测有两种包装 ——
    ① 整份多一层前缀（Krea2 int4 版的 ``model.diffusion_model.`` ✓，占绝对多数 ⇒ 按比例剥 ✓）；
    ② **all-in-one 大包**（``krea2TurboNSFWAIO_v10`` ✓：DiT 只占一部分，另有
       ``text_encoders.*`` 与 ``vae.*`` ⇒ 按比例剥**永远不成立** ✗）。
    ② 不能靠比例判 ⇒ 除主视图外，每个**出现过的**前缀都再试一遍（命中谁算谁 ✓）。
    """
    out: list[tuple[set[str], str, dict[str, TensorEntry]]] = []
    keys, stripped = _logical_keys(raw)
    out.append((keys, stripped, _logical_tensors(info, stripped)))
    for prefix in _COMMON_PREFIXES:
        if prefix == stripped or not any(k.startswith(prefix) for k in raw):
            continue
        inner = {k[len(prefix):] for k in raw if k.startswith(prefix)}
        inner = {k for k in inner if not any(k.endswith(s) for s in COMPANION_SUFFIXES)}
        out.append((inner, prefix, _logical_tensors(info, prefix)))
    return out


#: all-in-one 大包里顺带装着的部件（判族时不算 DiT，但值得报出来 ✓）
_EXTRA_PARTS: tuple[tuple[str, str], ...] = (
    ("text_encoders.", "内含 text_encoder"), ("vae.", "内含 VAE"),
    ("audio_encoder.", "内含 audio_encoder"),
)


def _extra_parts(raw: set[str]) -> tuple[str, ...]:
    return tuple(note for prefix, note in _EXTRA_PARTS if any(k.startswith(prefix) for k in raw))


def _int4_unverified(entries: Mapping[str, TensorEntry], specs: Any) -> tuple[str, ...]:
    """没读 ``comfy_quant`` 说明时，int8 存储的 2D 权重**可能**是 4bit 打包 ⇒ 必须说出来 ✓。

    为什么不当场 ×2：打包后的 ``(6144, 3072)`` 与一行一 scale 的 int8 **逐字相同** ✓
    （``quant.INT4_PACK_TOKENS`` 那段注释记的正是这个假绿）⇒ 按形状猜会把正常 int8 也读错 ✗。
    """
    if specs is not None:
        return ()
    if not any(k.endswith(".comfy_quant") for k in entries):
        return ()
    if not any(quant.is_int4_storage(e.dtype) and len(e.shape) == 2 for e in entries.values()):
        return ()
    return ("该文件自带 `comfy_quant` 说明但未读 ⇒ 量化层是否 4bit 打包未核（hidden 可能只是逻辑值的一半，"
            "改用 `detect_file` 走权威说明）",)


def detect(info: SafetensorsInfo, *,
           specs: Mapping[str, Mapping[str, Any]] | None = None) -> FamilySpec | None:
    """从一次 ``safetensors.inspect`` 的结论判族 ⇒ :class:`FamilySpec`；认不出 ⇒ ``None``。

    ``specs`` ＝ :func:`weights.quant_specs_of_reader` 的结论（可选 ✓）。给了才判得了
    「4bit 打包」（列数要 ×2 ✓）；不给就只在 ``unknown`` 里声明这一点 ✓（见
    :func:`_int4_unverified`）—— 形状本身分不出 int8 与打包 4bit ✗。
    """
    if not info.tensors:
        return None
    raw = set(info.tensors)
    views = _views(info, raw)
    primary_keys, _, _ = views[0]

    # H3 有自己的一套判据（``h3_form`` 认的是"分块打包"的 H3 形态），先问它
    if h3_form.looks_like_h3_form(primary_keys) or h3_form.looks_like_h3_form(raw):
        return FamilySpec(
            key="h3", label="MiniMax H3（分块打包 DiT）", block_prefix="", depth=0,
            evidence=("``h3_form.looks_like_h3_form`` 命中",),
            unknown=("详见 ``h3_form`` 与 ``dit.py``（它是已知实现，不走通用骨架）",),
        )

    parts = _extra_parts(raw)
    for rule in _RULES:
        for keys, stripped, entries in views:
            if not all(sentinel in keys for sentinel in rule.sentinels):
                continue
            packed = _packed_stems(specs, stripped)
            return _assemble(entries, rule, keys, stripped, packed,
                             extra_unknown=_int4_unverified(entries, specs),
                             extra_evidence=parts)

    for keys, stripped, entries in views:
        if not all(sentinel in keys for sentinel in _SDXL_SENTINELS):
            continue
        return FamilySpec(
            key="sdxl", label="SDXL（UNet）", block_prefix="input_blocks.", depth=0,
            hidden=_in_features(entries, "input_blocks.0.0.weight"),
            evidence=tuple(f"命中：`{s}`" for s in _SDXL_SENTINELS) + parts,
            unknown=("UNet 不走 DiT 骨架，走 ``sdxl.py`` 专用实现",),
        )
    return None


def detect_file(path: str | Path, *, with_specs: bool = True) -> FamilySpec | None:
    """:func:`detect` 的**文件级**入口 ✓ —— 顺手把 ``comfy_quant`` 说明读掉 ✓。

    为什么要多这一步：4bit 打包的列数是**声明**出来的 ✗（形状与 int8 逐字相同 ✓），
    只读头部永远判不出来 ⇒ 判出来的 ``hidden`` 会只有真值的一半 ✓✗，
    而「少一半」不报错、只是前向维度不对（或更糟：对上另一族）✗。
    """
    info = _inspect_file(str(path))
    if not info.tensors:
        return None
    specs: dict[str, dict[str, Any]] | None = None
    if with_specs:
        from .weights import quant_specs_of_reader  # 惰性：weights 依赖 torch 生态，判族不该被它拖住

        try:
            with TensorReader(path) as reader:
                specs = quant_specs_of_reader(reader)
        except Exception:  # noqa: BLE001 — 读说明失败不该让判族崩掉 ✓（会退化成 unknown 声明）
            specs = None
    return detect(info, specs=specs)


def detect_from_keys(keys: list[str] | tuple[str, ...] | set[str]) -> str | None:
    """只凭**键名**判族（没有形状时的降级口径）⇒ 族 key 或 ``None``。

    与 :func:`detect` 的区别：这里读不到 ``hidden``/``depth``，只回答"是哪一族"。
    用途：在只有头部键名、还没读形状的地方（如快速分诊）先分流。
    """
    logical, _ = _logical_keys(keys)
    if h3_form.looks_like_h3_form(logical) or h3_form.looks_like_h3_form(keys):
        return "h3"
    for rule in _RULES:
        if all(sentinel in logical for sentinel in rule.sentinels):
            return rule.key
    if all(sentinel in logical for sentinel in _SDXL_SENTINELS):
        return "sdxl"
    return None


def readout(spec: FamilySpec | None) -> str:
    """一句话人话（给 CLI/readiness 用）。"""
    if spec is None:
        return "族：**认不出**（头部键名不匹配任何已知族 ⇒ 不猜）"
    if spec.key == "h3":
        return "族：MiniMax H3（已知实现，走 `dit.py`）"
    if spec.key == "sdxl":
        return f"族：SDXL（UNet，`sdxl.py` 专用实现；in={spec.hidden}）"
    return (f"族：{spec.label} ⇒ 层数={spec.depth}　hidden={spec.hidden}　"
            f"注意力={spec.attn}{'+qknorm' if spec.qk_norm else ''}"
            f"{'+gate' if spec.attn_gate else ''}　MLP={spec.mlp}({spec.mlp_hidden})　"
            f"条件={spec.cond}{'/双流' if spec.dual_stream else '/单流'}")
