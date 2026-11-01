"""Qwen-Image **2.1** 的**文本塔** = Qwen3.5-9B **混合线性注意力**语言塔（24 层 Gated DeltaNet ＋ 8 层 GatedAttention ✓）。

事实来源（只参考行为 ✓ 实现自己写 ✓）：

* ``comfy/text_encoders/qwen35.py`` —— ``Qwen35Transformer`` / ``Qwen35TransformerBlock`` /
  ``GatedDeltaNet`` / ``GatedAttention`` / ``torch_chunk_gated_delta_rule`` ✓；
* ``QWEN35_MODELS["qwen35_9b"]`` —— ``hidden=4096`` / ``ffn=12288`` / ``depth=32`` / ``heads=16`` /
  ``kv_heads=4`` / ``head_dim=256`` / ``linear_head=16×128 + value_head=32×128`` / ``conv_kernel=4`` /
  ``partial_rotary_factor=0.25`` / ``rms_norm_eps=1e-6`` / ``rms_norm_add=True`` / ``rope_theta=1e7`` ✓；
* ``Qwen35ImageTokenizer`` —— 提示词模板与 ``<think>`` 尾段 ✓；``llama.Llama2_`` —— 取层用负索引、**不过** final norm ✓。

⚠️ 六条「装得上、算得出、但结果与提示词无关」的坑 ✓✗：

1. **RMSNorm 是 ``(1 + weight)`` 口径** ✓（``rms_norm_add=True`` ✓）。直接拿 ``weight`` 用，每层差一个缩放，
   32 层叠完 hidden 偏到另一个数量级 ✓✗。唯一例外是**线性注意力的门控 norm** ✓（``RMSNormGated`` 没传
   ``add`` ⇒ 那一格是朴素 ``weight`` ✓）—— 同一个文件里**两种口径并存** ✓✗。
2. **``q_proj`` 输出是 ``2×inner``** 且按**头交错**排：``view(B,L,heads,2*head_dim)`` 再前后劈 ✓ ——
   用 ``chunk(2)`` 会得到「前一半头全是 q、后一半头全是 gate」，**不报错**只是门控错了头 ✓✗。
3. **partial RoPE**：只转前 ``head_dim*0.25``（256 ⇒ 64）维 ✓，其余原样透传 ✓；且是 **split-half**
   （第 ``i`` 维配第 ``i+32`` 维 ✓），不是相邻配对 ✗。
4. **因果卷积左补后必须切尾**（``[:, :, :seq_len]`` ✓）：不切等于整体右移 3 格 ⇒ 每个 token 看到未来 ✓✗。
5. **``layer_idx = -2``** ⇒ 0-based 第 **30** 层之后 ✓、**不过** final norm ✓。
6. **``<|im_start|> = 248045``** ✓（**不是** Qwen2 的 151644 ✗ —— 照 Qwen2 拼就是另外几个字符 ✓✗）。

⚠️ 与 :mod:`.llm` 的分工 ✗：算子（RMSNorm / RoPE / SwiGLU）**只有一份** ✓ 从 :func:`llm.primitives` 取 ✓；
本模块只写这一族特有的两块（chunked delta rule ✓、带输出门的完整注意力 ✓）与**拼壳** ✓。

⚠️ **不建** ``lm_head`` ✗（与 ``embed_tokens`` 同形 ✓ 2 GB ✓ 文本→图用不到 ✓）与 ``model.visual.*`` ✗（549 键 ✓ 同一文件里 ✓）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "IM_END_ID", "IM_START_ID", "PAD_TOKEN_ID", "QWEN35_TEMPLATE", "THINK_CLOSE_ID", "THINK_OPEN_ID",
    "Qwen35Shape", "Qwen35TeRuntime", "ShapeError",
    "build_from_shapes", "expected_keys", "infer_shape", "tokens_of",
]

#: 填空 token ✓（``Qwen35Tokenizer.pad_token`` ✓）
PAD_TOKEN_ID = 248044

#: ``<|im_start|>`` ✓（**Qwen2 是 151644** ✗ —— 这一族把特殊 token 重排过 ✓ 2026-10-31 用 comfy 真跑的分词器核出 ✓）
IM_START_ID = 248045

#: ``<|im_end|>`` ✓
IM_END_ID = 248046

#: ``<think>`` / ``</think>`` ✓（``thinking=False`` 时**必定**追加这一段 ✓）
THINK_OPEN_ID = 248068
THINK_CLOSE_ID = 248069

#: 提示词模板 ✓（= ``Qwen35ImageTokenizer`` 拼出的原样字符串 ✓）。
#: ⚠️ ``{}`` 与 ``<|im_end|>`` 之间**没有**换行 ✓（照抄 Qwen2 那套会多一个 198 ✓✗）；
#: 尾段 ``<think>\\n</think>\\n`` 是**关闭思考**的声明 ✓ —— 去掉它模型会先"想"一段，语义就偏 ✓✗。
QWEN35_TEMPLATE = ("<|im_start|>user\n{}<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n</think>\n")

#: ``partial_rotary_factor`` ✓（权重形状里没有它 ✓ 只能按发布配置写死 ✓）
PARTIAL_ROTARY_FACTOR = 0.25

#: 每层**都有**的公共后缀 ✓（缺一个 = 权重没下全 ✗ 或不是这一族 ✗）
_COMMON_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

#: 线性注意力层（Gated DeltaNet ✓）独有的后缀 ✓
_LINEAR_SUFFIXES: tuple[str, ...] = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
)

#: 完整注意力层（GatedAttention ✓）独有的后缀 ✓
_FULL_SUFFIXES: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


class ShapeError(ValueError):
    """权重形状推不出可装的结构 ✓（**明确报错** ✗ 不拿出厂结构硬装 ✓）。"""


@dataclass(frozen=True)
class Qwen35Shape:
    """Qwen3.5 文本塔的**全部**结构参数 ✓（除 ``rms_eps``/``rope_theta``/``partial`` 外都由形状读出 ✓）。"""

    vocab: int
    hidden: int
    depth: int
    ffn: int
    # ---- 完整注意力（GatedAttention）----
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    # ---- 线性注意力（Gated DeltaNet）----
    linear_key_heads: int
    linear_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int
    # ---- 层型表（``linear_attention`` / ``full_attention`` ✓ 逐层不同 ✓）----
    layer_types: tuple[str, ...]
    rms_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    #: ``rms_norm_add`` ✓ —— 层 norm / QK-norm **是** ``(1+w)`` ✓；门控 norm **不是** ✓
    norm_add: bool = True
    #: 取哪一层（负索引 ✓）：``-2`` = 第 30 层（共 32 层）**之后**的 hidden ✓
    layer_idx: int = -2

    @property
    def key_dim(self) -> int:
        return self.linear_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        """因果卷积的通道数 ✓ = ``2*key_dim + value_dim`` ✓（q/k/v 挤在**一条** depthwise 卷积里 ✓）。"""
        return 2 * self.key_dim + self.value_dim

    @property
    def inner(self) -> int:
        return self.heads * self.head_dim

    def target_layer(self) -> int:
        """把 ``layer_idx``（可负 ✓）折成 **0-based 层号** ✓；越界 ⇒ :class:`ShapeError` ✗。"""
        index = self.layer_idx if self.layer_idx >= 0 else self.depth + self.layer_idx
        if not 0 <= index < self.depth:
            raise ShapeError(
                f"layer_idx={self.layer_idx} 在这 {self.depth} 层上越界 ✗"
                f"（折算成 {index} ⇒ 必须在 0..{self.depth - 1} ✓）")
        return index


#: 同一文件里**不属于**语言塔的键前缀 ✓（视觉塔 ✓ / 输出头 ✓）
_FOREIGN_PREFIXES: tuple[str, ...] = ("model.visual.", "visual.", "lm_head.", "model.lm_head.")


def _is_foreign(name: str) -> bool:
    return name.startswith(_FOREIGN_PREFIXES)


def _language_shapes(shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """剥掉 ``model.language_model.`` 前缀 ✓ 并**丢掉**视觉塔 / ``lm_head`` ✓ ⇒ 只剩语言塔的键 ✓。

    ⚠️ 盘上那份 ``..._pe_t2i.int8_convrot.safetensors`` 语言塔与视觉塔**同文件** ✓
    （828 ＋ 549 键）✓；不剥前缀 ⇒ 一个 ``model.layers.N.*`` 都匹配不到 ✓ 会误报「不是这一族」✗。
    ⚠️ 视觉塔（``model.visual.*`` ✓ 549 键 ✓）与 ``lm_head`` 必须**真的丢掉** ✗ ——
    留着它们既白占 2 GB 显存 ✓，又会让加载器报出几百条「多余键」把真问题淹掉 ✓✗。
    """
    plain: dict[str, tuple[int, ...]] = {}
    for name, value in shapes.items():
        for prefix in ("model.language_model.", "language_model."):
            if name.startswith(prefix):
                plain["model." + name[len(prefix):]] = value
                break
        else:
            if name.startswith("model."):
                plain[name] = value
    return {name: value for name, value in plain.items() if not _is_foreign(name)}


def expected_keys(shape: Qwen35Shape) -> set[str]:
    """这套结构**应该**有的权重键 ✓（与 HF / comfy 的键名**逐字**相同 ✓）。

    ⚠️ 用途是**对账** ✗：:func:`build_from_shapes` 建出来的 ``state_dict()`` 必须与它相等 ✓。
    多一个 ⇒ 加载时报「多余键」✓；少一个（比如把 ``norm.weight`` 写成裸 ``Parameter`` ✓）
    ⇒ 那一格**静默落空** ✓✗ —— 形状全对、结果全偏，只有这条能拦住 ✓。
    """
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for index, kind in enumerate(shape.layer_types):
        own = _LINEAR_SUFFIXES if kind == "linear_attention" else _FULL_SUFFIXES
        for suffix in (*_COMMON_SUFFIXES, *own):
            keys.add(f"model.layers.{index}.{suffix}")
    return keys


def infer_shape(shapes: dict[str, tuple[int, ...]]) -> Qwen35Shape:
    """按权重**形状**反推结构 ✓（只读头部的事实 ✗ 不猜层数/宽度 ✓）。

    ⚠️ ``linear_key_head_dim`` **读不出来** ✗ —— 形状里只有 value 侧有 ``norm.weight`` 可读 ✓，
    key 侧唯一的线索是 ``in_proj_qkv`` 的总宽 ✓。这里按「两侧 head_dim 相同」反推 ✓
    （Qwen3.5 全族两处都是 128 ✓），并用 ``key_dim % head_dim == 0`` 做硬校验 ✓ ——
    两侧真不等时，这条约束会当场拒掉 ✗（不会静默把 key 头数折半 ✓）。
    """
    plain = _language_shapes(shapes)

    def size(name: str) -> tuple[int, ...]:
        try:
            return tuple(int(v) for v in plain[name])
        except KeyError as err:
            raise ShapeError(
                f"缺权重 {name} ✗ —— 这不是 Qwen3.5 的文本塔 ✓（或者权重没下全 ✗），"
                f"不拿出厂结构硬装 ✓") from err

    vocab, hidden = size("model.embed_tokens.weight")
    if len(size("model.norm.weight")) != 1:
        raise ShapeError("model.norm.weight 必须是一维（RMSNorm 的 weight ✗）")

    # ⚠️ 层号必须**连续** 0..n-1 ✓：缺一层 ⇒ nn.ModuleList 按序号命名 ✓ 后面所有层的键**整位移** ✓✗
    layer_ids: dict[int, set[str]] = {}
    for name in plain:
        matched = _LAYER_RE.match(name)
        if matched is not None:
            layer_ids.setdefault(int(matched.group(1)), set()).add(matched.group(2))
    if not layer_ids:
        raise ShapeError("一个 model.layers.N.* 都没有 ✗ —— 这不是 Qwen3.5 文本塔 ✓")
    depth = len(layer_ids)
    if sorted(layer_ids) != list(range(depth)):
        raise ShapeError(f"层号不连续 ✗：拿到 {sorted(layer_ids)[:5]}…（共 {depth} 层）⇒ 权重不全 ✓")

    layer_types: list[str] = []
    for index in range(depth):
        suffixes = layer_ids[index]
        if "linear_attn.in_proj_qkv.weight" in suffixes:
            kind, own = "linear_attention", _LINEAR_SUFFIXES
        elif "self_attn.q_proj.weight" in suffixes:
            kind, own = "full_attention", _FULL_SUFFIXES
        else:
            raise ShapeError(
                f"第 {index} 层既没有 linear_attn.in_proj_qkv 也没有 self_attn.q_proj ✗ ⇒ 不是这一族的层 ✓")
        missing = [item for item in (*_COMMON_SUFFIXES, *own) if item not in suffixes]
        if missing:
            raise ShapeError(f"第 {index} 层（{kind}）缺 {missing} ✗ ⇒ 权重没下全 ✓")
        layer_types.append(kind)

    if "full_attention" not in layer_types:
        raise ShapeError("一层完整注意力都没有 ✗ —— 那不是 Qwen3.5 混合塔 ✓")
    if "linear_attention" not in layer_types:
        raise ShapeError("一层线性注意力都没有 ✗ —— 那不是 Qwen3.5 混合塔 ✓")

    full_index = layer_types.index("full_attention")
    prefix = f"model.layers.{full_index}.self_attn."
    head_dim = size(prefix + "q_norm.weight")[0]
    if size(prefix + "k_norm.weight")[0] != head_dim:
        raise ShapeError("q_norm 与 k_norm 的宽度不一致 ✗")
    q_out = size(prefix + "q_proj.weight")[0]
    if q_out % (2 * head_dim):
        raise ShapeError(
            f"q_proj 输出 {q_out} 不是 2×head_dim（{2 * head_dim}）的整数倍 ✗ ⇒ 推不出头数 ✓"
            f"（这一族的 q_proj **同时**出 query 与 gate ✓ 所以是 2 倍 ✓）")
    heads = q_out // (2 * head_dim)
    kv_out = size(prefix + "k_proj.weight")[0]
    if size(prefix + "v_proj.weight")[0] != kv_out:
        raise ShapeError("v_proj 与 k_proj 的输出维不一致 ✗（GQA 的 KV 头必须同宽 ✓）")
    if kv_out % head_dim:
        raise ShapeError(f"k_proj 输出 {kv_out} 不能被 head_dim（{head_dim}）整除 ✗")
    kv_heads = kv_out // head_dim
    if heads % kv_heads:
        raise ShapeError(f"heads（{heads}）不是 kv_heads（{kv_heads}）的整数倍 ✗")
    if size(prefix + "o_proj.weight")[1] != heads * head_dim:
        raise ShapeError("o_proj 的入维与 heads*head_dim 不一致 ✗")
    rotary_dim = int(head_dim * PARTIAL_ROTARY_FACTOR)
    if rotary_dim % 2 or not 0 < rotary_dim <= head_dim:
        raise ShapeError(f"partial RoPE 的维数 {rotary_dim} 非法 ✗（head_dim={head_dim} ✓）")

    linear_index = layer_types.index("linear_attention")
    lin = f"model.layers.{linear_index}.linear_attn."
    value_head_dim = size(lin + "norm.weight")[0]
    value_heads = size(lin + "in_proj_b.weight")[0]
    for name in (lin + "in_proj_a.weight",):
        if size(name)[0] != value_heads:
            raise ShapeError(f"{name} 的输出维与 in_proj_b 不一致 ✗（a/b 每个 value 头一个标量 ✓）")
    if size(lin + "A_log") != (value_heads,):
        raise ShapeError(f"A_log 形状 {size(lin + 'A_log')} ≠ ({value_heads},) ✗")
    if size(lin + "dt_bias") != (value_heads,):
        raise ShapeError(f"dt_bias 形状 {size(lin + 'dt_bias')} ≠ ({value_heads},) ✗")
    value_dim = size(lin + "in_proj_z.weight")[0]
    if value_dim != value_heads * value_head_dim:
        raise ShapeError(f"in_proj_z 输出 {value_dim} ≠ value_heads×value_head_dim ✗")
    qkv_out = size(lin + "in_proj_qkv.weight")[0]
    if qkv_out <= value_dim or (qkv_out - value_dim) % 2:
        raise ShapeError(f"in_proj_qkv 输出 {qkv_out} 推不出 key 维 ✗（应 = 2×key_dim + {value_dim} ✓）")
    key_dim = (qkv_out - value_dim) // 2
    key_head_dim = value_head_dim
    if key_dim % key_head_dim:
        raise ShapeError(
            f"key 维 {key_dim} 不能被 key_head_dim（{key_head_dim}）整除 ✗ ⇒ "
            f"这一族的 key/value 头宽不等 ✓ 本模块按「相等」反推 ✗ 不硬装 ✓")
    conv = size(lin + "conv1d.weight")
    if conv[0] != qkv_out or (len(conv) == 3 and conv[1] != 1):
        raise ShapeError(f"conv1d 形状 {conv} 不像 depthwise（应为 ({qkv_out}, 1, kernel) ✓）✗")
    conv_kernel = conv[-1]
    if size(lin + "out_proj.weight") != (hidden, value_dim):
        raise ShapeError(
            f"linear_attn.out_proj 形状 {size(lin + 'out_proj.weight')} ≠ (hidden, value_dim) = "
            f"({hidden}, {value_dim}) ✗")
    ffn = size("model.layers.0.mlp.gate_proj.weight")[0]
    # ⚠️ down_proj 是 ``(hidden, ffn)`` ✓ —— 入维（``[1]``）才是 ffn ✗（``[0]`` 是 hidden ✓）
    if size("model.layers.0.mlp.down_proj.weight") != (hidden, ffn):
        raise ShapeError(
            f"mlp.down_proj 形状 {size('model.layers.0.mlp.down_proj.weight')} ≠ (hidden, ffn)"
            f" = ({hidden}, {ffn}) ✗")
    return Qwen35Shape(
        vocab=vocab, hidden=hidden, depth=depth, ffn=ffn,
        heads=heads, kv_heads=kv_heads, head_dim=head_dim, rotary_dim=rotary_dim,
        linear_key_heads=key_dim // key_head_dim, linear_value_heads=value_heads,
        key_head_dim=key_head_dim, value_head_dim=value_head_dim, conv_kernel=conv_kernel,
        layer_types=tuple(layer_types),
    )


def tokens_of(text: str) -> str:
    """套上提示词模板 ✓ ⇒ 送进分词器的**原样字符串** ✓（不 strip ✗ 不改大小写 ✗）。"""
    return QWEN35_TEMPLATE.format(text)


@dataclass(frozen=True)
class Qwen35TeRuntime:
    """后端要的「文本编码器配置」那股形状 ✓（``output_dim`` / ``max_length`` / ``to_dict`` ✓）。

    ⚠️ 为什么不直接把 :class:`Qwen35Shape` 交出去 ✗：后端 ``self._textConfig`` 那个位置原本放的是
    ``text_encoder.TextEncoderConfig`` ✓（``max_length`` / ``output_dim`` / ``to_dict()`` ✓），
    而 :class:`Qwen35Shape` **没有**这些字段 ✓✗ ⇒ 直接塞进去，``encode_text`` 会在
    ``config.max_length`` 上 ``AttributeError`` ✓（报得还像"编码器挂了"✓✗）。这层壳只做**翻译** ✓。
    """

    shape: Qwen35Shape
    max_length: int = 512
    template: str = QWEN35_TEMPLATE

    @property
    def output_dim(self) -> int:
        """文本塔吐的宽度 ✓（= ``hidden`` ✓ = 4096 ✓；须与主干 ``text_dim`` 一致 ✓）。"""
        return int(self.shape.hidden)

    @property
    def vocab_size(self) -> int:
        return int(self.shape.vocab)

    def prompt(self, text: str) -> str:
        """套模板 ✓（**唯一**加模板的地方 ✓ —— 后端不要在别处再拼一次 ✓✗）。"""
        return self.template.format("" if text is None else str(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "qwen35-te",
            "family": "Qwen3.5-9B 混合线性注意力（Qwen-Image 2.1 文本塔 ✓）",
            "outputDim": self.output_dim, "vocabSize": self.vocab_size,
            "maxLength": int(self.max_length), "layerIdx": int(self.shape.layer_idx),
            "targetLayer": int(self.shape.target_layer()),
            "depth": int(self.shape.depth), "hidden": int(self.shape.hidden),
            "heads": int(self.shape.heads), "kvHeads": int(self.shape.kv_heads),
            "headDim": int(self.shape.head_dim), "rotaryDim": int(self.shape.rotary_dim),
            "linearHeads": int(self.shape.linear_key_heads),
            "linearValueHeads": int(self.shape.linear_value_heads),
            "convKernel": int(self.shape.conv_kernel),
            "layerTypes": list(self.shape.layer_types),
            "template": self.template,
            "note": "取倒数第 2 层 hidden ✓、**不过 final norm** ✓（同参考实现 ✓）；RMSNorm 走 (1+w) ✓",
        }


def _rms_norm(x: Any, weight: Any, eps: float, *, add: bool = False) -> Any:
    """``x / sqrt(mean(x²)+eps) * w`` ✓，``add=True`` 时 ``w = weight + 1`` ✓。

    ⚠️ 走 ``F.rms_norm``（torch 2.4+ ✓）而不是手搓 ✓：参考实现就是 ``comfy.rmsnorm.rms_norm``
    ⇒ 同一个 kernel ✓ 才有机会**逐位相同** ✓（手搓在 bf16 下尾数路径不同 ✓ 会差 1e-3 量级 ✓）。
    """
    import torch  # noqa: PLC0415

    wide = weight + 1.0 if add else weight
    kernel = getattr(torch.nn.functional, "rms_norm", None)
    if kernel is not None:
        return kernel(x, (int(x.shape[-1]),), weight=wide.to(dtype=x.dtype), eps=eps)
    source = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    return (value * wide.float()).to(source)


def _rope_tables(length: int, rotary_dim: int, theta: float, device: Any) -> tuple[Any, Any]:
    """RoPE 的 cos/sin ✓（``(length, rotary_dim//2)`` ✓ **一律 float32** ✓ —— 角度精度要紧 ✓）。"""
    import torch  # noqa: PLC0415

    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device,
                                            dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_partial_rope(x: Any, cos: Any, sin: Any, rotary_dim: int) -> Any:
    """只转**前 ``rotary_dim``** 维 ✓、**split-half** 配对 ✓、其余维度原样透传 ✓。

    ⚠️ 配对是「第 i 维 ↔ 第 i+half 维」✓（``rope_matrix`` 把 ``cos[:half], -sin, sin, cos[half:]``
    拼成 2×2 ✓）—— 相邻配对（``i ↔ i+1``）**不报错**，只是把语义转到别的维度上 ✓✗。
    """
    import torch  # noqa: PLC0415

    if rotary_dim <= 0:
        return x
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    first, second = rot[..., :half].float(), rot[..., half:].float()
    co = cos.to(torch.float32)[None, None]
    si = sin.to(torch.float32)[None, None]
    turned = torch.cat([first * co - second * si, first * si + second * co], dim=-1)
    return torch.cat([turned.to(x.dtype), passthrough], dim=-1)


def _chunk_gated_delta_rule(query: Any, key: Any, value: Any, g: Any, beta: Any,
                            chunk_size: int = 64) -> Any:
    """**chunked delta rule** ✓（= ``qwen35.py::torch_chunk_gated_delta_rule`` 的逐行移植 ✓）。

    入参都是 ``(B, S, H, D)`` ✓，返回 ``(B, S, H, Dv)`` ✓（**输入精度** ✓ —— 内部一律 float32 ✓）。

    ⚠️ 这是「线性注意力」的全部内容 ✓，四步都不能省 ✓：
    ``q/k 归一化 → ``beta`` 缩放 → 块内下三角求逆（``attn`` 那段递归）→ 块间状态递推 ✓。
    块内那段递归（``for i in range(1, chunk_size)``）**必须**按原序逐行做 ✓ —— 它是
    ``(I + 严格下三角)^-1`` 的原地求解 ✓，换成矩阵求逆数值不同 ✓（不报错 ✓✗）。

    ⚠️ ``g`` 是**累积后的**对数衰减 ✓（进来先 ``cumsum`` ✓）；``decay_mask`` 是
    ``exp(g_i - g_j)`` 的下三角 ✓ —— 下标写反（``g_j - g_i``）不报错、只是衰减变成放大 ✓✗。
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    source = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [item.transpose(1, 2).contiguous().to(torch.float32)
                                 for item in (query, key, value, beta, g)]
    batch, heads, length, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total = length + pad
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        item.reshape(item.shape[0], item.shape[1], -1, chunk_size, item.shape[-1])
        for item in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                      diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch, heads, k_dim, v_dim).to(value)
    out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                      diagonal=1)
    for i in range(0, total // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        block = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - (k_cumdecay[:, :, i] @ state)
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + block @ v_new
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
                 @ v_new)
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])
    out = out[:, :, :length]
    return out.transpose(1, 2).contiguous().to(source)


def build_from_shapes(shapes: dict[str, tuple[int, ...]], *,
                      layer_idx: int = -2, dtype: Any = None) -> Any:
    """按形状建**文本塔**（不含 ``lm_head`` ✓ / 不含视觉塔 ✓）⇒ ``nn.Module`` ✓。

    ``dtype`` 走 ``set_default_dtype`` ✓：9B 的 bf16 权重若先建 fp32 再转 ⇒ 峰值 **74 GB** ✗✗
    （与 :func:`~.llm.build_llm` 同一条手法 ✓）。默认 ``None`` = 不碰全局默认（fp32 ✓，CPU 自检用 ✓）。
    """
    from . import llm as llm_mod  # noqa: PLC0415 —— 局部 ✓：纯函数（infer_shape/tokens_of）不该拖 torch ✓

    shape = infer_shape(shapes)
    if layer_idx != shape.layer_idx:
        shape = replace(shape, layer_idx=layer_idx)
    shape.target_layer()                          # 越界当场炸 ✓（不要等前向 ✗）
    prim = llm_mod.primitives()
    nn, torch, F = prim["nn"], prim["torch"], prim["F"]

    class _RmsNorm(nn.Module):
        """Qwen3.5 的 ``RMSNorm`` ✓：``add=True`` ⇒ ``(1 + weight)`` ✓（层 norm 与 QK-norm 都用它 ✓）。"""

        def __init__(self, dim: int, add: bool = True) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.add = bool(add)

        def forward(self, x: Any) -> Any:
            return _rms_norm(x, self.weight, shape.rms_eps, add=self.add)

    class _GatedNorm(nn.Module):
        """``RMSNormGated`` ✓：**朴素** ``weight``（没有 1+w ✓）再乘 ``silu(gate)`` ✓。

        ⚠️ 与 :class:`_RmsNorm` 差的就是那个 ``+1`` ✓ —— 参考实现里这一格没传 ``add`` ✓；
        照抄成 ``add=True`` 会让线性注意力的输出整体放大 ✓（不报错 ✓✗）。
        """

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Any, gate: Any) -> Any:
            return _rms_norm(x, self.weight, shape.rms_eps, add=False) * F.silu(gate.to(x.dtype))

    class _GatedDeltaNet(nn.Module):
        """线性注意力层 ✓（``Gated DeltaNet`` ✓）：因果卷积 ＋ delta rule ＋ 门控 norm ＋ 输出投影 ✓。"""

        def __init__(self) -> None:
            super().__init__()
            self.in_proj_qkv = nn.Linear(shape.hidden, shape.conv_dim, bias=False)
            self.in_proj_z = nn.Linear(shape.hidden, shape.value_dim, bias=False)
            self.in_proj_b = nn.Linear(shape.hidden, shape.linear_value_heads, bias=False)
            self.in_proj_a = nn.Linear(shape.hidden, shape.linear_value_heads, bias=False)
            self.out_proj = nn.Linear(shape.value_dim, shape.hidden, bias=False)
            self.dt_bias = nn.Parameter(torch.zeros(shape.linear_value_heads))
            self.A_log = nn.Parameter(torch.zeros(shape.linear_value_heads))
            # ⚠️ depthwise：groups=conv_dim ⇒ 每个通道一套自己的 4 个系数 ✓；padding=kernel-1 后**切尾** ✓
            self.conv1d = nn.Conv1d(shape.conv_dim, shape.conv_dim, kernel_size=shape.conv_kernel,
                                    groups=shape.conv_dim, bias=False,
                                    padding=shape.conv_kernel - 1)
            self.norm = _GatedNorm(shape.value_head_dim)

        def forward(self, x: Any) -> Any:
            batch, length, _ = x.shape
            mixed = self.in_proj_qkv(x)                    # (B, S, 2*key+v)
            z = self.in_proj_z(x)                          # (B, S, v)
            b = self.in_proj_b(x)
            a = self.in_proj_a(x)
            mixed = F.silu(self.conv1d(mixed.transpose(1, 2))[:, :, :length]).transpose(1, 2)
            query, key, value = mixed.split(
                [shape.key_dim, shape.key_dim, shape.value_dim], dim=-1)
            # ⚠️ g_decay / dt_bias 一律 float32 ✓（参考实现 ``cast_to_device(..., torch.float32)`` ✓）
            decay = -self.A_log.float().exp()
            g = decay * F.softplus(a.float() + self.dt_bias.float())
            beta = b.sigmoid()
            query = query.reshape(batch, length, shape.linear_key_heads, shape.key_head_dim)
            key = key.reshape(batch, length, shape.linear_key_heads, shape.key_head_dim)
            value = value.reshape(batch, length, shape.linear_value_heads, shape.value_head_dim)
            if shape.linear_value_heads != shape.linear_key_heads:
                # ⚠️ **repeat_interleave** ✓（不是 ``repeat`` ✗）—— 头必须**相邻复制**才对得上 value 头序 ✓
                repeat = shape.linear_value_heads // shape.linear_key_heads
                query = query.repeat_interleave(repeat, dim=2)
                key = key.repeat_interleave(repeat, dim=2)
            core = _chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
            core = self.norm(core.reshape(-1, shape.value_head_dim),
                             z.reshape(-1, shape.value_head_dim))
            return self.out_proj(core.reshape(batch, length, -1))

    class _GatedAttention(nn.Module):
        """完整注意力层 ✓：``q_proj`` 出 **query+gate** ✓、partial RoPE ✓、输出乘 ``sigmoid(gate)`` ✓。"""

        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(shape.hidden, shape.inner * 2, bias=False)
            self.k_proj = nn.Linear(shape.hidden, shape.kv_heads * shape.head_dim, bias=False)
            self.v_proj = nn.Linear(shape.hidden, shape.kv_heads * shape.head_dim, bias=False)
            self.o_proj = nn.Linear(shape.inner, shape.hidden, bias=False)
            self.q_norm = _RmsNorm(shape.head_dim, add=shape.norm_add)
            self.k_norm = _RmsNorm(shape.head_dim, add=shape.norm_add)

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            batch, length, _ = x.shape
            # ⚠️ 头**交错**：view(B,L,H,2*hd) 再前后劈 ✓（chunk(2) 会把 gate 劈到另外一半头上 ✓✗）
            qg = self.q_proj(x).view(batch, length, shape.heads, shape.head_dim * 2)
            query, gate = qg[..., : shape.head_dim], qg[..., shape.head_dim:]
            gate = gate.reshape(batch, length, -1)
            key = self.k_proj(x).view(batch, length, shape.kv_heads, shape.head_dim)
            value = self.v_proj(x).view(batch, length, shape.kv_heads, shape.head_dim)
            query = self.q_norm(query).transpose(1, 2)
            key = self.k_norm(key).transpose(1, 2)
            value = value.transpose(1, 2)
            query = _apply_partial_rope(query, cos, sin, shape.rotary_dim)
            key = _apply_partial_rope(key, cos, sin, shape.rotary_dim)
            gqa = {"enable_gqa": True} if shape.heads != shape.kv_heads else {}
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True, **gqa)
            out = out.transpose(1, 2).reshape(batch, length, -1)
            return self.o_proj(out * gate.sigmoid())

    class _Mlp(nn.Module):
        """SwiGLU ✓，参数名照 HF（``gate_proj``/``up_proj``/``down_proj`` ✓）—— 与权重键逐字一致 ✓。"""

        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
            self.up_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
            self.down_proj = nn.Linear(shape.ffn, shape.hidden, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class _Block(nn.Module):
        """pre-norm 一层 ✓（``norm → attn → 残差 → norm → mlp → 残差`` ✓，与参考实现同序 ✓）。"""

        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind
            if kind == "linear_attention":
                self.linear_attn = _GatedDeltaNet()
            else:
                self.self_attn = _GatedAttention()
            self.mlp = _Mlp()
            self.input_layernorm = _RmsNorm(shape.hidden, add=shape.norm_add)
            self.post_attention_layernorm = _RmsNorm(shape.hidden, add=shape.norm_add)

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            normed = self.input_layernorm(x)
            if self.kind == "linear_attention":
                mixed = self.linear_attn(normed)
            else:
                mixed = self.self_attn(normed, cos, sin)
            x = x + mixed
            return x + self.mlp(self.post_attention_layernorm(x))

    class Qwen35TextTowerModule(nn.Module):
        """文本塔本体 ✓：``model.embed_tokens`` / ``model.layers.N.*`` / ``model.norm`` ✓。

        ⚠️ ``model.norm`` **建了但不用** ✗：盘上有这份权重 ✓（装载要对齐键 ✓），
        而参考实现取的是**未过 final norm** 的中间层 hidden ✓（``layer_norm_hidden_state=False`` ✓）。
        """

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(shape.vocab, shape.hidden)
            self.model.layers = nn.ModuleList([_Block(kind) for kind in shape.layer_types])
            self.model.norm = _RmsNorm(shape.hidden, add=shape.norm_add)
            self.shape = shape

        def forward(self, ids: Any) -> Any:
            """``ids`` ⇒ **第 ``layer_idx`` 层之后的 hidden** ✓ ``(1, s, hidden)`` ✓。

            ⚠️ batch 只支持 **1** ✗：因果掩码是「整条都看」✓，**没有**按条 pad 的那一路 ✓
            ⇒ 一次编码一条 ✓，多条由调用方逐条走 ✓（编码很快，别拿正确性换 ✓）。
            """
            batch, length = int(ids.shape[0]), int(ids.shape[1])
            if batch != 1:
                raise ShapeError(f"文本塔一次只编码一条 ✗（收到 batch={batch} ✓）—— 逐条调用 ✓")
            if length < 1:
                raise ShapeError("ids 长度为 0 ✗")
            x = self.model.embed_tokens(ids)
            cos, sin = _rope_tables(length, shape.rotary_dim, shape.rope_theta, ids.device)
            target = shape.target_layer()
            hidden = None
            for index, block in enumerate(self.model.layers):
                x = block(x, cos, sin)
                if index == target:
                    hidden = x                       # ⚠️ 不过 final norm ✓（参考实现同一口径 ✓）
            if hidden is None:
                raise ShapeError(f"取不到第 {target} 层 ✗")
            return hidden

    previous = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)             # ⚠️ 全局状态 ⇒ 必须 finally 复原 ✓
    try:
        module = Qwen35TextTowerModule()
    finally:
        torch.set_default_dtype(previous)
    return module if dtype is None else module.to(dtype)
