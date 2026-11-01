"""Qwen-Image（2512 t2i / Edit 2511）的**文本塔** = Qwen2.5-VL-7B 的**语言部分**
（因果注意力 ✓、取**最后一层** hidden ✓、**不做** final norm ✓、**q/k/v 带 bias** ✓、**没有** QK-norm ✗）。

事实来源（只参考行为、实现自己写 ✓）：

* ``reference/comfy-org/ComfyUI/comfy/text_encoders/llama.py::Qwen25_7BVLI_Config`` ——
  ``vocab_size=152064, hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
  num_attention_heads=28, num_key_value_heads=4, head_dim=128, rms_norm_eps=1e-6,
  rope_theta=1e6, qkv_bias=True, q_norm=None, k_norm=None, mlp_activation="silu",
  lm_head=False`` ✓。
* ``comfy/text_encoders/qwen_image.py::Qwen25_7BVLIModel`` —— ``layer="last"`` ＋
  ``layer_norm_hidden_state=False`` ✓ ⇒ 取的是**最后一层块之后、``model.norm`` 之前**的 hidden ✓；
  ``attention_mask=True`` ✓（但见下面「不补位」那条 ✓）。
* 同文件 ``QwenImageTokenizer`` —— 两套模板 ✓（无图 = :data:`TE_TEMPLATE` ✓，有图 = :data:`TE_TEMPLATE_EDIT` ✓），
  且 ``encode_token_weights`` 会把**系统段整段切掉** ✓（见 :func:`trim_start` ✓）。
* 同文件 ``Qwen25_7BVLITokenizer`` —— ``embedding_size=3584`` / ``max_length=99999999`` /
  ``pad_token=151643`` / ``has_start_token=False`` ✓；词表与 Z-Image 那份 ``qwen25_tokenizer`` **同源** ✓。

⚠️ 与 Z-Image 文本塔（:mod:`.qwen3_te` ✓）的四格差别，一格都不能串 ✗：

==================================  ==============================  ==========================
                                   Z-Image（`qwen3_te` ✓）          Qwen-Image（本模块 ✓）
==================================  ==============================  ==========================
取哪一层                            **倒数第 2 层**（`-2` ✓）        **最后一层**（`-1` ✓）
q/k/v 偏置                          **没有** ✗                     **有**（三根 ✓）
QK-norm（per-head）                 **有**（`q_norm`/`k_norm` ✓）   **没有** ✗
提示词模板                          `user` 单段 ✓                   `system` + `user` 两段 ✓
==================================  ==============================  ==========================

⇒ 四处「装得上、算得出、但结果与提示词无关」的换法：

① 取 ``-2`` 而不是 ``-1`` ⇒ 差的是**整个语义空间** ✓（不报错 ✗）；
② 少 q/k/v 的 bias ⇒ 每层少一个常数偏置 ✓，形状全对、只是提示词跟随明显变弱 ✓✗；
③ 多做一次 ``model.norm`` ⇒ 送进主干的是被 RMSNorm 缩放过的向量 ✗（出图整体偏灰/偏曝 ✓✗）；
④ 模板只给 ``user`` 段 ⇒ 少了「请详细描述颜色/形状/空间关系」这句**指令** ✓，
   Qwen-Image 是**指令跟随**口径 ✓ ⇒ 长提示词的结构会被读成"续写"而不是"照做" ✓✗。

⚠️ 本模块**只管语言塔** ✗：Edit 2511 的参考图要走 Qwen2.5-VL 的**视觉塔**（``visual.*`` 714 张量 ✓）
把 ``<|image_pad|>`` 换成图像嵌入 ✓ —— 那部分不在本模块 ✓，所以 :func:`prompt_of` 在
``with_image=True`` 而没有图像嵌入时**明确报错** ✓，而不是把占位符当普通文本 ✓✗。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .llm import LlmConfig

__all__ = [
    "EDIT_SENTINEL_TOKEN",
    "IMG_PAD_ID",
    "IM_START_ID",
    "LAYER_IDX",
    "PAD_TOKEN_ID",
    "PROMPT_HEAD_TOKEN",
    "QWEN25_HEAD_DIM",
    "TE_TEMPLATE",
    "TE_TEMPLATE_EDIT",
    "Qwen25TeRuntime",
    "Qwen25TeShape",
    "ShapeError",
    "build_from_shapes",
    "expected_keys",
    "infer_shape",
    "prompt_of",
    "trim_start",
    "unclaimed_extras",
]

#: Qwen 系的**填空** token ✓（与 Z-Image 同一份词表 ⇒ 同一个 id ✓）
PAD_TOKEN_ID = 151643

#: 模板里的特殊 token id ✓（``<|im_start|>`` / ``<|im_end|>`` / ``user`` / ``\n`` ✓）
IM_START_ID = 151644
IM_END_ID = 151648
USER_ID = 872
NEWLINE_ID = 198

#: 视觉占位符 ✓（Edit 模板里那三个 ✓ —— 它们的嵌入必须由视觉塔给出 ✗ 不是词表嵌入 ✓）
IMG_START_TOKEN, IMG_PAD_TOKEN, IMG_END_TOKEN = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"

#: ``<|image_pad|>`` 的 id ✓（参考实现按 151655 认这一格 ✓ —— 视觉塔的输出就替换这些位置 ✓）
IMG_PAD_ID = 151655
EDIT_SENTINEL_TOKEN = IMG_PAD_TOKEN

#: 无图（2512 t2i ✓）模板 ✓ —— ``{}`` 里是**用户原话**（不裁剪 ✗ 不加工 ✗）。
TE_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

#: 有图（Edit 2511 ✓）模板 ✓ —— ``<|vision_start|><|image_pad|><|vision_end|>`` 在前 ✓。
TE_TEMPLATE_EDIT = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets the user's requirements while "
    "maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

#: 这一族的取层口径 ✓（``layer="last"`` ⇒ ``-1`` ✓，不是 Z-Image 的 ``-2`` ✗）
LAYER_IDX = -1

#: ``head_dim`` 在权重里**读不出来** ✗（没有 q_norm/k_norm ✗，q_proj 是方阵 ✓）⇒ 只能按发布配置给 ✓。
#: 出处：``Qwen25_7BVLI_Config.head_dim = 128`` ✓，且与权重**互核**：``q_proj`` 出 3584 = 28×128 ✓、
#: ``k_proj`` 出 512 = 4×128 ✓（两个不同宽度的张量同时被 128 整除，且 3584/512 = 7 = 28/4 ✓）。
QWEN25_HEAD_DIM = 128

#: 模板里第二个 ``<|im_start|>`` 之后紧跟的那两个 token ✓（``user`` ＋ 换行 ✓ —— 一并切掉 ✓）
PROMPT_HEAD_TOKEN = (USER_ID, NEWLINE_ID)

_LAYER_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

#: 盘上**有、但本模块不建**的键 ✓：``lm_head.weight``（comfy 的 ``lm_head=False`` ⇒ 不建 ✓，
#: 文本塔只出 hidden ✓）、视觉塔 ``visual.*``（另一步 ✓）、fp8 的 ``*.scale_input``/``*.scale_weight``
#: （反量化时被吃掉 ✓）。⚠️ 它们在「键对账」里必须被**认领** ✓，否则会被当成多余键误报 ✗。
ALLOWED_EXTRA_PREFIXES: tuple[str, ...] = ("visual.", "lm_head.")
#: 顶层另有 **0 元素哨兵** ``scaled_fp8`` ✓（dtype ``F8_E4M3`` ✓、形状 ``(0,)`` ✓）——
#: 它不参与计算 ✓，只自述「这份权重是 fp8_scaled 存法 ✓」（与 DocDiT 的
#: ``__index_timestep_zero__`` 同一物种 ✓）。不认领 ⇒ 键对账把它当"多余键"误报 ✗。
ALLOWED_EXTRA_NAMES: tuple[str, ...] = ("scaled_fp8",)
ALLOWED_EXTRA_SUFFIXES: tuple[str, ...] = (".scale_input", ".scale_weight")


class ShapeError(ValueError):
    """权重形状推不出可装的结构 ✓（**明确报错** ✗ 不拿出厂形状硬装 ✓）。"""


@dataclass(frozen=True)
class Qwen25TeShape:
    """Qwen-Image 文本塔的**全部**结构参数 ✓（除 ``eps``/``theta``/``head_dim`` 外都从形状读出 ✓）。"""

    vocab: int
    hidden: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    qkv_bias: bool = True
    qk_norm: bool = False
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    #: 取哪一层 ✓：``-1`` = **最后一层块之后**（``layer="last"`` ✓）+ 不过 ``model.norm`` ✓
    layer_idx: int = LAYER_IDX

    def llm_config(self) -> LlmConfig:
        """摊成 :class:`~.llm.LlmConfig` ✓ —— 算子与校验只有一份 ✓。"""
        return LlmConfig(
            vocab_size=self.vocab, hidden=self.hidden, depth=self.depth,
            heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim,
            ffn=self.ffn, rope_theta=self.rope_theta, rms_eps=self.rms_eps,
            tie_embeddings=False,          # ⚠️ 不建 lm_head ✗
            qk_norm=self.qk_norm, qkv_bias=self.qkv_bias, pad_id=PAD_TOKEN_ID,
        )

    def target_layer(self) -> int:
        """``layer_idx``（可负 ✓）⇒ **0-based 层号** ✓；越界 ⇒ :class:`ShapeError` ✗。"""
        index = self.layer_idx if self.layer_idx >= 0 else self.depth + self.layer_idx
        if not 0 <= index < self.depth:
            raise ShapeError(
                f"layer_idx={self.layer_idx} 在这 {self.depth} 层上越界 ✗"
                f"（折算成 {index} ⇒ 必须在 0..{self.depth - 1} ✓）")
        return index


def infer_shape(shapes: dict[str, tuple[int, ...]], *,
                head_dim: int | None = None) -> Qwen25TeShape:
    """按权重**形状**反推结构 ✓（层数/宽度/头数都读得出来 ✓，``head_dim`` 只能由调用方给 ✓）。

    ⚠️ ``head_dim``：Qwen3 那族靠 ``self_attn.q_norm.weight`` 的宽度反推 ✓；
    Qwen2.5 **没有** QK-norm ✗，且 ``q_proj`` 是 ``(3584, 3584)`` 方阵 ✓ ⇒
    「28 头 × 128」与「14 头 × 256」在形状上**一样** ✓✗ ⇒ 必须显式传 ✓（默认取发布配置
    :data:`QWEN25_HEAD_DIM` ✓），再用两条整除关系**互核** ✓。
    """
    def size(name: str) -> tuple[int, ...]:
        try:
            return tuple(int(v) for v in shapes[name])
        except KeyError as err:
            raise ShapeError(
                f"缺权重 {name} ✗ —— 这不是 Qwen2.5-VL 的语言塔 ✓（或者权重没下全 ✗）") from err

    vocab, hidden = size("model.embed_tokens.weight")
    if len(size("model.norm.weight")) != 1:
        raise ShapeError("model.norm.weight 必须是一维 ✗")

    layer_ids: dict[int, set[str]] = {}
    for name in shapes:
        matched = _LAYER_RE.match(name)
        if matched is None:
            continue
        layer_ids.setdefault(int(matched.group(1)), set()).add(matched.group(2))
    if not layer_ids:
        raise ShapeError("一个 model.layers.N.* 都没有 ✗ —— 这不是 Qwen 系语言塔 ✓")
    depth = len(layer_ids)
    if sorted(layer_ids) != list(range(depth)):
        raise ShapeError(f"层号不连续 ✗：拿到 {sorted(layer_ids)[:5]}…（共 {depth} 层）⇒ 权重不全 ✓")
    for index, suffixes in layer_ids.items():
        missing = [item for item in _LAYER_SUFFIXES if item not in suffixes]
        if missing:
            raise ShapeError(f"第 {index} 层缺 {missing} ✗ ⇒ 权重没下全 ✓（或不是这一族 ✗）")

    q_out = size("model.layers.0.self_attn.q_proj.weight")[0]
    kv_out = size("model.layers.0.self_attn.k_proj.weight")[0]
    ffn = size("model.layers.0.mlp.gate_proj.weight")[0]
    has_qk_norm = ("model.layers.0.self_attn.q_norm.weight" in shapes
                   and "model.layers.0.self_attn.k_norm.weight" in shapes)
    if has_qk_norm:
        raise ShapeError(
            "第 0 层带 ``self_attn.q_norm/k_norm`` ✗ ⇒ 这是 Qwen3 那一族（Z-Image 文本塔 ✓），"
            "不是 Qwen2.5-VL ✓ —— 取层与 bias 口径都不一样 ✗")

    #: ⚠️ bias 的三根键是**这一族**的指纹 ✓：`qkv_bias=True` ✓。有 QK-norm 的族没有它们 ✗。
    biases = ("model.layers.0.self_attn.q_proj.bias",
              "model.layers.0.self_attn.k_proj.bias",
              "model.layers.0.self_attn.v_proj.bias")
    present = [name for name in biases if name in shapes]
    if present and len(present) != len(biases):
        raise ShapeError(f"q/k/v 的 bias 只有 {present} ✗ ⇒ 权重不全 ✓")
    qkv_bias = bool(present)
    if not qkv_bias:
        raise ShapeError(
            "缺 q/k/v 的三根 bias ✗ ⇒ 不是 Qwen2.5-VL 的语言塔 ✓"
            "（Qwen3 那族才没有它们 ✓；本模块的 ``qkv_bias`` 是写死的族指纹 ✓）")
    if "model.layers.0.self_attn.o_proj.bias" in shapes:
        raise ShapeError("``o_proj`` 带 bias ✗ —— Qwen2.5 的 o_proj 是无偏置的 ✓（不是这一族 ✗）")

    dim = QWEN25_HEAD_DIM if head_dim is None else int(head_dim)
    if dim <= 0 or dim % 2 != 0:
        raise ShapeError(f"head_dim 必须是正偶数 ✗（收到 {dim} ✗）—— RoPE 要配对旋转 ✓")
    if q_out % dim or kv_out % dim:
        raise ShapeError(
            f"q/kv 输出维（{q_out}/{kv_out}）不能被 head_dim（{dim}）整除 ✗")
    heads, kv_heads = q_out // dim, kv_out // dim
    if heads % kv_heads:
        raise ShapeError(f"GQA：heads（{heads}）不能被 kv_heads（{kv_heads}）整除 ✗")
    if heads * dim != hidden:
        raise ShapeError(
            f"heads × head_dim = {heads * dim} ≠ hidden {hidden} ✗"
            f"（这一族 q_proj 是方阵 ⇒ 两者必须相等 ✓；不等说明 head_dim 给错了 ✓）")
    if size("model.layers.0.self_attn.v_proj.weight")[0] != kv_out:
        raise ShapeError("v_proj 与 k_proj 的输出维不一致 ✗（GQA 的 KV 头必须同宽 ✓）")
    if size("model.layers.0.self_attn.o_proj.weight")[1] != q_out:
        raise ShapeError("o_proj 的入维与 q_proj 的输出维不一致 ✗")
    #: ⚠️ ``down_proj`` 是 ``(hidden, ffn)`` ✓ ⇒ 入维 ``[1]`` 才是 ffn ✗（``[0]`` 是 hidden ✓）
    if size("model.layers.0.mlp.down_proj.weight") != (hidden, ffn):
        raise ShapeError(
            f"mlp.down_proj 形状 {size('model.layers.0.mlp.down_proj.weight')} "
            f"≠ (hidden, ffn) = ({hidden}, {ffn}) ✗")
    return Qwen25TeShape(vocab=vocab, hidden=hidden, depth=depth, heads=heads,
                         kv_heads=kv_heads, head_dim=dim, ffn=ffn, qkv_bias=qkv_bias)


def expected_keys(shape: Qwen25TeShape) -> set[str]:
    """本模块**会建**的每一根键 ✓（与 :meth:`forward` 用到的参数一一对应 ✓）。"""
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for index in range(shape.depth):
        head = f"model.layers.{index}."
        for suffix in _LAYER_SUFFIXES:
            keys.add(head + suffix)
        if shape.qkv_bias:
            for name in ("q_proj", "k_proj", "v_proj"):
                keys.add(f"{head}self_attn.{name}.bias")
        if shape.qk_norm:
            keys.update({f"{head}self_attn.q_norm.weight",
                         f"{head}self_attn.k_norm.weight"})
    return keys


def unclaimed_extras(shapes: Mapping[str, Any], shape: Qwen25TeShape) -> list[str]:
    """盘上有、本模块既不建也**不认识**的键 ✓（应恒为空 ✓）。

    认领规则**只有一份** ✓（``ALLOWED_EXTRA_*`` ✓）—— 判据与自检共用它 ✓，别各写一遍 ✗。
    """
    extras = {name for name in shapes if name not in expected_keys(shape)}
    return sorted(
        name for name in extras
        if name not in ALLOWED_EXTRA_NAMES
        and not name.startswith(ALLOWED_EXTRA_PREFIXES)
        and not name.endswith(ALLOWED_EXTRA_SUFFIXES))


def trim_start(ids: Sequence[int]) -> int:
    """返回**送进主干**的起点下标 ✓：切掉模板前两段，只留「正文 + 收尾」✓。

    与参考实现逐条对应 ✓（``QwenImageTEModel.encode_token_weights`` ✓）：

    1. 找到**第二个** ``<|im_start|>`` 的下标 ⇒ 起点 ✓（第 1 个开了 ``system`` 段 ✓，整段丢掉 ✓）；
    2. 若紧接着是 ``user`` + ``\\n`` ⇒ 起点再 +3 ✓（连角色名一起丢掉 ✓）。

    ⚠️ 于是真正进 DiT 的文本 token 是 ``[正文…, <|im_end|>, \\n, <|im_start|>, assistant, \\n]`` ✓
    —— **不是**整条模板 ✓✗。多送 system 段不会报错 ✗，只是主干看到的 token 数与内容都变了 ✓✗
    （这也是「同一份提示词、换个引擎结果就不一样」的常见来源之一 ✓）。

    ⚠️ 少于两个 ``<|im_start|>`` ⇒ 报错 ✓（说明调用方给的**不是**套过模板的字符串 ✗，
    参考实现在这种输入上会退化成「只留最后 1 个 token」✓✗ —— 那是坏掉，不是口径 ✓）。
    """
    seen = 0
    start: int | None = None
    for index, token in enumerate(ids):
        if int(token) == IM_START_ID and seen < 2:
            start = index
            seen += 1
    if start is None or seen < 2:
        raise ShapeError(
            f"模板里应有 2 个 <|im_start|>（{IM_START_ID}）✗，实际只数到 {seen} 个 ⇒ "
            f"这段不是套过模板的字符串 ✓（请走 :func:`prompt_of` ✓）")
    if len(ids) > start + 3 and int(ids[start + 1]) == PROMPT_HEAD_TOKEN[0] \
            and int(ids[start + 2]) == PROMPT_HEAD_TOKEN[1]:
        start += 3
    return start


def prompt_of(text: str, *, with_image: bool = False, image_embeds: bool = False) -> str:
    """套模板 ⇒ 送进分词器的**原样字符串** ✓（**唯一**拼模板的地方 ✓，别处别再手拼 ✗）。

    ⚠️ ``with_image=True`` 而 ``image_embeds=False`` ⇒ **当场报错** ✓✗：
    Edit 模板里 ``<|vision_start|><|image_pad|><|vision_end|>`` 那三格的嵌入**必须**由视觉塔给出 ✓，
    拿词表嵌入顶替 ⇒ 装得上、算得出、Edit 能力却没有 ✓✗（最典型的静默错 ✓）。
    所以「有图模板」与「有图像嵌入」必须成对出现 ✓ —— 这就是这条断言的全部意义 ✓。
    """
    pattern = TE_TEMPLATE_EDIT if with_image else TE_TEMPLATE
    if with_image and not image_embeds:
        raise ShapeError(
            f"用了 Edit 模板（里面有 {IMG_PAD_TOKEN} ✓）却没给图像嵌入 ✗ ⇒ "
            f"这三格必须是视觉塔的输出 ✓，本模块只出语言塔 ✗。"
            f"（要么先接视觉塔 ✓，要么走无图模板 ✗）")
    if image_embeds and not with_image:
        raise ShapeError(
            "给了图像嵌入却用无图模板 ✗ ⇒ 模板里没有占位符，嵌入会凭空多出一段 ✗")
    body = "" if text is None else str(text)
    #: ⚠️ 参考实现的两条旁支 ✓（照抄 ✓，否则「同一句话」在两边分词成不同序列 ✓✗）：
    #: ① 已经自带 ``<|im_start|>`` / ``<|start_header_id|>`` 的串**不再套模板** ✓；
    #: ② 空串补一个**空格** ✓（``prevent_empty_text`` ✓）—— 空串会分出 0 个 token ⇒ 主干拿到
    #: ``(1, 0, d)`` 会静默出图 ✓✗（没有文本条件的图 ✓）。
    if body.startswith("<|im_start|>") or body.startswith("<|start_header_id|>"):
        return body
    return pattern.format(body if body else " ")


@dataclass(frozen=True)
class Qwen25TeRuntime:
    """后端要的「文本编码器配置」那股形状 ✓（``output_dim`` / ``max_length`` / ``to_dict`` ✓）。"""

    shape: Qwen25TeShape
    max_length: int = 512
    template: str = TE_TEMPLATE

    @property
    def output_dim(self) -> int:
        """文本塔吐的宽度 ✓（= ``hidden`` = 3584 ✓；须与主干 ``text_dim`` 一致 ✓）。"""
        return int(self.shape.hidden)

    @property
    def vocab_size(self) -> int:
        return int(self.shape.vocab)

    def prompt(self, text: str) -> str:
        return self.template.format("" if text is None else str(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "qwen25-vl-te", "family": "Qwen2.5-VL-7B 语言塔（Qwen-Image 文本塔 ✓）",
            "outputDim": self.output_dim, "vocabSize": self.vocab_size,
            "maxLength": int(self.max_length), "layerIdx": int(self.shape.layer_idx),
            "targetLayer": int(self.shape.target_layer()),
            "depth": int(self.shape.depth), "hidden": int(self.shape.hidden),
            "heads": int(self.shape.heads), "kvHeads": int(self.shape.kv_heads),
            "headDim": int(self.shape.head_dim), "qkvBias": bool(self.shape.qkv_bias),
            "template": self.template,
            "note": "取**最后一层** hidden ✓、不过 final norm ✓、切掉 system 段 ✓（同参考实现 ✓）",
        }


def build_from_shapes(shapes: dict[str, tuple[int, ...]], *,
                      head_dim: int | None = None, layer_idx: int = LAYER_IDX,
                      dtype: Any = None) -> Any:
    """按形状建**文本塔**（不含 ``lm_head`` ✓ 不含视觉塔 ✗）⇒ ``nn.Module`` ✓。"""
    from . import llm as llm_mod  # noqa: PLC0415 —— 局部 ✓：纯函数不该拖 torch ✓

    shape = infer_shape(shapes, head_dim=head_dim)
    if layer_idx != shape.layer_idx:
        shape = replace(shape, layer_idx=layer_idx)
    shape.target_layer()                          # 越界当场炸 ✓（不要等前向 ✗）
    prim = llm_mod.primitives()
    nn, torch, config = prim["nn"], prim["torch"], shape.llm_config()

    class _TeMlp(nn.Module):
        """SwiGLU ✓，参数名照 HF（``gate_proj``/``up_proj``/``down_proj`` ✓、**无** bias ✓）。"""

        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
            self.up_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
            self.down_proj = nn.Linear(shape.ffn, shape.hidden, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down_proj(prim["F"].silu(self.gate_proj(x)) * self.up_proj(x))

    class _TeBlock(nn.Module):
        """pre-norm 一层 ✓（HF Qwen2 顺序：norm → attn → 残差 → norm → mlp → 残差 ✓）。"""

        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = prim["RMSNorm"](shape.hidden, shape.rms_eps)
            self.self_attn = prim["GQAAttention"](config)
            self.post_attention_layernorm = prim["RMSNorm"](shape.hidden, shape.rms_eps)
            self.mlp = _TeMlp()

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            attn, _ = self.self_attn(self.input_layernorm(x), cos, sin)
            x = x + attn
            return x + self.mlp(self.post_attention_layernorm(x))

    class Qwen25TextTowerModule(nn.Module):
        """``model.embed_tokens`` / ``model.layers.N.*`` / ``model.norm`` ✓（``model.norm`` 建而不用 ✓）。

        ⚠️ batch 只支持 **1** ✗：因果掩码是纯 ``triu`` ✓，**没有**「按条长度 pad」那一路 ✓
        ⇒ batch 里两条不同长度会互相看见 ✓✗（静默错 ✗）。一次编码一条 ✓。
        """

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(shape.vocab, shape.hidden)
            self.model.layers = nn.ModuleList([_TeBlock() for _ in range(shape.depth)])
            self.model.norm = prim["RMSNorm"](shape.hidden, shape.rms_eps)
            self.shape = shape

        def forward(self, ids: Any, *, trim: bool = True) -> Any:
            """``ids (1, s)`` ⇒ **第 ``layer_idx`` 层之后的 hidden** ✓ ``(1, s', hidden)`` ✓。

            ``trim=True`` 时按 :func:`trim_start` 切掉模板前两段 ✓（位置编码在**完整序列**上算 ✓
            再切 ✗ —— 参考实现也是先编码后切 ✓；先切后编码会让位置整体前移 ✓✗，
            不报错、但语义偏移 ✓）。
            """
            batch, length = int(ids.shape[0]), int(ids.shape[1])
            if batch != 1:
                raise ShapeError(f"文本塔一次只编码一条 ✗（收到 batch={batch} ✓）")
            if length < 1:
                raise ShapeError("ids 长度为 0 ✗")
            x = self.model.embed_tokens(ids)
            positions = torch.arange(length, device=ids.device, dtype=torch.long)
            cos, sin = prim["rope_freqs"](positions, shape.head_dim, shape.rope_theta)
            target = shape.target_layer()
            hidden = None
            for index, block in enumerate(self.model.layers):
                x = block(x, cos, sin)
                if index == target:
                    hidden = x                     # ⚠️ 不过 model.norm ✓
            if hidden is None:
                raise ShapeError(f"取不到第 {target} 层 ✗")
            if trim:
                start = trim_start(ids[0].tolist())
                hidden = hidden[:, start:]
            return hidden

    previous = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)             # ⚠️ 全局状态 ⇒ 必须 finally 复原 ✓
    try:
        module = Qwen25TextTowerModule()
    finally:
        torch.set_default_dtype(previous)
    return module if dtype is None else module.to(dtype)
