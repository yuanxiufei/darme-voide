"""Z-Image 的**文本塔** = Qwen3-4B dense（因果注意力 ✓、取**倒数第 2 层** hidden ✓、**不做** final norm ✓）。

事实来源（只参考行为、实现自己写 ✓）：

* ``reference/comfy-org/ComfyUI/comfy/text_encoders/z_image.py`` —— ``Qwen3_4BModel(...)`` 的构造口径：
  ``vocab_size=151936, hidden=2560, num_layers=36, num_heads=32, num_kv_heads=8,
  intermediate_size=9728, rms_norm_eps=1e-6, qkv_bias=False`` ✓；并且 ``layer="hidden"`` /
  ``layer_idx=-2`` / ``layer_norm_hidden_state=False`` ✓；``pad_token_id=151643`` ✓；
  词表用 ``qwen25_tokenizer``（与 H3 那份同源 ✓）；提示词模板
  ``<|im_start|>user\\n{}<|im_end|>\\n<|im_start|>assistant\\n`` ✓。
* ``comfy/text_encoders/llama.py`` 的 ``Llama2_``（``BaseQwen3`` 的父类）—— 前向里**有因果掩码**
  （``mask = torch.full((L, L), -inf) + torch.triu(diagonal=1)`` ✓ 再叠逐条的 ``attention_mask``），
  而 ``intermediate_output`` 用的是**负索引**（``len(layers) + (-2)`` ⇒ 0-based 第 34 层**之后** ✓），
  ``final_layer_norm_intermediate=False`` ⇒ 取到的 hidden **不过 final norm** ✓。

⚠️ 这几条一条都不能猜 ✗ —— 猜错全都「装得上、算得出、但结果与提示词无关」✓✗：

* 取错层（``-1`` / ``-3``）差的是**整个语义空间** ✓；
* 多做一次 final norm ⇒ 送进主干的是被 RMSNorm 缩放过的向量 ✗（scale 差一个常数，出图整体偏灰/偏曝 ✓✗）；
* 把因果换成双向更狠 —— 词表里每个 token 都能看到它**后面**的 token ✓✗。

⚠️ 与 :mod:`.llm` 的分工 ✗：算子（RMSNorm / GQA+RoPE / SwiGLU）**只有一份** ✓，从
:func:`llm.primitives` 取 ✓；本模块只负责**拼壳**（参数名与 HF/Qwen 权重键名逐字一致 ✓：
``model.layers.0.self_attn.q_proj.weight`` ✓）和**取层** ✓。**没有** ``lm_head`` ✗ ——
盘上 ``qwen_3_4b.safetensors`` 的 398 张量里根本没有它 ✓（文本塔只出 hidden ✓）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .llm import LlmConfig

__all__ = [
    "PAD_TOKEN_ID",
    "TE_TEMPLATE",
    "Qwen3TeRuntime",
    "Qwen3TeShape",
    "ShapeError",
    "build_from_shapes",
    "infer_shape",
    "tokens_of",
]

#: Qwen 系的**填空** token ✓（``z_image.py::pad_token_id`` ✓ 与 ``qwen25_tokenizer`` 的 ``<|endoftext|>`` 同一个 id ✓）
PAD_TOKEN_ID = 151643

#: 提示词模板 ✓（``z_image.py::ZImageTokenizer`` ✓）—— ``{}`` 里是**用户原话**（不裁剪 ✗ 不加工 ✗）。
#: ⚠️ Z-Image 是**指令跟随**口径 ✓：模型看到的是「user 说了什么 / 现在轮到 assistant」✓，
#: 所以这里的 ``assistant`` 起手不是装饰 ✗ —— 去掉它，出图与提示词的对应会明显变松 ✓✗。
TE_TEMPLATE = "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"

#: 每层必须**全有**的权重后缀 ✓（缺一个都是「不是这一族」✗ 或者「权重没下全」✗ —— 两种都要明说 ✓）
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


class ShapeError(ValueError):
    """权重形状推不出可装的结构 ✓（**明确报错** ✗ 不拿出厂形状硬装 ✓）。"""


@dataclass(frozen=True)
class Qwen3TeShape:
    """Z-Image 文本塔的**全部**结构参数 ✓（每一项都是从权重形状读出来的 ✓，除了 eps/theta ✓）。"""

    vocab: int
    hidden: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    qk_norm: bool = True
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    #: 取哪一层（负索引 ✓ 同参考实现 ✓）：``-2`` = 倒数第二层**之后**的 hidden ✓
    layer_idx: int = -2

    def llm_config(self) -> LlmConfig:
        """摊成 :class:`~.llm.LlmConfig` ✓ —— 算子复用同一份校验（heads 整除 / head_dim 偶数 … ✓）。"""
        return LlmConfig(
            vocab_size=self.vocab, hidden=self.hidden, depth=self.depth,
            heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim,
            ffn=self.ffn, rope_theta=self.rope_theta, rms_eps=self.rms_eps,
            tie_embeddings=False,          # ⚠️ 没有 lm_head ✗ —— 不 tie 只是把话说清 ✓
            qk_norm=self.qk_norm, pad_id=PAD_TOKEN_ID,
        )

    def target_layer(self) -> int:
        """把 ``layer_idx``（可负 ✓）折成 **0-based 层号** ✓；越界 ⇒ :class:`ShapeError` ✗。"""
        index = self.layer_idx if self.layer_idx >= 0 else self.depth + self.layer_idx
        if not 0 <= index < self.depth:
            raise ShapeError(
                f"layer_idx={self.layer_idx} 在这 {self.depth} 层上越界 ✗"
                f"（折算成 {index} ⇒ 必须在 0..{self.depth - 1} ✓）")
        return index


def infer_shape(shapes: dict[str, tuple[int, ...]]) -> Qwen3TeShape:
    """按权重**形状**反推结构 ✓（只读头部的事实 ✗ 不猜层数/宽度 ✓）。

    ⚠️ ``rms_eps`` / ``rope_theta`` **读不出来** ✗ —— safetensors 头里没有这两项 ✓，
    Qwen3-4B 的取值（``1e-6`` / ``1e6`` ✓）只能按发布配置写默认值 ✓；所以它们在
    :class:`Qwen3TeShape` 上是**有默认值的字段** ✓，不假装是从权重读出来的 ✗。
    """
    def size(name: str) -> tuple[int, ...]:
        try:
            return tuple(int(v) for v in shapes[name])
        except KeyError as err:
            raise ShapeError(
                f"缺权重 {name} ✗ —— 这不是 Z-Image 的文本塔 ✓（或者权重没下全 ✗），"
                f"不拿出厂结构硬装 ✓") from err

    vocab, hidden = size("model.embed_tokens.weight")
    if len(size("model.norm.weight")) != 1:
        raise ShapeError("model.norm.weight 必须是一维（RMSNorm 的 weight ✗）")

    # ⚠️ 层号必须**连续** 0..n-1 ✓：中间缺一层就说明权重不全 ✗ —— 而 nn.ModuleList 按序号
    #    命名 ✓，缺一层会让后面所有层的键**整位移** ✓✗（装上去不报错、算出来全错 ✗✗）。
    layer_ids: dict[int, set[str]] = {}
    for name in shapes:
        matched = _LAYER_RE.match(name)
        if matched is None:
            continue
        layer_ids.setdefault(int(matched.group(1)), set()).add(matched.group(2))
    if not layer_ids:
        raise ShapeError("一个 model.layers.N.* 都没有 ✗ —— 这不是 Qwen3 文本塔 ✓")
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
    has_norm = ("model.layers.0.self_attn.q_norm.weight" in shapes
                and "model.layers.0.self_attn.k_norm.weight" in shapes)
    if has_norm:
        head_dim = size("model.layers.0.self_attn.q_norm.weight")[0]
    else:                                   # 没有 QK-norm ⇒ 只能按 Qwen3 的 head_dim=128 还？
        raise ShapeError(                          # 不还 ✗ —— 推不出来就明说 ✓
            "缺 self_attn.q_norm/k_norm ✗ —— 推不出 head_dim ✓（Qwen3 都有它 ✓；"
            "没有它的族不该走这条 ✓）")
    if q_out % head_dim or kv_out % head_dim:
        raise ShapeError(f"q/kv 输出维（{q_out}/{kv_out}）不能被 head_dim（{head_dim}）整除 ✗")
    heads, kv_heads = q_out // head_dim, kv_out // head_dim
    if size("model.layers.0.self_attn.v_proj.weight")[0] != kv_out:
        raise ShapeError("v_proj 与 k_proj 的输出维不一致 ✗（GQA 的 KV 头必须同宽 ✓）")
    if size("model.layers.0.self_attn.o_proj.weight")[1] != q_out:
        raise ShapeError("o_proj 的入维与 q_proj 的输出维不一致 ✗")
    # ⚠️ down_proj 是 ``(hidden, ffn)`` ✓ —— 入维（``[1]``）才是 ffn ✗（``[0]`` 是 hidden ✓）；
    #    写反了会把**每一份真权重**都拒掉 ✓✗（2026-10-31 首次跑就撞在这个下标上 ✓）。
    if size("model.layers.0.mlp.down_proj.weight") != (hidden, ffn):
        raise ShapeError(
            f"mlp.down_proj 的形状 {size('model.layers.0.mlp.down_proj.weight')} ≠ (hidden, ffn)"
            f" = ({hidden}, {ffn}) ✗")
    return Qwen3TeShape(vocab=vocab, hidden=hidden, depth=depth, heads=heads,
                        kv_heads=kv_heads, head_dim=head_dim, ffn=ffn, qk_norm=True)


def tokens_of(text: str) -> str:
    """套上提示词模板 ✓ ⇒ 送进分词器的**原样字符串** ✓（不做 strip ✗ 不改大小写 ✗）。"""
    return TE_TEMPLATE.format(text)


@dataclass(frozen=True)
class Qwen3TeRuntime:
    """后端要的「文本编码器配置」那股形状 ✓（``output_dim`` / ``max_length`` / ``to_dict`` ✓）。

    ⚠️ 为什么不直接把 :class:`Qwen3TeShape` 交出去 ✗：后端里 ``self._textConfig`` 的位置原本放的是
    ``text_encoder.TextEncoderConfig`` ✓（``max_length`` ✓ ``to_dict()`` ✓ ``output_dim`` ✓），
    而 :class:`Qwen3TeShape` **没有**这些字段 ✓✗ ⇒ 直接塞进去，``encode_text`` 会在
    ``config.max_length`` 上 ``AttributeError`` ✓（报得还像"编码器挂了"✓✗）。这层壳只做**翻译** ✓，
    结构参数仍是同一个 ``shape`` 对象 ✓（不复制第二份真源 ✓）。
    """

    shape: Qwen3TeShape
    max_length: int = 512
    template: str = TE_TEMPLATE

    @property
    def output_dim(self) -> int:
        """文本塔吐的宽度 ✓（= ``hidden`` ✓；须与主干 ``text_dim`` 一致 ✓ → ``_check_text_encoder`` ✓）。"""
        return int(self.shape.hidden)

    @property
    def vocab_size(self) -> int:
        return int(self.shape.vocab)

    def prompt(self, text: str) -> str:
        """套模板 ✓（**唯一**加模板的地方 ✓ —— 后端不要在别处再拼一次 ✓✗）。"""
        return self.template.format("" if text is None else str(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "qwen3-te", "family": "Qwen3-4B dense（Z-Image 文本塔 ✓）",
            "outputDim": self.output_dim, "vocabSize": self.vocab_size,
            "maxLength": int(self.max_length), "layerIdx": int(self.shape.layer_idx),
            "targetLayer": int(self.shape.target_layer()),
            "depth": int(self.shape.depth), "hidden": int(self.shape.hidden),
            "heads": int(self.shape.heads), "kvHeads": int(self.shape.kv_heads),
            "headDim": int(self.shape.head_dim),
            "template": self.template,
            "note": "取倒数第 2 层 hidden ✓、**不过 final norm** ✓（同参考实现 ✓）",
        }


def build_from_shapes(shapes: dict[str, tuple[int, ...]], *,
                      layer_idx: int = -2, dtype: Any = None) -> Any:
    """按形状建**文本塔**（不含 ``lm_head`` ✓）⇒ ``nn.Module`` ✓。

    ``dtype`` 走 ``set_default_dtype`` ✓：7.6 GB 的 bf16 权重若先建 fp32 再转 ⇒ 峰值 15 GB ✗✗
    （与 :func:`~.llm.build_llm` 同一条手法 ✓）。默认 ``None`` = 不碰全局默认（fp32 ✓，CPU 自检用 ✓）。
    """
    from . import llm as llm_mod  # noqa: PLC0415 —— 局部 ✓：本模块的纯函数（infer_shape/tokens_of）不该拖 torch ✓

    shape = infer_shape(shapes)
    if layer_idx != shape.layer_idx:
        shape = Qwen3TeShape(**{**shape.__dict__, "layer_idx": layer_idx})
    shape.target_layer()                          # 越界当场炸 ✓（不要等前向 ✗）
    prim = llm_mod.primitives()
    nn, torch, config = prim["nn"], prim["torch"], shape.llm_config()

    class _TeMlp(nn.Module):
        """SwiGLU ✓，但**参数名照 HF**（``gate_proj``/``up_proj``/``down_proj`` ✓）：

        ⚠️ 不能直接复用 ``llm.SwiGLU`` ✗ —— 它的参数叫 ``gate``/``up``/``down`` ✓，
        与盘上 ``model.layers.0.mlp.gate_proj.weight`` ✗ 对不上 ⇒ 装载会报 108 个缺键 ✗。
        算式一模一样 ✓（``down(silu(gate) * up)`` ✓）。
        """

        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
            self.up_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
            self.down_proj = nn.Linear(shape.ffn, shape.hidden, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down_proj(prim["F"].silu(self.gate_proj(x)) * self.up_proj(x))

    class _TeBlock(nn.Module):
        """pre-norm 一层 ✓（HF Qwen3 的顺序：norm → attn → 残差 → norm → mlp → 残差 ✓）。"""

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

    class Qwen3TextTowerModule(nn.Module):
        """文本塔本体 ✓：``model.embed_tokens`` / ``model.layers.N.*`` / ``model.norm`` ✓。

        ⚠️ ``model.norm`` **建了但不用** ✗：盘上有这份权重 ✓（装载要对齐键 ✓），
        而参考实现取的是**未过 final norm** 的中间层 hidden ✓（``layer_norm_hidden_state=False`` ✓）。
        建它不代表用它 ✓ —— 用不用由 :meth:`forward` 的 ``layer_idx`` 口径决定 ✓。
        """

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(shape.vocab, shape.hidden)
            self.model.layers = nn.ModuleList([_TeBlock() for _ in range(shape.depth)])
            self.model.norm = prim["RMSNorm"](shape.hidden, shape.rms_eps)
            self.shape = shape

        def forward(self, ids: Any) -> Any:
            """``ids`` ⇒ **第 ``layer_idx`` 层之后的 hidden** ✓ ``(b, s, hidden)`` ✓。

            ⚠️ batch 只支持 **1** ✗：``GQAAttention`` 的掩码是纯因果（``triu`` ✓），
            **没有**「按条长度 pad」的那一路 ✓ —— batch 里两条不同长度会被互相看见 ✓✗
            （静默错 ✗）。⇒ 一次编码一条 ✓；多条由调用方逐条走 ✓（编码很快，别拿正确性换 ✗）。
            """
            batch, length = int(ids.shape[0]), int(ids.shape[1])
            if batch != 1:
                raise ShapeError(f"文本塔一次只编码一条 ✗（收到 batch={batch} ✓）—— 逐条调用 ✓")
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
                    hidden = x                     # ⚠️ 不过 final norm ✓（参考实现同一口径 ✓）
            if hidden is None:                     # 只有 depth==0 才可能走到（construct 已挡 ✓）
                raise ShapeError(f"取不到第 {target} 层 ✗")
            return hidden

    previous = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)             # ⚠️ 全局状态 ⇒ 必须 finally 复原 ✓
    try:
        module = Qwen3TextTowerModule()
    finally:
        torch.set_default_dtype(previous)
    return module if dtype is None else module.to(dtype)
