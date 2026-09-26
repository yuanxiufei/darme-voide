"""**CLIP 文本编码器**（SDXL 的「双文本塔」这一半 ✓）—— 自研、纯 torch、零外部依赖。

## 为什么有这一层

`engine/sdxl.py` 的 UNet 只吃**条件张量** ✗ —— 从「提示词」到「条件张量」这段
（CLIP-L + OpenCLIP-bigG 两个文本塔 + 拼接）在引擎里一直是空的 ✓，本模块补它
（`image_generation` 的 `provider=engine` 那一路要它 ✓）。

## 结构事实（逐条有出处，不猜 ✓）

**两塔的数值**取自 `ComfyUI/comfy/sd1_clip_config.json` / `clip_config_bigg.json`
（配置数值 = 模型接口 ✓，与 HF `openai/clip-vit-large-patch14` 一致 ✓），并已用**本机真权重头**复核 ✓：

| | CLIP-L | OpenCLIP-bigG |
|---|---|---|
| hidden_size | 768 | 1280 |
| num_hidden_layers | 12 | 32 |
| num_attention_heads | 12 | 20 |
| intermediate_size | 3072 | 5120 |
| hidden_act | ``quick_gelu`` | ``gelu`` |
| vocab_size | 49408 | 49408 |
| max_position_embeddings | 77 | 77 |
| layer_norm_eps | 1e-5 | 1e-5 |
| text_projection | **无** | 1280（**有** ✓） |
| pad_token | 49407（= eos ✓） | **0** ✓ |

**块结构**（出处 `ComfyUI/comfy/clip_model.py` —— 只读事实，未抄代码 ✗）：
`layer_norm1 → self_attn(q/k/v/out_proj **全带偏置**) → 残差` + `layer_norm2 → mlp(fc1 → act → fc2) → 残差`；
嵌入 = `token_embedding + position_embedding`（**无** scale ✓）；`final_layer_norm` 在全部层**之后** ✓；
`pooled` = **EOS 位置**那一行 ✓（再过 `text_projection` 才是 SDXL 用的 pooled ✓）。

⚠️ **两塔的 pad 口径不一样** ✗：CLIP-L 用 `pad_with_end=True` ⇒ 补 **eos(49407)**；
bigG 用 `pad_with_end=False` ⇒ 补 **0** ✓（出处 `ComfyUI/comfy/sdxl_clip.py` ✓）。

**SDXL 的两塔用法**（出处同上 ✓）：
* 两塔都取 **倒数第二层**（`layer_idx=-2` ✓）且**不做** `final_layer_norm`（`layer_norm_hidden_state=False` ✓）；
* `context = cat([l_out, g_out], dim=-1)`，**各自先截到较短的那个长度** ✓ ⇒ 768 + 1280 = **2048** ✓；
* `pooled` 只取 **bigG** 的（经 `text_projection` ✓ ⇒ 1280 ✓ = SDXL UNet `label_emb` 吃的那半 ✓）；
* 两塔都**不做** padding 掩码（只做因果掩码 ✓，出处 `sd1_clip.py` 的 `enable_attention_masks=False` 默认 ✓）。

## ⚠️ 检查点命名：同一个模型有两种叫法（**本机真权重实测** ✓）

`sd_xl_base_1.0.safetensors`（只读 safetensors 头量出 ✓：2515 键 = conditioner 587 + first_stage_model 248 + model 1680）：

* **CLIP-L** 是 **HF 命名** ✓：`conditioner.embedders.0.transformer.text_model.*`（197 键 ✓，含一个
  `embeddings.position_ids` 缓冲 ✓ —— 那是常量，本模块**不注册**它 ✗，会在装载时如实回报为「未用键」✓）；
* **bigG** 是 **OpenCLIP 命名** ✓：`conditioner.embedders.1.model.transformer.resblocks.N.attn.in_proj_weight`
  这类（390 键 ✓，另有 `model.logit_scale` ✓ —— 那是 CLIPModel 的对比学习标量，文本塔**不用** ✗）。

⇒ 换算口径（出处 `ComfyUI/comfy/supported_models.py` 的 SDXL 段 + `comfy/utils.py` ✓）：
* L：只**剥前缀** ✓（本来就是 HF 命名 ✓）；
* G：`attn.in_proj_weight/bias` **按 hidden_size 三等分成 q/k/v_proj** ✓（顺序 q,k,v ✓）、
  `ln_1→layer_norm1`、`ln_2→layer_norm2`、`mlp.c_fc→mlp.fc1`、`mlp.c_proj→mlp.fc2`、
  `attn.out_proj→self_attn.out_proj`、`positional_embedding→embeddings.position_embedding.weight`、
  `token_embedding.weight→embeddings.token_embedding.weight`、`ln_final→final_layer_norm` ✓；
* ⚠️⚠️ **`text_projection` 必须转置** ✗✗：OpenCLIP 的用法是 `x @ W` ✓、HF 的 `nn.Linear` 是 `x @ W.T` ✓
  ⇒ 不转置就是**看着对的错值**（形状一样、数值全错 ✓✗）—— 出处 `comfy/utils.py` 的
  `clip_text_transformers_convert`（对裸 `text_projection` 做 `transpose(0, 1)` ✓）。

## 不猜·边界

* 只认**这两种命名** ✓：别的前缀/结构 ⇒ **报错** ✗（不猜着装 ✗）；
* `text_projection` 与 `logit_scale` **按名字分流** ✓：前者转置装载 ✓、后者如实跳过后回报 ✓；
* **tokenizer 不内置** ✓（词表是权重的一部分 ✗ —— 与 `engine/text_encoder.py` 同一条纪律 ✓）；
  本模块只给「ids → 定长 77」的**组装口径**（`:func:`sdxl_token_ids`` ✓，BOS/EOS/pad 是接口 ✓ 不是词表 ✓）；
* 只做**架构 + 前向 + 装载** ✓；采样循环 / VAE 解码不在本模块（引擎另有 ✓）。

## 验证口径

`tests/engine_clip_text_test.py`：缩小版走**同一条前向** ✓（不下载真权重就能验 ✓）、
因果性/池化位置/倒数第二层/投影转置都有**能自己算出来**的判据 ✓；
本机有真权重时**逐键逐形状**核对（只读头 ✓ 不载张量 ✓），找不到就**显式 SKIP** ✓。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

try:  # pragma: no cover - 环境相关
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False

__all__ = [
    "CLIP_BOS_TOKEN_ID",
    "CLIP_EOS_TOKEN_ID",
    "CLIP_G",
    "CLIP_L",
    "CLIP_MAX_LENGTH",
    "SDXL_CLIP_G_KEY_PREFIX",
    "SDXL_CLIP_L_KEY_PREFIX",
    "SDXL_CONTEXT_DIM",
    "ClipTextConfig",
    "ClipTextError",
    "ClipTextOutput",
    "activation_names",
    "build_clip_text_model",
    "clip_text_key_names",
    "encode_sdxl_conditioning",
    "has_torch",
    "hf_clip_state_dict",
    "load_clip_text_state_dict",
    "open_clip_state_dict",
    "sdxl_token_ids",
]

#: CLIP 的 BOS / EOS（两个塔**同值** ✓ —— 那是 tokenizer 接口，不是词表内容 ✓）。
CLIP_BOS_TOKEN_ID = 49406
CLIP_EOS_TOKEN_ID = 49407

#: 两塔的上下文长度（位置嵌入只有 77 行 ✓）。
CLIP_MAX_LENGTH = 77

#: SDXL 条件张量的通道数：768 + 1280 ✓（= UNet 交叉注意力的 `context_dim` ✓）。
SDXL_CONTEXT_DIM = 2048

#: 官方 SDXL 检查点里两塔的键前缀（**真权重实测** ✓）。
SDXL_CLIP_L_KEY_PREFIX = "conditioner.embedders.0."
SDXL_CLIP_G_KEY_PREFIX = "conditioner.embedders.1."

#: 激活函数（口径见模块头 ✓ —— ``quick_gelu`` 是 CLIP 的 1.702 近似 sigmoid ✓）。
_QUICK_GELU_SLOPE = 1.702

#: OpenCLIP 的 ``resblocks`` 键名 → HF 的 `encoder.layers` 键名 ✓（出处 `comfy/utils.py` ✓）。
_OPEN_CLIP_BLOCK_RENAMES = {
    "ln_1": "layer_norm1",
    "ln_2": "layer_norm2",
    "mlp.c_fc": "mlp.fc1",
    "mlp.c_proj": "mlp.fc2",
    "attn.out_proj": "self_attn.out_proj",
}

#: OpenCLIP 顶层键名 → HF 的 ``text_model`` 内层键名 ✓（出处 `comfy/utils.py` ✓）。
_OPEN_CLIP_ROOT_RENAMES = {
    "positional_embedding": "embeddings.position_embedding.weight",
    "token_embedding.weight": "embeddings.token_embedding.weight",
    "ln_final.weight": "final_layer_norm.weight",
    "ln_final.bias": "final_layer_norm.bias",
}

#: 本模块的模块名（HF 命名 ✓）—— 装载时按它拼键 ✓。
_TEXT_MODEL_PREFIX = "transformer.text_model."


def _rename_open_clip_block(rest: str) -> str | None:
    """块内键改名 ✓ —— **按前缀匹配** ✓（真权重里是 `attn.out_proj.bias` 这类带后缀的键 ✗，
    精确匹配 `.get(rest)` 会漏掉它们 ✓ —— 参考实现对整个键名做替换 ⇒ 后缀自然保留 ✓）。"""
    for source, target in _OPEN_CLIP_BLOCK_RENAMES.items():
        if rest == source or rest.startswith(source + "."):
            return target + rest[len(source):]
    return None

_BLOCK_RE = re.compile(r"^transformer\.resblocks\.(\d+)\.(.+)$")


class ClipTextError(RuntimeError):
    """CLIP 文本塔的配置/装载/前向出错（**明确报错**，不静默兜底 ✗）。"""


def has_torch() -> bool:
    """本进程里 torch 是否可用（无 torch 的环境要能 import 本模块 ✓）。"""
    return _HAS_TORCH


def activation_names() -> tuple[str, ...]:
    """支持的激活函数名（配置里只认这几个 ✓，认不出就报错 ✗）。"""
    return ("quick_gelu", "gelu", "gelu_pytorch_tanh")


def _activation(name: str):
    if not _HAS_TORCH:  # pragma: no cover - 由调用方先查
        raise ClipTextError("没装 torch ✗ ⇒ 取不到激活函数 ✓")
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(_QUICK_GELU_SLOPE * x)
    if name == "gelu":
        return F.gelu
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    raise ClipTextError(
        f"认不出的激活函数 {name!r} ✗ ⇒ 只认 {activation_names()} ✓（不静默当 gelu ✗）")


@dataclass(frozen=True)
class ClipTextConfig:
    """一个文本塔的全部架构参数 —— **逐个显式** ✓（默认是 CLIP-L ✓，改任一字段都会改变键集/形状 ✓）。

    ⚠️ ``projection_dim=0`` 表示**没有** `text_projection` ✓（CLIP-L 就是这样 ✓）；
    ⚠️ ``pad_token`` 两塔**不同** ✗（L=49407、G=0 ✓，见模块头 ✓）；
    ⚠️ ``load_prefix`` 是「官方 SDXL 检查点里这个塔的键前缀」✓（装载时用来剥 ✓）。
    """

    name: str = "clip_l"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 49408
    max_position_embeddings: int = CLIP_MAX_LENGTH
    layer_norm_eps: float = 1e-5
    hidden_act: str = "quick_gelu"
    projection_dim: int = 0
    pad_token: int = CLIP_EOS_TOKEN_ID
    load_prefix: str = SDXL_CLIP_L_KEY_PREFIX

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ClipTextError("文本塔必须有名字 ✗（空名字会让日志/报错说不清是哪个塔 ✓）")
        for field in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                      "intermediate_size", "vocab_size", "max_position_embeddings"):
            value = getattr(self, field)
            if not isinstance(value, int) or value <= 0:
                raise ClipTextError(
                    f"{self.name}: {field} 必须是正整数 ✗（拿到 {value!r} ✓）")
        if self.projection_dim < 0:
            raise ClipTextError(
                f"{self.name}: projection_dim 只能是 0（无投影 ✓）或正数 ✗（拿到 {self.projection_dim!r} ✓）")
        if self.hidden_size % self.num_attention_heads:
            raise ClipTextError(
                f"{self.name}: hidden_size({self.hidden_size}) 必须被 "
                f"num_attention_heads({self.num_attention_heads}) 整除 ✗")
        if self.hidden_act not in activation_names():
            raise ClipTextError(
                f"{self.name}: 认不出的 hidden_act {self.hidden_act!r} ✗ ⇒ 只认 {activation_names()} ✓")
        if not 0 <= int(self.pad_token) < self.vocab_size:
            raise ClipTextError(
                f"{self.name}: pad_token({self.pad_token}) 必须落在 [0, vocab_size) 内 ✗")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def has_projection(self) -> bool:
        return self.projection_dim > 0

    def describe(self) -> dict[str, Any]:
        """如实自述（给别的模块/前端读 ✓，不让调用方猜 ✗）。"""
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "projection_dim": self.projection_dim,
            "pad_token": self.pad_token,
            "load_prefix": self.load_prefix,
        }


#: SDXL 的 **CLIP-L** 塔（HF 命名 ✓、无投影 ✓、pad = eos ✓）。
CLIP_L = ClipTextConfig(
    name="clip_l",
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="quick_gelu",
    projection_dim=0,
    pad_token=CLIP_EOS_TOKEN_ID,
    load_prefix=SDXL_CLIP_L_KEY_PREFIX,
)

#: SDXL 的 **OpenCLIP-bigG** 塔（OpenCLIP 命名 ✓、有 1280 投影 ✓、pad = 0 ✓）。
CLIP_G = ClipTextConfig(
    name="clip_g",
    hidden_size=1280,
    num_hidden_layers=32,
    num_attention_heads=20,
    intermediate_size=5120,
    hidden_act="gelu",
    projection_dim=1280,
    pad_token=0,
    load_prefix=SDXL_CLIP_G_KEY_PREFIX,
)

#: 无 torch 时用 ``object`` 当基类 ✓（本模块要能**无 torch 也 import** ✓ 同 `engine/sdxl.py` ✓）。
_ModuleBase = nn.Module if _HAS_TORCH else object


def _causal_mask(length: int, dtype: Any, device: Any):
    """因果掩码（上三角 ``-inf`` ✓）—— 出处 `comfy/clip_model.py`（``triu_(1)`` ✓）。"""
    mask = torch.full((length, length), float("-inf"), dtype=dtype, device=device)
    return torch.triu(mask, diagonal=1)


def _resolve_layer_index(layer_idx: int, num_layers: int, name: str) -> int:
    """把（可负的）层号解析成下标 ✓ —— 越界**报错** ✗（不夹到范围内 ✗）。"""
    if not isinstance(layer_idx, int) or isinstance(layer_idx, bool):
        raise ClipTextError(f"{name}: layer_idx 必须是整数 ✗（拿到 {layer_idx!r} ✓）")
    index = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    if not 0 <= index < num_layers:
        raise ClipTextError(
            f"{name}: layer_idx({layer_idx}) 越界 ✗ ⇒ 该塔只有 {num_layers} 层"
            f"（合法范围 -{num_layers}…{num_layers - 1} ✓）")
    return index


def _pool_at_eos(hidden: Any, input_ids: Any, eos_token_id: int):
    """取**第一个** EOS 位置那一行 ✓ —— 找不到就**报错** ✗（不静默取最后一个 ✗）。"""
    matches = input_ids == eos_token_id
    if not bool(matches.any(dim=-1).all()):
        raise ClipTextError(
            f"输入里没有 eos({eos_token_id}) ✗ ⇒ 取不了 pooled ✓"
            "（出处口径就是「EOS 那一行」✓，本模块不拿别的位置凑 ✗）")
    first = matches.int().argmax(dim=-1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, first]


class _ClipAttention(_ModuleBase):
    """多头自注意力（q/k/v/out_proj **全带偏置** ✓、因果掩码 ✓、**不做** padding 掩码 ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = config.head_dim ** -0.5

    def forward(self, x: Any, causal_mask: Any = None) -> Any:
        batch, length, _ = x.shape
        heads, head_dim = self.config.num_attention_heads, self.config.head_dim
        q = self.q_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
        if causal_mask is None:
            causal_mask = _causal_mask(length, x.dtype, x.device)
        weights = torch.softmax(q @ k.transpose(-1, -2) * self.scale + causal_mask, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, heads * head_dim)
        return self.out_proj(out)


class _ClipMlp(_ModuleBase):
    """前馈（``fc1 → act → fc2`` ✓；激活按配置取 ✓ 出处 `comfy/clip_model.py` ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = _activation(config.hidden_act)

    def forward(self, x: Any) -> Any:
        return self.fc2(self.activation(self.fc1(x)))


class _ClipLayer(_ModuleBase):
    """一层：``x += attn(ln1(x))``；``x += mlp(ln2(x))`` ✓（pre-norm + 残差 ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = _ClipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _ClipMlp(config)

    def forward(self, x: Any, causal_mask: Any = None) -> Any:
        x = x + self.self_attn(self.layer_norm1(x), causal_mask)
        return x + self.mlp(self.layer_norm2(x))


class _ClipEncoder(_ModuleBase):
    """``encoder.layers.N`` ✓（键名与真权重一致 ⇒ 装载能逐键对上 ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ClipLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: Any, causal_mask: Any = None) -> tuple[Any, list[Any]]:
        outputs: list[Any] = []
        for layer in self.layers:
            x = layer(x, causal_mask)
            outputs.append(x)
        return x, outputs


class _ClipEmbeddings(_ModuleBase):
    """``token_embedding + position_embedding`` ✓（**无** scale ✓；按 ``arange(长度)`` 取位置 ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids: Any) -> Any:
        length = int(input_ids.shape[1])
        if length > self.config.max_position_embeddings:
            raise ClipTextError(
                f"{self.config.name}: 序列长 {length} 超过位置嵌入上限 "
                f"{self.config.max_position_embeddings} ✗（调用方须先裁到 77 ✓）")
        positions = torch.arange(length, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)


class _ClipTextCore(_ModuleBase):
    """``transformer.text_model.*`` ✓（嵌入 + 编码器 + 末尾 LayerNorm ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = _ClipEmbeddings(config)
        self.encoder = _ClipEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: Any,
        *,
        layer_idx: int = -1,
        layer_norm_hidden_state: bool = True,
        return_all_hidden_states: bool = False,
        eos_token_id: int = CLIP_EOS_TOKEN_ID,
    ) -> dict[str, Any]:
        if input_ids.dim() != 2:
            raise ClipTextError(
                f"{self.config.name}: 输入必须是 (批, 长度) 的 int id 张量 ✗"
                f"（拿到形状 {tuple(input_ids.shape)} ✓）")
        length = int(input_ids.shape[1])
        if length <= 0:
            raise ClipTextError(f"{self.config.name}: 输入长度为 0 ✗ ⇒ 没有可编码的 token ✓")
        embeds = self.embeddings(input_ids)
        mask = _causal_mask(length, embeds.dtype, embeds.device)
        encoded, layer_outputs = self.encoder(embeds, mask)
        index = _resolve_layer_index(layer_idx, len(layer_outputs), self.config.name)
        # ⚠️ 口径（出处 `comfy/clip_model.py` ✓）：**最后一层输出过 `final_layer_norm` 才是 x** ✓；
        #    中间层（intermediate）是否过 norm 由调用方开关决定 ✓（SDXL 两塔都是 **False** ✓）。
        last_hidden_state = self.final_layer_norm(encoded)
        selected = layer_outputs[index]
        if layer_norm_hidden_state:
            selected = self.final_layer_norm(selected)
        pooled = _pool_at_eos(last_hidden_state, input_ids, eos_token_id)
        return {
            "last_hidden_state": last_hidden_state,
            "selected": selected,
            "pooled": pooled,
            "layer_index": index,
            "hidden_states": ((embeds,) + tuple(layer_outputs)) if return_all_hidden_states else (),
        }


class ClipTextTransformer(_ModuleBase):
    """键前缀 ``transformer.`` ✓（与真权重 `…transformer.text_model…` 对齐 ✓）。"""

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.config = config
        self.text_model = _ClipTextCore(config)


@dataclass(frozen=True)
class ClipTextOutput:
    """一次前向的全部产物（**字段名与含义一一对应** ✓，不让调用方猜 ✓）。"""

    #: 最后一层输出**过** ``final_layer_norm`` ✓（= 参考实现的 ``x`` ✓；HF 的 last_hidden_state 同义 ✓）。
    last_hidden_state: Any
    #: 选定层输出 ✓（``layer_norm_hidden_state=True`` 时才过 norm ✓ —— SDXL 用这个 ✓）。
    selected: Any
    #: ``last_hidden_state`` 里 **EOS 位置**那一行 ✓（未投影 ✓）。
    pooled: Any
    #: ``text_projection(pooled)`` ✓；本塔无投影时为 ``None`` ✓（**不假装有** ✗）。
    projected_pooled: Any
    #: 每层输出（含嵌入输出，长度 = 层数 + 1 ✓）；没要求时是空元组 ✓（省内存 ✓）。
    hidden_states: tuple[Any, ...]
    #: 实际选中的层下标（已归一成正数 ✓）。
    layer_index: int

    def describe(self) -> dict[str, Any]:
        return {
            "last_hidden_state": tuple(self.last_hidden_state.shape),
            "selected": tuple(self.selected.shape),
            "pooled": tuple(self.pooled.shape),
            "projected_pooled": (tuple(self.projected_pooled.shape)
                                 if self.projected_pooled is not None else None),
            "hidden_states": len(self.hidden_states),
            "layer_index": self.layer_index,
        }


class ClipTextModel(_ModuleBase):
    """一个完整的文本塔 ✓（``transformer`` + 可选的 ``text_projection`` ✓）。

    ⚠️ 前向返回 :class:`ClipTextOutput` ✓——`selected` 给「倒数第二层」这类用法 ✓、
    `pooled`/`projected_pooled` 给「池化条件」用法 ✓（SDXL 只吃 bigG 的 `projected_pooled` ✓）。
    """

    def __init__(self, config: ClipTextConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = ClipTextTransformer(config)
        if config.has_projection:
            self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        else:
            self.text_projection = None

    def forward(
        self,
        input_ids: Any,
        *,
        layer_idx: int = -1,
        layer_norm_hidden_state: bool = True,
        return_all_hidden_states: bool = False,
        eos_token_id: int = CLIP_EOS_TOKEN_ID,
    ) -> ClipTextOutput:
        out = self.transformer.text_model(
            input_ids,
            layer_idx=layer_idx,
            layer_norm_hidden_state=layer_norm_hidden_state,
            return_all_hidden_states=return_all_hidden_states,
            eos_token_id=eos_token_id,
        )
        pooled = out["pooled"]
        projected = self.text_projection(pooled) if self.text_projection is not None else None
        return ClipTextOutput(
            last_hidden_state=out["last_hidden_state"],
            selected=out["selected"],
            pooled=pooled,
            projected_pooled=projected,
            hidden_states=out["hidden_states"],
            layer_index=int(out["layer_index"]),
        )


#: 允许「出现在检查点里但**不装载**」的键 ✓ —— 都要**如实回报** ✗（不静默吞掉 ✓）。
_IGNORED_CHECKPOINT_KEYS = frozenset({
    # 常量缓冲（真权重里是 F16 的 [1, 77] ✓）—— 本模块前向按 arange 取位置 ✓，不注册它 ✓。
    "transformer.text_model.embeddings.position_ids",
    # CLIPModel 的对比学习标量 ✓ —— 文本塔前向用不到 ✓。
    "logit_scale",
})


def build_clip_text_model(
    config: ClipTextConfig = CLIP_L,
    *,
    device: Any = None,
    dtype: Any = None,
) -> "Any":
    """按配置建一个文本塔 ✓（默认 CLIP-L ✓）。

    ⚠️ 没装 torch ⇒ **报错** ✗（本仓不静默降级 ✗，同 `engine/sdxl.py` ✓）。
    """
    if not _HAS_TORCH:
        raise ClipTextError(
            "没装 torch ✗ ⇒ 建不了 CLIP 文本塔 ✓（要纯结构信息请用 `ClipTextConfig.describe()` ✓）")
    if not isinstance(config, ClipTextConfig):
        raise ClipTextError(f"config 得是 ClipTextConfig ✗（拿到 {type(config).__name__} ✓）")
    model = ClipTextModel(config)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def clip_text_key_names(config: ClipTextConfig = CLIP_L) -> tuple[str, ...]:
    """算出一个塔**应该有的全部键名** ✓ —— 不建模型、不载张量就能核对 ✓（体检/测试用 ✓）。"""
    if not isinstance(config, ClipTextConfig):
        raise ClipTextError(f"config 得是 ClipTextConfig ✗（拿到 {type(config).__name__} ✓）")
    names = [
        f"{_TEXT_MODEL_PREFIX}embeddings.token_embedding.weight",
        f"{_TEXT_MODEL_PREFIX}embeddings.position_embedding.weight",
    ]
    for index in range(config.num_hidden_layers):
        prefix = f"{_TEXT_MODEL_PREFIX}encoder.layers.{index}."
        for block in ("layer_norm1", "layer_norm2"):
            names += [f"{prefix}{block}.weight", f"{prefix}{block}.bias"]
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            names += [f"{prefix}self_attn.{proj}.weight", f"{prefix}self_attn.{proj}.bias"]
        for fc in ("fc1", "fc2"):
            names += [f"{prefix}mlp.{fc}.weight", f"{prefix}mlp.{fc}.bias"]
    names += [
        f"{_TEXT_MODEL_PREFIX}final_layer_norm.weight",
        f"{_TEXT_MODEL_PREFIX}final_layer_norm.bias",
    ]
    if config.has_projection:
        names.append("text_projection.weight")
    return tuple(names)


def hf_clip_state_dict(state_dict: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    """**只剥前缀** ✓（HF 命名 = 本模块命名 ✓）—— 剥完为空 ⇒ **报错** ✗（前缀不对 ✓）。"""
    text = str(prefix)
    if not text:
        raise ClipTextError("剥前缀得给一个**非空**前缀 ✗（要原样传入请直接传 dict ✓）")
    out = {key[len(text):]: value for key, value in state_dict.items() if key.startswith(text)}
    if not out:
        raise ClipTextError(f"没有以 {text!r} 开头的键 ✗ ⇒ 前缀不对（或这不是那个塔的权重 ✓）")
    return out


def open_clip_state_dict(
    state_dict: Mapping[str, Any],
    config: ClipTextConfig = CLIP_G,
    *,
    source_prefix: str = "model.",
    target_prefix: str = _TEXT_MODEL_PREFIX,
) -> dict[str, Any]:
    """把 **OpenCLIP 命名**换算成本模块（HF ✓）命名 —— 口径逐条见模块头 ✓。

    ⚠️ 三处**不能少** ✗：`in_proj_*` 三等分成 q/k/v ✓、`ln_1`/`mlp.c_fc` 等改名 ✓、
    **`text_projection` 转置** ✓（OpenCLIP 是 `x @ W`、HF 是 `x @ W.T` ✗）。
    认不出的键 ⇒ **报错** ✗（不猜着装 ✗）。
    """
    if not isinstance(config, ClipTextConfig):
        raise ClipTextError(f"config 得是 ClipTextConfig ✗（拿到 {type(config).__name__} ✓）")
    stripped = dict(state_dict)
    if source_prefix:
        stripped = {k[len(source_prefix):]: v
                    for k, v in stripped.items() if k.startswith(source_prefix)}
    if not stripped:
        raise ClipTextError(f"没有以 {source_prefix!r} 开头的键 ✗ ⇒ 这批不是 OpenCLIP 命名的权重 ✓")

    hidden = config.hidden_size
    out: dict[str, Any] = {}
    for key, value in stripped.items():
        block = _BLOCK_RE.match(key)
        if block is not None:
            index = int(block.group(1))
            rest = block.group(2)
            if index >= config.num_hidden_layers:
                raise ClipTextError(
                    f"检查点里有 resblocks.{index} ✗ ⇒ {config.name} 只有 "
                    f"{config.num_hidden_layers} 层 ✓（配置与权重不是同一套 ✗）")
            if rest.startswith("attn.in_proj_"):
                suffix = rest[len("attn.in_proj_"):]
                if suffix not in ("weight", "bias"):
                    raise ClipTextError(f"认不出的 OpenCLIP 键 {key!r} ✗")
                got = int(value.shape[0])
                if got != 3 * hidden:
                    raise ClipTextError(
                        f"{key} 的第 0 维是 {got} ✗ ⇒ 三等分要求 {3 * hidden}"
                        f"（3 × hidden_size ✓）⇒ 配置与权重不是同一套 ✗")
                # ⚠️ 顺序固定 q, k, v ✓（出处 `comfy/utils.py` 的 ``[q_proj, k_proj, v_proj]`` ✓）。
                for slot, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                    out[f"{target_prefix}encoder.layers.{index}.self_attn.{proj}.{suffix}"] = \
                        value[hidden * slot: hidden * (slot + 1)]
                continue
            renamed = _rename_open_clip_block(rest)
            if renamed is None:
                raise ClipTextError(
                    f"认不出的 OpenCLIP 键 {key!r} ✗ ⇒ 本模块不猜着装 ✗"
                    "（换算表见 `_OPEN_CLIP_BLOCK_RENAMES` / `_OPEN_CLIP_ROOT_RENAMES` ✓）")
            out[f"{target_prefix}encoder.layers.{index}.{renamed}"] = value
            continue
        if key == "text_projection":
            if not config.has_projection:
                raise ClipTextError(
                    f"{config.name} 配的是**没有** text_projection ✗ ⇒ 权重里却有它 ✓（塔选错了 ✗）")
            # ⚠️ 唯一一处**方向**换算 ✗✗：不转置 = 形状对、数值全错 ✓✗（见模块头 ✓）。
            out["text_projection.weight"] = value.transpose(0, 1).contiguous()
            continue
        if key == "text_projection.weight":
            out[key] = value  # 已经是 HF 语义 ✓ ⇒ **不**转置 ✓
            continue
        renamed = _OPEN_CLIP_ROOT_RENAMES.get(key)
        if renamed is not None:
            out[f"{target_prefix}{renamed}"] = value
            continue
        out[key] = value  # 例如 ``logit_scale`` ✓ —— 原样留着，由装载**如实回报** ✓。
    return out


def _detect_clip_prefix(state_dict: Mapping[str, Any], config: ClipTextConfig) -> str:
    """认出这批权重的前缀 ✓ —— 三种叫法都认，认不出就**报错** ✗（不猜 ✗）。"""
    for candidate in (config.load_prefix, f"{config.name}."):
        if any(key.startswith(candidate) for key in state_dict):
            return candidate
    if any(key.startswith(_TEXT_MODEL_PREFIX) or key.startswith("text_model.")
           or key.startswith("transformer.") or key.startswith("text_projection")
           for key in state_dict):
        return ""
    raise ClipTextError(
        f"认不出这批权重是不是 {config.name} 的 ✗ —— 试过 {config.load_prefix!r} / "
        f"{config.name + '.'!r} / 已剥前缀（以 `transformer.text_model.` 开头 ✓）都不匹配 ✓"
        "⇒ 不猜 ✗（也可以显式传 prefix= ✓）")


def _normalize_clip_state_dict(
    state_dict: Mapping[str, Any],
    config: ClipTextConfig,
) -> dict[str, Any]:
    """把两种命名统一成 HF 命名 ✓（口径 = 参考实现的两分支 ✓ 见模块头 ✓）。"""
    if any(key.startswith("transformer.resblocks.") for key in state_dict):
        return open_clip_state_dict(state_dict, config, source_prefix="")
    if any(key.startswith("model.transformer.resblocks.") for key in state_dict):
        return open_clip_state_dict(state_dict, config)
    out = dict(state_dict)
    if any(key.startswith("text_model.") for key in out):
        # ⚠️ 真权重实测：ComfyUI 官方单独发的 `clip_l.safetensors` 键是 `text_model.*` ✓
        #（比本模块少一层 `transformer.` ✗）—— 补上那一层 ✓，其余不动 ✓。
        out = {(f"transformer.{key}" if key.startswith("text_model.") else key): value
               for key, value in out.items()}
    if "text_projection" in out:  # 已剥前缀但仍是 OpenCLIP 的裸投影 ✓ ⇒ 只转置 ✓
        if not config.has_projection:
            raise ClipTextError(
                f"{config.name} 配的是**没有** text_projection ✗ ⇒ 权重里却有它 ✓（塔选错了 ✗）")
        out["text_projection.weight"] = out.pop("text_projection").transpose(0, 1).contiguous()
    return out


def _check_clip_shapes(model: Any, state_dict: Mapping[str, Any], config: ClipTextConfig) -> None:
    """装载**前**逐键比形状 ✓ —— 报中文错，而不是让 torch 抛天书 ✗。"""
    for key, param in model.named_parameters():
        if key not in state_dict:
            continue
        want = tuple(int(v) for v in param.shape)
        got = tuple(int(v) for v in state_dict[key].shape)
        if want != got:
            raise ClipTextError(
                f"{config.name}: 键 {key} 形状不符 ✗ —— 权重 {got} vs 模型 {want} ✓"
                "（配置与权重不是同一套 ✓）")


def load_clip_text_state_dict(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    prefix: str | None = None,
    strict: bool = True,
) -> list[str]:
    """把检查点里**一个塔**的权重装进 `model` ✓，返回**未装载**的键（可忽略的那些 ✓）。

    - 三种叫法都认 ✓：官方 SDXL 前缀（`conditioner.embedders.0.` / `.1.` ✓）、
      ComfyUI 内部前缀（`clip_l.` / `clip_g.` ✓）、已剥前缀 ✓；
    - OpenCLIP 命名（bigG ✓）自动换算 ✓（含 `in_proj` 三分与 `text_projection` 转置 ✓）；
    - 缺键 / 多出认不得的键 ⇒ **报错** ✗（⚠️ 不许用 `strict=False` 糊过去 ✓，否则留下随机初始化的层 ✓✗）；
    - 形状不符 ⇒ **报错** ✗（中文，指明是哪个键 ✓）；
    - `strict=False` ⇒ 缺键**不报错**但并入返回值 ✓（如实报告，不假装完整 ✓）。
    """
    if not _HAS_TORCH:
        raise ClipTextError("没装 torch ✗ ⇒ 装不了权重 ✓")
    config = getattr(model, "config", None)
    if not isinstance(config, ClipTextConfig):
        raise ClipTextError(
            "要装进的是本模块 `build_clip_text_model` 建的模型 ✗（它得带 `config: ClipTextConfig` ✓）")

    if prefix is None:
        prefix = _detect_clip_prefix(state_dict, config)
    scoped = hf_clip_state_dict(state_dict, prefix=prefix) if prefix else dict(state_dict)
    sd = _normalize_clip_state_dict(scoped, config)

    expected = set(clip_text_key_names(config))
    missing = sorted(expected - set(sd))
    if missing and strict:
        raise ClipTextError(
            f"{config.name}: 权重缺 {len(missing)} 个键 ✗（例如 {missing[:3]} ✓）"
            "⇒ 不装半个模型 ✗（确实允许缺键请显式 strict=False ✓）")
    extra = sorted(set(sd) - expected - _IGNORED_CHECKPOINT_KEYS)
    if extra:
        raise ClipTextError(
            f"{config.name}: 权重里多出 {len(extra)} 个认不得的键 ✗（例如 {extra[:3]} ✓）"
            "⇒ 命名对不上/不是这一套 ✓（不静默跳过 ✗）")
    _check_clip_shapes(model, sd, config)
    model.load_state_dict({key: value for key, value in sd.items() if key in expected}, strict=strict)
    unused = sorted((set(sd) - expected) & _IGNORED_CHECKPOINT_KEYS)
    if not strict:
        unused = sorted(set(unused) | set(missing))
    return unused


def encode_sdxl_conditioning(
    clip_l: ClipTextOutput,
    clip_g: ClipTextOutput,
    *,
    expected_dim: int | None = SDXL_CONTEXT_DIM,
) -> tuple[Any, Any]:
    """两塔输出 → SDXL 的 `(context, pooled)` ✓（口径见模块头 ✓）。

    ⚠️ 两个入参得是**同一口径**的前向结果 ✗：SDXL 用
    ``layer_idx=-2, layer_norm_hidden_state=False`` ✓（倒数第二层、不做末尾 norm ✓）。
    ⚠️ `pooled` 只取 bigG 的**投影后**结果 ✓；那塔没有投影 ⇒ **报错** ✗（不拿未投影的凑 ✗）。
    ⚠️ ``expected_dim`` 默认按 **SDXL 的 2048** 核 ✓；缩小版模型做单测时**显式**传别的值 ✓
    （传 ``None`` = 不核宽度 ✓ —— 那是显式放弃判据，不是默认行为 ✗）。
    """
    if not _HAS_TORCH:
        raise ClipTextError("没装 torch ✗ ⇒ 拼不了条件 ✓")
    for name, value in (("clip_l", clip_l), ("clip_g", clip_g)):
        if not isinstance(value, ClipTextOutput):
            raise ClipTextError(
                f"{name} 得是本模块前向返回的 ClipTextOutput ✗（拿到 {type(value).__name__} ✓）")
    left, right = clip_l.selected, clip_g.selected
    if int(left.shape[0]) != int(right.shape[0]):
        raise ClipTextError(
            f"两塔批大小不一致 ✗（{int(left.shape[0])} vs {int(right.shape[0])} ✓）"
            "⇒ 它们不是同一次前向的条件 ✓")
    cut = min(int(left.shape[1]), int(right.shape[1]))
    context = torch.cat([left[:, :cut], right[:, :cut]], dim=-1)
    if expected_dim is not None and int(context.shape[-1]) != int(expected_dim):
        raise ClipTextError(
            f"拼出来的条件宽 {int(context.shape[-1])} ≠ {int(expected_dim)} ✗"
            "⇒ 不是这一对塔 ✓（核对 hidden_size / 塔选对没有 ✓）")
    pooled = clip_g.projected_pooled
    if pooled is None:
        raise ClipTextError(
            "bigG 这塔没有 text_projection ✗ ⇒ SDXL 的 pooled 取不到 ✓"
            "（别拿未投影的 pooled 凑 ✗）")
    return context, pooled


def sdxl_token_ids(
    tokenizer: Any,
    text: str,
    config: ClipTextConfig = CLIP_L,
    *,
    max_length: int | None = None,
    truncate: bool = True,
) -> tuple[list[int], bool]:
    """把一段文本组装成**定长 id 序列** ✓：``BOS + 编码 + EOS + pad`` ✓。

    ⚠️ 注入的 tokenizer **不许自带特殊 token** ✗（本模块只依赖 ``vocab_size`` + ``encode`` ✓
    —— 与 `engine/text_encoder.py` 同一条纪律：词表是权重的一部分，引擎不内置 ✗）。

    ⚠️ 返回 ``(ids, truncated)`` ✓：超长被裁**如实报告** ✗（不静默改输入 ✓）。
    """
    if not isinstance(config, ClipTextConfig):
        raise ClipTextError(f"config 得是 ClipTextConfig ✗（拿到 {type(config).__name__} ✓）")
    if not hasattr(tokenizer, "encode"):
        raise ClipTextError(
            f"tokenizer 得能 ``encode(text) -> list[int]`` ✗（拿到 {type(tokenizer).__name__} ✓）")
    limit = int(max_length) if max_length is not None else config.max_position_embeddings
    if limit < 3:
        raise ClipTextError(f"max_length({limit}) 太小 ✗（至少要放得下 BOS、一个 token、EOS ✓）")
    if limit > config.max_position_embeddings:
        raise ClipTextError(
            f"max_length({limit}) 超过 {config.name} 的位置嵌入上限 "
            f"{config.max_position_embeddings} ✗（位置嵌入只有那么多行 ✓）")
    ids = [int(item) for item in tokenizer.encode(text)]
    bad = [item for item in ids if not 0 <= item < config.vocab_size]
    if bad:
        raise ClipTextError(f"tokenizer 给出的 id 超出词表范围 ✗（例如 {bad[:3]} ✓）")
    truncated = False
    if len(ids) > limit - 2:
        if not truncate:
            raise ClipTextError(
                f"{config.name}: 文本编出 {len(ids)} 个 token ✗ ⇒ 加上 BOS/EOS 超过上限 {limit} ✓"
                "（``truncate=False`` ⇒ 不裁 ✗）")
        ids = ids[: limit - 2]
        truncated = True
    full = [CLIP_BOS_TOKEN_ID, *ids, CLIP_EOS_TOKEN_ID]
    full.extend([config.pad_token] * (limit - len(full)))
    return full, truncated





