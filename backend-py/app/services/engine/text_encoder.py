"""**文本编码器**（本项目自己的实现 ✓ 真 `torch.nn.Module` ✓）+ **可注入 tokenizer** ✓。

## 为什么分成「编码器」与「tokenizer」两块 ✓

* **编码器**（token id → 条件张量 ✓）是**结构** ✓ —— 本模块自己实现 ✓（标准 pre-norm transformer ✓）；
* **tokenizer**（文本 → token id ✓）是**数据/词表** ✓ —— 它是**权重的一部分** ✓，
  本模块**不内置任何词表** ✗（内置一个"看起来能用"的词表只会在真机上错得莫名其妙 ✗）。

⇒ 所以 tokenizer 走**注入式** ✓：给 :class:`Tokenizer` 协议的任何实现都行 ✓：

* :class:`StubTokenizer` —— 确定性哈希 ✓（**自检用** ✓ 不需要下载 ✓；⚠️ 它**不代表真词表** ✗）；
* :class:`HFTokenizer` —— `transformers` 的适配器 ✓（懒导入 ✓；没装就给出可行动报错 ✓）。
  ⚠️ 2026-09-20 起 `transformers` **已登记为本仓的可选依赖** ✓
  （`torch_backend.DEPENDENCIES` 里 `optional=True` ✓ —— 缺它**不阻断**后端 ✓；用镜像装：
  `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers` ✓）；
* ⭐ :class:`~app.services.engine.tokenizer_bpe.BpeTokenizer` —— **本仓自研的字节级 BPE** ✓
  （2026-09-20 补 ✓ 零依赖 ✓ 离线 ✓）：直接读**随权重一起来的**词表文件 ✓
  （``tokenizer.json`` ✓ 或 ``vocab.json`` + ``merges.txt`` ✓，用
  :func:`~app.services.engine.tokenizer_bpe.load_tokenizer` 嗅探 ✓）⇒ **不需要外部 `transformers`** ✗。
  ⚠️ 它需要的是**词表文件**（那是**权重的一部分** ✗ 本模块依然不内置任何词表 ✓ ——
  内置一个"看起来能用"的词表只会在真机上错得莫名其妙 ✓✗）。
* ⭐⭐ :class:`~app.services.engine.tokenizer_hub.TokenizerHub` —— **总入口（优化改造层）** ✓
  （2026-09-20 ✓）：**形态嗅探** → 自研 BPE 优先 ✓ → ``Unigram``/``WordPiece``/``Metaspace``
  等自研**未覆盖**的形态**回退参考实现** ✓（`transformers` ✓ 已登记为可选依赖 ✓）+
  批量 ✓ LRU 缓存 ✓ 离线/关遥测/缓存目录 ✓ 运行期互校 ✓。**给路径即用** ✓
  （``TorchBackend.attach_text_encoder(tokenizer_path=…)`` 走的就是它 ✓）。

## 与 :mod:`app.services.engine.dit` 的接口

``encode(prompt) -> (1, L, output_dim)`` ✓ —— DiT 的 ``forward`` 收 3 维上下文 ✓（见那里的维度归一化 ✓）。

⚠️ **注意力掩码**：本参考实现**不做 padding mask** ✗（真 TE 要做 ✓）⇒ 返回值里额外给 ``mask`` ✓，
调用方想用就用 ✓；`dit` 目前忽略它 ✓（如实写在这里，免得以为已经处理了 ✗）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "HFTokenizer",
    "StubTokenizer",
    "TextEncoderConfig",
    "TextEncoderError",
    "Tokenizer",
    "build_text_encoder",
    "tokenize_prompt",
]


class TextEncoderError(ValueError):
    """配置/tokenizer/形状不合法 ✓（**明确报错**，不悄悄截断到能跑为止 ✗）。"""


class Tokenizer(Protocol):
    """把文本变成 id ✓（本模块只依赖这两件事 ✓）。"""

    vocab_size: int

    def encode(self, text: str) -> list[int]:
        """文本 → id 列表 ✓。"""


@dataclass(frozen=True)
class TextEncoderConfig:
    """结构参数 ✓（**全部显式** ✓ —— 不猜 H3 的 TE ✗）。"""

    vocab_size: int = 1024
    hidden: int = 64
    depth: int = 2
    heads: int = 4
    #: 最长 token 数 ✓（超了**如实回报**截断数 ✓，不静默切 ✗）
    max_length: int = 32
    #: 输出给 DiT 的条件维度 ✓（应与 ``DiTConfig.text_dim`` 一致 ✓）
    output_dim: int = 64
    mlp_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or self.hidden <= 0 or self.depth <= 0 or self.heads <= 0:
            raise TextEncoderError(f"配置不合法：{self.to_dict()} ✗")
        if self.hidden % self.heads:
            raise TextEncoderError(f"hidden={self.hidden} 不能被 heads={self.heads} 整除 ✗")
        if self.max_length <= 0 or self.output_dim <= 0:
            raise TextEncoderError("max_length / output_dim 必须为正 ✗")

    def to_dict(self) -> dict[str, Any]:
        return {"vocabSize": self.vocab_size, "hidden": self.hidden, "depth": self.depth,
                "heads": self.heads, "maxLength": self.max_length,
                "outputDim": self.output_dim, "mlpRatio": self.mlp_ratio}


class StubTokenizer:
    """**确定性**哈希 tokenizer ✓（自检/干跑用 ✓）。

    ⚠️ 它出的 id 与**任何真词表都无关** ✗ ⇒ 只用于验证"管道通不通" ✓，
    **不能**用它得出任何关于语义的结论 ✗（如实标注 ✓）。
    """

    name = "stub"

    def __init__(self, vocab_size: int = 1024, *, max_length: int = 32) -> None:
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)

    def encode(self, text: str) -> list[int]:
        """按字符生成 id ✓（同一个字总是同一个 id ✓ ⇒ 可复现 ✓）。"""
        import hashlib  # noqa: PLC0415

        ids: list[int] = []
        for index, char in enumerate(str(text or "")):
            digest = hashlib.sha256(f"{index}:{char}".encode("utf-8")).digest()
            ids.append(int.from_bytes(digest[:4], "big") % self.vocab_size)
        return ids


class HFTokenizer:
    """`transformers` 的 tokenizer 适配器 ✓（懒导入 ✓ —— 没装就给出**可行动**报错 ✓）。"""

    name = "hf"

    def __init__(self, repo_or_path: str, *, local_files_only: bool = True) -> None:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as err:
            raise TextEncoderError(
                f"未安装 transformers（{err} ✓）⇒ `pip install transformers` ✓"
                f"（注意本机 PyPI 下不动 ✓：加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` ✓）；"
                f"在此之前可用 `StubTokenizer` 验管道 ✓，或直接用**本仓自研 BPE** ✓"
                f"（`tokenizer_bpe.load_tokenizer(词表目录)` ✓ 零依赖 ✓ 离线 ✓）。") from err
        self.repo = str(repo_or_path)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.repo, local_files_only=local_files_only)
        except Exception as err:  # noqa: BLE001 —— ⚠️ **不许把外部库的裸异常漏出去** ✗
            # ⚠️⚠️ 2026-09-20 实测：装了 `transformers` 之后，坏仓库名会抛 `OSError`
            #    （`Repo id must use alphanumeric chars…` ✓）—— 那是**外部库的天书** ✓✗，
            #    调用方（与自检）拿不到「该怎么办」✓ ⇒ 统一包成**可行动**的 `TextEncoderError` ✓。
            raise TextEncoderError(
                f"装载 HF 分词器失败（仓库/路径 `{self.repo}` ✓）：{err} ✓"
                f" ⇒ 先核对仓库名或本机是否已有该词表 ✓（离线时传**本地目录** ✓）；"
                f"也可改用**本仓自研 BPE** ✓（`tokenizer_bpe.load_tokenizer(词表目录)` ✓ "
                f"零依赖 ✓ 离线 ✓）。") from err
        self.vocab_size = int(getattr(self._tokenizer, "vocab_size", 0) or 0)

    def encode(self, text: str) -> list[int]:
        return [int(item) for item in self._tokenizer.encode(str(text or ""))]


def tokenize_prompt(tokenizer: Tokenizer, text: str, *, max_length: int,
                    pad_id: int = 0) -> tuple[list[int], int, bool]:
    """文本 → ``(ids, 原长, 是否被截断)`` ✓ —— **截断必须能被调用方看到** ✗（静默切最坏 ✓）。"""
    ids = list(tokenizer.encode(text))
    original = len(ids)
    truncated = original > int(max_length)
    if truncated:
        ids = ids[: int(max_length)]
    if not ids:
        ids = [int(pad_id)]
    return ids, original, truncated


def _build() -> type:
    """造出 :class:`TextEncoder` 类 ✓（模块级定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。"""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class Block(nn.Module):
        """标准 **pre-norm** transformer 块 ✓（自注意力 + MLP ✓，与 DiT 的 adaLN 块不同用途 ✓）。"""

        def __init__(self, config: TextEncoderConfig) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.hidden)
            self.attn = nn.MultiheadAttention(config.hidden, config.heads, batch_first=True)
            self.norm2 = nn.LayerNorm(config.hidden)
            inner = int(config.hidden * config.mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(config.hidden, inner), nn.GELU(),
                                     nn.Linear(inner, config.hidden))

        def forward(self, tokens: Any, mask: Any = None) -> Any:
            normed = self.norm1(tokens)
            attended, _ = self.attn(normed, normed, normed, need_weights=False,
                                    key_padding_mask=mask)
            tokens = tokens + attended
            return tokens + self.mlp(self.norm2(tokens))

    class TextEncoder(nn.Module):
        """token id → 条件张量 ✓（输出 ``(B, L, output_dim)`` ✓）。"""

        def __init__(self, config: TextEncoderConfig) -> None:
            super().__init__()
            self.config = config
            self.embed = nn.Embedding(config.vocab_size, config.hidden)
            self.pos = nn.Parameter(torch.zeros(1, config.max_length, config.hidden))
            self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
            self.norm = nn.LayerNorm(config.hidden)
            self.out = (nn.Identity() if config.output_dim == config.hidden
                        else nn.Linear(config.hidden, config.output_dim))

        def forward(self, ids: Any, mask: Any = None) -> Any:
            """``(B, L)`` → ``(B, L, output_dim)`` ✓；``mask`` 为 ``True`` 的位置是**padding** ✓。"""
            shape = tuple(ids.shape)
            if len(shape) != 2:
                raise TextEncoderError(f"token id 张量应为 (B, L) ✓，收到 {shape} ✗")
            if int(shape[1]) > self.config.max_length:
                raise TextEncoderError(
                    f"token 数 {int(shape[1])} 超过 maxLength={self.config.max_length} ✗"
                    f"（请先用 `tokenize_prompt` **显式**截断 ✓ —— 它会回报截断数 ✓）")
            if mask is not None:
                shape_mask = tuple(mask.shape)
                if shape_mask != shape:
                    raise TextEncoderError(f"mask 形状 {shape_mask} 应与 ids {shape} 一致 ✗")
                # 全 padding 行会让 softmax 出 NaN ✗ ⇒ 至少放开第一个位置 ✓（真实实现也这么做 ✓）
                mask = mask.clone()
                mask[:, 0] = False
            tokens = self.embed(ids) + self.pos[:, : int(shape[1])]
            for block in self.blocks:
                tokens = block(tokens, mask)
            return self.out(self.norm(tokens))

    return TextEncoder


_ENCODER_CLASS: Any = None


def build_text_encoder(config: TextEncoderConfig) -> Any:
    """构造文本编码器 ✓（**懒导入 torch** ✓）。"""
    global _ENCODER_CLASS
    if _ENCODER_CLASS is None:
        _ENCODER_CLASS = _build()
    return _ENCODER_CLASS(config)
