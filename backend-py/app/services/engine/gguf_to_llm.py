"""GGUF 权重 → :class:`llm.LlmModel` 的**装载**（2026-09-25 起）。

## 为什么单独一层

:mod:`gguf_dequant` 反量化出 ``{张量名: fp32 数组}``，但**没映射到 LlmModel 的参数名**，也没把
ggml 的维序摆成 numpy 序 ✗。本模块补上这一步 ✓：把 llama.cpp 的 ggml 命名（``blk.N.attn_q.weight`` ✓）
映射成 LlmModel 的命名（``blocks.N.attn.q_proj.weight`` ✓），并按 ggml 维序把数组**摆正** ✓
（``reversed(dims)`` ✓，见下面第 2 条 —— 不是 transposed ✗）。

## 三条口径（不猜 ✗，事实来源：llama.cpp 的 ggml 约定 ✓）

1. **命名**：GGUF 用 ggml 的 ``blk.N.*`` ✓ → LlmModel 用 ``blocks.N.*`` ✓（映射表见下 ✓）；
2. **数组摆法（本模块最容易错的一步 ✗）**：GGUF 的 ``dims`` 是 ggml 的 ``ne[]`` ✓，``ne[0]`` 是
   **最内维（变化最快）** ✗；而 numpy 的 C 序是**最后一维**变化最快 ✗ ⇒
   ``numpy_shape = tuple(reversed(dims))`` ✓，**摆好即是 PyTorch 布局、不需要再 transpose** ✓。
   例：``attn_q`` 的 dims ``[4096(in), 4096(out)]`` → reshape ``(4096, 4096)`` = ``(out, in)`` ✓，
   ``nn.Linear.weight`` 要的就是 ``(out, in)`` ✓；``token_embd`` 的 dims ``[4096, 151936]`` →
   ``(151936, 4096)`` = ``(vocab, hidden)`` ✓，``nn.Embedding.weight`` 要的也是这个 ✓。
   ⚠️ **曾经的错法**（2026-10-29 真机 qwen3:8b 全程乱码的唯一根因 ✗✗）：按 dims 正序
   ``reshape(ne0, ne1)`` 再 ``.T`` —— 形状一模一样 ✓、装载不报错 ✓、自造往返用例也过 ✗，
   但元素被整张重排 ⇒ 权重成了「同量级的随机矩阵」✗（embedding 行间余弦 0.014 ✗，
   ``1+1=`` 的下一 token 熵 6.9 bits ✗）。**形状相等不等于摆法正确**，`dequantize_tensor` 出来的
   是 ggml 顺序的一维流 ✓ ⇒ 只有 ``reversed(dims)`` 这一种摆法 ✓；
3. **tie embeddings**：``config.tie_embeddings=True`` 时 LlmModel 的 ``lm_head.weight is embed.weight`` ✓
   ⇒ GGUF 里的 ``output.weight`` **跳过** ✗（已经 tie ✓ —— 强制装会多出一个形状一样的 key ✗）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import gguf as gguf_mod
from . import gguf_dequant
from . import llm as llm_mod
# ⚠️ 2026-10-29 修：``infer_llm_config`` 的拒绝分支里直接用 ``GgufDequantError`` ✗，
# 而本模块只 ``from . import gguf_dequant`` ✓ ⇒ 一进拒绝分支就 **NameError** ✗✗
# （本该是「名字对、原因清楚」的拒绝，变成「名字都没了」的崩 ✗；真机 qwen3:8b 首次撞出来 ✓）。
from .gguf_dequant import GgufDequantError

__all__ = ["gguf_state_dict", "infer_llm_config", "load_llm_from_gguf", "map_name", "numpy_shape"]

#: GGUF 层名 → LlmModel 层名 ✓（``{}`` 是层号 ✓）
_LAYER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"blk\.(\d+)\.attn_q\.weight", "blocks.{}.attn.q_proj.weight"),
    (r"blk\.(\d+)\.attn_k\.weight", "blocks.{}.attn.k_proj.weight"),
    (r"blk\.(\d+)\.attn_v\.weight", "blocks.{}.attn.v_proj.weight"),
    (r"blk\.(\d+)\.attn_output\.weight", "blocks.{}.attn.o_proj.weight"),
    (r"blk\.(\d+)\.ffn_gate\.weight", "blocks.{}.ffn.gate.weight"),
    (r"blk\.(\d+)\.ffn_up\.weight", "blocks.{}.ffn.up.weight"),
    (r"blk\.(\d+)\.ffn_down\.weight", "blocks.{}.ffn.down.weight"),
    (r"blk\.(\d+)\.attn_norm\.weight", "blocks.{}.norm1.weight"),
    (r"blk\.(\d+)\.ffn_norm\.weight", "blocks.{}.norm2.weight"),
)

#: ⭐ Qwen3 的 **per-head QK 归一化**（2026-10-29 补 ✓，真机 qwen3:8b 实测：36 层 × 2 = 72 个张量
#: 原实现**一个都映射不上** ✗✗ ⇒ 8B/14B 全部装不进 ✗）。形状 ``(head_dim,)`` ✓ **不转置** ✗。
#: ⚠️ 单列表、单开关：``qk_norm=False`` 的模型若也套这张表 ⇒ 多出两个 key ⇒ ``strict=True`` 当场报
#: 「unexpected key」✗（所以不能并进 ``_LAYER_PATTERNS`` ✗）。
_QK_NORM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"blk\.(\d+)\.attn_q_norm\.weight", "blocks.{}.attn.q_norm.weight"),
    (r"blk\.(\d+)\.attn_k_norm\.weight", "blocks.{}.attn.k_norm.weight"),
)

#: ⚠️ 旧实现曾有一张 :data:`_TRANSPOSE_SUFFIXES`（把 2D 权重 ``.T`` 一次）✗ —— **2026-10-29 整张删掉** ✓：
#: 它与「按 ``dims`` 正序 reshape」配对时，两条错互相抵消掉形状差、却把元素整张重排 ✗✗
#: （见模块 docstring 第 2 条；真机 qwen3:8b 只因此就全程乱码 ✗）。现在摆法是
#: ``arr.reshape(tuple(reversed(dims)))`` ✓，**恰好等于 PyTorch 布局 ⇒ 不再需要任何转置** ✓。


def numpy_shape(dims: Any) -> tuple[int, ...]:
    """ggml 的 ``ne[]`` → numpy 形状 ✓（``ne[0]`` 最内 ⇒ 反转 ✓）。

    ``[4096, 151936]`` → ``(151936, 4096)`` ✓（``token_embd`` → ``nn.Embedding.weight`` ✓）；
    ``[4096, 4096]`` → ``(4096, 4096)`` ✓；1D（norm / QK-norm）反转无影响 ✓。
    """
    return tuple(int(dim) for dim in reversed(tuple(dims)))


def map_name(gguf_name: str, *, tie_embeddings: bool, qk_norm: bool = False) -> str | None:
    """GGUF 张量名 → LlmModel 参数名 ✓（``None`` = 跳过：tie 时的 ``output.weight`` ✓）。

    ⚠️ 未知名字也返回 ``None`` ✗（不是抛错 ✓）—— 真正的「缺不漏」由 ``load_state_dict(strict=True)``
    兜底 ✓（漏一个 key 它就会报「missing key」✗）。
    ``qk_norm`` 控制 ``attn_q_norm``/``attn_k_norm`` 这两族（见 :data:`_QK_NORM_PATTERNS` ✓）。
    """
    if gguf_name == "token_embd.weight":
        return "embed.weight"
    if gguf_name == "output_norm.weight":
        return "norm.weight"
    if gguf_name == "output.weight":
        return None if tie_embeddings else "lm_head.weight"
    patterns = _LAYER_PATTERNS + (_QK_NORM_PATTERNS if qk_norm else ())
    for pattern, template in patterns:
        match = re.match(pattern, gguf_name)
        if match:
            return template.format(match.group(1))
    return None


def gguf_state_dict(weights: dict[str, Any], config: llm_mod.LlmConfig) -> dict[str, Any]:
    """``{GGUF 名: (ggml dims, 一维 flat 数组)}`` → LlmModel 的 ``state_dict`` ✓。

    ⚠️ 入参必须带 **ggml 的 dims** ✓：形状信息不能从数组自己反推 ✗（
    ``dequantize_tensor`` 吐的就是一维流 ✓，同一个流按「正序 reshape」与「反序 reshape」
    得到的 2D 数组**形状不同、元素顺序也不同**，而两者再各自 ``.T`` 都可能凑出目标形状 ✗
    ⇒ 只有 dims 在场才能唯一定位 ✓）。
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    state: dict[str, Any] = {}
    for gguf_name, payload in weights.items():
        target = map_name(gguf_name, tie_embeddings=config.tie_embeddings, qk_norm=config.qk_norm)
        if target is None:
            continue
        dims, flat = payload
        arr = np.reshape(np.asarray(flat), numpy_shape(dims))
        state[target] = torch.from_numpy(np.ascontiguousarray(arr))
    return state


def load_llm_from_gguf(path: str | Path, config: llm_mod.LlmConfig, *,
                       device: Any = "cpu", dtype: Any = None) -> Any:
    """GGUF 文件 → 装载好权重的 :class:`llm.LlmModel` ✓。

    ⚠️ 2026-10-29 修（真机 qwen3:8b 撞出来 ✓✗）：原实现先把**全部**张量反量化进一个 ``dict``、
    再整份转 torch、再 ``load_state_dict`` ⇒ 峰值 = fp32 权重 × **3**（numpy 字典 ＋ torch state
    ＋ 模型本体）✗✗ —— 8B 就是约 98 GB，在 96 GB 机器上直接进交换区（实测工作集冲到 86 GB 还没装完 ✗）。
    现在改为**逐张量流式装配** ✓：读出 → 反量化 → 转置 → 直接 ``copy_`` 进模型参数 → 丢弃 ✓，
    峰值 = 模型本体 ＋ 单张张量 ✓（内存账从 3× 降到 1× ✓）。

    ⚠️ 流式装配之后 ``load_state_dict(strict=True)`` 那句不能用了 ✗（没有整份 state 可传 ✗）⇒
    在同一语义上**手写对账** ✓：装完比 ``model.state_dict()`` 的键集，**缺**与**多**都当场报 ✓✗
    （等价于 strict=True ✓，不静默装半套 ✗）。

    ⚠️ **CPU / GPU 同一条装载路径** ✓（2026-10-29 补）：模型按 ``device``/``dtype`` **直接建** ✓
    （见 :func:`~.llm.build_llm`），反量化结果留在 CPU 上 ✓，``copy_`` 时**跨设备拷一次** ✓ ⇒
    峰值 = 设备上的模型 ＋ CPU 上一张张量 ✓（不是「CPU 装完再整份搬」✗ —— 那等于两份权重 ✗✗）。
    默认 ``device="cpu"`` ✓：择优（``auto``）属于配置层 :class:`~.llm_backend.LlmBackend` ✓，不在内核里 ✓。
    """
    info = gguf_mod.inspect(path)
    # ⭐ QK-norm **只能由张量表判定** ✓（GGUF 没有任何 metadata 键声明它 ✗，见真机 qwen3:8b 实测 ✓）：
    # 文件里有 ``blk.N.attn_q_norm.weight`` 就是有 ✓，没有就是没有 ✓ —— 这不是猜 ✗，是看文件 ✓。
    # 调用方把 ``config.qk_norm`` 设错也不至于装出半套：设 True 而文件没这两项 ⇒ 对账报 missing ✗；
    # 设 False 而文件有 ⇒ 多出 72 个张量装不进去 ✗ ⇒ 这里按文件**纠正**并把话说明白 ✓。
    has_qk_norm = any(name.endswith(("attn_q_norm.weight", "attn_k_norm.weight"))
                      for name in info.tensors)
    if has_qk_norm != config.qk_norm:
        import dataclasses  # noqa: PLC0415
        config = dataclasses.replace(config, qk_norm=has_qk_norm)
    model = llm_mod.build_llm(config, device=device, dtype=dtype)
    _load_into_model(path, info, model, config)
    return model


def _load_into_model(path: str | Path, info: Any, model: Any, config: llm_mod.LlmConfig) -> None:
    """逐张量读进**已建好的**模型 ✓（流式 ✓ 峰值 = 模型 ＋ 单张张量 ✓）+ 键集对账 ✓。"""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    expected = model.state_dict()
    loaded: dict[str, tuple[int, ...]] = {}
    unknown: list[str] = []
    with torch.no_grad():
        for name, tensor in info.tensors.items():
            target = map_name(name, tie_embeddings=config.tie_embeddings, qk_norm=config.qk_norm)
            if target is None:
                continue
            if target not in expected:
                unknown.append(f"{name} → {target}")
                continue
            arr = _read_tensor(path, info, tensor)
            # ⚠️ 这里**不再转置** ✓：``_read_tensor`` 已按 ``reversed(dims)`` 摆成 PyTorch 布局 ✓
            # （曾经的 ``.T`` 与「按 dims 正序 reshape」配对，把元素整张重排 ✗✗ —— 全程乱码的根因）。
            # ``copy_`` 也不能直接吃非连续视图 ✗ ⇒ 非连续时 ascontiguousarray 一份 ✓；
            # 这份拷贝与参数本体同寿于一次循环 ✓ 下一轮就被回收 ✓。
            weight = torch.from_numpy(arr if arr.flags.c_contiguous else np.ascontiguousarray(arr))
            if weight.dtype != expected[target].dtype:
                weight = weight.to(expected[target].dtype)
            if tuple(weight.shape) != tuple(expected[target].shape):
                raise GgufDequantError(
                    f"张量形状不符 ✗：{name} → {target} 装上后 {tuple(weight.shape)} ✓ "
                    f"而模型要 {tuple(expected[target].shape)} ✗")
            expected[target].copy_(weight)
            loaded[target] = tuple(weight.shape)
            del arr, weight
    unexpected = sorted(unknown)
    if unexpected:  # 防御：map_name 产出的名字必须都在模型里 ✓（漏了就是映射表写错了 ✗）
        raise GgufDequantError(
            f"{len(unexpected)} 个张量映射到了模型里**没有**的参数 ✗（映射表与模型对不上 ✗）："
            f"{unexpected[:8]}{' …' if len(unexpected) > 8 else ''}")
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise GgufDequantError(
            f"GGUF 里缺 {len(missing)} 个张量 ✗（模型要、文件没有 ⇒ 装不了 ✗）：{missing[:8]}"
            f"{' …' if len(missing) > 8 else ''}")


def _read_tensor(path: str | Path, info: Any, tensor: Any) -> Any:
    """读一个张量并**摆成 numpy 序** ✓（``reversed(dims)`` ✓，见模块 docstring 第 2 条）。"""
    with open(path, "rb") as handle:
        handle.seek(info.data_start + tensor.offset)
        blob = handle.read(tensor.nbytes)
    arr = gguf_dequant.dequantize_tensor(blob, tensor.type_id, tensor.numel)
    return arr.reshape(numpy_shape(tensor.shape))


def infer_llm_config(metadata: dict[str, Any], *, arch: str | None = None) -> llm_mod.LlmConfig:
    """从 GGUF 元数据推导 :class:`llm.LlmConfig` ✓（架构参数**全从权重读** ✓ 不猜 ✗）。

    ⚠️ GGUF 的架构参数前缀是 ``general.architecture``（``llama`` / ``qwen2`` / ``qwen3`` ✓），
    字段名同族：``{arch}.embedding_length`` / ``{arch}.block_count`` / ``{arch}.attention.head_count``
    / ``{arch}.attention.head_count_kv``（GQA ✓）/ ``{arch}.feed_forward_length`` / ``{arch}.rope.freq_base``
    / ``{arch}.attention.layer_norm_rms_epsilon`` 等 ✓（Qwen3 GGUF 偶尔沿用 ``llama.`` 前缀 ⇒ 回退读 ✓）。

    ⚠️ 缺必需字段 ⇒ **具名拒绝** ✗（不猜默认值 ✗ —— 猜了会装出「名字对、形状全错」的模型 ✗）。
    """
    if arch is None:
        arch = str(metadata.get("general.architecture") or "llama")

    def _get(name: str) -> Any:
        for prefix in (arch, "llama"):
            value = metadata.get(f"{prefix}.{name}")
            if value is not None:
                return value
        return None

    vocab_size = _get("vocab_size")
    if vocab_size is None:
        # ⚠️ 2026-10-29 修（真机 qwen3:8b / ollama blob 撞出来）：llama.cpp 的 GGUF **不写**
        # ``{arch}.vocab_size`` ✗ —— 词表大小就是 ``tokenizer.ggml.tokens`` 的**元素个数** ✓。
        # 原实现把它当必需字段 ⇒ 所有真实 Ollama/qwen GGUF 一律被拒 ✗✗（「不猜默认值」没错 ✓，
        # 但这里不是猜 ✗：词表就在文件里摆着 ✓）。
        # ⚠️ 别再写成 ``len(tokens)`` ✗✗：:func:`gguf.inspect` 对数组**截断保存** ✓（元数据只做摘要 ✓），
        # 字符串数组的真长度在 ``{"__array__": "string", "len": N, "head": 前 16 项}`` ✓ ——
        # ``len()`` 会拿到 **3**（三个键）而不是 151936 ✗（这句是实测撞出来的 ✓）。
        tokens = metadata.get("tokenizer.ggml.tokens")
        if isinstance(tokens, dict) and tokens.get("len") is not None:
            vocab_size = int(tokens["len"])
        elif isinstance(tokens, list):
            vocab_size = len(tokens)
    hidden = _get("embedding_length")
    depth = _get("block_count")
    heads = _get("attention.head_count")
    kv_heads = _get("attention.head_count_kv")
    ffn = _get("feed_forward_length")
    # 同理：GGUF 写的是 ``attention.key_length``（= ``value_length`` ✓），不是 ``attention.head_dim`` ✗
    # ⇒ 先读 key_length ✓，没有再退回 hidden/heads ✓（GQA 时两者可能不等 ✗ 所以顺序不能反 ✓）。
    head_dim = _get("attention.head_dim") or _get("attention.key_length")
    rope_theta = _get("rope.freq_base")
    rms_eps = _get("attention.layer_norm_rms_epsilon")

    required = {"vocab_size": vocab_size, "embedding_length": hidden, "block_count": depth,
                "attention.head_count": heads, "feed_forward_length": ffn}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise GgufDequantError(
            f"GGUF 元数据缺架构参数：{missing} ✗（arch={arch} ✓）⇒ 推不出 LlmConfig ✗ "
            f"（不猜默认值 ✗）")

    if kv_heads is None:
        kv_heads = heads
    if head_dim is None:
        if int(hidden) % int(heads) != 0:
            raise GgufDequantError(
                f"embedding_length（{hidden}）不被 head_count（{heads}）整除 ✗ ⇒ head_dim 推不出 ✗")
        head_dim = int(hidden) // int(heads)

    eos = metadata.get("tokenizer.ggml.eos_token_id")
    pad = metadata.get("tokenizer.ggml.pad_token_id")
    return llm_mod.LlmConfig(
        vocab_size=int(vocab_size), hidden=int(hidden), depth=int(depth),
        heads=int(heads), kv_heads=int(kv_heads), head_dim=int(head_dim),
        ffn=int(ffn), rope_theta=float(rope_theta if rope_theta is not None else 10000.0),
        rms_eps=float(rms_eps if rms_eps is not None else 1e-6),
        tie_embeddings=bool(metadata.get(f"{arch}.tie_embeddings", False)),
        pad_id=int(pad) if pad is not None else 0,
        eos_id=int(eos) if eos is not None else None,
    )
