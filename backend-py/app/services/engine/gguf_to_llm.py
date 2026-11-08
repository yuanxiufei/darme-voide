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
   ⭐ 2026-11-07 真机补（``Qwen3-4B-Instruct-2507-GGUF`` Q4_K_M ✓✗）：**这份文件既没有
   ``qwen3.tie_embeddings`` 元数据键、也没有 ``output.weight`` 张量** ✓（llama.cpp 允许省略 ✓）。
   原实现只看元数据、缺键按 ``False`` ⇒ 对账要 ``lm_head.weight`` ⇒ 报「缺 1 个张量 ✗」装不进 ✗✗。
   现在「是不是 tie」以**张量表**为准（:func:`tie_embeddings_of` ✓：没有 ``output.weight`` ⇒ 只能 tie ✓），
   且对账认得**真共享的键**（:func:`_tied_keys` ✓：凭据是 ``lm_head.weight is embed.weight`` 这件事 ✓，
   不是配置位 ✗ —— ``state_dict()`` 会把同一张 Parameter 在两个名字下各列一次 ✗）。

## 第四条：反量化产物落盘（G3 ✓ 2026-11-07）

反量化是**纯 CPU** 的活 ✓（`Q4_K` 向量化后 100.6 M 元素/秒 ✓）—— 一份 4B 文本塔约 **31 秒**，
而进程**每次启动**都要再付一遍 ✗✗。`materialize` 把「装进模型的那张张量的**原始字节**」写成
safetensors ✓ ⇒ 第二次起只是读盘 ✓，省 CPU ＋ 省峰值内存（不再需要 fp32 中间量 ✓）。

三条不许动的口径 ✗✗：

* 物化的 dtype **就是装载目标的 dtype** ✓，字节取自 **torch 张量本身**（``.view(int16)`` ✓）⇒
  「命中与不命中装出来的权重**逐位相等**」是构造出来的 ✓（自己写舍入公式会差 1 ulp ✗✗）；
* **键集不变**：计划里的张量 = 装载循环**真会写**的那些 ✓（``map_name`` 有值且模型里真有 ✓）；
  少一个 ⇒ `materialize.Writer` 当场拒并且**不留产物** ✓（半份产物装得上、跑得动、输出不对 ✗）；
* **全份或不做** ✗：文件里一个量化张量都没有（全 fp32/fp16）⇒ 整份物化等于抄一份 ✓✗ ⇒ 不做，
  并如实带 ``reason`` ✓。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from . import gguf as gguf_mod
from . import gguf_dequant
from . import llm as llm_mod
from . import materialize as materialize_mod
# ⚠️ 2026-10-29 修：``infer_llm_config`` 的拒绝分支里直接用 ``GgufDequantError`` ✗，
# 而本模块只 ``from . import gguf_dequant`` ✓ ⇒ 一进拒绝分支就 **NameError** ✗✗
# （本该是「名字对、原因清楚」的拒绝，变成「名字都没了」的崩 ✗；真机 qwen3:8b 首次撞出来 ✓）。
from .gguf_dequant import GgufDequantError

__all__ = ["detect_features", "gguf_state_dict", "infer_llm_config", "load_llm_from_gguf",
           "map_name", "numpy_shape", "tie_embeddings_of"]

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

#: ⭐ **q/k/v 的偏置**（2026-10-31 补 ✗）：Qwen2 / Qwen2.5 每层有三根 ``attn_{q,k,v}.bias`` ✓
#: （``o_proj`` 与 MLP 三项**没有** ✓）。⚠️ 与 QK-norm 同一纪律：``qkv_bias=False`` 的模型
#: （Qwen3 / Llama）若也套这张表 ⇒ 多出键 ⇒ 对账当场报「映射到了模型里没有的参数」✗。
#: 漏了这张表 ⇒ Qwen2.5 的 3×层数 根键**无处可放** ✓✗ —— 而且因为 ``LlmConfig.qkv_bias`` 默认
#: False、模型压根不建 bias 参数 ⇒ 连「缺键」都不会报 ✓✗：**装得进、形状全对、只是每层少一个常数**
#: ⇒ 提示词跟随明显变弱 ✓✗（本仓最恨的那类静默降质 ✓）。
_QKV_BIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"blk\.(\d+)\.attn_q\.bias", "blocks.{}.attn.q_proj.bias"),
    (r"blk\.(\d+)\.attn_k\.bias", "blocks.{}.attn.k_proj.bias"),
    (r"blk\.(\d+)\.attn_v\.bias", "blocks.{}.attn.v_proj.bias"),
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


def map_name(gguf_name: str, *, tie_embeddings: bool, qk_norm: bool = False,
             qkv_bias: bool = False) -> str | None:
    """GGUF 张量名 → LlmModel 参数名 ✓（``None`` = 跳过：tie 时的 ``output.weight`` ✓）。

    ⚠️ 未知名字也返回 ``None`` ✗（不是抛错 ✓）—— 真正的「缺不漏」由 ``load_state_dict(strict=True)``
    兜底 ✓（漏一个 key 它就会报「missing key」✗）。
    ``qk_norm`` 控制 ``attn_q_norm``/``attn_k_norm`` 这两族（见 :data:`_QK_NORM_PATTERNS` ✓）；
    ``qkv_bias`` 控制 ``attn_{q,k,v}.bias`` 这三族（见 :data:`_QKV_BIAS_PATTERNS` ✓）。
    """
    if gguf_name == "token_embd.weight":
        return "embed.weight"
    if gguf_name == "output_norm.weight":
        return "norm.weight"
    if gguf_name == "output.weight":
        return None if tie_embeddings else "lm_head.weight"
    patterns = (_LAYER_PATTERNS
                + (_QK_NORM_PATTERNS if qk_norm else ())
                + (_QKV_BIAS_PATTERNS if qkv_bias else ()))
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
        target = map_name(gguf_name, tie_embeddings=config.tie_embeddings,
                          qk_norm=config.qk_norm, qkv_bias=config.qkv_bias)
        if target is None:
            continue
        dims, flat = payload
        arr = np.reshape(np.asarray(flat), numpy_shape(dims))
        state[target] = torch.from_numpy(np.ascontiguousarray(arr))
    return state


def detect_features(config: llm_mod.LlmConfig, tensor_names: Iterable[str]) -> llm_mod.LlmConfig:
    """按**张量表**纠正 GGUF metadata 不给的两件事 ✓ —— ``qk_norm`` 与 ``qkv_bias`` ✓。

    ⚠️ 为什么只能看张量表 ✗：GGUF 的 metadata 里**没有任何键**声明这两项 ✓（真机 qwen3:8b 实测 ✓）。
    设错的两种后果差得很远 ✗：
    · ``qk_norm`` 设错 ⇒ 对账能响 ✓（多 72 个张量装不进去 / 缺 72 个 key ✗）；
    · ``qkv_bias`` 设 False 而文件里有 ⇒ 模型**压根不建** bias 参数 ⇒ 那三根键被当「未知名字」跳过 ✓✗，
      **连 missing 都不会报** ✓✗ ⇒ 每层少一个常数、提示词跟随变弱 ✓✗（2026-10-31 修 ✓）。

    这不是猜 ✓：文件里有 ``blk.N.attn_q_norm.weight`` / ``blk.N.attn_q.bias`` 就是有 ✓，没有就是没有 ✓。
    """
    names = list(tensor_names)
    has_qk_norm = any(name.endswith(("attn_q_norm.weight", "attn_k_norm.weight")) for name in names)
    has_qkv_bias = any(name.endswith(("attn_q.bias", "attn_k.bias", "attn_v.bias")) for name in names)
    if has_qk_norm == config.qk_norm and has_qkv_bias == config.qkv_bias:
        return config
    import dataclasses  # noqa: PLC0415
    return dataclasses.replace(config, qk_norm=has_qk_norm, qkv_bias=has_qkv_bias)


def load_llm_from_gguf(path: str | Path, config: llm_mod.LlmConfig, *,
                       device: Any = "cpu", dtype: Any = None,
                       materialize: bool | str | Path | None = None) -> Any:
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

    ``materialize``（G3 ✓ 2026-11-07）：``None`` = 跟随总开关（``DARME_MATERIALIZE`` ✓ 默认开 ✓）；
    ``False`` = 关；``True`` = 开（默认目录 ✓）；给路径 = 用这个目录 ✓。结论挂在 ``model.materialized``
    上 ✓（``{hit, gib, path, reason}`` ✓ —— 命中时说清**凭什么** ✓，不做时说清**为什么** ✓✗）。
    """
    info = gguf_mod.inspect(path)
    config = detect_features(config, info.tensors)
    model = llm_mod.build_llm(config, device=device, dtype=dtype)
    handle, note = _open_materializer(path, info, model, config, materialize)
    report = _load_into_model(path, info, model, config, materializer=handle)
    model.materialized = (report.to_dict() if report is not None
                          else {"enabled": False, "reason": note or "没有可用计划 ✓"})
    return model


#: 转换器版本 ✓ —— ⭐ **反量化/摆法逻辑一改就要 +1** ✗✗（它进产物文件名的哈希 ✓：
#: 不加 ⇒ 换了公式之后旧产物仍被当成有效 ✓✗ —— 那是「装得上、数值不对」的形状 ✓）。
CONVERTER = "gguf-dequant-v1"

#: 这几种 GGML 类型**源就是浮点** ✓（0=f32 / 1=f16 / 32=bf16 ✓）⇒ 不当作「有量化张量」。
_FLOAT_TYPES = frozenset({0, 1, 32})


def _dtype_label(dtype: Any) -> str | None:
    """torch dtype → safetensors 头里的 dtype 名 ✓（不在表里 ⇒ ``None`` ✓ 不放行 ✗）。"""
    import torch  # noqa: PLC0415

    return {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}.get(dtype)


def _plan_specs(info: Any, config: llm_mod.LlmConfig,
                expected: dict[str, Any]) -> tuple[list[Any], str]:
    """物化**计划** ✓ ⇒ ``(张量表 ✓, dtype 标签／不做时的原因 ✓)``。

    ⚠️ 键集 = **装载循环真会写的那些** ✗✗：``map_name`` 有值（tie 的 ``output.weight`` 没有 ✓）
    且模型里真有这个参数 ✓ —— 否则产物会多 key（读侧得跳过并解释 ✗）或少 key（装到一半才发现 ✗）。
    ⚠️ 全是浮点源 ⇒ **不做** ✓：那种文件物化等于把权重原样抄一份（16 GiB 换 0 秒 ✗）。
    """
    rows: list[Any] = []
    labels: set[str] = set()
    for name, tensor in info.tensors.items():
        target = map_name(name, tie_embeddings=config.tie_embeddings,
                          qk_norm=config.qk_norm, qkv_bias=config.qkv_bias)
        if target is None or target not in expected:
            continue
        label = _dtype_label(expected[target].dtype)
        if label is None:
            return [], f"目标 dtype {expected[target].dtype} 不在 safetensors 头表里 ✗ ⇒ 不物化 ✓"
        labels.add(label)
        rows.append(materialize_mod.Spec.of(name, tuple(expected[target].shape), label))
    if not rows:
        return [], "没有可映射的张量 ⇒ 没什么可物化的 ✓"
    if all(info.tensors[row.name].type_id in _FLOAT_TYPES for row in rows):
        return [], "源张量全是 fp32/fp16/bf16 ✓（文件里没有量化 ✓）⇒ 物化等于抄一份 ✗ 不做 ✓"
    return rows, "|".join(sorted(labels))


def _open_materializer(path: str | Path, info: Any, model: Any, config: llm_mod.LlmConfig,
                       choice: bool | str | Path | None) -> tuple[Any, str]:
    """按 ``materialize`` 决定要不要物化 ✓ ⇒ ``(把手／None ✓, 不做时的原因 ✓)``。"""
    if choice is False:
        return None, "调用方显式关掉了（materialize=False ✓）"
    if choice is None and not materialize_mod.enabled():
        return None, f"总开关关着（{materialize_mod.ENV_SWITCH}=0 ✓）"
    directory = choice if isinstance(choice, (str, Path)) else None
    specs, note = _plan_specs(info, config, model.state_dict())
    if not specs:
        return None, note
    try:
        return materialize_mod.Materializer(
            path, converter=CONVERTER, dtype=note, directory=directory,
            quota_gib=materialize_mod.limit_gib()), ""
    except materialize_mod.MaterializeSkip as err:      # 源指纹读不到 ⇒ 不物化 ✓ 不拦装载 ✓
        return None, str(err)


def _receipt_path(materializer: Any) -> str:
    """回执里的路径 ✓（源指纹读不到时 :meth:`Materializer.path` 会抛 ✓ ⇒ 别让它盖掉真错误 ✗）。"""
    try:
        return str(materializer.path())
    except Exception:  # noqa: BLE001
        return ""


def _load_into_model(path: str | Path, info: Any, model: Any, config: llm_mod.LlmConfig, *,
                     materializer: Any = None) -> Any:
    """逐张量读进**已建好的**模型 ✓（流式 ✓ 峰值 = 模型 ＋ 单张张量 ✓）+ 键集对账 ✓。

    命中产物 ⇒ **一次反量化都不做** ✓（只读盘 ＋ ``copy_`` ✓）；否则走反量化 ✓，并在**同一次循环**
    里顺手把「装进去的那张张量的原始字节」写进产物 ✓（不为了缓存多算一遍 ✗）。
    """
    expected = model.state_dict()
    tied = _tied_keys(model)
    loaded: dict[str, tuple[int, ...]] = {}
    unknown: list[str] = []
    if materializer is not None:
        ok, why = materializer.valid()
        if ok:
            with materializer.reader() as reader:
                _fill_from_cache(reader, info, config, expected, loaded, unknown)
            _check_accounting(expected, loaded, unknown, tied=tied)
            total = sum(expected[name].numel() * expected[name].element_size() for name in loaded)
            return materialize_mod.Report(
                hit=True, path=_receipt_path(materializer), reason=why,
                gib=materialize_mod.gib(total), tensor_count=len(loaded),
                notes=("命中产物 ⇒ 反量化 0 次 ✓（省的是 CPU 与 fp32 中间量 ✓）",))

    writer = None
    note = ""
    if materializer is not None:
        #: ⚠️ 计划在 :func:`_open_materializer` 里算过一次 ✓，这里**重算**只为拿到那张表 ✓ ——
        #: 纯元数据（不读盘 ✓ 不反量化 ✓）且必然同结果 ✓；宁可重算，也不把计划挂到把手上当状态 ✗。
        specs, _label = _plan_specs(info, config, expected)
        if specs:
            try:
                writer = materializer.open_writer(specs)
            except materialize_mod.MaterializeSkip as err:
                note = str(err)       # 余量/配额不够 ✓ ⇒ 退化成「慢」✓ 不拦装载 ✓✗
    if writer is not None:
        with writer:
            _fill_by_dequant(path, info, config, expected, loaded, unknown, writer)
        report = writer.report
    else:
        _fill_by_dequant(path, info, config, expected, loaded, unknown, None)
        report = None
    _check_accounting(expected, loaded, unknown, tied=tied)
    if materializer is None:
        return None
    return report or materialize_mod.Report(hit=False, path=_receipt_path(materializer),
                                            reason=note or "本次不物化 ✓")


def _tensor_bytes(weight: Any) -> bytes:
    """张量**原始字节** ✓（低精度取 ``.view(int16)`` 的位模式 ✓ —— 不是「转 numpy 浮点再自己打包」✗：
    后者要自己实现舍入 ✓✗，而 RNE 与向零差 1 ulp 就是「同一个问题两个答案」✗）。"""
    import torch  # noqa: PLC0415

    flat = weight.detach().reshape(-1).contiguous()
    if flat.dtype in (torch.bfloat16, torch.float16):
        return flat.view(torch.int16).numpy().tobytes()
    return flat.numpy().tobytes()


def _fill_from_cache(reader: Any, info: Any, config: llm_mod.LlmConfig,
                     expected: dict[str, Any], loaded: dict[str, tuple[int, ...]],
                     unknown: list[str]) -> None:
    """从产物读 ✓（**不反量化** ✗）：字节 ⇒ ``view`` 回目标 dtype ⇒ ``copy_`` ✓。

    ⚠️ 字节是 :func:`_tensor_bytes` 写进去的**位模式** ✓ ⇒ ``view`` 回来就是**同一个张量** ✓；
    任何「再 cast 一次」都是白扔精度 ✓✗。
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        for name, tensor in info.tensors.items():
            target = map_name(name, tie_embeddings=config.tie_embeddings,
                              qk_norm=config.qk_norm, qkv_bias=config.qkv_bias)
            if target is None:
                continue
            if target not in expected:
                unknown.append(f"{name} → {target}")
                continue
            want = expected[target]
            try:
                raw = reader.read(name)
            except Exception as err:  # noqa: BLE001 —— 缺 key / 读坏 ⇒ 都是「不能半装」✗
                raise GgufDequantError(
                    f"物化产物里读不出 {name!r} ✗（{type(err).__name__}: {err}）"
                    f"⇒ 产物与源文件对不上（键集应当一致 ✓）⇒ 不许半装 ✗") from err
            need = want.numel() * want.element_size()
            if len(raw) != need:
                raise GgufDequantError(
                    f"物化产物里 {name!r} 字节数不符 ✗：{len(raw)} ✓ 而模型要 {need} ✓"
                    f"（dtype·形状对不上 ⇒ 产物作废 ✓）")
            weight = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            weight = weight.view(want.dtype).reshape(tuple(want.shape))
            want.copy_(weight)
            loaded[target] = tuple(want.shape)
            del raw, weight


def _fill_by_dequant(path: str | Path, info: Any, config: llm_mod.LlmConfig,
                     expected: dict[str, Any], loaded: dict[str, tuple[int, ...]],
                     unknown: list[str], writer: Any) -> None:
    """反量化装配 ✓（``writer`` 非空时顺手把**装进去的字节**写进产物 ✓）。"""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    with torch.no_grad():
        for name, tensor in info.tensors.items():
            target = map_name(name, tie_embeddings=config.tie_embeddings,
                              qk_norm=config.qk_norm, qkv_bias=config.qkv_bias)
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
            if writer is not None:
                #: ⭐ 物化的字节 = **刚 copy_ 进去那张张量的位模式** ✓ ⇒ 命中时逐位相同是构造的 ✓✗
                writer.add(name, _tensor_bytes(weight))
            del arr, weight


#: ⭐ 共享权重（tie embeddings ✓）：GGUF 里**省掉**的那张张量由哪根参数顶上 ✓。
#: ⚠️ 2026-11-07 补（真机 ``Qwen3-4B-Instruct-2507-GGUF`` 撞出来 ✓✗）：``nn.Module.state_dict()``
#: 会把同一张 Parameter 在 ``embed.weight`` 与 ``lm_head.weight`` **两个名字下各列一次** ✗
#: ⇒ tie 模型（文件里本来就没有 ``output.weight``）必然被报「缺 ``lm_head.weight``」✗✗ ——
#: 而它其实就是 ``embed.weight`` 本人 ✓。放行凭据是**对象同一性** ✓，不是配置位 ✗。
_TIED_KEYS: dict[str, str] = {"lm_head.weight": "embed.weight"}


def _tied_keys(model: Any) -> dict[str, str]:
    """这个模型里**真**共享的键 ✓ ⇒ ``{满足不了的键: 满足它的键}`` ✓（不共享 ⇒ 空 dict ✓）。

    ⚠️ 只认 ``model.lm_head.weight is model.embed.weight`` ✓：配置位写 tie 而实物没 tie 时
    **不放行** ✗（那样会把「权重真缺了」伪装成「已经共享了」✗✗）。
    """
    weight = getattr(getattr(model, "lm_head", None), "weight", None)
    if weight is not None and weight is getattr(getattr(model, "embed", None), "weight", None):
        return dict(_TIED_KEYS)
    return {}


def _check_accounting(expected: dict[str, Any], loaded: dict[str, tuple[int, ...]],
                      unknown: list[str], *, tied: dict[str, str] | None = None) -> None:
    """键集对账 ✓（等价 ``strict=True`` ✗：**多**与**缺**都当场报 ✓✗）。

    ``tied``：真共享的键 ⇒ 由另一根参数顶上 ✓（见 :func:`_tied_keys` ✓，默认空 ⇒ 老行为不动 ✓）。
    """
    unexpected = sorted(unknown)
    if unexpected:  # 防御：map_name 产出的名字必须都在模型里 ✓（漏了就是映射表写错了 ✗）
        raise GgufDequantError(
            f"{len(unexpected)} 个张量映射到了模型里**没有**的参数 ✗（映射表与模型对不上 ✗）："
            f"{unexpected[:8]}{' …' if len(unexpected) > 8 else ''}")
    tied = tied or {}
    missing = sorted(name for name in set(expected) - set(loaded)
                     if not (name in tied and tied[name] in loaded))
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


def tie_embeddings_of(declared: Any, tensor_names: Iterable[str] | None = None) -> bool:
    """这张 GGUF 到底是不是**共享输入/输出词嵌入** ✓ —— 以**张量表**为准，元数据只当声明 ✓。

    ⚠️ 2026-11-07 真机撞出来（``Qwen3-4B-Instruct-2507-GGUF`` 的 Q4_K_M ✓✗）：那份文件
    **既没有** ``qwen3.tie_embeddings`` 元数据键 ✗、**也没有** ``output.weight`` 张量 ✗
    （llama.cpp 对 tie 模型允许省掉那份权重 ✓）。原实现只看元数据、缺键按 ``False`` ✗
    ⇒ 模型建出独立的 ``lm_head`` ⇒ 键集对账要求 ``lm_head.weight`` ⇒ 报「缺 1 个张量 ✗」
    ⇒ 一个**完好的文件**被判成装不了 ✗✗。

    判据只有一条、且不是猜 ✓：**没有 ``output.weight`` 就只能 tie** ✓（非 tie 的模型，那份
    权重没有任何别的地方能来 ✗）。三种落法：
    · 有 ``output.weight`` ⇒ 跟随声明 ✓（缺声明按 ``False`` ✓ —— 真要 tie 时那份权重与词表
      逐位相同 ✓，装成独立 ``lm_head`` 数值等价 ✓）；
    · 没有 ``output.weight``、有 ``token_embd.weight`` ⇒ ``True`` ✓；
    · 没有 ``output.weight`` 而声明还写着 ``False`` ⇒ **具名拒绝** ✗（自相矛盾，不猜 ✗）。
    ``tensor_names=None``（只给元数据 ✓）⇒ 老行为一字不动 ✓（看不到张量表就不假设 tie ✗）。
    """
    names = None if tensor_names is None else set(tensor_names)
    if names is not None and "token_embd.weight" in names and "output.weight" not in names:
        if declared is False:
            raise GgufDequantError(
                "GGUF 元数据声明 tie_embeddings=False ✗，但文件里**没有** output.weight ✗ "
                "⇒ 自相矛盾（lm_head 的权重无处可来 ⇒ 装不了 ✗）；不猜 ✗")
        return True
    return bool(declared)


def infer_llm_config(metadata: dict[str, Any], *, arch: str | None = None,
                     tensor_names: Iterable[str] | None = None) -> llm_mod.LlmConfig:
    """从 GGUF 元数据推导 :class:`llm.LlmConfig` ✓（架构参数**全从权重读** ✓ 不猜 ✗）。

    ⚠️ GGUF 的架构参数前缀是 ``general.architecture``（``llama`` / ``qwen2`` / ``qwen3`` ✓），
    字段名同族：``{arch}.embedding_length`` / ``{arch}.block_count`` / ``{arch}.attention.head_count``
    / ``{arch}.attention.head_count_kv``（GQA ✓）/ ``{arch}.feed_forward_length`` / ``{arch}.rope.freq_base``
    / ``{arch}.attention.layer_norm_rms_epsilon`` 等 ✓（Qwen3 GGUF 偶尔沿用 ``llama.`` 前缀 ⇒ 回退读 ✓）。

    ⚠️ ``tie_embeddings`` **不能只读元数据** ✗（真文件常常两个都不给 ✓✗）⇒ 传 ``tensor_names``
    让 :func:`tie_embeddings_of` 按张量表定夺 ✓；不传 ⇒ 退回「只信声明」的老行为 ✓。

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
        tie_embeddings=tie_embeddings_of(_get("tie_embeddings"), tensor_names),
        pad_id=int(pad) if pad is not None else 0,
        eos_id=int(eos) if eos is not None else None,
    )
