"""自研 decoder-only LLM 推理（2026-09-25 起）。

## 为什么有这一层

本仓文本生成此前走 **ollama**（本地 11434 的 HTTP 服务）✗ —— 那是**调用**，不是**实现** ✗。
本模块把它换成**自己实现**的 decoder-only transformer ✓：RMSNorm + GQA（分组查询注意力）+
RoPE + SwiGLU + **因果掩码** + **KV cache** ✓ —— Qwen3 / Llama 这类 LLM 的架构 ✓。

## 与其它模块的分工（不重叠 ✓）

* :mod:`tokenizer_bpe` / :mod:`tokenizer_hub` —— 文本 → token id ✓（词表是**权重的一部分** ✓）；
* :mod:`gguf` —— 读 GGUF 元数据（张量表 / 量化方案 ✓；**张量反量化见后续** ✓）；
* 本模块 —— token id → **逐 token 生成**（前向 + KV cache + 采样 ✓）；
* :mod:`quant` —— 低精度权重反量化（GGUF 的 k-quant 尚未接 ✓ 已写进缺口 ✓）。

## 两条硬约束（本仓纪律 ✓）

1. **架构参数全部显式** ✗（:class:`LlmConfig` ✓）—— 不写死 Qwen3 的具体值 ✓，从权重/配置读 ✓；
2. **懒导入 torch** ✓（模块级不 import ✓ ⇒ 没装也能读配置/常量 ✓）。

## 验证

自检（``engine_llm_test`` ✓）用**缩小版**模型走**同一条**前向 ✓（本仓纪律 ✓）：
因果掩码 ✓ / RoPE 旋转不变量 ✓ / GQA 分组 ✓ / KV cache 逐 token 不变量 ✓ / 采样形状 ✓。

## 一台机器、两种设备（2026-10-29 补 ✓）

**同一份**前向代码在 CPU 与 CUDA 上跑 ✓（不写两套 ✗）：设备只由 ``torch.device`` 决定 ✓，
算子全是设备无关的 ✓（``arange``/``full``/``softmax``/``matmul``/``multinomial`` ✓ —— 每处都从
**输入张量**取 ``device`` ✓，没有一个地方写死 ``"cpu"`` ✓）。

三条口径（都是「看着能跑、其实不一样」的形状 ✓✗）：

1. **``auto`` 才允许挑** ✓：``device=None/"auto"`` ⇒ 有卡用卡、没卡用 CPU ✓；**显式要 ``cuda`` 而
   用不了 ⇒ 具名拒绝** ✗✗（静默回落 CPU 会让「8B 慢」被当成模型慢 ✓✗）；
2. **默认 fp32** ✓：CPU/GPU 的 fp32 差异只在归约顺序（~1e-5 ✓）⇒ 能当**硬判据** ✓；
   要省显存才显式 ``dtype="bfloat16"`` ✓（那时一致性只能按容差判 ✓）；
3. **低精度不在算子缝里炸** ✗✗：``RMSNorm`` 升 fp32 再降回 ✓、cos/sin 降到输入精度 ✓、
   因果掩码用 ``scores.dtype`` ✓、采样统一走 fp32 ✓ —— 这四处漏一个就是
   ``expected scalar type BFloat16 but found Float`` ✓（**只在 bf16/fp16 下炸，fp32 全绿** ✗✗）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "KV_GROUP",
    "KV_MODES",
    "LlmConfig",
    "LookupDrafter",
    "Q8Kv",
    "SPEC_MIN_MATCH",
    "SPEC_WINDOW",
    "apply_layer_plan",
    "apply_rope",
    "available_devices",
    "build_llm",
    "dequantize_kv",
    "device_facts",
    "generate_llm",
    "KvPrefixCache",
    "kv_bytes_per_token",
    "kv_cache_bytes",
    "layer_weights_bytes",
    "materialize_kv",
    "model_device",
    "plan_layer_devices",
    "primitives",
    "quantize_kv",
    "repeat_kv",
    "resolve_device",
    "resolve_dtype",
    "resolve_kv_cache_bytes",
    "resolve_kv_mode",
    "resolve_spec_lookup",
    "resolve_spec_min_match",
    "resolve_spec_window",
    "rope_freqs",
    "truncate_kv",
    "update_kv_cache",
]


class LlmError(ValueError):
    """LLM 配置/输入不合法 ✓（**明确报错** ✗ 不静默猜 ✓）。"""


#: :func:`_build` 的积木缓存 ✓（见它 docstring：不缓存 ⇒ 每次拿到的是**新类** ⇒ ``isinstance`` 恒假 ✗✗）
_PRIMITIVES: dict[str, Any] | None = None


@dataclass(frozen=True)
class LlmConfig:
    """decoder-only LLM 的**显式**结构参数 ✓（不猜具体模型 ✗）。

    ⚠️ ``kv_heads`` 是 **GQA** 的关键 ✓：``kv_heads < heads`` 表示分组查询注意力
    （若干 query 头共享一组 KV ✓）；``kv_heads == heads`` 退化为标准 MHA ✓。
    """

    vocab_size: int
    hidden: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    #: ⭐ **Qwen3 的 per-head QK 归一化**（2026-10-29 补 ✓）：真机 ``qwen3:8b`` 的 GGUF 里每层都带
    #: ``blk.N.attn_q_norm.weight`` / ``attn_k_norm.weight``（形状 ``(head_dim,)`` ✓ = 128 ✓），
    #: 且**没有**任何 metadata 键来声明它 ✗ ⇒ 只能由「张量表里有没有这两项」判定 ✓
    #: （见 :func:`gguf_to_llm.load_llm_from_gguf` 的自动检测 ✓，不猜 ✗）。
    #: 缺这条时：399 个张量里有 72 个映射不上 ✗（36 层 × 2 ✓）⇒ 8B/14B 一律装不进 ✗✗。
    qk_norm: bool = False
    #: ⭐ **q/k/v 的偏置**（2026-10-31 补 ✓）：Qwen2 / Qwen2.5 的 ``q_proj`` / ``k_proj`` / ``v_proj``
    #: 都带 ``bias`` ✓（``o_proj`` 与 MLP 三项**没有** ✗）—— 盘上 ``qwen_2.5_vl_7b_fp8_scaled``
    #: 每层正好三根 ``*.bias``（``(3584,)`` / ``(512,)`` × 2 ✓）。
    #: ⚠️ 漏了这条 ⇒ 那三根键**无处可放** ✓✗（装载报缺键 ✓；若强行忽略 ⇒ 全是「少一个常数」的偏差 ✓，
    #: 形状全对、不报错、提示词跟随明显变弱 ✓✗）。Qwen3 / Llama 一律 ``False`` ✓（它们真的没有 bias ✓）。
    qkv_bias: bool = False
    pad_id: int = 0
    eos_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise LlmError(f"vocab_size 必须 > 1（收到 {self.vocab_size} ✗）")
        for name, value in (("hidden", self.hidden), ("depth", self.depth),
                            ("heads", self.heads), ("head_dim", self.head_dim),
                            ("ffn", self.ffn), ("kv_heads", self.kv_heads)):
            if value <= 0:
                raise LlmError(f"{name} 必须 > 0（收到 {value} ✗）")
        if self.heads % self.kv_heads != 0:
            raise LlmError(f"GQA：heads（{self.heads}）必须被 kv_heads（{self.kv_heads}）整除 ✗")
        if self.head_dim % 2 != 0:
            raise LlmError(f"head_dim 必须是偶数（RoPE 配对旋转 ✗ 收到 {self.head_dim} ✗）")
        if self.rope_theta <= 0 or self.rms_eps <= 0:
            raise LlmError("rope_theta / rms_eps 必须 > 0 ✗")


# ===== 设备 / 精度口径（2026-10-29 补 ✓ —— CPU 与 GPU **同一份前向** ✓）=================
#
# 事实来源（**只参考行为、没有复制代码** ✓）：
# * ``llama.cpp``（MIT，``ggml/src/ggml-cuda`` 与 ``src/llama-model.cpp``）—— 同一份 ggml 图在
#   CPU 后端与 CUDA 后端上跑 ✓，这也是「**同一套算子、两种设备**」这个形状的出处 ✓；
# * ``huggingface/transformers``（Apache-2.0，`LlamaRMSNorm` / `Qwen3RMSNorm`）—— 低精度权重下
#   **norm 内部升 fp32 再降回** 的写法 ✓（bf16 直接 `mean` 会掉精度 ✓）；
# * PyTorch 文档（BSD-3）—— ``Module.to(device=…, dtype=…)``、``set_default_dtype`` 的语义 ✓。
# 本模块的实现在上面三处之外**自己写** ✓：参数解析、拒绝口径、一致性判据都是本仓的形状 ✓。

#: 精度别名 → 规范名 ✓（只收真能跑的三种 ✓；``int8``/``fp8`` 这类权重精度**不在这里** ✗ —— 那是反量化的事 ✓）
_DTYPE_ALIASES: dict[str, str] = {
    "float32": "float32", "fp32": "float32", "float": "float32", "single": "float32",
    "float16": "float16", "fp16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
}


def device_facts() -> dict[str, Any]:
    """本机**实测**的设备事实 ✓（没有 torch 也照答 ✓ 不抛 ✗）。"""
    try:
        import torch  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001 —— 可选依赖 ✓
        return {"torch": False, "torchVersion": "", "cuda": False, "cudaCount": 0,
                "names": [], "devices": ["cpu"], "reason": f"没有 torch ✗（{type(err).__name__}）"}
    cuda = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda else 0
    names: list[str] = []
    for index in range(count):
        try:
            names.append(str(torch.cuda.get_device_name(index)))
        except Exception:  # noqa: BLE001 —— 拿不到型号也不许把事实搞崩 ✗
            names.append("?")
    return {"torch": True, "torchVersion": str(torch.__version__), "cuda": cuda,
            "cudaCount": count, "names": names,
            "devices": ["cpu"] + [f"cuda:{index}" for index in range(count)], "reason": ""}


def available_devices() -> list[str]:
    """本机**能用**的设备名 ✓（``["cpu", "cuda:0", …]`` ✓ —— 顺序即优先级 ✓）。"""
    return list(device_facts()["devices"])


def resolve_device(spec: Any = None) -> str:
    """把 ``auto`` / ``cpu`` / ``cuda`` / ``cuda:1`` 解析成**具体**设备名 ✓（``"cuda:0"`` ✓）。

    ⚠️⚠️ **探不到就具名拒绝，绝不静默回落 cpu** ✗✗：显式点名 ``cuda`` 的调用方要的是「用卡」✓，
    悄悄给 CPU 会让「8B 跑得慢」被当成模型慢 ✗（本仓老账：好默认值把未实现伪装成已实现 ✗）。
    ``auto``（默认）才做「有卡用卡、没卡用 CPU」的**择优** ✓ —— 那是唯一允许挑的地方 ✓。
    """
    text = "auto" if spec is None else str(spec).strip()
    if text.lower() in ("", "auto", "none", "default"):
        text = "auto"
    try:
        import torch  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001
        if text.lower() == "cpu":
            return "cpu"
        raise LlmError(f"没有 torch ⇒ 只能用 cpu ✗（请求的是 {spec!r} ✗ —— 不静默回落 ✗）"
                       ) from err
    if text.lower() == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(text)
    except (RuntimeError, ValueError) as err:
        raise LlmError(f"设备 {spec!r} 不认 ✗（可写 auto / cpu / cuda / cuda:1 ✓；"
                       f"本机可用：{available_devices()} ✓）") from err
    # ⚠️ **真探一次** ✓（`cuda.is_available()` 为真也可能选到不存在的卡号 ✓；mps/xpu 也没有统一探测口 ✗）
    try:
        concrete = torch.zeros(1, device=device)
    except Exception as err:  # noqa: BLE001 —— 任何探测失败都算「用不了」✓
        raise LlmError(f"设备 {text} 用不了 ✗（{type(err).__name__}: {err}）"
                       f"⇒ 本机可用：{available_devices()} ✓（**不静默回落 cpu** ✗）") from err
    return str(concrete.device)


def resolve_dtype(spec: Any = None, *, device: str = "cpu") -> Any:
    """把精度别名解析成 ``torch.dtype`` ✓（``None``/``auto`` ⇒ **看设备挑** ✓，见下）。

    ⚠️ **``auto`` 在两种设备上给的不是同一个精度** ✓，这是**有意的** ✗✗：
    * CPU ⇒ **float32** ✓（CPU 上 bf16 只剩慢路径 ✓，fp32 才是常规 ✓）；
    * CUDA ⇒ **bfloat16**（卡支持 bf16 时 ✓，Ampere+ 都支持 ✓）：8B 的 fp32 权重 **32 GB** ✗，
      24 GB 卡根本装不下 ✓ ⇒ 默认 fp32 会让「GPU 模式」直接 OOM ✗✗（不是慢，是跑不起来 ✗）。
    * 显式 ``dtype="float32"`` ⇒ 两端都 fp32 ✓：这是**跨设备逐位对齐**的硬判据口径 ✓
      （差异只在归约顺序 ~1e-5 ✓）；代价是显存按 fp32 算 ✓。
    ⚠️ CPU 上 fp16 直接拒 ✗（一半算子不支持/退化到慢路径 ✓）—— 要省内存用 bfloat16 ✓。
    """
    try:
        import torch  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001
        raise LlmError(f"没有 torch ⇒ 谈不上精度 ✗（{type(err).__name__}）") from err
    if isinstance(spec, torch.dtype):
        # ⚠️ **幂等** ✓：后端先把请求解析成 dtype，再交给装载/建模型 ⇒ 会二次解析 ✓
        # 不许把已经解析好的 ``torch.float32`` 当成「不认识的字符串」拒掉 ✗✗（实测就在这条上炸过 ✓）。
        if spec is torch.float16 and torch.device(device).type == "cpu":
            raise LlmError(f"CPU 上 float16 不做 ✗（设备 {device} ✓ ⇒ 用 float32 ✓ 或 bfloat16 ✓）")
        return spec
    key = "auto" if spec is None else str(spec).strip().lower()
    if key in ("", "none", "default"):
        key = "auto"
    if key == "auto":
        if torch.device(device).type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:  # noqa: BLE001 —— 老版本 torch 没这个函数 ✓ 退回 fp32 ✓
                pass
        return torch.float32
    name = _DTYPE_ALIASES.get(key)
    if name is None:
        raise LlmError(f"精度 {spec!r} 不认 ✗（可写 auto / float32 / bfloat16 / float16 ✓；"
                       f"int8/fp8 是**权重**精度 ✓ 由反量化负责 ✓ 不是这里的参数 ✗）")
    dtype = getattr(torch, name)
    if dtype is torch.float16 and torch.device(device).type == "cpu":
        raise LlmError(f"CPU 上 float16 不做 ✗（设备 {device} ✓ ⇒ 用 float32 ✓ 或 bfloat16 ✓）")
    return dtype


def model_device(model: Any) -> Any:
    """模型**实际**在哪块设备上 ✓（取第一个参数的 ``device`` ✓ 不猜 ✗）。"""
    try:
        for parameter in model.parameters():
            return parameter.device
    except Exception:  # noqa: BLE001 —— 没有参数的假模型 ✓
        pass
    return _build()["torch"].device("cpu")


def _block_param_count(config: LlmConfig) -> int:
    """一层 :class:`TransformerBlock` 的**参数量** ✓（**算出来** ✓ 不抄表 ✗）。

    与 :class:`TransformerBlock` / :class:`GQAAttention` / :class:`SwiGLU` 的构造**逐项对应** ✓：
    ``q``/``k``/``v``/``o`` 四个投影 ✓、``qkv_bias`` 的偏置 ✓、``qk_norm`` 的两根 ``(head_dim,)`` ✓、
    两条 RMSNorm 的 ``(hidden,)`` ✓、SwiGLU 的三项 ✓。
    """
    q_dim = config.heads * config.head_dim
    kv_dim = config.kv_heads * config.head_dim
    attention = (config.hidden * q_dim          # q_proj
                 + 2 * (config.hidden * kv_dim)  # k_proj / v_proj
                 + q_dim * config.hidden)        # o_proj
    if config.qkv_bias:
        attention += q_dim + 2 * kv_dim
    if config.qk_norm:
        attention += 2 * config.head_dim
    return attention + 2 * config.hidden + 3 * config.hidden * config.ffn


def layer_weights_bytes(config: LlmConfig, dtype_bytes: int = 2) -> int:
    """**一层权重**的字节数 ✓（= 参数量 × 精度字节 ✓ —— 后端算预算就读这一个口子 ✓）。

    ⚠️ 为什么单列一个公开函数 ✗✗：预算计算散在「后端 / 预检 / 回执」三处各写一遍乘加 ⇒ 结构一变
    三处漂移 ✓（一处漏掉 ``qkv_bias`` ⇒ 预算少算、装到一半 OOM ✓✗）。
    """
    if int(dtype_bytes) <= 0:
        raise LlmError(f"dtype_bytes 必须 > 0 ✗（收到 {dtype_bytes!r}）")
    return _block_param_count(config) * int(dtype_bytes)


#: KV cache 的存储口径 ✓（``fp`` = 原样浮点 ✓ 老路径 ✓；``q8`` = INT8 分组量化 ✓ 2026-11-08 补 ✓）
KV_MODES: tuple[str, ...] = ("fp", "q8")

#: q8 的**分组宽度** ✓（沿 ``head_dim`` 每 64 个元素共用一个 scale ✓）。
#: ⚠️ 这是 Strata ``kv_q8.hpp`` 同款口径 ✓ —— 64 是「组内量程差异」与「scale 自身开销」的折中 ✓：
#: 32 ⇒ scale 多一倍（相对开销 +6.25% ⇒ 见 :func:`kv_bytes_per_token` 的账 ✓）；128 在长上下文上
#: 组内离群值会把整组压坏 ✓。**改这个数必须同时改判据** ✗（测试按它逐字节核对）。
KV_GROUP: int = 64


def resolve_kv_mode(mode: Any) -> str:
    """KV 存储口径归一 ✓（``None``/空/``off``/``fp`` ⇒ ``fp`` ✓；``q8``/``int8`` ⇒ ``q8`` ✓）。

    ⚠️ 拼错**当场拒** ✗✗：``"q08"`` 静默当 ``fp`` ⇒ 人会以为「开了量化还这么占显存」（其实是没开 ✓✗），
    反过来把 ``"fp"`` 当 ``q8`` 更糟 —— 数值悄悄变差、没人知道是量化干的 ✓✗。
    """
    raw = "" if mode is None else str(mode).strip().lower()
    if raw in ("", "off", "none", "false", "no", "fp", "float", "fp16", "bf16"):
        return "fp"
    if raw in ("q8", "int8", "8bit", "on", "true", "yes", "auto"):
        return "q8"
    raise LlmError(f"kvMode 不认 {mode!r} ✗（可写 off / fp / q8 ✓）")


def kv_bytes_per_token(config: LlmConfig, mode: Any = "fp", *,
                       dtype_bytes: int = 2, group: int = KV_GROUP) -> dict[str, Any]:
    """**每 token** 的 KV cache 字节数 ✓（``fp`` 与 ``q8`` 同一把尺子 ✓ —— 回执只报这个数 ✓）。

    口径（**算**出来 ✓ 不抄表 ✗）：

    * K/V 各 ``kv_heads × head_dim`` 个元素／层 ✓，全模型 ``depth`` 层 ✓；
    * ``fp``：元素 × ``dtype_bytes`` ✓；
    * ``q8``：元素 × 1 B（int8 ✓）＋ 每个 64 元素组一份 fp32 scale ✓
      ⇒ 每 token 每层 ``ceil(head_dim/group) × kv_heads × 4`` B 的额外开销 ✓。

    ⚠️ 为什么把「省了多少」算成字节而不是百分比 ✗✗：上下文长度是变量 ✓ —— 只有 per-token 字节
    能与「目标 32K tokens」这类真实问题相乘 ✓（百分比读不出「还剩几个 token 的余量」✓✗）。
    """
    if int(dtype_bytes) <= 0:
        raise LlmError(f"dtype_bytes 必须 > 0 ✗（收到 {dtype_bytes!r}）")
    if int(group) <= 0:
        raise LlmError(f"group 必须 > 0 ✗（收到 {group!r}）")
    chosen = resolve_kv_mode(mode)
    elements = config.kv_heads * config.head_dim          # 每层 K（或 V）的元素数 ✓
    groups = -(-config.head_dim // int(group))            # ceil ⇒ 除不尽时最后一组也占一份 scale ✓
    scale_bytes = 0
    element_bytes = int(dtype_bytes)
    if chosen == "q8":
        element_bytes, scale_bytes = 1, groups * config.kv_heads * 4
    per_side = elements * element_bytes + scale_bytes      # 每层、K 或 V 一侧 ✓
    fp_side = elements * int(dtype_bytes)
    total = 2 * per_side * config.depth                    # K ＋ V × 层数 ✓
    fp_total = 2 * fp_side * config.depth
    return {
        "mode": chosen,
        "group": int(group) if chosen == "q8" else 0,
        "kBytes": per_side,
        "vBytes": per_side,
        "totalBytes": total,
        "fpTotalBytes": fp_total,
        #: ``fp`` 口径下恒为 1.0 ✓（不写 ``None`` ✗ —— 回执里一个数比一句「不适用」好读 ✓）
        "ratio": round(total / fp_total, 4) if fp_total else None,
        "note": (f"每 token {total} B（{'INT8 分组 %d 量化' % group if chosen == 'q8' else '浮点'} ✓；"
                 f"fp 口径 {fp_total} B ⇒ 省 {round((1 - total / fp_total) * 100, 1)}% ✓"
                 if fp_total else ""),
    }


#: 跨调用 KV 前缀复用的**默认字节预算** ✓（512 MiB ⇒ q8 口径约 6700 token ✓，见 :func:`kv_cache_bytes`）。
#: ⚠️ 为什么给预算而不是「全留」✗✗：常驻 cache 是**跨调用**活着的 ✓ —— 不设上限时它会跟着最长那次
#: 上下文一路涨到卡满 ✓，而卡满的报错会落在**下一张图**上（离真因十几分钟 ✓✗）。
KV_CACHE_DEFAULT_BYTES = 512 * 2 ** 20

_BYTES_UNITS = {"": 1, "b": 1, "k": 1024, "kb": 1024, "kib": 1024, "m": 2 ** 20, "mb": 2 ** 20,
                "mib": 2 ** 20, "g": 2 ** 30, "gb": 2 ** 30, "gib": 2 ** 30}


def kv_group_for(head_dim: int, group: int = KV_GROUP) -> int:
    """**实际生效的组宽** ✓ = ``min(group, head_dim)`` ✓（2026-11-08 补 ✓）。

    ⚠️ 为什么要夹 ✗✗：组宽 > ``head_dim`` 时（小模型 ``head_dim=8`` ✓ 常见 ✓）整条头只剩**一组**
    ✓，尾部 56 个通道全是对齐用的**零** ✓ ⇒ 量化后的宽度被撑到 64 ✓（实测 8 个通道的 cache 变成
    int8 64 个通道 ＋ scale ✓ 比 fp 还大 ✓✗ —— ⑨″ 就是被这个咬了一口 ✓）。
    夹到 ``head_dim`` 后：真模型（``head_dim=128 > 64`` ✓）行为**一字不变** ✓，小模型退回「一条头一组」✓。
    """
    g = int(group or 0)
    if g <= 0:
        raise LlmError(f"KV 组宽必须 > 0 ✗（收到 {group!r}）")
    d = int(head_dim or 0)
    if d <= 0:
        raise LlmError(f"KV 的 head_dim 必须 > 0 ✗（收到 {head_dim!r}）")
    return min(g, d)


def kv_cache_bytes(past: Any) -> int:
    """数**一段真实 cache** 占多少字节 ✓（``q8`` 段 = int8 数据 ＋ fp32 scale ✓，``fp`` 段 = 原样 ✓）。

    ⚠️ 为什么不按「每 token 字节 × 长度」估 ✗✗（:func:`kv_bytes_per_token` 那个数是**预算** ✓）：
    预算算的是「装满会占多少」✓，这里要答「**现在**真占多少」✗ —— quantize 的尾部 pad、拼接过中间态、
    分组数取整都会让两者差一点点 ✓，而回执里报的是后者 ⇒ 必须逐张量数 ✓（实测差 0 ✓ 见 §7.2.1 ✓）。
    """
    total = 0
    for layer in past or ():
        for item in layer:
            if hasattr(item, "q"):                  # :class:`Q8Kv` ✓
                total += int(item.q.numel() * item.q.element_size())
                total += int(item.scale.numel() * item.scale.element_size())
            else:
                total += int(item.numel() * item.element_size())
    return total


def resolve_kv_cache_bytes(value: Any,
                           *, default: int = KV_CACHE_DEFAULT_BYTES) -> int:
    """把 ``kvCacheBytes`` 这个旋钮归一成**字节数** ✓（``0`` = 关 ✓）。

    可写（大小写/空格不敏感 ✓）：``off``/``0``/``none``/``false`` ⇒ 0 ✓；``on``/``auto``/``default``
    或**不写** ⇒ ``default`` ✓；``536870912`` ⇒ 原值 ✓；``"512MiB"``/``"1GiB"``/``"256m"`` ⇒ 按单位换 ✓。

    ⚠️ 拼错**当场拒** ✗✗（``"512MB"`` 之外的乱写若悄悄当 0 ⇒ 人以为开着复用 ✓ 实际一次没命中 ✗✗）。
    """
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(default) if value else 0
    if isinstance(value, (int, float)):
        if int(value) < 0:
            raise LlmError(f"kvCacheBytes 不能为负 ✗（收到 {value!r}）")
        return int(value)
    raw = str(value).strip().lower().replace("_", "")
    if raw in ("", "on", "true", "yes", "auto", "default"):
        return int(default)
    if raw in ("off", "false", "no", "none", "disabled", "disable"):
        return 0
    digits = raw.rstrip("bkmigb") if raw[-1:].isalpha() else raw
    unit = raw[len(digits):]
    try:
        number = float(digits)
    except ValueError as err:
        raise LlmError(f"kvCacheBytes 不认 {value!r} ✗（可写 0 / off / N / 512MiB / 1GiB ✓）") from err
    if unit not in _BYTES_UNITS:
        raise LlmError(f"kvCacheBytes 的单位 {unit!r} 不认 ✗（可用 b/k/kb/kib/m/mb/mib/g/gb/gib ✓）")
    total = int(number * _BYTES_UNITS[unit])
    if total < 0:
        raise LlmError(f"kvCacheBytes 不能为负 ✗（收到 {value!r}）")
    return total


#: 投机解码的默认窗口 ✓（一拍最多吃 ``1 + SPEC_WINDOW`` 行 ✓）：8 与参考实现
#: ``spec::SuffixDrafter`` 的经验值同档 ✓。**调大不一定更快** ✗ —— 全接受的收益是线性的，
#: 而每多一行就多一份被拒后白算的 KV ✓（lookup 草稿的命中率随长度掉得很快 ✓）。
SPEC_WINDOW: int = 8
#: 后缀匹配的**最短**长度 ✓（3 ＝「中文短语／代码行」的经验值 ✓，见 ``SuffixDrafter`` ✓）。
SPEC_MIN_MATCH: int = 3


def resolve_spec_lookup(value: Any) -> bool:
    """投机解码开关归一 ✓（``None``/空/``off``/``0`` ⇒ **关** ✓ 老路径 ✓；``on``/``true``/``1`` ⇒ 开 ✓）。

    ⚠️ 默认**关** ✗ 不是「忘了开」：本仓实测收益 1.02~1.03× ✓（逐字复述时最长匹配贴着当前末尾
    ⇒ 匹配越准越不接受 ✓ 这是 lookup 的结构性盲区 ✓），且**只在 ``temperature=0``** 成立 ✓
    ⇒ 随机采样口开了也只会在回执里报 ``enabled=False`` ✓。
    ⚠️ 拼错**当场拒** ✗✗（``"ture"`` 静默当关 ⇒ 人以为开了投机 ✓ 实际一拍一 token ✗）。
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    raw = str(value).strip().lower()
    if raw in ("", "off", "false", "no", "none", "disabled", "disable", "0"):
        return False
    if raw in ("on", "true", "yes", "1", "auto"):
        return True
    raise LlmError(f"specLookup 不认 {value!r} ✗（可写 off / on / true / false ✓）")


def resolve_spec_window(value: Any, *, default: int = SPEC_WINDOW) -> int:
    """草稿窗口归一 ✓（不写 ⇒ ``default`` ✓；``< 1`` **当场拒** ✗）。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return int(default)
    try:
        window = int(value)
    except (TypeError, ValueError) as err:
        raise LlmError(f"specWindow 不认 {value!r} ✗（要正整数 ✓）") from err
    if window < 1:
        raise LlmError(f"specWindow 必须 ≥ 1 ✗（收到 {value!r} ⇒ 0 草稿＝白跑一次窗口前向 ✗）")
    return window


def resolve_spec_min_match(value: Any, *, default: int = SPEC_MIN_MATCH) -> int:
    """后缀匹配最短长度归一 ✓（不写 ⇒ ``default`` ✓；``< 1`` **当场拒** ✗）。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return int(default)
    try:
        need = int(value)
    except (TypeError, ValueError) as err:
        raise LlmError(f"specMinMatch 不认 {value!r} ✗（要正整数 ✓）") from err
    if need < 1:
        raise LlmError(f"specMinMatch 必须 ≥ 1 ✗（收到 {value!r} ⇒ 单 token 匹配＝几乎必中、草稿必错 ✗）")
    return need


def plan_layer_devices(config: LlmConfig, *, gpu_bytes: int, dtype_bytes: int = 2,
                       gpu_device: str = "cuda", cpu_device: str = "cpu",
                       reserve_bytes: int = 0,
                       want_gpu_layers: int = 0) -> tuple[list[str], dict[str, Any]]:
    """按显存预算算**连续前缀**的层放置 ✓（返回 ``(devices, report)`` ✓）。

    口径（参考 Strata ``plan/plan.hpp`` 的「放置是算术结果、算不平即如实报」 ✓）：

    * 每层字节 = :func:`_block_param_count` × ``dtype_bytes`` ✓ —— **算出来** ✓ 不查配额表 ✗；
    * 卡上只放**连续前缀**（第 ``0..n-1`` 层）✓：跨界只发生在**一处边界** ✓，每步只搬一份激活
      （MB 级 ✓），而不是每层都搬权重（那是 :mod:`~.residency` 的 ``park`` 档，每步几十 GB ✗）；
    * 一层都放不上 ⇒ ``gpuLayers=0`` 且 ``why`` **写清原因** ✓（**不静默给全 CPU** ✗✗）；
    * ``want_gpu_layers`` 大于算得出来的层数 ⇒ ``shortfall`` 如实报 ✓（**不悄悄减配** ✗）。

    ⚠️ 这里只做**算术**、不碰 torch ✓ ⇒ 没有 torch 也能先算后装 ✓（回执/预检可以先算 ✓）。
    ⚠️ ``dtype_bytes`` 要按**实际装载精度**给 ✓（bf16 ⇒ 2 ✓、fp32 ⇒ 4 ✓）；给 1 会得到
      「纸面能装、真装 OOM」的假结论 ✗。
    """
    for name, value in (("gpu_bytes", gpu_bytes), ("dtype_bytes", dtype_bytes),
                        ("reserve_bytes", reserve_bytes), ("want_gpu_layers", want_gpu_layers)):
        if int(value) < 0:
            raise LlmError(f"{name} 不能为负 ✗（收到 {value!r}）")
    if int(dtype_bytes) <= 0:
        raise LlmError(f"dtype_bytes 必须 > 0 ✗（收到 {dtype_bytes!r}）")
    layer_params = _block_param_count(config)
    layer_bytes = layer_params * int(dtype_bytes)
    usable = max(0, int(gpu_bytes) - int(reserve_bytes))
    gpu_layers = min(config.depth, usable // layer_bytes) if layer_bytes > 0 else 0
    cpu_layers = config.depth - gpu_layers
    devices = [gpu_device] * gpu_layers + [cpu_device] * cpu_layers
    why = ""
    if gpu_layers == 0:
        why = (f"单层 {layer_bytes} 字节 > 可用 {usable} 字节（预算 {gpu_bytes} − 预留 "
               f"{reserve_bytes}）⇒ 一层也放不上卡，整份留 {cpu_device} ✓")
    shortfall = max(0, int(want_gpu_layers) - gpu_layers)
    report = {
        "layerParams": layer_params,
        "layerBytes": layer_bytes,
        "dtypeBytes": int(dtype_bytes),
        "gpuBytes": int(gpu_bytes),
        "reserveBytes": int(reserve_bytes),
        "usableBytes": usable,
        "gpuLayers": gpu_layers,
        "cpuLayers": cpu_layers,
        "gpuWeightBytes": gpu_layers * layer_bytes,
        "cpuWeightBytes": cpu_layers * layer_bytes,
        "allOnGpu": gpu_layers == config.depth,
        "wantGpuLayers": int(want_gpu_layers),
        "shortfall": shortfall,
        "why": why,
    }
    return devices, report


def apply_layer_plan(model: Any, layer_devices: Any) -> list[Any]:
    """把每层权重**落到各自设备** ✓，并把 ``embed``/``norm``/``lm_head`` 钉在**末层** ✓。

    ⚠️ 为什么三者跟**末层**而不是首层 ✗：``tie_embeddings`` 时 ``lm_head.weight is embed.weight``
    （**同一个 Parameter** ✓）—— 两处 ``.to()`` 到不同设备会让它**变成两份** ✗✗（白涨一份词表 ✓）。
    钉在末层还带来一个额外好处 ✓：算完 ``norm`` 直接乘 ``lm_head`` ✓ ⇒ 采样与下一轮喂入都在同一设备 ✓，
    每步只多一次「token 搬回首层」的小拷贝 ✓。

    返回**实际落地**的设备列表 ✓（``.to()`` 之后从参数读回 ✓ 不照抄请求值 ✗ —— ``"cuda"`` 与
    ``"cuda:0"`` 不是同一个 ``torch.device`` ✓，照抄会让 ``forward`` 里的比较恒真 ⇒ 每层都白搬 ✗✗）。
    """
    torch = _build()["torch"]
    blocks = list(model.blocks)
    requested = [torch.device(str(item)) for item in layer_devices]
    if len(requested) != len(blocks):
        raise LlmError(f"层放置长度 {len(requested)} ≠ 层数 {len(blocks)} ✗"
                       f"（每层都要给 ✓ 不补默认值 ✗）")
    resolved: list[Any] = []
    for block, target in zip(blocks, requested):
        block.to(device=target)
        landed = target
        for parameter in block.parameters():
            landed = parameter.device
            break
        resolved.append(landed)
    tail = resolved[-1] if resolved else torch.device(str(layer_devices[0]))
    model.embed.to(device=tail)
    model.norm.to(device=tail)
    model.lm_head.to(device=tail)
    model._layer_devices = resolved
    return resolved


#: ⚠️ ``build_llm`` 的 ``device`` **默认值哨兵** ✓✗：老语义的默认是 ``"cpu"`` ✓，但「没传」与
#: 「显式写 ``device="cpu"``」在**互斥判据**里必须分开 ✓ —— 前者允许同时给 ``layer_devices`` ✓、
#: 后者必须当场拒 ✗（不知道该听谁 ✗）。用 ``None`` 当哨兵**不行** ✗：``resolve_device(None)`` 是
#: ``auto`` ✓ ⇒ 会把老路径的默认从 cpu 悄悄改成「有卡用卡」✗✗（老测试会静默跑到卡上 ✓）。
_UNSET_DEVICE: Any = object()


def build_llm(config: LlmConfig, *, device: Any = _UNSET_DEVICE, dtype: Any = None,
              layer_devices: Any = None) -> Any:
    """构造 :class:`LlmModel` ✓（**懒导入 torch** ✓）。

    ⚠️ ``device``/``dtype`` 在**建模型时**就生效 ✓（不是建完再搬 ✗）：``set_default_dtype`` 让参数
    一开始就是目标精度 ✓ ⇒ 8B 在 bf16 下**从头到尾**只占 16 GB ✓（先建 fp32 再转 = 峰值 48 GB ✗✗）。
    ⚠️ ``device`` **不传** ⇒ ``cpu`` ✓（**不是** ``auto`` ✗）：这一层是引擎内核 ✓，**在哪算必须由调用方说清** ✓；
    「有卡用卡」这种**择优**只许发生在配置层（:class:`~.llm_backend.LlmBackend`）✓ ——
    否则老测试会静默跑到卡上 ✗、CPU 口径的失败也没人发现 ✗✗。要择优请显式写 ``device="auto"`` ✓。

    ⭐ **分层放置**（2026-11-08 补 ✓）：``layer_devices`` 给 ``config.depth`` 个设备名 ⇒ 每层落到
    各自设备 ✓（由 :func:`apply_layer_plan` 执行 ✓）。此时：

    * ``device`` **必须**保持**没传**或 ``auto``/``None`` ✓ —— 两个都点名就抛 :class:`LlmError` ✗
      （不知道该听谁 ✓，**不静默二选一** ✗）。⚠️ 判据靠 :data:`_UNSET_DEVICE` 哨兵 ✓：
      显式写 ``device="cpu"`` **也**算点名 ✗（`"cpu"` 恰好等于老默认值 ⇒ 拿默认值比对分不出来 ✗）；
    * 精度按**首层设备**解析 ✓（首层通常在卡上 ⇒ ``auto`` 得 bf16 ✓ 正是省显存的目的 ✓）；
      尾巴在 CPU 也用同一精度 ✓ —— CPU 上 bf16 慢但**省内存** ✓，要 fp32 就显式给 ``dtype`` ✓；
    * 不给 ``layer_devices`` ⇒ 与老路径**逐字节相同** ✓（``model.to(device=resolved_device)`` ✓）。
    """
    build = _build()
    if layer_devices is not None:
        if not list(layer_devices):
            raise LlmError("layer_devices 是空表 ✗（要么不给 ✓ 要么每层都给 ✓）")
        if device is not _UNSET_DEVICE and str(device).strip().lower() not in ("", "none", "auto", "default"):
            raise LlmError(f"``device``（{device!r}）与 ``layer_devices`` 同时点名 ⇒ 不知道听谁 ✗"
                           "（分层放置自带设备表 ✓；要单设备就别传 ``layer_devices`` ✓）")
        resolved_device = resolve_device(list(layer_devices)[0])
    else:
        # ⚠️ **没传** ⇒ 老语义 ``cpu`` ✓；``None``/``auto`` 仍按 `auto` 择优 ✓（与改动前逐字相同 ✓）
        resolved_device = resolve_device("cpu" if device is _UNSET_DEVICE else device)
    resolved_dtype = resolve_dtype(dtype, device=resolved_device)
    torch = build["torch"]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(resolved_dtype)      # ⚠️ 全局状态 ⇒ 必须 finally 复原 ✓
    try:
        model = build["LlmModel"](config)
    finally:
        torch.set_default_dtype(previous)
    if layer_devices is not None:
        apply_layer_plan(model, list(layer_devices))
        return model
    return model.to(device=resolved_device)


def primitives() -> dict[str, Any]:
    """公开**积木**：``torch`` / ``RMSNorm`` / ``GQAAttention`` / ``SwiGLU`` / ``TransformerBlock`` ✓。

    ⚠️ 为什么需要这个口子 ✗：别的族要的是**同一套算子**（Z-Image 的文本塔就是 Qwen3 dense ✓
    见 :mod:`.qwen3_te` ✓），但它们的参数名得与**权重键名逐字一致**（``model.layers.0.self_attn
    .q_proj.weight`` ✓ HF 命名 ✓）；而 :class:`LlmModel` 那层壳是 ``embed``/``blocks``/``lm_head``
    ✗ —— 对不上，且多一个**权重里根本没有的** ``lm_head`` ✗✗（装载时会被报成缺键 ✓）。
    ⇒ 公开积木、各层各自拼壳 ✓：算子只有一份 ✓（不复制实现 ✗）。
    """
    return dict(_build())


def _build() -> dict[str, Any]:
    """造出需要 torch 的类/函数 ✓（模块级定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。

    ⚠️ **结果要缓存** ✗✗（2026-11-08 补 ✓）：这个工厂里定义的类是**嵌套类** ✓ —— 每次调用都造一批
    新类 ⇒ ``isinstance(x, llm.Q8Kv)`` 恒假 ✓✗（模型手里那个 Q8Kv 来自另一次 ``_build()`` ✓），
    而 ``llm.Q8Kv is llm.Q8Kv`` 也恒假 ✓ ⇒ 任何「按类型判」的调用方全失效 ✓（GQA/Block 那几层也
    一样 ⇒ 外部想 ``isinstance(block, llm.TransformerBlock)`` 从来就没成功过 ✓ 只是没人写过 ✓）。
    顺带省掉每次取属性都重新执行一遍类体 ✗（``primitives()`` 在装配线上是热路径 ✓）。
    """
    global _PRIMITIVES
    if _PRIMITIVES is not None:
        return _PRIMITIVES
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    class RMSNorm(nn.Module):
        """**RMSNorm** ✓：``x / sqrt(mean(x²) + eps) * weight`` ✓（LLM 用这个，不是 LayerNorm ✗）。"""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            # ⚠️ 低精度（bf16/fp16）下**先升 fp32 再降回** ✓ —— 与 HF 的 ``LlamaRMSNorm`` 同一口径 ✓：
            # bf16 直接 ``mean`` 只有 8 位尾数 ✓，8B 深 36 层的残差会把这点误差放大 ✗✗。
            # ⚠️ fp32 输入时 ``x.float()`` / ``.to(fp32)`` **都是零拷贝** ✓ ⇒ 老路径结果一字不变 ✓。
            source = x.dtype
            wide = x.float()
            normalized = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + self.eps)
            return normalized.to(source) * self.weight

    def rope_freqs(positions: Any, head_dim: int, theta: float) -> tuple[Any, Any]:
        """预计算 RoPE 的 cos/sin ✓（``(seq, head_dim//2)`` ✓ —— **GPT-NeoX / Qwen 风格** ✓）。"""
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float()
                                    / head_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(x: Any, cos: Any, sin: Any) -> Any:
        """对最后两维做旋转 ✓（**前半/后半配对** ✓ —— GPT-NeoX / Qwen 风格 ✓）。

        ⚠️ cos/sin 一律在 **fp32** 里算出来 ✓（角度精度要紧 ✓），但**乘之前降到 x 的精度** ✓：
        否则 bf16 的 q 乘 fp32 的 cos ⇒ 广播出 fp32 张量 ✗ ⇒ 后面 ``attn @ v`` 直接撞
        ``expected scalar type BFloat16 but found Float`` ✗✗（**只在低精度下炸** ✓ CPU/GPU 都炸 ✓）。
        """
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def repeat_kv(kv: Any, n_rep: int) -> Any:
        """GQA：``(b, kv_heads, s, d)`` → ``(b, kv_heads*n_rep, s, d)`` ✓（query 头对齐 KV 头 ✓）。"""
        if n_rep == 1:
            return kv
        b, kv_h, s, d = kv.shape
        return kv[:, :, None, :, :].expand(b, kv_h, n_rep, s, d).reshape(b, kv_h * n_rep, s, d)

    # ── KV cache 量化 ✓（2026-11-08 补 ✓ 口径参考 Strata ``kv_q8.hpp`` ✓）
    def quantize_kv(x: Any, group: int = KV_GROUP) -> tuple[Any, Any]:
        """``(b, h, s, d)`` 浮点 → **(int8 值, fp32 组 scale)** ✓（**对称**量化 ✓ 零点恒 0 ✓）。

        三条不许动的口径：

        1. **分组沿最后一维（``head_dim``）** ✓ 不沿序列 ✗✗：沿序列分组＝把「这个 token 的 k」与
           「上个 token 的 k」共用一个 scale ✓（两者量级无关 ✗）；沿 ``head_dim`` 分组才是「同一个头
           的同一批通道」✓ —— 它们的量级本来就相近 ✓（这是 ``kv_q8`` 选 64 的理由 ✓）。
        2. **全零组不许让 scale 变 0** ✗✗：``0/0`` 出 NaN ⇒ softmax 吃 NaN ⇒ 整段输出 NaN ✓，
           而报错位置离这里十万八千里 ✗（症状＝「模型突然说胡话/吐 NaN」✓）。此处给 ``scale=1``
           ⇒ 反量化得 0 ✓ 精确 ✓，代价只是该组白占一份 scale ✓（全零组本来就该是 0 ✓）。
        3. ``head_dim`` 除不尽 ``group`` 时**尾部补零到整组** ✓（不 pad 就 reshape 不动 ✗，
           而 pad 出去的那几个 0 反量化后要**裁掉** ✓ —— 见 :func:`dequantize_kv` 的 ``head_dim`` ✓）。
        """
        g = int(group)
        if g <= 0:
            raise LlmError(f"quantize_kv 的 group 必须 > 0 ✗（收到 {group!r}）")
        if x.dim() != 4:
            raise LlmError(f"quantize_kv 要 (b, h, s, d) 四维 ✗（收到 {tuple(x.shape)}）")
        b, h, s, d = (int(v) for v in x.shape)
        g = kv_group_for(d, g)      # ⚠️ 组宽不许大于 head_dim ✓（否则整条头只剩一组 + 一堆补零 ✗✗）
        pad = (-d) % g
        # ⚠️ ``contiguous()`` 是**必须**的 ✓：k/v 是 ``transpose(1, 2)`` 的产物（非连续 ✓），
        # ``reshape`` 会自己复制一份**同尺寸**的 ✓ —— 显式写出来，别让读者以为零成本 ✗。
        work = (x if pad == 0 else F.pad(x, (0, pad))).contiguous().reshape(b * h * s, -1, g)
        amax = work.abs().amax(dim=-1).float()                       # (n, groups) ✓
        scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        q = (work / scale.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
        return q.view(b, h, s, -1), scale.view(b, h, s, -1)

    def dequantize_kv(q: Any, scale: Any, group: int = KV_GROUP,
                      *, dtype: Any = None, head_dim: int | None = None) -> Any:
        """``(int8, scale)`` → 浮点 ✓（``head_dim`` 给了就裁掉量化时的尾部补零 ✓）。"""
        g = int(group)
        b, h, s, groups = (int(v) for v in scale.shape)
        if q.shape[-1] != groups * g:
            raise LlmError(f"q 的最后一维（{q.shape[-1]}）≠ 组数×组宽（{groups}×{g}）✗"
                           "（这两样必须来自同一次 quantize_kv ✓）")
        out = q.to(torch.float32).view(b, h, s, groups, g) * scale.to(torch.float32).unsqueeze(-1)
        out = out.view(b, h, s, groups * g)
        if head_dim is not None and int(head_dim) != groups * g:
            out = out[..., : int(head_dim)]
        return out if dtype is None else out.to(dtype)

    class Q8Kv:
        """**一段量化后的** KV ✓（对外只暴露 ``shape``/``device``/``dequant``/``append`` ✓）。

        ⚠️ 为什么要有这层壳而不是直接把 int8 张量塞进 cache ✗✗：``forward`` 里到处要
        ``past_kv[0][0].shape[2]``（序列长度 ✓）。裸 int8 也能给出 ``shape`` ✓，但**读回时的语义**
        完全变了 —— 没有 scale 的 int8 张量是垃圾 ✓✗。壳把「值 + scale + 组宽 + 原始 head_dim」
        绑在一起 ✓ ⇒ 谁拿到它都不可能只拿走一半 ✓。
        """

        __slots__ = ("q", "scale", "group", "headDim")

        def __init__(self, q: Any, scale: Any, *, group: int = KV_GROUP,
                     head_dim: int | None = None) -> None:
            self.q = q
            self.scale = scale
            self.group = int(group)
            self.headDim = int(head_dim) if head_dim is not None else int(scale.shape[-1]) * self.group

        @classmethod
        def from_tensor(cls, x: Any, *, group: int = KV_GROUP) -> "Q8Kv":
            # ⚠️ 存进壳里的组宽必须是**夹过的那一个** ✓（存 64 而 q 只有 8 列 ⇒ 读回时
            # 「q 的宽度 ≠ 组数×组宽」当场拒 ✓ 或者更糟：不拒、静默按 64 解 ✗✗）
            g = kv_group_for(int(x.shape[-1]), group)
            q, scale = quantize_kv(x, group=g)
            return cls(q, scale, group=g, head_dim=int(x.shape[-1]))

        @property
        def shape(self) -> tuple[int, int, int, int]:
            """与**量化前**逐字相同 ✓（``head_dim`` 是原始值 ✓ 不是 pad 后的 ✓）。"""
            return (int(self.q.shape[0]), int(self.q.shape[1]), int(self.q.shape[2]), self.headDim)

        @property
        def device(self) -> Any:
            return self.q.device

        @property
        def dtype(self) -> Any:
            return self.q.dtype

        def dequant(self, dtype: Any = None) -> Any:
            return dequantize_kv(self.q, self.scale, self.group,
                                 dtype=dtype, head_dim=self.headDim)

        def append(self, other: "Q8Kv") -> "Q8Kv":
            """沿序列拼接 ✓（**旧段的 int8 原样搬走** ✓ 不重复量化 ✗✗）。

            ⚠️ 为什么不能「把整段反量化、和新段拼起来、再整段量化」✗✗：那样每一步都对同一批历史值
            再量化一次 ✓ ⇒ 误差随步数累积（32K 上下文 = 3 万次重复量化 ✓ 数值会烂掉 ✗）。这里旧段
            只做 ``cat`` ✓，一个数都不碰 ✓ —— 量化只发生一次、就在它刚算出来那一刻 ✓。
            """
            if self.group != other.group or self.headDim != other.headDim:
                raise LlmError(f"拼接两段 KV 的量化口径不同 ✗（group {self.group}/{other.group}、"
                               f"headDim {self.headDim}/{other.headDim}）")
            if int(self.q.shape[-1]) != int(other.q.shape[-1]):
                raise LlmError("拼接两段 KV 的量化宽度不同 ✗（不是同一次 quantize_kv 的产物 ✓）")
            return Q8Kv(torch.cat([self.q, other.q], dim=2), torch.cat([self.scale, other.scale], dim=2),
                        group=self.group, head_dim=self.headDim)

    def materialize_kv(x: Any, *, dtype: Any = None, device: Any = None) -> Any:
        """cache 里的一段 → **能直接做矩阵乘的浮点张量** ✓（``fp`` 段原样 ✓）。"""
        out = x.dequant(dtype) if isinstance(x, Q8Kv) else x
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
        if device is not None and out.device != device:
            out = out.to(device)
        return out

    def update_kv_cache(past: Any, fresh: Any, mode: Any = "fp", *,
                        group: int = KV_GROUP) -> Any:
        """把**这一拍新算出来**的 k/v 接进 cache ✓（返回新 cache 段 ✓ 不改入参 ✗）。

        ⚠️ ``mode="fp"`` 是老路径**逐字节相同**的写法 ✓（``torch.cat`` ✓ 老测试靠它当基准 ✓）。
        """
        if resolve_kv_mode(mode) == "fp":
            return fresh if past is None else torch.cat([past, fresh], dim=2)
        quantized = Q8Kv.from_tensor(fresh, group=group)
        return quantized if past is None else past.append(quantized)

    def _slice_kv(x: Any, length: int) -> Any:
        """沿序列维裁一段 ✓（``Q8Kv`` 要**值和 scale 一起裁** ✗ 只裁一半＝长度对不上、静默错位 ✗✗）。"""
        if isinstance(x, Q8Kv):
            return Q8Kv(x.q[:, :, :length].contiguous(), x.scale[:, :, :length].contiguous(),
                        group=x.group, head_dim=x.headDim)
        return x[:, :, :length].contiguous()

    def truncate_kv(past: Any, length: int) -> Any:
        """把整份 cache 按序列维裁到 ``length`` ✓（前缀复用「同一条提示词」时留一个 token 给前向 ✓）。

        ⚠️ 裁完要 ``contiguous()`` ✓：切片是**视图** ⇒ 底层那份长张量会一直被吊着 ✓（「裁了但没省」✗）。
        """
        if not past:
            return past
        length = max(0, int(length))
        if length >= int(past[0][0].shape[2]):
            return past
        return [(_slice_kv(key, length), _slice_kv(value, length)) for key, value in past]

    class KvPrefixCache:
        """跨调用 **KV 前缀复用** ✓（单槽 ＋ 字节预算 ＋ 命中回执 ✓）。

        口径三条（每条都是「不做就静默错」的坑 ✓）：

        1. **只认逐 token 相同的公共前缀** ✓（按 id 比 ✓ 不按文本比 ✗ —— 文本相等不保证 id 相等 ✓，
           模板空白、特殊符、解码/编码往返都会让同一句话变成另一串 id ✗）。前缀长度还必须
           ``≤ 序列长 − 1`` ✓：复用整段就没有查询可算、拿不到 logits ⇒ 采样无从谈起 ✗✗；
        2. **口径必须同一套** ✓：``fp`` 的 cache 喂给 ``q8`` 跑＝长度对、数值两套 ✗（反向也是 ✗）；
        3. **装不下就不装** ✓（:func:`kv_cache_bytes` 逐张量数 ✓，超预算直接不存并把原因写进回执 ✗
           不许「先存着试试」——那一拍正是别的件要显存的时候 ✗✗）。

        ⚠️ 为什么复用在 ``q8`` 下也**数值无损** ✓✗：量化是沿 ``head_dim`` 分组的 ✓ ⇒ **每个 token 的
        scale 只由它自己那 64 个元素决定** ✓，与前后 token 无关 ✓ ⇒ 「先算好的段」与「重算一遍」
        逐位相同 ✓（这就是当初把分组放在 ``head_dim`` 而不是序列维的第二个理由 ✓）。
        """

        __slots__ = ("_budget", "_mode", "_group", "_ids", "_past", "_bytes",
                     "_hits", "_misses", "_skips", "_reused", "_stored", "_note")

        def __init__(self, budget_bytes: Any = KV_CACHE_DEFAULT_BYTES, *,
                     mode: Any = "fp", group: int = KV_GROUP) -> None:
            self._budget = max(0, int(budget_bytes))
            self._mode = resolve_kv_mode(mode)
            self._group = int(group)
            self._ids: list[int] = []
            self._past: Any = None
            self._bytes = 0
            self._hits = self._misses = self._skips = self._reused = self._stored = 0
            self._note = "" if self._budget else "预算 0 ⇒ 前缀复用关闭 ✓"

        @property
        def enabled(self) -> bool:
            return self._budget > 0

        @property
        def mode(self) -> str:
            return self._mode

        def clear(self, note: str = "") -> None:
            """丢掉常驻 cache ✓（换权重/换口径/按需腾显存时调 ✓）。"""
            self._ids = []
            self._past = None
            self._bytes = 0
            if note:
                self._note = note

        def lookup(self, ids: Any) -> tuple[Any, int]:
            """问「这串 id 能复用多少」 ✓ ⇒ ``(cache 段 ／ 可复用 token 数)``（0 = 没命中 ✓）。"""
            if not self.enabled or self._past is None:
                self._misses += 1
                return None, 0
            wanted = [int(x) for x in ids]
            matched = 0
            for old, new in zip(self._ids, wanted):
                if old != new:
                    break
                matched += 1
            if matched == 0:
                # ⚠️ 一点共同前缀都没有 ⇒ 这份 cache 对**当前**这串永远没用 ✓ ⇒ 立刻放掉腾显存 ✗
                # 不许「留着万一」✓✗（卡满会报在别的件上 ✓ 见模块抬头）。
                self.clear("无共同前缀 ⇒ 释放常驻 cache ✓")
                self._misses += 1
                return None, 0
            if matched >= len(wanted):
                matched = len(wanted) - 1     # 至少留一个 token 给前向 ✓（见 class docstring ①）
            if matched < 1:
                self._misses += 1
                return None, 0
            self._hits += 1
            self._reused += matched
            return self._past, matched

        def store(self, ids: Any, past: Any) -> int:
            """存下「这串 id ⇒ 这份 cache」 ✓ ⇒ 存下的字节数（0 = 没存 ✓，原因进 ``note`` ✓）。"""
            if not self.enabled:
                self._skips += 1
                return 0
            if not past:
                self._note = "没有 cache 可存 ✓"
                self._skips += 1
                return 0
            tokens = int(past[0][0].shape[2])
            size = kv_cache_bytes(past)
            if size > self._budget:
                self.clear(f"{tokens} token 的 cache 要 {size} B > 预算 {self._budget} B ⇒ 不常驻 ✓")
                self._skips += 1
                return 0
            self._ids = [int(x) for x in ids]
            self._past = past
            self._bytes = size
            self._stored += 1
            self._note = ""
            return size

        def describe(self) -> dict[str, Any]:
            return {
                "enabled": self.enabled,
                "mode": self._mode,
                "group": self._group if self._mode == "q8" else 0,
                "budgetBytes": self._budget,
                "storedTokens": len(self._ids),
                "bytes": self._bytes,
                "hits": self._hits,
                "misses": self._misses,
                "skipped": self._skips,
                "reusedTokens": self._reused,
                "stored": self._stored,
                "note": self._note,
            }

    class GQAAttention(nn.Module):
        """分组查询注意力 ✓（GQA ✓ + RoPE ✓ + **因果掩码** ✓ + KV cache ✓）。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.heads = config.heads
            self.kv_heads = config.kv_heads
            self.head_dim = config.head_dim
            self.n_rep = config.heads // config.kv_heads
            self.qk_norm = config.qk_norm
            #: ⚠️ 只有 q/k/v 吃 ``qkv_bias`` ✓ —— ``o_proj`` 在任何一族都是无偏置 ✗（Qwen2.5 亦然 ✓）
            bias = bool(config.qkv_bias)
            self.q_proj = nn.Linear(config.hidden, config.heads * config.head_dim, bias=bias)
            self.k_proj = nn.Linear(config.hidden, config.kv_heads * config.head_dim, bias=bias)
            self.v_proj = nn.Linear(config.hidden, config.kv_heads * config.head_dim, bias=bias)
            self.o_proj = nn.Linear(config.heads * config.head_dim, config.hidden, bias=False)
            # QK-norm（Qwen3 ✓）：**per-head** 归一化，weight 形状 ``(head_dim,)`` ✓
            # ⇒ 广播到最后一维即可 ✓（不加 ModuleList ✗，瓶颈在 per-head 的 RMS ✓）。
            if config.qk_norm:
                self.q_norm = RMSNorm(config.head_dim, config.rms_eps)
                self.k_norm = RMSNorm(config.head_dim, config.rms_eps)

        def forward(self, x: Any, cos: Any, sin: Any,
                    past_kv: tuple[Any, Any] | None = None,
                    kv_mode: Any = "fp") -> tuple[Any, tuple[Any, Any]]:
            b, s, _ = x.shape
            q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            # ⚠️ 顺序是**硬事实**（llama.cpp 的 ``build_qwen3`` ✓）：proj → reshape → **QK-norm** → RoPE ✗✗。
            # 先 RoPE 后 norm 也对得上 shape ✓ ⇒ 错了**不会报错**、只会让效果变差 ✗（所以按参考顺序写死 ✓）。
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            mode = resolve_kv_mode(kv_mode)
            past_k = past_v = None
            past_len = 0
            if past_kv is not None:
                past_k, past_v = past_kv
                past_len = int(past_k.shape[2])
            if mode == "q8":
                # ⚠️ 注意力**必须读量化后的值** ✓✗：写入时用的是原 k ✓、读回时是 dequant(k) ✓ ——
                # 两者不同就会「同一个 token 在 prefill 与 decode 看到两套历史」✗（数值上表现为
                # 首 token 正常、越往后越飘 ✓）。所以这里先量化、再**反量化回来**算注意力 ✓。
                new_kv = (update_kv_cache(past_k, k, "q8"),
                          update_kv_cache(past_v, v, "q8"))
                k = materialize_kv(new_kv[0], dtype=q.dtype, device=q.device)
                v = materialize_kv(new_kv[1], dtype=q.dtype, device=q.device)
            else:
                k = update_kv_cache(past_k, k, "fp")
                v = update_kv_cache(past_v, v, "fp")
                new_kv = (k, v)
            k_rep = repeat_kv(k, self.n_rep)
            v_rep = repeat_kv(v, self.n_rep)
            scores = q @ k_rep.transpose(-2, -1) / math.sqrt(self.head_dim)
            # ⭐ 因果掩码 ✓：当前位置（past_len+i）只许看 ≤ 它自己的位置 ✓✗
            # ⚠️ mask 的 dtype 必须跟 scores 一致 ✗：fp32 的 mask 加到 bf16 的 scores 上会**广播成 fp32**
            # ⇒ 后面 ``attn @ v`` 撞 `expected scalar type BFloat16 but found Float` ✗✗（低精度专炸 ✓）。
            # ⚠️ 单 token（解码步 ``s == 1``）**不造 mask** ✓：唯一的查询就是最后一位 ⇒ 它能看见
            # ``past_len + 1`` 个位置里的全部 ✓，掩码全零 ⇒ 造出来只是白占一份 ``(1, past+1)`` 的显存 ✗
            # （每生成一个 token 白造一次、每层一次 ✓✗）。数值上与原路径**逐位相同** ✓（加零 ≠ 变数 ✓）。
            if s == 1:
                masked = scores
            else:
                mask = torch.triu(torch.full((s, past_len + s), float("-inf"),
                                             device=x.device, dtype=scores.dtype),
                                  diagonal=past_len + 1)
                masked = scores + mask
            # fp16 的 softmax 升 fp32 再降回 ✓（溢出 + 精度两个理由 ✓）；fp32/bf16 原样 ✓。
            if masked.dtype == torch.float16:
                attn = torch.softmax(masked.float(), dim=-1).to(masked.dtype)
            else:
                attn = torch.softmax(masked, dim=-1)
            out = attn @ v_rep
            out = out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
            return self.o_proj(out), new_kv

    class SwiGLU(nn.Module):
        """**SwiGLU** MLP ✓：``gate`` 与 ``up`` 两条 ✓ ``silu(gate) * up`` ✓（无 bias ✓）。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.gate = nn.Linear(config.hidden, config.ffn, bias=False)
            self.up = nn.Linear(config.hidden, config.ffn, bias=False)
            self.down = nn.Linear(config.ffn, config.hidden, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class TransformerBlock(nn.Module):
        """pre-norm 一层 ✓：RMSNorm → 注意力 → 残差 → RMSNorm → SwiGLU → 残差 ✓。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.norm1 = RMSNorm(config.hidden, config.rms_eps)
            self.attn = GQAAttention(config)
            self.norm2 = RMSNorm(config.hidden, config.rms_eps)
            self.ffn = SwiGLU(config)

        def forward(self, x: Any, cos: Any, sin: Any,
                    past_kv: tuple[Any, Any] | None = None,
                    kv_mode: Any = "fp") -> tuple[Any, tuple[Any, Any]]:
            attended, new_kv = self.attn(self.norm1(x), cos, sin, past_kv, kv_mode)
            x = x + attended
            x = x + self.ffn(self.norm2(x))
            return x, new_kv

    class LlmModel(nn.Module):
        """token id → logits ✓（``(b, s)`` → ``(b, s, vocab)`` ✓ + KV cache ✓）。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.config = config
            self.head_dim = config.head_dim
            self.rope_theta = config.rope_theta
            self.embed = nn.Embedding(config.vocab_size, config.hidden, padding_idx=config.pad_id)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.depth)])
            self.norm = RMSNorm(config.hidden, config.rms_eps)
            self.lm_head = nn.Linear(config.hidden, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.embed.weight
            #: 分层放置的**实际**设备表 ✓（``None`` ⇒ 整模型同设备、即老路径 ✓ 由
            #: :func:`apply_layer_plan` 写入 ✓ 不在这里猜 ✗）
            self._layer_devices: list[Any] | None = None
            #: KV 存储口径 ✓（``fp`` 老路径 ✓；``q8`` 分组量化 ✓）—— 可由
            #: :func:`generate_llm` 的 ``kv_mode`` 逐步覆盖 ✓，也可在外部按住不动 ✓。
            self.kv_mode: str = "fp"

        def forward(self, ids: Any, past_kv: Any = None,
                    kv_mode: Any = None) -> tuple[Any, Any]:
            batch, seq = ids.shape
            mode = resolve_kv_mode(self.kv_mode if kv_mode is None else kv_mode)
            past_len = 0 if past_kv is None else int(past_kv[0][0].shape[2])
            plan = getattr(self, "_layer_devices", None)
            embed_device = self.embed.weight.device
            if plan is not None and ids.device != embed_device:
                ids = ids.to(embed_device)          # 采样在末层设备 ⇒ 每步把 token 送回 embed ✓
            positions = torch.arange(past_len, past_len + seq, device=embed_device)
            cos, sin = rope_freqs(positions, self.head_dim, self.rope_theta)
            x = self.embed(ids)
            rope_by_device: dict[Any, tuple[Any, Any]] = {}
            new_kv: list[tuple[Any, Any]] = []
            for index, block in enumerate(self.blocks):
                layer_kv = None if past_kv is None else past_kv[index]
                if plan is None:
                    layer_cos, layer_sin = cos, sin
                else:
                    target = plan[index]
                    if x.device != target:
                        x = x.to(target)            # 跨界只发生在**边界** ✓ 每步一份激活（MB 级）✓
                    rope = rope_by_device.get(target)
                    if rope is None:
                        rope = (cos.to(target), sin.to(target))   # 每个设备只搬一次 ✓ 不是每层 ✓
                        rope_by_device[target] = rope
                    layer_cos, layer_sin = rope
                x, kv = block(x, layer_cos, layer_sin, layer_kv, mode)
                new_kv.append(kv)
            x = self.norm(x)
            return self.lm_head(x), new_kv

    class LookupDrafter:
        """三词元后缀 → 历史里出现过的后继 ✓（**零权重查表** ✓ 与 ``strata::spec::SuffixDrafter`` 同构）。

        口径四条（每一条决定的都是「接受了几个」✓ 而不是「会不会崩」✗ —— 所以抄错了不报错 ✗✗）：

        1. **三词元后缀**作键 ✓（``(hist[n-3], hist[n-2], hist[n-1])``）：后缀越长命中越少、草稿越准 ✓，
           参考实现选的 3 是「中文短语／代码行」的经验值 ✓；
        2. 每个键存 ``ways`` 个**最近**位置 ✓（新位置插头部 ✓）⇒ :meth:`propose` 里「严格大于才换」
           的实际语义就是**等长匹配保留更近的** ✓（抄成 ``>=`` 会变成保留更早的 ✓ 结果不同但不报错 ✗）；
        3. **允许草稿与当前后缀重叠** ✓✗ —— ``suffix_drafter.cpp:83`` 原文写着「continuation may run
           into the current suffix (periodic text); reading history up to ``cur`` is valid」✓。
           ⚠️ 我曾把这条当成 bug 删掉（判「草稿落在后缀里就没用」✓）**方向是错的** ✗；
        4. 取草稿从 ``best_end + 1`` 起、**不超过当前末尾 ``cur``** ✓（历史里没有的东西猜不出来 ✓）。

        ⚠️ 第 4 条是这套机制的结构性盲区 ✓（见 ``docs/reference-audit-llm.md`` §7.4 的实测）：
        逐字复述时最长匹配会一路扩到当前末尾 ⇒ ``best_end`` 贴着 ``cur`` ⇒ 草稿只剩与后缀重叠的那几个
        ⇒ **匹配越准越不接受** ✓。这里**照参考实现保留**这行为 ✗ 不擅自加「错位修正」✗ ——
        那已经是另一种策略（要改成偏好 ``best_end + 1 + k <= cur`` 的匹配或一次投多个候选 ✓），
        不在「照搬」范围内 ✓。
        """

        __slots__ = ("hist", "table", "ways", "min_match", "max_match")

        def __init__(self, *, ways: int = 3, min_match: int = 3,
                     max_match: int = 64) -> None:
            self.hist: list[int] = []
            self.table: dict[tuple[int, int, int], list[int]] = {}
            self.ways = max(1, int(ways))
            self.min_match = max(1, int(min_match))
            self.max_match = max(1, int(max_match))

        def observe(self, token: Any) -> None:
            """吃进一个 token ✓（凑够 3 个才建索引 ✓ —— 键是三词元 ✓）。"""
            tid = int(token)
            n = len(self.hist)
            if n >= 2:
                key = (self.hist[n - 2], self.hist[n - 1], tid)
                slot = self.table.get(key)
                if slot is None:
                    slot = self.table[key] = []
                slot.insert(0, n)              # 最近优先 ✓（位置 n 就是 tid 自己 ✓）
                del slot[self.ways:]           # 只留最近 ways 个 ✓
            self.hist.append(tid)

        def append(self, tokens: Any) -> None:
            """整段吃进 ✓（提示词也要吃 ✓ —— 后缀表不完整时草稿质量直接塌 ✗）。"""
            for token in tokens:
                self.observe(token)

        def propose(self, max_k: int) -> list[int]:
            """取最多 ``max_k`` 个草稿 ✓（返回**空列表** = 没匹配上 ✓ 不是错 ✓）。"""
            if max_k <= 0:
                return []
            n = len(self.hist)
            if n < 3:
                return []
            slot = self.table.get((self.hist[n - 3], self.hist[n - 2], self.hist[n - 1]))
            if not slot:
                return []
            cur = n - 1
            best_end, best_len = -1, 0
            for pos in slot:
                if pos >= cur:                 # 当前后缀自己 ✓（它「之后」没有内容可抄 ✗）
                    continue
                ln = 0
                while (ln < self.max_match and ln <= pos
                       and self.hist[pos - ln] == self.hist[cur - ln]):
                    ln += 1
                if ln > best_len:              # 严格大于 ⇒ 等长保留更近的（槽内已最近优先 ✓）
                    best_len, best_end = ln, pos
            if best_end < 0 or best_len < self.min_match:
                return []
            out: list[int] = []
            q = best_end + 1
            while q <= cur and len(out) < max_k:   # 允许与当前后缀重叠 ✓（见口径 3 ✓）
                out.append(self.hist[q])
                q += 1
            return out

    def generate_llm(model: Any, ids: Any, *, max_new_tokens: int = 64,
                     temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
                     eos_id: int | None = None, seed: int | None = None,
                     stop_ids: Any = None, on_token: Any = None,
                     prefill_chunk: int = 1024, kv_mode: Any = "fp",
                     kv_cache: Any = None, spec_lookup: bool = False,
                     spec_window: int = 8, spec_min_match: int = 3,
                     spec_report: Any = None) -> Any:
        """逐 token 生成 ✓（greedy / temperature / top-p / top-k ✓ + 早停 ✓）。

        ⚠️ ``temperature=0`` ⇒ greedy ✓（直接 argmax ✓ —— 保证可复现 ✓，也是**跨设备一致**的判据 ✓）。
        ⚠️ ``seed`` 只对**随机采样**有用 ✓：它种的是 torch **本进程全部设备**的 RNG ✓，所以同一设备、
        同一份权重/输入 ⇒ 逐 token 复现 ✓；⚠️ **跨设备不保证逐 token 相同** ✗ —— 两端 logits 本身就差
        1e-5 量级 ✓，再从概率分布里抽样 ⇒ 尾部 token 会分叉 ✓（这是数学 ✓ 不是 bug ✓）。
        要跨设备逐字相同 ⇒ 用 ``temperature=0`` ✓（:class:`~.llm_backend.LlmBackend` 的一致性测试就这么判 ✓）。

        流式两的口子（2026-11-01 补 ✓，**默认关闭 ⇒ 老调用方字节不变** ✓）：

        * ``on_token(tid: int)``：**每步一调** ✓，把刚采到的 token **id** 递出去（**不解码、不判 stop** ✗ ——
          解码与停用词判定属于 :meth:`~.llm_backend.LlmBackend.generate_stream` ✓，本层只管"步" ✓）。
          回调**抛异常即中止生成** ✓（异常原样上抛、不吞 ✗）——这正是「文字级停用词命中」与
          「客户端断开」两种中止的实现方式 ✓；已生成的张量**不返回**（抛出去了 ✓）。
        * ``stop_ids``：命中的 token **计入输出后**收尾 ✓（字符级停用词由上层按解码文本裁 ✓，
          本层只处理"整 token 级"的哨兵 ✓）。

        ⚠️⚠️ **整段跑在 ``torch.inference_mode()`` 里**（2026-11-07 修 ✓✗ —— 这条是**真 bug** ✗✗）：
        本函数原来没有关梯度 ✓，而参数默认 ``requires_grad=True`` ⇒ 每一步前向都在建 **autograd 图** ✗，
        36 层的所有中间量（注意力分数 ``(heads, L, L)`` 最大 ✓）被 ``SoftmaxBackward``/``MulBackward``
        当 saved tensor **吊住不放** ✓✗。代价是 L² 乘层数：实测 4B（heads=32/depth=36）上
        **3205 词元**的提示词 ⇒ PyTorch 已分配 **94.62 GiB**（卡只有 22.49 GiB ✗ —— Windows 的 WDDM
        把它甩进共享内存硬撑了 60 秒才 OOM ✓）。关梯度后同一跑峰值 ≈ 权重 8.1 + 三张分数 1.8 GiB ✓。
        **看着能跑、只是慢/吃显存** ⇒ 没有任何报错指向这里 ✗✗（``/tune`` 真跑才露头 ✓）。

        长提示词还要**分块预填充** ✗：``prefill_chunk`` 把提示词按词元数切成若干次前向 ✓（带 KV cache ✓
        数学上等价 ✓ —— 每块只对**自己那批查询**算注意力 ✓），峰值从 ``O(L²)`` 降到
        ``O(prefill_chunk × L)`` ✓。默认 1024 词元 ✓（3205 词元的 extractor 提示词 ⇒ 4 块 ✓）；
        写成 ``>= ids.shape[1]`` 即退回「一次吃完」的老路径 ✓（数值上逐位相同 ✓）。

        ``kv_mode``（2026-11-08 补 ✓）：``fp``（默认 ✓ 老路径逐位不变 ✓）／``q8``（KV cache 存成
        INT8 分组量化 ＋ fp32 组 scale ✓，见 :func:`quantize_kv`）。q8 是**有损**的 ✓ —— 输出不再
        与 fp 逐字相同（长上下文里可能分叉 ✓），但每 token 的 KV 占用降到约一半 ✓
        （4B 档实测见 ``docs/reference-audit-llm.md`` §7.2 的账 ✓）。

        ``kv_cache``（2026-11-08 补 ✓）：传 :class:`KvPrefixCache` ⇒ **跨调用复用公共前缀的 KV** ✓
        （agent 工具循环每轮都带同一段 system/工具声明 ✓、``/tune`` 拿同一提示词调参数 ✓
        —— 这些场景每轮都在重算**一字不差**的前缀 ✓✗）。**默认 ``None`` ⇒ 老路径逐位不变** ✓；
        命中时只把尾部 ``len(ids) − 复用数`` 个词元喂进 prefill ✓（位置由 ``past_len`` 自然接上 ✓）。

        ⚠️ 「无损」只到**哪一位**要说清 ✗✗：复用的 **KV 段本身与重算逐位相同** ✓（因果注意力下每个
        词元的 KV 只由它自己及之前的词元决定 ✓，q8 的分组又沿 ``head_dim`` ✓），但**末位那一次前向的
        查询块形状变了** ✓（``1×N`` 还是 ``M×N`` 的矩阵乘 ✓）⇒ 末位 logits 有 **bf16 累加顺序**级别的
        差 ✓（4B 真权重实测最大 0.145，相对 0.56% ✓ 见 §7.3.1）⇒ 贪心在**近似并列**处可能翻成另一个
        token（实测：多/少一个空格 ✓）。要**逐字复现**（判据基准、缓存命中比对）就别开复用 ✓。

        ``spec_lookup``（2026-11-08 补 ✓ **默认 False ⇒ 老路径逐位不变** ✓）：**投机解码**，草稿源用
        「后缀查表」（:class:`LookupDrafter` ✓ **零权重** ✓），一次前向验 ``1 + spec_window`` 行、
        接受其中一段前缀 ✓（机制照搬 ``strata::spec::SuffixDrafter`` ＋ ``Verifier::commit`` ✓
        见 ``docs/reference-audit-llm.md`` §7.4 ✓）。

        三条口径（每条都是「不写清就会静默错」的 ✓✗）：

        1. **只在 ``temperature == 0`` 下启用** ✓：贪心是唯一有**逐 token 等价判据**的场景 ✓ ——
           开投机与不开必须**逐 token 相同** ✓（``engine_llm_test`` 就拿这条判 ✓）。
           随机采样要概率接受（``spec_prob.hpp`` 那套点质量规则 ✓），且**必须**做 Philox 计数空间隔离
           才能与不开时落在同一条随机序列上 ✗ 本仓没有这套 ⇒ ``temperature > 0`` 时**自动退回老路径** ✓
           并写在回执里 ✓（不静默降级成「看起来开了」✗✗）；
        2. **窗口第 0 行 = target 自产、已产出但尚未喂入的那个 token** ✓（``generate.cpp:20700`` 的
           ``window[0]`` 口径 ✓）—— 草稿从第 1 行起 ✓。窗口 = ``[t, d1..dk]``、前向 ``T = k+1`` 行，
           ``outv[i] == window[i+1]`` 即接受 ✓，接受数 ``a`` ⇒ 一轮产出 ``a+1`` 个 token（含 bonus）✓；
        3. **回滚只有一件事：把 KV 裁到"真正接受的长度"** ✓（``truncate_kv`` 已在 ③ 为前缀复用写好 ✓）。
           本仓 KV 是**真实张量** ⇒ 必须裁 ✗；参考实现是**预分配定长缓冲** ⇒「被拒的行从未写入」✓
           （两种实现等价、写法不同 ✓）。⚠️ 裁的**时机**与**目标长度**写错**不报错** ✗✗，
           只让后续位置整体错位 ✓✗。

        ⚠️ 收益要如实说 ✓：真实产物形状上 A/B 两场景实测**每轮平均接受 0.01~0.02 个词元** ⇒
        同尺度加速 **1.02~1.03×**（§7.4 的账 ✓）。这套框架的价值在**「接草稿头的前置设施」** ✓，
        不在 lookup 本身 ✓。
        """
        if max_new_tokens < 1:
            raise LlmError(f"max_new_tokens 必须 ≥1（收到 {max_new_tokens} ✗）")
        if temperature < 0:
            raise LlmError(f"temperature 不能为负（收到 {temperature} ✗）")
        if prefill_chunk < 1:
            raise LlmError(f"prefill_chunk 必须 ≥1（收到 {prefill_chunk} ✗）")
        if spec_window < 1:
            raise LlmError(f"spec_window 必须 ≥1（收到 {spec_window} ✗）")
        if spec_min_match < 1:
            raise LlmError(f"spec_min_match 必须 ≥1（收到 {spec_min_match} ✗）")
        if seed is not None:
            torch.manual_seed(int(seed))       # ⚠️ 种全部设备 ✓（cuda 的 generator 也在内 ✓）
        mode = resolve_kv_mode(kv_mode)
        ids = ids.to(model_device(model))      # ⚠️ 拼 id 的调用方常忘了搬 ✓（HF 同口径 ✓ 移动而不报错 ✓）
        total = int(ids.shape[1])
        if total < 1:
            raise LlmError("ids 不能是空序列 ✗（至少要有一个词元 ✓ 空提示词没有可续写的上下文 ✗）")
        generated: list[Any] = []

        def _sample_token(next_logits: Any) -> Any:
            """从**一行** logits 采一个 token ✓（形状 ``(1, 1)`` ✓ —— 与老路径逐位同 ✓）。"""
            if temperature == 0:
                return torch.argmax(next_logits, dim=-1, keepdim=True)
            # ⚠️ 采样一律在 **fp32** 里做 ✓：top-p 的阈值比较与多项式抽样都对**概率精度**敏感 ✓，
            # bf16 只有 8 位尾数 ⇒ 尾部概率会被压平/抹掉 ✓（argmax 不受影响 ✓ —— 它只比大小 ✓）。
            # 副作用是好的 ✓：概率在两端算得更接近 ⇒ 随机采样也不至于一上来就分叉 ✓。
            probs = torch.softmax(next_logits.float() / temperature, dim=-1)
            if top_k > 0:
                topk = torch.topk(probs, min(top_k, probs.shape[-1]))
                cut = torch.full_like(probs, float("-inf"))
                cut.scatter_(1, topk.indices, topk.values)
                probs = torch.softmax(cut, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_probs[remove] = 0.0
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            return torch.multinomial(probs, 1)

        def _emit(token: Any) -> bool:
            """产出一个 token ✓ 返回「该停了吗」✓（**先回调再判停** ✓ 与老路径同序 ✓）。"""
            # ⚠️ 形状统一成 ``(1, 1)`` ✓：老路径从 ``_sample_token`` 拿到的是 ``(1, 1)`` ✓，
            # 投机路径送进来的是草稿 id 的 ``(1,)`` ⇒ 不归一，收尾 ``torch.cat(dim=1)`` 会当场炸 ✗✗。
            generated.append(token.reshape(1, 1))
            token_id = int(token.reshape(-1)[0].item())
            if on_token is not None:
                # ⚠️ 顺序固定：**先回调再判停** ✓ —— 让流式端也看见收尾那个 token（不看见就得靠上层猜 ✗）；
                # 回调抛异常=主动中止（文字级停用词/客户端断开 ✓），原样上抛 ✓。
                on_token(token_id)
            if eos_id is not None and bool((token == eos_id).all()):
                return True
            if stop_ids is not None and token_id in stop_ids:
                return True
            return False

        with torch.inference_mode():           # ⚠️ 见 docstring：不关梯度 = 每层中间量全部被吊住 ✗✗
            past_kv = None
            logits = None
            offset = 0
            if kv_cache is not None and getattr(kv_cache, "mode", mode) == mode:
                cached, reuse = kv_cache.lookup(ids[0].tolist())
                if reuse:
                    # ⚠️ 复用段必须**先裁到 reuse** 再喂前向 ✗（多留一个 token 会让位置与 cache 错位 ✗✗：
                    # 位置按 ``past_len`` 接 ✓，cache 长一格而位置不动 ⇒ 注意力看见一个"幽灵位置" ✓✗）。
                    past_kv = truncate_kv(cached, reuse)
                    offset = int(reuse)
            while offset < total:
                end = min(offset + prefill_chunk, total)
                logits, past_kv = model(ids[:, offset:end], past_kv, kv_mode=mode)
                offset = end
            if logits is None:                     # 全段命中时 lookup 应保 ``reuse ≤ len-1`` ✓
                raise LlmError("KV 复用命中了整段 ids ⇒ 没有可采样的 logits ✗"
                               "（lookup 应保证 reuse ≤ len(ids)-1 ✓，这里说明那条保护失效了 ✗✗）")
            drafter = None
            if spec_lookup:
                if temperature == 0:
                    drafter = LookupDrafter(min_match=spec_min_match)
                    # ⚠️ 后缀表要吃**全部**历史 ✓（提示词也算 ✓）：只吃生成段 ⇒ 前几句的格式壳
                    # （「角色名＋冒号」那种）完全进不了表 ⇒ 草稿退化成空 ✗（不报错 ✗✗）。
                    drafter.append(ids[0].tolist())
                elif spec_report is not None:
                    # ⚠️ 如实退回 ✓ 不静默：随机采样要概率接受 ＋ Philox 计数空间隔离才能落在同一条
                    # 随机序列上 ✗ 本仓没有那套 ⇒ 不开（不假装开了 ✓）。
                    spec_report.update(enabled=False, reason="temperature>0", window=0,
                                       rounds=0, drafted=0, accepted=0)
            if drafter is None:
                for _ in range(max_new_tokens):
                    token = _sample_token(logits[:, -1, :])
                    if _emit(token):
                        break
                    logits, past_kv = model(token, past_kv, kv_mode=mode)
            else:
                # —— 投机主循环 ✓（``T == 1`` 时窗口就是 ``[t]`` ⇒ 与上面逐 token 版**逐位等价** ✓）——
                past_len = int(past_kv[0][0].shape[2])
                rounds = drafted = accepted = 0
                token = _sample_token(logits[:, -1, :])
                while len(generated) < max_new_tokens:
                    if _emit(token):
                        break
                    tid = int(token.reshape(-1)[0].item())
                    drafter.observe(tid)
                    remain = max_new_tokens - len(generated)
                    if remain <= 0:
                        break                  # 产满了 ⇒ 不再白送一次前向 ✓（老路径会空跑一拍 ✗）
                    draft = drafter.propose(spec_window)[:remain]
                    window = [tid] + draft
                    win = torch.tensor([window], device=ids.device, dtype=ids.dtype)
                    win_logits, win_kv = model(win, past_kv, kv_mode=mode)
                    rounds += 1
                    drafted += len(draft)
                    # 接受判据：第 i 行的 argmax 就是「窗口第 i 行**之后**那个 token」✓ ⇒ 与 window[i+1] 比 ✓
                    a = 0
                    while a < len(draft):
                        if int(torch.argmax(win_logits[:, a, :], dim=-1)) != draft[a]:
                            break
                        a += 1
                    accepted += a
                    stopped = False
                    for i in range(a):
                        if _emit(torch.tensor([draft[i]], device=ids.device, dtype=ids.dtype)):
                            stopped = True
                            break
                        drafter.observe(draft[i])
                    # 回滚：窗口算了 ``1 + len(draft)`` 行 ✓ 只有 ``1 + a`` 行是真接受的 ✓。
                    # ⚠️ 上限压到「已产出总数」✓：stop 打断时 cache 不许比 ids 长 ✗（否则 store 存错配 ✗✗）。
                    keep = min(past_len + 1 + a, total + len(generated))
                    past_kv = truncate_kv(win_kv, keep)
                    past_len = keep
                    if stopped:
                        break
                    # bonus / 被拒处的修正 ✓：第 a 行 logits 就是下一个 token 的依据 ✓（a == len(draft) 即 bonus ✓）
                    token = _sample_token(win_logits[:, a, :])
                if spec_report is not None:
                    spec_report.update(enabled=True, reason="greedy", window=spec_window,
                                       rounds=rounds, drafted=drafted, accepted=accepted)
            out = torch.cat([ids] + generated, dim=1)
            if kv_cache is not None and getattr(kv_cache, "mode", mode) == mode:
                # 存**整段**（提示词＋续写 ✓）而不是只存提示词：下一轮 agent 循环带回来的历史里
                # 就包含这段续写 ✓ ⇒ 连上一轮的输出也不必重算 ✓（装不下由 store 自己拒 ✓）。
                kv_cache.store(out[0].tolist(), past_kv)
            return out

    _PRIMITIVES = {
        "torch": torch, "nn": nn, "F": F, "RMSNorm": RMSNorm, "GQAAttention": GQAAttention,
        "SwiGLU": SwiGLU, "TransformerBlock": TransformerBlock, "LlmModel": LlmModel,
        "rope_freqs": rope_freqs, "apply_rope": apply_rope, "repeat_kv": repeat_kv,
        "generate_llm": generate_llm,
        # KV cache 量化（2026-11-08 ✓）：值 + scale 绑在 :class:`Q8Kv` 里 ✓ 不许只拿走一半 ✗
        "Q8Kv": Q8Kv, "quantize_kv": quantize_kv, "dequantize_kv": dequantize_kv,
        "materialize_kv": materialize_kv, "update_kv_cache": update_kv_cache,
        # 跨调用前缀复用（2026-11-08 ✓）：裁段与常驻 cache 是**一对** ✓（只给 cache 不给 truncate ⇒
        # 「多留一个 token」的错位会静默发生 ✗✗）
        "truncate_kv": truncate_kv, "KvPrefixCache": KvPrefixCache,
        # 投机解码草稿源（2026-11-08 ✓）：零权重后缀查表 ✓ 与 ``truncate_kv`` 成对用 ✓
        # （草稿器单独拿出来没意义 ✗ —— 窗口前向 ＋ 接受前缀 ＋ 回滚三件缺一不可 ✓）
        "LookupDrafter": LookupDrafter,
        # 旋钮归一（2026-11-08 ✓）：三个口子（普通／流式／agent）必须共用 ✗ 各抄一遍必漂移 ✗
        "resolve_spec_lookup": resolve_spec_lookup,
        "resolve_spec_window": resolve_spec_window,
        "resolve_spec_min_match": resolve_spec_min_match,
        "SPEC_WINDOW": SPEC_WINDOW, "SPEC_MIN_MATCH": SPEC_MIN_MATCH,
    }
    return _PRIMITIVES


def __getattr__(name: str) -> Any:
    """惰性导出 ✓（**先查 ``__all__``** ✓ 再建积木 ✓ —— 与 h3_form 同一纪律 ✓）。"""
    if name.startswith("_") or name not in __all__:
        raise AttributeError(f"llm 没有 {name} ✓")
    return _build()[name]
