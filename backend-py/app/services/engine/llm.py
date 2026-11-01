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
    "LlmConfig",
    "apply_rope",
    "available_devices",
    "build_llm",
    "device_facts",
    "generate_llm",
    "model_device",
    "primitives",
    "repeat_kv",
    "resolve_device",
    "resolve_dtype",
    "rope_freqs",
]


class LlmError(ValueError):
    """LLM 配置/输入不合法 ✓（**明确报错** ✗ 不静默猜 ✓）。"""


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


def build_llm(config: LlmConfig, *, device: Any = "cpu", dtype: Any = None) -> Any:
    """构造 :class:`LlmModel` ✓（**懒导入 torch** ✓）。

    ⚠️ ``device``/``dtype`` 在**建模型时**就生效 ✓（不是建完再搬 ✗）：``set_default_dtype`` 让参数
    一开始就是目标精度 ✓ ⇒ 8B 在 bf16 下**从头到尾**只占 16 GB ✓（先建 fp32 再转 = 峰值 48 GB ✗✗）。
    ⚠️ 默认 ``device="cpu"`` ✓ **不是** ``auto`` ✗：这一层是引擎内核 ✓，**在哪算必须由调用方说清** ✓；
    「有卡用卡」这种**择优**只许发生在配置层（:class:`~.llm_backend.LlmBackend`）✓ ——
    否则老测试会静默跑到卡上 ✗、CPU 口径的失败也没人发现 ✗✗。要择优请显式写 ``device="auto"`` ✓。
    """
    build = _build()
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, device=resolved_device)
    torch = build["torch"]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(resolved_dtype)      # ⚠️ 全局状态 ⇒ 必须 finally 复原 ✓
    try:
        model = build["LlmModel"](config)
    finally:
        torch.set_default_dtype(previous)
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
    """造出需要 torch 的类/函数 ✓（模块级定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。"""
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
                    past_kv: tuple[Any, Any] | None = None) -> tuple[Any, tuple[Any, Any]]:
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
            past_len = 0
            if past_kv is not None:
                pk, pv = past_kv
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
                past_len = int(pk.shape[2])
            new_kv = (k, v)
            k_rep = repeat_kv(k, self.n_rep)
            v_rep = repeat_kv(v, self.n_rep)
            scores = q @ k_rep.transpose(-2, -1) / math.sqrt(self.head_dim)
            # ⭐ 因果掩码 ✓：当前位置（past_len+i）只许看 ≤ 它自己的位置 ✓✗
            # ⚠️ mask 的 dtype 必须跟 scores 一致 ✗：fp32 的 mask 加到 bf16 的 scores 上会**广播成 fp32**
            # ⇒ 后面 ``attn @ v`` 撞 `expected scalar type BFloat16 but found Float` ✗✗（低精度专炸 ✓）。
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
                    past_kv: tuple[Any, Any] | None = None) -> tuple[Any, tuple[Any, Any]]:
            attended, new_kv = self.attn(self.norm1(x), cos, sin, past_kv)
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

        def forward(self, ids: Any, past_kv: Any = None) -> tuple[Any, Any]:
            batch, seq = ids.shape
            past_len = 0 if past_kv is None else int(past_kv[0][0].shape[2])
            positions = torch.arange(past_len, past_len + seq, device=ids.device)
            cos, sin = rope_freqs(positions, self.head_dim, self.rope_theta)
            x = self.embed(ids)
            new_kv: list[tuple[Any, Any]] = []
            for index, block in enumerate(self.blocks):
                layer_kv = None if past_kv is None else past_kv[index]
                x, kv = block(x, cos, sin, layer_kv)
                new_kv.append(kv)
            x = self.norm(x)
            return self.lm_head(x), new_kv

    def generate_llm(model: Any, ids: Any, *, max_new_tokens: int = 64,
                     temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
                     eos_id: int | None = None, seed: int | None = None) -> Any:
        """逐 token 生成 ✓（greedy / temperature / top-p / top-k ✓ + 早停 ✓）。

        ⚠️ ``temperature=0`` ⇒ greedy ✓（直接 argmax ✓ —— 保证可复现 ✓，也是**跨设备一致**的判据 ✓）。
        ⚠️ ``seed`` 只对**随机采样**有用 ✓：它种的是 torch **本进程全部设备**的 RNG ✓，所以同一设备、
        同一份权重/输入 ⇒ 逐 token 复现 ✓；⚠️ **跨设备不保证逐 token 相同** ✗ —— 两端 logits 本身就差
        1e-5 量级 ✓，再从概率分布里抽样 ⇒ 尾部 token 会分叉 ✓（这是数学 ✓ 不是 bug ✓）。
        要跨设备逐字相同 ⇒ 用 ``temperature=0`` ✓（:class:`~.llm_backend.LlmBackend` 的一致性测试就这么判 ✓）。
        """
        if max_new_tokens < 1:
            raise LlmError(f"max_new_tokens 必须 ≥1（收到 {max_new_tokens} ✗）")
        if temperature < 0:
            raise LlmError(f"temperature 不能为负（收到 {temperature} ✗）")
        if seed is not None:
            torch.manual_seed(int(seed))       # ⚠️ 种全部设备 ✓（cuda 的 generator 也在内 ✓）
        ids = ids.to(model_device(model))      # ⚠️ 拼 id 的调用方常忘了搬 ✓（HF 同口径 ✓ 移动而不报错 ✓）
        past_kv = None
        current = ids
        generated: list[Any] = []
        for _ in range(max_new_tokens):
            logits, past_kv = model(current, past_kv)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                # ⚠️ 采样一律在 **fp32** 里做 ✓：top-p 的阈值比较与多项式抽样都对**概率精度**敏感 ✓，
                # bf16 只有 8 位尾数 ⇒ 尾部概率会被压平/抹掉 ✓（argmax 不受影响 ✓ —— 它只比大小 ✓）。
                # 副作用是好的 ✓：概率在两端算得更接近 ⇒ 随机采样也不至于一上来就分叉 ✓。
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                if top_k > 0:
                    topk = torch.topk(probs, min(top_k, probs.shape[-1]))
                    mask = torch.full_like(probs, float("-inf"))
                    mask.scatter_(1, topk.indices, topk.values)
                    probs = torch.softmax(mask, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_probs[remove] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                token = torch.multinomial(probs, 1)
            current = token
            generated.append(token)
            if eos_id is not None and bool((token == eos_id).all()):
                break
        return torch.cat([ids] + generated, dim=1)

    return {
        "torch": torch, "nn": nn, "F": F, "RMSNorm": RMSNorm, "GQAAttention": GQAAttention,
        "SwiGLU": SwiGLU, "TransformerBlock": TransformerBlock, "LlmModel": LlmModel,
        "rope_freqs": rope_freqs, "apply_rope": apply_rope, "repeat_kv": repeat_kv,
        "generate_llm": generate_llm,
    }


def __getattr__(name: str) -> Any:
    """惰性导出 ✓（**先查 ``__all__``** ✓ 再建积木 ✓ —— 与 h3_form 同一纪律 ✓）。"""
    if name.startswith("_") or name not in __all__:
        raise AttributeError(f"llm 没有 {name} ✓")
    return _build()[name]
