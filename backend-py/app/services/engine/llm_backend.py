"""自研文本后端（2026-09-25 起）—— **进程内** LLM 推理 ✓，不 HTTP 调 ollama ✗。

## 定位

:mod:`llm`（架构 ✓）+ :mod:`gguf_dequant`（反量化 ✓）+ :mod:`gguf_to_llm`（装载 ✓）三块拼图之后，
本模块把它们收成**一个后端** ✓（像 :class:`TorchBackend` 那样 ✓）：``describe`` 自述 + ``load`` 装载 +
``generate`` 生成 ✓ —— 文本生成由此能**绕过 ollama** ✓（「文本不依赖第三方」的最后一跳 ✓）。

## 能力自述（describe ✓）

权重 / 词表路径就绪与否、模型装载与否 ⇒ ``canGenerate`` ✓ —— ⚠️ **真权重没到就说 False** ✓ 不冒充 ✗
（与 :class:`TorchBackend` 同一条纪律 ✓）。

## 生成（generate ✓）

``prompt(+system) → 对话骨架 → tokenizer.encode → llm.generate_llm → tokenizer.decode`` ✓。

对话骨架（chat 模板 ✓ **2026-09-25 补** ✓）走 :mod:`chat_template` ✓：模板**从权重元数据取** ✓
（``tokenizer.chat_template`` ✓ —— **不内置任何模型骨架** ✗✗）；**没有** ⇒ 按裸拼接走 ✓（``describe``
如实标 ✓）、**坏了** ⇒ **当场拒** ✗（不静默回落裸拼接 ✗✗）。

## 词表从哪来（2026-10-29 补 ✓）

Ollama 库里的模型**只有 GGUF** ✗ ⇒ 词表走 :mod:`.tokenizer_gguf`（读 ``tokenizer.ggml.*`` ✓）；
给了 HF 词表路径就走 :mod:`.tokenizer_hub` ✓。⚠️ ``add_special_tokens=True`` 在 GGUF 词表上
**不加** BOS/EOS ✗ —— Qwen3 的 ``<|im_start|>`` 等由对话骨架写进文本 ✓，而生成侧若把 EOS
拼在提示词尾部会**直接算错** ✗✗（llama.cpp 里 ``add_bos``/``add_eos`` 也是可关的 ✓）。

## 在 CPU / GPU 上跑（2026-10-29 补 ✓）

**择优只在这一层** ✓：``device="auto"``（默认）⇒ 有卡用卡、没卡用 CPU ✓；显式 ``device="cuda"``
而用不了 ⇒ **具名报错** ✗✗（不静默回落 ✗ —— 回落会让「8B 慢」被当成模型慢 ✗）。默认精度随设备
（CPU ⇒ fp32 ✓、CUDA ⇒ bf16 ✓，因为 8B 的 fp32 是 32 GB ✗ 装不进 24 GB 卡 ✓）；``dtype="float32"``
可在卡上也强用 fp32 ✓（跨设备逐位对齐的判据口径 ✓）。

``describe`` 里 ``device`` 是**实际**值（``load()`` 后才有 ✓）、``deviceRequested`` 是请求值 ✓ ——
两栏分开才不会把「要了卡、其实在 CPU 跑」看成「已经用上卡」✗✗。

## 两条约束

1. **同步** ✓：``generate`` 会阻塞 ✗（CPU 更明显 ✗）—— 接线时用 ``asyncio.to_thread`` 包 ✓（见 text_generation ✓）；
2. **懒导入 torch** ✓（模块级不 import ✓）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import chat_template
from . import device_policy
from . import gguf
from . import gguf_to_llm
from . import llm as llm_mod
from . import tokenizer_gguf
from . import tokenizer_hub

__all__ = ["LlmBackend", "LlmBackendUnavailable"]

#: 精度名 ⇒ 字节数 ✓（算层预算用 ✓）。⚠️ 不走 ``torch.dtype.itemsize`` ✗：那一格要 torch ≥2.1 ✓，
#: 而「算预算」这件事**本来就不该拖 torch** ✓（回执/预检可以先算 ✓）。
_DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "float": 4, "float64": 8, "double": 8,
    "float16": 2, "half": 2, "bfloat16": 2,
}

#: prefill 分块的**默认档**（词元）✓ —— 与 :func:`~.llm.generate_llm` 的 ``prefill_chunk`` 默认值
#: 同源 ✓（两处不同 ⇒ 回执写 1024、实际按别的数跑 ✗✗，谁也发现不了 ✓）。
PREFILL_CHUNK_DEFAULT = 1024

#: 空闲显存下限（GiB）⇒ prefill 块大小 ✓：**只缩不涨** ✗（``≥ 4 GiB`` 就是默认档 ⇒ 常见环境
#: 一个字节都不变 ✓；涨没有实测收益却把 ``O(chunk × 总长)`` 的峰值翻倍 ✓✗）。
PREFILL_CHUNK_TIERS: tuple[tuple[float, int], ...] = (
    (4.0, PREFILL_CHUNK_DEFAULT),
    (2.0, 512),
    (1.0, 256),
    (0.0, 128),
)


class LlmBackendUnavailable(RuntimeError):
    """后端不可用 ✓（权重/词表没就绪、或没装载 ✓）—— ``reason`` 说明缺什么 ✓。"""


class _StopHit(Exception):
    """内部哨兵 ✓：文字级停用词命中 ⇒ 借「回调抛异常=中止采样」收尾 ✓（不对外 ✓ 不进 ``__all__`` ✓）。"""

    def __init__(self, mark: str) -> None:
        super().__init__(mark)
        self.mark = mark


class _Aborted(Exception):
    """内部哨兵 ✓：调用方要求中止（客户端断开 ✓）⇒ 与 ``_StopHit`` 同一手法，但**不**再补发文本 ✗。

    ⚠️ 为什么必须有这个哨兵 ✗✗：``asyncio.to_thread`` 包住的同步推理**取消不了** ✗ —— 任务被
    ``task.cancel()`` 之后线程仍在把 ``max_new_tokens`` 烧完 ✓✗（本地 CPU/GPU 推理尤其明显 ✗）。
    真正的止损只能在**采样循环内部**判一个 ``threading.Event`` ✓。
    """


def _context_length_of(metadata: Any) -> int:
    """元数据 ⇒ **上下文窗口** ✓（``<arch>.context_length`` ✓）；没有 ⇒ ``0`` ✓（**不猜** ✗）。

    ⚠️ 键名前缀随架构变（``qwen3.context_length`` / ``llama.context_length`` ✓）⇒ 按**后缀**认 ✓，
    并且只认第一个非零整数 ✓（同一份元数据里没有第二个同名语义键 ✓）。
    """
    for key, value in (metadata or {}).items():
        if not str(key).endswith(".context_length"):
            continue
        if isinstance(value, bool):        # bool 是 int 的子类 ⇒ 必须先挡 ✓
            continue
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return 0


class LlmBackend:
    """自研文本后端 ✓（``describe`` / ``load`` / ``generate`` / ``generate_messages`` ✓）。"""

    name = "llm"
    #: 张量与推理都是**真的** ✓（不是 dryrun 的假张量 ✗）—— 但**没装真权重时 generate 会拒** ✗
    realTensors = True

    def __init__(self, config: llm_mod.LlmConfig, *, gguf_path: str | None = None,
                 tokenizer_path: str | None = None, device: Any = "auto",
                 dtype: Any = None, gpu_layers: Any = None, kv_mode: Any = None,
                 kv_cache_bytes: Any = None, enable_thinking: bool | None = None,
                 spec_lookup: Any = None, spec_window: Any = None,
                 spec_min_match: Any = None, fast_path: Any = None,
                 kv_window: Any = None, kv_sink: Any = None,
                 cuda_graph: Any = None) -> None:
        self._config = config
        self._gguf_path = gguf_path
        self._tokenizer_path = tokenizer_path
        #: ``None`` = 不定义该模板变量 ✓（由骨架自己的默认决定 ✓）；``False`` = 不要思考过程 ✓
        self._enable_thinking = enable_thinking
        # ⚠️ ``device``/``dtype`` 只在 **load() 那一刻**解析 ✓（构造函数不许碰 torch ✗：describe 要能离线跑 ✓）。
        self._device_request = "auto" if device is None else str(device)
        self._dtype_request = "" if dtype is None else str(dtype)
        #: ⭐ 分层放置的**请求** ✓（``None``/空 = 不分层 ✓ 老路径逐字节相同 ✓；``0`` = 全留 CPU；
        #: ``N`` = 前 N 层上卡；``auto`` = 按此刻可用显存铺满 ✓）。只在 ``load()`` 那一刻算 ✓。
        self._gpu_layers_request = "" if gpu_layers is None else str(gpu_layers)
        #: ⭐ KV cache 口径 ✓（``fp`` 老路径 ✓／``q8`` INT8 分组量化 ✓ 2026-11-08 补 ✓）。
        #: 请求值原样存着 ✓（拼错在 ``load()`` 那一刻**具名拒** ✗ —— 构造期不抛 ⇒ describe 仍能离线出回执 ✓）。
        self._kv_mode_request = "" if kv_mode is None else str(kv_mode)
        #: 生效口径 ✓（装载后才确定 ✓；装载失败退回 ``fp`` ✓ —— 绝不「以为开了量化其实没开」✗✗）
        self._kv_mode: str = "fp"
        #: ⭐ 跨调用 KV 前缀复用的**预算** ✓（``0`` = 关 ✓；``None`` = 默认 512 MiB ✓ 2026-11-08 补 ✓）。
        #: ⚠️ 归一放在**构造期**（纯算术、不碰 torch ✓）⇒ describe 在装载前也能报准这个数 ✓。
        try:
            self._kv_cache_budget = llm_mod.resolve_kv_cache_bytes(kv_cache_bytes)
        except Exception:  # noqa: BLE001 —— 拼错别在构造期砸掉 describe ✗：装载那一刻具名拒 ✓
            self._kv_cache_budget = -1
        self._kv_cache_request = kv_cache_bytes
        #: 生效的常驻 cache ✓（``load()`` 里建 ✓；``None`` = 关／还没装 ✓）
        self._kv_cache: Any = None
        #: ⭐ 投机解码（2026-11-08 ✓）：零权重后缀查表草稿 ✓。**默认关** ✓ —— 只在贪心下成立 ✓
        #: （随机采样要概率接受 ＋ Philox 空间隔离 ✗ 本仓没有 ⇒ 开了也只会在回执里报退回原因 ✓）。
        #: ⚠️ 归一失败**不在构造期抛** ✗（与 ``kvCacheBytes`` 同一条纪律 ✓：describe 要能离线出回执 ✓）
        #: —— 记下那句错 ✓ 由 ``load()`` 在花几十秒装权重**之前**具名拒 ✗✗。
        self._spec_request = None if spec_lookup is None else str(spec_lookup)
        self._spec_error: str = ""
        try:
            self._spec_lookup = llm_mod.resolve_spec_lookup(spec_lookup)
            self._spec_window = llm_mod.resolve_spec_window(spec_window)
            self._spec_min_match = llm_mod.resolve_spec_min_match(spec_min_match)
        except Exception as err:  # noqa: BLE001 —— 拼错别砸掉 describe ✗：装载那一刻具名拒 ✓
            self._spec_lookup = False
            self._spec_window = llm_mod.SPEC_WINDOW
            self._spec_min_match = llm_mod.SPEC_MIN_MATCH
            self._spec_error = str(err)
        #: 上一次生成的**投机回执** ✓（``None`` = 还没生成过 ✓ —— 与「生成了但没开」不是一回事 ✗✗）
        self._spec_last: dict[str, Any] | None = None
        #: ⭐ 快路径（2026-11-08 ✓）：``norm``＝融合 RMSNorm ✓、``attn``＝SDPA＋GQA ✓。
        #: **默认关** ✓ —— 两个都**改数值**（累加顺序不同 ✓）⇒ 不许默认开 ✗✗。归一失败同样
        #: 不在构造期抛 ✗（describe 要能离线出回执 ✓）⇒ 记错、由 ``load()`` 在装权重前具名拒 ✓。
        self._fast_request = None if fast_path is None else str(fast_path)
        self._fast_error: str = ""
        try:
            self._fast_path = llm_mod.resolve_fast_path(fast_path)
        except Exception as err:  # noqa: BLE001 —— 拼错别砸掉 describe ✗
            self._fast_path = frozenset()
            self._fast_error = str(err)
        #: ⭐ KV 滑窗（2026-11-08 补 ✓）：尾窗行数 ＋ 开头 sink 行数 ✓。``0`` = **不滑窗**（老路径 ✓）。
        #: ⚠️ 这是**有损**档 ✗✗（中间那段历史真的丢掉 ✓）⇒ 默认必须是 0 ✓；且与跨调用前缀复用
        #: **结构互斥** ✓（复用存的是一整段真实历史 ✓，滑窗后那段已经不存在了 ✗）—— 见 :meth:`_kv_window_call`。
        #: 归一失败同样不在构造期抛 ✗（describe 要离线能出回执 ✓）⇒ 记错、``load()`` 装权重前具名拒 ✓。
        self._kv_window_request = None if kv_window is None else str(kv_window)
        self._kv_sink_request = None if kv_sink is None else str(kv_sink)
        self._kv_win_error: str = ""
        try:
            self._kv_window = llm_mod.resolve_kv_window(kv_window)
            self._kv_sink = llm_mod.resolve_kv_sink(kv_sink)
        except Exception as err:  # noqa: BLE001 —— 拼错别砸掉 describe ✗
            self._kv_window = 0
            self._kv_sink = llm_mod.KV_SINK_DEFAULT
            self._kv_win_error = str(err)
        #: ⭐ CUDA Graph（2026-11-08 补 ✓）：把 **decode 单步**捕成一张图再重放 ✓（省 launch ✓ 不改算力 ✓）。
        #: **默认关** ✓ —— 掩码把 softmax 归约长度改成桶容量 ✓ ⇒ 与老路**不逐位相同** ✓ 不许默认换 ✗✗。
        #: 归一失败同样不在构造期抛 ✗（describe 要离线能出回执 ✓）⇒ 记错、``load()`` 装权重前具名拒 ✓。
        self._graph_request = None if cuda_graph is None else str(cuda_graph)
        self._graph_error: str = ""
        #: 三档原文 ✓（2026-11-09 补 ✓）：``off``／``on``／``auto`` ✓ —— ``_cuda_graph`` 只能回答「可能开 ✓」，
        #: 传下去必须用本字段 ✗✗（否则 ``auto`` 又变成「无条件开」✓，短问答白花 100 ms 捕获 ✓✗）。
        self._graph_policy = "off"
        try:
            self._graph_policy = llm_mod.resolve_cuda_graph_policy(cuda_graph)
            self._cuda_graph = self._graph_policy != "off"
        except Exception as err:  # noqa: BLE001 —— 拼错别砸掉 describe ✗
            self._cuda_graph = False
            self._graph_error = str(err)
        #: 上一次生成的**图化回执** ✓（``None`` = 还没生成过 ✓ ≠「生成了但没开」✗✗）
        self._graph_last: dict[str, Any] | None = None
        #: 装载后的**层放置回执** ✓（``None`` = 没分层／还没装 ✓ —— 与「装了但算出 0 层」不是一回事 ✗✗）
        self._layers: dict[str, Any] | None = None
        self._device = ""          # 装载后**实际**设备 ✓（如 ``cuda:0`` ✓）
        self._dtype = ""           # 装载后**实际**精度 ✓（如 ``bfloat16`` ✓）
        self._model: Any = None
        self._hub: Any = None
        self._load_error: str = ""
        self._chat_template: chat_template.ChatTemplate | None = None
        self._chat_template_error: str = ""
        self._chat_template_note: str = "未读取 ✓（load() 之后才有结论 ✓）"
        #: 反量化产物缓存（G3 ✓）的结论 ✓：``None`` = **还没装载** ✓（≠ "没命中" ✗）——
        #: 装载后**原样**搬 :attr:`gguf_to_llm` 挂的 ``model.materialized`` ✓（不在这里重算口径 ✗）。
        self._materialized: dict[str, Any] | None = None
        #: 权重声明的**上下文窗口** ✓（GGUF 元数据的 ``<arch>.context_length`` ✓）——``0`` = 没声明 ✓。
        #: 只用来「超窗当场拒」✗，**不拿它当默认出生成长度** ✗（那是调用方的 ``max_new_tokens`` ✓）。
        self._context_length: int = 0
        #: ⭐ **显存侧**的 KV 上限 ✓（2026-11-08 补 ✓）：见 :meth:`refresh_kv_budget` ✓。
        #: ``None`` = 还没算过 ✓（≠「算出来是 0」✗ —— 0 是「一个 token 都放不下」✓）。
        #: ⚠️ 与 ``_context_length``（**权重的**上限 ✗）是**两本账** ✗✗：低显存下先撞的永远是这一本 ✓。
        self._kv_budget: dict[str, Any] | None = None

    # ── 自述 ✓
    def describe(self) -> dict[str, Any]:
        """能力自述 ✓（**只查路径/状态，不做张量操作** ✓）。

        ⚠️ 设备栏必须分**请求值**与**实际值**两栏 ✓✗：只报请求值 ⇒ 「要了 cuda、其实在 CPU 跑」
        看不出来 ✗✗（本仓老账：好默认值把「没生效」伪装成「已生效」✗）。``load()`` 之前 ``device`` 是空 ✓。
        """
        missing: list[str] = []
        if not self._gguf_path or not Path(self._gguf_path).exists():
            missing.append("GGUF 权重未就绪")
        source, source_note = self._tokenizer_source()
        if source is None:
            missing.append(f"词表未就绪（{source_note}）" if source_note else "词表未就绪")
        if self._model is None:
            missing.append("模型未装载（load() 后即可用）")
        devices = llm_mod.device_facts() if self._model is None else {}
        return {
            "name": self.name,
            "realTensors": self.realTensors,
            "available": not missing,
            "reason": "；".join(missing),
            "modelLoaded": self._model is not None,
            "tokenizerReady": self._hub is not None,
            "canGenerate": self._model is not None and self._hub is not None,
            "ggufPath": self._gguf_path or "",
            "tokenizerPath": self._tokenizer_path or "",
            "tokenizerSource": source or "",
            "tokenizerNote": source_note,
            "loadError": self._load_error,
            "deviceRequested": self._device_request,
            "device": self._device,
            "deviceNote": self._device_note(),
            #: ⭐ 分层放置（2026-11-08 ✓）：请求值 ＋ **算出来的**层放置 ✓。
            #: ``layers`` 为 ``None`` ⇒ 没配／还没装 ✓；装了且算出 0 层也**是**一条回执（带 ``why`` ✓）。
            "gpuLayersRequested": self._gpu_layers_request or "off",
            "layers": self._layers,
            #: ⭐ KV cache 口径（2026-11-08 ✓）：请求值 ＋ **生效值** ＋ 每 token 字节 ✓。
            #: ⚠️ 只报百分比没用 ✗ —— 上下文长度是变量 ✓，能乘起来的 per-token 字节才是可判读的 ✓。
            "kvModeRequested": self._kv_mode_request or "off",
            "kv": self._kv_report(),
            #: ⭐ **显存侧**的 KV 上限（2026-11-08 补 ✓）：``None`` = 还没装载／没算过 ✓；
            #: 装载后是 ``{mode, perToken, maxTokens, effectiveTokens, limitedBy, freeBytes, ...}`` ✓。
            #: ⚠️ 与 ``kv.contextLength``（**权重**声明 ✓）是两本账 ✗✗ —— ``limitedBy`` 就是「先撞哪一本」✓。
            "kvBudget": self._kv_budget,
            #: ⭐ KV 滑窗（2026-11-08 补 ✓）：``0`` = 关（老路径 ✓）；开了就是「cache 恒为
            #: sink+window 行」✓（**有损** ✗ —— 中间历史真丢 ✓，位置仍按绝对位置 ✓）。
            "kvWindow": self._kv_window_report(),
            #: ⭐ 跨调用 KV 前缀复用（2026-11-08 ✓）：预算 ＋ 命中/复用 token/当前占用 ✓。
            #: ⚠️ 命中数必须**真的报出来** ✗✗：不报 ⇒ 「配了复用」与「一次没命中（前缀其实不同）」分不清 ✗。
            "kvCache": self._kv_cache_report(),
            #: ⭐ 投机解码（2026-11-08 ✓）：开关 ＋ 参数 ＋ **上一次真实的**轮数/草稿/接受数 ✓。
            "spec": self._spec_report(),
            #: ⭐ 快路径（2026-11-08 ✓）：``norm``/``attn`` 两个融合档的生效与否 ＋ 代价说明 ✓。
            "fastPath": self._fast_report(),
            #: ⭐ 图化 decode 单步（2026-11-08 补 ✓）：开关 ＋ **上一次真实的**捕获/重放账 ✓
            #: （``enabled=False`` 时 ``reason`` 必写明是没要求、还是被哪条前置挡了 ✗✗）。
            "cudaGraph": self._graph_report(),
            #: ⭐ prefill 分块（2026-11-08 ✓）：显存一紧就缩块 ✓（峰值 ``O(chunk × 总长)`` ✓）；
            #: 默认档＝1024 ⇒ 与老路径逐字相同 ✓。**只缩不涨** ✗（涨没收益、峰值翻倍 ✓✗）。
            "prefill": self._prefill_report(),
            "dtypeRequested": self._dtype_request or "auto",
            "dtype": self._dtype,
            "devices": devices.get("devices", []) if devices else [],
            #: G3 反量化产物缓存 ✓：``None`` = **还没装载** ✓（不是"没命中" ✗）；
            #: 装载后是 ``{hit, gib, path, reason, ...}`` ✓ / 显式关掉时 ``{enabled: False, reason}`` ✓。
            "materialize": self._materialized,
            "chatTemplate": {
                "present": self._chat_template is not None,
                "origin": self._chat_template.origin if self._chat_template is not None else "",
                #: ``None`` = 没管这个变量 ✓（骨架按自己的默认走 ✓，本层不替它编值 ✗）
                "enableThinking": self._enable_thinking,
                "note": self._chat_template_note,
            },
        }

    def _kv_report(self) -> dict[str, Any]:
        """KV cache 的**账** ✓（每 token 字节 ＋ 按窗口乘出来的总值 ✓）。

        ⚠️ 装载**之前**也照样给数 ✗✗（不是「未知」✓）：拿不准的只有精度那一格 ✓ —— 那就在 ``note``
        里写明「按 2 B 估 ✓」✓ 而不是把整栏留空 ✗（留空的回执读不出「省了多少」✓ 等于没报 ✗）。

        口径按**请求值**归一 ✓（装载前配了 ``q8`` 就报 q8 的字节 ✓）—— 否则「配了 q8 的预检回执
        报 fp」会让人以为配置没吃 ✓✗。拼错在 describe 这一层**不抛** ✗（describe 要离线能跑 ✓）：
        记进 ``note`` ✓，真正的拒绝在 :meth:`load` 那一刻 ✗。
        """
        assumed = not self._dtype
        dtype_bytes = _DTYPE_BYTES.get(self._dtype or "", 2)
        mode, prefix = self._kv_mode, ""
        if self._model is None:
            try:
                mode = llm_mod.resolve_kv_mode(self._kv_mode_request)
                prefix = "（装载后生效 ✓）" if mode != "fp" else ""
            except Exception as err:  # noqa: BLE001 —— 只在这里标注 ✗ 不在 describe 抛 ✗
                mode, prefix = "fp", f"（请求值 {self._kv_mode_request!r} 不认 ✗：{err}）"
        try:
            report: dict[str, Any] = dict(llm_mod.kv_bytes_per_token(
                self._config, mode, dtype_bytes=dtype_bytes))
        except Exception as err:  # noqa: BLE001 —— 算不出就得说清 ✗ 不许编 0 ✗
            return {"mode": mode, "note": f"算不出每 token 预算 ✗（{err}）"}
        report["mode"] = mode
        report["dtypeBytes"] = dtype_bytes
        report["assumedDtype"] = assumed
        report["contextLength"] = int(self._context_length or 0)
        if report["contextLength"]:
            report["contextBytes"] = report["totalBytes"] * report["contextLength"]
            report["fpContextBytes"] = report["fpTotalBytes"] * report["contextLength"]
        bits = f"（精度未定 ⇒ 按 {dtype_bytes} B 估 ✓）" if assumed else ""
        # ⚠️ **省了字节就必须交出花掉的时间** ✗✗（2026-11-09 探针 ⑦ 实测 ✓，与 cudaGraph 的 ``captureMs``
        # 同一条纪律 ✓）：``kv_bytes_per_token`` 只会说「省 47% B」✓ ⇒ 人会读成白拿 ✓✗。实测 q8 每步
        # **+8–10 ms（×1.29–1.38）**，且 256 行与 1024 行**两个点几乎一样** ⇒ 更接近**固定成本** ✓
        # ⇒ 别按「上下文越长越亏」外推 ✗（这句话本探针没量出来就成了想当然 ✓）。
        # 关着的档也要带这句 ✓：不打开的人**永远看不到**打开的代价 ✓✗（与 cudaGraph 默认关同一个坑 ✓）。
        cost = ("· 实测每步 **+8–10 ms（×1.29–1.38）** ✓（4B-Q4_K_M／A5000，256 与 1024 行两个点 ≈ 同量级 "
                "⇒ 更接近固定成本 ✓ 别按「越长越亏」外推 ✗）"
                if mode == "q8" else
                "· ``q8`` 可省约 47% KV 字节，代价是每步 **+8–10 ms（×1.3 左右）** ⇒ 它省**显存**不省时间 ✓")
        report["note"] = f"{prefix}{report['note']}{bits}{cost}"
        return report

    def _kv_window_report(self) -> dict[str, Any]:
        """KV 滑窗的回执 ✓（装载前／装载后、开／关都能出 ✓ 不抛 ✗）。

        ⚠️ 必须单列一栏 ✗✗：判 ``kv`` 里的每 token 字节看不出「历史会不会无限涨」✓ ——
        那是**行数**问题不是每行字节问题 ✓；人问「长对话会不会 OOM」时唯一能回答的就是这一栏 ✓。
        """
        if self._kv_win_error:
            return {"enabled": False, "window": 0, "sink": 0,
                    "note": f"kvWindow/kvSink 不认 ✗（{self._kv_win_error}）—— 装载时具名拒 ✓"}
        if not self._kv_window:
            return {"enabled": False, "window": 0, "sink": 0, "rows": 0,
                    "note": ("关 ✓（老路径：历史全留 ⇒ KV 随长度线性涨 ✓；"
                             "低显存下判断「还能放多少 token」看 ``kvBudget`` ✓）")}
        rows = self._kv_sink + self._kv_window
        return {"enabled": True, "window": self._kv_window, "sink": self._kv_sink, "rows": rows,
                "note": (f"滑窗**有损** ✓：留开头 {self._kv_sink} 行（attention sink ✓）"
                         f"＋最近 {self._kv_window} 行 ⇒ cache 恒定 {rows} 行 ✓"
                         f"（位置按**绝对**位置走 ✓ 不跳相位 ✓）；"
                         f"与 kvCache 前缀复用互斥 ⇒ 复用已暂停 ✓")}

    def _kv_cache_report(self) -> dict[str, Any]:
        """跨调用前缀复用的回执 ✓（装载前／装载后都能出 ✓ 不抛 ✗）。"""
        if self._kv_window:
            # ⚠️ 不许静默：滑窗开着 ⇒ 本次生成的 cache 一定是 ``None`` ✓（见 :meth:`_kv_cache_for_call`）✓
            # 不报 ⇒ 人按「复用在跑」算时间/显存 ✓ 而实际一拍都没命中 ✓✗。
            base = dict(self._kv_cache.describe()) if self._kv_cache is not None else {
                "budgetBytes": int(self._kv_cache_budget)}
            base.update({"enabled": False, "suspendedBy": "kvWindow",
                         "note": (f"暂停 ✓（kvWindow={self._kv_window} 与完整历史复用**结构互斥** ✓："
                                  f"滑窗后中间那段不存在了 ⇒ 复用回来会让模型看见「本不该记得」的行 ✓）")})
            return base
        budget = int(self._kv_cache_budget)
        if self._kv_cache is not None:
            report = dict(self._kv_cache.describe())
            if not report.get("note"):
                report["note"] = ("常驻中 ✓ —— 命中就跳过公共前缀的 prefill ✓（装不下/前缀不同都如实记 ✓）"
                                  if report.get("enabled") else "关闭 ✓")
            return report
        if budget < 0:
            return {"enabled": False, "budgetBytes": None, "mode": self._kv_mode,
                    "note": (f"kvCacheBytes={self._kv_cache_request!r} 不认 ✗ "
                             "（可写 0 / off / N / 512MiB / 1GiB ✓）—— 装载时具名拒 ✓")}
        return {"enabled": budget > 0, "budgetBytes": budget, "mode": self._kv_mode,
                "storedTokens": 0, "bytes": 0, "hits": 0, "misses": 0, "skipped": 0,
                "reusedTokens": 0, "stored": 0,
                "note": ("装载后生效 ✓（命中就跳过公共前缀的 prefill ✓）" if budget
                         else "预算 0 ⇒ 跨调用复用关闭 ✓（每次调用都从头 prefill ✓）")}

    def _spec_report(self) -> dict[str, Any]:
        """投机解码回执 ✓（装载前／装载后、开／关、生成前／生成后都能出 ✓ 不抛 ✗）。

        ⚠️ 「开了」与「**真的投机了**」必须分开报 ✗✗：只报一个开关 ⇒ 温度 > 0 时
        ``enabled=False`` 也能被读成「优化在跑」✓（实际一拍一 token ✓）。
        """
        report: dict[str, Any] = {
            "requested": self._spec_request or "off",
            "window": self._spec_window,
            "minMatch": self._spec_min_match,
            "rounds": 0,
            "drafted": 0,
            "accepted": 0,
            "rate": None,
            "last": None,
        }
        if self._spec_error:
            report.update(enabled=False, acceptedRate=None,
                          note=f"{self._spec_error} —— 装载时具名拒 ✓")
            return report
        if not self._spec_lookup:
            report.update(enabled=False, acceptedRate=None,
                          note=("关闭 ✓（一拍一 token ✓ 老路径逐字节相同 ✓ —— "
                                "本仓实测收益 1.02~1.03× ✓ 默认不值得开 ✓）"))
            return report
        report["enabled"] = True
        last = self._spec_last
        if not last:
            report["note"] = ("已开 ✓ 还没生成过 ✓（**只在 temperature=0 生效** ✗ 随机采样会在 "
                              "``last.reason`` 里如实报退回 ✓）")
            return report
        for key in ("window", "rounds", "drafted", "accepted", "enabled", "reason"):
            if key in last:
                report[key] = last[key]
        drafted = int(report.get("drafted") or 0)
        report["rounds"] = int(last.get("rounds") or 0)
        report["last"] = dict(last)
        #: 接受率＝accepted / drafted ✓（**不是** accepted / 窗口行数 ✗：没草稿的轮次不进分母 ✓）
        report["acceptedRate"] = (round(int(report.get("accepted") or 0) / drafted, 4)
                                  if drafted else None)
        if report.get("enabled") is False:
            report["note"] = (f"上次调用**退回**了 ✓ 原因 {last.get('reason')!r} "
                              f"（退回＝逐 token 老路径 ✓ 结果与不开一致 ✓）")
        else:
            report["note"] = (f"上次调用：{report['rounds']} 轮 / 草稿 {drafted} / "
                              f"接受 {report.get('accepted')} ✓")
        return report

    def _fast_report(self) -> dict[str, Any]:
        """快路径回执 ✓（装载前／装载后、开／关都能出 ✓ 不抛 ✗）。

        ⚠️ **配了 ≠ 生效** ✗✗：``norm`` 要靠 ``F.rms_norm``（torch ≥ 2.4 ✓），缺了会静默退回老写法 ✓
        ⇒ ``fusedNormAvailable`` 单独报一格 ✓（与 ``spec`` 里「配了 ≠ 真的投机了」同一条纪律 ✓）。
        """
        report: dict[str, Any] = {
            "requested": self._fast_request or "off",
            "norm": "norm" in self._fast_path,
            "attn": "attn" in self._fast_path,
            "kvbuf": "kvbuf" in self._fast_path,
            "items": sorted(self._fast_path),
            "enabled": bool(self._fast_path),
            "fusedNormAvailable": bool(llm_mod.fused_norm_available()),
        }
        if self._fast_error:
            report.update(enabled=False, norm=False, attn=False, kvbuf=False, items=[],
                          note=f"{self._fast_error} —— 装载时具名拒 ✓")
            return report
        if not self._fast_path:
            report["note"] = ("关闭 ✓（老路径**逐位不变** ✓ —— ``norm``/``attn`` 改数值 ⇒ 默认不开 ✓、"
                              "``kvbuf`` 只改存储布局 ✓ 但矩阵乘读跨步视图 ⇒ 也默认不开 ✓；"
                              "要省 launch 配 ``norm`` ✓、要省 GQA 物化配 ``attn`` ✓、"
                              "要省每步 KV 整段重拷配 ``kvbuf`` ✓、全都要配 ``all`` ✓）")
            return report
        parts: list[str] = []
        if "norm" in self._fast_path:
            # ⚠️ 次数栏写**实测**的数 ✗ 别写「5 个 kernel × 73 次」那种估算 ✓：真机 4B-Q4_K_M
            # （36 层、带 QK-norm ⇒ 一步 **145** 次调用 ✓）实测老路一次 **16** 个 kernel ✓
            # （bf16→fp32 往返 ＋ 两段 reduction ✓）⇒ 2320→290 次 launch ✓（`data/experiments/
            # llm_attn_cost/fast_path.py` 实测 ✓ 与整步降幅 4449−2030=2419 **逐次对平** ✓）。
            parts.append("RMSNorm → fused ✓（实测一步 2320→290 次 launch ✓ 墙钟 −15.6% ✓ "
                         "数值差约 1 个 bf16 ulp ✓）"
                         if report["fusedNormAvailable"]
                         else "RMSNorm → 本机**没有** F.rms_norm ✗ ⇒ 如实退回老写法（一步也没省 ✗）")
        if "attn" in self._fast_path:
            parts.append("注意力 → SDPA＋enable_gqa ✓（不吃 repeat_kv 物化 ✓ 数值与手写式不同 ✓）")
        if "kvbuf" in self._fast_path:
            # ⚠️ 别写「省 launch」✗✗：实测 4B/2048 词元一步 kernel 数 **4449→4449 一个不少** ✓
            # （72 次 ``aten::cat`` 换成 72 次 ``copy_`` ✓ 一比一 ✓）—— 省的是**带宽**：每步拷贝量
            # 288 MiB → 288 KiB ✓、CUDA 50.86→47.23 ms ✓、墙钟 −5.8% ✓、logit 最大差 0.000e+00 ✓。
            parts.append("KV → 定长缓冲＋原位写 ✓（decode 每步从「整段重拷 288 MiB」降到「写 1 行 288 KiB」✓ "
                         "墙钟 −5.8% ✓ **不是**省 launch（kernel 数不变 ✓）＋ 数值逐位相同 ✓ "
                         "＋ 已分配容量按翻倍涨 ✓）")
        report["note"] = "；".join(parts)
        return report

    def _fast_call(self) -> dict[str, Any]:
        """生成调用要递下去的快路径参数 ✓（**空集 ⇒ 空字典** ✓ ⇒ 调用签名与老路径逐字相同 ✓）。

        ⚠️ 不做成「三处各写一遍 ``fast_path=self._fast_path``」✗✗：三处手抄必漂移 ✓（本仓已栽过
        同一形状的坑 ✓）—— 判据 ``engine_llm_backend_test`` 的 ⑫⁸ 盯着三处调用都必须展开这一份 ✓。
        """
        return {"fast_path": self._fast_path} if self._fast_path else {}

    def _graph_call(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """生成调用要递下去的图化参数 ✓ ⇒ ``(kwargs, sink)``（**关着 ⇒ 空字典 ＋ None** ✓ 老签名不变 ✓）。

        ⚠️ 返回值是**二元组**而不是字典 ✗✗：``cuda_graph_report`` 要的是一块**每轮新建**的账本 ✓
        （三个生成口各跑各的 ✓ 共用一块会把上一轮的重放数留在这一轮 ✓），所以没法像
        :meth:`_fast_call` 那样塞进同一个字典里让调用方 ``**`` 展开 ✓ —— 那会把同一个字典对象发给三处 ✓。
        """
        if not self._cuda_graph:
            return {}, None
        sink: dict[str, Any] = {}
        # ⚠️ 递的必须是**三档原文** ✗ 不是 ``True`` ✗✗（2026-11-09 补）：``on`` 与 ``auto`` 在
        # ``generate_llm`` 里行为不同 ✓（后者短生成主动不捕 ✓）。递 ``True`` ⇒ auto 被悄悄升成 on ✓、
        # 而回执照旧报 ``auto`` ✓✗ —— 同一份回执里「说的」和「做的」不一致，是本仓明令禁止的形状。
        return {"cuda_graph": self._graph_policy, "cuda_graph_report": sink}, sink

    def _graph_report(self) -> dict[str, Any]:
        """回执：图化档的请求值 ＋ **上一次真实**的捕获/重放账 ✓（装载前也能出 ✓ 不抛 ✗）。

        ⚠️ 三态必须分清 ✗✗：``requested`` ＝配置想不想开 ✓；``last`` ＝**上一次生成**实际怎样 ✓
        （``None`` ＝还没生成过 ✓，**不是**「开了但没生效」✗）；``note`` ＝人话说明 ✓。
        """
        #: ★ ``policy`` 是**三档原文** ✓（2026-11-09 补）：``enabled=True`` 只说「可能开」✓，
        #: 光看它分不出 ``on``（每次都捕）与 ``auto``（短生成主动不捕 ✓）——
        #: 而这两档在用户眼里的含义完全不同 ✓✗（一个「按你要求」，一个「按长度替你决定」）。
        report: dict[str, Any] = {
            "requested": self._graph_request or "off",
            "policy": self._graph_policy,
            "enabled": self._cuda_graph,
            "last": self._graph_last,
        }
        if self._graph_error:
            report["note"] = f"{self._graph_error} —— 装载时具名拒 ✓"
            return report
        auto = self._graph_policy == "auto"
        auto_min = llm_mod.CUDA_GRAPH_AUTO_MIN_TOKENS
        if not self._cuda_graph:
            # ⚠️ **关着也要交收益** ✗✗（2026-11-09 补）：只写「关 ✓」⇒ 没打开的人**永远看不到**打开值不值 ✓
            # （enabled 分支里那些实测数只有已经开了才看得到 ✓✗ 这是个自锁 ✓）。代价＋收益在这里一并给，
            # 由读数的人决定 —— 实测口径来自探针 ⑥（每档都重新捕获＝每次请求的真实账）。
            report["note"] = ("关 ✓（老路径逐位不变 ✓ —— 掩码把归约长度改成桶容量 ⇒ 只有显式开才换 ✓）"
                              "· 打开的话：实测 128–512 词元 **×1.24–1.27**（4B-Q4_K_M／A5000），"
                              "代价是每请求多花 ~100 ms 捕获 ＋ `bitExact=False`（512 词元里 443 个与老路不同）"
                              "⇒ 要逐字复现就别开 ✓ 长文吞吐要快就值得开 ✓"
                              f"· 怕短问答吃亏可以写 `auto` ✓：按 maxOutputTokens ≥ {auto_min} 才捕"
                              f"（实测 8 词元 ×0.91＝净亏 ~28 ms、16 词元才回本 ✓）")
            return report
        last = self._graph_last
        if last is None:
            report["note"] = ("已开 ✓，还没生成过 ✓（首次生成时捕获一次 ⇒ 之后逐步重放 ✓ "
                              "隐含 fastPath 的 kvbuf 档 ✓ 回执 last.impliedKvbuf 会写明）"
                              if not auto else
                              f"auto ✓，还没生成过 ✓（按长度决定：maxOutputTokens ≥ {auto_min} 才捕获 ✓、"
                              f"< {auto_min} 直接走老路 ✓ —— 实测 8 词元开图 ×0.91＝净亏 ~28 ms ✓）")
        elif last.get("enabled"):
            pad = last.get("padRows")
            if pad is None:
                pad_text = ""
            elif int(pad) <= 16:
                # 默认预留 = ``1 + specWindow(8) + 1`` ＝ 10 行 ✓ 实测 8/16/32 词元 0 分叉 ✓ ⇒ 这一档报事实
                # ⚠️ 但**别外推到长生成** ✗✗：同一档 padRows=10 下 64 词元已分叉 11 个、128 词元 72 个 ✓
                pad_text = f"、多读 {int(pad)} 行 padding（默认预留量级 ✓ 短生成实测一致 ✓ 长生成仍会累积漂移 ✓）"
            else:
                # ⚠️ 是 ``kv_reserve_rows`` 被放大才到这里的 ✓ ⇒ 差异会静默变大，必须显式警示 ✗✗
                pad_text = (f"、多读 {int(pad)} 行 padding"
                            f" ⚠️ 超出默认预留 —— 实测 96 行时 bf16 |Δlogits| 1.9e-1、32 词元 25 个分叉"
                            f" ⇒ 输出可能与老路不同 ✓")
            # ⚠️ 捕获成本必须报 ✗✗：它**真实计在第一次生成里** ✓ ⇒ 只报「重放 N 步很快」而瞒住「捕获花了
            # 40 ms」，就是拿一次 32 词元的读数冒充「任何长度都 ×1.15」✓✗（生成越短越吃亏 ✓，实测见探针 ⑥）。
            ms = last.get("captureMs")
            cap_text = "" if ms is None else f"、捕获另花 {float(ms):.1f} ms（计在第一次生成里 ✓ 短生成先扣它 ✓）"
            # ⚠️ auto 档要说明「这次为什么捕」✗✗：不写的话人只会看到「auto ✓ 但这次真在重放」，
            # 读不出它是按长度算出来的 ✓（下一个短请求会自己退回老路 ✓）—— 所以整段拼完再追加，
            # **不能**插在隐式字符串拼接中间 ✗✗（那会把剩下的 f-string 变成语法错误）。
            auto_text = f"（auto 档：本次 maxOutputTokens ≥ {auto_min} ⇒ 捕 ✓）" if auto else ""
            report["note"] = (f"上一次真在重放 ✓：桶容 {last.get('cap')} 行、起步历史 "
                              f"{last.get('pastLen')} 行{pad_text}{cap_text}、重放 {last.get('replays')} 步 ✓ "
                              f"{auto_text}"
                              f"⚠️ bitExact=False（不承诺与老路逐字相同 —— 要逐字比对就别开 ✓）"
                              # ★ 收益与代价必须**成对**交出去 ✗✗：只写 bitExact=False 人会读成「风险换了个
                              # 不知值不值的收益」✓；实测口径（4B-Q4_K_M／A5000／truth 见探针 ⑥）：
                              # 128–512 词元 ×1.24–1.27，而 512 词元里 443 个词元已与老路不同 ✓
                              f" · 实测 128–512 词元 ×1.24–1.27（512 词元里 443 个已与老路不同 ⇒ "
                              f"长生成要逐字复现就别开 ✓）")
        else:
            reason = str(last.get("reason") or "?")
            if auto and reason.startswith("autoTooShort"):
                # ⚠️ auto 的「没捕」**不是故障** ✗✗（2026-11-09 补）：它按长度算出来这一档开图净亏 ✓
                # ⇒ 若和「让路/前置不满足」并在一句话里，人下次会去查前置、查不到东西 ✓✗
                report["note"] = (f"上一次**没捕图** ✓：{reason} ⇒ auto 档按长度主动走老路 ✓"
                                  f"（这一档开图净亏 ~28 ms ✓ 不是故障、也不用调 ✓ "
                                  f"要强制捕请把档位改 `on` ✓）")
            else:
                report["note"] = (f"上一次**没在跑图** ✓：{reason} ✓"
                                  f"（是让路/前置不满足，不是静默关闭 ✓）")
        return report

    def _kv_window_call(self) -> dict[str, Any]:
        """生成调用要递下去的滑窗参数 ✓（**关着 ⇒ 空字典** ✓ ⇒ 调用签名与老路径逐字相同 ✓）。

        ⚠️ 为什么滑窗是**后端口径**而不是每次生成的参数 ✗✗：它会同时改「KV 缓冲该开多大」
        （:func:`~.llm.generate_llm` 里的 ``kv_rows`` 预算 ✓）⇒ 与 ``kvMode`` 同层 ✓；
        挂在实例上 ⇒ 三个生成口共用一份 ✓（``engine_llm_backend_test`` 的 ⑫⁸/⑫⁷ 盯着别漏口 ✓）。
        """
        if not self._kv_window:
            return {}
        return {"kv_window": self._kv_window, "kv_sink": self._kv_sink}

    def _prefill_chunk(self) -> int:
        """这一次 prefill 切多大块（词元）✓（显存紧 ⇒ 缩小 ✓；探不到／够用 ⇒ 1024 逐字不变 ✓）。

        ⚠️ 为什么块大小归**后端**而不是调用方 ✗✗：块大小只解决一件事 —— **峰值**
        （:func:`~.llm.generate_llm` 里分数矩阵是 ``O(chunk × 总长)`` ✓，与 KV 容量无关 ✓✗）。
        显存有多紧只有后端知道（它刚 ``refresh_kv_budget`` 过 ✓）；调用方拿不到这个数 ✓，
        会被迫猜或写死 ✓（写死 1024 的那天，2 GiB 空闲的卡上 prefill 3240 词元就炸 ✓✗）。

        只缩不涨 ✗（``≥ 4 GiB`` 就是 1024 ⇒ 常见环境**一个字节都不变** ✓）；涨没有实测收益、
        却把峰值翻倍 ✓✗。显存一紧就缩，代价是前向次数变多（慢一点）✓ 而不是 OOM（跑不了）✓✗。
        """
        free = (self._kv_budget or {}).get("freeBytes")
        if not isinstance(free, int):
            return PREFILL_CHUNK_DEFAULT
        gib = free / 2 ** 30
        for floor_gib, chunk in PREFILL_CHUNK_TIERS:
            if gib >= floor_gib:
                return chunk
        return PREFILL_CHUNK_TIERS[-1][1]

    def _prefill_call(self) -> dict[str, Any]:
        """生成调用要递下去的分块参数 ✓（**默认档 ⇒ 空字典** ✓ ⇒ 调用签名与老路径逐字相同 ✓）。"""
        chunk = self._prefill_chunk()
        return {} if chunk == PREFILL_CHUNK_DEFAULT else {"prefill_chunk": chunk}

    def _prefill_report(self) -> dict[str, Any]:
        """回执：这次 prefill 切多大块 ＋ **为什么是这一档** ✓（装载前也能出 ✓ 不抛 ✗）。"""
        chunk = self._prefill_chunk()
        free = (self._kv_budget or {}).get("freeBytes")
        if chunk == PREFILL_CHUNK_DEFAULT:
            note = (f"默认档 {PREFILL_CHUNK_DEFAULT} ✓（空闲 ≥4 GiB、或还没探到显存 ✓）"
                    f"—— 老路径逐字不变 ✓")
        else:
            note = (f"空闲只有 {free / 2 ** 30:.1f} GiB ⇒ 分块收到 {chunk} ✓ "
                    f"（峰值 O(chunk × 总长) ✓ 慢一点、但不 OOM ✓✗）")
        return {"chunk": chunk, "default": PREFILL_CHUNK_DEFAULT,
                "freeBytes": free, "note": note}

    def _kv_cache_for_call(self) -> Any:
        """这一次生成能用哪个前缀缓存 ✓（滑窗开着 ⇒ ``None`` ✗✗）。

        ⚠️ 两者**结构互斥** ✗：前缀缓存存的是**完整**历史 ✓，滑窗已经把中间那段删了 ✓ ⇒
        复用回来模型会看见「本不该记得」的行 ✓（数值错、不报错 ✓✗）。本仓选**滑窗优先** ✓：
        滑窗解决的是「根本装不下」，复用只是「重复前缀省时间」✓ —— 但**必须如实报** ✓
        （静默把 ``kvCache`` 停掉 ⇒ 人按「复用在跑」算时间会算错 ✓✗）。
        """
        return None if self._kv_window else self._kv_cache

    def clear_kv_cache(self, note: str = "调用方要求清空 ✓") -> dict[str, Any]:
        """清掉常驻的 KV 前缀 ✓（换提示词模板／要腾显存时调 ✓）⇒ 清完的回执 ✓。"""
        if self._kv_cache is not None:
            self._kv_cache.clear(note)
        return self._kv_cache_report()

    def _device_note(self) -> str:
        """设备/精度这一栏**人话说明** ✓（装载前 / 装载后各一段 ✓）。"""
        if self._model is not None and self._layers:
            gpu = int(self._layers.get("gpuLayers") or 0)
            cpu = int(self._layers.get("cpuLayers") or 0)
            return (f"分层放置 ✓：前 {gpu} 层在 {self._device}、后 {cpu} 层在 CPU，"
                    f"以 {self._dtype or '?'} 跑 ✓（跨界只搬一份激活 ✓ 不是每层搬权重 ✗）")
        if self._model is not None and self._device:
            return f"在 {self._device} 上以 {self._dtype or '?'} 跑 ✓（同一份前向 ✓）"
        if self._device_request.lower() == "auto":
            facts_devices = llm_mod.device_facts().get("devices", [])
            return (f"装载时择优 ✓（现在能用：{facts_devices} ✓ ⇒ "
                    f"有卡用卡·bf16 ✓、没卡用 CPU·fp32 ✓；显式写 dtype 可改 ✓）")
        return f"装载时解析 ✓（请求 {self._device_request} ✓ —— 用不了会**具名报错** ✗ 不回落 ✗）"

    # ── 分层放置 ✓（2026-11-08）
    def _gpu_layers_wanted(self) -> int | str | None:
        """``gpuLayers`` 请求 ⇒ ``None``（不分层）／整数（0..depth）／``"auto"`` ✓。

        ⚠️ 拼错**当场拒** ✗（``"atou"`` 悄悄当成分层或不分层，都会让人以为配生效了 ✓✗）。
        """
        raw = self._gpu_layers_request.strip().lower()
        if raw in ("", "none", "off", "false", "no"):
            return None
        if raw in ("auto", "true", "on", "yes"):
            return "auto"
        try:
            value = int(raw)
        except ValueError as err:
            raise llm_mod.LlmError(
                f"gpuLayers 不认 {self._gpu_layers_request!r} ✗（可写 off / 0 / N / auto ✓）") from err
        if value < 0:
            raise llm_mod.LlmError(f"gpuLayers 不能为负 ✗（收到 {value}）")
        return value

    def _layer_placement(self, device: str, dtype: Any) -> tuple[list[str] | None, dict[str, Any] | None]:
        """算**层放置** ✓ ⇒ ``(每层设备表 ／ None=不分层, 回执)``。

        口径（参考 Strata ``plan/plan.hpp``：**放置是算术结果，算不平就拒** ✓✗）：

        * 没配 ⇒ ``(None, None)`` ✓ —— 老路径（整模型一处 ✓）逐字节不变 ✓；
        * ``0`` ⇒ **全留 CPU** ✓（唯一的「一条都不上卡」表达 ✓；回执写明是**显式要求** ✓）；
        * ``N`` ⇒ 卡上放**前 N 层** ✓，放不下（``N × 层字节 > 可用``）⇒ **具名拒** ✗✗（不悄悄减配 ✗）；
        * ``auto`` ⇒ 按**此刻**可用显存（:func:`~.device_policy.probe` 的 ``freeBytes`` ✓ 含可复用缓存 ✓）
          铺满前 n 层 ✓；没有可用的卡 ⇒ 全 CPU 且回执写清**为什么** ✓（不是静默回落 ✗）。

        ⚠️ 预算只扣**层权重** ✓：``embed``/``norm``/``lm_head`` 被 :func:`~.llm.apply_layer_plan`
        钉在**末层**（分层时＝CPU ✓）⇒ 卡上占的就是前 n 层的权重 ✓，激活由 ``HEADROOM_BYTES`` 兜 ✓。
        """
        want = self._gpu_layers_wanted()
        if want is None:
            return None, None
        depth = int(self._config.depth)
        dtype_bytes = _DTYPE_BYTES.get(str(dtype).replace("torch.", ""))
        if dtype_bytes is None:
            raise llm_mod.LlmError(f"精度 {dtype!r} 的字节数不知道 ⇒ 算不了层预算 ✗")
        layer_bytes = llm_mod.layer_weights_bytes(self._config, dtype_bytes)
        base = {"layerParams": layer_bytes // dtype_bytes, "layerBytes": layer_bytes,
                "dtypeBytes": dtype_bytes, "depth": depth, "request": want,
                "gpuDevice": device, "cpuDevice": "cpu"}

        if want == 0:
            return (["cpu"] * depth,
                    {**base, "gpuLayers": 0, "cpuLayers": depth, "gpuBytes": 0,
                     "usableBytes": 0, "why": "调用方显式要求 0 层上卡 ⇒ 整份留 cpu ✓"})
        if want != "auto" and int(want) > depth:
            raise llm_mod.LlmError(
                f"gpuLayers={want} 超过模型层数 {depth} ✗（多出来的层不存在 ✓；"
                "要「尽量都上卡」请写 auto ✓）")

        pressure = device_policy.probe() if str(device).split(":")[0] == "cuda" else None
        if pressure is None:
            if want == "auto":
                return (["cpu"] * depth,
                        {**base, "gpuLayers": 0, "cpuLayers": depth, "gpuBytes": 0,
                         "usableBytes": 0,
                         "why": f"设备 {device} 上没有可用的 CUDA 卡 ⇒ 分层放置退化为全 CPU ✓"})
            raise llm_mod.LlmError(
                f"gpuLayers={want} 要求前 {want} 层上卡 ✗，但设备 {device} 上没有可用的 CUDA 卡 ⇒ 拒绝 ✓"
                "（不静默退回全 CPU ✗；要全 CPU 请显式写 gpuLayers=0 ✓）")

        free_bytes = int(pressure.freeBytes)
        reserve = int(device_policy.HEADROOM_BYTES)
        usable = max(0, free_bytes - reserve)
        if want == "auto":
            _devices, report = llm_mod.plan_layer_devices(
                self._config, gpu_bytes=free_bytes, dtype_bytes=dtype_bytes,
                gpu_device=device, reserve_bytes=reserve)
            report.update({key: base[key] for key in ("layerParams", "depth", "request")})
            report["freeBytes"] = free_bytes
            if not report["why"]:
                report["why"] = (f"此刻空闲 {free_bytes} 字节 − 预留 {reserve} ⇒ "
                                 f"前 {report['gpuLayers']} 层上卡 ✓")
            return _devices, report

        need = int(want) * layer_bytes
        if need > usable:
            raise llm_mod.LlmError(
                f"gpuLayers={want} 需要 {need} 字节，可用只有 {usable} 字节"
                f"（此刻空闲 {free_bytes} − 预留 {reserve}）⇒ 放不下 ✗"
                "（**不悄悄减配** ✗；改小 gpuLayers ✓、或写 auto 让它按余量算 ✓）")
        _devices, report = llm_mod.plan_layer_devices(
            self._config, gpu_bytes=free_bytes, dtype_bytes=dtype_bytes,
            gpu_device=device, reserve_bytes=reserve, want_gpu_layers=int(want))
        # ⚠️ ``plan_layer_devices`` 的 ``want_gpu_layers`` 是**下限**（不够时报 ``shortfall`` ✓），
        # 它会给「预算能装多少就装多少」⇒ 显式写 ``N`` 时必须**截到 N** ✓✗：不截就是「配了 1 层、
        # 实际 2 层全上卡」✓（配置的意图与落点不一致 ✓ 而且卡上白占一层 ✗✗）。
        capped = min(int(want), int(report["gpuLayers"]))
        if capped != int(report["gpuLayers"]):
            _devices = _devices[:capped] + ["cpu"] * (depth - capped)
            report.update({"gpuLayers": capped, "cpuLayers": depth - capped,
                           "gpuWeightBytes": capped * layer_bytes,
                           "cpuWeightBytes": (depth - capped) * layer_bytes,
                           "allOnGpu": capped == depth})
        if report["gpuLayers"] != int(want):                    # pragma: no cover —— 上面已拦 ✓
            raise llm_mod.LlmError(f"内部分层算错：要 {want} 层、算出 {report['gpuLayers']} 层 ✗")
        report.update({key: base[key] for key in ("layerParams", "depth")})
        report["request"] = int(want)
        report["freeBytes"] = free_bytes
        report["why"] = (f"显式要前 {want} 层上卡 ✓：{need} 字节 ≤ 可用 {usable} 字节"
                         f"（空闲 {free_bytes} − 预留 {reserve}）✓")
        return _devices, report

    # ── 装载 ✓
    def load(self) -> "LlmBackend":
        """装载 GGUF 权重 + 词表 ✓（⚠️ 重复调用**幂等** ✓ 只装一次 ✓）。

        GPU 口径 ✓：``device="auto"`` 有卡就上卡 ✓（显式写 ``cuda`` 用不了则**报错** ✗ 不回落 ✗）；
        ``dtype`` 交给 :func:`~.llm.resolve_dtype` ✓（CPU ⇒ fp32 ✓，CUDA ⇒ bf16 ✓ —— 8B 的 fp32
        是 32 GB ✗，24 GB 卡装不下 ✓）。

        分层放置 ✓（``gpu_layers``，2026-11-08 补）：见 :meth:`_layer_placement` ✓ —— 卡上前 N 层 ＋
        CPU 尾巴 ✓，跨界只搬一份激活 ✓。**装不下就拒** ✗，不悄悄退化成全 CPU ✗。

        KV cache 口径 ✓（``kv_mode``，2026-11-08 补）：``off``/``fp`` = 原样浮点（老路径 ✓）、
        ``q8`` = INT8 按 ``head_dim`` 每 64 元素分组量化 ✓。⚠️ **装载前**归一 ✗（拼错当场拒 ✓
        —— 装完 8 GiB 才报「kvMode 写错」等于白等一轮 ✓✗）。

        跨调用前缀复用 ✓（``kv_cache_bytes``，2026-11-08 补）：预算 > 0 就建一个
        :class:`~.llm.KvPrefixCache` ✓，交给三处 ``generate_llm`` 同用 ✓（``0`` = 关 ✓，
        不写 = 512 MiB ✓）。⚠️ 换权重/换口径都走本方法 ⇒ **重建 cache** ✓（绝不复用别份权重的 KV ✗✗）。
        """
        if self._model is not None and self._hub is not None:
            return self
        self._load_error = ""
        try:
            # ⚠️ KV 口径在**装载前**就归一 ✓（拼错要当场拒 ✗）：装完 8 GiB 权重才发现 ``kvMode``
            # 写错了 ⇒ 白等一轮 ✓（而且人只会看到「装完了但报错」✓ 以为权重有问题 ✗）。
            self._kv_mode = llm_mod.resolve_kv_mode(self._kv_mode_request)
            # ⚠️ 预算也在这里归一 ✓（同一个理由：拼错就在**花 30 秒装权重之前**拒掉 ✗✗）
            self._kv_cache_budget = llm_mod.resolve_kv_cache_bytes(self._kv_cache_request)
            # ⚠️ 投机旋钮也在这里收口 ✓（构造期归一失败的那三条，同样在**装权重之前**拒 ✗✗）
            if self._spec_error:
                raise LlmBackendUnavailable(self._spec_error)
            # ⚠️ 快路径档位同一个理由 ✓：拼错要在**装权重之前**具名拒 ✗✗（装完 8 GiB 才说
            # ``fastPath="nrom"`` ⇒ 人以为慢是权重问题 ✓✗）。
            if self._fast_error:
                raise LlmBackendUnavailable(self._fast_error)
            # ⚠️ 滑窗两个口同样在**装权重之前**具名拒 ✗✗（``kvWindow="3k"`` 静默当 0 ⇒ 人以为
            # 开了滑窗、显存却照涨 ✓✗）。
            if self._kv_win_error:
                raise LlmBackendUnavailable(self._kv_win_error)
            # ⚠️ 图化档位同一个理由 ✓：``cudaGraph="grpah"`` 静默当关 ⇒ 人以为在跑图 ✓ 其实每步还在
            # 发几千次 launch ✗✗ ⇒ **装权重之前**具名拒 ✓。
            if self._graph_error:
                raise LlmBackendUnavailable(self._graph_error)
            # ⚠️ ``kv_sink >= kv_window`` 是**必错**组合 ✗（sink 是窗口之外另留的开头几行 ⇒ 相等就没有
            # 可滚的窗口 ✓）—— 现在拒 ⇒ 不必等到第一次生成才炸 ✓（那时人已经在看输出 ✓）。
            if self._kv_window and self._kv_sink >= self._kv_window:
                raise LlmBackendUnavailable(
                    f"kvSink={self._kv_sink} 必须小于 kvWindow={self._kv_window} ✗："
                    f"sink 是窗口**之外**另留的开头几行 ✓，相等就没有可滚的窗口了 ✗")
            if self._model is None:
                if not self._gguf_path or not Path(self._gguf_path).exists():
                    raise LlmBackendUnavailable("GGUF 权重未就绪 ✗（describe 里已报 ✓）")
                device = llm_mod.resolve_device(self._device_request)
                dtype = llm_mod.resolve_dtype(self._dtype_request or None, device=device)
                layer_devices, layers = self._layer_placement(device, dtype)
                if layer_devices is None:
                    self._model = gguf_to_llm.load_llm_from_gguf(
                        self._gguf_path, self._config, device=device, dtype=dtype)
                else:
                    self._model = gguf_to_llm.load_llm_from_gguf(
                        self._gguf_path, self._config, dtype=dtype, layer_devices=layer_devices)
                self._layers = layers
                if layers is not None:
                    # ⚠️ 分层时 ``model_device`` 是**首张参数** ⇒ embed 被钉在末层（CPU ✓）⇒ 报出来
                    # 会是 "cpu" ✗✗（人一看「要了卡却在 cpu」）。分层口径只认**首层**在哪 ✓。
                    landed = list(getattr(self._model, "_layer_devices", None) or layer_devices)
                    self._device = str(landed[0])
                    first = next(iter(self._model.blocks[0].parameters()), None)
                    self._dtype = str(first.dtype).replace("torch.", "") if first is not None else ""
                else:
                    actual = llm_mod.model_device(self._model)
                    self._device = str(actual)
                    self._dtype = str(next(self._model.parameters()).dtype).replace("torch.", "")
                #: ⭐ 如实抬出来 ✓：命中/物化/放弃哪个、凭什么 ✓ —— 否则「这次快了 31 秒」
                #: 是缓存起了作用还是权重变小了，看回执的人分不出来 ✗✗（G3 的可见面 ✓）。
                self._materialized = dict(getattr(self._model, "materialized", None) or {}) or None
            if self._hub is None:
                source, note = self._tokenizer_source()
                if source is None:
                    raise LlmBackendUnavailable(f"词表未就绪 ✗（describe 里已报 ✓）{note}")
                self._hub = self._load_tokenizer(source)
        except Exception as err:  # noqa: BLE001 —— 装载失败要记下理由 ✗ 不半装 ✗
            self._model = None
            self._hub = None
            self._device = ""
            self._dtype = ""
            self._layers = None            # 失败 ⇒ 没有「这次的层放置」✓ 不留上一次的 ✗✗
            self._materialized = None      # 装载失败 ⇒ 没有「这次的缓存结论」✓ 不留上一次的 ✗✗
            self._kv_cache = None          # 装载失败 ⇒ 不留常驻 KV ✗✗（它占的是真显存 ✓）
            self._kv_budget = None         # 装载失败 ⇒ 没有「这块卡能放多少」✓ 不留上一次的 ✗✗
            self._load_error = str(err)
            raise
        self._read_chat_template()
        # ⚠️ 常驻 cache **每次装载重建** ✓：它只对**这份权重 + 这个口径**有效 ✓（换权重还留着＝拿旧
        # 模型的 K/V 喂新模型 ✗✗ —— shape 对得上、数值全错，且不报错 ✓✗）。
        self._kv_cache = (llm_mod.KvPrefixCache(self._kv_cache_budget, mode=self._kv_mode)
                          if self._kv_cache_budget > 0 else None)
        # ⭐ 这时候权重已经真的在卡上了 ⇒ 才是问「KV 还能放多少」的正确时刻 ✓（2026-11-08 补 ✓）。
        # ⚠️ 不许吞异常 ✗：算不出来要能从回执看出来 ✓（但也不许因此让装载失败 ✗ —— 预算只是守门 ✓）。
        try:
            self.refresh_kv_budget()
        except Exception as err:  # noqa: BLE001 —— 守门算不出 ⇒ 不拦装载 ✓ 但如实留痕 ✓
            self._kv_budget = {"ok": False, "maxTokens": None, "perToken": None,
                              "note": f"算不出显存侧 KV 上限 ✗（{type(err).__name__}: {err}）"
                                      f"⇒ 只按权重窗口守门 ✓"}
        return self

    # ── 生成 ✓
    def _spec_call(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """投机解码的 kwargs ＋ 回执槽 ✓（**三个调用口共用** ✗ 各抄一遍必漂移 ✗✗）。

        ⚠️ **关着也要送** ``spec_report`` ✓：``enabled=False`` 的**原因**（关／温度退回）本身就是回执 ✓，
        不送 ⇒ 回执永远空白 ✓ 人分不出「没开」与「开了没生效」✗✗。
        """
        sink: dict[str, Any] = {}
        return ({"spec_lookup": self._spec_lookup, "spec_window": self._spec_window,
                 "spec_min_match": self._spec_min_match, "spec_report": sink}, sink)

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 1.0,
                 max_new_tokens: int = 256, top_p: float = 1.0, top_k: int = 0,
                 seed: int | None = None) -> str:
        """进程内生成 ✓（⚠️ **同步** ✗ CPU 推理会阻塞 ✗ —— 调用方用 ``asyncio.to_thread`` 包 ✓）。

        ⚠️ ``seed`` 只影响随机采样 ✓；**跨设备逐字相同**要 ``temperature=0`` ✓（见 :func:`~.llm.generate_llm` ✓）。
        """
        if self._model is None or self._hub is None:
            raise LlmBackendUnavailable(self.describe()["reason"] or "模型/词表未装载 ✗（先 load() ✓）")

        import torch  # noqa: PLC0415

        # ⚠️ **必须 False** ✗✗（2026-10-31 修 ✓）：本文件抬头那条铁律（「生成侧若把 EOS 拼在提示词尾部会
        # 直接算错」）在 GGUF 词表上成立 ✓，但 HF 词表这条路原先传的是 ``True`` ✗ ⇒
        # `TokenizerBpe.encode(add_special_tokens=True)` 会在**尾部补 EOS** ✗✗（它同时补 BOS 和 EOS ✗）。
        # 提示词以 `<|im_start|>assistant\n` 结尾时再补一个 EOS ⇒ 模型被要求「文档已结束」⇒ 续写在 EOS 之后 ✓✗，
        # 输出量塌、跟随变弱 ✓✗，而且**不报任何错** ✓✗。
        # 参考口径与 ollama / HF 的 `apply_chat_template(...)` 后 `add_special_tokens=False` 一致 ✓；
        # 提示词里的 BOS/特殊符由 chat 模板自己产出 ✓（见 :meth:`_build_prompt_text` ✓）。
        ids = self._hub.encode(self._build_prompt_text(prompt, system), add_special_tokens=False)
        device = llm_mod.model_device(self._model)          # ⚠️ 张量必须跟模型**同设备** ✗ 否则 cuda 上直接崩 ✗✗
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        spec_kwargs, spec_sink = self._spec_call()
        fast_kwargs = self._fast_call()
        graph_kwargs, graph_sink = self._graph_call()    # 图化 decode 单步 ✓（默认关 ⇒ 空字典 ✓）
        out = llm_mod.generate_llm(
            self._model, ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, eos_id=self._config.eos_id, seed=seed,
            kv_mode=self._kv_mode, kv_cache=self._kv_cache_for_call(),
            **spec_kwargs, **fast_kwargs, **graph_kwargs,
            **self._kv_window_call(), **self._prefill_call(),
        )
        self._spec_last = spec_sink
        self._graph_last = graph_sink
        # ⚠️ 只能解**新生成**的那段 ✗✗（``generate_llm`` 返回的是 prompt＋续写 ✓）：整条解出去
        # ⇒ 调用方会看到自己刚发的那句对话（真机 qwen3:8b 实测：``OUT>>user\\n你好…\\nassistant\\n\\n好的…``✗）。
        # 用 **token 数**裁 ✓（不按文本找 ⇒ 不受空白/特殊 token 解码差异影响 ✓）。
        new_tokens = out[0][len(ids):].tolist()
        return self._hub.decode(new_tokens, skip_special_tokens=True).strip()

    # ── 多轮 + 工具声明 ✓（2026-11-07 补 ✓）
    def generate_messages(self, messages: Any, *, tools: Any = None, documents: Any = None,
                          extra: Any = None, temperature: float = 1.0,
                          max_new_tokens: int = 256, top_p: float = 1.0, top_k: int = 0,
                          seed: int | None = None) -> str:
        """**多轮消息 + 工具声明** ⇒ 续写文本 ✓（agent 工具循环的底层口 ✓）。

        与 :meth:`generate` 只差一件事 ✓：``generate`` 自己把 ``(prompt, system)`` 拼成
        ``system+user`` 两条消息 ✗ ⇒ 表达不了「assistant 调过工具 / tool 回了结果」这类历史 ✓✗。
        这里**原样收消息列表** ✓（含 ``role="tool"`` 与 ``assistant.tool_calls`` ✓），
        ``tools`` 按 OpenAI 形状透传给骨架 ✓（Qwen3/Hermes 系骨架自己会渲染成
        ``<tools>`` 声明 + 要求 ``<tool_call>`` 输出 ✓）。

        ⚠️ **权重不带 chat 模板 ⇒ 当场拒** ✗✗：没有骨架就没有「工具声明」这段，模型看见的
        只是几条裸文本 ✓✗ ⇒ 工具名/参数格式全靠猜 ✓✗，而调用方会以为自己在用原生 function
        calling ✗。**不静默回落裸拼接** ✓（与 :meth:`_build_prompt_text` 同一条纪律 ✓）。

        ⚠️ **超窗当场拒** ✗✗（:meth:`_guard_context` ✓）：本层宁可报错也不让采样在半路
        越界或由后端静默截断 ✓ —— 截断掉的往往是开头那段 system/工具声明 ✓✗，
        症状是「模型突然不按格式走」且**没有任何报错** ✓✗。
        """
        if self._model is None or self._hub is None:
            raise LlmBackendUnavailable(self.describe()["reason"] or "模型/词表未装载 ✗（先 load() ✓）")
        if self._chat_template_error:
            raise chat_template.ChatTemplateError(self._chat_template_error)
        if self._chat_template is None:
            raise chat_template.ChatTemplateError(
                "权重元数据里**没有 chat 模板** ✗ ⇒ 多轮消息没有骨架可渲染 ✓（本层不内置任何"
                "模型骨架 ✗、也不拼裸文本 ✗）—— 换一份自带模板的 GGUF ✓")

        import torch  # noqa: PLC0415

        extra_vars: dict[str, Any] = {}
        if self._enable_thinking is not None:
            extra_vars["enable_thinking"] = bool(self._enable_thinking)
        if extra:
            extra_vars.update(dict(extra))
        tokens = getattr(self._hub, "special_tokens", None) or {}
        text = self._chat_template.render(
            messages,
            bos_token=tokens.get("bos_token", ""), eos_token=tokens.get("eos_token", ""),
            tools=tools, documents=documents, extra=extra_vars)
        ids = self._hub.encode(text, add_special_tokens=False)
        self._guard_context(len(ids), max_new_tokens)
        device = llm_mod.model_device(self._model)
        spec_kwargs, spec_sink = self._spec_call()
        fast_kwargs = self._fast_call()
        graph_kwargs, graph_sink = self._graph_call()
        out = llm_mod.generate_llm(
            self._model, torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, eos_id=self._config.eos_id, seed=seed,
            kv_mode=self._kv_mode, kv_cache=self._kv_cache_for_call(),
            **spec_kwargs, **fast_kwargs, **graph_kwargs,
            **self._kv_window_call(), **self._prefill_call(),
        )
        self._spec_last = spec_sink
        self._graph_last = graph_sink
        return self._hub.decode(out[0][len(ids):].tolist(), skip_special_tokens=True).strip()

    def refresh_kv_budget(self) -> dict[str, Any]:
        """按**此刻**剩余显存重算「KV 还放得下多少 token」✓（2026-11-08 补 ✓）。

        为什么必须有这一本账 ✗✗：``_context_length`` 是**权重的**上限 ✓（Qwen3-4B 声明 32K ✓），
        而卡上真正能放多少由**剩余显存**决定 ✓ —— 8 GiB 卡装完 4B（反量化后 ~8 GiB 权重 ✓）
        剩不到 1 GiB ⇒ KV 连 8K 都放不下 ✓✗，可 `_guard_context` 老版只看权重窗口 ✓ ⇒
        **装得进、跑到第三千个 token 才炸** ✓，且报错落在 ``baddbmm`` 上、离真因几十步 ✓✗。

        ⚠️ 这是**快照** ✗（不是活值 ✓）：装载那一刻算一次 ✓；之后别的件上卡/退卡都会让它变 ✓
        ⇒ 换阶段（``enter_phase``）之后要**再调一次** ✓。本层不自动调 —— 每次生成都探卡要付一次
        全卡同步 ✓（decode 一步才 20 毫秒级 ✓，那笔开销不能白花 ✗）。
        """
        cfg = getattr(self._model, "config", self._config)
        free = None
        device = self._device or ""
        if device.startswith("cuda"):
            try:
                import torch  # noqa: PLC0415 —— 只有真装了才需要 ✓（describe 离线不许 import ✗）

                free, _total = torch.cuda.mem_get_info(device)
                free = int(free)
            except Exception:  # noqa: BLE001 —— 探不到 ⇒ 不判 ✓ 不编 0 ✗
                free = None
        budget = llm_mod.kv_room_tokens(cfg, self._kv_mode, free_bytes=free)
        budget["device"] = device or "cpu"
        #: 另一本账（**权重**声明窗口 ✓）也并进来 ✓ —— 调用方一眼能看到「先撞哪一本」✓
        budget["weightContext"] = int(self._context_length or 0)
        room = budget.get("maxTokens")
        if room is not None and budget["weightContext"]:
            budget["effectiveTokens"] = min(int(room), int(budget["weightContext"]))
            budget["limitedBy"] = "vram" if int(room) < int(budget["weightContext"]) else "weight"
        elif room is not None:
            budget["effectiveTokens"] = int(room)
            budget["limitedBy"] = "vram"
        else:
            budget["effectiveTokens"] = int(budget["weightContext"]) or None
            budget["limitedBy"] = "weight" if budget["weightContext"] else None
        self._kv_budget = budget
        return budget

    def _guard_context(self, prompt_tokens: int, max_new_tokens: int) -> None:
        """``prompt + 新生`` 超**两本账里较小的那本** ⇒ **拒** ✗（``0``/``None`` = 那本没结论 ⇒ 不拿它判 ✓）。

        两本账（2026-11-08 补 ✓）：

        * **权重窗口** ✓（``self._context_length`` ✓）—— 模型的先天上限 ✓，老口径不变 ✓；
        * **显存上限** ✓（``self._kv_budget["maxTokens"]`` ✓）—— 卡上真放得下的 token 数 ✓。

        ⚠️ 报错必须说清**是哪一本**卡住的 ✗✗：说「超窗」而其实是显存不够 ✓ 会让人去改提示词长度 ✓
        （改了也没用 ✓，因为权重窗口还有 32K ✓）。两条的处置办法完全不同 ✓（见报错正文 ✓）。
        """
        total = int(prompt_tokens) + int(max_new_tokens)
        weight_limit = int(self._context_length or 0)
        budget = self._kv_budget or {}
        room = budget.get("maxTokens")
        room = None if room is None else int(room)
        limit = weight_limit or 0
        if room is not None and (not limit or room < limit):
            limit = room
            by = "vram"
        elif limit:
            by = "weight"
        else:
            return
        if total <= limit:
            return
        if by == "vram":
            per = budget.get("perToken")
            raise llm_mod.LlmError(
                f"显存放不下这么长的上下文 ✗：提示词 {prompt_tokens} + 预留生成 {max_new_tokens} "
                f"= {total} 词元，而**这块卡**的 KV 只放得下 {limit} 词元 ✓（每 token {per} B、"
                f"可用 {budget.get('freeBytes')} B − 余量 {budget.get('reserveBytes')} B ✓）"
                f"⇒ **不静默截断** ✗ ⇒ 四条路：缩上下文 ✓／减 ``maxOutputTokens`` ✓／"
                f"``kvMode=q8``（每 token 字节减半 ✓）／先让别的件退卡（DiT 等 ✓）再 "
                f"``refresh_kv_budget()`` 重算 ✓")
        raise llm_mod.LlmError(
            f"上下文超窗 ✗：提示词 {prompt_tokens} + 预留生成 {max_new_tokens} = {total} 词元，"
            f"超过权重窗口 {limit} ✗ ⇒ **不静默截断** ✗（减 ``maxOutputTokens`` ✓、"
            f"或收短历史/工具声明 ✓）")

    # ── 流式生成 ✓（2026-11-01 补）
    def generate_stream(self, prompt: str, *, system: str | None = None,
                       temperature: float = 1.0, max_new_tokens: int = 256,
                       top_p: float = 1.0, top_k: int = 0, seed: int | None = None,
                       stop: Any = (), on_event: Any = None, abort: Any = None) -> str:
        """**同步**流式生成 ✓ —— 每出一段就 ``on_event("delta", {"text": ...})`` ✓，返回值是**整段**文本 ✓。

        ⚠️ 本方法**仍是同步阻塞**的 ✗（CPU 推理会占满线程 ✓）：调用方用 ``asyncio.to_thread`` 包住 ✓，
        回调里**只许做线程安全的事**（放到 ``asyncio.Queue`` 要经 ``loop.call_soon_threadsafe`` ✓）。

        三条口径（漏一条就会「看着在流、其实整段才出来」✓✗）：

        1. **增量按"解码全文前缀"算** ✗✗：BPE 的词边界上逐 token 解码会出半个字符（``\\ufffd`` ✓），
           所以每步解**累积 id** 的全文 ✓，只在「本轮全文以上一轮已发文本为前缀」时才切增量 ✓
           （不满足就这一轮不发 ✓、下一轮自然对齐 ✓ —— 宁可晚一字也不许吐半个字节 ✗）；
        2. **文字级 ``stop`` 是本层的事** ✗（``llm.generate_llm`` 的 ``stop_ids`` 只管整 token 哨兵 ✓）：
           停用词可能跨若干 token ✓ ⇒ 命中后**只把命中点之前的部分发出去** ✓，再用内部异常中止采样 ✓；
        3. **``on_event`` 为 ``None`` 时等价于「带 stop 裁剪的 generate」** ✓（便于无流式需求时复用 ✓）；
        4. **``abort``**（任意带 ``is_set()`` 的对象，通常是 :class:`threading.Event` ✓）：每一步先看一眼，
           已置位就**立刻跳出采样循环** ✓ 并返回**已生成的那半段** ✓（不再补发、不再解后续 token ✗）。
           这是「客户端断开 ⇒ 别再烧卡」的**唯一**有效通路 ✗✗：外层 ``task.cancel()`` 只能停掉
           ``asyncio`` 那边 ✓，线程里的采样循环照跑到底 ✗ ⇒ 由 :mod:`..text_stream` 在取消时置位 ✓。
        """
        if self._model is None or self._hub is None:
            raise LlmBackendUnavailable(self.describe()["reason"] or "模型/词表未装载 ✗（先 load() ✓）")

        import torch  # noqa: PLC0415

        stops = tuple(str(item) for item in (stop or ()) if str(item))
        ids = self._hub.encode(self._build_prompt_text(prompt, system), add_special_tokens=False)
        device = llm_mod.model_device(self._model)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        new_ids: list[int] = []
        sent = ""            # 已经发出去的前缀（含被 stop 截断后的那一段 ✓）

        def _emit(text: str) -> None:
            nonlocal sent
            if len(text) <= len(sent):
                return
            chunk = text[len(sent):]
            sent = text
            if on_event is not None:
                on_event("delta", {"text": chunk})

        def _on_token(token_id: int) -> None:
            # ⚠️ 中止判定必须在**一切都之前** ✗（append 之后再抛，那一个 token 已经进 new_ids ✓ 会被
            # 最后那段 decode 补发出去 ✗ —— 客户端都走了还多发字 ✓✗）
            if abort is not None and abort.is_set():
                raise _Aborted()
            new_ids.append(token_id)
            text = self._hub.decode(new_ids, skip_special_tokens=True)
            if not text.startswith(sent):
                return          # 边界不稳（半个 UTF-8 / 特殊 token 拼接）⇒ 本轮不发 ✓
            for mark in stops:
                at = text.find(mark)
                if at >= 0:
                    _emit(text[:at])
                    raise _StopHit(mark)
            _emit(text)

        spec_kwargs, spec_sink = self._spec_call()
        fast_kwargs = self._fast_call()
        graph_kwargs, graph_sink = self._graph_call()
        try:
            llm_mod.generate_llm(
                self._model, ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, eos_id=self._config.eos_id, seed=seed, on_token=_on_token,
                kv_mode=self._kv_mode, kv_cache=self._kv_cache_for_call(),
                **spec_kwargs, **fast_kwargs, **graph_kwargs,
                **self._kv_window_call(), **self._prefill_call(),
            )
        except _StopHit:
            pass
        except _Aborted:
            self._spec_last = spec_sink      # 中止也要留回执 ✓（否则「跑到一半被掐」看不出投机是否在跑 ✗）
            self._graph_last = graph_sink    # 图化同理 ✓（被掐在重放途中 ⇒ 重放了几步也该看得见 ✓）
            # 客户端已走 ⇒ **不许**再 ``_emit``（那份 emit 会经 call_soon_threadsafe 落到一个
            # 已经没人读的队列里 ✓）；直接返回已生成的那半段 ✓（调用方那边是取消路径，值会被丢弃 ✓）。
            return self._hub.decode(new_ids, skip_special_tokens=True).strip()
        self._spec_last = spec_sink
        self._graph_last = graph_sink
        full = self._hub.decode(new_ids, skip_special_tokens=True)
        for mark in stops:
            at = full.find(mark)
            if at >= 0:
                full = full[:at]
                break
        _emit(full)
        return full.strip()

    # ── 词表来源 ✓（2026-10-29 补 ✓）
    def _tokenizer_source(self) -> tuple[str | None, str]:
        """词表该从哪读 ✓ → ``(路径, 说明)``（``None`` = 没有 ✓ 说明写清为什么 ✗）。

        ⚠️ 真机撞出来的缺口 ✓✗：Ollama 库里的模型**只有 GGUF** ✗，没有 ``tokenizer.json`` ✗
        ⇒ 原来「``tokenizer_path`` 不在就报词表未就绪」把**词表就在权重里**的模型一律判死 ✗✗。
        词表本来就存在 GGUF 元数据里 ✓（``tokenizer.ggml.tokens`` ✓）⇒ 回落读权重文件 ✓。

        ⚠️⚠️ 第二次踩：**判 GGUF 不许看后缀** ✗✗ —— Ollama 的文件是 ``blobs\\sha256-<hash>`` ✓
        **没有后缀** ✗，按 ``.gguf`` 分派会一路走到 HF 词表装载、最后报「没有可识别的词表」✗
        （实测：权重装了两分钟才炸在词表这步 ✗）。⇒ 一律用 :func:`.gguf.looks_like_gguf` 看**正文** ✓。
        """
        if self._tokenizer_path and Path(self._tokenizer_path).exists():
            return self._tokenizer_path, ""
        if self._gguf_path and gguf.looks_like_gguf(self._gguf_path):
            note = "词表读自 GGUF 元数据 ✓"
            if self._tokenizer_path:
                note += f"（给的词表 ``{self._tokenizer_path}`` 不存在 ✗ ⇒ 回落权重 ✓）"
            return self._gguf_path, note
        if self._tokenizer_path:
            return None, f"``{self._tokenizer_path}`` 不存在 ✗"
        return None, "既没有词表文件、也没有 GGUF 权重 ✗"

    def _load_tokenizer(self, source: str) -> Any:
        """按来源分派 ✓：**GGUF 正文** → :mod:`.tokenizer_gguf` ✓；别的一律按 HF 词表 ✓。"""
        if gguf.looks_like_gguf(source):
            return tokenizer_gguf.load_gguf_tokenizer(source)
        return tokenizer_hub.load(tokenizer_hub.HubConfig(path=source))

    # ── 对话骨架 ✓（2026-09-25 补 ✓）
    def _read_chat_template(self) -> None:
        """读**权重自带的**对话骨架 ✓（只读头部 ✓ 毫秒级 ✓）。

        ⚠️ 模板出问题**不让装载失败** ✗（装载与骨架无关 ✓）—— 但**必须留痕** ✓：``describe`` 如实报 ✓，
        「存在但坏了」在 ``generate`` 时**当场拒** ✗（不静默回落裸拼接 ✗✗）。
        """
        self._chat_template = None
        self._chat_template_error = ""
        try:
            info = gguf.inspect(self._gguf_path or "")
            # 顺手取窗口长度 ✓（同一份 metadata 只读一次 ✓ —— 另开一次 inspect 要重读文件头 ✓✗）
            self._context_length = _context_length_of(info.metadata)
            self._chat_template = chat_template.ChatTemplate.from_metadata(info.metadata)
        except chat_template.ChatTemplateError as err:
            self._chat_template_error = str(err)
            self._chat_template_note = f"模板**存在但不可用** ✗：{err}"
            return
        except Exception as err:  # noqa: BLE001 —— 读不出元数据 ⇒ 按「没有模板」报 ✓ 不假装有骨架 ✗
            self._chat_template_note = (
                f"元数据读不出模板 ⇒ 按**没有模板**处理 ✓（{type(err).__name__}: {err}）")
            return
        self._chat_template_note = (
            f"用上权重自带的骨架 ✓（来源 ``{self._chat_template.origin}`` ✓）"
            if self._chat_template is not None
            else "权重元数据里没有 chat 模板 ✓ ⇒ 按**裸拼接**走 ✗（不是「用过骨架」✗）")

    def _build_prompt_text(self, prompt: str, system: str | None = None) -> str:
        """``(prompt, system)`` ⇒ 送进词表的文本 ✓（有骨架走骨架 ✓ / 没有走裸拼接 ✓ / 坏了 ⇒ 拒 ✗✗）。"""
        if self._chat_template_error:
            raise chat_template.ChatTemplateError(self._chat_template_error)
        if self._chat_template is not None:
            # ⚠️ ``enable_thinking`` **只在显式给定时**才定义 ✓（不给 ⇒ 模板用 ``is defined`` 自行判定 ✓，
            # 与 HF ``apply_chat_template`` 一个口径 ✓）—— 本层不替模板编默认值 ✗。
            extra = {} if self._enable_thinking is None else {"enable_thinking": bool(self._enable_thinking)}
            # ⚠️ ``bos_token``/``eos_token`` **必须真的给它** ✗✗（2026-10-31 修 ✓）：
            # ``render`` 的这两个参数默认 ``""`` ✓，不给 ⇒ 模板里 ``{{ bos_token }}`` 渲染成空串 ✗
            # （Llama-3 系骨架首行就是它 ✓；``{% if not bos_token %}`` 一类分支还会走错 ✓）⇒
            # 少了 BOS、与 llama.cpp/HF 的输出对不上 ✓✗，**且不报错** ✓。参考侧是自动注入 ✓。
            tokens = getattr(self._hub, "special_tokens", None) or {}
            return self._chat_template.render(
                chat_template.build_messages(prompt, system),
                bos_token=tokens.get("bos_token", ""),
                eos_token=tokens.get("eos_token", ""), extra=extra)
        return (system + "\n\n" + prompt) if system else prompt
