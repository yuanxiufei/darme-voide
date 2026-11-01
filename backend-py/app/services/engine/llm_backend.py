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
from . import gguf
from . import gguf_to_llm
from . import llm as llm_mod
from . import tokenizer_gguf
from . import tokenizer_hub

__all__ = ["LlmBackend", "LlmBackendUnavailable"]


class LlmBackendUnavailable(RuntimeError):
    """后端不可用 ✓（权重/词表没就绪、或没装载 ✓）—— ``reason`` 说明缺什么 ✓。"""


class LlmBackend:
    """自研文本后端 ✓（``describe`` / ``load`` / ``generate`` ✓）。"""

    name = "llm"
    #: 张量与推理都是**真的** ✓（不是 dryrun 的假张量 ✗）—— 但**没装真权重时 generate 会拒** ✗
    realTensors = True

    def __init__(self, config: llm_mod.LlmConfig, *, gguf_path: str | None = None,
                 tokenizer_path: str | None = None, device: Any = "auto",
                 dtype: Any = None, enable_thinking: bool | None = None) -> None:
        self._config = config
        self._gguf_path = gguf_path
        self._tokenizer_path = tokenizer_path
        #: ``None`` = 不定义该模板变量 ✓（由骨架自己的默认决定 ✓）；``False`` = 不要思考过程 ✓
        self._enable_thinking = enable_thinking
        # ⚠️ ``device``/``dtype`` 只在 **load() 那一刻**解析 ✓（构造函数不许碰 torch ✗：describe 要能离线跑 ✓）。
        self._device_request = "auto" if device is None else str(device)
        self._dtype_request = "" if dtype is None else str(dtype)
        self._device = ""          # 装载后**实际**设备 ✓（如 ``cuda:0`` ✓）
        self._dtype = ""           # 装载后**实际**精度 ✓（如 ``bfloat16`` ✓）
        self._model: Any = None
        self._hub: Any = None
        self._load_error: str = ""
        self._chat_template: chat_template.ChatTemplate | None = None
        self._chat_template_error: str = ""
        self._chat_template_note: str = "未读取 ✓（load() 之后才有结论 ✓）"

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
            "dtypeRequested": self._dtype_request or "auto",
            "dtype": self._dtype,
            "devices": devices.get("devices", []) if devices else [],
            "chatTemplate": {
                "present": self._chat_template is not None,
                "origin": self._chat_template.origin if self._chat_template is not None else "",
                #: ``None`` = 没管这个变量 ✓（骨架按自己的默认走 ✓，本层不替它编值 ✗）
                "enableThinking": self._enable_thinking,
                "note": self._chat_template_note,
            },
        }

    def _device_note(self) -> str:
        """设备/精度这一栏**人话说明** ✓（装载前 / 装载后各一段 ✓）。"""
        if self._model is not None and self._device:
            return f"在 {self._device} 上以 {self._dtype or '?'} 跑 ✓（同一份前向 ✓）"
        if self._device_request.lower() == "auto":
            facts_devices = llm_mod.device_facts().get("devices", [])
            return (f"装载时择优 ✓（现在能用：{facts_devices} ✓ ⇒ "
                    f"有卡用卡·bf16 ✓、没卡用 CPU·fp32 ✓；显式写 dtype 可改 ✓）")
        return f"装载时解析 ✓（请求 {self._device_request} ✓ —— 用不了会**具名报错** ✗ 不回落 ✗）"

    # ── 装载 ✓
    def load(self) -> "LlmBackend":
        """装载 GGUF 权重 + 词表 ✓（⚠️ 重复调用**幂等** ✓ 只装一次 ✓）。

        GPU 口径 ✓：``device="auto"`` 有卡就上卡 ✓（显式写 ``cuda`` 用不了则**报错** ✗ 不回落 ✗）；
        ``dtype`` 交给 :func:`~.llm.resolve_dtype` ✓（CPU ⇒ fp32 ✓，CUDA ⇒ bf16 ✓ —— 8B 的 fp32
        是 32 GB ✗，24 GB 卡装不下 ✓）。
        """
        if self._model is not None and self._hub is not None:
            return self
        self._load_error = ""
        try:
            if self._model is None:
                if not self._gguf_path or not Path(self._gguf_path).exists():
                    raise LlmBackendUnavailable("GGUF 权重未就绪 ✗（describe 里已报 ✓）")
                device = llm_mod.resolve_device(self._device_request)
                dtype = llm_mod.resolve_dtype(self._dtype_request or None, device=device)
                self._model = gguf_to_llm.load_llm_from_gguf(
                    self._gguf_path, self._config, device=device, dtype=dtype)
                actual = llm_mod.model_device(self._model)
                self._device = str(actual)
                self._dtype = str(next(self._model.parameters()).dtype).replace("torch.", "")
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
            self._load_error = str(err)
            raise
        self._read_chat_template()
        return self

    # ── 生成 ✓
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
        out = llm_mod.generate_llm(
            self._model, ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, eos_id=self._config.eos_id, seed=seed,
        )
        # ⚠️ 只能解**新生成**的那段 ✗✗（``generate_llm`` 返回的是 prompt＋续写 ✓）：整条解出去
        # ⇒ 调用方会看到自己刚发的那句对话（真机 qwen3:8b 实测：``OUT>>user\\n你好…\\nassistant\\n\\n好的…``✗）。
        # 用 **token 数**裁 ✓（不按文本找 ⇒ 不受空白/特殊 token 解码差异影响 ✓）。
        new_tokens = out[0][len(ids):].tolist()
        return self._hub.decode(new_tokens, skip_special_tokens=True).strip()

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
