"""自研文本后端（2026-09-25 起）—— **进程内** LLM 推理 ✓，不 HTTP 调 ollama ✗。

## 定位

:mod:`llm`（架构 ✓）+ :mod:`gguf_dequant`（反量化 ✓）+ :mod:`gguf_to_llm`（装载 ✓）三块拼图之后，
本模块把它们收成**一个后端** ✓（像 :class:`TorchBackend` 那样 ✓）：``describe`` 自述 + ``load`` 装载 +
``generate`` 生成 ✓ —— 文本生成由此能**绕过 ollama** ✓（「文本不依赖第三方」的最后一跳 ✓）。

## 能力自述（describe ✓）

权重 / 词表路径就绪与否、模型装载与否 ⇒ ``canGenerate`` ✓ —— ⚠️ **真权重没到就说 False** ✓ 不冒充 ✗
（与 :class:`TorchBackend` 同一条纪律 ✓）。

## 生成（generate ✓）

``prompt(+system) → tokenizer.encode → llm.generate_llm → tokenizer.decode`` ✓。
⚠️ **chat 模板未做** ✗（Qwen3 的 ``<|im_start|>`` 对话骨架 ✗ 后续 ✓ —— 现在是裸拼接 ✓）。

## 两条约束

1. **同步** ✓：``generate`` 是 CPU 推理 ✗ 会阻塞 ✗ —— 接线时用 ``asyncio.to_thread`` 包 ✓（见 text_generation ✓）；
2. **懒导入 torch** ✓（模块级不 import ✓）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import gguf_to_llm
from . import llm as llm_mod
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
                 tokenizer_path: str | None = None) -> None:
        self._config = config
        self._gguf_path = gguf_path
        self._tokenizer_path = tokenizer_path
        self._model: Any = None
        self._hub: Any = None
        self._load_error: str = ""

    # ── 自述 ✓
    def describe(self) -> dict[str, Any]:
        """能力自述 ✓（**只查路径/状态，不做张量操作** ✓）。"""
        missing: list[str] = []
        if not self._gguf_path or not Path(self._gguf_path).exists():
            missing.append("GGUF 权重未就绪")
        if not self._tokenizer_path or not Path(self._tokenizer_path).exists():
            missing.append("词表未就绪")
        if self._model is None:
            missing.append("模型未装载（load() 后即可用）")
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
            "loadError": self._load_error,
        }

    # ── 装载 ✓
    def load(self) -> "LlmBackend":
        """装载 GGUF 权重 + 词表 ✓（⚠️ 重复调用**幂等** ✓ 只装一次 ✓）。"""
        if self._model is not None and self._hub is not None:
            return self
        self._load_error = ""
        try:
            if self._model is None:
                if not self._gguf_path or not Path(self._gguf_path).exists():
                    raise LlmBackendUnavailable("GGUF 权重未就绪 ✗（describe 里已报 ✓）")
                self._model = gguf_to_llm.load_llm_from_gguf(self._gguf_path, self._config)
            if self._hub is None:
                if not self._tokenizer_path or not Path(self._tokenizer_path).exists():
                    raise LlmBackendUnavailable("词表未就绪 ✗（describe 里已报 ✓）")
                self._hub = tokenizer_hub.load(tokenizer_hub.HubConfig(path=self._tokenizer_path))
        except Exception as err:  # noqa: BLE001 —— 装载失败要记下理由 ✗ 不半装 ✗
            self._model = None
            self._hub = None
            self._load_error = str(err)
            raise
        return self

    # ── 生成 ✓
    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 1.0,
                 max_new_tokens: int = 256, top_p: float = 1.0, top_k: int = 0) -> str:
        """进程内生成 ✓（⚠️ **同步** ✗ CPU 推理会阻塞 ✗ —— 调用方用 ``asyncio.to_thread`` 包 ✓）。"""
        if self._model is None or self._hub is None:
            raise LlmBackendUnavailable(self.describe()["reason"] or "模型/词表未装载 ✗（先 load() ✓）")

        import torch  # noqa: PLC0415

        text = (system + "\n\n" + prompt) if system else prompt
        ids = self._hub.encode(text, add_special_tokens=True)
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        out = llm_mod.generate_llm(
            self._model, ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, eos_id=self._config.eos_id,
        )
        return self._hub.decode(out[0].tolist(), skip_special_tokens=True).strip()
