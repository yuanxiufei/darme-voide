"""**对话骨架（chat 模板）** ✓ —— 从**权重元数据**取模板、渲染成对话骨架 ✓（2026-09-25 补 ✓）。

## 为什么要有这一层
``llm_backend.generate`` 此前是**裸拼接**（``system + "\\n\\n" + prompt``）✗：对「骨架决定格式」的模型
（Qwen3 的 ``<|im_start|>…<|im_end|>`` ✓），提示词**没有落进对话骨架** ✓✗ ⇒ 模型更可能**续写**而不是
**执行** ✗（画面不对，还查不出来 ✓）。

⚠️ 但**绝不能内置一个「看起来对」的骨架** ✗✗ —— 骨架是**权重自带的** ✓：GGUF 元数据里的
``tokenizer.chat_template`` ✓（llama.cpp 与 HF 落的是同一处口径 ✓）。本层只做三件事：
**取出来** ✓、**渲染** ✓、**坏了就拒** ✗（不猜 ✗、不翻译 ✗）。

## 三条判据（错了都是「格式悄悄不对」✓✗）
1. ⭐⭐ **「没有模板」与「模板坏了」是两回事** ✗✗：**没有**（键不在 / 值是空串）⇒ ``None`` ✓
   （调用方**如实报告**、按裸拼接走 ✓）；**坏了**（Jinja 编译不过 / 渲染时引用未定义变量）⇒ **抛**
   :class:`ChatTemplateError` ✓✗（**不静默回落裸拼接** ✗ —— 回落 = 用户以为用了骨架 ✓✗）；
2. ⭐⭐ **产出逐字可核** ✓：骨架**完全由模板决定** ⇒ 同一份模板 + 同一批消息必须**逐字**再现 ✓
   （含 ``add_generation_prompt`` 带出的 assistant 前缀 ✓）；
3. ⭐ **模板没要的东西一律不塞** ✗：``bos_token``/``eos_token`` 由调用方给 ✓（不给就是空串 ✓，
   不替模型猜 ✓）；``raise_exception`` 由本层注入 ✓（HF 生态的模板会用它**主动拒绝**非法角色 ✓）；
   其余未定义变量 ⇒ **报错** ✓✗。

## 边界（不做什么 ✗）
* ⚠️ **不做模板「翻译」** ✗：不把 Jinja 模板改写成 Python 拼接（改写必然漏特例 ✓✗）；
* ⚠️ **不注入 ``strftime_now``** ✗（部分模板拿它做时间戳）：注入 ⇒ 同一提示词两次渲染**不同** ✓✗
   ⇒ 用了就**如实报错** ✓✗；
* ⚠️ **不碰词表** ✗：BOS/EOS 加不加由词表层定 ✓（见 ``tokenizer_hub`` ✓）—— 本层只产出**文本** ✓。
"""
from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CHAT_TEMPLATE_KEYS",
    "ChatTemplate",
    "ChatTemplateError",
    "build_messages",
]

#: 元数据里的模板键 ✓（GGUF/llama.cpp 的 ``tokenizer.chat_template`` ✓；裸键是历史写法 ✓）。
CHAT_TEMPLATE_KEYS: tuple[str, ...] = ("tokenizer.chat_template", "chat_template")


class ChatTemplateError(ValueError):
    """模板**存在但不可用** ✗（编译不过 / 渲染失败）⇒ 宁可当场拒 ✗。

    ⚠️ 静默回落裸拼接 = 格式**悄悄**不对 ✓✗ —— 那正是本仓最忌讳的「看着接了、其实没接」✗。
    """


@functools.lru_cache(maxsize=8)
def _compile(source: str) -> Any:
    """模板原文 ⇒ Jinja 编译产物 ✓（**同一份模板只编译一次** ✓；编译不过 ⇒ 具名拒 ✗）。

    ⚠️ ``trim_blocks``/``lstrip_blocks`` 与 ``transformers`` 的 ``apply_chat_template`` **同口径** ✓✗：
    模板作者是按那套语义写的 ✓，换语义 ⇒ 空白/换行**悄悄变** ✓✗。
    """
    from jinja2 import StrictUndefined  # noqa: PLC0415 —— 懒导入 ✓
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415

    def _raise_exception(message: Any) -> Any:
        raise ChatTemplateError(f"模板**主动拒绝**（``raise_exception`` ✓）：{message}")

    # ⚠️ 必须 ``StrictUndefined`` ✓✗：模板引用了本层**没给**的变量 ⇒ **当场报错** ✗；默认的 ``Undefined``
    #    会**静默渲染成空串** ✓✗ —— 那正是「格式悄悄不对」✗（判据 3 ✓）。
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                undefined=StrictUndefined)
    environment.globals["raise_exception"] = _raise_exception
    try:
        return environment.from_string(source)
    except Exception as err:  # noqa: BLE001 —— jinja2 各色 TemplateError 一律收成具名错 ✓
        raise ChatTemplateError(
            f"chat 模板**编译**不过 ✗：{type(err).__name__}: {err}"
            f" ⇒ 模板在权重元数据里**存在但不可用** ✗✗（**不是**「没有模板」✓ ⇒ 不静默回落裸拼接 ✗）"
        ) from err


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    """外部消息 ⇒ 模板能吃的形状 ✓（⚠️ 形状不对 ⇒ **拒** ✗ —— 不替调用方猜角色 ✓）。"""
    if messages is None:
        return []
    if isinstance(messages, (str, bytes, Mapping)):
        raise ChatTemplateError(
            f"messages 要是**消息列表** ✗（收到 {type(messages).__name__} ✓）——"
            f" 单条消息请写成 ``[{{'role': …, 'content': …}}]`` ✓"
        )
    rows: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise ChatTemplateError(f"每条消息要是字典 ✗（收到 {type(item).__name__} ✓）")
        role = str(item.get("role") or "").strip()
        if not role:
            raise ChatTemplateError(
                f"消息缺 ``role`` ✗：{dict(item)!r} ⇒ 本层**不替调用方猜**角色 ✓✗"
            )
        rows.append({"role": role, "content": str(item.get("content") or "")})
    return rows


def build_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    """``(prompt, system)`` ⇒ 消息列表 ✓。

    ⚠️ **system 没给就不编造** ✗✗（塞一条空 system ⇒ 骨架里凭空多一段 ``system`` ✓✗ ⇒ 与参考实现的
    产出**不一致** ✓✗）。
    """
    rows: list[dict[str, str]] = []
    if system is not None and str(system).strip():
        rows.append({"role": "system", "content": str(system)})
    rows.append({"role": "user", "content": str(prompt or "")})
    return rows


@dataclass(frozen=True)
class ChatTemplate:
    """**权重自带的**对话骨架 ✓（只承载「模板原文 + 来源」✓ —— 本层**不内置任何模型骨架** ✗✗）。"""

    source: str
    origin: str = "tokenizer.chat_template"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None, *,
                      keys: Sequence[str] = CHAT_TEMPLATE_KEYS) -> "ChatTemplate | None":
        """元数据 ⇒ 模板 ✓（**没有 ⇒ ``None``** ✓；**有但坏 ⇒ 抛** ✗✗）。

        ⚠️ 两种「取不到」必须分开 ✗✗：**键不在**（或值为空白串）⇒ ``None`` ✓（如实报告「没有」✓）；
        **键在但编译不过** ⇒ 抛 :class:`ChatTemplateError` ✓（那是「坏了」✗，不是「没有」✓）。
        """
        source, origin = "", ""
        for key in keys:
            value = (metadata or {}).get(key)
            if isinstance(value, str) and value.strip():
                source, origin = value, key
                break
        if not source:
            return None
        _compile(source)          # ⚠️ **读取时就编译** ✓：语法错当场暴露 ✗（别拖到出片那一刻 ✓✗）
        return cls(source=source, origin=origin)

    def render(self, messages: Iterable[Mapping[str, Any]] | None, *,
               add_generation_prompt: bool = True, bos_token: str = "",
               eos_token: str = "") -> str:
        """消息列表 ⇒ 骨架文本 ✓（⚠️ 渲染失败**抛** ✗ —— **不静默回落裸拼接** ✗✗）。"""
        rows = _normalize_messages(messages)
        if not rows:
            raise ChatTemplateError("messages 是空的 ✗ ⇒ 没有东西可渲染 ✓（给 system/user 至少一条 ✓）")
        template = _compile(self.source)
        try:
            return str(template.render(
                messages=rows,
                add_generation_prompt=bool(add_generation_prompt),
                bos_token=str(bos_token or ""),
                eos_token=str(eos_token or ""),
            ))
        except ChatTemplateError:
            raise
        except Exception as err:  # noqa: BLE001
            raise ChatTemplateError(
                f"chat 模板**渲染**失败 ✗：{type(err).__name__}: {err}"
                f" ⇒ **不静默回落裸拼接** ✗（模板来源 ``{self.origin}`` ✓ ——"
                f" 模板引用了本层没给的变量？要么显式给、要么换模板 ✓）"
            ) from err
