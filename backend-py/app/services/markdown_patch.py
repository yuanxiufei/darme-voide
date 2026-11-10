"""增量文本变更集（学自 lobehub ``canvas/src/project/changes.ts``）。

**为什么本仓需要它**：模型改一段剧本/技能文档，现在只有 ``save_script`` 那种**整篇重写**
（``content`` 全量覆盖）。整篇重写的两个真代价：

* 模型必须把没改的部分**逐字复述**一遍 ⇒ 3000 字里想改一句话，得重打 3000 字，
  于是它开始"顺手优化"别处（改写漂移），或者写超上下文被截断；
* 覆盖式写法**没法发现冲突**：模型以为原文是 A、实际是 B，它照样写"成功"，
  然后整段 B 被 A 的复述版悄悄抹掉。

参考实现的三条真机制（缺一条就退化成"字符串替换凑合"）：

1. **``oldText`` 必须恰好命中 1 次**，0 次与 ≥2 次都拒。0 次＝模型记错了原文；
   ≥2 次＝这段话在文件里不唯一，改哪一处是歧义 —— 让它补上下文，而不是我们猜。
2. **失败句子必须是模型能直接照做的**（点名第几条、命中几次、缺什么、怎么改），
   这条错误会作为工具结果回灌给模型，它读得懂才有第二次机会。
3. **整批原子生效**：任何一条不过，**一条都不落**（本模块把改动攒在内存里，
   ``ok=False`` 时返回原文）。半套改动落盘比不改更糟 —— 前面的改动已经生效、
   后面的没有，模型看到"失败"会重试整批，于是前面的改动被**再改一遍**。

⚠️ 与本仓 ``save_script`` 的分工：整篇重写仍保留（模型真要重写时用它），
``edit_script`` 走本模块 —— **两者都不许静默降级成对方**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TextEdit",
    "PatchResult",
    "apply_edits",
    "coerce_edits",
    "count_occurrences",
    "describe_edit",
    "render_result",
]


@dataclass(frozen=True)
class TextEdit:
    """一条增量改动（``old_text`` 在原文里**必须恰好出现一次**）。"""

    old_text: str
    new_text: str
    reason: str = ""

    @property
    def noop(self) -> bool:
        return self.old_text == self.new_text


@dataclass
class PatchResult:
    """一次变更集的结果（``ok=False`` ⇒ 原文**一个字节都没动**）。"""

    ok: bool = True
    text: str = ""
    applied: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: 模型要读的那句（``ok=False`` 时必有）
    error: str = ""
    #: 出错的是第几条（从 0 起，与模型送来的数组下标一致）
    failed_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "failedIndex": self.failed_index,
            "applied": list(self.applied),
            "warnings": list(self.warnings),
            "charsBefore": self.applied[0]["charsBefore"] if self.applied else None,
            "charsAfter": len(self.text),
        }


def count_occurrences(text: str, needle: str) -> int:
    """``needle`` 在 ``text`` 里出现几次（**重叠不算**，与参考 ``countOccurrences`` 同口径）。"""
    if not needle:
        # Python 的 ``"".count("")`` 是 ``len+1``；参考实现的循环同理会给出 len+1，
        # 但那种"数字很大"的消息对模型毫无用处 ⇒ 由调用方单独判空串并给专门的句子。
        return 0
    count = 0
    index = text.find(needle)
    while index != -1:
        count += 1
        index = text.find(needle, index + len(needle))
    return count


def coerce_edits(raw: Any) -> tuple[list[TextEdit], str]:
    """把模型送来的数组归一成 :class:`TextEdit`（返回 ``(edits, error)``，二选一）。

    兼容 ``oldText/newText``（参考实现的字段名，模型最可能照抄）与 ``old_text/new_text``
    （本仓蛇形习惯）两种写法；``reason`` 接受 ``reason/note/why``。
    """
    if raw is None:
        return [], ""
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return [], (f"edits 必须是数组，收到的是 {type(raw).__name__}。"
                    "写法：edits: [{\"oldText\": \"原文片段\", \"newText\": \"改后片段\"}]")
    edits: list[TextEdit] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            return [], f"edits[{index}] 必须是对象，收到的是 {type(item).__name__}。"
        old = _first(item, ("oldText", "old_text", "old"))
        new = _first(item, ("newText", "new_text", "new"))
        reason = _first(item, ("reason", "note", "why")) or ""
        if old is None or new is None:
            return [], (f"edits[{index}] 必须同时给 \"oldText\" 与 \"newText\" 两个字符串"
                        f"（收到 oldText={_shown(old)}、newText={_shown(new)}）。")
        if not isinstance(old, str) or not isinstance(new, str):
            return [], (f"edits[{index}] 的 oldText / newText 必须是字符串，"
                        f"收到的是 {type(old).__name__} / {type(new).__name__}。")
        edits.append(TextEdit(old_text=old, new_text=new, reason=str(reason)))
    return edits, ""


def apply_edits(text: str, edits: Iterable[TextEdit]) -> PatchResult:
    """按顺序把 ``edits`` 应用到 ``text``（**整批原子**，任一条不过就整批不落）。"""
    original = text if isinstance(text, str) else ""
    working = original
    edits = list(edits)
    applied: list[dict[str, Any]] = []
    warnings: list[str] = []
    chars_before = len(original)

    for index, edit in enumerate(edits):
        if not edit.old_text:
            return PatchResult(
                ok=False, text=original, failed_index=index,
                error=(f"edits[{index}] 的 oldText 是空字符串。"
                       "空片段会被当成\"在文件任意位置插入\"，本工具不做这种猜测 —— "
                       "请给一段**在原文里唯一出现**的现有文字作为定位点。"),
            )
        hits = count_occurrences(working, edit.old_text)
        if hits != 1:
            return PatchResult(
                ok=False, text=original, failed_index=index,
                error=_mismatch_sentence(index, hits, edit.old_text),
            )
        if edit.noop:
            warnings.append(f"edits[{index}] 的 oldText 与 newText 相同 ⇒ 这条不改变任何内容。")
        working = working.replace(edit.old_text, edit.new_text, 1)
        applied.append({
            "index": index,
            "reason": edit.reason,
            "noop": edit.noop,
            "oldChars": len(edit.old_text),
            "newChars": len(edit.new_text),
            "charsBefore": chars_before,
            "charsAfter": len(working),
        })

    if not edits:
        warnings.append("edits 是空数组 ⇒ 什么都没改。")
    return PatchResult(ok=True, text=working, applied=applied, warnings=warnings)


def _mismatch_sentence(index: int, hits: int, old_text: str) -> str:
    """``oldText`` 命中次数不对时的模型可读句子（参考实现同款三段：事实 / 原因 / 怎么改）。"""
    head = f"edits[{index}] 的 oldText 在原文里出现 {hits} 次，必须恰好 1 次。"
    snippet = _one_line(old_text)
    if hits == 0:
        return (f"{head}（找不到这段：\"{snippet}\"）"
                "说明你记的原文与文件不一致 —— 先用 read 工具重新读一遍，"
                "把你**看到的那几行原样**贴进 oldText（含标点与换行）。")
    return (f"{head}（这段太短或太常见：\"{snippet}\"）"
            "请把 oldText 扩到**能唯一定位**它的范围：多带上一两行上下文，"
            "或者带上它前面那一行的场景标题。")


def describe_edit(edit: TextEdit, *, limit: int = 80) -> str:
    """一条改动的人话（日志/回执用）。"""
    mark = "（无变化）" if edit.noop else ""
    why = f" —— {edit.reason}" if edit.reason else ""
    return f"\"{_one_line(edit.old_text, limit)}\" → \"{_one_line(edit.new_text, limit)}\"{mark}{why}"


def render_result(result: PatchResult) -> str:
    """把结果渲染成一句人话（工具回执/日志共用）。"""
    if not result.ok:
        return f"未改动任何内容：{result.error}"
    if not result.applied:
        return "未改动任何内容（edits 为空）。"
    before = result.applied[0]["charsBefore"]
    after = len(result.text)
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"已应用 {len(result.applied)} 处改动（{before} → {after} 字，{sign}{delta}）。"


def _first(item: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _shown(value: Any, limit: int = 40) -> str:
    if value is None:
        return "缺失"
    text = _one_line(str(value), limit)
    return f"\"{text}\""


def _one_line(value: str, limit: int = 80) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"
