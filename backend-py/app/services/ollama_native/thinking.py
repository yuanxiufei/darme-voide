"""thinking 标签的**流式**解析器 —— 逐字移植 ``reference/ollama/thinking/parser.go``。

作用：把模型吐出的 ``<think>…</think>实际回答`` 流切成「思考」与「回答」两路，
且**必须能跨 chunk 工作** —— 标签可能被网线切成 ``<thi`` + ``nk>`` 两半，也可能
思考内容还没出现就先来了一堆空白。

为什么不用正则：正则要求整段可见，流式下每来一个 chunk 都要重扫全文，且无法
区分「这会成为 <think> 的前缀」和「这就是正文」。官方这个状态机的全部复杂度
都在处理这件事：遇到**可能是标签前缀**的尾巴时必须缓冲，不能提前吐给用户。

用法::

    parser = ThinkingParser()          # 默认 <think> / </think>
    for chunk in stream:
        thinking, answer = parser.add_content(chunk)
"""

from __future__ import annotations

from enum import IntEnum


class ThinkingState(IntEnum):
    """五个状态，与 Go 侧 ``thinkingState_*`` 一一对应（值也一致）。"""

    LOOKING_FOR_OPENING = 0
    """还在找开标签，且尚未见到任何非空白字符。"""
    THINKING_STARTED_EATING_WHITESPACE = 1
    """开标签已出现，但思考内容还没开始 —— 要吃掉标签后那段空白。"""
    THINKING = 2
    """已进入思考内容，尚未见到闭标签。"""
    THINKING_DONE_EATING_WHITESPACE = 3
    """闭标签已出现，要吃掉闭标签与正文之间的空白。"""
    THINKING_DONE = 4
    """思考已结束、正文已开始（此后全部算正文）。"""


DEFAULT_OPENING_TAG = "<think>"
DEFAULT_CLOSING_TAG = "</think>"


def _overlap(text: str, delim: str) -> int:
    """``text`` 的后缀与 ``delim`` 的前缀之间的**最长**重合长度。

    这是流式判断「这段尾巴是不是闭标签的开头」的唯一依据：
    返回 >0 表示 ``text`` 末尾有若干字符**可能**属于闭标签，必须缓冲。
    """
    limit = min(len(delim), len(text))
    for i in range(limit, 0, -1):
        if text.endswith(delim[:i]):
            return i
    return 0


class ThinkingParser:
    """有状态的 thinking 切分器（非线程安全，一次流一个实例）。

    :param opening_tag: 开标签，官方默认 ``<think>``（qwen3 / deepseek-r1 同款）
    :param closing_tag: 闭标签，官方默认 ``</think>``
    """

    def __init__(
        self,
        opening_tag: str = DEFAULT_OPENING_TAG,
        closing_tag: str = DEFAULT_CLOSING_TAG,
    ) -> None:
        if not opening_tag or not closing_tag:
            raise ValueError("opening_tag / closing_tag 不能为空")
        self.opening_tag = opening_tag
        self.closing_tag = closing_tag
        self.state = ThinkingState.LOOKING_FOR_OPENING
        self._acc = ""

    # ── 对外 ────────────────────────────────────────────────────────────
    def add_content(self, content: str) -> tuple[str, str]:
        """喂入一段原始输出，返回 ``(thinking, answer)``。

        两段都可能是空串。返回值里的内容**已经是确定的**，可以立刻给用户看；
        还不确定的尾巴留在内部缓冲里等下一个 chunk。
        """
        self._acc += content

        thinking_parts: list[str] = []
        remaining_parts: list[str] = []
        # 一次调用可能连穿多个状态（例如 ``<think>a</think>b`` 一段就完了），
        # 所以循环到状态机说「不能再推进」为止 —— 否则调用方要等下一次输入
        # 才拿得到已经确定的内容。
        while True:
            thinking, remaining, keep_looping = self._eat()
            thinking_parts.append(thinking)
            remaining_parts.append(remaining)
            if not keep_looping:
                break

        return "".join(thinking_parts), "".join(remaining_parts)

    def reset(self) -> None:
        """清空状态（同一实例要复用时调用）。"""
        self.state = ThinkingState.LOOKING_FOR_OPENING
        self._acc = ""

    # ── 状态机 ──────────────────────────────────────────────────────────
    def _eat(self) -> tuple[str, str, bool]:
        """推进一步。返回 ``(thinking, remaining, 是否要继续)``。"""
        state = self.state

        if state == ThinkingState.LOOKING_FOR_OPENING:
            trimmed = self._acc.lstrip()
            if trimmed.startswith(self.opening_tag):
                # 开标签之后可能还有更多思考内容，继续解析而不是这里就返回
                after = trimmed.split(self.opening_tag, 1)[1]
                after = after.lstrip()
                self._acc = after
                self.state = (
                    ThinkingState.THINKING_STARTED_EATING_WHITESPACE
                    if after == ""
                    else ThinkingState.THINKING
                )
                return "", "", True

            if self.opening_tag.startswith(trimmed):
                # 只看到开标签的一部分（或纯空白）⇒ 继续累积
                return "", "", False

            if trimmed == "":
                return "", "", False

            # 没有开标签但有内容 ⇒ 模型跳过了思考，后面全是正文。
            # ⚠️ 这里返回的是**未 trim** 的原文：既然没有标签，就不该吞掉正文的空白。
            untrimmed = self._acc
            self._acc = ""
            self.state = ThinkingState.THINKING_DONE
            return "", untrimmed, False

        if state == ThinkingState.THINKING_STARTED_EATING_WHITESPACE:
            trimmed = self._acc.lstrip()
            self._acc = ""
            if trimmed == "":
                return "", "", False
            self.state = ThinkingState.THINKING
            self._acc = trimmed
            return "", "", True

        if state == ThinkingState.THINKING:
            acc = self._acc
            if self.closing_tag in acc:
                thinking, remaining = acc.split(self.closing_tag, 1)
                remaining = remaining.lstrip()
                self._acc = ""
                self.state = (
                    ThinkingState.THINKING_DONE_EATING_WHITESPACE
                    if remaining == ""
                    else ThinkingState.THINKING_DONE
                )
                return thinking, remaining, False

            overlap_len = _overlap(acc, self.closing_tag)
            if overlap_len > 0:
                # 尾巴可能是闭标签的开头 ⇒ 只吐确定部分，尾巴留着等后续
                thinking = acc[: len(acc) - overlap_len]
                self._acc = acc[len(acc) - overlap_len :]
                return thinking, "", False

            # 全是思考内容，可以整段吐出
            self._acc = ""
            return acc, "", False

        if state == ThinkingState.THINKING_DONE_EATING_WHITESPACE:
            trimmed = self._acc.lstrip()
            self._acc = ""
            if trimmed != "":
                self.state = ThinkingState.THINKING_DONE
            return "", trimmed, False

        if state == ThinkingState.THINKING_DONE:
            acc = self._acc
            self._acc = ""
            return "", acc, False

        raise AssertionError(f"未知状态: {state!r}")


__all__ = [
    "DEFAULT_CLOSING_TAG",
    "DEFAULT_OPENING_TAG",
    "ThinkingParser",
    "ThinkingState",
]
