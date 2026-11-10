"""工具调用重复守卫（学自 lobehub ``agent-runtime/src/utils/toolCallRepeatGuard.ts``）。

**为什么需要**：Agent 打转（同一个工具、同一份参数反复调）是本仓最贵的失败形态 ——
出图/出视频一次几分钟、占满 22 GiB 卡，模型卡在循环里既烧时间也烧显存，而运行时
**从不叫停**：``max_steps`` 只限轮数，模型在**一轮之内**可以把同一个工具调 N 次。

参考实现三道闸门（本模块逐条对拍）：

1. **归一化签名** ``[toolName, normalizedArguments]``：``canonicalizeValue`` 递归按
   **键名排序** 归一对象（数组**保序**），字符串参数先 ``JSON.parse`` 再归一，解析不了
   就退原文 ``trim``。没有这一步，``{"a":1,"b":2}`` 与 ``{"b":2,"a":1}`` 会被当成两个
   不同调用 ⇒ 打转永远数不满 ⇒ 守卫是死的。
2. **重复闸门**：同一签名累计到 ``repeat_limit``（参考值 20）⇒ 叫停该轮。
3. **爆发闸门**：一轮里工具调用条数 ``> burst_limit``（参考值 50）⇒ **一条都不跑**、
   直接叫停（模型在刷格式，跑下去只是把 50 个副作用做完再说）。

⚠️⚠️ **参考实现的注释与代码不是一回事，本模块照代码**：注释写的是
"consecutive identical calls"，实现却是 ``counts[signature] = (previous ?? 0) + 1`` ——
**累计计数、中间穿插别的调用不清零**。真跑里最常见的打转恰恰是「调 A、调 B、再调 A」
这种非连续形态，照注释实现会漏掉它。本模块按代码语义（累计）实现，并在
:meth:`RepeatGuard.receipt` 里如实标注 ``counting="cumulative"``。

为什么是 20 而不是 3：合法的固定参数轮询（MCP 的 job status、本仓的长任务进度）本来
就要重复调用到完成为止；阈值太低会误杀正常等待。20 次的代价是一次 run 内浪费 20 步，
足以让 ``max_steps`` 先耗尽 —— 所以真正的作用是**把"静默耗尽"变成"显式叫停"**。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "REPEAT_LIMIT",
    "BURST_LIMIT",
    "GuardHit",
    "RepeatGuard",
    "canonicalize_value",
    "normalize_arguments",
    "tool_signature",
]

#: 同一签名累计多少次叫停（参考实现 ``TOOL_CALL_REPEAT_LIMIT``）
REPEAT_LIMIT = 20
#: 一轮里工具调用条数上限（参考实现 ``TOOL_CALL_BURST_LIMIT``）
BURST_LIMIT = 50

_SIGNATURE_SEP = "\x00"


def canonicalize_value(value: Any) -> Any:
    """递归归一：对象**按键名排序**、数组**保序**（与参考 ``canonicalizeValue`` 同口径）。

    数组保序是刻意的：``[1,2]`` 与 ``[2,1]`` 对工具来说**可能**是两次不同的调用
    （``ids`` 顺序决定处理顺序），不能像对象那样排序。
    """
    if isinstance(value, Mapping):
        return {str(key): canonicalize_value(item) for key, item in sorted(value.items(),
                                                                         key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize_value(item) for item in value]
    return value


def _dumps(value: Any) -> str:
    """紧凑 JSON（中文不转义、分隔符无空格 ⇒ 同一份参数恒得同一串）。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def normalize_arguments(raw: Any) -> str:
    """工具入参归一成**可比较的字符串**（参考 ``normalizeArguments``）。

    * ``None`` / 空串 ⇒ ``""``（无参工具的签名就是 ``name\\0``）；
    * 字符串 ⇒ 先试 ``json.loads`` 再归一（**这才是关键**：OpenAI 那条回的是 JSON 字符串，
      两份语义相同、键序不同的参数串不归一就会被数成两次不同调用）；解析不了 ⇒
      ``strip()`` 原文（模型给了半截 JSON 也是"同一个坏参数"）；
    * 其他 ⇒ 归一后 ``_dumps``。
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ""
        try:
            return _dumps(canonicalize_value(json.loads(text)))
        except (ValueError, TypeError):
            return text
    return _dumps(canonicalize_value(raw))


def tool_signature(name: Any, arguments: Any) -> str:
    """一次调用的签名（参考 ``[identifier, apiName, normalizedArguments].join('\\0')``）。

    本仓没有 identifier/apiName 之分（工具名即函数名），故只留 ``name\\0args``；
    ``\\0`` 作分隔符而不是 ``|``：工具名与参数串里都可能出现 ``|``。
    """
    return _SIGNATURE_SEP.join([str(name or ""), normalize_arguments(arguments)])


@dataclass(frozen=True)
class GuardHit:
    """一次叫停（``kind`` 分 ``repeat`` / ``burst``）。"""

    kind: str
    tool: str
    count: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "tool": self.tool, "count": self.count,
                "message": self.message}


@dataclass
class RepeatGuard:
    """一次 run 内的重复/爆发守卫（**按 run 新建**，不跨 run 共享）。

    ``repeat_limit`` / ``burst_limit`` 可以按跑覆盖（``options["toolGuard"]``），
    但**不允许关掉**（``None``/``0``/负数一律回落到默认值）—— 关掉守卫就等于回到
    "静默耗尽 max_steps"，那不是配置项，是事故。
    """

    repeat_limit: int = REPEAT_LIMIT
    burst_limit: int = BURST_LIMIT
    counts: dict[str, int] = field(default_factory=dict)
    hits: list[GuardHit] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.repeat_limit = _positive(self.repeat_limit, REPEAT_LIMIT)
        self.burst_limit = _positive(self.burst_limit, BURST_LIMIT)

    # -- 闸门 -------------------------------------------------------------
    def check_burst(self, call_count: int) -> GuardHit | None:
        """一轮的工具调用条数闸门（**先判**，命中则整批一条都不跑）。"""
        if call_count > self.burst_limit:
            hit = GuardHit(
                kind="burst",
                tool="",
                count=call_count,
                message=(
                    f"本轮一次要调 {call_count} 个工具，超过单轮上限 {self.burst_limit}，"
                    f"整批未执行。请把这一轮拆成几轮、每轮只调真正必要的那几个。"
                ),
            )
            self.hits.append(hit)
            return hit
        return None

    def observe(self, name: Any, arguments: Any) -> GuardHit | None:
        """记一次调用并判重复闸门（累计计数，见模块文档的 ⚠️ 说明）。"""
        signature = tool_signature(name, arguments)
        count = self.counts.get(signature, 0) + 1
        self.counts[signature] = count
        if count < self.repeat_limit:
            return None
        hit = GuardHit(
            kind="repeat",
            tool=str(name or ""),
            count=count,
            message=(
                f"工具 `{name}` 用同一份参数已经调了 {count} 次（上限 {self.repeat_limit}），"
                f"本轮就此停下。上一轮的结果就在对话里：参数一样时再调一次结果不会变 ——"
                f"换成别的工具、或者改参数，别再重试同一个调用。"
            ),
        )
        self.hits.append(hit)
        return hit

    # -- 回执 -------------------------------------------------------------
    def receipt(self) -> dict[str, Any]:
        """守卫这一跑的账（进 ``steps``，评测与排障都要读得到）。"""
        return {
            "counting": "cumulative",
            "repeatLimit": self.repeat_limit,
            "burstLimit": self.burst_limit,
            "signatures": len(self.counts),
            "calls": sum(self.counts.values()),
            "hits": [hit.as_dict() for hit in self.hits],
        }


def _positive(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def guard_from_options(raw: Any) -> RepeatGuard:
    """从 ``options["toolGuard"]`` 建守卫（**形状容错、绝不关闸**）。

    允许 ``None`` / ``{"repeatLimit": 8}`` / ``{"repeat_limit": 8}`` 三种写法；
    显式 ``{"enabled": False}`` **不认**（见 :class:`RepeatGuard` 文档）。
    """
    if not isinstance(raw, Mapping):
        return RepeatGuard()
    return RepeatGuard(
        repeat_limit=raw.get("repeatLimit", raw.get("repeat_limit", REPEAT_LIMIT)),
        burst_limit=raw.get("burstLimit", raw.get("burst_limit", BURST_LIMIT)),
    )


def render_tool_names(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    """从一批原始调用里取工具名（只给日志/回执用，不参与判定）。"""
    names: list[str] = []
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            names.append(str(function.get("name") or ""))
    return names
