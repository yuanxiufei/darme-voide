"""**上下文预算与压缩**（移植 ``Mini-Agent`` 的上下文压缩思路 ✓ 零依赖 ✓）。

## 为什么需要它

本项目 agent 的主循环（``agent/runtime.py`` 的 ``default_generate`` ✓）把
**整份消息列表**每步重发一次 ✓（``messages`` 一直 append assistant + tool 往返 ✓）。
而 skill 注入体量上限是 **60000 字符**（``agent/skills.py`` ✓）、剧本切片也没有硬上限 ✓
⇒ **越跑越长的多步工具链会把上下文撑爆** ✗ —— 现在**没有任何东西**在管这件事 ✗✗。

Mini-Agent 的做法是「**估算 → 超限就摘要**」✓；本模块把那件事拆成**两层**，
其中**第一层完全不调 LLM** ✓（这点很关键 ✓：压缩本身不能变成又一次昂贵的失败 ✗）：

| 层 | 做什么 | 成本 | 会不会失真 ✗ |
|---|---|---|---|
① **机械层**（本模块 ``apply_plan`` ✓） | 老的 **tool 输出**改成一行占位 ✓（保留个头 ✓ + 说明全文在哪 ✓） | 零 ✓ | 不改语义 ✓（模型知道"这里被压过"✓） |
② **摘要层**（``summarize_prompt`` ✓） | 把一段对话交给 LLM 写成要点 ✓ | 一次调用 ✓ | 会 ✓ ⇒ **只做要点、且由调用方决定要不要做** ✓ |

## ⚠️ 本模块最要紧的一条：**tool 往返必须成对** ✗

OpenAI 兼容协议要求：带 ``tool_calls`` 的 assistant 消息 ✓
**必须**紧跟每一条 ``role=tool`` 的应答（``tool_call_id`` 一一对应 ✓）。
初学者的压缩器常常"把老消息删掉" ✗ ⇒ 于是要么把 assistant 删了留下孤儿 tool 消息 ✗、
要么把某条 tool 应答删了留下**没人回答的 tool_call** ✗ —— 两者都会让**下一次请求直接 400** ✗✗。

所以本模块的机械层**只替换 ``content``、从不删消息** ✓，并且把成对关系当作**不变量**校验 ✓
（``pairsIntact`` ✓ —— 自检 ⑦ 专门钉它 ✓）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Budget", "CompressionPlan", "apply_plan", "estimate_message_tokens",
           "estimate_messages_tokens", "estimate_tokens", "plan_compression",
           "summarize_prompt", "tool_pairs_intact"]

#: ⚠️ **启发式**估算，不是 tokenizer ✓（刻意**不引依赖** ✗ —— 为了估个数字装个库不值 ✓）。
#: 系数取"偏保守"（宁可早压 ✓ 不可爆 ✗）；中文 ≈ 0.95 token/字 ✓、ASCII ≈ 0.28（≈1/3.6）✓。
#: 误差量级 ±20% ✓ ⇒ 预算要按**保守侧**设 ✓（见 :class:`Budget` 的注释 ✓）。
TOKENS_PER_CJK = 0.95
TOKENS_PER_ASCII = 0.28

#: 被机械压缩后，tool 输出保留多少字符 ✓（够模型认出"这是什么"即可 ✓）
ELIDED_PREFIX_CHARS = 240


@dataclass(frozen=True)
class Budget:
    """压缩预算 ✓ —— 全部是**输入**侧的阈值 ✓（不含要预留的输出 ✓）。"""

    #: 输入 token 上限 ✓。默认 24000 ≈ 1.2e5 字符中文 ✓（留足输出空间 ✓）
    limit_tokens: int = 24_000
    #: 最近 N 条**永不动** ✓（当前这一步的现场，压了就答非所问 ✗）
    keep_recent: int = 6
    #: 开头必须保住几条 ✓（system + 首条 user ⇒ 任务本身 ✗ 不能丢）
    keep_head: int = 2
    #: 要触发**摘要层**时，最少要压缩掉多少 token 才值得多花一次 LLM 调用 ✓
    summarize_threshold_tokens: int = 2_000

    def to_dict(self) -> dict[str, Any]:
        return {"limitTokens": self.limit_tokens, "keepRecent": self.keep_recent,
                "keepHead": self.keep_head,
                "summarizeThresholdTokens": self.summarize_threshold_tokens}


@dataclass
class CompressionPlan:
    """压缩方案 ✓（``apply_plan`` 做机械层 ✓；摘要层由调用方决定 ✓）。"""

    overBudget: bool = False          # noqa: N815 —— 与 JSON 字段同名 ✓
    tokensBefore: float = 0.0         # noqa: N815
    tokensAfter: float = 0.0          # noqa: N815
    elide: list[int] = field(default_factory=list)          # 要替换 content 的消息下标 ✓
    summarize: list[int] = field(default_factory=list)      # 建议交给 LLM 摘要的下标 ✓
    needsSummarize: bool = False      # noqa: N815 —— 机械层**不够** ✓
    pairsIntact: bool = True          # noqa: N815 —— 不变量：tool 往返成对 ✓
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"overBudget": self.overBudget, "tokensBefore": round(self.tokensBefore, 1),
                "tokensAfter": round(self.tokensAfter, 1), "elide": self.elide,
                "summarize": self.summarize, "needsSummarize": self.needsSummarize,
                "pairsIntact": self.pairsIntact, "notes": self.notes}


def estimate_tokens(text: Any) -> float:
    """粗估 token ✓（CJK 与 ASCII 分开算 ✓ —— 纯按字符数会**严重低估中文** ✗）。"""
    if not isinstance(text, str) or not text:
        return 0.0
    cjk = 0
    other = 0
    for char in text:
        if "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" \
                or "\u3000" <= char <= "\u303f" or "\uff00" <= char <= "\uffef":
            cjk += 1
        else:
            other += 1
    return cjk * TOKENS_PER_CJK + other * TOKENS_PER_ASCII


def estimate_message_tokens(message: Any) -> float:
    """一条消息的 token 估算 ✓（``content`` 与 ``tool_calls`` 的参数都算 ✓）。"""
    if not isinstance(message, dict):
        return 0.0
    total = estimate_tokens(message.get("content"))
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                function = call.get("function") or {}
                total += estimate_tokens(str(function.get("name") or ""))
                total += estimate_tokens(str(function.get("arguments") or ""))
    return total


def estimate_messages_tokens(messages: Any) -> float:
    """整份消息列表的 token 估算 ✓。"""
    if not isinstance(messages, list):
        return 0.0
    return sum(estimate_message_tokens(item) for item in messages)


def tool_pairs_intact(messages: Any) -> bool:
    """不变量检查 ✓：**每一条 ``role=tool`` 都有对应的 tool_call** ✓ 且反之亦然 ✓。

    ⚠️ 这是**协议级**要求（OpenAI 兼容端点会直接 400 ✗）——
    压缩器只要"删消息"就很容易破坏它 ✗（见模块文档 ✓）。
    """
    if not isinstance(messages, list):
        return False
    pending: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        if role == "assistant":
            if pending:
                return False                 # 上一条 assistant 的 tool_call 还没人应答 ✗
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    pending.add(str(call["id"]))
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending:
                return False                 # 孤儿 tool 应答 ✗
            pending.discard(call_id)
    return not pending


def _is_elidable(message: Any) -> bool:
    """只有 **tool 输出**能被机械压缩 ✓（assistant 的推理/工具选择不能替 ✗）。"""
    return isinstance(message, dict) and message.get("role") == "tool"


def plan_compression(messages: Any, budget: Budget | None = None) -> CompressionPlan:
    """算一份压缩方案 ✓（**不改动输入** ✓ —— 纯函数 ✓）。

    顺序 ✓：① 全部 tool 输出的**正文**替成占位 ✓（从最老的开始 ✓）；
    ② 还不够 ⇒ 标 ``needsSummarize`` ✓ 并把「建议摘要的区间」与提示词一起给出 ✓
    （**要不要花这次调用，由调用方决定** ✓ —— 本模块不擅自调模型 ✗）。
    """
    settings = budget or Budget()
    plan = CompressionPlan()
    if not isinstance(messages, list) or not messages:
        plan.notes.append("没有消息 ⇒ 无需压缩 ✓")
        return plan

    plan.tokensBefore = estimate_messages_tokens(messages)
    plan.pairsIntact = tool_pairs_intact(messages)
    if not plan.pairsIntact:
        # ⚠️ 进来就已经坏了 ⇒ 先说清楚 ✓（这不是压缩造成的 ✗，但压缩后一定更坏 ✓）
        plan.notes.append("⚠️ 调用方给进来的消息**本身**就破坏了 tool 往返成对 ✗"
                          "—— 压缩前请先修上游 ✓")
    if plan.tokensBefore <= settings.limit_tokens:
        plan.tokensAfter = plan.tokensBefore
        plan.notes.append(f"未超预算（{plan.tokensBefore:.0f} ≤ {settings.limit_tokens} ✓）")
        return plan

    plan.overBudget = True
    head = max(0, min(settings.keep_head, len(messages)))
    tail_start = max(head, len(messages) - max(0, settings.keep_recent))

    # ① 机械层：老的 tool 输出 ⇒ 占位正文 ✓（**消息本身留着** ✓ ⇒ 成对关系不破 ✓）
    saved = 0.0
    for index in range(head, tail_start):
        message = messages[index]
        if not _is_elidable(message):
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= ELIDED_PREFIX_CHARS:
            continue
        plan.elide.append(index)
        saved += estimate_tokens(content) - estimate_tokens(content[:ELIDED_PREFIX_CHARS])

    plan.tokensAfter = plan.tokensBefore - saved

    # ② 摘要层：机械层不够（且值得多花一次调用）⇒ 交给调用方 ✓
    if plan.tokensAfter > settings.limit_tokens:
        remaining = plan.tokensAfter - settings.limit_tokens
        plan.needsSummarize = True          # 确实还超 ✓（只是**值不值得**另说 ✓ 见下面的注释 ✓）
        plan.summarize = [index for index in range(head, tail_start)
                          if not _is_elidable(messages[index])]
        if remaining < settings.summarize_threshold_tokens:
            plan.notes.append(
                f"机械层已压到 {plan.tokensAfter:.0f} ✓ 只超 {remaining:.0f} token ⇒ "
                f"**不值得**再花一次 LLM 调用摘要 ✓（建议微调 keep_recent 或提示词 ✓）")
        else:
            plan.notes.append(
                f"机械层后仍超 {remaining:.0f} token ✗ ⇒ 建议对 {len(plan.summarize)} 条"
                f"中间消息做**一次**摘要 ✓（提示词见 ``summarize_prompt`` ✓）")
    else:
        plan.needsSummarize = False
        plan.notes.append(f"机械层足够 ✓：{plan.tokensBefore:.0f} → {plan.tokensAfter:.0f} token ✓")

    if plan.elide:
        plan.notes.append(
            f"压缩了 {len(plan.elide)} 条**老的 tool 输出**（保留前 {ELIDED_PREFIX_CHARS} 字符 ✓）"
            f"—— 消息条数不变 ✓ ⇒ tool 往返仍成对 ✓")
    return plan


def apply_plan(messages: list[dict[str, Any]], plan: CompressionPlan) -> list[dict[str, Any]]:
    """执行**机械层** ✓ ⇒ 返回新列表 ✓（**消息条数不变** ✓ —— 只换 ``content`` ✓）。

    ⚠️ 刻意不提供"删除消息"的选项 ✗：协议要求 tool 往返成对 ✓，
    删一条就 400 ✓（见模块文档 ✓）。要真的变短 ⇒ 走**摘要**（替换一段 ✓ 由调用方做 ✓）。
    """
    if not isinstance(messages, list):
        return messages
    out: list[dict[str, Any]] = []
    wanted = set(plan.elide or ())
    for index, message in enumerate(messages):
        if index not in wanted or not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str):
            out.append(message)
            continue
        head = content[:ELIDED_PREFIX_CHARS]
        note = (f"\n…（此工具输出共 {len(content)} 字符，已被**上下文压缩**截断 ✓ "
                f"如需全文请重新调用该工具 ✓）")
        out.append({**message, "content": head + note})
    return out


def summarize_prompt(messages: Any) -> str:
    """把「建议摘要的区间」写成**给 LLM 的摘要指令** ✓（第二层 ✓ 由调用方执行 ✓）。

    要点 ✓（照 Mini-Agent 的思路 ✓）：**要求保结构**（谁调了什么工具、得出了什么结论 ✓）、
    **不许编**（宁缺勿造 ✗）、且**明说这是压缩** ✓（免得模型以为这些是用户原话 ✗）。
    """
    if not isinstance(messages, list):
        messages = []
    body = json.dumps([{"role": item.get("role"), "content": item.get("content"),
                        "toolName": (item.get("tool_calls") or [{}])[0].get("function", {}).get("name")
                        if item.get("tool_calls") else None}
                       for item in messages if isinstance(item, dict)],
                      ensure_ascii=False, indent=1)
    return (
        "把下面这段 Agent 执行记录压缩成**要点**，用于继续同一项任务 ✓。\n"
        "要求：\n"
        "1. **保结构**：按顺序保留「调用了哪个工具 / 关键参数 / 得到什么结论」✓；\n"
        "2. **不许编**：原文没写的信息一律不要补 ✗（宁缺勿造 ✓）；\n"
        "3. **保留未完成事项**：哪些事还没做完、卡在哪儿 ✓；\n"
        "4. 用中文、分条、不引用原文大段内容 ✓；\n"
        "5. 开头注明这是**中间过程摘要**（不是用户原话 ✓）。\n\n"
        f"执行记录：\n{body}"
    ) if messages else ""
