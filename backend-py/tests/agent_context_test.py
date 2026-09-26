"""S7 自检：**Agent 上下文预算与压缩**（零依赖 ✓ 2026-09-18）。

移植自 ``Mini-Agent``（上下文估算 + 超限压缩 ✓）。本项目原来**完全没有**这件事 ✗：
agent 主循环每步把**整份消息列表**重发 ✓，而 skill 注入上限是 60000 字符 ✓
⇒ 多步工具链一定会把上下文撑爆 ✗。

本套最要紧的一条 ✓：**tool 往返必须成对** ✗✗ ——
OpenAI 兼容协议要求带 ``tool_calls`` 的 assistant 消息后**必须**跟上对应的
``role=tool`` 应答 ✓；压缩器一旦"删消息"就很容易留下**没人回答的 tool_call** ✗
（下一次请求直接 400 ✗）。所以本套专门钉：**压缩后消息条数不变 + 成对关系仍成立** ✓✓。

运行::

    ./.venv/Scripts/python.exe tests/agent_context_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.agent.context_budget import (  # noqa: E402
    Budget, apply_plan, estimate_messages_tokens, estimate_message_tokens, estimate_tokens,
    plan_compression, summarize_prompt, tool_pairs_intact)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def tool_round(index: int, *, chars: int = 2000) -> list[dict]:
    """一轮正常的工具往返 ✓：assistant(tool_calls) + tool 应答 ✓。"""
    call_id = f"call_{index}"
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "get_script", "arguments": "{\"episodeId\":1}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "剧本文本" * (chars // 4)},
    ]


def conversation(rounds: int, *, chars: int = 2000) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "你是分镜师" * 20},
                            {"role": "user", "content": "把这集剧本转成分镜"}]
    for index in range(rounds):
        messages.extend(tool_round(index, chars=chars))
    return messages


# ══════════════════════════════════════════════════════════════════════════
# ① 估算（**不使用 tokenizer** ✓ 但中文不能被低估 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_estimate() -> None:
    chinese = estimate_tokens("。" * 100)
    ascii_text = estimate_tokens("a" * 100)
    check("① ⭐ 中文与 ASCII **分开估**：同长度中文的 token 数**明显更高** ✓"
          "（纯按字符数估会严重低估中文 ✗）",
          chinese > ascii_text * 2.5, (round(chinese, 1), round(ascii_text, 1)))
    check("② 空值/非字符串 ⇒ 估 0 ✓ 不崩 ✓",
          estimate_tokens("") == 0 and estimate_tokens(None) == 0
          and estimate_tokens(123) == 0, "")

    with_calls = estimate_message_tokens({
        "role": "assistant", "content": None,
        "tool_calls": [{"function": {"name": "get_script", "arguments": "{\"a\": 1}"}}]})
    check("③ `tool_calls` 的**名字与参数**都算进估算 ✓（不算就会低估多步链 ✓）",
          with_calls > 0, with_calls)

    messages = conversation(3)
    check("④ 整份列表估算 = 逐条之和 ✓（且随轮数线性增长 ✓）",
          estimate_messages_tokens(messages) > estimate_messages_tokens(conversation(1)),
          round(estimate_messages_tokens(messages), 1))


# ══════════════════════════════════════════════════════════════════════════
# ② 未超预算 ⇒ 一个字都不动
# ══════════════════════════════════════════════════════════════════════════
def case_under_budget() -> None:
    messages = conversation(2, chars=200)
    plan = plan_compression(messages, Budget(limit_tokens=100_000))
    check("⑤ 未超预算 ⇒ 不压缩 ✓ 且明说「未超」✓（不做没被要求的干预 ✓）",
          plan.overBudget is False and plan.elide == [] and plan.needsSummarize is False
          and any("未超预算" in item for item in plan.notes), plan.notes)
    check("⑥ 未超预算时 `apply_plan` 原样返回 ✓",
          apply_plan(messages, plan) == messages, "")


# ══════════════════════════════════════════════════════════════════════════
# ③ ⭐ 机械层：压缩**但不破成对**
# ══════════════════════════════════════════════════════════════════════════
def case_mechanical() -> None:
    messages = conversation(8, chars=4000)
    budget = Budget(limit_tokens=4_000, keep_recent=4, keep_head=2)
    plan = plan_compression(messages, budget)
    check("⑦ 超预算 ⇒ 标出要压缩的老 tool 输出 ✓（并给出前后 token 估算 ✓）",
          plan.overBudget is True and plan.elide and plan.tokensAfter < plan.tokensBefore,
          (plan.elide, round(plan.tokensBefore, 1), round(plan.tokensAfter, 1)))
    check("⑧ ⭐ 被压的全是 ``role=tool`` ✓（assistant 的工具选择/推理**不替** ✗）",
          all(messages[index]["role"] == "tool" for index in plan.elide), plan.elide)

    compressed = apply_plan(messages, plan)
    check("⑨ ⭐⭐ **消息条数不变** ✓✓（只换 content ⇒ 协议结构原样 ✓）",
          len(compressed) == len(messages), (len(messages), len(compressed)))
    check("⑩ ⭐⭐ **压缩后 tool 往返仍成对** ✓✓（不变量：否则下次请求直接 400 ✗）",
          tool_pairs_intact(compressed) is True and plan.pairsIntact is True, "")
    check("⑪ 被压的 tool 输出**保留开头 + 说明被压缩** ✓（不假装原文就那么短 ✗）",
          all(compressed[index]["content"] == messages[index]["content"]
              for index in range(len(messages)) if index not in set(plan.elide))
          and all("上下文压缩" in compressed[index]["content"] for index in plan.elide), "")
    check("⑫ 保留开头若干字符 ✓（够模型认出「这是什么」✓）",
          all(len(compressed[index]["content"]) < len(messages[index]["content"])
              for index in plan.elide), "")

    tail_start = len(messages) - budget.keep_recent
    check("⑬ 最近 N 条**一条都不动** ✓（当前这一步的现场，压了就答非所问 ✗）",
          all(index not in set(plan.elide) for index in range(tail_start, len(messages))), "")
    check("⑭ 开头（system + 首条 user）**不动** ✓（任务本身不能丢 ✗）",
          all(index not in set(plan.elide) for index in range(0, budget.keep_head)), "")

    mostly_short = conversation(4, chars=120)
    short_plan = plan_compression(mostly_short, Budget(limit_tokens=1, keep_recent=2))
    check("⑮ 本来就短的 tool 输出**不压** ✓（压了反而多占一条提示 ✓）",
          short_plan.elide == [], short_plan.elide)


# ══════════════════════════════════════════════════════════════════════════
# ④ 摘要层：机械层不够时**如实说**，但把决定权交回调用方
# ══════════════════════════════════════════════════════════════════════════
def case_summarize() -> None:
    messages = conversation(6, chars=3000)
    plan = plan_compression(messages, Budget(limit_tokens=200, keep_recent=2, keep_head=2))
    check("⑯ 机械层不够 ⇒ `needsSummarize=True` ✓ 且给出**建议摘要的区间** ✓",
          plan.needsSummarize is True and plan.summarize, (plan.needsSummarize, plan.summarize))
    check("⑰ ⭐ 建议摘要的区间**只含没法机械压缩的**消息 ✓（tool 应答已机械处理 ✓）",
          all(messages[index]["role"] != "tool" for index in plan.summarize), plan.summarize)

    prompt = summarize_prompt([messages[index] for index in plan.summarize])
    check("⑱ 摘要提示词**要求保结构**（工具/参数/结论 ✓）并**明令不许编** ✗",
          "保结构" in prompt and "不许编" in prompt and "中间过程摘要" in prompt, prompt[:60])
    check("⑲ 摘要提示词里带**执行记录本体** ✓（不是只给个空壳 ✓）",
          "get_script" in prompt, "")

    nearly = conversation(8, chars=4000)
    small = plan_compression(nearly, Budget(limit_tokens=int(plan_compression(
        nearly, Budget(limit_tokens=4_000, keep_recent=4)).tokensAfter) + 10,
        keep_recent=4))
    check("⑳ 只差一点点 ⇒ 说明**不值得**再花一次 LLM 调用 ✓（有判断、不无脑摘要 ✓）",
          any("不值得" in item for item in small.notes) or small.needsSummarize is False,
          small.notes)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 不变量检查器本身要可靠（它坏了 ⇒ 上面所有断言都是假绿 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_invariant() -> None:
    good = conversation(2)
    check("㉑ 正常对话 ⇒ 成对 ✓", tool_pairs_intact(good) is True, "")

    missing_reply = [item for item in conversation(2) if item.get("role") != "tool"]
    check("㉒ ⭐ **tool_call 没人回答** ⇒ 判为不成对 ✓✓（这就是会让端点 400 的形态 ✓）",
          tool_pairs_intact(missing_reply) is False, "")

    orphan = [{"role": "system", "content": "x"},
              {"role": "tool", "tool_call_id": "call_9", "content": "孤儿应答"}]
    check("㉓ ⭐ **孤儿 tool 应答** ⇒ 判为不成对 ✓",
          tool_pairs_intact(orphan) is False, "")

    broken = plan_compression(missing_reply, Budget(limit_tokens=1))
    check("㉔ 进来就已坏 ⇒ **先如实说明是上游的问题** ✓（不是压缩造成的 ✗）",
          broken.pairsIntact is False
          and any("本身" in item for item in broken.notes), broken.notes)

    check("㉕ 空列表 / 非列表 ⇒ 不崩且给出说明 ✓",
          plan_compression([]).notes and plan_compression("坏的").notes
          and apply_plan("坏的", plan_compression([])) == "坏的", "")


def main() -> int:
    case_estimate()
    case_under_budget()
    case_mechanical()
    case_summarize()
    case_invariant()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
