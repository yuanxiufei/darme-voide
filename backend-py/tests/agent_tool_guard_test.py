"""工具调用守卫（P0-1 重复守卫 + P0-2 两段式人工介入）的守卫测试。

被测：``app/agent/repeat_guard.py``、``app/agent/intervention.py``、
``app/agent/runtime.py`` 的 ``_run_tool`` / ``_run_call_batch`` / ``AgentRunResult``。

盯的是这几处**"看起来实现了、其实没生效"**的形状：

* 参数不归一 ⇒ ``{"a":1,"b":2}`` 与 ``{"b":2,"a":1}`` 数成两次不同调用 ⇒ 打转永远数不满；
* 阈值差一（``>`` 与 ``>=``）⇒ 第 20 次不停、第 21 次才停，与参考实现差一拍；
* 中间穿插别的调用就清零 ⇒ 最常见的打转形态（A,B,A,B…）漏光；
* 审批键只按工具名 ⇒ 批准了「这个工具的任意参数」，护栏等于拆掉；
* ``pending`` 时工具**真的被调用了**（副作用已发生才回一句"等批准"）—— 那是最坏的一种假实现。

跑法：``.venv/Scripts/python.exe tests/agent_tool_guard_test.py``。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: 数据根在 **import app.core.db 之前**指到临时目录（否则动真库）
_DATA_ROOT = Path(tempfile.mkdtemp(prefix="agent_tool_guard_test_"))
os.environ["DATA_ROOT"] = str(_DATA_ROOT)
os.environ.setdefault("SKILLS_DIR", str(_DATA_ROOT / "skills"))

from app.agent import intervention as iv  # noqa: E402
from app.agent import repeat_guard as rg  # noqa: E402
from app.agent import runtime  # noqa: E402
from app.agent.tool import Tool, ToolRegistry  # noqa: E402

_failures: list[str] = []
_passed = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _passed
    if ok:
        _passed += 1
        print(f"  ✓ {label}")
        return True
    _failures.append(f"{label}{(' —— ' + detail) if detail else ''}")
    print(f"  ✗ {label}{(' —— ' + detail) if detail else ''}")
    return False


# ---------------------------------------------------------------------------
# 一、签名归一
# ---------------------------------------------------------------------------

def case_canonical() -> None:
    print("\n[1] 参数归一（不归一守卫就是死的）")
    a = rg.normalize_arguments({"b": 2, "a": 1})
    b = rg.normalize_arguments({"a": 1, "b": 2})
    check("键序不同 ⇒ 同一串", a == b, f"{a!r} vs {b!r}")
    check("数组保序（[1,2] ≠ [2,1]）",
          rg.normalize_arguments([1, 2]) != rg.normalize_arguments([2, 1]))

    check("JSON 字符串先解析再归一",
          rg.normalize_arguments('{"b": 2, "a": 1}') == rg.normalize_arguments({"a": 1, "b": 2}))
    check("同一份参数的字符串与 dict 同签名",
          rg.tool_signature("t", '{"a": 1}') == rg.tool_signature("t", {"a": 1}))
    check("坏 JSON ⇒ 退原文（trim）", rg.normalize_arguments('  {"a": 1  ') == '{"a": 1')
    check("None / 空串 ⇒ 空", rg.normalize_arguments(None) == ""
          and rg.normalize_arguments("   ") == "")
    check("签名用 \\0 分隔（工具名里的 | 不会串位）",
          rg.tool_signature("a|b", None) == "a|b\x00")
    check("嵌套对象也按键排序",
          rg.normalize_arguments({"x": {"b": 1, "a": 2}}) == '{"x":{"a":2,"b":1}}')


# ---------------------------------------------------------------------------
# 二、重复闸门 / 爆发闸门
# ---------------------------------------------------------------------------

def case_repeat_limits() -> None:
    print("\n[2] 重复闸门（累计计数，阈值 >= 才停）")
    guard = rg.RepeatGuard()
    check("默认阈值 20/50", (guard.repeat_limit, guard.burst_limit) == (20, 50))
    hits = [guard.observe("gen", {"prompt": "x"}) for _ in range(19)]
    check("前 19 次都不停", all(hit is None for hit in hits))
    hit = guard.observe("gen", {"prompt": "x"})
    check("第 20 次停（>= 而非 >）", hit is not None and hit.kind == "repeat",
          f"第 20 次={hit}")
    check("停止消息里点名工具与上限",
          hit is not None and "gen" in hit.message and "20" in hit.message)
    check("同签名再调仍然停（粘住语义）", guard.observe("gen", {"prompt": "x"}) is not None)

    other = rg.RepeatGuard()
    for _ in range(10):
        other.observe("gen", {"prompt": "x"})
        other.observe("read", {"id": 1})     # 穿插别的调用
    for _ in range(9):
        other.observe("gen", {"prompt": "x"})
    hit = other.observe("gen", {"prompt": "x"})
    check("穿插别的调用**不清零**（累计语义，参考代码而非其注释）",
          hit is not None, "第 20 次没停 ⇒ 中间被清零了")
    check("回执如实标注 cumulative", other.receipt()["counting"] == "cumulative")

    burst = rg.RepeatGuard()
    check("50 个不触发 burst", burst.check_burst(50) is None)
    hit = burst.check_burst(51)
    check("51 个触发 burst", hit is not None and hit.kind == "burst")
    check("burst 不记调用计数（一条都没跑）",
          burst.counts == {}, f"{burst.counts}")

    forced = rg.guard_from_options({"repeatLimit": 0, "burstLimit": -3})
    check("阈值传 0/负数 ⇒ 回落默认（**守卫不可关闭**）",
          (forced.repeat_limit, forced.burst_limit) == (20, 50))
    custom = rg.guard_from_options({"repeat_limit": 3})
    check("允许按跑调小（snake_case 也认）", custom.repeat_limit == 3)


# ---------------------------------------------------------------------------
# 三、规则引擎
# ---------------------------------------------------------------------------

def case_matching() -> None:
    print("\n[3] 匹配规则（wildcard / 冒号前缀 / regex / AND 语义）")
    check("通配 *", iv.match_pattern("create*", "create_skill"))
    check("冒号前缀精确等于前缀本身", iv.match_pattern("git add:*", "git add"))
    check("冒号前缀带参数（参考只认 `git add:`；本仓多认空格，否则真实命令永不命中）",
          iv.match_pattern("git add:*", "git add foo.md"))
    check("冒号前缀的原文形状也认", iv.match_pattern("git add:*", "git add:foo.md"))
    check("冒号前缀不误伤别的命令", not iv.match_pattern("git add:*", "git commit -m x"))
    check("转义的点号不当通配（a.b 不匹配 axb）",
          not iv.match_pattern("a.b", "axb"))
    check("转义的点号命中自身", iv.match_pattern("a.b", "a.b"))

    check("exact", iv.match_argument({"type": "exact", "pattern": "rm"}, "rm"))
    check("exact 不做子串", not iv.match_argument({"type": "exact", "pattern": "rm"}, "rm -rf /"))
    check("prefix", iv.match_argument({"type": "prefix", "pattern": "rm "}, "rm -rf /"))
    check("regex（`(?!\\.pub)` 负向前瞻可用）",
          not iv.match_argument({"type": "regex", "pattern": r"id_rsa(?!\.pub)"}, "id_rsa.pub"))
    check("未知 type ⇒ 不命中（不猜）",
          not iv.match_argument({"type": "glob", "pattern": "*"}, "x"))
    check("裸字符串＝wildcard", iv.match_argument("*rm*", "sudo rm -rf /"))

    rule = {"match": {"command": "*rm*", "path": {"type": "prefix", "pattern": "/home"}}}
    check("AND 语义：两个都中才算中",
          iv._rule_matches(rule, {"command": "rm x", "path": "/home/a"}))
    check("AND 语义：只中一个不算", not iv._rule_matches(rule, {"command": "rm x", "path": "/tmp"}))
    check("参数缺失 ⇒ 不命中（不是空串命中）",
          not iv._rule_matches({"match": {"path": "*"}}, {}))
    check("参数为 None ⇒ 不命中",
          not iv._rule_matches({"match": {"path": "*"}}, {"path": None}))


def case_should_intervene() -> None:
    print("\n[4] 判定顺序（安全优先 ⇒ 配置 ⇒ 从严兜底）")
    policy, reason = iv.should_intervene(None, {"x": 1})
    check("无配置 ⇒ never（默认不打扰）", policy == "never", policy)

    policy, _ = iv.should_intervene("required", {})
    check("字符串策略直取", policy == "required")

    policy, _ = iv.should_intervene(
        [{"match": {"command": "*git push*"}, "policy": "always"}], {"command": "git push origin"})
    check("规则命中 ⇒ 取该规则的 policy", policy == "always", policy)

    policy, reason = iv.should_intervene(
        [{"match": {"command": "*git push*"}, "policy": "always"}], {"command": "ls"})
    check("配了规则但一条都没命中 ⇒ required（从严兜底）", policy == "required", reason)

    policy, reason = iv.should_intervene(
        "never", {"command": "cat ~/.ssh/id_rsa"})
    check("**安全黑名单压过 never**（创作者怎么配都拦不住）",
          policy == "required", f"{policy} / {reason}")
    check("黑名单的原因进 reason", "黑名单" in reason)

    policy, _ = iv.should_intervene("never", {"command": "cat ~/.ssh/id_rsa.pub"})
    check("公钥不算私钥（(?!\\.pub) 生效）", policy == "never", policy)

    policy, _ = iv.should_intervene("never", {"path": "data/drama.db"})
    check("本仓主库路径进黑名单", policy == "required", policy)

    policy, reason = iv.should_intervene({"policy": "也许"}, {})
    check("认不出的 policy ⇒ required（不静默放行）", policy == "required", reason)


def case_hash_key() -> None:
    print("\n[5] 审批键（必须带参数指纹）")
    check("键序无关", iv.hash_arguments({"b": 2, "a": 1}) == iv.hash_arguments({"a": 1, "b": 2}))
    check("值不同 ⇒ 键不同",
          iv.hash_arguments({"path": "a.md"}) != iv.hash_arguments({"path": "b.md"}))
    check("空参数也有稳定值", isinstance(iv.hash_arguments({}), str) and iv.hash_arguments({}) == "0")
    check("tool_key 带指纹", iv.tool_key("save", iv.hash_arguments({"a": 1}))
          == "save#" + iv.hash_arguments({"a": 1}))
    check("tool_key 无指纹时只有名字", iv.tool_key("save") == "save")


# ---------------------------------------------------------------------------
# 六、闸口（记忆 / 拒绝 / pending）
# ---------------------------------------------------------------------------

def case_gate() -> None:
    print("\n[6] 两段式：先登记 pending，下一跑带批准")
    gate = iv.InterventionGate(interventions={"delete_episode": "required"})
    decision = gate.decide("delete_episode", {"episodeId": 7}, call_id="c1")
    check("首跑 ⇒ pending（不自动执行）", decision.kind == "pending", decision.kind)
    check("pending 清单带 callId/tool/key",
          gate.pending and gate.pending[0].as_dict()["callId"] == "c1")
    check("pending 消息告诉模型'没有执行'",
          iv.pending_message(gate.pending[0])["status"] == "pending_approval")
    check("pending 消息不许它重试", "不要重试" in iv.pending_message(gate.pending[0])["note"])

    key = gate.pending[0].key
    approved = iv.InterventionGate(interventions={"delete_episode": "required"},
                                     approvals={key: {"action": "approve", "remember": True}})
    decision = approved.decide("delete_episode", {"episodeId": 7})
    check("批准 ⇒ run", decision.run, decision.kind)
    decision = approved.decide("delete_episode", {"episodeId": 7})
    check("remember ⇒ 同一参数第二次不再问", decision.run and decision.remembered)
    decision = approved.decide("delete_episode", {"episodeId": 8})
    check("**换个参数 ⇒ 仍然要批**（键带指纹的意义）", decision.kind == "pending", decision.kind)

    remembered_via_second_run = iv.InterventionGate(
        interventions={"delete_episode": "required"},
        approvals={key: "approve"},   # 字符串简写、不带 remember
    )
    remembered_via_second_run.decide("delete_episode", {"episodeId": 7})
    again = iv.InterventionGate(interventions={"delete_episode": "required"},
                                 approvals={key: "approve"})
    again.decide("delete_episode", {"episodeId": 7})
    again.decide("delete_episode", {"episodeId": 7})
    check("不 remember ⇒ 每次都批（不是一次批终身）",
          len(again.confirmed) == 0, f"{again.confirmed}")

    always = iv.InterventionGate(interventions={"gen": "always"},
                                  approvals={iv.tool_key("gen", iv.hash_arguments({})): {"action": "approve", "remember": True}})
    first = always.decide("gen", {})
    second = always.decide("gen", {})
    check("always 不吃记忆（每次都批）",
          first.run and second.run and not always.confirmed, f"{always.confirmed}")

    rejected = iv.InterventionGate(interventions={"gen": "required"},
                                    approvals={iv.tool_key("gen", iv.hash_arguments({})): "reject"})
    decision = rejected.decide("gen", {})
    check("拒绝 ⇒ rejected（不是 pending）", decision.kind == "rejected", decision.kind)
    check("拒绝消息明确说'被拒绝，别绕路'",
          "拒绝" in iv.rejected_message(rejected.rejected[0])["note"])

    wildcard = iv.InterventionGate(interventions={"*": "required"})
    decision = wildcard.decide("mcp__fs__write", {"path": "a"})
    check("`*` 兜底键生效并记账", decision.kind == "pending" and wildcard.wildcard_used)

    exact_wins = iv.InterventionGate(interventions={"*": "required", "read": "never"})
    check("精确键优先于 `*`", exact_wins.decide("read", {}).run)

    normalized = iv.gate_from_options({"approvals": [{"key": "k#1", "action": "approve"}]})
    check("approvals 列表形状也认", normalized.approvals.get("k#1", {}).get("action") == "approve")


# ---------------------------------------------------------------------------
# 七、接进运行时（唯一执行口）
# ---------------------------------------------------------------------------

def _registry() -> tuple[ToolRegistry, list[dict]]:
    calls: list[dict] = []

    def make(tool_id: str):
        def run(args: dict):
            calls.append({"tool": tool_id, "args": args})
            return {"ok": True, "tool": tool_id}
        return run

    tools = {tool_id: Tool(id=tool_id, description=tool_id, input_schema={"type": "object"},
                           execute=make(tool_id))
             for tool_id in ("gen", "read")}
    return ToolRegistry(tools), calls


def case_runtime_gate() -> None:
    print("\n[7] _run_tool 里的闸口（**副作用没发生**才算真拦住）")
    tools, calls = _registry()
    guard = runtime._ToolCallGuard(
        guard=rg.RepeatGuard(), gate=iv.InterventionGate(interventions={"gen": "required"}))
    token = runtime._TOOL_GUARD.set(guard)
    try:
        result = asyncio.run(runtime._run_tool(tools, "gen", {"p": 1}, [], []))
    finally:
        runtime._TOOL_GUARD.reset(token)
    check("pending ⇒ 工具**一次都没被调用**", calls == [], f"{calls}")
    check("pending ⇒ 回执里 status=pending_approval", "pending_approval" in result, result[:80])
    check("pending 进了闸口清单", len(guard.gate.pending) == 1)

    tools2, calls2 = _registry()
    approved_key = iv.tool_key("gen", iv.hash_arguments({"p": 1}))
    guard2 = runtime._ToolCallGuard(
        guard=rg.RepeatGuard(),
        gate=iv.InterventionGate(interventions={"gen": "required"},
                                  approvals={approved_key: {"action": "approve", "remember": True}}))
    token = runtime._TOOL_GUARD.set(guard2)
    try:
        asyncio.run(runtime._run_tool(tools2, "gen", {"p": 1}, [], []))
        asyncio.run(runtime._run_tool(tools2, "gen", {"p": 1}, [], []))
    finally:
        runtime._TOOL_GUARD.reset(token)
    check("批准后真执行（两次）", len(calls2) == 2, f"{calls2}")
    check("remember ⇒ 第二次不问", len(guard2.gate.pending) == 0)

    tools3, calls3 = _registry()
    guard3 = runtime._ToolCallGuard(guard=rg.RepeatGuard(repeat_limit=2),
                                    gate=iv.InterventionGate())
    token = runtime._TOOL_GUARD.set(guard3)
    try:
        first = asyncio.run(runtime._run_tool(tools3, "read", {"id": 1}, [], []))
        second = asyncio.run(runtime._run_tool(tools3, "read", {"id": 1}, [], []))
        third = asyncio.run(runtime._run_tool(tools3, "read", {"id": 1}, [], []))
    finally:
        runtime._TOOL_GUARD.reset(token)
    # 语义与参考实现一致：**第 limit 次**就拦（阈值 2 ⇒ 第 2 次不执行、只跑了 1 次）
    check("第 2 次（阈值 2）就不再执行", len(calls3) == 1, f"{calls3}")
    check("被拦的那次回执说 skipped", "skipped" in second and "skipped" in third,
          f"{second[:60]} / {third[:60]}")
    check("被判之前那次正常跑", "skipped" not in first)
    check("闸口 stop 置位并带 kind", guard3.stop is not None and guard3.stop.kind == "repeat")
    check("回执含重复账与审批账",
          {"repeat", "intervention"} <= set(guard3.receipt()))


def case_batch_pairs() -> None:
    print("\n[8] 批级：burst 时每条调用都有对应结果（③-23 成对约束）")
    tools, calls = _registry()
    guard = runtime._ToolCallGuard(guard=rg.RepeatGuard(burst_limit=2),
                                    gate=iv.InterventionGate())
    token = runtime._TOOL_GUARD.set(guard)
    raw_calls = [{"id": f"c{i}", "function": {"name": "read",
                                              "arguments": json.dumps({"id": i})}}
                 for i in range(3)]
    call_log: list[dict] = []
    results: list[dict] = []
    try:
        stop = asyncio.run(runtime._run_call_batch(
            tools, raw_calls, call_log, results,
            lambda call, name, rendered: None,
        ))
    finally:
        runtime._TOOL_GUARD.reset(token)
    check("burst ⇒ 返回叫停", stop is not None and stop.kind == "burst")
    check("三个工具**一个都没跑**", calls == [], f"{calls}")
    check("calls 与 results 逐条成对",
          len(call_log) == 3 and len(results) == 3, f"{len(call_log)}/{len(results)}")
    check("每条回执都说明未执行",
          all("skipped" in json.dumps(item, ensure_ascii=False) for item in results))


def case_run_result_shape() -> None:
    print("\n[9] AgentRunResult 携带守卫账")
    from app.agent.runtime import AgentRunResult
    result = AgentRunResult(model="m", text="t", tool_calls=[], tool_results=[],
                            protocol=None, protocol_errors=[], usage=None)
    check("默认无 pending（老调用方零变化）", result.pending_approvals == [])
    check("默认 stop=None", result.stop is None)
    check("默认 guard=None", result.guard is None)


def case_wiring() -> None:
    print("\n[10] 接线（源码级：三循环都走批口，没有漏网的 for 循环）")
    source = (_ROOT / "app" / "agent" / "runtime.py").read_text(encoding="utf-8")
    check("三条循环都走 _run_call_batch", source.count("await _run_call_batch(") == 3,
          f"{source.count('await _run_call_batch(')}")
    # 唯一允许的 `for call in calls:` 就在 `_run_call_batch` 里（其余三条循环必须走批口）
    check("逐条循环只剩批口自己那一个",
          source.count("for call in calls:") == 1, f"{source.count('for call in calls:')}")
    check("闸口经 ContextVar 传入", "_TOOL_GUARD.set(" in source and "_TOOL_GUARD.reset(" in source)
    check("每次 driver 尝试都新建闸口（fallback 换模型不继承上轮计数）",
          source.count("_ToolCallGuard(guard=guard_from_options") == 1)
    check("叫停写进 steps", 'steps.append({"guardStop"' in source)


def main() -> int:
    case_canonical()
    case_repeat_limits()
    case_matching()
    case_should_intervene()
    case_hash_key()
    case_gate()
    case_runtime_gate()
    case_batch_pairs()
    case_run_result_shape()
    case_wiring()

    total = _passed + len(_failures)
    print(f"\nSUMMARY: {_passed}/{total} passed")
    if _failures:
        print("失败项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
