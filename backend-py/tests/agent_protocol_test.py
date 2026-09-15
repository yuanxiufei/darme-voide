"""S5 第一块自检：Agent 输出协议 + 工具基座（``services/agents/``）。

协议是**下游（前端/评测/优化/管线）消费的唯一稳定接口**，所以：
* 契约文本必须逐字稳定（Agent 靠它产出 YAML）；
* 解析器的**六类失败分支**与英文文案都要对齐（``empty text`` / ``no yaml fence found`` /
  ``yaml parse error: …`` / ``protocol is not an object`` / ``invalid status: …`` /
  ``missing or empty summary``）；
* ⚠️ ``invalid status`` 要区分**缺键**（``undefined``）与**显式 null**（``null``）——
  这正是 JS ``JSON.stringify(undefined)`` 与 ``JSON.stringify(null)`` 的差别。

运行::

    ./.venv/Scripts/python.exe tests/agent_protocol_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="agentproto_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.protocol import build_protocol_contract, parse_agent_protocol  # noqa: E402
from app.agent.tool import (  # noqa: E402
    Tool,
    ToolRegistry,
    array_of,
    json_number,
    json_string,
    object_schema,
)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    # ================= 契约文本 =================
    contract = build_protocol_contract()
    check(
        "契约: 含 yaml 围栏与 status/summary 示例",
        "```yaml" in contract and "status: ok" in contract and "summary:" in contract
        and contract.startswith("## 输出协议（必须遵守）"),
        contract[:40],
    )
    check(
        "契约: 三条要求齐全（ok/failed、一句话、失败也要输出）",
        "- status 只能填 ok 或 failed" in contract
        and "- summary 用一句话概括" in contract
        and "即使中途工具调用失败" in contract,
    )
    check("契约: 多次调用结果完全一致（无随机/无时间）",
          build_protocol_contract() == contract)

    # ================= 正常解析 =================
    good = parse_agent_protocol("完成了。\n\n```yaml\nstatus: ok\nsummary:  已保存 12 个分镜  \n```\n")
    check(
        "解析: ok 协议通过，且 summary 被 trim",
        good["errors"] == [] and good["protocol"] == {"status": "ok", "summary": "已保存 12 个分镜"},
        good,
    )
    check("解析: failed 也合法（失败也要输出协议）",
          parse_agent_protocol("```yaml\nstatus: failed\nsummary: 工具调用出错\n```")["errors"] == [])
    extra = parse_agent_protocol(
        "```yaml\nstatus: ok\nsummary: 完成\nsaved_storyboards: 12\nwarnings: [a]\n```")
    check(
        "解析: Agent 附加的结构化字段**保留**（供下游交叉验证）",
        extra["protocol"].get("saved_storyboards") == 12 and extra["protocol"].get("warnings") == ["a"],
        extra["protocol"],
    )
    check("解析: ```yml（三字母）围栏也认",
          parse_agent_protocol("```yml\nstatus: ok\nsummary: x\n```")["errors"] == [])
    check("解析: 取**第一个**围栏（后面还有别的代码块时不受影响）",
          parse_agent_protocol(
              "```yaml\nstatus: ok\nsummary: 第一个\n```\n```json\n{}\n```"
          )["protocol"]["summary"] == "第一个")

    # ================= 六类失败分支 =================
    check("失败: 空文本 -> 'empty text'",
          parse_agent_protocol("")["errors"] == ["empty text"]
          and parse_agent_protocol(None)["errors"] == ["empty text"])
    check("失败: 没有围栏 -> 'no yaml fence found'",
          parse_agent_protocol("我完成了，但没写协议")["errors"] == ["no yaml fence found"])
    check("失败: YAML 语法错 -> 'yaml parse error: …' 前缀",
          parse_agent_protocol("```yaml\nstatus: [ok\n```")["errors"][0].startswith("yaml parse error: "),
          parse_agent_protocol("```yaml\nstatus: [ok\n```")["errors"])
    check("失败: 解析出来不是对象（数组）-> 'protocol is not an object'",
          parse_agent_protocol("```yaml\n- a\n- b\n```")["errors"] == ["protocol is not an object"])
    check("失败: status 值非法 -> 带引号的 JSON 形式",
          parse_agent_protocol('```yaml\nstatus: "weird"\nsummary: x\n```')["errors"]
          == ['invalid status: "weird"'],
          parse_agent_protocol('```yaml\nstatus: "weird"\nsummary: x\n```')["errors"])
    # ⭐ 缺键 vs 显式 null：对应 JS 的 undefined 与 null
    check("失败: **缺 status 键** -> `invalid status: undefined`（JS 的 undefined 语义）",
          parse_agent_protocol("```yaml\nsummary: x\n```")["errors"]
          == ["invalid status: undefined"],
          parse_agent_protocol("```yaml\nsummary: x\n```")["errors"])
    check("失败: **status 显式为 null** -> `invalid status: null`",
          parse_agent_protocol("```yaml\nstatus:\nsummary: x\n```")["errors"]
          == ["invalid status: null"],
          parse_agent_protocol("```yaml\nstatus:\nsummary: x\n```")["errors"])
    check("失败: summary 缺失 -> 'missing or empty summary'",
          parse_agent_protocol("```yaml\nstatus: ok\n```")["errors"]
          == ["missing or empty summary"])
    check("失败: summary 是纯空白也判失败（trim 后为空）",
          parse_agent_protocol("```yaml\nstatus: ok\nsummary: '   '\n```")["errors"]
          == ["missing or empty summary"])
    check("失败: summary 不是字符串（数字）也算失败",
          parse_agent_protocol("```yaml\nstatus: ok\nsummary: 12\n```")["errors"]
          == ["missing or empty summary"])

    # ================= 工具基座 =================
    async def _echo(args: dict):
        return {"echo": args.get("n")}

    def _sync(args: dict):
        return {"sync": True}

    registry = ToolRegistry.of(ToolRegistry({
        "echo": Tool(
            id="echo", description="回声",
            input_schema=object_schema({"n": json_number(), "tags": array_of(json_string())}),
            execute=_echo,
        ),
        "sync": Tool(id="sync", description="同步工具",
                     input_schema=object_schema({}, required=[]), execute=_sync),
    }))
    check("工具: 注册表可取值/列 id", registry.get("echo") is not None
          and registry.get("nope") is None and registry.ids() == ["echo", "sync"])
    check(
        "工具: OpenAI 契约形状（type=function + parameters 是 JSON Schema）",
        registry.as_openai_tools()[0]["type"] == "function"
        and registry.as_openai_tools()[0]["function"]["name"] == "echo"
        and registry.as_openai_tools()[0]["function"]["parameters"]["type"] == "object",
    )
    check(
        "工具: Gemini 契约形状（functionDeclarations）",
        registry.as_gemini_tools()[0]["functionDeclarations"][0]["name"] == "echo",
        registry.as_gemini_tools(),
    )
    check("工具: object_schema 默认全部必填（与 zod 一致）",
          registry.get("echo").input_schema["required"] == ["n", "tags"]
          and registry.get("sync").input_schema["required"] == [])
    check("工具: 异步 execute 可 await",
          asyncio.run(registry.get("echo").run({"n": 7})) == {"echo": 7})
    check("工具: 同步 execute 也走同一条 run（内部统一 await）",
          asyncio.run(registry.get("sync").run()) == {"sync": True})
    check("工具: 不传参数按空对象处理", asyncio.run(registry.get("sync").run(None)) == {"sync": True})

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
