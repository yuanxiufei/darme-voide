"""S6 自检：MCP 工具接入层（``agents/mcp.py`` ← ``agents/mcp.ts`` 285 行）+ 路由（3 端点）。

本层最值钱的不是「连上」（那很短），而是**连不上时会怎样**：

* 配置文件缺失 / 坏 JSON / 非法条目 ⇒ 空列表，**对现有 Agent 零影响**；
* 单个 server 连接失败 ⇒ **只降级、绝不 throw**（外部工具挂了不能拖垮 Agent）；
* **single-flight + 指纹缓存**：并发调用共享一次发现；配置变了才重建；
* 命名隔离 ``mcp__<server>__<tool>`` + 重名跳过。

测试用**假 stdio MCP server**（临时脚本，说 JSON-RPC）跑通整条握手链路，不打网络。

运行::

    ./.venv/Scripts/python.exe tests/mcp_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="mcp_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.mcp import client as mcpmod  # noqa: E402  （mcp 已是顶层包；模块名不能写在 agent 的 import 位）
from app.agent.tool import Tool, ToolRegistry  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: 假 MCP server：说 stdio 的按行 JSON-RPC，只认三个方法
_FAKE_SERVER = '''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    if "id" not in msg:          # 通知（initialized）不回
        continue
    rid = msg["id"]
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        # 故意重复一个名字：验证「重名跳过」
        result = {"tools": [
            {"name": "echo", "description": "Echo text"},
            {"name": "no.desc"},
            {"name": "echo", "description": "重复的同名工具"},
        ]}
    elif method == "tools/call":
        params = msg.get("params") or {}
        result = {"content": [{"type": "text", "text": "called:" + str(params.get("name"))},
                              {"type": "text", "text": str(sorted((params.get("arguments") or {}).items()))}]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
    sys.stdout.flush()
'''

#: 立刻退出的假 server（模拟连接成功但列表失败 → 走降级）
_BROKEN_SERVER = 'import sys\nsys.exit(3)\n'


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:  # noqa: C901
    workdir = Path(tempfile.mkdtemp(prefix="mcp_srv_"))
    good_script = workdir / "fake_server.py"
    broken_script = workdir / "broken_server.py"
    _write(good_script, _FAKE_SERVER)
    _write(broken_script, _BROKEN_SERVER)
    configs = workdir / "mcp-servers.json"
    os.environ["MCP_SERVERS_PATH"] = str(configs)

    def configure(items: object) -> None:
        _write(configs, items if isinstance(items, str)
               else json.dumps(items, ensure_ascii=False))

    def good_entry(name: str = "demo", **extra) -> dict:
        return {"name": name, "command": sys.executable, "args": [str(good_script)], **extra}

    # ================= 配置净化 =================
    configs.unlink(missing_ok=True)
    check("配置: 文件缺失 -> 空列表（对现有 Agent 零影响）", mcpmod.load_mcp_servers() == [])
    configure("{坏 JSON")
    check("配置: 解析失败 -> 空列表（不抛异常）", mcpmod.load_mcp_servers() == [])
    configure({"not": "array"})
    check("配置: 非数组 -> 空列表", mcpmod.load_mcp_servers() == [])
    configure([
        None, "x", {"no_name": 1}, {"name": "   "}, {"name": "  pad  "},
        {"name": "off", "enabled": False}, {"name": "on", "enabled": True, "command": "c"},
    ])
    names = [c["name"] for c in mcpmod.load_mcp_servers()]
    check("配置: 非法条目跳过 + 名字 trim + enabled=False 跳过",
          names == ["pad", "on"], names)

    # ================= 传输推断 / 净化 / 结果序列化 =================
    check("传输: 显式 transport 优先 > command 推 stdio > 否则 http",
          mcpmod.infer_transport({"transport": "sse", "command": "x"}) == "sse"
          and mcpmod.infer_transport({"command": "x"}) == "stdio"
          and mcpmod.infer_transport({"url": "http://a"}) == "http",
          [mcpmod.infer_transport({"transport": "sse", "command": "x"}),
           mcpmod.infer_transport({"command": "x"}), mcpmod.infer_transport({"url": "http://a"})])
    check("净化: 非 [a-zA-Z0-9_-] 字符换成下划线（中文/点/斜杠都不合法）",
          mcpmod.sanitize_segment("a.b/c d") == "a_b_c_d"
          and mcpmod.sanitize_segment("工具") == "__"
          and mcpmod.sanitize_segment("ok-name_1") == "ok-name_1",
          [mcpmod.sanitize_segment("a.b/c d"), mcpmod.sanitize_segment("工具")])
    check("序列化: structuredContent 优先（合并 + isError）",
          mcpmod.serialize_tool_result({"structuredContent": {"a": 1}, "isError": True})
          == {"a": 1, "isError": True})
    check("序列化: content 里的 text 片段用换行拼接",
          mcpmod.serialize_tool_result({"content": [
              {"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]})
          == {"text": "a\nb", "isError": False})
    check("序列化: 其余情况落 raw 紧凑 JSON（含 null）",
          mcpmod.serialize_tool_result({"other": 1})["raw"] == '{"other":1}'
          and mcpmod.serialize_tool_result(None)["raw"] == "null",
          mcpmod.serialize_tool_result({"other": 1}))

    # ================= 发现（假 server 端到端）=================
    # ⚠️ 必须收进**同一次** `asyncio.run`：stdio 子进程的管道绑定在创建它的那个事件循环上，
    #    跨 `asyncio.run` 复用会在「管道已随旧循环销毁」时炸出 `'NoneType' has no attribute 'send'`。
    #    （生产环境整进程一个循环，天然没这个问题。）
    async def async_checks() -> None:
        async def discover() -> dict:
            return await mcpmod.discover_mcp_tools()

        configure([good_entry()])
        tools = await discover()
        check("发现: 工具名是 `mcp__<server>__<tool>` 三段式（点号也被净化成下划线）",
              set(tools) == {"mcp__demo__echo", "mcp__demo__no_desc"}, sorted(tools))
        check("发现: 带描述的工具描述是 `[MCP/<server>] <原描述>`",
              tools["mcp__demo__echo"].description == "[MCP/demo] Echo text",
              tools["mcp__demo__echo"].description)
        check("发现: **无描述**的工具回落 `[MCP/x] external tool <名>`",
              "mcp__demo__no_desc" in tools
              and tools["mcp__demo__no_desc"].description == "[MCP/demo] external tool no.desc",
              tools.get("mcp__demo__no_desc"))
        check("发现: 入参 schema 是**宽松对象**（不还原 MCP 的 JSON Schema）",
              tools["mcp__demo__echo"].input_schema
              == {"type": "object", "additionalProperties": True})
        check("发现: 同名工具只留第一个（重名跳过，且不炸）",
              len([k for k in tools if k.endswith("echo")]) == 1, sorted(tools))

        called = await tools["mcp__demo__echo"].run({"b": 2, "a": 1})
        check("调用: 工具真的走了一趟 MCP（结果被序列化成 {text, isError}）",
              called.get("isError") is False and "called:echo" in called.get("text", "")
              and "('a', 1)" in called.get("text", ""),
              called)

        cached = await discover()
        check("缓存: 配置未变 -> 直接复用缓存（同一批对象，不重连）",
              cached["mcp__demo__echo"] is tools["mcp__demo__echo"])

        configure([good_entry(connectTimeoutMs=9_000)])
        changed = await discover()
        check("缓存: 配置指纹变了 -> 重建（工具是**新对象**）",
              changed["mcp__demo__echo"] is not tools["mcp__demo__echo"])

        status = mcpmod.get_mcp_status()
        check("状态: configured 来自配置、connected 带 transport/toolCount/tools",
              status["configured"] == ["demo"]
              and status["connected"] == [{
                  "server": "demo", "transport": "stdio",
                  "toolCount": 2, "tools": ["mcp__demo__echo", "mcp__demo__no_desc"],
              }],
              status)

        # 连接失败只降级
        configure([good_entry("bad", command=str(broken_script), args=[]), good_entry("demo2")])
        mixed = await discover()
        check("降级: 一个 server 挂掉 -> **不抛异常**，其余 server 的工具照常可用",
              set(mixed) == {"mcp__demo2__echo", "mcp__demo2__no_desc"}, sorted(mixed))
        check("降级: 挂掉的 server 不进 connected（状态里看得到）",
              [s["server"] for s in mcpmod.get_mcp_status()["connected"]] == ["demo2"],
              mcpmod.get_mcp_status()["connected"])

        configure([good_entry("dead", command=str(workdir / "不存在"))])
        check("降级: 连**不存在**的可执行文件也不抛（返回空工具集）",
              (await discover()) == {})

        # ================= test_mcp_server =================
        check("测试: 缺 name / name 非字符串 -> ok=false（不是异常）",
              await mcpmod.test_mcp_server({}) == {"ok": False, "error": "server name is required"}
              and await mcpmod.test_mcp_server({"name": 123})
              == {"ok": False, "error": "server name is required"})
        tested = await mcpmod.test_mcp_server(good_entry())
        check("测试: 连接成功 -> ok=true + transport/toolCount/tools（重名**不去重**，原样列出）",
              tested["ok"] is True and tested["server"] == "demo"
              and tested["transport"] == "stdio"
              and tested["tools"] == ["echo", "no.desc", "echo"],
              tested)
        failed = await mcpmod.test_mcp_server({"name": "x", "command": str(broken_script)})
        check("测试: 连接失败 -> ok=false + error（**测试连接不写缓存**）",
              failed["ok"] is False and failed["server"] == "x" and failed["error"],
              failed)
        sse = await mcpmod.test_mcp_server(
            {"name": "s", "transport": "sse", "url": "http://x"})
        check("测试: sse 传输**未移植** -> ok=false 且说明原因（不静默降级成 http）",
              sse["ok"] is False and "未移植" in sse["error"], sse)

    async def run_phase() -> None:
        await async_checks()
        # 收尾：关掉所有子进程连接，并给事件循环一次机会回收失败分支残留的传输对象
        # （否则循环关闭后 GC 会打印 `RuntimeError: Event loop is closed` 噪音）
        await mcpmod._close_all_servers()  # noqa: SLF001 —— 测试刻意收尾
        await asyncio.sleep(0.05)

    asyncio.run(run_phase())

    # ================= 路由 =================
    client = TestClient(app)
    configure([good_entry()])
    status_resp = client.get("/api/v1/mcp/status")
    check("路由: GET /status 是标准成功信封（code/data/message）",
          status_resp.status_code == 200
          and status_resp.json()["code"] == 200
          and status_resp.json()["data"]["configured"] == ["demo"],
          status_resp.json().get("data"))
    refreshed = client.post("/api/v1/mcp/refresh")
    check("路由: POST /refresh 返回**刷新后的状态**（connected 已含 demo）",
          refreshed.status_code == 200
          and refreshed.json()["data"]["connected"][0]["server"] == "demo",
          refreshed.json().get("data"))
    no_name = client.post("/api/v1/mcp/test", json={})
    check("路由: POST /test 缺 name -> 400 `server name is required`",
          no_name.status_code == 400
          and no_name.json() == {"code": 400, "message": "server name is required"},
          no_name.json())
    ok_test = client.post("/api/v1/mcp/test", json=good_entry())
    check("路由: POST /test 成功 -> 200 成功信封包住 {ok: true, ...}",
          ok_test.status_code == 200 and ok_test.json()["data"]["ok"] is True,
          ok_test.json().get("data"))
    bad_json = client.post("/api/v1/mcp/test", content=b"{oops",
                           headers={"Content-Type": "application/json"})
    check("路由: POST /test 坏 JSON -> **500 且带 data:null**（与别的域的信封不同）",
          bad_json.status_code == 500
          and set(bad_json.json()) == {"code", "data", "message"}
          and bad_json.json()["code"] == 500 and bad_json.json()["data"] is None,
          (bad_json.status_code, bad_json.json()))
    not_truthy = client.post("/api/v1/mcp/test", json={"name": 123})
    check("路由: POST /test 里 name 只判 truthy -> 123 放行到业务层（200，ok=false）",
          not_truthy.status_code == 200
          and not_truthy.json()["data"] == {"ok": False, "error": "server name is required"},
          not_truthy.json().get("data"))

    # ================= 工具查找回归（id 兜底）=================
    camel = Tool(id="snake_id", description="d", input_schema={}, execute=lambda a: None)
    registry = ToolRegistry({"camelKey": camel})
    check("回归: ToolRegistry.get 按 **tool.id** 也能取到（键与 id 不一致时不静默失败）",
          registry.get("snake_id") is camel, registry.get("snake_id"))

    # ================= 汇总 =================
    os.environ.pop("MCP_SERVERS_PATH", None)
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
