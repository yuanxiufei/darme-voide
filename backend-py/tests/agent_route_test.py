"""S5 ④ 自检：Agent 聊天域（``routers/agent.py`` ← ``routes/agent.ts`` 57 行，2 端点）。

本域很短，但**文案与顺序**上有五处「改了就静默错」的细节：两处非法类型的文案不同
（``chat`` 带类型名、``debug`` 不带）、校验顺序（start 日志在校验 id **之前**）、
缺 id 用 falsy 判定（``0`` 也算缺）、运行期异常一律 **400 而非 500**、``usage`` 是
camelCase 或 null。

运行时被替换成假的（`run_agent_with_retry` 打桩）⇒ 本测试**不打网络、不碰 LLM**。

运行::

    ./.venv/Scripts/python.exe tests/agent_route_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="agentroute_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.routers import agent as agent_route  # noqa: E402
from app.services.agent_registry import VALID_AGENT_TYPES  # noqa: E402
from app.agent.runtime import AgentRunResult, TokenUsage  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_CALLS: list[dict] = []


def _stub(result: object) -> None:
    """把运行时换掉：记录入参 + 返回预设结果（或抛错）。"""

    async def _fake(conn, agent_type, episode_id, drama_id, message, options=None, **kwargs):
        _CALLS.append({
            "type": agent_type, "episodeId": episode_id, "dramaId": drama_id,
            "message": message, "options": options,
        })
        if isinstance(result, BaseException):
            raise result
        return result

    agent_route.run_agent_with_retry = _fake  # type: ignore[assignment]


def _run_result(**overrides) -> AgentRunResult:
    data = {
        "model": "m1",
        "text": "完成",
        "tool_calls": [{"toolName": "save_script", "args": {"x": 1}}],
        "tool_results": [{"toolName": "save_script", "result": "ok"}],
        "protocol": {"status": "ok", "summary": "已保存"},
        "protocol_errors": [],
        "usage": TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
    }
    data.update(overrides)
    return AgentRunResult(**data)  # type: ignore[arg-type]


def main() -> int:  # noqa: C901
    client = TestClient(app)

    # ================= debug =================
    check("debug: 六个合法类型都 valid（且不与 agent_configs 前缀撞车）",
          all(client.get(f"/api/v1/agent/{t}/debug").json()["data"]
              == {"agent_type": t, "valid": True} for t in VALID_AGENT_TYPES),
          list(VALID_AGENT_TYPES))
    dbg = client.get("/api/v1/agent/nope/debug")
    check("debug: 非法类型 -> 400，文案**不带**类型名",
          dbg.status_code == 400 and dbg.json() == {"code": 400, "message": "Invalid agent type"},
          (dbg.status_code, dbg.json()))

    # ================= chat: 类型与必填 =================
    bad = client.post("/api/v1/agent/nope/chat", json={"message": "hi"})
    check("chat: 非法类型 -> 400，文案**带**类型名（与 debug 不同）",
          bad.status_code == 400
          and bad.json() == {"code": 400, "message": "Invalid agent type: nope"},
          (bad.status_code, bad.json()))

    missing = client.post("/api/v1/agent/voice_assigner/chat", json={"message": "hi"})
    check("chat: 缺 drama_id + episode_id -> 400",
          missing.status_code == 400
          and missing.json()["message"] == "drama_id and episode_id are required",
          missing.json())
    half = client.post("/api/v1/agent/voice_assigner/chat",
                       json={"message": "hi", "drama_id": 1})
    check("chat: 只给一个 id 也算缺（两个都必填）",
          half.status_code == 400
          and half.json()["message"] == "drama_id and episode_id are required", half.json())
    zero = client.post("/api/v1/agent/voice_assigner/chat",
                       json={"message": "hi", "drama_id": 1, "episode_id": 0})
    check("chat: **falsy 判定** —— episode_id=0 也当缺（不是只判 None）",
          zero.status_code == 400, zero.json())
    broken = client.post("/api/v1/agent/voice_assigner/chat",
                         content=b"{not json", headers={"Content-Type": "application/json"})
    check("chat: 坏 JSON -> 走空 body -> 400（**不是 500**）",
          broken.status_code == 400
          and broken.json()["message"] == "drama_id and episode_id are required",
          (broken.status_code, broken.json()))

    # ================= chat: 成功 =================
    _CALLS.clear()
    _stub(_run_result())
    ok = client.post("/api/v1/agent/voice_assigner/chat", json={
        "message": "给角色配音", "drama_id": 7, "episode_id": 9})
    body = ok.json()
    check("chat: 成功信封 `{code, data, message}` 且 data.type 固定 done",
          ok.status_code == 200 and body["code"] == 200 and body["message"] == "success"
          and set(body["data"]) == {"type", "text", "toolCalls", "toolResults", "usage"}
          and body["data"]["type"] == "done",
          sorted(body.get("data") or {}))
    check("chat: 回执把入参透传给运行时（type / episode / drama / message）",
          _CALLS == [{"type": "voice_assigner", "episodeId": 9, "dramaId": 7,
                      "message": "给角色配音", "options": {"maxSteps": 20}}],
          _CALLS)
    check("chat: toolCalls / toolResults 原样传出（camelCase 键）",
          body["data"]["toolCalls"] == [{"toolName": "save_script", "args": {"x": 1}}]
          and body["data"]["toolResults"] == [{"toolName": "save_script", "result": "ok"}],
          body["data"]["toolCalls"])
    check("chat: usage 是 **camelCase**（inputTokens/outputTokens/totalTokens）",
          body["data"]["usage"] == {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
          body["data"]["usage"])

    _CALLS.clear()
    _stub(_run_result(usage=None))
    bare = client.post("/api/v1/agent/voice_assigner/chat",
                       json={"message": "hi", "drama_id": 1, "episode_id": 1}).json()
    check("chat: 上游没给用量 -> usage 为 **null**（键仍在）",
          bare["data"]["usage"] is None and "usage" in bare["data"], bare["data"]["usage"])

    # message 缺失（合法 id 齐备时）不该 400，而是空串
    _CALLS.clear()
    _stub(_run_result())
    no_msg = client.post("/api/v1/agent/voice_assigner/chat",
                         json={"drama_id": 1, "episode_id": 2})
    check("chat: 没传 message 也放行（透传空串，不是必填）",
          no_msg.status_code == 200 and _CALLS[0]["message"] == "", _CALLS)

    # ================= chat: 失败 =================
    _stub(RuntimeError("模型全挂"))
    err = client.post("/api/v1/agent/voice_assigner/chat",
                      json={"message": "hi", "drama_id": 1, "episode_id": 1})
    check("chat: 运行期异常 -> **400**（不是 500），文案取 str(err)",
          err.status_code == 400
          and err.json() == {"code": 400, "message": "模型全挂"},
          (err.status_code, err.json()))

    _stub(RuntimeError(""))
    blank = client.post("/api/v1/agent/voice_assigner/chat",
                        json={"message": "hi", "drama_id": 1, "episode_id": 1})
    check("chat: err 文案为空 -> 兜底 `Agent execution failed`",
          blank.json()["message"] == "Agent execution failed", blank.json())

    _stub(RuntimeError("Invalid agent type: x"))
    vt = client.post("/api/v1/agent/orchestrator/chat",
                     json={"message": "hi", "drama_id": 1, "episode_id": 1})
    check("chat: orchestrator 是**合法**类型（放行到运行时，不在路由层拦）",
          vt.json()["message"] == "Invalid agent type: x", vt.json())

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok_, detail in _RESULTS:
        print(("PASS  " if ok_ else "FAIL  ") + name + ("" if ok_ else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
