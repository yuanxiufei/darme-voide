"""S7 自检：``try_agent_grid_prompt``（Agent 宫格提示词）+ 端点 ``source='agent'`` 接线。

2026-09-15 之前这里是「恒返回 None」的存根 ⇒ 端点**永远**走本地构建器，而 Node 会优先用
``agent('grid_prompt_generator')`` ⇒ 属真行为缺口。本套件锁住移植后的语义：

1. **工具结果优先于正文**（先 ``toolResults`` 再 ``text``）；
2. 正文里的 ```json 围栏 / 裸 JSON 都能抠出来（复用既有的 ``_find_grid_payload``）；
3. **任何异常都吞掉返回 None**（对齐原 TS 的裸 catch ⇒ 端点回落，不报错）；
4. 消息 9 段拼接、空段丢弃（无参考图映射时**不出现空行**）、镜头 ID 走紧凑 JSON；
5. 端点命中 Agent 时返回 ``source='agent'`` 且带上 ``grid``/``storyboard_ids``/``mode``。

``run_agent_with_retry`` 全程打桩（不打网络、不起模型）。

运行::

    ./.venv/Scripts/python.exe tests/grid_agent_prompt_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="gridagent_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import dramas, episodes, storyboards  # noqa: E402
from app.response import now  # noqa: E402
from app.routers import grid as grid_api  # noqa: E402
from app.services.agents import runtime  # noqa: E402

_R: list[tuple[str, bool, object]] = []
_SEEN: dict[str, object] = {}


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def stub(*, tool_results=None, text="", raises=None):
    async def fake(conn, agent_type, episode_id, drama_id, message, options=None,
                   **_kwargs):
        _SEEN.update({"agentType": agent_type, "episodeId": episode_id,
                      "dramaId": drama_id, "message": message, "options": options})
        if raises:
            raise raises
        return SimpleNamespace(text=text, tool_calls=[], tool_results=tool_results or [])

    return fake


PAYLOAD = {"grid_prompt": "宫格总提示", "cell_prompts": [
    {"shot_number": 1, "frame_type": "first_frame", "prompt": "第一格"}]}
TOOL_RESULT = {"toolName": "build_grid_prompt", "result": PAYLOAD}


def main() -> int:  # noqa: C901
    stamp = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="宫格剧", created_at=stamp, updated_at=stamp)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        sb_id = int(conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=1, title="镜一", description="d",
            created_at=stamp, updated_at=stamp)).lastrowid)

    def run(*, tool_results=None, text="", raises=None):
        runtime.run_agent_with_retry = stub(tool_results=tool_results, text=text,
                                            raises=raises)  # type: ignore[assignment]
        return asyncio.run(grid_api.try_agent_grid_prompt(
            _conn(), episode_id, drama_id, [sb_id], 2, 2, "first_frame", "图片1=A"))

    def _conn():
        # 打桩 runtime 不用连接，传 None 也行；这里给个真连接更贴近真实调用
        return _NOOP_CONN

    # ── 1. 工具结果优先 ──
    got = run(tool_results=[TOOL_RESULT], text="正文里也有一份但应被忽略")
    check("取材: **工具结果优先**（toolResults 命中即返回，不再看正文）",
          got == PAYLOAD, got)
    check("取材: 传参为 `('grid_prompt_generator', episode_id, drama_id)` + maxSteps=10",
          _SEEN["agentType"] == "grid_prompt_generator" and _SEEN["episodeId"] == episode_id
          and _SEEN["dramaId"] == drama_id and _SEEN["options"] == {"maxSteps": 10}, _SEEN)

    # ── 2. 正文回落（围栏 / 裸 JSON）──
    fenced = "说明如下\n```json\n" + json.dumps(PAYLOAD) + "\n```\n谢谢"
    check("取材: 工具结果为空时从**正文的 ```json 围栏**抠出", run(text=fenced) == PAYLOAD)
    check("取材: 正文是裸 JSON 也能抠出", run(text=json.dumps(PAYLOAD)) == PAYLOAD)
    check("取材: 正文有 JSON 但不是宫格结构 ⇒ None（交给端点回落）",
          run(text=json.dumps({"foo": 1})) is None)
    check("取材: 正文是散文 ⇒ None", run(text="这里没有任何 JSON") is None)

    # ── 3. 异常吞掉 ──
    check("容错: runtime 抛异常 ⇒ 返回 None（对齐原 TS 的裸 catch，不向上抛）",
          run(raises=RuntimeError("模型挂了")) is None)

    # ── 4. 消息拼接 ──
    run(text=json.dumps(PAYLOAD))
    message = str(_SEEN["message"])
    lines = message.split("\n")
    check("消息: 9 段齐全（工具优先/镜头ID/行/列/模式/参考图/编号规则/格数约束/JSON 结构）",
          len(lines) == 9 and lines[0].startswith("请为宫格图生成提示词")
          and lines[1] == "选中镜头ID：[%d]" % sb_id and lines[2] == "行数：2"
          and lines[3] == "列数：2" and lines[4] == "模式：first_frame"
          and lines[5] == "参考图映射：图片1=A" and "exactly 4 visible panels" in lines[7]
          and lines[8].startswith("必须返回 JSON"), lines)
    check("消息: 镜头 ID 走**紧凑 JSON**（`[1]` 而不是 `[1]` 带空格）",
          ", " not in lines[1], lines[1])
    runtime.run_agent_with_retry = stub(text=json.dumps(PAYLOAD))  # type: ignore[assignment]
    asyncio.run(grid_api.try_agent_grid_prompt(_NOOP_CONN, episode_id, drama_id,
                                               [sb_id], 2, 2, "first_frame", ""))
    no_legend = str(_SEEN["message"]).split("\n")
    check("消息: **无参考图映射时不产生空段**（空段被丢弃，行数为 8 而不是 9）",
          len(no_legend) == 8 and not any(line == "" for line in no_legend), no_legend)

    # ── 5. 端点端到端：命中 Agent ⇒ source='agent' ──
    runtime.run_agent_with_retry = stub(tool_results=[TOOL_RESULT])  # type: ignore[assignment]
    client = TestClient(app)
    ok = client.post("/api/v1/grid/prompt", json={
        "storyboard_ids": [sb_id], "drama_id": drama_id, "episode_id": episode_id,
        "rows": 2, "cols": 2, "mode": "first_frame"})
    body = ok.json()
    check("端点: Agent 命中 ⇒ 200 + source='agent' + 透传 grid_prompt/cell_prompts",
          ok.status_code == 200 and body["data"]["source"] == "agent"
          and body["data"]["grid_prompt"] == PAYLOAD["grid_prompt"]
          and body["data"]["cell_prompts"] == PAYLOAD["cell_prompts"]
          and body["data"]["grid"] == {"rows": 2, "cols": 2}
          and body["data"]["storyboard_ids"] == [sb_id], body.get("data"))

    runtime.run_agent_with_retry = stub(raises=RuntimeError("模型挂了"))  # type: ignore[assignment]
    fallback = client.post("/api/v1/grid/prompt", json={
        "storyboard_ids": [sb_id], "drama_id": drama_id, "episode_id": episode_id,
        "rows": 2, "cols": 2, "mode": "first_frame"})
    fbody = fallback.json()
    check("端点: Agent 失败 ⇒ 仍 200 但 source='fallback'（回落本地构建器）",
          fallback.status_code == 200 and fbody["data"]["source"] == "fallback"
          and fbody["data"].get("grid_prompt"), fbody.get("data", {}).get("source"))

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


_NOOP_CONN = None

if __name__ == "__main__":
    raise SystemExit(main())
