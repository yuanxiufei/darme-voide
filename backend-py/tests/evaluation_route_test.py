"""S6 自检：评测执行器 + 评测域前 2 个端点（``evaluation/evaluator.ts`` 210 行 + ``routes/evaluation.ts``）。

评测执行器是「**seed → 跑 Agent → 抽取工具入参 → 打分 → 物理清理**」这条闭环。它有三处
**错了不报错、只是结果不可信**的地方：

1. **seed 必须先提交**：Agent 的工具开着**自己的**连接读库（SQLite 单写者）。若 seed 攒在
   请求事务里，Agent 看不到临时的剧本/角色/场景 ⇒ 评分变成「对空气评分」；
2. **抽取口径各不相同**：分镜/剧本取**最后一次**调用，角色/场景/音色**跨调用累加**；
3. **cleanup 必须删干净**：否则每评测一次就往库里留一套 `[bench] xxx` 剧组。

外加两条端点契约：`/cases` 无 try（坏 JSON ⇒ 500）、`/evaluate` 的三个失败分支状态码不同
（404 / 400 / 400），以及**未迁的 3 个端点仍走兜底 501**（验证没被新路由遮蔽）。

运行::

    ./.venv/Scripts/python.exe tests/evaluation_route_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="evalroute_"))
# 未迁端点要么 501（不反代）、要么反代到 Node；本测试只验前者
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (  # noqa: E402
    characters,
    dramas,
    episode_characters,
    episode_scenes,
    episodes,
    scenes,
    storyboard_characters,
    storyboards,
)
from app.core.response import now  # noqa: E402
from app.agent.evaluation import catalog as cat  # noqa: E402
from app.agent.evaluation import evaluator as ev  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


FAKE_MODEL = "bench-model-x"


class FakeRuntime:
    """替身：记录入参 + 在**独立连接**上确认 seed 已提交，再返回预设工具调用。"""

    def __init__(self, tool_calls: list[dict], *, raise_error: Exception | None = None) -> None:
        self.tool_calls = tool_calls
        self.raise_error = raise_error
        self.calls: list[dict] = []
        self.seed_visible: bool | None = None

    async def __call__(self, conn, agent_type, episode_id, drama_id, instructions, message, *a, **kw):
        self.calls.append({
            "agentType": agent_type, "episodeId": episode_id, "dramaId": drama_id,
            "instructions": instructions, "message": message,
        })
        # ⭐ 关键：用**另一个**连接查 seed 数据 —— 没提交的话这里必然查不到
        with engine.begin() as probe:
            row = probe.execute(
                select(episodes.c.id).where(episodes.c.id == episode_id)
            ).first()
        self.seed_visible = row is not None
        if self.raise_error is not None:
            raise self.raise_error

        class _Result:
            model = FAKE_MODEL
            text = "{}"

        result = _Result()
        result.tool_calls = self.tool_calls  # type: ignore[attr-defined]
        return result





def main() -> int:  # noqa: C901
    # ================= 工具名归一 =================
    check("归一: 驼峰/下划线/空格/大写 归一到同一形（否则抽取 0 命中）",
          ev.norm_tool_name("saveStoryboards") == "savestoryboards"
          and ev.norm_tool_name("save_storyboards") == "savestoryboards"
          and ev.norm_tool_name("SAVE STORYBOARDS") == "savestoryboards"
          and ev.norm_tool_name(None) == "",
          [ev.norm_tool_name(x) for x in ("saveStoryboards", "save_storyboards", None)])

    # ================= 抽取口径 =================
    calls = [
        {"toolName": "save_storyboards", "args": {"storyboards": [{"shot_number": 1}]}},
        {"toolName": "read_storyboard_context", "args": {}},
        {"toolName": "saveStoryboards", "args": {"storyboards": [{"shot_number": 9}]}},
    ]
    check("抽取: 分镜只看 save_storyboards 且取**最后一次**",
          ev.extract_storyboards(calls) == [{"shot_number": 9}],
          ev.extract_storyboards(calls))
    check("抽取: 分镜 args 缺失/为空 -> 空列表（不抛）",
          ev.extract_storyboards([{"toolName": "save_storyboards", "args": {}}]) == []
          and ev.extract_storyboards([]) == [])
    dup_calls = [
        {"toolName": "save_dedup_characters", "args": {"characters": [{"name": "甲"}]}},
        {"toolName": "saveDedupCharacters", "args": {"characters": [{"name": "乙"}]}},
        {"toolName": "save_dedup_scenes", "args": {"scenes": [{"location": "客栈"}]}},
        {"toolName": "save_dedup_scenes", "args": {"scenes": [{"location": "长街"}]}},
    ]
    check("抽取: 角色/场景 **跨调用累加**（与分镜的「取最后一次」不同）",
          [c["name"] for c in ev.extract_characters(dup_calls)] == ["甲", "乙"]
          and [s["location"] for s in ev.extract_scenes(dup_calls)] == ["客栈", "长街"],
          (ev.extract_characters(dup_calls), ev.extract_scenes(dup_calls)))
    check("抽取: 剧本取最后一次 save_script；音色取三个字段并累加",
          ev.extract_script([
              {"toolName": "save_script", "args": {"content": "旧"}},
              {"toolName": "saveScript", "args": {"content": "新"}},
          ]) == "新"
          and ev.extract_voice_assignments([
              {"toolName": "assign_voice", "args": {"character_id": 1, "voice_id": "v1",
                                                   "reason": "r", "extra": "x"}},
          ]) == [{"character_id": 1, "voice_id": "v1", "reason": "r"}],
          ev.extract_voice_assignments([{"toolName": "assignVoice", "args": {}}]))

    # ================= seed / cleanup =================
    seed = ev.seed_case(case_id="unit-case", script="剧本", content="原文",
                        characters_in=[{"name": "林昭", "role": "主角", "appearance": "青衫"}],
                        scenes_in=[{"location": "客栈", "time": "夜", "prompt": "烛火"}])
    with engine.begin() as conn:
        drama = conn.execute(select(dramas).where(dramas.c.id == seed["dramaId"])).first()
        episode = conn.execute(select(episodes).where(episodes.c.id == seed["episodeId"])).first()
        scene_row = conn.execute(select(scenes).where(scenes.c.drama_id == seed["dramaId"])).first()
        char_row = conn.execute(select(characters)
                                .where(characters.c.drama_id == seed["dramaId"])).first()
        links = (
            len(conn.execute(select(episode_characters.c.character_id)).all()),
            len(conn.execute(select(episode_scenes.c.scene_id)).all()),
        )
    check("seed: drama 是 `[bench] <caseId>` + draft；episode 是第 1 集「评测集」",
          drama.title == "[bench] unit-case" and drama.status == "draft"
          and episode.episode_number == 1 and episode.title == "评测集"
          and episode.script_content == "剧本" and episode.content == "原文",
          (drama.title, episode.title))
    check("seed: 场景同时挂 drama_id 与 episode_id，且角色/场景都建了关联",
          scene_row.drama_id == seed["dramaId"] and scene_row.episode_id == seed["episodeId"]
          and char_row is not None and links[0] >= 1 and links[1] >= 1,
          (scene_row.drama_id, scene_row.episode_id, links))
    check("seed: legal 集合就是本 case 造出来的 id",
          seed["legal"]["characterIds"] == {char_row.id}
          and seed["legal"]["sceneIds"] == {scene_row.id})

    # 模拟 Agent 跑出来的分镜 + 关联，验证 cleanup 也会清掉
    with engine.begin() as conn:
        sb_id = int(conn.execute(storyboards.insert().values(
            episode_id=seed["episodeId"], storyboard_number=1, title="临时",
            created_at=now(), updated_at=now(),
        )).lastrowid)
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_id, character_id=char_row.id))
    ev.cleanup_case(seed)
    with engine.begin() as conn:
        leftover = (
            len(conn.execute(select(dramas.c.id).where(dramas.c.id == seed["dramaId"])).all()),
            len(conn.execute(select(episodes.c.id)
                             .where(episodes.c.id == seed["episodeId"])).all()),
            len(conn.execute(select(scenes.c.id)
                             .where(scenes.c.drama_id == seed["dramaId"])).all()),
            len(conn.execute(select(characters.c.id)
                             .where(characters.c.drama_id == seed["dramaId"])).all()),
            len(conn.execute(select(storyboards.c.id)
                             .where(storyboards.c.episode_id == seed["episodeId"])).all()),
            len(conn.execute(select(storyboard_characters.c.storyboard_id)).all()),
        )
    check("cleanup: drama/episode/角色/场景/分镜/分镜角色关联 **全部清干净**",
          leftover == (0, 0, 0, 0, 0, 0), leftover)
    ev.cleanup_case(seed)  # 再删一次（数据已不在）
    check("cleanup: 对已删除的数据重复调用**不抛**（失败只告警）", True)

    # ================= evaluate_case（四种 kind + 事务边界 + 清理）=================
    storyboard_calls = [
        {"toolName": "read_storyboard_context", "args": {}},
        {"toolName": "save_storyboards", "args": {"storyboards": [
            {"shot_number": 1, "title": "开场", "action": "推门", "duration": 8,
             "video_prompt": "<location>客栈"},
            {"shot_number": 2, "title": "反应", "action": "回头", "duration": 7,
             "video_prompt": "<location>客栈"},
        ]}},
    ]
    fake = FakeRuntime(storyboard_calls)
    ev.run_agent_with_instructions = fake  # type: ignore[assignment]
    case_def = cat.load_case_by_id("sb-case-001")
    report = asyncio.run(ev.evaluate_case(conn=None, case_def=case_def, instructions="候选"))  # type: ignore[arg-type]
    check("执行: 分镜 case 走 storyboard_breaker + 逐字消息",
          fake.calls[0]["agentType"] == "storyboard_breaker"
          and fake.calls[0]["message"] == ev.STORYBOARD_MESSAGE
          and fake.calls[0]["instructions"] == "候选",
          fake.calls[0])
    check("执行: **seed 在 Agent 跑之前就已提交**（独立连接查得到 episode）",
          fake.seed_visible is True)
    check("执行: 报告填了 caseId 与 runtimeModel，kind 为 storyboard",
          report is not None and report["caseId"] == "sb-case-001"
          and report["runtimeModel"] == FAKE_MODEL and report["kind"] == "storyboard",
          report)
    # ⚠️ 用**案例自己的 rubric** 推期望值，别硬编（真实 case 的镜头数区间是 4~8 ⇒ 2 镜是 5 分）
    rubric = case_def["rubric"]
    in_range = rubric["minShots"] <= 2 <= rubric["maxShots"]
    check("执行: 打的是**真实 rubric**（镜头数维度按该案例区间给 10 或 5）",
          report["dimensions"][0]["score"] == (10 if in_range else 5)
          and str(rubric["minShots"]) in report["dimensions"][0]["detail"],
          report["dimensions"][0])
    with engine.begin() as conn:
        leaked = len(conn.execute(select(dramas.c.id)
                                  .where(dramas.c.title.like("[bench]%"))).all())
    check("执行: 跑完 **临时剧组已物理删除**（库里不留 [bench] 数据）", leaked == 0, leaked)

    kinds = [
        ("ext-case-001", "extractor", ev.EXTRACTOR_MESSAGE,
         [{"toolName": "save_dedup_characters", "args": {"characters": [{"name": "甲"}]}}]),
        ("script-rewriter-001", "script_rewriter", ev.SCRIPT_REWRITER_MESSAGE,
         [{"toolName": "save_script", "args": {"content": "## S01 | 内景 · 客栈 | 夜\n" + "字" * 120}}]),
        ("voice-assigner-001", "voice_assigner", ev.VOICE_ASSIGNER_MESSAGE,
         [{"toolName": "assign_voice", "args": {"character_id": 1, "voice_id": "v1",
                                                "reason": "少年感"}}]),
    ]
    for case_id, expected_type, expected_message, tool_calls in kinds:
        case = cat.load_case_by_id(case_id)
        fake_kind = FakeRuntime(tool_calls)
        ev.run_agent_with_instructions = fake_kind  # type: ignore[assignment]
        out = asyncio.run(ev.evaluate_case(conn=None, case_def=case, instructions="p"))  # type: ignore[arg-type]
        check(f"执行: {expected_type} case 用对了 Agent/消息，且报告带 caseId",
              fake_kind.calls[0]["agentType"] == expected_type
              and fake_kind.calls[0]["message"] == expected_message
              and out["caseId"] == case_id and out["kind"] == case["kind"]
              and fake_kind.seed_visible is True,
              (fake_kind.calls[0]["agentType"], out["caseId"]))
    with engine.begin() as conn:
        leaked2 = len(conn.execute(select(dramas.c.id)
                                   .where(dramas.c.title.like("[bench]%"))).all())
    check("执行: 四个 kind 跑完后**没有任何残留**剧组", leaked2 == 0, leaked2)

    unknown = asyncio.run(ev.evaluate_case(
        conn=None, case_def={"id": "x", "kind": "没有这种", "statement": {}, "rubric": {}},  # type: ignore[arg-type]
        instructions="p"))
    check("执行: 未知 kind -> None（TS 的 switch 没 default）", unknown is None)

    # 异常路径：Agent 抛错也要清理（finally）
    failing = FakeRuntime([], raise_error=RuntimeError("模型全挂"))
    ev.run_agent_with_instructions = failing  # type: ignore[assignment]
    raised = ""
    try:
        asyncio.run(ev.evaluate_case(conn=None, case_def=cat.load_case_by_id("sb-case-001"),  # type: ignore[arg-type]
                                     instructions="p"))
    except Exception as err:  # noqa: BLE001
        raised = str(err)
    with engine.begin() as conn:
        leaked3 = len(conn.execute(select(dramas.c.id)
                                   .where(dramas.c.title.like("[bench]%"))).all())
    check("执行: Agent 抛错时 **finally 仍然清理干净**（否则每失败一次留一套剧组）",
          raised == "模型全挂" and leaked3 == 0, (raised, leaked3))

    # ================= 端点 =================
    client = TestClient(app)
    cases = client.get("/api/v1/evaluation/cases")
    check("端点: GET /cases -> 200 成功信封 + 四个 case 元信息",
          cases.status_code == 200 and len(cases.json()["data"]) == 4
          and all(set(m) == {"id", "kind", "agentType"} for m in cases.json()["data"]),
          cases.json().get("data"))

    missing = client.post("/api/v1/evaluation/evaluate/不存在")
    check("端点: case 不存在 -> **404** + `未知基准 case：x`（全角冒号）",
          missing.status_code == 404
          and missing.json()["message"] == "未知基准 case：不存在",
          (missing.status_code, missing.json()))

    # ⚠️ 「未知 case kind -> 400」那支**实际不可达**：catalog 扫描时就把 AGENT_BY_KIND 里没有的
    #    kind 过滤掉了（所以那种文件连 loadCaseById 都取不到）⇒ 这里只能验到 404。
    unknown_file = client.post("/api/v1/evaluation/evaluate/kind-unknown")
    check("端点: 未知 case 一律 404（`未知 case kind` 的 400 分支不可达，与 TS 同）",
          unknown_file.status_code == 404, unknown_file.status_code)

    ev.run_agent_with_instructions = FakeRuntime([  # type: ignore[assignment]
        {"toolName": "save_storyboards", "args": {"storyboards": [{"title": "t"}]}}])
    ok = client.post("/api/v1/evaluation/evaluate/sb-case-001")
    check("端点: 评测成功 -> 200 成功信封，data 含 caseId/kind/dimensions/total/runtimeModel",
          ok.status_code == 200
          and set(ok.json()["data"]) >= {"caseId", "kind", "dimensions", "total", "runtimeModel"}
          and ok.json()["data"]["caseId"] == "sb-case-001",
          ok.json().get("data"))

    ev.run_agent_with_instructions = FakeRuntime([], raise_error=ValueError("没有可用模型"))  # type: ignore[assignment]
    failed = client.post("/api/v1/evaluation/evaluate/sb-case-001")
    check("端点: 运行期异常 -> **400**（不是 500），message 取 str(err)",
          failed.status_code == 400 and failed.json()["message"] == "没有可用模型",
          (failed.status_code, failed.json()))

    # 2026-09-13：评测域 **5/5 端点全部迁完**（optimize/scheduler/run 已本地化，见
    # optimizer_test.py 与 evaluation_scheduler_test.py）⇒ 这里断言整域**不再有**兜底 501。
    domain_paths = {
        "GET /cases": client.get("/api/v1/evaluation/cases"),
        "POST /evaluate/x": client.post("/api/v1/evaluation/evaluate/x"),
        "POST /optimize/x": client.post("/api/v1/evaluation/optimize/x"),
        "GET /scheduler": client.get("/api/v1/evaluation/scheduler"),
        "POST /run": client.post("/api/v1/evaluation/run"),
    }
    check("端点: 评测域 **5/5 已迁**（没有任何一条落到 501 兜底）",
          all(r.status_code != 501 for r in domain_paths.values()),
          {k: v.status_code for k, v in domain_paths.items()})

    # 坏 JSON 的基准目录：/cases 无 try ⇒ 500（保真）
    broken_dir = Path(tempfile.mkdtemp(prefix="bench_bad_"))
    (broken_dir / "bad.json").write_text("{坏", encoding="utf-8")
    os.environ["BENCHMARKS_DIR"] = str(broken_dir)
    try:
        client.get("/api/v1/evaluation/cases")
        status = 200
    except Exception:  # noqa: BLE001 —— TestClient 默认 raise_server_exceptions=True
        status = 500
    os.environ.pop("BENCHMARKS_DIR", None)
    check("端点: GET /cases **无 try** —— 基准目录坏 JSON 直接 500（不是 400，与 TS 一致）",
          status == 500, status)

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_, detail in _RESULTS:
        print(("PASS  " if ok_ else "FAIL  ") + name + ("" if ok_ else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
