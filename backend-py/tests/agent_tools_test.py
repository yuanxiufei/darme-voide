"""S5 工具层自检：宫格工具集（``agents/tools/grid_prompt_tools.py``，首个工具集）。

工具层最容易出的错是「**agent 路径与路由 fallback 悄悄分叉**」—— 同一个宫格 prompt，
Agent 生成的与路由兜底生成的必须**逐字相同**（画风走同一解析链、参考图走同一收集器）。
本测试的核心断言就是这条不变量。

另外两处细节：
* 返回的 ``reference_assets`` 是 **snake_case**（``image_label``），不是 TS 的 camelCase；
* ``shot_schema`` 里**只有 shot_number 必填**（其余对应 zod 的 ``.optional()``）。

运行::

    ./.venv/Scripts/python.exe tests/agent_tools_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="agenttools_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import scenes, storyboards  # noqa: E402
from app.core.response import now  # noqa: E402
from app.agent.tools.grid_prompt_tools import create_grid_prompt_tools  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _call(tools: dict, tool_id: str, arguments: dict | None = None):
    return asyncio.run(tools[tool_id].run(arguments))


def main() -> int:  # noqa: C901
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE agent 工具"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(scenes.update().where(scenes.c.id == scene_id)
                     .values(image_url="static/images/s.png"))

    def new_sb(number: int, **fields) -> int:
        sid = client.post("/api/v1/storyboards", json={
            "episode_id": episode_id, "title": f"镜头{number}", "storyboard_number": number,
        }).json()["data"]["id"]
        if fields:
            with engine.begin() as conn:
                conn.execute(storyboards.update().where(storyboards.c.id == sid).values(**fields))
        return sid

    # sb1 有 description；sb2 只有 title（验 `description || title || ''` 的回落链）
    sb1 = new_sb(1, scene_id=scene_id, description="少年入门", movement="推镜",
                 first_frame_image="static/images/f1.png")
    sb2 = new_sb(2, scene_id=scene_id, first_frame_image="static/images/f2.png")

    # 工厂只注入 id（工具自开短事务），所以这里不需要持有连接
    tools = create_grid_prompt_tools(episode_id, drama_id)

    # ================= 工具形状 =================
    check("工具: 工厂产出 2 个工具，键即工具 id",
          set(tools) == {"read_shots_for_grid", "generate_grid_prompt"}
          and all(key == tool.id for key, tool in tools.items()),
          sorted(tools))
    read_schema = tools["read_shots_for_grid"].input_schema
    gen_schema = tools["generate_grid_prompt"].input_schema
    check("工具: read 的入参 shot_ids 必填且是数字数组",
          read_schema["required"] == ["shot_ids"]
          and read_schema["properties"]["shot_ids"]["items"]["type"] == "number",
          read_schema)
    check("工具: generate 四参必填",
          gen_schema["required"] == ["shots", "rows", "cols", "mode"], gen_schema["required"])
    check(
        "工具: 逐镜 schema 里**只有 shot_number 必填**（其余对应 zod .optional()）",
        gen_schema["properties"]["shots"]["items"]["required"] == ["shot_number"],
        gen_schema["properties"]["shots"]["items"]["required"],
    )

    # ================= read_shots_for_grid =================
    check("读取: 不传 shot_ids -> 空列表（**不去查库**）",
          _call(tools, "read_shots_for_grid", {}) == {"shots": []}
          and _call(tools, "read_shots_for_grid", {"shot_ids": []}) == {"shots": []})

    read = _call(tools, "read_shots_for_grid", {"shot_ids": [sb1, sb2]})
    shots = read["shots"]
    check("读取: 按传入 id 过滤、按镜号排序", [s["shot_number"] for s in shots] == [1, 2], shots)
    check(
        "读取: description 回落 title（sb2 只有 title）",
        shots[0]["description"] == "少年入门" and shots[1]["description"] == "镜头2",
        [s["description"] for s in shots],
    )
    check(
        "读取: 除 shot_number（数字）外，缺失字段一律回落空串（不是 None）",
        all(isinstance(value, str) for shot in shots for key, value in shot.items()
            if key != "shot_number")
        and shots[1]["dialogue"] == "" and shots[0]["movement"] == "推镜",
        shots[1],
    )
    check("读取: 只读本集镜头（传别的集 id 也拿不到）",
          _call(tools, "read_shots_for_grid", {"shot_ids": [999999]}) == {"shots": []})

    # ================= generate_grid_prompt =================
    empty = _call(tools, "generate_grid_prompt", {"shots": [], "rows": 1, "cols": 2, "mode": "x"})
    check("生成: 没有 shots -> 带 error 的空结果（不抛错）",
          empty == {"error": "No shots provided", "grid_prompt": "", "cell_prompts": []}, empty)

    generated = _call(tools, "generate_grid_prompt", {
        "shots": [{"shot_number": 1}, {"shot_number": 2}],
        "rows": 1, "cols": 2, "mode": "first_frame"})
    check(
        "生成: 返回三件套（grid_prompt / cell_prompts / reference_assets）",
        set(generated) == {"grid_prompt", "cell_prompts", "reference_assets"}
        and generated["grid_prompt"].startswith("1x2 grid layout"),
        sorted(generated),
    )
    check(
        "生成: reference_assets 是 **snake_case**（image_label/path/label）",
        generated["reference_assets"]
        and set(generated["reference_assets"][0]) == {"image_label", "path", "label"},
        generated["reference_assets"][:1],
    )
    # ⭐ 核心不变量：agent 路径 == 路由 fallback 路径（逐字相同）
    with engine.begin() as conn:
        rows = conn.execute(
            storyboards.select().where(storyboards.c.id.in_([sb1, sb2]))
            .order_by(storyboards.c.storyboard_number)
        ).all()
        style = pu.resolve_effective_art_style(conn, drama_id)
        assets = pu.collect_grid_reference_assets(conn, rows)
        expected_prompt = pu.build_grid_prompt(conn, "first_frame", rows, 1, 2, style, assets)
        expected_cells = pu.build_grid_cell_prompts(conn, "first_frame", rows, 1, 2, assets)
    check(
        "生成: **agent 路径与路由 fallback 逐字一致**（grid_prompt）",
        generated["grid_prompt"] == expected_prompt, generated["grid_prompt"][:60],
    )
    check(
        "生成: 逐格 prompt 也一致 + 格数 = rows×cols（本例 2 格）",
        generated["cell_prompts"] == expected_cells and len(generated["cell_prompts"]) == 2,
        len(generated["cell_prompts"]),
    )
    multi = _call(tools, "generate_grid_prompt", {
        "shots": [{"shot_number": 1}], "rows": 2, "cols": 2, "mode": "multi_ref"})
    check("生成: multi_ref 模式下格数按 rows×cols（4 格）",
          len(multi["cell_prompts"]) == 4, len(multi["cell_prompts"]))

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
