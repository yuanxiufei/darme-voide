"""S5 工具层自检：提取工具集的**保存侧**（``agents/tools/extract_tools.py``，3 个 save 工具）。

保存侧的核心是**去重与合并**，错了不会报错、只会悄悄产生重复实体或丢字段：

* 角色：按**名字**去重；同名合并时 **agent 值 > 已有值**，但**保留已有 ID**；
  ``role_type`` 缺失时按 role 推断；``core_features``/``costumes`` 落库为**紧凑 JSON**；
  提示词兜底链 agent > 已有 > **自动构建**（不依赖 LLM 必填）；
* 场景：去重键 = **地点 + 时间段**（同地点**不同时段算新场景**）；
* 物品：按名字去重；新增时 ``category`` 默认「道具」、``keyClue`` 默认「否」、
  **``negativePrompt`` 兜底是空串**（不是构建出来的负面词）。

另含三条必需的不变量：全部**关联到当前集**、关联**幂等**（重复保存不产生重复关联）、
三条回执文案逐字对上。

运行::

    ./.venv/Scripts/python.exe tests/agent_extract_tools_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="extracttools_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (
    characters,
    episode_characters,
    episode_props,
    episode_scenes,
    prop_templates,
    scenes,
)  # noqa: E402
from app.agent.tools.extract_tools import create_extract_tools  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _call(tools: dict, tool_id: str, arguments: dict | None = None):
    return asyncio.run(tools[tool_id].run(arguments))


def main() -> int:  # noqa: C901
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 提取工具"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    tools = create_extract_tools(episode_id, drama_id)

    # ================= 角色：新增 =================
    first = _call(tools, "save_dedup_characters", {"characters": [{
        "name": "林昭", "role": "女主角", "appearance": "短发", "clothing": "青衫",
        "core_features": ["冷峻眉眼", "刀疤"], "personality": "隐忍",
    }]})
    check("角色: 新增回执（中文文案 + created=1）",
          first == {"message": "角色保存完成：新增 1，合并更新 0", "created": 1, "merged": 0},
          first)
    with engine.begin() as conn:
        row = conn.execute(select(characters).where(characters.c.name == "林昭")).first()
        links = conn.execute(select(episode_characters.c.character_id)
                             .where(episode_characters.c.episode_id == episode_id)).all()
    check("角色: role_type 由 role 推断（「女主角」-> 主角）", row.role_type == "主角", row.role_type)
    check("角色: core_features 落库为**紧凑 JSON**（共用列）",
          row.core_features == '["冷峻眉眼","刀疤"]', row.core_features)
    check("角色: 提示词兜底链生效（未给 image_prompt 时自动构建，非空）",
          bool(row.custom_prompt) and bool(row.negative_prompt),
          (row.custom_prompt or "")[:40])
    check("角色: 已关联到当前集", [link[0] for link in links] == [row.id], links)
    original_id = row.id

    # ================= 角色：同名合并 =================
    second = _call(tools, "save_dedup_characters", {"characters": [{
        "name": "林昭", "personality": "刚烈", "role_type": "反派",
    }]})
    with engine.begin() as conn:
        merged_row = conn.execute(select(characters).where(characters.c.name == "林昭")).first()
        count = len(conn.execute(select(characters.c.id)
                                 .where(characters.c.drama_id == drama_id)).all())
    check("角色: 同名 -> 合并而非新增（ID 不变，且库里仍只有 1 个）",
          second["merged"] == 1 and second["created"] == 0
          and merged_row.id == original_id and count == 1,
          (second, merged_row.id, count))
    check("角色: 合并时 **agent 值覆盖**、未给的字段回退已有值",
          merged_row.personality == "刚烈" and merged_row.clothing == "青衫"
          and merged_row.appearance == "短发",
          (merged_row.personality, merged_row.clothing))
    check("角色: 显式 role_type 优先于推断（给了「反派」就不看 role）",
          merged_row.role_type == "反派", merged_row.role_type)
    check("角色: 合并不会清空已有的 core_features",
          merged_row.core_features == '["冷峻眉眼","刀疤"]', merged_row.core_features)

    # ================= 场景：地点 + 时段 =================
    scenes_payload = [{"location": "客栈", "time": "夜晚", "atmosphere": "压抑"}]
    scene_first = _call(tools, "save_dedup_scenes", {"scenes": scenes_payload})
    check("场景: 新增回执文案",
          scene_first == {"message": "场景保存完成：新增 1，复用已有 0", "created": 1, "reused": 0},
          scene_first)
    with engine.begin() as conn:
        scene_row = conn.execute(select(scenes).where(scenes.c.location == "客栈")).first()
        scene_links = conn.execute(select(episode_scenes.c.scene_id)
                                   .where(episode_scenes.c.episode_id == episode_id)).all()
    check("场景: prompt 兜底为 **location**、time 默认空串",
          scene_row.prompt == "客栈" and scene_row.time == "夜晚", scene_row.prompt)
    check("场景: 负面词兜底是 SCENE_IMAGE_NEGATIVE（不是空串）",
          bool(scene_row.negative_prompt), (scene_row.negative_prompt or "")[:30])
    check("场景: 已关联到当前集", [link[0] for link in scene_links] == [scene_row.id])

    reused = _call(tools, "save_dedup_scenes", {"scenes": [
        {"location": "客栈", "time": "夜晚", "lighting": "烛光"}]})
    with engine.begin() as conn:
        scene_row2 = conn.execute(select(scenes).where(scenes.c.location == "客栈")).first()
        scene_count = len(conn.execute(select(scenes.c.id)
                                       .where(scenes.c.drama_id == drama_id)).all())
    check("场景: 同地点同时段 -> **复用**（不新增，且合并了新字段）",
          reused["reused"] == 1 and scene_count == 1 and scene_row2.lighting == "烛光",
          (reused, scene_count))

    new_time = _call(tools, "save_dedup_scenes", {"scenes": [{"location": "客栈", "time": "白天"}]})
    check("场景: **同地点不同时段算新场景**（这是去重键的核心规则）",
          new_time["created"] == 1, new_time)

    # 同义写法必须合并（2026-10-30 实测：一集重跑三次，同一地点长出 3 行 ——
    # 「民国上海街头 / 民国上海街道」、「夜晚 / 深夜」都是模型给法的漂移，不是不同场景）
    drift = _call(tools, "save_dedup_scenes", {"scenes": [
        {"location": "客栈大堂", "time": "深夜"}]})
    check("场景: 同义地点 + 同时段桶（客栈 / 客栈大堂、夜晚 / 深夜）-> **复用**",
          drift["reused"] == 1 and drift["created"] == 0, drift)

    # 场景小标题解析（供 extract 阶段清理陈旧关联）
    from app.services.scene_match import (  # noqa: PLC0415
        locations_match, normalize_time, script_scene_headings, same_scene)
    headings = script_scene_headings(
        "## S01 | 外景 · 民国上海街道 | 黄昏\n## 场景2：东家大厅 夜\n正文\n")
    check("场景表: 剧本小标题解析出地点/时段（竖线写法 + 冒号写法）",
          [(h["location"], h["time"]) for h in headings]
          == [("民国上海街道", "黄昏"), ("东家大厅", "夜")], headings)
    check("场景表: 内景/外景前缀被剥掉",
          not any(h["location"].startswith(("内景", "外景")) for h in headings), headings)
    check("场景表: 地点同义判定（街头/街道 合并，大厅/书房 不合并）",
          locations_match("民国上海街头", "民国上海街道")
          and not locations_match("东家商号大厅", "东家商号书房"), "")
    check("场景表: 时段同桶（深夜/夜晚 同桶，黄昏 不同桶）",
          normalize_time("深夜") == normalize_time("夜晚") == "夜晚"
          and normalize_time("黄昏") == "黄昏", normalize_time("深夜"))
    check("场景表: same_scene 要求地点同义 **且** 时段同桶",
          same_scene("客栈", "夜晚", "客栈大堂", "深夜")
          and not same_scene("客栈", "夜晚", "客栈", "白天"), "")

    # ================= 物品 =================
    prop_first = _call(tools, "save_dedup_props", {"props": [{
        "name": "玉坠", "appearance": "羊脂白玉", "holder": "林昭"}]})
    check("物品: 新增回执文案",
          prop_first == {"message": "物品保存完成：新增 1，复用已有 0", "created": 1, "reused": 0},
          prop_first)
    with engine.begin() as conn:
        prop_row = conn.execute(select(prop_templates)
                                .where(prop_templates.c.name == "玉坠")).first()
        prop_links = conn.execute(select(episode_props.c.prop_id)
                                  .where(episode_props.c.episode_id == episode_id)).all()
    check("物品: category 默认「道具」、keyClue 默认「否」",
          prop_row.category == "道具" and prop_row.key_clue == "否",
          (prop_row.category, prop_row.key_clue))
    check("物品: negativePrompt 兜底是**空串**（原 TS 如此，不构建负面词）",
          prop_row.negative_prompt == "", repr(prop_row.negative_prompt))
    check("物品: image_prompt 列自动构建（非空；prop_templates 不叫 custom_prompt）",
          bool(prop_row.image_prompt))
    check("物品: 已关联到当前集", [link[0] for link in prop_links] == [prop_row.id])

    prop_reuse = _call(tools, "save_dedup_props", {"props": [
        {"name": "玉坠", "category": "信物", "key_clue": "是"}]})
    with engine.begin() as conn:
        prop_row2 = conn.execute(select(prop_templates)
                                 .where(prop_templates.c.name == "玉坠")).first()
        prop_count = len(conn.execute(select(prop_templates.c.id)
                                      .where(prop_templates.c.drama_id == drama_id)).all())
    check("物品: 同名 -> 复用并合并且 category/keyClue",
          prop_reuse["reused"] == 1 and prop_count == 1
          and prop_row2.category == "信物" and prop_row2.key_clue == "是",
          (prop_reuse, prop_count))

    # ================= 关联幂等 =================
    _call(tools, "save_dedup_characters", {"characters": [{"name": "林昭"}]})
    _call(tools, "save_dedup_props", {"props": [{"name": "玉坠"}]})
    with engine.begin() as conn:
        char_links = conn.execute(select(episode_characters.c.character_id)
                                  .where(episode_characters.c.episode_id == episode_id)).all()
        prop_links2 = conn.execute(select(episode_props.c.prop_id)
                                   .where(episode_props.c.episode_id == episode_id)).all()
    check("幂等: 重复保存**不产生重复关联**（角色/物品各 1 条）",
          len(char_links) == 1 and len(prop_links2) == 1,
          (len(char_links), len(prop_links2)))
    # ⚠️ 2026-10-28 改契约：角色的**空提交**曾被当成「成功 created=0」记进日志（界面看着像做完了），
    #    而 qwen3:14b 被催「去保存」时正是这样交差 ⇒ 角色/场景改成回灌 ok=False + 可执行错误，
    #    逼模型带实际内容重来。物品暂无此防护（旧契约），两者分开断言，别让一条摆动拖死另一条。
    empty_chars = _call(tools, "save_dedup_characters", {"characters": []})
    empty_scenes = _call(tools, "save_dedup_scenes", {"scenes": []})
    empty_props = _call(tools, "save_dedup_props", {"props": []})
    check("空提交: 角色/场景空数组 -> ok=False + 可执行错误（不再假装成功）",
          empty_chars.get("ok") is False and "空数组" in (empty_chars.get("error") or "")
          and empty_scenes.get("ok") is False and "空数组" in (empty_scenes.get("error") or ""),
          (empty_chars, empty_scenes))
    check("空提交: 物品空数组 -> 0/0（未加防护，仍是旧契约）",
          empty_props.get("created") == 0, empty_props)
    with engine.begin() as conn:
        char_links_after = conn.execute(select(episode_characters.c.character_id)
                                        .where(episode_characters.c.episode_id == episode_id)).all()
        scene_links_after = conn.execute(select(episode_scenes.c.scene_id)
                                        .where(episode_scenes.c.episode_id == episode_id)).all()
    check("空提交: **一行都没落库**（关联数不变）",
          len(char_links_after) == len(char_links) and len(scene_links_after) >= 1,
          (len(char_links), len(char_links_after), len(scene_links_after)))

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
