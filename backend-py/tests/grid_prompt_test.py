"""S2 第 6 块自检：宫格 prompt + 运镜构图指导（``prompt-utils.ts`` 收尾块）。

三块**直接决定生成结果**的逻辑：

1. **运镜表**（``camera_movement_guides.py``）：先**精确**（含别名）再**包含**（key 长度 > 1）
   —— 顺序反了「跟拍」会被「斜线跟拍」捕获；单字 key（摇/推/拉）不参与包含匹配；
2. **参考资产收集**（``collect_grid_reference_assets``）：顺序决定「图片 N」编号，
   且**上限 6 张** -> 顺序错会改变哪几张图能进编号；
3. **宫格 prompt 三模式**（``build_grid_prompt`` / ``build_grid_cell_prompts``）：
   换行拼接会**丢掉空串**（与第 2 块的 ``, `` 拼接相反），三个分支的**标点各不相同**。

运行::

    ./.venv/Scripts/python.exe tests/grid_prompt_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="grid_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import characters, scenes, storyboard_characters, storyboards  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import camera_movement_guides as cmg  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    # ================= 运镜构图指导 =================
    comp = cmg.get_camera_movement_composition

    check("运镜: 表 29 条", len(cmg.CAMERA_MOVEMENT_GUIDES) == 29)
    check(
        "运镜: 精确匹配（左摇 / 向左摇别名）",
        comp("左摇", "start") == "画面聚焦右侧区域，预留向左摇镜空间"
        and comp("向左摇", "end") == "画面展现左侧区域，完成向左摇镜",
        comp("左摇", "start"),
    )
    # * 表序敏感点：斜线跟拍（第 2 条）含「跟拍」，但「跟拍」必须命中第 27 条的**通用跟拍**
    check(
        "运镜: 精确优先 —— 「跟拍」命中通用跟拍而非「斜线跟拍」",
        comp("跟拍", "start") == "主体入画，前后与侧方预留跟拍空间",
        comp("跟拍", "start"),
    )
    check("运镜: 别名「跟随」也走精确（→ 跟拍）",
          comp("跟随", "end") == "主体随镜头穿越空间，保持视觉关系", comp("跟随", "end"))
    check(
        "运镜: 包含兜底 —— 「快速横移」含「横移」（key 长 2 > 1）",
        comp("快速横移", "end") == "主体完成横向平移，画面横向延展",
        comp("快速横移", "end"),
    )
    check(
        "运镜: **单字 key 不参与包含匹配**（「摇」长 1）-> 含摇但不含任何多字 key 时不命中",
        comp("大幅度摇动", "start") is None,
        comp("大幅度摇动", "start"),
    )
    check("运镜: 单字 key 的**精确**匹配仍然有效（摇 → 摇镜）",
          comp("摇", "start") == "主体偏置一侧，预留摇镜空间", comp("摇", "start"))
    check("运镜: 未命中返回 None（跳切）", comp("跳切", "start") is None)
    check("运镜: 空 / None / 纯空白 -> None",
          comp("", "start") is None and comp(None, "end") is None and comp("   ", "start") is None)
    check("运镜: 首尾帧取的是不同短语（推镜）",
          comp("推镜", "start") == "全景开场，完整场景入画，主体在画面中较小"
          and comp("推镜", "end") == "紧贴主体的特写，细节充满画面")
    check("运镜: 值首尾空白会被 trim", comp("  左摇  ", "start") == comp("左摇", "start"))
    check("运镜: 360 环绕有 4 个别名（'360度环绕拍摄' 精确命中）",
          comp("360度环绕拍摄", "start") == "主体居中，镜头位于360度环绕起点")

    # ================= 造数据 =================
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 宫格 prompt"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    char_id = _insert_character(drama_id, "林昭", image_url="static/images/c.png")
    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(scenes.update().where(scenes.c.id == scene_id)
                     .values(image_url="static/images/s.png"))

    def new_sb(number: int) -> int:
        return client.post("/api/v1/storyboards", json={
            "episode_id": episode_id, "title": f"镜头{number}", "storyboard_number": number,
        }).json()["data"]["id"]

    # ⚠ 这些字段**不能**通过 POST body 传（分镜创建接口不收，会静默忽略）-> 走 UPDATE
    sb1 = new_sb(1)
    sb2 = new_sb(2)
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb1).values(
            scene_id=scene_id,
            first_frame_image="static/images/f1.png",
            last_frame_image="static/images/l1.png",
            reference_images='["static/images/r1.png", "", null]',
            image_prompt="少年立于客栈门前", description="描述1", action="进门",
            movement="推镜", shot_type="wide", location="客栈"))
        conn.execute(storyboards.update().where(storyboards.c.id == sb2).values(
            scene_id=scene_id, first_frame_image="static/images/f2.png",
            description="回头一瞥", action="拔剑", movement="左摇",
            shot_type="close-up", location="客栈内"))
        conn.execute(storyboard_characters.insert().values(storyboard_id=sb1, character_id=char_id))
        conn.execute(storyboard_characters.insert().values(storyboard_id=sb2, character_id=char_id))

    with engine.begin() as conn:
        rows = conn.execute(
            select(storyboards).where(storyboards.c.id.in_([sb1, sb2])).order_by(storyboards.c.id)
        ).all()
        assets = pu.collect_grid_reference_assets(conn, rows)
        legend = pu.build_reference_legend(assets)

    # ================= 参考资产收集 =================
    paths = [asset["path"] for asset in assets]
    check(
        "资产: 顺序 = 分镜(首帧→尾帧→镜头图→参考图) → 场景 → 角色",
        paths == [
            "static/images/f1.png", "static/images/l1.png", "static/images/r1.png",
            "static/images/f2.png", "static/images/s.png", "static/images/c.png",
        ],
        paths,
    )
    check("资产: imageLabel 从「图片1」连续编号", assets[0]["imageLabel"] == "图片1"
          and assets[-1]["imageLabel"] == "图片6", [a["imageLabel"] for a in assets])
    check(
        "资产: 标签用 storyboardNumber（镜头1首帧 / 客栈（夜晚）场景 / 林昭角色）",
        assets[0]["label"] == "镜头1首帧" and assets[4]["label"] == "客栈（夜晚）场景"
        and assets[5]["label"] == "林昭角色",
        [a["label"] for a in assets],
    )
    check("资产: 损坏/空元素被过滤（`[\"x\", \"\", null]` 只留 1 条）",
          len([p for p in paths if p == "static/images/r1.png"]) == 1)
    check("资产: kind 与归属 id 正确",
          assets[0]["kind"] == "storyboard" and assets[0]["storyboardId"] == sb1
          and assets[4]["kind"] == "scene" and assets[4]["sceneId"] == scene_id
          and assets[5]["kind"] == "character" and assets[5]["characterId"] == char_id)
    check("资产: 同一路径去重（char 与 scene 路径不同故都在）", len(set(paths)) == len(paths))
    check("资产: legend 用全角分号且形如 `图片1=镜头1首帧`",
          legend.startswith("图片1=镜头1首帧；图片2=镜头1尾帧"), legend[:50])
    check("资产: 空列表 -> legend 是空串", pu.build_reference_legend([]) == "")

    # 上限 6：再加两个分镜（同路径会被去重），后到的场景/角色应被上限挤掉
    sb3 = new_sb(3)
    sb4 = new_sb(4)
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id.in_([sb3, sb4])).values(
            first_frame_image="static/images/f3.png"))
    with engine.begin() as conn:
        rows4 = conn.execute(select(storyboards).where(
            storyboards.c.id.in_([sb1, sb2, sb3, sb4])).order_by(storyboards.c.id)).all()
        capped = pu.collect_grid_reference_assets(conn, rows4)
    check(
        "资产: **上限 6 张** —— 排最后的**角色立绘**被挤掉（顺序决定谁能进编号）",
        len(capped) == 6 and capped[-1]["path"] == "static/images/s.png"
        and "static/images/c.png" not in [a["path"] for a in capped],
        [a["path"] for a in capped],
    )

    # ================= build_grid_prompt =================
    with engine.begin() as conn:
        rows = conn.execute(select(storyboards).where(
            storyboards.c.id.in_([sb1, sb2])).order_by(storyboards.c.id)).all()
        assets = pu.collect_grid_reference_assets(conn, rows)

        ff = pu.build_grid_prompt(conn, "first_frame", rows, 1, 2, "ink-wash", assets)
        ff_unknown = pu.build_grid_prompt(conn, "first_frame", rows, 1, 2, "nope", assets)
        fl = pu.build_grid_prompt(conn, "first_last", rows, 1, 4, "ink-wash", assets)
        mr = pu.build_grid_prompt(conn, "multi_ref", rows, 2, 2, "ink-wash", assets)
        other = pu.build_grid_prompt(conn, "weird", rows, 1, 2, "ink-wash", assets)
        ff_no_refs = pu.build_grid_prompt(conn, "first_frame", rows, 1, 2, "ink-wash", [])

    style_known = pu.DRAMA_ART_STYLE_MAP["ink-wash"]
    ff_lines = ff.split("\n")
    check(
        "宫格 first_frame: 首行 = 尺寸 + 画风层 + 尾逗号",
        ff_lines[0] == f"1x2 grid layout, consistent art style, {style_known},",
        ff_lines[0][:80],
    )
    check("宫格 first_frame: 第二行是参考图映射（全角冒号）",
          ff_lines[1] == f"参考图映射：{legend}".replace(legend, pu.build_reference_legend(assets)),
          ff_lines[1][:60])
    check("宫格 first_frame: 第三行是固定中文指令",
          ff_lines[2] == "当画面涉及角色或场景时，优先使用对应的图片编号来约束一致性。")
    check("宫格 first_frame: 格子数 = 分镜数（**不是** rows×cols）", len(ff_lines) == 3 + 2 + 1,
          len(ff_lines))
    check(
        "宫格 first_frame: 格标签 `格1（row 1 col 1）` + 半角冒号 + 角色外观 + 参考图",
        ff_lines[3].startswith("格1（row 1 col 1）: Characters: 林昭")
        and "参考图片1（镜头1首帧）" in ff_lines[3]
        and ff_lines[3].endswith("少年立于客栈门前"),
        ff_lines[3][:150],
    )
    check("宫格 first_frame: 末行是收口词",
          ff_lines[-1] == "high quality, cinematic lighting, no text, no watermark")
    # ⚠ 与第 2 块的 `, ` 拼接相反：这里的空串会被丢掉
    check(
        "宫格: 空行被丢掉（无参考图时**不**留空行，`filter(Boolean)` 语义）",
        "\n\n" not in ff_no_refs and "参考图映射" not in ff_no_refs,
        ff_no_refs.split("\n")[:3],
    )
    check(
        "宫格: 画风未命中时回退**字面量** `cinematic illustration style`"
        "（不是 VISUAL_STYLE_MASTER）",
        ff_unknown.split("\n")[0]
        == "1x2 grid layout, consistent art style, cinematic illustration style,",
        ff_unknown.split("\n")[0],
    )
    check("宫格: 未命中不看 VISUAL_STYLE_MASTER", pu.VISUAL_STYLE_MASTER not in ff_unknown)

    fl_lines = fl.split("\n")
    check("宫格 first_last: 格子数 = rows×cols（4 格）", len(fl_lines) == 3 + 4 + 1, len(fl_lines))
    check("宫格 first_last: 第三行是首尾帧节奏说明",
          fl_lines[2] == "first/last frame visual rhythm, alternating opening and closing beats across the grid,")
    check(
        "宫格 first_last: 偶数格 opening moment + 运镜 start 构图",
        fl_lines[3].startswith("格1（row 1 col 1）: ")
        and fl_lines[3].endswith("少年立于客栈门前, opening moment, 全景开场，完整场景入画，主体在画面中较小"),
        fl_lines[3][-90:],
    )
    # ⚠️ 第 2 格取的是 views[1]（镜头 2，movement=左摇、action=拔剑），不是镜头 1
    check(
        "宫格 first_last: 奇数格 closing moment + 该镜自己的 action 前缀 + 运镜 end 构图",
        "拔剑, closing moment, subtle motion change" in fl_lines[4]
        and fl_lines[4].endswith("画面展现左侧区域，完成向左摇镜"),
        fl_lines[4][-110:],
    )
    check("宫格 first_last: 分镜不足时**循环取用**（i % len -> 第 3 格回到镜头1）",
          fl_lines[5].startswith("格3（row 1 col 3）:") and "少年立于客栈门前" in fl_lines[5],
          fl_lines[5][:60])
    check("宫格 first_last: 末行是连续运镜收口",
          fl_lines[-1] == "continuous motion implied between left and right, high quality, no text")

    mr_lines = mr.split("\n")
    check("宫格 multi_ref: 格子数 = rows×cols（4 格）", len(mr_lines) == 3 + 4 + 1, len(mr_lines))
    check("宫格 multi_ref: 首行点明 `same scene different angles`",
          "same scene different angles and compositions" in mr_lines[0], mr_lines[0][:90])
    check("宫格 multi_ref: 有 `main scene:` 行", mr_lines[2] == "main scene: 少年立于客栈门前,")
    check(
        "宫格 multi_ref: 每格用**legend 整体**做参考（不是逐格 refs）+ 角度表轮转",
        mr_lines[3].startswith("格1（row 1 col 1）: 参考图片1=镜头1首帧；")
        and mr_lines[3].endswith(", wide establishing shot")
        and mr_lines[4].endswith(", medium shot character focus"),
        mr_lines[3][-60:],
    )
    check("宫格: 未知 mode -> 兜底短语（含画风）",
          other == f"1x2 grid, {style_known}, storyboard frames, high quality", other)

    # ================= build_grid_cell_prompts =================
    with engine.begin() as conn:
        cells_ff = pu.build_grid_cell_prompts(conn, "first_frame", rows, 1, 2, assets)
        cells_fl = pu.build_grid_cell_prompts(conn, "first_last", rows, 2, 2, assets)
        cells_mr = pu.build_grid_cell_prompts(conn, "multi_ref", rows, 1, 2, assets)
        cells_none = pu.build_grid_cell_prompts(conn, "first_frame", [], 1, 2, assets)
        cells_1x1 = pu.build_grid_cell_prompts(conn, "first_frame", rows, 1, 1, assets)

    check("逐格: 空分镜 -> 空数组（**不抛错**）", cells_none == [])
    check(
        "逐格 first_frame: frame_type / shot_number / 全角冒号 + `, opening scene`",
        len(cells_ff) == 2 and cells_ff[0]["frame_type"] == "first_frame"
        and cells_ff[0]["shot_number"] == 1
        and cells_ff[0]["prompt"].startswith("格1（row 1 col 1）：")
        and cells_ff[0]["prompt"].endswith("客栈, wide, opening scene, 全景开场，完整场景入画，主体在画面中较小"),
        cells_ff[0]["prompt"][-80:],
    )
    check(
        "逐格 first_frame: 只取前 rows×cols 个（`slice(0, rows*cols)`）",
        len(cells_1x1) == 1,
        len(cells_1x1),
    )
    check(
        "逐格 first_last: 标点是 `，首帧：` / `，尾帧：`（**全角**，与另两个分支不同）",
        cells_fl[0]["prompt"].startswith("格1（row 1 col 1），首帧：")
        and cells_fl[1]["prompt"].startswith("格2（row 1 col 2），尾帧：")
        and cells_fl[0]["frame_type"] == "first_frame"
        and cells_fl[1]["frame_type"] == "last_frame",
        cells_fl[1]["prompt"][:40],
    )
    check(
        "逐格 first_last: 尾帧带 motion（action 优先）+ 运镜 end 构图",
        "拔剑" in cells_fl[1]["prompt"] and cells_fl[1]["prompt"].endswith("画面展现左侧区域，完成向左摇镜"),
        cells_fl[1]["prompt"][-70:],
    )
    check(
        "逐格 multi_ref: 标点是半角 `: ` + frame_type=reference + 角度轮转",
        cells_mr[0]["prompt"].startswith("格1（row 1 col 1）: ")
        and cells_mr[0]["frame_type"] == "reference"
        and cells_mr[1]["prompt"].endswith(", medium shot character focus"),
        cells_mr[1]["prompt"][-50:],
    )
    # 4 格循环取 2 个分镜：格3 → views[0]（推镜，start）、格4 → views[1]（左摇，end）
    check(
        "逐格 first_last: 运镜 start/end 交替取用（偶数格 start、奇数格 end）",
        cells_fl[2]["prompt"].endswith("全景开场，完整场景入画，主体在画面中较小")
        and cells_fl[3]["prompt"].endswith("画面展现左侧区域，完成向左摇镜"),
        cells_fl[3]["prompt"][-50:],
    )

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
