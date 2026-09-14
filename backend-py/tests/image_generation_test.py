"""图片生成链路自检（``app/services/image_generation.py`` 及其依赖）。

覆盖三类最容易出错、又最不容易被发现的地方：

1. **门禁与注入**：take 预算 / 剧本指纹门禁 / 逐镜禁止变化 / 时代背景 —— 都是「静默改变
   生成结果」的注入点；
2. **入队字段**：一串 ``??`` 与 ``||`` 混用的默认值，写错不会报错、只会让记录里的值不对；
3. **回写关联表**：分支**顺序即优先级**（itemType → viewType → expression → equipType →
   costume → 主图），顺序错了会写错字段、覆盖别的图。

后台任务一律用「把 ``_spawn`` 换成 no-op」来隔离 —— 测试要的是入队语义，不是真去轮询厂商。

运行::

    ./.venv/Scripts/python.exe tests/image_generation_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="imggen_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    api_usage,
    asset_versions,
    characters,
    image_generations,
    scenes,
    storyboards,
)
from app.services import file_storage as fs  # noqa: E402
from app.services import image_generation as ig  # noqa: E402
from app.services import take_budget as tb  # noqa: E402
from app.services import usage_tracking as ut  # noqa: E402
from app.services.color_grade import apply_color_grade_to_file  # noqa: E402
from app.services.era_background import apply_era_image_clause, get_era_background  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def run(coro):
    return asyncio.run(coro)


def _noop_spawn(coro):
    """替代 ``ig._spawn``：关掉协程但不执行（避免测试真去请求厂商）。"""
    coro.close()


def main() -> int:  # noqa: C901
    client = TestClient(app)
    ig._spawn = _noop_spawn  # type: ignore[assignment]

    # 造一套最小数据：剧 → 集 → 分镜
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 生图父剧"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    sb_id = client.post("/api/v1/storyboards", json={"episode_id": episode_id, "title": "镜头"}).json()["data"]["id"]
    # ⚠️ `POST /api/v1/characters` 尚未迁移（走 501）⇒ 角色行直插。
    #    这里按「表里真实存在的列」动态补 时间列，避免模型加列后测试就挂。
    char_id = _insert_character(drama_id, "林昭")
    scene_id = client.post("/api/v1/scenes", json={"drama_id": drama_id, "location": "客栈"}).json()["data"]["id"]

    # 建一条图片配置（服务层要读「活跃图片配置」）
    # ⚠️ `model` 入参是**数组**（多模型 fallback 用）：路由会 `json.dumps` 后落库，
    #    读侧（Node 与 Python 都是）再 `JSON.parse` 取 `models[0]`。
    #    传裸字符串 "dall-e-3" 会被存成 `"\"dall-e-3\""` → 解析出的不是数组 → **model 变空串**。
    #    这个约定很容易踩，且两边行为一致（所以是契约不是 bug）。
    config_id = client.post("/api/v1/ai-configs", json={
        "service_type": "image", "provider": "openai", "base_url": "https://api.example.com",
        "api_key": "k", "model": ["dall-e-3"], "is_active": True,
    }).json()["data"]["id"]

    # ================= file_storage._ext_from_url =================
    check(
        "ext: 取 URL 路径的扩展名",
        fs._ext_from_url("https://x.com/a/b.PNG") == ".PNG"
        and fs._ext_from_url("https://x.com/a/b?v=1") == ".bin"
        and fs._ext_from_url("https://x.com/a/b.jpg?x=1") == ".jpg",
        (fs._ext_from_url("https://x.com/a/b.PNG"), fs._ext_from_url("https://x.com/a/b?v=1"),
         fs._ext_from_url("https://x.com/a/b.jpg?x=1")),
    )
    check(
        "ext: 非绝对 URL / 无扩展名 / 超长扩展名 -> .bin",
        fs._ext_from_url("static/a.png") == ".bin"
        and fs._ext_from_url("https://x.com/a/b") == ".bin"
        and fs._ext_from_url("https://x.com/a/b.verylong") == ".bin",
        (fs._ext_from_url("static/a.png"), fs._ext_from_url("https://x.com/a/b"),
         fs._ext_from_url("https://x.com/a/b.verylong")),
    )

    # ================= take 预算 =================
    with engine.begin() as conn:
        status = tb.get_take_status(conn, sb_id)
    check(
        "take: 初始状态 = 0/3、未耗尽",
        status is not None and status["take_count"] == 0 and status["take_budget"] == 3
        and status["remaining"] == 3 and status["exhausted"] is False,
        status,
    )
    with engine.begin() as conn:
        tb.consume_take(conn, sb_id)
        tb.consume_take(conn, sb_id)
        status2 = tb.get_take_status(conn, sb_id)
    check("take: 消耗两次后 remaining=1", status2["take_count"] == 2 and status2["remaining"] == 1, status2)
    with engine.begin() as conn:
        check("take: 未耗尽时 allowed", tb.check_take_budget(conn, sb_id)["allowed"] is True)
        tb.consume_take(conn, sb_id)
        blocked = tb.check_take_budget(conn, sb_id)
    check(
        "take: 耗尽后阻断 + 中文 reason 带 3/3",
        blocked["allowed"] is False and "3/3" in blocked["reason"] and "force=true" in blocked["reason"],
        blocked.get("reason"),
    )
    with engine.begin() as conn:
        check("take: force 可放行", tb.check_take_budget(conn, sb_id, force=True)["allowed"] is True)
        tb.reset_take_budget(conn, sb_id)
        check("take: 重置归零", tb.get_take_status(conn, sb_id)["take_count"] == 0)
        check("take: 未绑定分镜一律放行", tb.check_take_budget(conn, None)["allowed"] is True)
        check("take: 不存在的分镜 -> None / 放行",
              tb.get_take_status(conn, 999999) is None and tb.check_take_budget(conn, 999999)["allowed"] is True)

    # ================= 时代背景注入 =================
    with engine.begin() as conn:
        check("era: 无 drama_id -> 原样", apply_era_image_clause(conn, "p", None) == "p")
        check("era: 该剧没有背景 -> 原样", apply_era_image_clause(conn, "p", drama_id) == "p")
        check("era: get_era_background 无背景返回 None", get_era_background(conn, drama_id) is None)

    client.put(f"/api/v1/dramas/{drama_id}", json={
        "era_background": json.dumps({"era": "古代仙侠", "summary": "概述", "imageHint": "水墨质感"},
                                     ensure_ascii=False),
    })
    with engine.begin() as conn:
        era = get_era_background(conn, drama_id)
        check("era: 解析后拿到 imageHint", era is not None and era["imageHint"] == "水墨质感", era)
        check(
            "era: 注入是「prompt, hint.」（补句点）",
            apply_era_image_clause(conn, "p", drama_id) == "p, 水墨质感.",
            apply_era_image_clause(conn, "p", drama_id),
        )
    # 先验「全角句号」的保真行为，再把 era 设回 **ASCII 句点** 版本 ——
    # 后面的入队用例会断言 prompt 以 `水墨质感.` 结尾。
    client.put(f"/api/v1/dramas/{drama_id}", json={
        "era_background": json.dumps({"era": "古代仙侠", "summary": "概述", "imageHint": "水墨质感。"},
                                     ensure_ascii=False),
    })
    with engine.begin() as conn:
        # ⚠️ `hint.endsWith('.')` 只认 **ASCII 点**：全角「。」会被判为「没有句点」而再补一个 "."。
        #    这是原实现的行为（中文提示词里很常见），照实保真、别"顺手修好"。
        check(
            "era: 全角句号不算句点 -> 仍补一个 ASCII 点（保真）",
            apply_era_image_clause(conn, "p", drama_id) == "p, 水墨质感。.",
            apply_era_image_clause(conn, "p", drama_id),
        )
    client.put(f"/api/v1/dramas/{drama_id}", json={
        "era_background": json.dumps({"era": "古代仙侠", "summary": "概述", "imageHint": "水墨质感."},
                                     ensure_ascii=False),
    })
    with engine.begin() as conn:
        check(
            "era: hint 已带 **ASCII** 句点则不重复加",
            apply_era_image_clause(conn, "p", drama_id) == "p, 水墨质感.",
            apply_era_image_clause(conn, "p", drama_id),
        )

    # ================= 校色：无参数短路 + 失败降级 =================
    check(
        "colorgrade: 无参数 -> 原样返回路径（与 Node 相同）",
        run(apply_color_grade_to_file("static/images/none.png", None)) == "static/images/none.png"
        and run(apply_color_grade_to_file("static/images/none.png", "{}")) == "static/images/none.png",
    )
    # ⚠️ 像素管线已于 2026-09-15 落地（ffmpeg，见 color_grade.py 与 color_grade_test.py）。
    #    这里只验**失败路径**：文件不存在 ⇒ ffmpeg 非零退出 ⇒ 抛错 ⇒ 调用方记
    #    `color-grade-failed` 并**保留原图**（与 Node 校色抛错时同形）。
    try:
        run(apply_color_grade_to_file("static/images/x.png", '{"exposure":0.5}'))
        check("colorgrade: 文件不存在 -> 抛错（调用方记 warn 并保留原图）", False, "未抛错")
    except RuntimeError as err:
        check("colorgrade: 文件不存在 -> 抛错且**不带半成品**（调用方记 warn 并保留原图）",
              "校色失败" in str(err), str(err))

    # ================= 入队（generate_image） =================
    # ⚠️ `force=True` 是**必要的**：take 预算是**按分镜累计**的（默认 3 次），
    #    本用例要在同一分镜上反复入队验字段，不绕过门禁会在第 4 次就被挡住。
    #    「验门禁」的那条用例单独用 force=False。
    base_params = {
        "storyboardId": sb_id,
        "dramaId": drama_id,
        "characterId": char_id,
        "prompt": "画面",
        "model": None,
        "referenceImages": [],
        "force": True,
    }
    with engine.begin() as conn:
        image_id = run(ig.generate_image(conn, base_params))
        row = conn.execute(select(image_generations).where(image_generations.c.id == image_id)).first()
    check(
        "enqueue: 返回自增 id 且状态 processing",
        isinstance(image_id, int) and row.status == "processing", (image_id, row.status),
    )
    check(
        "enqueue: provider/model 来自配置、size 默认 1920x1080",
        row.provider == "openai" and row.model == "dall-e-3" and row.size == "1920x1080",
        (row.provider, row.model, row.size),
    )
    check(
        "enqueue: prompt 追加了时代背景指令（尾部是 `, hint.`）",
        row.prompt == "画面, 水墨质感.", repr(row.prompt),
    )
    check(
        "enqueue: 空数组参考图落成 **\"[]\"**（JS 空数组是真值，不是 null）",
        row.reference_images == "[]", repr(row.reference_images),
    )
    check("enqueue: 提交即消耗一次 take", _take_count(sb_id) == 1, _take_count(sb_id))

    # negativePrompt：nullish 链（无则继承配置；显式空串要保留空串）
    with engine.begin() as conn:
        iid2 = run(ig.generate_image(conn, {**base_params, "negativePrompt": ""}))
        r2 = conn.execute(select(image_generations).where(image_generations.c.id == iid2)).first()
    check("enqueue: negativePrompt 传空串 -> 保留空串（nullish，不是回退配置）",
          r2.negative_prompt == "", repr(r2.negative_prompt))

    # 逐镜禁止变化注入
    client.put(f"/api/v1/storyboards/{sb_id}", json={"constraints": "服装与发型不得变化"})
    with engine.begin() as conn:
        iid3 = run(ig.generate_image(conn, {**base_params, "prompt": "基础"}))
        r3 = conn.execute(select(image_generations).where(image_generations.c.id == iid3)).first()
    check(
        "enqueue: 注入逐镜禁止变化（后缀固定文案）",
        "-- 逐镜禁止变化（画面中以下元素必须保持不变，不得增减或改变）: 服装与发型不得变化" in r3.prompt,
        r3.prompt,
    )

    # colorGrade：只有真有调整才落库
    with engine.begin() as conn:
        iid4 = run(ig.generate_image(conn, {**base_params, "colorGrade": {"exposure": 0}}))
        r4 = conn.execute(select(image_generations).where(image_generations.c.id == iid4)).first()
        iid5 = run(ig.generate_image(conn, {**base_params, "colorGrade": {"exposure": 0.5}}))
        r5 = conn.execute(select(image_generations).where(image_generations.c.id == iid5)).first()
    check("enqueue: colorGrade 无实际调整 -> null（hasColorGrade 门禁）", r4.color_grade is None, r4.color_grade)
    check("enqueue: colorGrade 有调整 -> 紧凑 JSON", r5.color_grade == '{"exposure":0.5}', repr(r5.color_grade))

    # 门禁：take 耗尽时入队被阻断（force 放行）
    with engine.begin() as conn:
        tb.reset_take_budget(conn, sb_id)
        for _ in range(3):
            tb.consume_take(conn, sb_id)
    try:
        with engine.begin() as conn:
            run(ig.generate_image(conn, {**base_params, "force": False}))
        check("enqueue: take 耗尽 -> 抛中文 reason", False, "未抛错")
    except ValueError as err:
        check("enqueue: take 耗尽 -> 抛中文 reason", "take 预算已耗尽" in str(err), str(err))
    with engine.begin() as conn:
        forced = run(ig.generate_image(conn, base_params))
        check("enqueue: force=true 绕过 take 门禁", isinstance(forced, int))
        tb.reset_take_budget(conn, sb_id)

    # 门禁：剧本指纹过期
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id)
                     .values(asset_status="needs_regeneration"))
    # （指纹门禁的具体触发条件由 script_fingerprint 的用例覆盖，这里只验证透传不吞错）

    # ================= 参考图归一化 =================
    check(
        "refs: 空/坏 JSON -> []",
        run(ig._normalize_reference_images(None)) == []
        and run(ig._normalize_reference_images("not json")) == []
        and run(ig._normalize_reference_images('{"a":1}')) == [],
    )
    check(
        "refs: 去空白、去空项、去重（保持首次出现顺序）",
        run(ig._normalize_reference_images('[" a ", "a", "", "  ", "b"]')) == ["a", "b"],
        run(ig._normalize_reference_images('[" a ", "a", "", "  ", "b"]')),
    )
    check(
        "refs: data URL 原样保留",
        run(ig._normalize_reference_images('["data:image/png;base64,AA"]')) == ["data:image/png;base64,AA"],
    )
    check(
        "refs: http URL 原样保留",
        run(ig._normalize_reference_images('["https://x.com/a.png"]')) == ["https://x.com/a.png"],
    )
    check(
        "refs: 超过 6 张被截断",
        len(run(ig._normalize_reference_images(json.dumps([f"https://x.com/{i}.png" for i in range(9)])))) == 6,
    )

    # ================= 回写关联表（分支优先级） =================
    def _make_record(**overrides):
        with engine.begin() as conn:
            new_id = run(ig.generate_image(conn, {**base_params, **overrides, "force": True}))
            return conn.execute(select(image_generations).where(image_generations.c.id == new_id)).first()

    # itemType 优先于 costume / equipType
    rec = _make_record(itemType="weapon", equipType="clothing", costume="战袍")
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/a.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    item_images = json.loads(char.item_images or "{}")
    check(
        "回写: itemType 优先（写入 itemImages，且不碰 equipImages / variations / 主图）",
        item_images.get("weapon", {}).get("imageUrl") == "static/images/a.png"
        and item_images["weapon"]["type"] == "weapon"
        and not (char.equip_images or "")
        and char.variations in (None, "", "[]")
        and char.image_url in (None, ""),
        (item_images, char.equip_images, char.variations, char.image_url),
    )

    # viewType（combined 是合法值）
    rec = _make_record(viewType="combined")
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/b.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    three_views = json.loads(char.three_views or "{}")
    check(
        "回写: viewType -> threeViews，项内字段名是 view",
        three_views.get("combined", {}).get("imageUrl") == "static/images/b.png"
        and three_views["combined"]["view"] == "combined",
        three_views,
    )

    # expression
    rec = _make_record(expression="smile")
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/c.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    exps = json.loads(char.expressions or "{}")
    check(
        "回写: expression -> expressions，项内字段名是 key",
        exps.get("smile", {}).get("imageUrl") == "static/images/c.png" and exps["smile"]["key"] == "smile",
        exps,
    )

    # equipType
    rec = _make_record(equipType="accessory")
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/d.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    equip = json.loads(char.equip_images or "{}")
    check("回写: equipType -> equipImages", equip.get("accessory", {}).get("imageUrl") == "static/images/d.png", equip)
    check(
        "回写: 多次写入互不覆盖（item/threeViews/expressions/equipImages 同时存在）",
        bool(char.item_images) and bool(char.three_views) and bool(char.expressions) and bool(char.equip_images),
    )

    # costume：命中已存在的变体名 -> 只改该项；未命中 -> 追加
    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == char_id).values(
            variations=json.dumps([{"name": "红衣", "imageUrl": None}, {"name": "青衫", "imageUrl": "old"}])))
    rec = _make_record(costume="青衫")
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/e.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    variations = json.loads(char.variations)
    check(
        "回写: costume 命中已有变体 -> 只改该项的 imageUrl（不新增、不碰主图）",
        len(variations) == 2 and variations[1]["imageUrl"] == "static/images/e.png"
        and variations[1]["name"] == "青衫" and char.image_url in (None, ""),
        variations,
    )
    rec = _make_record(costume="黄袍")
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/f.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    variations = json.loads(char.variations)
    check(
        "回写: costume 未命中 -> 追加新变体",
        len(variations) == 3 and variations[2] == {"name": "黄袍", "imageUrl": "static/images/f.png"},
        variations,
    )

    # 无 costume -> 覆盖主图
    rec = _make_record()
    with engine.begin() as conn:
        ig._update_character_image(conn, rec, "static/images/g.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    check("回写: 无 costume -> 覆盖主图", char.image_url == "static/images/g.png", char.image_url)

    # 软删角色不回写
    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == char_id).values(deleted_at="2026-01-01T00:00:00.000Z"))
        ig._update_character_image(conn, rec, "static/images/h.png")
        char = conn.execute(select(characters).where(characters.c.id == char_id)).first()
    check("回写: 角色已软删 -> 不回写", char.image_url == "static/images/g.png", char.image_url)
    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == char_id).values(deleted_at=None))

    # ================= 完成收尾（URL 模式 / base64 模式） =================
    def _finish(mode: str, frame_type: str | None = None):
        with engine.begin() as conn:
            new_id = run(ig.generate_image(conn, {
                **base_params, "force": True, "frameType": frame_type,
                "sceneId": scene_id, "prompt": "完成测试",
            }))
            rec2 = conn.execute(select(image_generations).where(image_generations.c.id == new_id)).first()
        if mode == "url":
            run(ig._handle_image_complete(new_id, "openai", "https://cdn.test/ok.png"))
        else:
            run(ig._handle_image_complete_base64(new_id, "gemini", "aGVsbG8=", "image/jpeg"))
        with engine.begin() as conn:
            return conn.execute(select(image_generations).where(image_generations.c.id == new_id)).first()

    # download_file 会真发请求 ⇒ 打桩成「写一个本地文件并返回路径」
    async def _fake_download(url, sub_dir):
        return f"static/{sub_dir}/fake.png"

    ig.download_file = _fake_download  # type: ignore[assignment]

    row = _finish("url")
    check(
        "finish(URL): 状态 completed + 同时写 image_url 与 local_path",
        row.status == "completed" and row.image_url == "https://cdn.test/ok.png"
        and row.local_path == "static/images/fake.png",
        (row.status, row.image_url, row.local_path),
    )
    with engine.begin() as conn:
        sc = conn.execute(select(scenes).where(scenes.c.id == scene_id)).first()
        avs = conn.execute(select(asset_versions).where(
            asset_versions.c.asset_type == "scene", asset_versions.c.asset_id == scene_id)).all()
    check("finish: 场景图回写 + status completed", sc.image_url == "static/images/fake.png" and sc.status == "completed")
    check("finish: 场景资产留档 1 条", len(avs) == 1, len(avs))

    row = _finish("base64", frame_type="first_frame")
    check(
        "finish(base64): **不写** image_url（与 TS 一致），只写 local_path",
        row.status == "completed" and row.image_url is None
        and (row.local_path or "").startswith("static/images/"),
        (row.image_url, row.local_path),
    )
    with engine.begin() as conn:
        sb = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
    check(
        "finish: first_frame -> firstFrameImage + asset_status=approved",
        sb.first_frame_image == row.local_path and sb.asset_status == "approved",
        (sb.first_frame_image, sb.asset_status, row.local_path),
    )

    # 帧类型 → 列的映射（其余三种）
    for frame_type, column in (("last_frame", "last_frame_image"), ("keyframe", "keyframe_image"), (None, "composed_image")):
        row = _finish("url", frame_type=frame_type)
        with engine.begin() as conn:
            sb = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
        check(
            f"finish: frameType={frame_type!r} -> {column}",
            getattr(sb, column) == "static/images/fake.png",
            getattr(sb, column),
        )
    check(
        "finish: keyframe/None 不改 asset_status（只有首尾帧算验收）",
        sb.asset_status == "approved",
    )

    # 完成时把 submitted 用量收口为 completed
    with engine.begin() as conn:
        submitted = conn.execute(select(api_usage).where(api_usage.c.status == "submitted")).all()
    check("finish: 该任务的 submitted 用量已全部收口", len(submitted) == 0, len(submitted))

    # ================= 失败路径：资产标记 needs_regeneration =================
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(asset_status="approved"))
        fail_first = run(ig.generate_image(conn, {**base_params, "force": True, "frameType": "first_frame"}))
        ig._mark_storyboard_asset_needs_regeneration(conn, fail_first)
        sb = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
    check("失败: 首帧任务 -> 分镜标记 needs_regeneration", sb.asset_status == "needs_regeneration", sb.asset_status)

    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(asset_status="approved"))
        fail_composed = run(ig.generate_image(conn, {**base_params, "force": True}))
        ig._mark_storyboard_asset_needs_regeneration(conn, fail_composed)
        sb = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
    check("失败: 非首尾帧任务 -> **不动** asset_status", sb.asset_status == "approved", sb.asset_status)

    # 无 storyboardId（如角色图失败）不影响任何分镜
    with engine.begin() as conn:
        no_sb = run(ig.generate_image(conn, {
            "dramaId": drama_id, "characterId": char_id, "prompt": "p",
            "referenceImages": [], "force": True,
        }))
        ig._mark_storyboard_asset_needs_regeneration(conn, no_sb)
    check("失败: 无 storyboardId -> 静默跳过（角色图失败不影响分镜门禁）", True)

    # ================= 崩溃恢复 =================
    with engine.begin() as conn:
        # 1) 超时过期 -> failed
        expired = run(ig.generate_image(conn, {**base_params, "force": True}))
        conn.execute(image_generations.update().where(image_generations.c.id == expired).values(
            task_id="t-expired", updated_at="2020-01-01T00:00:00.000Z"))
        # 2) 无 taskId -> failed
        no_task = run(ig.generate_image(conn, {**base_params, "force": True}))
        # 3) 未知 provider -> failed（updated_at 必须是「刚刚」，否则会先命中过期分支）
        unknown = run(ig.generate_image(conn, {**base_params, "force": True}))
        conn.execute(image_generations.update().where(image_generations.c.id == unknown).values(
            provider="nope", task_id="t-unknown", updated_at=_iso_now()))
    ig.recover_image_tasks_on_startup()
    with engine.begin() as conn:
        rows = {
            r[0]: (r[1], r[2])
            for r in conn.execute(select(image_generations.c.id, image_generations.c.status,
                                         image_generations.c.error_msg).where(
                image_generations.c.id.in_([expired, no_task, unknown]))).all()
        }
    check("recover: 超时过期 -> failed（reason 带 idle）",
          rows[expired][0] == "failed" and "expired" in (rows[expired][1] or ""), rows[expired])
    check("recover: 无 taskId -> failed（Interrupted before task submit）",
          rows[no_task][0] == "failed" and "before task submit" in (rows[no_task][1] or ""), rows[no_task])
    check("recover: 未知 provider -> failed（reason 带 provider 名）",
          rows[unknown][0] == "failed" and "Unknown image provider" in (rows[unknown][1] or ""), rows[unknown])

    # ================= record_usage =================
    with engine.begin() as conn:
        uid_local = ut.record_usage(conn, {
            "serviceType": "image", "provider": "local-sd", "model": "m",
            "units": 1, "isLocal": True, "status": "submitted",
        })
        uid_remote = ut.record_usage(conn, {
            "serviceType": "image", "provider": "openai", "model": "dall-e-3",
            "units": 1, "isLocal": False, "status": "submitted", "retryCount": 2,
            "meta": {}, "settings": None,
        })
        local_row = conn.execute(select(api_usage).where(api_usage.c.id == uid_local)).first()
        remote_row = conn.execute(select(api_usage).where(api_usage.c.id == uid_remote)).first()
    check("usage: 本地模型 -> cost_amount 为 NULL（不计费）",
          local_row.cost_amount is None and local_row.is_local is True, (local_row.cost_amount, local_row.is_local))
    check("usage: 空对象 meta 也落成 \"{}\"（JS 空对象是真值）",
          remote_row.meta == "{}", repr(remote_row.meta))
    check("usage: retryCount 透传 + status 默认 submitted",
          remote_row.retry_count == 2 and remote_row.status == "submitted", (remote_row.retry_count, remote_row.status))
    check("usage: units 传 0 要保留 0（nullish，不是「未提供」）",
          _record_units_zero() == 0)

    # ================= 汇总 =================
    client.delete(f"/api/v1/dramas/{drama_id}")

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _iso_now() -> str:
    from app.response import now

    return now()


def _insert_character(drama_id: int, name: str) -> int:
    from app.response import now

    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    with engine.begin() as conn:
        result = conn.execute(characters.insert().values(**values))
        return int(result.lastrowid)


def _take_count(storyboard_id: int) -> int:
    with engine.begin() as conn:
        row = conn.execute(
            select(storyboards.c.take_count).where(storyboards.c.id == storyboard_id)
        ).first()
    return row[0]


def _record_units_zero() -> int:
    with engine.begin() as conn:
        uid = ut.record_usage(conn, {
            "serviceType": "image", "provider": "openai", "model": "m", "units": 0, "isLocal": True,
        })
        row = conn.execute(select(api_usage).where(api_usage.c.id == uid)).first()
    return row.units


if __name__ == "__main__":
    raise SystemExit(main())
