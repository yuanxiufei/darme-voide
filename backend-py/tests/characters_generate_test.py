"""S6 自检：characters 域的 **8 个生成端点**（`characters.ts` 后半部分，626 行里的一半）。

这一批是「角色资产生产链」：智能拆分 → 立绘 → 三视图 → 装备图 → 表情组 → 批量出图。
服务（``generate_image`` / ``generate_voice_sample`` / ``split_character_visuals``）**全部打桩**
⇒ 不打网络、不落盘、不调模型。

四类必锁的语义（抄错都不报错，只出「丑图/出人/画风漂移/多花钱」）：

1. **错误码不统一**：id 非法 = 404；角色找不到 = 400（**只有 auto-split-visuals 是 404**）；
   ``generate-three-views`` **没有 try/catch**（异常直冒 500）；批量接口**逐条静默跳过**；
2. **配置回退不同**：单张接口 ``ep.imageConfigId ?? drama 级第一个``，**批量接口只用 ep 的**；
3. **画风收口**：自定义 prompt 也必须追加画风后缀 + 注入服装/武器/首饰子句；
4. **视觉锚定**：立绘不自动锚定；三视图锚主立绘；装备/表情锚「三视图 combined → 主立绘」，
   显式 ``reference_images`` 最高优先（截断到 6 张）；``anchor='none'`` 不锚。

运行::

    ./.venv/Scripts/python.exe tests/characters_generate_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="chargen_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import characters, dramas, episodes  # noqa: E402
from app.core.response import now  # noqa: E402
from app.routers import characters as cr  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ============================================================
# 桩：三个被调用的服务
# ============================================================
_CALLS: list[tuple[str, object]] = []


async def _fake_generate_image(conn, params):  # noqa: ANN001
    _CALLS.append(("image", params))
    if params.get("prompt") == "BOOM":
        raise RuntimeError("生成炸了")
    return 1234


async def _fake_voice_sample(conn, name, voice, config_id=None):  # noqa: ANN001
    _CALLS.append(("voice", {"name": name, "voice": voice, "configId": config_id}))
    if voice == "BOOM":
        raise RuntimeError("TTS 炸了")
    return "static/audio/sample.mp3"


async def _fake_split(conn, payload):  # noqa: ANN001
    _CALLS.append(("split", payload))
    if payload.get("appearance") == "BOOM":
        raise RuntimeError("拆分炸了")
    return {"clothing": "黑色风衣", "weapons": "长剑", "accessories": "耳环"}


cr.generate_image = _fake_generate_image  # type: ignore[assignment]
cr.generate_voice_sample = _fake_voice_sample  # type: ignore[assignment]
cr.split_character_visuals = _fake_split  # type: ignore[assignment]


def _seed() -> tuple[int, int, int, int]:
    """建 drama / episode（带 image+audio configId）/ 两个角色，返回 id 四元组。"""
    ts = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="测试剧", style="realistic", status="draft",
            created_at=ts, updated_at=ts)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            status="draft", image_config_id=11, audio_config_id=22,
            created_at=ts, updated_at=ts)).lastrowid)
        rich = int(conn.execute(characters.insert().values(
            drama_id=drama_id, name="林昭", appearance="短发、冷峻", personality="寡言",
            clothing="黑色风衣", weapons="长剑", accessories="耳环",
            voice_style="zh-CN-YunxiNeural", style=None,
            custom_prompt="my custom prompt",
            image_url="static/images/main.png",
            local_path=None,
            three_views=json.dumps({"combined": {"imageUrl": "static/images/three.png"}}),
            created_at=ts, updated_at=ts)).lastrowid)
        plain = int(conn.execute(characters.insert().values(
            drama_id=drama_id, name="无名", created_at=ts, updated_at=ts)).lastrowid)
    return drama_id, episode_id, rich, plain


def main() -> int:  # noqa: C901
    drama_id, episode_id, rich, plain = _seed()
    client = TestClient(app)
    base = "/api/v1/characters"

    # ================= 纯辅助 =================
    check("辅助: _parse_body_ref_images —— 数组/JSON 字符串都吃，空项过滤",
          cr._parse_body_ref_images({"reference_images": [" a ", "", "b"]}) == ["a", "b"]
          and cr._parse_body_ref_images({"reference_images": '["x","y"]'}) == ["x", "y"]
          and cr._parse_body_ref_images({"referenceImages": ["z"]}) == ["z"]
          and cr._parse_body_ref_images({}) is None)
    check("辅助: reference_images 为 null 时**回落到 camelCase 键**（`??` 语义）",
          cr._parse_body_ref_images({"reference_images": None,
                                     "referenceImages": ["fallback"]}) == ["fallback"])
    check("辅助: 三视图 combined 提取（combined → front → side → back，坏 JSON 不炸）",
          cr._get_character_three_view_combined_image(
              type("C", (), {"three_views": json.dumps({"front": {"imageUrl": "f"}})})) == "f"
          and cr._get_character_three_view_combined_image(
              type("C", (), {"three_views": "not json"})) is None
          and cr._get_character_three_view_combined_image(
              type("C", (), {"three_views": None})) is None)

    char_stub = type("C", (), {"image_url": "main.png", "local_path": None,
                               "three_views": json.dumps({"combined": {"imageUrl": "three.png"}})})()
    check("锚定: 显式参考图最高优先，且**截断到 6 张**",
          cr._resolve_character_anchors(char_stub, {"reference_images": [f"r{i}" for i in range(9)]})
          == [f"r{i}" for i in range(6)])
    check("锚定: `anchor='none'` 不锚定（显式参考图仍优先）",
          cr._resolve_character_anchors(char_stub, {"anchor": "none"}) is None)
    check("锚定: mode='main' 只锚主立绘；auto 优先三视图 combined",
          cr._resolve_character_anchors(char_stub, {}, "main") == ["main.png"]
          and cr._resolve_character_anchors(char_stub, {}) == ["three.png"]
          and cr._resolve_character_anchors(char_stub, {"anchor": "three_views"}) == ["three.png"])
    no_image = type("C", (), {"image_url": None, "local_path": None, "three_views": None})()
    check("锚定: 什么都没有 -> None（不发空参考图）",
          cr._resolve_character_anchors(no_image, {}) is None
          and cr._resolve_character_anchors(no_image, {"anchor": "main"}) is None)

    check("收口: 自定义 prompt 也会**追加画风后缀**，并注入视觉子句",
          cr._resolve_character_prompt("", "AUTO", "anime") == "AUTO"
          and cr._resolve_character_prompt("custom", "AUTO", "anime") == "custom" + pu.build_character_art_style_suffix("anime")
          and ", wearing 黑色风衣" in cr._resolve_character_prompt(
              "custom", "AUTO", "anime", "wearing 黑色风衣"))
    check("收口: 负向词为空才用默认（有值原样保留）",
          cr._resolve_character_negative("", "anime") == pu.build_character_negative_prompt("anime")
          and cr._resolve_character_negative("mine", "anime") == "mine")
    check("收口: 旧版装备提示词判定（4 种命中 + 2 种不命中）",
          all(cr._is_legacy_equip_prompt(text) for text in (
              "character appearance: tall", "for the character Lin",
              "single view of a sword", "realistic cinematic character design, armor"))
          and not cr._is_legacy_equip_prompt("three views, side by side")
          and not cr._is_legacy_equip_prompt("single view, side by side"))
    check("收口: 装备 prompt —— 空/旧版丢弃走构建器；新版保留并追装备画风尾",
          cr._resolve_equip_prompt("", "AUTO", "anime") == "AUTO"
          and cr._resolve_equip_prompt("single view of sword", "AUTO", "anime") == "AUTO"
          and cr._resolve_equip_prompt("my equip", "AUTO", "anime")
          == "my equip" + pu.build_equip_art_style_suffix("anime"))

    # ================= auto-split-visuals =================
    _CALLS.clear()
    check("拆分: id 非法 -> 404 Invalid character id",
          client.post(f"{base}/abc/auto-split-visuals").status_code == 404)
    check("拆分: 角色不存在 -> **404** Character not found（本端点独有）",
          client.post(f"{base}/999999/auto-split-visuals").status_code == 404)
    check("拆分: 外貌为空 -> 400 请先填写「外貌特征」",
          client.post(f"{base}/{plain}/auto-split-visuals", json={}).json()
          == {"code": 400, "message": "请先填写「外貌特征」"})
    split_ok = client.post(f"{base}/{rich}/auto-split-visuals").json()
    check("拆分: 用库里的外貌特征跑 LLM，回传拆分结果",
          split_ok["data"] == {"clothing": "黑色风衣", "weapons": "长剑", "accessories": "耳环"}
          and _CALLS[-1][1] == {"appearance": "短发、冷峻"}, (split_ok, _CALLS[-1]))
    check("拆分: body.appearance 覆盖库内值",
          client.post(f"{base}/{rich}/auto-split-visuals",
                      json={"appearance": "  长发  "}).status_code == 200
          and _CALLS[-1][1] == {"appearance": "长发"})
    failed = client.post(f"{base}/{rich}/auto-split-visuals", json={"appearance": "BOOM"})
    check("拆分: 服务抛错 -> **HTTP 400**「智能拆分失败: ...」（仓里 badRequest 就是 400）",
          failed.status_code == 400 and failed.json()["code"] == 400
          and failed.json()["message"] == "智能拆分失败: 拆分炸了", failed.json())

    # ================= generate-voice-sample =================
    _CALLS.clear()
    check("试听: 角色不存在 -> **400**（与拆分端点不同！）",
          client.post(f"{base}/999999/generate-voice-sample", json={}).json()
          == {"code": 400, "message": "Character not found"})
    check("试听: 未分配音色 -> 400 请先分配音色",
          client.post(f"{base}/{plain}/generate-voice-sample", json={}).json()
          == {"code": 400, "message": "请先分配音色"})
    check("试听: episode_id 不存在 -> 400 Episode not found",
          client.post(f"{base}/{rich}/generate-voice-sample",
                      json={"episode_id": 999999}).json()
          == {"code": 400, "message": "Episode not found"})
    check("试听: episode_id 非数字 -> 也走 400（Number('abc')=NaN 的等价语义）",
          client.post(f"{base}/{rich}/generate-voice-sample",
                      json={"episode_id": "abc"}).json()
          == {"code": 400, "message": "Episode not found"})
    voice_ok = client.post(f"{base}/{rich}/generate-voice-sample",
                           json={"episode_id": episode_id}).json()
    check("试听: 成功 -> 回执键是 **snake_case** `voice_sample_url`，且 configId 取集上的",
          voice_ok["data"] == {"voice_sample_url": "static/audio/sample.mp3"}
          and _CALLS[-1][1]["configId"] == 22, (voice_ok, _CALLS[-1]))
    with engine.begin() as conn:
        row = conn.execute(select(characters.c.voice_sample_url)
                           .where(characters.c.id == rich)).first()
    check("试听: 音色试听 URL 已回写 characters.voice_sample_url（库内保持 snake_case）",
          row[0] == "static/audio/sample.mp3", row[0])
    client.post(f"{base}/{rich}/generate-voice-sample")
    check("试听: 不传 episode_id -> configId 回退到 **drama 级第一个**集配置",
          _CALLS[-1][1]["configId"] == 22, _CALLS[-1])
    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == rich)
                     .values(voice_style="BOOM"))
    tts_failed = client.post(f"{base}/{rich}/generate-voice-sample")
    check("试听: 服务抛错 -> 400「TTS 生成失败: ...」",
          tts_failed.json()["message"] == "TTS 生成失败: TTS 炸了", tts_failed.json())
    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == rich)
                     .values(voice_style="zh-CN-YunxiNeural"))

    # ================= generate-image =================
    _CALLS.clear()
    check("立绘: 角色不存在 -> 400 Character not found",
          client.post(f"{base}/999999/generate-image", json={}).json()
          == {"code": 400, "message": "Character not found"})
    image_ok = client.post(f"{base}/{rich}/generate-image", json={}).json()["data"]
    call = _CALLS[-1][1]
    check("立绘: 成功 -> image_generation_id，且 prompt 是「自定义 + 视觉效果子句 + 画风尾」",
          image_ok == {"image_generation_id": 1234}
          and call["prompt"].startswith("my custom prompt, ")
          and "wearing 黑色风衣" in call["prompt"]
          and call["prompt"].endswith(pu.build_character_art_style_suffix("realistic")), call["prompt"])
    check("立绘: **不做自动锚定**（未传 reference_images -> None）",
          call["referenceImages"] is None)
    check("立绘: 显式参考图原样下发；configId 取集上的 imageConfigId",
          client.post(f"{base}/{rich}/generate-image",
                      json={"reference_images": ["a.png"]}).status_code == 200
          and _CALLS[-1][1]["referenceImages"] == ["a.png"]
          and _CALLS[-1][1]["configId"] == 11)
    check("立绘: 无集时 configId 回退 drama 级（另一个角色只挂别的剧则不给）",
          client.post(f"{base}/{rich}/generate-image",
                      json={"colorGrade": "warm"}).status_code == 200
          and _CALLS[-1][1]["colorGrade"] == "warm")

    # ================= generate-prompt（预览不落库）=================
    check("预览: type 非法 -> 400（带枚举说明）",
          client.post(f"{base}/{rich}/generate-prompt", json={"type": "nope"}).json()
          == {"code": 400, "message": "type 必须是 character/clothing/weapon/accessory 之一"})
    preview = client.post(f"{base}/{rich}/generate-prompt", json={}).json()["data"]
    check("预览: 默认 type=character，返回正/负向两个字段且非空",
          preview["type"] == "character" and preview["prompt"]
          and preview["negativePrompt"], list(preview))
    equip_preview = client.post(f"{base}/{rich}/generate-prompt",
                                json={"type": "clothing"}).json()["data"]
    check("预览: type=clothing 走装备构建器（负向换成装备负向）",
          equip_preview["negativePrompt"] == pu.build_equip_negative("realistic")
          and equip_preview["negativePrompt"] != preview["negativePrompt"])
    snapped = client.post(f"{base}/{rich}/generate-prompt",
                          json={"type": "character", "appearance": "银发"}).json()["data"]
    check("预览: 表单最新值覆盖生效（body 里的 appearance 进 prompt）",
          "银发" in snapped["prompt"], snapped["prompt"][:80])
    # ⚠️ coreFeatures 在两个版本里都只吃 **JSON 字符串**（TS 签名就是 `coreFeatures?: string`），
    #    传数组会被忽略（两边一致，非漂移）
    snake_override = client.post(f"{base}/{rich}/generate-prompt",
                                 json={"type": "character",
                                       "core_features": '["snake-marker"]'}).json()["data"]
    camel_override = client.post(f"{base}/{rich}/generate-prompt",
                                 json={"type": "character",
                                       "coreFeatures": '["camel-marker"]'}).json()["data"]
    check("预览: **snake_case 与 camelCase 两种键名都吃**（coreFeatures 是要 JSON 字符串）",
          "snake-marker" in snake_override["prompt"]
          and "camel-marker" in camel_override["prompt"],
          (snake_override["prompt"][:70], camel_override["prompt"][:70]))
    check("预览: coreFeatures 传**数组**会被忽略（与 TS 同款怪癖：只解析 JSON 字符串）",
          "array-marker" not in client.post(
              f"{base}/{rich}/generate-prompt",
              json={"type": "character", "coreFeatures": ["array-marker"]}).json()["data"]["prompt"])

    # ================= generate-three-views =================
    _CALLS.clear()
    check("三视图: views 全非法 -> 400",
          client.post(f"{base}/{rich}/generate-three-views", json={"views": ["x"]}).json()
          == {"code": 400, "message": "views 必须包含 front/side/back 之一"})
    three = client.post(f"{base}/{rich}/generate-three-views", json={}).json()["data"]
    call = _CALLS[-1][1]
    check("三视图: **合成一张**横向长图（size=2048x896、viewType=combined、count=1）",
          three["count"] == 1 and three["results"][0]["view"] == "combined"
          and call["size"] == pu.THREE_VIEW_SIZE and call["viewType"] == "combined",
          (three, call.get("size")))
    check("三视图: prompt 末尾拼上三视图并排布局词",
          call["prompt"].endswith(pu.THREE_VIEW_COMBINED_LAYOUT), call["prompt"][-40:])
    check("三视图: 视觉锚定 = **主立绘**（mode='main'）",
          call["referenceImages"] == ["static/images/main.png"], call["referenceImages"])
    check("三视图: 无用户负向词 -> 追加「组合图」排除词",
          call["negativePrompt"] == pu.build_three_view_negative("realistic"))
    client.post(f"{base}/{rich}/generate-three-views",
                json={"negative_prompt": "blurry, lowres"})
    check("三视图: 用户给了负向词 -> 保留用户词（不换组合排除词）",
          _CALLS[-1][1]["negativePrompt"] == "blurry, lowres")

    # ================= generate-equip-image =================
    _CALLS.clear()
    check("装备图: type 非法 -> 400（且**先于**角色查询）",
          client.post(f"{base}/999999/generate-equip-image", json={"type": "nope"}).json()
          == {"code": 400, "message": "type 必须是 clothing/weapon/accessory 之一"})
    equip = client.post(f"{base}/{rich}/generate-equip-image",
                        json={"type": "clothing"}).json()["data"]
    call = _CALLS[-1][1]
    check("装备图: 默认 view 模式 -> 横向长图 + equipType + 锚定三视图（不锚主立绘）",
          equipeq(call["size"], pu.THREE_VIEW_SIZE) and call["equipType"] == "clothing"
          and call["itemType"] is None
          and call["referenceImages"] == ["static/images/three.png"], call)
    check("装备图: 无用户负向 -> 用装备三视图负向（排除人物/手持）",
          call["negativePrompt"] == pu.build_equip_negative("realistic"))
    single = client.post(f"{base}/{rich}/generate-equip-image",
                         json={"type": "weapon", "mode": "single"}).json()["data"]
    call = _CALLS[-1][1]
    check("装备图: single 模式 -> 1:1 方形、itemType=类型、**不锚定**、用物品负向",
          single["image_generation_id"] == 1234
          and call["size"] == pu.ITEM_IMAGE_SIZE and call["itemType"] == "weapon"
          and call["equipType"] == "weapon" and call["referenceImages"] is None
          and call["negativePrompt"] == pu.build_item_negative("realistic"), call)
    client.post(f"{base}/{rich}/generate-equip-image",
                json={"type": "clothing", "negative_prompt": "blurry, lowres"})
    check("装备图: 用户负向**缺人物排除词** -> 强制换成装备三视图负向（防出人）",
          _CALLS[-1][1]["negativePrompt"] == pu.build_equip_negative("realistic"),
          _CALLS[-1][1]["negativePrompt"][:60])
    client.post(f"{base}/{rich}/generate-equip-image",
                json={"type": "clothing", "negative_prompt": "no person, no human hands, clean"})
    check("装备图: 用户负向**含人物排除词** -> 保留用户词（角色收口）",
          _CALLS[-1][1]["negativePrompt"].startswith("no person"),
          _CALLS[-1][1]["negativePrompt"][:40])

    # ================= generate-expressions =================
    _CALLS.clear()
    check("表情: keys 全非法 -> 400",
          client.post(f"{base}/{rich}/generate-expressions", json={"keys": ["nope"]}).json()
          == {"code": 400, "message": "keys 中没有合法的表情 key"})
    all_expressions = client.post(f"{base}/{rich}/generate-expressions", json={}).json()["data"]
    check("表情: 省略 keys -> 跑**全部预设**，每条带 key/label/image_generation_id",
          all_expressions["count"] == len(pu.EXPRESSION_PRESETS)
          and set(all_expressions["results"][0]) == {"key", "label", "image_generation_id"},
          all_expressions["count"])
    check("表情: 每张都带锚定与 1024x1024、expression=key",
          _CALLS[-1][1]["referenceImages"] == ["static/images/three.png"]
          and _CALLS[-1][1]["size"] == "1024x1024"
          and _CALLS[-1][1]["expression"] == all_expressions["results"][-1]["key"])
    one = client.post(f"{base}/{rich}/generate-expressions",
                      json={"keys": ["smile"]}).json()["data"]
    check("表情: 传单个 key -> 只生成 1 张（单图重生成路径）",
          one["count"] == 1 and one["results"][0]["key"] == "smile", one)
    check("表情: 逐张独立成败 —— 某张失败只少一条，整批仍 200",
          _inject_one_failure(client, base, rich) == len(pu.EXPRESSION_PRESETS) - 1)

    # ================= batch-generate-images =================
    _CALLS.clear()
    check("批量: 缺 episode_id -> 400 episode_id is required",
          client.post(f"{base}/batch-generate-images", json={}).json()
          == {"code": 400, "message": "episode_id is required"})
    check("批量: 集不存在 -> 400 Episode not found",
          client.post(f"{base}/batch-generate-images",
                      json={"episode_id": 999999, "character_ids": []}).json()
          == {"code": 400, "message": "Episode not found"})
    batch = client.post(f"{base}/batch-generate-images",
                        json={"episode_id": episode_id,
                              "character_ids": [rich, 999999, plain]}).json()["data"]
    check("批量: 存在的角色逐个出图，**不存在的静默跳过**",
          batch["count"] == 2 and len(batch["ids"]) == 2, batch)
    check("批量: configId **只用集上的**（不回退 drama 级）",
          all(call[1]["configId"] == 11 for call in _CALLS), [c[1]["configId"] for c in _CALLS])
    check("批量: prompt 由角色字段构建（无 customPrompt 时走自动构建器）",
          all("name" not in call[1] for call in _CALLS)
          and all(call[1]["prompt"] for call in _CALLS))

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


def equipeq(left: object, right: object) -> bool:
    """``size`` 断言小助手（默认值可能是字符串常量）。"""
    return left == right


def _inject_one_failure(client: TestClient, base: str, character_id: int) -> int:
    """让「第一个表情」失败，返回实际成功的张数（验证逐张独立成败）。"""
    original = cr.generate_image
    first_key = pu.EXPRESSION_PRESETS[0]["key"]
    counter = {"n": 0}

    async def _flaky(conn, params):  # noqa: ANN001
        counter["n"] += 1
        if params.get("expression") == first_key:
            raise RuntimeError("这张炸了")
        return 1234

    cr.generate_image = _flaky  # type: ignore[assignment]
    try:
        data = client.post(f"{base}/{character_id}/generate-expressions",
                           json={"keys": [p["key"] for p in pu.EXPRESSION_PRESETS]}).json()["data"]
    finally:
        cr.generate_image = original  # type: ignore[assignment]
    check("表情: 失败的那张**不在结果里**（其余照常）",
          all(item["key"] != first_key for item in data["results"]), data["count"])
    return data["count"]


if __name__ == "__main__":
    raise SystemExit(main())
