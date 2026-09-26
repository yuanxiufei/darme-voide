"""S6 自检：props / scenes 各一条 ``POST /{id}/generate-image``。

两条端点都短，但各有「抄错不报错」的语义：

* **props**：找不到物品是 **404 中文文案**（本域特例）；``category='屏幕留白'`` 走留白构建器；
  ``config_id`` 是 **falsy 判定**；回执键是 **camelCase**（``imageGenerationId`` + **回显 prompt**）；
* **scenes**：id 非法 404 / 场景不存在 400；**跨集一致参考图**（同剧同地点已出图的其他场景，
  最多 2 张）——**地点为空必须跳过**，否则会把所有「无地点」场景串成一张图；
  置 ``processing`` 后**成功不重置**、**失败置 ``failed``**。

生成服务打桩（不打网络、不落盘）。

运行::

    ./.venv/Scripts/python.exe tests/props_scenes_generate_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="propscenes_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas, episodes, prop_templates, scenes  # noqa: E402
from app.core.response import now  # noqa: E402
from app.routers import props as props_router  # noqa: E402
from app.routers import scenes as scenes_router  # noqa: E402
from app.services.prompt_utils import build_scene_negative_prompt  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_CALLS: list[dict] = []


async def _fake_generate_image(conn, params):  # noqa: ANN001
    _CALLS.append(params)
    if params.get("prompt") == "BOOM":
        raise RuntimeError("生成炸了")
    return 4321


async def _fail_with_empty_message(conn, params):  # noqa: ANN001
    _CALLS.append(params)
    raise RuntimeError("")


props_router.generate_image = _fake_generate_image  # type: ignore[assignment]
scenes_router.generate_image = _fake_generate_image  # type: ignore[assignment]


def main() -> int:  # noqa: C901
    ts = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="剧", style="realistic", created_at=ts, updated_at=ts)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="一", content="x",
            status="draft", image_config_id=11, created_at=ts, updated_at=ts)).lastrowid)
        normal_prop = int(conn.execute(prop_templates.insert().values(
            drama_id=drama_id, name="青铜剑", category="道具", appearance="锈迹斑斑",
            description="主角家传", size_hint="三尺", created_at=ts, updated_at=ts)).lastrowid)
        plate_prop = int(conn.execute(prop_templates.insert().values(
            drama_id=drama_id, name="留白框", category="屏幕留白", description="供叠字",
            created_at=ts, updated_at=ts)).lastrowid)
        # ⚠️ scenes 的 time / prompt / location 都是 NOT NULL ⇒ 种子必须给值
        #    （「无地点」在库里是**空串**，正是端点要防的那种）
        def _scene(**fields: object) -> int:
            values: dict[str, object] = {"drama_id": drama_id, "time": "夜", "prompt": "",
                                         "location": "", "created_at": ts, "updated_at": ts}
            values.update(fields)
            return int(conn.execute(scenes.insert().values(**values)).lastrowid)

        base_scene = _scene(location="客栈大堂", prompt="木桌、灯笼")
        city_scene = _scene(location="城门口", prompt="", time="日")
        _scene(location="客栈大堂", image_url="static/images/a.png")
        same_location2 = _scene(location=" 客栈大堂 ", image_url="static/images/b.png")
        _scene(location="客栈大堂", image_url="static/images/c.png")
        no_location = _scene(image_url="static/images/none.png")
        _scene(location="客栈大堂", image_url="static/images/deleted.png", deleted_at=ts)

    client = TestClient(app)

    # ================= props =================
    base = "/api/v1/props"
    _CALLS.clear()
    check("物品图: id 非法 -> 404 Invalid prop id（本域 parseId 更严：1.5 也算非法）",
          client.post(f"{base}/abc/generate-image").status_code == 404
          and client.post(f"{base}/1.5/generate-image").status_code == 404)
    check("物品图: 物品不存在 -> 404「物品不存在」（中文文案，本域特例）",
          client.post(f"{base}/999999/generate-image").json()
          == {"code": 404, "message": "物品不存在"})
    plain = client.post(f"{base}/{normal_prop}/generate-image", json={}).json()["data"]
    call = _CALLS[-1]
    check("物品图: 回执是 camelCase `imageGenerationId` 且**回显 prompt**",
          plain["imageGenerationId"] == 4321 and plain["prompt"]
          and set(plain) == {"imageGenerationId", "prompt"}, list(plain))
    check("物品图: 普通分类走物品构建器（prompt 里有物品名）",
          "青铜剑" in plain["prompt"] and call["propId"] == normal_prop
          and call["dramaId"] == drama_id, plain["prompt"][:60])
    plate = client.post(f"{base}/{plate_prop}/generate-image", json={}).json()["data"]
    check("物品图: `category='屏幕留白'` 走**留白图**构建器（prompt 与普通构建器不同）",
          plate["prompt"] != plain["prompt"] and "留白框" in plate["prompt"],
          plate["prompt"][:60])
    check("物品图: body.prompt 优先于两种构建器",
          client.post(f"{base}/{normal_prop}/generate-image",
                      json={"prompt": "custom prompt"}).json()["data"]["prompt"] == "custom prompt")
    check("物品图: config_id 是 **falsy 判定**（0/'' 都视为未传 -> None）",
          client.post(f"{base}/{normal_prop}/generate-image",
                      json={"config_id": 0}).status_code == 200
          and _CALLS[-1]["configId"] is None
          and client.post(f"{base}/{normal_prop}/generate-image",
                          json={"config_id": 7}).status_code == 200
          and _CALLS[-1]["configId"] == 7)
    check("物品图: negativePrompt 取物品自己的（空则 None）",
          _CALLS[-1]["negativePrompt"] is None)
    props_router.generate_image = _fail_with_empty_message  # type: ignore[assignment]
    empty_msg = client.post(f"{base}/{normal_prop}/generate-image", json={})
    props_router.generate_image = _fake_generate_image  # type: ignore[assignment]
    check("物品图: 服务抛错 -> 400，文案空则兜底「生成失败」",
          empty_msg.status_code == 400 and empty_msg.json()["message"] == "生成失败",
          empty_msg.json())

    # ================= scenes =================
    sbase = "/api/v1/scenes"
    _CALLS.clear()
    check("场景图: id 非法 -> 404 Invalid scene id",
          client.post(f"{sbase}/abc/generate-image").status_code == 404)
    check("场景图: 场景不存在 -> **400** Scene not found（与 props 的 404 不同）",
          client.post(f"{sbase}/999999/generate-image", json={}).json()
          == {"code": 400, "message": "Scene not found"})
    check("场景图: episode_id 不存在 / 非数字 -> 400 Episode not found",
          client.post(f"{sbase}/{base_scene}/generate-image",
                      json={"episode_id": 999999}).json()
          == {"code": 400, "message": "Episode not found"}
          and client.post(f"{sbase}/{base_scene}/generate-image",
                          json={"episode_id": "abc"}).json()
          == {"code": 400, "message": "Episode not found"})
    scene_ok = client.post(f"{sbase}/{base_scene}/generate-image",
                           json={"episode_id": episode_id}).json()["data"]
    call = _CALLS[-1]
    check("场景图: 成功 -> image_generation_id + configId 取集上的",
          scene_ok == {"image_generation_id": 4321} and call["configId"] == 11, (scene_ok, call["configId"]))
    check("场景图: `scene.prompt` 非空 -> 直接用 prompt（**地点/时间不并进**，原 TS 如此）",
          "木桌、灯笼" in call["prompt"] and "客栈大堂" not in call["prompt"],
          call["prompt"][:80])
    check("场景图: `scene.prompt` 为空 -> 退回「地点 + 时间氛围」",
          client.post(f"{sbase}/{city_scene}/generate-image", json={}).status_code == 200
          and "城门口" in _CALLS[-1]["prompt"] and "日 lighting and atmosphere" in _CALLS[-1]["prompt"],
          _CALLS[-1]["prompt"][:80])
    check("场景图: 负向词回退链 —— 无用户/场景负向则用场景默认负向",
          call["negativePrompt"] == build_scene_negative_prompt("realistic"))
    check("场景图: **跨集一致参考图** = 同剧同地点的其他场景图（自身排除、软删排除）",
          sorted(call["referenceImages"]) == ["static/images/a.png", "static/images/b.png"],
          call["referenceImages"])
    check("场景图: 参考图**最多 2 张**（第三个同地点场景不进）",
          len(call["referenceImages"]) == 2 and "static/images/c.png" not in call["referenceImages"])
    check("场景图: 地点为空的场景 **不注入任何参考图**（防串图）",
          client.post(f"{sbase}/{no_location}/generate-image", json={}).status_code == 200
          and _CALLS[-1]["referenceImages"] is None)
    check("场景图: 地点会 trim 后比较（` 客栈大堂 ` 也算同地点）",
          client.post(f"{sbase}/{same_location2}/generate-image", json={}).status_code == 200
          and sorted(_CALLS[-1]["referenceImages"]) == ["static/images/a.png",
                                                        "static/images/c.png"],
          _CALLS[-1]["referenceImages"])
    with engine.begin() as conn:
        status_after = conn.execute(select(scenes.c.status)
                                    .where(scenes.c.id == base_scene)).first()[0]
    check("场景图: 成功路径**不改状态**（仍是 process，等出图回调收尾）",
          status_after == "processing", status_after)
    check("场景图: body.prompt / custom_prompt / 构建器 三级回退（body 优先）",
          client.post(f"{sbase}/{base_scene}/generate-image",
                      json={"prompt": "custom"}).status_code == 200
          and _CALLS[-1]["prompt"] == "custom")
    scenes_router.generate_image = _fail_with_empty_message  # type: ignore[assignment]

    async def _boom(conn, params):  # noqa: ANN001
        _CALLS.append(params)
        raise RuntimeError("生成炸了")

    scenes_router.generate_image = _boom  # type: ignore[assignment]
    failed = client.post(f"{sbase}/{base_scene}/generate-image", json={})
    scenes_router.generate_image = _fake_generate_image  # type: ignore[assignment]
    with engine.begin() as conn:
        failed_status = conn.execute(select(scenes.c.status)
                                     .where(scenes.c.id == base_scene)).first()[0]
    check("场景图: 失败 -> 400 + 场景置 **failed**（与原 TS 的补偿写一致）",
          failed.status_code == 400 and failed.json()["message"] == "生成炸了"
          and failed_status == "failed", (failed.json(), failed_status))

    # ================= 汇总 =================
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
