"""S7 自检：重生成镜头帧 ``POST /storyboards/{id}/regenerate-frame``。

锁三条容易看错的语义：**帧类型白名单**（只有 ``last_frame``/``keyframe`` 被认，其余一律
``first_frame``）、**帧画面内容的优先级**（请求体 ``prompt`` > 库里对应的帧 prompt）、
**最终提示词 = 标准构建器 + 帧画面内容 + 帧提示词**（三段用 ``, `` 连接，空段丢弃）。

``generate_image`` 打桩 ⇒ 不打网络、不起任务。

运行::

    ./.venv/Scripts/python.exe tests/regenerate_frame_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="rgf_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ai_service_configs, dramas, episodes, storyboards  # noqa: E402
from app.response import now  # noqa: E402
from app.routers import storyboards as sb_api  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SEEN: dict[str, object] = {}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    async def fake_generate_image(conn, params):
        _SEEN["params"] = params
        return 777

    sb_api.generate_image = fake_generate_image  # type: ignore[assignment]

    stamp = now()
    with engine.begin() as conn:
        cfg = int(conn.execute(ai_service_configs.insert().values(
            name="本地绘图", service_type="image", provider="local-sd",
            base_url="http://localhost:7860", api_key="",  # ⚠️ api_key 也是 NOT NULL（本地 provider 用空串）
            model="sdxl-base", priority=1, created_at=stamp, updated_at=stamp)).lastrowid)
        drama_id = int(conn.execute(dramas.insert().values(
            title="帧剧", created_at=stamp, updated_at=stamp)).lastrowid)
        ep = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            image_config_id=cfg, created_at=stamp, updated_at=stamp)).lastrowid)
        sb = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=1, title="镜一",
            description="雨夜街头，主角撑伞回望",
            location="旧城雨巷", shot_type="中景", angle="平视",
            first_frame_prompt="库里首帧提示", last_frame_prompt="库里尾帧提示",
            keyframe_prompt="库里关键帧提示", negative_prompt="库里负面词",
            created_at=stamp, updated_at=stamp)).lastrowid)

    client = TestClient(app)

    # ── 白名单 ──
    r = client.post(f"/api/v1/storyboards/{sb}/regenerate-frame", json={})
    check("白名单: 不传 frame_type -> first_frame（默认，不报错）",
          r.status_code == 200 and r.json()["data"]["frame_type"] == "first_frame",
          r.text[:140])
    first_params = dict(_SEEN["params"])  # type: ignore[arg-type]
    check("白名单: 未知值也当 first_frame（原 TS 是真值判断，不是白名单报错）",
          client.post(f"/api/v1/storyboards/{sb}/regenerate-frame",
                      json={"frame_type": "abc"}).json()["data"]["frame_type"] == "first_frame")
    check("first_frame: 帧内容取**库里 first_frame_prompt**（未传 body.prompt）",
          "库里首帧提示" in str(first_params["prompt"]), first_params["prompt"][-160:])
    check("first_frame: 帧提示词含 opening frame / establishing the scene",
          "opening frame, establishing the scene" in str(first_params["prompt"]))
    check("params: generate_image 收到 frameType=first_frame + model/force/configId 透传",
          first_params["frameType"] == "first_frame" and first_params["configId"] == cfg
          and "model" in first_params and "force" in first_params, first_params)

    # ── last_frame ──
    r = client.post(f"/api/v1/storyboards/{sb}/regenerate-frame",
                    json={"frame_type": "last_frame"})
    last_params = dict(_SEEN["params"])  # type: ignore[arg-type]
    check("last_frame: 帧类型回显 + 帧内容取 last_frame_prompt + closing frame 提示",
          r.json()["data"]["frame_type"] == "last_frame"
          and "库里尾帧提示" in str(last_params["prompt"])
          and "closing frame, final composition" in str(last_params["prompt"]),
          r.json()["data"])

    # ── keyframe ──
    client.post(f"/api/v1/storyboards/{sb}/regenerate-frame", json={"frame_type": "keyframe"})
    key_params = dict(_SEEN["params"])  # type: ignore[arg-type]
    check("keyframe: 帧内容取 keyframe_prompt + mid-action keyframe 提示",
          "库里关键帧提示" in str(key_params["prompt"])
          and "mid-action keyframe, subject mid-motion" in str(key_params["prompt"]))

    # ── body.prompt 优先级最高 ──
    client.post(f"/api/v1/storyboards/{sb}/regenerate-frame",
                json={"frame_type": "last_frame", "prompt": "请求体帧描述"})
    override = dict(_SEEN["params"])  # type: ignore[arg-type]
    check("优先级: **请求体 prompt 覆盖**库里帧 prompt（两者不共存）",
          "请求体帧描述" in str(override["prompt"])
          and "库里尾帧提示" not in str(override["prompt"]), override["prompt"][-160:])

    # ── 提示词拼接：三段用 ", " 连接 ──
    prompt = str(override["prompt"])
    check("拼接: 段落顺序 = 构建器基底 → 帧内容 → 帧提示词（三段独立）",
          prompt.index("雨夜街头") < prompt.index("请求体帧描述")
          < prompt.index("closing frame"), prompt[:200])
    check("拼接: 基底注入了角色/场景链（含 location 与景别/机位）",
          "旧城雨巷" in prompt and ("中景" in prompt or "medium shot" in prompt))

    # ── 参考图与负面词 ──
    client.post(f"/api/v1/storyboards/{sb}/regenerate-frame",
                json={"reference_images": ["static/images/a.png"],
                      "negative_prompt": "请求体负面词"})
    ref_params = dict(_SEEN["params"])  # type: ignore[arg-type]
    check("参考图: 请求体 reference_images **优先**（非空即用，不查库）",
          ref_params["referenceImages"] == ["static/images/a.png"], ref_params["referenceImages"])
    check("负面词: 请求体 negative_prompt 优先于库里",
          ref_params["negativePrompt"] == "请求体负面词", ref_params["negativePrompt"])
    client.post(f"/api/v1/storyboards/{sb}/regenerate-frame", json={})
    fallback_params = dict(_SEEN["params"])  # type: ignore[arg-type]
    check("负面词: 未传时回退**库里 negative_prompt**", 
          fallback_params["negativePrompt"] == "库里负面词", fallback_params["negativePrompt"])
    check("参考图: 未传且库里没有 -> 空列表（仍传给 generate_image）",
          fallback_params["referenceImages"] == [], fallback_params["referenceImages"])

    # ── 错误分支 ──
    check("错误: 非法 id -> 404 'Invalid storyboard id'",
          client.post("/api/v1/storyboards/abc/regenerate-frame", json={}).status_code == 404)
    check("错误: 镜头不存在 -> 404 '镜头不存在'",
          client.post("/api/v1/storyboards/999999/regenerate-frame", json={}).json()
          == {"code": 404, "message": "镜头不存在"})

    async def boom(conn, params):
        raise RuntimeError("生图失败")

    sb_api.generate_image = boom  # type: ignore[assignment]
    bad = client.post(f"/api/v1/storyboards/{sb}/regenerate-frame", json={})
    check("错误: 生图抛错 -> 400 信封（带错误原文）",
          bad.status_code == 400 and bad.json()["message"] == "生图失败", bad.json())
    sb_api.generate_image = fake_generate_image  # type: ignore[assignment]

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
