"""自检：引擎**画布计划层 + 用途档位表**（零依赖 ✓ 2026-11-01）。

为什么这套值得写：画布错法是**不报错**的 ✓✗ —— 参数全对、图"就是不像"（LoRA 与负向词是在
``1344×768`` 上调的 ✓，换到 ``1792×1024`` 会哪儿都不对 ✓），而回执上看不出是谁定的 ✓✗。
所以这套钉的不是"数字对不对"✓，而是**三条纪律**：
① 档位表里的数**能追溯到工作流**（每档带文件名 ✓）且**与采样器/调度器注册表自洽** ✗（拼错就红 ✓）；
② 给了档位就**精确复现那一张画布** ✓（不是"再拿 MP 算一遍" ✗ —— ``1344×768`` 与 ``16:9`` 不闭合 ✓）；
③ **没点名就一点都不变** ✓（旧路径逐位相同 ✓ —— 这是"零回归"的硬证据 ✗ 不是口头承诺 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_canvas_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import canvas as canvas_mod  # noqa: E402
from app.services.engine import geometry, pipeline as pipe  # noqa: E402
from app.services.engine import profiles as profiles_mod  # noqa: E402
from app.services.engine import sampler as sampler_mod  # noqa: E402
from app.services.engine import schedules as schedules_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def req(**overrides: Any) -> pipe.GenerationRequest:
    base: dict[str, Any] = {"prompt": "雨夜霓虹街头，主角回头", "seed": 7, "steps": 6,
                            "stage": "qwen_image",
                            "outputs_dir": tempfile.mkdtemp(prefix="engine_canvas_")}
    base.update(overrides)
    return pipe.GenerationRequest(**base)


def raises(fn: Any, needle: str = "") -> str:
    """跑一下，返回错误文案 ✓（没抛 ⇒ 空串 ✓ 便于 ``check`` 一眼看出红在哪 ✗）。"""
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else ""
    return ""


# ══════════════════════════════════════════════════════════════════════════
# ① 档位表自洽（拼错、漏字段、口径漂移都在这里红 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_profiles_table() -> None:
    check("① 档位表非空且 id 唯一", len(profiles_mod.PROFILES) >= 10, len(profiles_mod.PROFILES))
    bad_prefix = [pid for pid, item in profiles_mod.PROFILES.items()
                  if pid.split("/", 1)[0] != item.stage]
    check("② 每档 id 前缀 == stage（``qwen_image/xxx`` ✓ 不许挂错族 ✗）", not bad_prefix, bad_prefix)

    unknown_stage = [item.id for item in profiles_mod.PROFILES.values()
                     if item.stage not in canvas_mod.STAGE_CANVAS]
    check("③ 每档的 stage 都在 ``STAGE_CANVAS`` 里（否则永远算不出网格 ✓✗）",
          not unknown_stage, unknown_stage)

    bad_grid = [item.id for item in profiles_mod.PROFILES.values()
                if item.width is not None
                and (item.width % canvas_mod.STAGE_CANVAS[item.stage][0]
                     or item.height % canvas_mod.STAGE_CANVAS[item.stage][0])]
    check("④ 每档画布都是该 stage 吸附步长的整数倍 ✓（否则档位本身就得先被吸 ✓✗）",
          not bad_grid, bad_grid)

    bad_sampler = [item.id for item in profiles_mod.PROFILES.values()
                   if item.sampler is not None and item.sampler not in sampler_mod.SAMPLERS]
    check("⑤ 每档的 sampler 都在注册表里 ✓（拼错的档位跑不了 ✓ 但要**现在**就红 ✗ 不是跑到才红 ✗）",
          not bad_sampler, bad_sampler)

    bad_schedule = [item.id for item in profiles_mod.PROFILES.values()
                    if item.schedule is not None and item.schedule not in schedules_mod.SCHEDULES]
    check("⑥ 每档的 schedule 都在注册表里 ✓", not bad_schedule, bad_schedule)

    no_source = [item.id for item in profiles_mod.PROFILES.values() if not item.source]
    check("⑦ 每档都写了**出处工作流文件名** ✓（数值对不上时能回去核 ✗ 不靠记忆 ✓）",
          not no_source, no_source)

    partial = [item.id for item in profiles_mod.PROFILES.values()
               if (item.width is None) != (item.height is None)]
    check("⑧ 画布要么两个都有、要么两个都没有 ✓（单边有 = 表本身就在猜 ✗）",
          not partial, partial)

    stages_match = set(canvas_mod.STAGE_CANVAS) == set(profiles_mod.STAGES)
    pipeline_stages = {pipe.VIDEO_STAGE, pipe.IMAGE_STAGE, pipe.ZIMAGE_STAGE,
                       pipe.QWEN21_STAGE, pipe.QWEN_IMAGE_STAGE, pipe.KREA2_STAGE}
    check("⑨ ``STAGE_CANVAS`` 与 ``profiles.STAGES`` 同集 ✓", stages_match, canvas_mod.STAGE_CANVAS)
    check("⑩ 这两个集又与 ``pipeline`` 的 stage 常量**同值** ✓（手抄字符串必漂移 ✗）",
          set(canvas_mod.STAGE_CANVAS) == pipeline_stages, pipeline_stages)

    entry = profiles_mod.PROFILES["qwen_image/character-sheet"]
    check("⑪ 定妆板档：2048×1536 / 30 步 / cfg 5.5（工作流实测值 ✓ 不是文档抄的 ✗）",
          (entry.width, entry.height, entry.steps, entry.cfg) == (2048, 1536, 30, 5.5), entry)
    last = profiles_mod.PROFILES["qwen_image/storyboard-last"]
    check("⑫ 尾帧档带 ``denoise=0.55`` ✓（这一格以前在引擎里根本没有 ✗）",
          last.denoise == 0.55, last.denoise)
    ref = profiles_mod.PROFILES["qwen_image/reference-sheet"]
    check("⑬ Edit 2511 的调度器是 ``normal`` ✓（flow 模型上的真口径 ✓ 不是统一 simple ✗）",
          ref.schedule == "normal", ref.schedule)
    check("⑬¹ ``requiresInitImage`` 是**派生量** ✓：只有 ``denoise<1`` 的档为真 ✓"
          "（尾帧档 ✓；定妆板 / H3 那两档 denoise=None ⇒ 假 ✓ —— 少这条前端照抄参数必撞 400 ✗）",
          last.requires_init_image is True
          and profiles_mod.PROFILES["qwen_image/character-sheet"].requires_init_image is False
          and profiles_mod.PROFILES["h3/storyboard-768p"].requires_init_image is False,
          (last.to_dict().get("requiresInitImage"),
           profiles_mod.PROFILES["h3/storyboard-768p"].to_dict().get("requiresInitImage")))


# ══════════════════════════════════════════════════════════════════════════
# ② 查档（命中即精确复现；不认识就报错列合法 ✗ 不回落 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_lookup() -> None:
    planned = canvas_mod.plan_canvas("qwen_image", target="character-sheet")
    check("⑭ 命中定妆板档 ⇒ 2048×1536 ✓ 且 source 写明取自哪一档 ✓",
          (planned.width, planned.height) == (2048, 1536)
          and planned.source == "profile:qwen_image/character-sheet", planned.to_dict())
    check("⑮ 回执带 ``profile`` 与「取自档位」的说明 ✓（读的人不用猜这数是谁定的 ✓）",
          planned.profile == "qwen_image/character-sheet"
          and any("取自档位" in note for note in planned.notes), planned.notes)

    scene = canvas_mod.plan_canvas("qwen_image", target="scene-card")
    check("⑯ 场景卡档 ⇒ 1344×768 ✓（**不是**拿 1.03 MP 重算的 1360×768 ✓✗）",
          (scene.width, scene.height) == (1344, 768), (scene.width, scene.height))

    tight = canvas_mod.plan_canvas("qwen_image", target="scene-card", ratio="16:9")
    check("⑰ 1344×768（1.75）与 16:9（1.778）差 1.6% ⇒ **保留档位像素** ✓ 并说明 ✓",
          (tight.width, tight.height) == (1344, 768)
          and any("保留档位像素" in note for note in tight.notes), tight.notes)

    wide = canvas_mod.plan_canvas("qwen_image", target="character-sheet", ratio="16:9")
    check("⑱ 4:3 的档位遇到 16:9（差 33% ✓）⇒ **按档位面积重算** ✓ 并写明 ✓",
          wide.source == "profile:qwen_image/character-sheet+ratio"
          and wide.width > wide.height and any("重算" in note for note in wide.notes),
          (wide.width, wide.height, wide.notes))

    check("⑲ 未知用途 ⇒ 报错并**列出该 stage 的合法档位** ✗✗（不回落默认档 ✗）",
          bool(raises(lambda: canvas_mod.plan_canvas("qwen_image", target="character-seet"),
                      "scene-card")), "no raise")
    check("⑳ 没给用途名 ⇒ 报错（含合法档位列表）", bool(raises(
        lambda: profiles_mod.resolve("qwen_image", ""), "合法档位")), "no raise")
    check("㉑ 空格容错：`` scene-card `` 能命中 ✓",
          profiles_mod.resolve("qwen_image", " scene-card ").id == "qwen_image/scene-card")

    krea = canvas_mod.plan_canvas("krea2", megapixels=1.0, ratio="1:1")
    check("㉒ Krea2 的 1.00 MP @1:1 必须**正好 1024×1024** ✓（与官方画布逐像素相同 ✓ 旧口径曾落到 1040 ✗）",
          (krea.width, krea.height) == (1024, 1024), (krea.width, krea.height))

    qwen = canvas_mod.plan_canvas("qwen_image", megapixels=1.76, ratio="1:1")
    check("㉓ Qwen 的 1.76 MP @1:1 ⇒ 1328×1328 ✓（**不能**换成 ResolutionSelector 口径 ⇒ 1360 ✗）",
          (qwen.width, qwen.height) == (1328, 1328), (qwen.width, qwen.height))

    video = canvas_mod.plan_canvas("h3", megapixels=0.65, ratio="16:9")
    check("㉔ H3 的 0.65 MP @16:9 ⇒ 1088×608 ✓（32 步长吸附 ✓ 与既有锚点一致 ✓）",
          (video.width, video.height) == (1088, 608), (video.width, video.height))


# ══════════════════════════════════════════════════════════════════════════
# ③ 入口冲突与"不猜"（这是本仓最容易退化的地方 ✓✗）
# ══════════════════════════════════════════════════════════════════════════
def case_refusals() -> None:
    check("㉕ 只给宽不给高 ⇒ 报错 ✗（另一边**不猜** ✓）",
          bool(raises(lambda: canvas_mod.plan_canvas("qwen_image", width=1344), "不猜")), "no raise")
    check("㉖ 同时给像素与用途档 ⇒ 报错（两处口径 ✗）",
          bool(raises(lambda: canvas_mod.plan_canvas("qwen_image", target="scene-card",
                                                     width=1024, height=1024), "两处口径")),
          "no raise")
    check("㉗ 同时给像素与 megapixels ⇒ 报错",
          bool(raises(lambda: canvas_mod.plan_canvas("qwen_image", megapixels=1.0,
                                                     width=1024, height=1024), "两处口径")),
          "no raise")
    check("㉘ 三样都不给 ⇒ 报错 ✗✗（**不编一个默认画布** ✓ 本仓最恨这类静默 ✗）",
          bool(raises(lambda: canvas_mod.plan_canvas("qwen_image"), "不猜")), "no raise")
    check("㉙ 未知 stage ⇒ 报错并列出合法 stage ✗",
          bool(raises(lambda: canvas_mod.plan_canvas("flux", megapixels=1.0), "未知 stage")), "no raise")
    check("㉚ H3 档（画布由导演台出）⇒ 报错**说清为什么没有数** ✗（不是回 0 或 None ✗）",
          bool(raises(lambda: canvas_mod.plan_canvas("h3", target="storyboard-768p"), "没有定画布")),
          "no raise")

    odd = canvas_mod.plan_canvas("qwen_image", width=1343, height=769)
    check("㉛ 非 16 倍数的宽高 ⇒ 就近吸附为 1344×768 ✓ 并**如实写进 notes** ✓（不静默改 ✗）",
          (odd.width, odd.height) == (1344, 768)
          and any("吸附" in note for note in odd.notes), (odd.width, odd.height, odd.notes))


# ══════════════════════════════════════════════════════════════════════════
# ④ 管线接入（点名了才变 ✓ 没点名一点都不变 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_pipeline_wiring() -> None:
    baseline = pipe.build_plan(req())
    check("㉜ 没点名 ⇒ 走旧路径 ✓ 且 ``plan.canvas is None`` ✓（= 「这张不是表定的」✗ 不是「没画布」✗）",
          baseline.canvas is None, baseline.to_dict().get("canvas"))

    same = pipe.build_plan(req(megapixels=1.76, ratio="1:1"))
    check("㉝ 旧路径算出来的仍是 1328×1328 ✓（值没被本轮改动带跑 ✓）",
          (same.width, same.height) == (1328, 1328), (same.width, same.height))

    targeted = pipe.build_plan(req(target="character-sheet"))
    check("㉞ ``target`` ⇒ 画布精确命中 2048×1536 ✓ 且回执里 source/profile 齐全 ✓",
          (targeted.width, targeted.height) == (2048, 1536)
          and (targeted.canvas or {}).get("profile") == "qwen_image/character-sheet",
          targeted.to_dict().get("canvas"))
    check("㉟ 档位说明也进 ``warnings`` ✓（前端不用自己拼文案 ✓）",
          any("取自档位" in item for item in targeted.warnings), targeted.warnings[:2])

    planned = canvas_mod.plan_canvas("krea2", target="scene-card")
    via_canvas = pipe.build_plan(req(stage="krea2", canvas=planned))
    check("㊱ 直接给 ``canvas`` 计划 ⇒ 照计划执行 ✓（1344×768 ✓）",
          (via_canvas.width, via_canvas.height) == (1344, 768), (via_canvas.width, via_canvas.height))

    check("㊲ ``canvas`` 与 ``target`` 同时给 ⇒ 管线报错（两处口径 ✗）",
          bool(raises(lambda: pipe.build_plan(req(target="scene-card", canvas=planned)),
                      "二选一")), "no raise")
    check("㊳ ``canvas`` 塞了个 dict（而不是计划对象）⇒ 报错并指路 ``/engine/canvas-plan`` ✗",
          bool(raises(lambda: pipe.build_plan(req(canvas={"width": 1344, "height": 768})),
                      "canvas-plan")), "no raise")
    check("㊴ 未知用途经管线 ⇒ 报错列出合法档位 ✗",
          bool(raises(lambda: pipe.build_plan(req(target="nope")), "scene-card")), "no raise")

    sdxl_canvas = canvas_mod.plan_canvas("sdxl", width=1344, height=768)
    check("㊵ SDXL **拒绝**画布覆盖 ✗✗（它只能按训练预算收口 ✓ 外面塞 1344×768 会进坏区 ✗）",
          bool(raises(lambda: pipe.build_plan(req(stage="sdxl", canvas=sdxl_canvas)),
                      "训练预算")), "no raise")

    check("㊶ ``denoise=1.0``（默认）⇒ 正常 ✓ 且四格如实（1.0 / steps / σ=1.0 / noise ✓）",
          pipe.build_plan(req()).to_dict().get("initFrom") == "noise"
          and pipe.build_plan(req()).to_dict().get("startSigma") == 1.0)
    # ⭐ denoise 的**两半必须同时给** ✓✗（本轮接上 ✓）：只砍 σ 不换起点 ⇒ 与参考图无关的灰图 ✓✗
    check("㊷ ``denoise=0.55`` **没有** ``initImage`` ⇒ 当场拒 ✗✗（只砍 σ 那一半 = 骗人 ✓✗）",
          bool(raises(lambda: pipe.build_plan(req(denoise=0.55)), "initImage")), "no raise")
    check("㊸ ``denoise=0`` ⇒ 报错（(0,1] 之外 ✓）",
          bool(raises(lambda: pipe.build_plan(req(denoise=0.0)), "denoise")), "no raise")
    check("㊹ 给了 ``initImage`` 却 ``denoise=1`` ⇒ 拒 ✗（起点是纯噪声 ✓ 图会被**整个丢掉** ✗✗）",
          bool(raises(lambda: pipe.build_plan(req(init_image="x.png")), "denoise")), "no raise")
    check("㊹¹ 视频侧 ``denoise<1`` ⇒ 拒 ✗（接上一段走 pin/lastFrame ✓ 单图广播=死帧 ✗）",
          bool(raises(lambda: pipe.build_plan(req(stage="h3", denoise=0.5, init_image="x.png")),
                      "不支持")), "no raise")

    img2img = pipe.build_plan(req(stage="zimage", steps=20, denoise=0.55, init_image="x.png"))
    check("㊹² ``denoise=0.55`` + ``initImage`` ⇒ σ 梯子按 ``int(20/0.55)=36`` 算**再取尾巴** ✓"
          "（实跑仍 20 步 ✓ 起点 σ≈0.55 而不是 1.0 ✓）",
          img2img.to_dict().get("totalSteps") == 36 and img2img.to_dict().get("steps") == 20
          and img2img.to_dict().get("initFrom") == "image"
          and 0.3 < img2img.to_dict().get("startSigma") < 1.0,
          img2img.to_dict().get("startSigma"))
    check("㊹³ 截出来的梯子**单调下降 + 末位 0** ✓（尾巴取错会造出重复 σ / 缺收尾 ✗✗）",
          all(b < a for a, b in zip(img2img.sigmas, img2img.sigmas[1:]))
          and img2img.sigmas[-1] == 0.0 and len(img2img.sigmas) == img2img.steps + 1,
          img2img.sigmas[:3])
    check("㊹⁴ ``normal`` 在 flow 域 ⇒ **σ 等距**（≠ flow 的 t 等距 ✓ 也不回落告警 ✓✗）",
          pipe.build_plan(req(stage="zimage", steps=8, schedule="normal")).sigmas
          != pipe.build_plan(req(stage="zimage", steps=8, schedule="flow")).sigmas
          and not any("EDM" in item for item in
                      pipe.build_plan(req(stage="zimage", schedule="normal")).warnings))


# ══════════════════════════════════════════════════════════════════════════
# ⑤ API（计划接口 + 请求体串联 + capabilities 自述）
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    try:
        from fastapi.testclient import TestClient

        from app.main import app
    except Exception as err:  # noqa: BLE001
        _SKIPS.append(f"FastAPI TestClient 不可用：{type(err).__name__}: {err}")
        return
    client = TestClient(app)

    planned = client.post("/api/v1/engine/canvas-plan",
                          json={"stage": "qwen_image", "target": "character-sheet"})
    data = planned.json().get("data") or {}
    check("㊺ POST /engine/canvas-plan 命中档位 ⇒ 2048×1536 + 该档采样参数 ✓",
          planned.status_code == 200 and (data.get("width"), data.get("height")) == (2048, 1536)
          and (data.get("sampling") or {}).get("cfg") == 5.5,
          (planned.status_code, data.get("width"), data.get("sampling")))
    check("㊻ 回执列出该 stage 的全部可选档位 ✓（前端据此做下拉 ✓ 不靠读代码 ✗）",
          isinstance(data.get("availableTargets"), list) and len(data["availableTargets"]) >= 5,
          len(data.get("availableTargets") or []))

    last_plan = client.post("/api/v1/engine/canvas-plan",
                            json={"stage": "qwen_image", "target": "storyboard-last"})
    last_data = last_plan.json().get("data") or {}
    check("㊻¹ ``denoise<1`` 的档在回执里就标明**要图** ✓（``sampling.requiresInitImage`` ✓ + "
          "下拉项同键 ✓）—— 否则前端照抄 ``denoise=0.55`` 会撞一个看不懂的 400 ✗✗",
          (last_data.get("sampling") or {}).get("requiresInitImage") is True
          and any(item.get("target") == "storyboard-last"
                  and item.get("requiresInitImage") is True
                  for item in (last_data.get("availableTargets") or [])),
          (last_data.get("sampling"),
           [item.get("requiresInitImage") for item in (last_data.get("availableTargets") or [])]))

    missing = client.post("/api/v1/engine/canvas-plan",
                          json={"stage": "qwen_image", "target": "no-such-thing"})
    check("㊼ 未知档位 ⇒ **400 + 列出合法的** ✗（不回落 ✗）",
          missing.status_code == 400 and "scene-card" in missing.text, missing.status_code)

    by_mp = client.post("/api/v1/engine/canvas-plan",
                        json={"stage": "h3", "megapixels": 0.65, "aspectRatio": "16:9"})
    payload = by_mp.json().get("data") or {}
    check("㊽ ``megapixels`` + ``aspectRatio`` ⇒ 1088×608 ✓（前端 camelCase 也收 ✓）",
          (payload.get("width"), payload.get("height")) == (1088, 608),
          (payload.get("width"), payload.get("height")))

    bad = client.post("/api/v1/engine/canvas-plan", json={"stage": "qwen_image", "width": 1024})
    check("㊾ 只给宽 ⇒ 400 ✗（不是 200 + 猜一个高 ✗）",
          bad.status_code == 400 and "不猜" in bad.text, bad.status_code)

    planned_req = client.post("/api/v1/engine/plan", json={
        "stage": "qwen_image", "prompt": "雨夜霓虹", "target": "scene-card", "steps": 8})
    plan_data = planned_req.json().get("data") or {}
    check("㊿ ``target`` 经 **/engine/plan** 生效 ⇒ 回执 canvas 里能看出取自哪一档 ✓",
          planned_req.status_code == 200
          and (plan_data.get("canvas") or {}).get("profile") == "qwen_image/scene-card"
          and (plan_data.get("width"), plan_data.get("height")) == (1344, 768),
          (planned_req.status_code, plan_data.get("canvas")))

    passed_canvas = client.post("/api/v1/engine/plan", json={
        "stage": "krea2", "prompt": "雨夜霓虹",
        "canvas": {"width": 1344, "height": 768, "profile": "krea2/scene-card"}})
    canvas_data = passed_canvas.json().get("data") or {}
    check("⑤¹ ``canvas`` 对象进请求体 ⇒ 生效 ✓（以前这个字段**会在白名单外被静默丢掉** ✗✗）",
          passed_canvas.status_code == 200
          and (canvas_data.get("width"), canvas_data.get("height")) == (1344, 768),
          (passed_canvas.status_code, canvas_data.get("width"), canvas_data.get("height")))

    # ⭐ 本轮补上 HTTP 口 ✓：``GET /api/v1/engine/capabilities`` ✓（以前**没有**路由 ✗ ⇒
    # 前端想知道"认哪些采样器/调度器/档位"只能读代码猜 ✓✗，猜错就是静默按默认跑 ✓✗）。
    http_caps = client.get("/api/v1/engine/capabilities")
    http_data = http_caps.json().get("data") or {}
    check("⑤⁵ ``GET /engine/capabilities`` 真走 HTTP 200 ✓（不再是 ``not_migrated`` 501 ✗✗）"
          "且与函数同源 ✓（不手抄 ✗）",
          http_caps.status_code == 200 and "euler" in (http_data.get("samplers") or [])
          and len(http_data.get("profiles") or []) >= 10,
          http_caps.status_code)

    from app.services.engine import facade as facade_mod  # noqa: PLC0415

    caps = facade_mod.capabilities()
    check("⑤² capabilities 自述 ``samplers`` / ``schedules`` ✓（调用方不用读代码猜 ✓）",
          "euler" in (caps.get("samplers") or []) and "simple" in (caps.get("schedules") or []),
          (caps.get("samplers"), caps.get("schedules")))
    check("⑤³ capabilities 自述 ``profiles`` 与每 stage 的几何纪律 ✓",
          len(caps.get("profiles") or []) >= 10
          and (caps.get("canvasStages") or {}).get("h3", {}).get("grid") == 32,
          (len(caps.get("profiles") or []), caps.get("canvasStages")))
    check("⑤⁴ capabilities 的 ``attentionBackends`` **如实分两栏** ✓（只有 sdpa 可用 ✓ 不装样子 ✗）",
          (caps.get("attentionBackends") or {}).get("available") == ["sdpa"],
          caps.get("attentionBackends"))


def main() -> int:
    case_profiles_table()
    case_lookup()
    case_refusals()
    case_pipeline_wiring()
    case_api()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
