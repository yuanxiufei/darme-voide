"""自检：**H3 形态（FL2VA / Ref2VA）决策的接线**（2026-09-24 ✓）。

背景（「能力接不出去不算功能」✗ 的第二处）：``conditioning.select_h3_task`` 把那条**硬约束**
写得明明白白（**有参考素材 ⇒ 只能 Ref2VA** ✓✗），但它此前**只有自检在调** ✗✗ ——
生产里的形态决策在别处，而且是「``scene_type`` 正则 + ``or`` 兜底」✓✗：

* 后端 ``video_adapters.resolve_h3_checkpoint`` ✓：``action/silent/…`` ⇒ FL2VA，
  **完全不看有没有参考素材** ✓✗ ⇒ ``scene_type=action`` 且带角色声线样本的镜头会挑到 FL2VA 权重、
  参考素材**静默失效** ✓✗；
* 8765 薄封装 ``server.py`` ✓：**又按场景猜了一遍** ✓✗，还**从来没读过 ``body.model``** ✗✗
  ⇒ 后端算出来的权重**静默落空**、永远用模板自带那一个 ✓✗。

本套钉的判据：
1. ⭐⭐ 有参考素材 ⇒ **ref2va**（即使场景偏好 FL2VA ✓✗）；
2. ⭐⭐ 「硬首帧」与「参考素材」互斥 ⇒ 引擎**拒** ✗，接线侧 ``blocked=True`` + **带理由** ✓
   （回退不炸流程 ✓，但**不许装作按规则选的** ✓✗）；
3. ⭐⭐ 薄封装**尊重调用方给的权重** ✗✗（且要用 ComfyUI 的 ``/models/diffusion_models`` **核过** ✓，
   核不上就**告警 + 兜底** ✓ —— 不许静默换模型 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/h3_form_routing_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="h3form_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.core.models import video_generations  # noqa: E402
from app.core.response import now  # noqa: E402
from app.local_services.h3 import server as h3srv  # noqa: E402
from app.services import vendor_errors  # noqa: E402
from app.services import video_generation as vg  # noqa: E402
from app.services.adapters import get_video_adapter  # noqa: E402
from app.services.adapters.video_adapters import plan_h3_form, resolve_h3_checkpoint  # noqa: E402
from app.services.engine import conditioning  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def run(coro):
    return asyncio.run(coro)


def refusal(fn, *args: object, **kwargs: object) -> str | None:
    try:
        fn(*args, **kwargs)
    except conditioning.H3TaskError as err:
        return str(err)
    return None


MAP = {"fl2va": "MiniMaxH3/fl2va_int8", "ref2va": "MiniMaxH3/ref2va_int8"}
CFG = {"model": "h3-base", "settings": {"checkpoint_map": MAP}}


# ══════════════════════════════════════════════════════════════════════════
# ① 引擎规则（新补的互斥那条 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_engine_rules() -> None:
    both = refusal(conditioning.select_h3_task, primary_model_kind="ref2va",
                   has_optional_fl2va=True, has_references=True, has_first_frame=True)
    check("① ⭐⭐ 「硬首帧」+「参考素材」**同时要** ⇒ 拒 ✗✗（此前会挑 FL2VA ⇒ 参考被静默失效 ✓✗）",
          both is not None and "同时" in both and "二选一" in both, both)
    check("①′ 拒的时候要**说清两个都不能丢** ✓ 并提示「自动挂上的别顶掉硬设定」✗",
          both is not None and "硬首帧" in both and "参考素材" in both and "自动" in both)
    check("①″ 只有其中一条 ⇒ **照常**（不是把这条规则写成了「一律拒」✗）",
          conditioning.select_h3_task(primary_model_kind="ref2va", has_optional_fl2va=True,
                                      has_references=True) == "ref2va"
          and conditioning.select_h3_task(primary_model_kind="ref2va", has_optional_fl2va=True,
                                          has_first_frame=True) == "fl2va")


# ══════════════════════════════════════════════════════════════════════════
# ② 规划层（纯函数 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_plan() -> None:
    single = plan_h3_form({"model": "only-one"}, {"prompt": "p"})
    check("② 没配 ``checkpoint_map`` ⇒ 单模型模式 ✓（``kind=None`` ✓ 用 base ✓ 不是判错 ✗）",
          single["kind"] is None and single["checkpoint"] == "only-one"
          and single["blocked"] is False, single)

    action = {"sceneType": "action", "referenceMode": "none"}
    plain = plan_h3_form(CFG, action)
    check("②′ 场景偏好（action）+ 无参考 ⇒ **fl2va** ✓（旧行为的正面情形必须保住 ✓✗）",
          plain["kind"] == "fl2va" and plain["checkpoint"] == MAP["fl2va"]
          and plain["blocked"] is False, plain)

    with_audio = plan_h3_form(CFG, {**action, "referenceAudioUrls": '["a.mp3"]'})
    check("②″ ⭐⭐ 同一场景**带参考音频** ⇒ 改走 **ref2va** ✗✗（这就是原先被绕过的硬约束 ✓）",
          with_audio["kind"] == "ref2va" and with_audio["checkpoint"] == MAP["ref2va"]
          and with_audio["has_references"] is True, with_audio)

    as_list = plan_h3_form(CFG, {**action, "referenceImageUrls": ["https://cdn.test/a.png"]})
    check("②‴ ⭐ 参考素材**两种形态都认** ✗✗（落库是 JSON 字符串 ✓、直接调用是列表 ✓ —— "
          "只认一种 ⇒ 另一种被判成「没有」⇒ 硬约束又绕过去了 ✓✗）",
          as_list["has_references"] is True and as_list["kind"] == "ref2va", as_list)
    check("②⁗ 空列表**不算**参考素材 ✓（``[]`` 不是诉求 ✓，别把它判成「有参考」✗）",
          plan_h3_form(CFG, {**action, "referenceImageUrls": []})["has_references"] is False)
    check("②⁵ 非首尾帧的 ``imageUrl`` **不算首帧诉求** ✗✗（每张 I2V 都有首帧图 ⇒ "
          "拿它当诉求会把所有 I2V 都判成「必须 FL2VA」✓✗）",
          plan_h3_form(CFG, {**action, "imageUrl": "static/x.png"})["has_first_frame"] is False)

    tail = plan_h3_form(CFG, {"sceneType": "dialogue", "referenceMode": "first_last"})
    check("②⁶ 首尾帧目标（``referenceMode=first_last``）⇒ **fl2va** ✓（锁定结束画面 ✓）",
          tail["kind"] == "fl2va" and tail["has_first_frame"] is True, tail)

    conflict = plan_h3_form(CFG, {"sceneType": "dialogue", "referenceMode": "first_last",
                                  "referenceAudioUrls": '["a.mp3"]'})
    check("②⁷ ⭐⭐ 互斥诉求 ⇒ ``blocked=True`` + **理由**（引擎写的可行动文案 ✓）+ "
          "``kind=None``（**不装作按规则选的** ✓✗）",
          conflict["blocked"] is True and conflict["kind"] is None
          and "二选一" in str(conflict["reason"]), conflict)
    check("②⁸ 但 ``checkpoint`` **仍给一个** ✓（回退不炸既有流程 ✓）：回退口径 = **有 Ref2VA 就用它** ✓✗"
          "（参考素材只有它能表达 ✓，首尾帧字段照样在请求体里 ✓ ⇒ 这一侧没丢内容 ✓）",
          conflict["checkpoint"] == MAP["ref2va"], conflict)

    only_ref = {"model": "m", "settings": {"checkpoint_map": {"ref2va": MAP["ref2va"]}}}
    ref_plan = plan_h3_form(only_ref, {"sceneType": "action", "referenceAudioUrls": '["a.mp3"]'})
    check("②⁹ 只有 Ref2VA 权重 + 有参考 ⇒ 正常走 ref2va ✓（不 blocked ✓）",
          ref_plan["kind"] == "ref2va" and ref_plan["blocked"] is False, ref_plan)
    tail_plan = plan_h3_form(only_ref, {"referenceMode": "first_last"})
    check("②¹⁰ 只有 Ref2VA 却要首尾帧 ⇒ ``blocked`` + 理由 ✓，但 checkpoint **还是 ref2va** ✓"
          "（**输出与旧实现一致** ✓ —— 差别只在这回它**说出来了** ✓✗）",
          tail_plan["blocked"] is True and tail_plan["checkpoint"] == MAP["ref2va"]
          and "FL2VA" in str(tail_plan["reason"]), tail_plan)

    only_fl = {"model": "m", "settings": {"checkpoint_map": {"fl2va": MAP["fl2va"]}}}
    fl_plan = plan_h3_form(only_fl, {"sceneType": "dialogue", "referenceAudioUrls": '["a.mp3"]'})
    check("②¹¹ 只有 FL2VA 却有参考素材 ⇒ ``blocked`` + 理由含「丢素材」✓（硬约束在这一侧 ✓）",
          fl_plan["blocked"] is True and "丢素材" in str(fl_plan["reason"]), fl_plan)


# ══════════════════════════════════════════════════════════════════════════
# ③ 适配器接线（请求真的带上了规划结果 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_adapter() -> None:
    adapter = get_video_adapter("minimax")
    record = {"prompt": "p", "sceneType": "action", "referenceMode": "none",
              "referenceAudioUrls": '["a.mp3"]', "duration": 5, "aspectRatio": "16:9"}
    request = adapter.build_generate_request(CFG, record)
    check("③ ⭐ 请求体里的 ``model`` == 规划出来的权重 ✓（不是另一处再猜一遍 ✓✗）",
          request["body"]["model"] == MAP["ref2va"], request["body"]["model"])
    check("③′ ``resolve_h3_checkpoint`` 与规划**同源** ✓（旧的公开入口还在，但不再自己判 ✓）",
          resolve_h3_checkpoint(CFG, record) == plan_h3_form(CFG, record)["checkpoint"])
    check("③″ ``formPlan`` 随请求描述出去 ✓ 但 **body 里没有它** ✗✗（不许发到厂商那边 ✓）",
          isinstance(request.get("formPlan"), dict)
          and "formPlan" not in request["body"], list(request))


# ══════════════════════════════════════════════════════════════════════════
# ④ 8765 薄封装：尊重调用方的权重 ✓
# ══════════════════════════════════════════════════════════════════════════
class FakeComfy:
    """假 ComfyUI 客户端 ✓（鸭子类型 ✓）——**只**实现 ``models`` ✓（与 ``accel`` 那条接缝同套手法 ✓）。"""

    def __init__(self, files: list[str] | None = None, boom: bool = False) -> None:
        self.files = files or []
        self.boom = boom
        self.calls: list[str] = []

    def models(self, folder: str) -> list[str]:
        self.calls.append(folder)
        if self.boom:
            raise RuntimeError("connect refused")
        return self.files


def case_wrapper_checkpoint() -> None:
    files = ["MiniMaxH3/fl2va_int8", "MiniMaxH3/ref2va_int8"]
    task: dict[str, Any] = {"body": {"scene_type": "action"}}

    chosen = h3srv._pick_checkpoint(FakeComfy(files), task, "MiniMaxH3/ref2va_int8", MAP)
    check("④ ⭐⭐ 调用方给的权重**在目录里 ⇒ 就用它** ✗✗（此前 ``body.model`` 根本没被读过 ✓✗，"
          "后端按参考素材选出来的权重**静默落空** ✓✗）；⚠️ 只剩「模板是 FL2VA 形态」那条**该有的**告警 ✓",
          chosen == "MiniMaxH3/ref2va_int8"
          and not any("不在" in w or "没核" in w for w in (task.get("warnings") or [])), (chosen, task))

    task2: dict[str, Any] = {"body": {"scene_type": "action"}}
    bad = h3srv._pick_checkpoint(FakeComfy(files), task2, "MiniMaxH3/not_there.safetensors", MAP)
    check("④′ 目录里**没有** ⇒ 不用它 ✓ 且**告警**（不静默换别的权重 ✗✗），并落到兜底 ✓",
          bad == MAP["fl2va"] and any("not_there" in w for w in task2["warnings"]), (bad, task2))

    task3: dict[str, Any] = {"body": {"scene_type": "action"}}
    unchecked = h3srv._pick_checkpoint(FakeComfy(boom=True), task3, "MiniMaxH3/ref2va_int8", MAP)
    check("④″ 清单取不到 ⇒ **说「没核」** ✗（不是默默放行 ✓✗）且**照样走兜底** ✓"
          "（提前 return 会把兜底也跳掉 ✓✗）",
          unchecked == MAP["fl2va"] and any("没核" in w for w in task3["warnings"]), (unchecked, task3))

    task4: dict[str, Any] = {"body": {"scene_type": "action"}}
    legacy = h3srv._pick_checkpoint(FakeComfy(files), task4, "", MAP)
    check("④‴ 调用方**没给**权重 ⇒ 退回 ``checkpoint_map`` 的老口径 ✓（向后兼容 ✓✗）",
          legacy == MAP["fl2va"] and task4.get("warnings") in (None, []), (legacy, task4))


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 接线可达：真跑一次视频提交，看日志里有没有那份决策 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_video_logging() -> None:
    logged: list[tuple[str, str, dict]] = []
    original = (vg.log_task_payload, vg.log_task_warn)
    vg.log_task_payload = lambda cat, action, meta=None: logged.append(("payload", action, meta or {}))
    vg.log_task_warn = lambda cat, action, meta=None: logged.append(("warn", action, meta or {}))

    def handler(request: httpx.Request) -> httpx.Response:
        # ⚠️ 故意回一个**解析不出任务**的响应 ✓：既不进轮询（会拖几十秒 ✓✗），
        #    也照样走完「建请求 → 记日志」这一段 ✓ —— 要验的正是**发请求之前**那步 ✓
        return httpx.Response(200, json={"nope": True})

    def submit(video_id: int, config: dict[str, Any]) -> None:
        try:
            run(vg._process_video_generation(video_id, config))
        except Exception:  # noqa: BLE001 —— 末次尝试会把「解析不出任务」抛上来 ✓ 与本次判据无关 ✓
            pass

    try:
        vendor_errors._vendor_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = {"provider": "minimax", "baseUrl": "http://api.minimax.test", "apiKey": "k",
                  "model": "MiniMaxH3/fl2va_int8", "models": ["MiniMaxH3/fl2va_int8"],
                  "settings": {"checkpoint_map": MAP}}

        def insert(**values: Any) -> int:
            with engine.begin() as conn:
                row = conn.execute(video_generations.insert().values(
                    created_at=now(), updated_at=now(), status="processing", provider="minimax",
                    model="MiniMaxH3/fl2va_int8", prompt="p", duration=5, aspect_ratio="16:9",
                    **values,
                ))
            return int(row.inserted_primary_key[0])

        ok_id = insert(scene_type="action", reference_mode="multiple",
                       reference_image_urls='["https://cdn.test/a.png"]')
        submit(ok_id, config)
        forms = [meta for kind, action, meta in logged if action == "h3 form"]
        check("⑤ ⭐ 真跑提交链路 ⇒ 决策**落进任务日志** ✓（``kind`` 可在日志里核对 ✓）",
              len(forms) == 1 and forms[0]["kind"] == "ref2va", forms)

        logged.clear()
        bad_id = insert(scene_type="dialogue", reference_mode="first_last",
                        reference_image_urls='["https://cdn.test/a.png"]',
                        reference_audio_urls='["https://cdn.test/v.mp3"]')
        submit(bad_id, config)
        blocked = [meta for kind, action, meta in logged if action == "h3-form-blocked"]
        check("⑤′ ⭐⭐ 互斥那一单 ⇒ 日志里是 ``h3-form-blocked`` + 理由 ✓✗（不静默挑一个 ✓）",
              len(blocked) == 1 and "二选一" in str(blocked[0]["reason"])
              and blocked[0]["hasReferences"] is True and blocked[0]["hasFirstFrame"] is True, blocked)
    finally:
        vg.log_task_payload, vg.log_task_warn = original


def main() -> int:
    case_engine_rules()
    case_plan()
    case_adapter()
    case_wrapper_checkpoint()
    case_video_logging()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
