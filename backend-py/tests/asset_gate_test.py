"""S7 自检：**资产清单 + 验收门**（零依赖 ✓ 2026-09-18）。

移植自 ``reference/short-drama-agent``（``asset-to-video-pipeline.md`` ✓）。它守的是一条**产品硬规则**：

> 「不要从被拒或缺失的锚定资产**提交付费视频任务**。」

本项目原来只有「记录/激活资产版本」✓（那是**存**），**没有任何东西阻止**"角色参考图还是
``to-generate`` 就去跑付费视频" ✗ —— 本套就是钉住这道门 ✓。

两条**刻意防假绿**的判据 ✓：

* **``generated`` 不算验收通过** ✗（参考项目把「图生成了」与「人验收了」当两件事 ✓）——
  若这里写成"生成了就算过"，门就形同虚设 ✗；
* **成环必须点名** ✓（只说"有环"没用 ✗ —— 执行器要么死循环要么随机跳过 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/asset_gate_test.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.asset_manifest import (  # noqa: E402
    gate_for_shots, generation_order, validate_manifest)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def asset(asset_id: str, *, kind: str = "character", status: str = "approved",
          needed_by: str = "", deps: list[str] | None = None,
          criteria: bool = True, prompt: bool = True, path: bool = True) -> dict:
    item: dict = {"asset_id": asset_id, "type": kind, "status": status,
                  "name": asset_id, "first_needed_by": needed_by,
                  "dependencies": list(deps or [])}
    if prompt:
        item["prompt"] = f"{asset_id} 的生成提示词"
    if path:
        item["target_path"] = f"video/ep001/assets/{asset_id}.png"
    if criteria:
        item["acceptance_criteria"] = ["同一发型", "同一服装", "脸部清晰"]
    return item


def manifest(assets: list[dict], **extra: object) -> dict:
    return {"episode": 1, "title": "第一集",
            "style_bible": {"visual_style": "真人写实", "palette": "低饱和冷蓝"},
            "assets": assets, **extra}


# ══════════════════════════════════════════════════════════════════════════
# ① 清单校验
# ══════════════════════════════════════════════════════════════════════════
def case_validate() -> None:
    good = validate_manifest(manifest([
        asset("char_shenyan", needed_by="kf001"),
        asset("scene_crime_room", kind="scene", needed_by="kf001"),
    ]))
    check("① ⭐ 全部验收通过 ⇒ **门开着** ✓（canGenerateVideo=True ✓ 无理由 ✓）",
          good.canGenerateVideo is True and good.gateReasons == [] and good.problems == [],
          good.to_dict())

    check("② 清单不是对象 / 没有 assets ⇒ 报问题而**不抛异常** ✓（校验器崩了比校验失败更糟 ✗）",
          validate_manifest("不是对象").problems and validate_manifest({}).problems
          and validate_manifest({"assets": []}).problems, "")

    dup = validate_manifest(manifest([asset("a"), asset("a")]))
    check("③ ⭐ `asset_id` 重复 ⇒ 报问题（引用会指向不确定的那个 ✗）",
          any("重复" in item for item in dup.problems), dup.problems)

    unknown_type = validate_manifest(manifest([asset("bg", kind="background")]))
    check("④ 类型不在六种里 ⇒ 报问题 ✓ 且提示 `ui_plate` 的用途 ✓（空屏留给后期叠字 ✓）",
          any("type=" in item for item in unknown_type.problems)
          and any("ui_plate" in item for item in unknown_type.problems), unknown_type.problems)

    unknown_status = validate_manifest(manifest([asset("a", status="看起来还行")]))
    check("⑤ 状态不认识 ⇒ 报问题 ✓（不接受「看起来还行」这种 ✓）",
          any("status=" in item for item in unknown_status.problems), unknown_status.problems)

    no_id = validate_manifest(manifest([{"type": "prop", "status": "approved"}]))
    check("⑥ 缺 asset_id / 缺必填字段 ⇒ 报问题 ✓",
          any("asset_id" in item for item in no_id.problems), no_id.problems)

    missing_field = validate_manifest(manifest([{"asset_id": "a", "type": "prop"}]))
    check("⑦ 缺 status ⇒ 报问题 ✓（不许默认成通过 ✗）",
          any("status" in item for item in missing_field.problems), missing_field.problems)


# ══════════════════════════════════════════════════════════════════════════
# ② 依赖：悬空 / 自依赖 / **成环**
# ══════════════════════════════════════════════════════════════════════════
def case_dependencies() -> None:
    dangling = validate_manifest(manifest([asset("a", deps=["不存在"])]))
    check("⑧ 依赖指向**不存在**的资产 ⇒ 报问题（悬空引用是最常见的清单事故 ✓）",
          any("不存在" in item for item in dangling.problems), dangling.problems)

    self_dep = validate_manifest(manifest([asset("a", deps=["a"])]))
    check("⑨ 依赖自己 ⇒ 报问题 ✓", any("自己" in item for item in self_dep.problems),
          self_dep.problems)

    chain = validate_manifest(manifest([
        asset("front", status="to-generate"),
        asset("three_view", status="to-generate", deps=["front"]),
        asset("closeup", status="to-generate", deps=["three_view"]),
    ]))
    check("⑩ ⭐ **生成顺序 = 拓扑序**（先有正面图才谈得上基于它做三视图 ✓）",
          chain.order == ["front", "three_view", "closeup"], chain.order)

    cycle = validate_manifest(manifest([
        asset("a", status="to-generate", deps=["b"]),
        asset("b", status="to-generate", deps=["a"]),
    ]))
    check("⑪ ⭐⭐ **成环 ⇒ 点名环上的资产** ✓（只说「有环」没法修 ✗）",
          cycle.cycles == [["a", "b"]] and cycle.canGenerateVideo is False,
          (cycle.cycles, cycle.to_dict()["gateReasons"]))
    check("⑫ 环与**正常链**混在一起：只点名环上的 ✓ 不误伤链上的 ✓",
          validate_manifest(manifest([
              asset("ok1", status="to-generate"),
              asset("ok2", status="to-generate", deps=["ok1"]),
              asset("x", status="to-generate", deps=["y"]),
              asset("y", status="to-generate", deps=["x"]),
          ])).cycles == [["x", "y"]], "")

    approved_skipped = validate_manifest(manifest([
        asset("done", status="approved"),
        asset("todo", status="to-generate", deps=["done"]),
    ]))
    check("⑬ 已批准的资产**不进**生成顺序 ✓（没必要重做 ✓）",
          approved_skipped.order == ["todo"], approved_skipped.order)

    three_cycle = validate_manifest(manifest([
        asset("a", status="to-generate", deps=["c"]),
        asset("b", status="to-generate", deps=["a"]),
        asset("c", status="to-generate", deps=["b"]),
    ]))
    check("⑭ 三元环也能抓到 ✓（`a→c→b→a` 任意起点都成 ✓）",
          three_cycle.cycles and sorted(three_cycle.cycles[0]) == ["a", "b", "c"],
          three_cycle.cycles)

    order_a, cycles_a = generation_order({"p": {"dependencies": []}, "q": {"dependencies": []}})
    order_b, _ = generation_order({"q": {"dependencies": []}, "p": {"dependencies": []}})
    check("⑮ 同层资产按 id **稳定排序** ✓（同一份清单每次跑顺序一致 ✓ 便于对着操作 ✓）",
          order_a == order_b == ["p", "q"], (order_a, order_b))


# ══════════════════════════════════════════════════════════════════════════
# ③ ⭐ 验收门（本套核心）
# ══════════════════════════════════════════════════════════════════════════
def case_gate() -> None:
    pending = validate_manifest(manifest([asset("char_a", status="to-generate",
                                                needed_by="kf001")]))
    check("⑯ ⭐ 资产还是 `to-generate` ⇒ **门锁着** ✓ 且说清是哪一镜被谁卡住 ✓",
          pending.canGenerateVideo is False and pending.blockedShots == {"kf001": ["char_a"]}
          and any("kf001" in item for item in pending.gateReasons),
          (pending.blockedShots, pending.gateReasons))

    generated = validate_manifest(manifest([asset("char_a", status="generated",
                                                   needed_by="kf001")]))
    check("⑰ ⭐⭐ **`generated` 不等于验收通过** ⇒ 门**仍然锁着** ✗✓"
          "（若这条宽松了，门就形同虚设 ✓）",
          generated.canGenerateVideo is False
          and generated.blockedShots == {"kf001": ["char_a"]}, generated.counts)

    rejected = validate_manifest(manifest([asset("char_a", status="needs_regeneration",
                                                  needed_by="kf001")]))
    check("⑱ `needs_regeneration` / `rejected` / `missing` ⇒ 一律锁门 ✓",
          rejected.canGenerateVideo is False
          and validate_manifest(manifest([asset("a", status="rejected", needed_by="k1")])
                                ).canGenerateVideo is False
          and validate_manifest(manifest([asset("a", status="missing", needed_by="k1")])
                                ).canGenerateVideo is False, "")

    underscore = validate_manifest(manifest([asset("char_a", status="to_generate",
                                                   needed_by="kf001")]))
    check("⑲ 状态写法 `to_generate`（下划线 ✓）也认 ⇒ 不会因为写法差异**漏过门** ✗✓",
          underscore.canGenerateVideo is False
          and underscore.assets["char_a"]["status"] == "to-generate", underscore.assets)

    approved = validate_manifest(manifest([asset("char_a", status="approved",
                                                 needed_by="kf001")]))
    check("⑳ 人验收过 ⇒ 门开 ✓（机器不自己提状态 ✓ —— 只能由人给 approved ✓）",
          approved.canGenerateVideo is True and approved.blockedShots == {}, "")

    no_shot = validate_manifest(manifest([asset("char_a", status="to-generate")]))
    check("㉑ 未验收但**没有被任何镜头需要** ⇒ 不算阻断 ✓（还没排到它 ✓）",
          no_shot.canGenerateVideo is True and no_shot.blockedShots == {},
          (no_shot.canGenerateVideo, no_shot.blockedShots))

    mixed = validate_manifest(manifest([
        asset("char_a", status="approved", needed_by="kf001"),
        asset("prop_knife", kind="prop", status="to-generate", needed_by="kf003"),
        asset("clue_7", kind="clue", status="approved", needed_by="kf003"),
    ]))
    check("㉒ 逐镜判定 ✓：kf001 放行、kf003 被 `prop_knife` 卡住（`clue_7` 已过 ⇒ 不列 ✓）",
          mixed.blockedShots == {"kf003": ["prop_knife"]}, mixed.blockedShots)

    per_shot = gate_for_shots(manifest([
        asset("char_a", status="to-generate", needed_by="kf001"),
        asset("scene_b", kind="scene", status="approved", needed_by="kf002"),
    ]), ["kf001", "kf002", "kf999"])
    check("㉓ ⭐ `gate/check` 式门禁：**执行器调用前**逐镜判 ✓（未列到的镜头不受影响 ✓）",
          per_shot == {"kf001": ["char_a"]}, per_shot)

    counts = validate_manifest(manifest([
        asset("a", status="approved"), asset("b", status="to-generate"),
        asset("c", status="needs_regeneration"),
    ])).counts
    check("㉔ ⭐ 进度计数 ✓（键是**归一化后**的连字符形态 ✓ —— 下划线写法在这里曾漏过一道门 ✗）",
          counts == {"approved": 1, "to-generate": 1, "needs-regeneration": 1,
                     "total": 3, "blocking": 2}, counts)

    # ⭐ 回归：这个 bug 的**真实后果**不是计数 ✗，而是「待重做的资产**没进生成顺序**」✗✓
    regen_order = validate_manifest(manifest([
        asset("front", status="approved"),
        asset("face", status="needs_regeneration", deps=["front"]),
        asset("fresh", status="to-generate", deps=["face"]),
    ]))
    check("㉕′ ⭐⭐ **待重做的资产必须进生成顺序**（顺序就是「接下来该重做什么」✓）"
          "—— 下划线/连字符不一致时它会**整条消失** ✗✓",
          regen_order.order == ["face", "fresh"], regen_order.order)

    # 提示类（不阻断 ✓ 但影响"能不能重做"✓）
    warned = validate_manifest(manifest([
        asset("a", status="to-generate", needed_by="k1", prompt=False,
              criteria=False, path=False)]))
    check("㉖ 未验收却**没有 prompt / 验收标准 / 落点** ⇒ 只进 warnings ✓（不阻断 ✓ 但都该报 ✓）",
          warned.canGenerateVideo is False and len(warned.warnings) == 3, warned.warnings)


# ══════════════════════════════════════════════════════════════════════════
# ④ 路由
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    payload = manifest([asset("char_a", status="to-generate", needed_by="kf001"),
                        asset("scene_b", kind="scene", status="approved", needed_by="kf001")])
    response = client.post("/api/v1/assets/manifest/validate", json={"manifest": payload})
    data = response.json().get("data") or {}
    check("㉖ POST /assets/manifest/validate 真能调用 ✓（200 + 门锁着 + 逐条理由 ✓）",
          response.status_code == 200 and data.get("canGenerateVideo") is False
          and data.get("blockedShots") and data.get("gateReasons"),
          (response.status_code, data.get("gateReasons")))

    cycle_payload = manifest([asset("x", status="to-generate", deps=["y"]),
                              asset("y", status="to-generate", deps=["x"])])
    cycle_data = client.post("/api/v1/assets/manifest/validate",
                             json={"manifest": cycle_payload}).json()["data"]
    check("㉗ 成环经 API 也点名 ✓（前端可直接把环画出来 ✓）",
          cycle_data.get("cycles") == [["x", "y"]], cycle_data.get("cycles"))

    gate = client.post("/api/v1/assets/gate/check",
                       json={"manifest": payload, "shots": ["kf001", "kf002"]})
    gate_data = gate.json().get("data") or {}
    check("㉘ POST /assets/gate/check 真能调用 ✓（`allowed=False` + 逐镜明细 ✓）",
          gate.status_code == 200 and gate_data.get("allowed") is False
          and gate_data.get("blocked") == {"kf001": ["char_a"]}, gate_data)

    check("㉙ 缺 shots ⇒ 400 ✓（不默认「全部放行」✗）",
          client.post("/api/v1/assets/gate/check",
                      json={"manifest": payload}).status_code == 400)

    check("㉚ 传坏 JSON（字符串清单）⇒ 200 + problems ✓（不 500 ✗）",
          client.post("/api/v1/assets/manifest/validate",
                      json={"manifest": "坏的"}).json()["data"]["problems"] != [])


def main() -> int:
    case_validate()
    case_dependencies()
    case_gate()
    case_api()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
