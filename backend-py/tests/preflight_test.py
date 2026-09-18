"""S7 自检：**开跑前体检**（把四块能力串成一次调用 ✓ 零依赖 ✓ 2026-09-18）。

本套钉三件事 ✓：

1. **判据只有一份** ✓ —— 体检报告里的 ``sections`` 必须与**直接调用**那四个模块的结果
   逐项一致 ✓（否则就是"又抄了一遍判据"✗ ⇒ 以后改一处、漏一处 ✓）；
2. **该断的断、该警的警** ✗—— 质感层问题（约束不够硬 ✓）与 §10「可删或可并」✓
   **只预警不阻断** ✓✓（它们是"画得差点/可以删"，不是"接不上"✓）；
3. ⭐ ``nextActions`` 必须**按成本排序** ✓：先补映射/改分镜（零成本 ✓）、
   再验收资产（要人工 ✓）、最后才生成（**要花钱** ✓）。

运行::

    ./.venv/Scripts/python.exe tests/preflight_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.continuity import check_continuity  # noqa: E402
from app.services.production_preflight import run_preflight  # noqa: E402
from app.services.prompt_polish import PolishInputs, build_prompt_layer  # noqa: E402
from app.services.shot_placeholders import PlaceholderMaps, resolve_placeholders  # noqa: E402
from app.services.asset_manifest import validate_manifest  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


CLEAN_PLAN: dict = {
    "locations": [{"id": "L1", "name": "案发房间", "left": "门", "right": "破窗",
                   "back": "白布尸体", "foreground": "湿水泥地", "axis": "左入右出"}],
    "characters": [{"id": "R5", "name": "沈砚", "appearance": "黑色湿短发", "current_state": "刚醒"}],
    "props": [{"id": "P1", "name": "刀", "first_shot": "kf001", "first_state": "floor"}],
    "clues": [{"id": "C1", "name": "数字7", "first_visible_shot": "kf001"}],
    "shots": [
        {"shot_id": "kf001", "location": "L1", "characters": ["R5"],
         "prop_states": {"P1": "floor"},
         "actions": [{"actor": "R5", "action": "睁眼", "effects": ["information"]}],
         "clues_visible": ["C1"]},
    ],
}
CLEAN_MANIFEST: dict = {"episode": 1, "assets": [
    {"asset_id": "char_a", "type": "character", "status": "approved", "name": "沈砚",
     "first_needed_by": "kf001", "dependencies": [],
     "target_path": "video/ep001/char_a.png", "prompt": "正面参考图",
     "acceptance_criteria": ["同一发型", "脸部清晰"]}]}
CLEAN_POLISH: dict = {
    "themeTags": ["真人写实", "犯罪悬疑", "冷色调"], "characterScene": "沈砚，黑色湿短发",
    "lens": "Sony Venice + Canon K-35", "palette": "低饱和灰蓝", "texture": "潮湿水泥反光",
    "shotType": "近景", "movement": "轻微跟焦", "screenDirection": "朝右看",
    "slice": "沈砚睁眼", "sounds": ["呼吸声"], "imperfections": ["血迹", "沾泥"],
    "handheld": True}


# ══════════════════════════════════════════════════════════════════════════
# ① 干净 ⇒ 可开跑
# ══════════════════════════════════════════════════════════════════════════
def case_clean() -> None:
    report = run_preflight({
        "promptText": "场景 <location>L1</location> 的 <role>R5</role>",
        "maps": {"locations": {"L1": "旧城区案发房间"}, "roles": {"R5": "沈砚"}},
        "polish": CLEAN_POLISH, "plan": CLEAN_PLAN, "manifest": CLEAN_MANIFEST})
    check("① ⭐ 四段全给且都干净 ⇒ `ready=True` ✓ 且四个环节都进了 `checked` ✓",
          report.ready is True and set(report.checked) == {"placeholders", "polish",
                                                           "continuity", "assetGate"},
          (report.ready, report.checked, report.blockers))
    check("② 可开跑时给出**诚实的行动项** ✓（明说这只代表事前检查过 ✓ 不代表质量 ✗）",
          any("可以开跑" in item and "事前" in item for item in report.nextActions),
          report.nextActions)
    check("③ 四个环节的**完整原报告**都在 `sections` 里 ✓（想深挖不必再调一次 ✓）",
          set(report.sections) == {"placeholders", "polish", "continuity", "assetGate"},
          list(report.sections))


# ══════════════════════════════════════════════════════════════════════════
# ② ⭐ 判据只有一份：体检 == 直接调那四个模块
# ══════════════════════════════════════════════════════════════════════════
def case_no_duplicate_criteria() -> None:
    maps = {"locations": {"L1": "旧城区案发房间"}, "roles": {"R5": "沈砚"}}
    text = "场景 <location>L1</location> 的 <role>R5</role><duration-ms>6000</duration-ms>"
    report = run_preflight({"promptText": text, "maps": maps, "plan": CLEAN_PLAN,
                            "manifest": CLEAN_MANIFEST, "polish": CLEAN_POLISH})

    direct_resolve = resolve_placeholders(text, PlaceholderMaps(
        locations=maps["locations"], roles=maps["roles"]))
    check("④ ⭐ 占位符那段与**直接调用** `resolve_placeholders` 结果**逐字段一致** ✓"
          "（体检不自己抄判据 ✓）",
          report.sections["placeholders"]["text"] == direct_resolve.text
          and report.sections["placeholders"]["used"] == direct_resolve.used
          and report.sections["placeholders"]["durationsMs"] == [6000],
          report.sections["placeholders"]["text"])

    direct_plan = check_continuity(CLEAN_PLAN)
    check("⑤ ⭐ 连续性那段与**直接调用** `check_continuity` 一致 ✓（含 propTimeline ✓）",
          report.sections["continuity"]["violations"] == direct_plan.violations
          and report.sections["continuity"]["ok"] == direct_plan.ok, "")

    direct_gate = validate_manifest(CLEAN_MANIFEST)
    check("⑥ ⭐ 验收门那段与**直接调用** `validate_manifest` 一致 ✓",
          report.sections["assetGate"]["canGenerateVideo"] == direct_gate.canGenerateVideo
          and report.sections["assetGate"]["counts"] == direct_gate.counts, "")

    direct_polish = build_prompt_layer(PolishInputs(**{
        "theme_tags": tuple(CLEAN_POLISH["themeTags"]),
        "character_scene": CLEAN_POLISH["characterScene"], "lens": CLEAN_POLISH["lens"],
        "palette": CLEAN_POLISH["palette"], "texture": CLEAN_POLISH["texture"],
        "shot_type": CLEAN_POLISH["shotType"], "movement": CLEAN_POLISH["movement"],
        "screen_direction": CLEAN_POLISH["screenDirection"],
        "slice_text": CLEAN_POLISH["slice"], "sounds": tuple(CLEAN_POLISH["sounds"]),
        "imperfections": tuple(CLEAN_POLISH["imperfections"]), "handheld": True}))
    check("⑦ ⭐ 质感层那段与**直接调用** `build_prompt_layer` 一致 ✓",
          report.sections["polish"]["text"] == direct_polish.text
          and report.sections["polish"]["issues"] == direct_polish.issues, "")


# ══════════════════════════════════════════════════════════════════════════
# ③ 该断的断、该警的警
# ══════════════════════════════════════════════════════════════════════════
def case_severity() -> None:
    unresolved = run_preflight({"promptText": "<location>L9</location> 的 <role>R5</role>",
                                "maps": {"roles": {"R5": "沈砚"}}})
    check("⑧ 未映射编号 ⇒ **阻断** ✓（裸编号进提示词 = 模型不认识 ✓）",
          unresolved.ready is False
          and any(item["stage"] == "placeholders" for item in unresolved.blockers),
          unresolved.blockers)

    tagged = run_preflight({"promptText": "<location>L1</role> 写错闭合",
                            "maps": {"locations": {"L1": "案发房间"}}})
    check("⑨ 畸形标签 ⇒ 阻断 ✓（宽松检测抓得到 ✓）",
          tagged.ready is False
          and any(item["stage"] == "placeholders" for item in tagged.blockers), tagged.blockers)

    thin_polish = run_preflight({"polish": {"themeTags": ["写实"], "slice": "睁眼"}})
    check("⑩ ⭐ 质感层的问题**只预警不阻断** ✓✓（⚠️ 约束不够硬 ⇒ 画得差点 ✓ 不是接不上 ✗）",
          thin_polish.ready is True and thin_polish.blockers == []
          and len(thin_polish.warnings) >= 2, (thin_polish.ready, thin_polish.warnings))

    jump_plan = {**CLEAN_PLAN, "shots": [dict(CLEAN_PLAN["shots"][0]), {
        "shot_id": "kf002", "location": "L1", "characters": ["R5"],
        "prop_states": {"P1": "hand"},
        "actions": [{"actor": "R5", "action": "站起", "effects": ["plot"]}]}]}
    broken_shot = run_preflight({"plan": jump_plan})
    check("⑪ 道具状态**没有交代就变了** ⇒ 阻断 ✓（§8 的核心 ✓）",
          broken_shot.ready is False
          and any(item.get("code") == "prop-timeline" for item in broken_shot.blockers),
          broken_shot.blockers)

    inert_plan = {**CLEAN_PLAN, "shots": [{**CLEAN_PLAN["shots"][0],
                  "actions": [{"actor": "R5", "action": "看了一眼", "effects": []}]}]}
    inert = run_preflight({"plan": inert_plan})
    check("⑫ ⭐ §10「可删或可并」**只预警不阻断** ✓（删戏是创作决定 ✓ 由人拍板 ✓）",
          inert.ready is True
          and any("可删或可并" in str(item.get("message")) for item in inert.warnings),
          inert.warnings)


# ══════════════════════════════════════════════════════════════════════════
# ④ 验收门 + ⭐ nextActions **按成本排序**
# ══════════════════════════════════════════════════════════════════════════
def case_gate_and_actions() -> None:
    blocked_manifest = {"episode": 1, "assets": [
        {"asset_id": "char_a", "type": "character", "status": "to-generate", "name": "沈砚",
         "first_needed_by": "kf001", "dependencies": [], "target_path": "x.png",
         "prompt": "正面参考图", "acceptance_criteria": ["同一发型"]}]}
    gate_only = run_preflight({"manifest": blocked_manifest})
    check("⑬ 未验收资产卡住镜头 ⇒ 阻断 ✓ 且行动项里带**生成顺序** ✓",
          gate_only.ready is False
          and any("char_a" in item for item in gate_only.nextActions)
          and any(item["stage"] == "assetGate" for item in gate_only.blockers),
          gate_only.nextActions)

    cyclic = {"episode": 1, "assets": [
        {"asset_id": "x", "type": "character", "status": "to-generate", "name": "x",
         "dependencies": ["y"], "target_path": "x.png", "prompt": "p"},
        {"asset_id": "y", "type": "character", "status": "to-generate", "name": "y",
         "dependencies": ["x"], "target_path": "y.png", "prompt": "p"}]}
    cycle_report = run_preflight({"manifest": cyclic})
    check("⑭ 依赖成环 ⇒ 行动项里**明确要求先人工打断** ✓（顺序无解 ✓）",
          any("成环" in item and "人工" in item for item in cycle_report.nextActions),
          cycle_report.nextActions)

    everything = run_preflight({
        "promptText": "<location>L9</location>",
        "continuityHint": "", "plan": {**CLEAN_PLAN, "shots": [dict(CLEAN_PLAN["shots"][0]), {
            "shot_id": "kf002", "location": "L1", "characters": ["R5"],
            "prop_states": {"P1": "hand"},
            "actions": [{"actor": "R5", "action": "站起", "effects": ["plot"]}]}]},
        "manifest": blocked_manifest})
    joined = " ".join(everything.nextActions)
    check("⑮ ⭐⭐ `nextActions` **按成本排序**：补映射（①）→ 修连续性（②）→ 处理资产（③）✓✓"
          "（先零成本、后花钱 ✓）",
          everything.ready is False
          and joined.index("补占位符映射") < joined.index("修连续性") < joined.index("处理资产"),
          everything.nextActions[:3])
    check("⑯ 三个 stage 的阻断项都逐条带 `stage` 标 ✓（前端可分栏 ✓）",
          {item["stage"] for item in everything.blockers}
          >= {"placeholders", "continuity", "assetGate"},
          [item["stage"] for item in everything.blockers])


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 缺省与坏输入：**如实说没检查**（而不是假装干净 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_missing() -> None:
    empty = run_preflight({})
    check("⑰ ⭐ 什么都没给 ⇒ **明说「结果不代表任何」** ✓ 而不是 `ready=True` 让人误以为干净 ✗",
          empty.checked == []
          and any("不代表任何" in str(item.get("message")) for item in empty.warnings),
          empty.warnings)

    partial = run_preflight({"plan": CLEAN_PLAN})
    check("⑱ 只给一段 ⇒ `checked` **如实只列那一段** ✓（不冒充全查过 ✓）",
          partial.checked == ["continuity"] and partial.ready is True, partial.checked)

    check("⑲ 非对象输入 ⇒ 阻断且不崩 ✓",
          run_preflight("坏的").ready is False
          and run_preflight("坏的").blockers, "")

    check("⑳ `sections` 只含**真跑过**的环节 ✓（没跑的不放空壳 ✓）",
          set(partial.sections) == {"continuity"}, list(partial.sections))


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 路由
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post("/api/v1/production/preflight", json={
        "plan": CLEAN_PLAN, "manifest": CLEAN_MANIFEST, "polish": CLEAN_POLISH})
    data = response.json().get("data") or {}
    check("㉑ POST /production/preflight 真能调用 ✓（200 + ready + 行动项 ✓）",
          response.status_code == 200 and data.get("ready") is True
          and data.get("nextActions") and data.get("order") == [
              "placeholders", "polish", "continuity", "assetGate"],
          (response.status_code, data.get("ready"), data.get("nextActions")))

    blocked = client.post("/api/v1/production/preflight", json={
        "plan": CLEAN_PLAN,
        "manifest": {"episode": 1, "assets": [
            {"asset_id": "a", "type": "character", "status": "missing", "name": "a",
             "first_needed_by": "kf001", "dependencies": [], "target_path": "a.png",
             "prompt": "p"}]}}).json()["data"]
    check("㉒ 有卡点 ⇒ `ready=False` 且 `blockers` 带 stage ✓",
          blocked.get("ready") is False
          and any(item.get("stage") == "assetGate" for item in blocked.get("blockers") or []),
          blocked.get("blockers"))

    schema = client.get("/api/v1/production/preflight/schema").json().get("data") or {}
    check("㉓ `GET /preflight/schema` 报出**接受哪些段 + 顺序 + 注意事项** ✓"
          "（前端照它拼请求 ✓）",
          schema.get("order") and set(schema.get("sections") or {}) >= {
              "promptText", "plan", "manifest"}, list(schema))

    # ⚠️ 注意本项目的约定 ✓：``read_json`` 会把非法 JSON / 非对象**归一成 ``{}``** ✓
    #    （有意为之 ✓ 对齐 TS ✓）⇒ 路由里的 ``isinstance(body, dict)`` 是**死代码** ✗，
    #    判据只能写在**业务层** ✓ ⇒ 这里测的是「一段都没给」⇒ 400 ✓。
    check("㉔ ⭐ 一段都没给 ⇒ **400** ✓（不假装体检过 ✗；也不把归一后的空 dict 当成功 ✓）",
          client.post("/api/v1/production/preflight", json={}).status_code == 400
          and client.post("/api/v1/production/preflight", json="坏的").status_code == 400,
          (client.post("/api/v1/production/preflight", json={}).status_code,
           client.post("/api/v1/production/preflight", json="坏的").status_code))

    check("㉕ 给了任意一段 ⇒ 200 ✓（判据是「有没有可体检的东西」✓ 不是「body 是不是对象」✗）",
          client.post("/api/v1/production/preflight",
                      json={"plan": CLEAN_PLAN}).status_code == 200, "")


def main() -> int:
    case_clean()
    case_no_duplicate_criteria()
    case_severity()
    case_gate_and_actions()
    case_missing()
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
