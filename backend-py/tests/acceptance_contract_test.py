"""验收报告合同自检 ✓（``app/services/acceptance_report.py`` ✓）。

它守的是一句话：**这一件产物凭什么算过？** 三处最容易写成"假绿"的地方各有一条钉：

* **覆盖率门禁**：声明要 ``audio`` 却只交 ``text`` ⇒ 必须**改判 fail** ✓
  （只报一条 warning 就等于 ``requiredEvidence`` 是装饰 ✗）；
* **没有证据的 pass**：必须降级成 ``blocked`` ✓
  （否则会出现"有 problem 但 verdict=pass"这种自相矛盾的报告 ✗）；
* **计划里没执行的条目**：必须进 ``unexecuted`` 且 ``verdict`` 至多 ``partial`` ✓
  （"我没验"被数成"通过"是这类报告最典型的假绿 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/acceptance_contract_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.acceptance_report import (  # noqa: E402
    DEFAULT_VERIFIER,
    EVIDENCE_TYPES,
    STATUSES,
    SURFACES,
    VERDICTS,
    VERIFIERS,
    check_report,
    coerce_cases,
    coerce_plan,
    derive_verdict,
    evidence_type_of,
    plan_from_manifest,
    render_report,
)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def item(pid: str = "asset:char_a", *, verifier: str = "agent",
         required: tuple[str, ...] = ("screenshot",), **over: object) -> dict:
    base: dict = {
        "id": pid,
        "title": f"{pid} 通过",
        "category": "资产验收",
        "verifier": verifier,
        "method": "对照准则看图",
        "expected": "同一发型；同一服装",
        "requiredEvidence": list(required),
    }
    base.update(over)
    return base


def case(cid: str = "asset:char_a", *, status: str = "pass", evidence=None,
         surface: str = "cli", observation: str = "看了三张，发型与服装一致",
         **over: object) -> dict:
    payload: dict = {
        "id": cid,
        "name": f"{cid} 通过",
        "category": "资产验收",
        "surface": surface,
        "status": status,
        "observation": observation,
        "evidence": evidence if evidence is not None else [ev("out/char_a.png", "正面半身，发型与基准一致")],
    }
    payload.update(over)
    return payload


def ev(path: str, description: str = "看到了什么", **over: object) -> dict:
    entry: dict = {"path": path, "description": description}
    entry.update(over)
    return entry


def report(*, plan=None, cases=None, summary=None) -> dict:
    payload: dict = {"plan": plan if plan is not None else [item()],
                     "cases": cases if cases is not None else [case()]}
    if summary is not None:
        payload["summary"] = summary
    return payload


# ══════════════════════════════════════════════════════════════════════════
# ① 闭集与默认值
# ══════════════════════════════════════════════════════════════════════════
def case_vocab() -> None:
    check("① verifier 闭集是 program／agent／llm ✓", VERIFIERS == ("program", "agent", "llm"), VERIFIERS)
    check("① 默认 verifier 是 agent ✓（参考原文默认值 ✓）", DEFAULT_VERIFIER == "agent", DEFAULT_VERIFIER)
    check("① 证据介质闭集 8 值 ✓（一个不多一个不少 ✓）",
          EVIDENCE_TYPES == ("screenshot", "gif", "video", "audio", "text", "markdown",
                             "dom_snapshot", "transcript"), EVIDENCE_TYPES)
    check("① 反验：闭集里**没有** image／img ✗（参考用 screenshot ✓ 拼错要当场红 ✓）",
          "image" not in EVIDENCE_TYPES and "img" not in EVIDENCE_TYPES, None)
    check("① case 状态三值 ✓（blocked 是合法出口 ✓ 不是失败 ✓）",
          STATUSES == ("pass", "fail", "blocked"), STATUSES)
    check("① verdict 三值 ✓（没验到落 partial ✓ 不落 pass ✓）",
          VERDICTS == ("pass", "fail", "partial"), VERDICTS)
    check("① surface 闭集含 cli／desktop／web ✓",
          {"cli", "desktop", "web"} <= set(SURFACES), SURFACES)


# ══════════════════════════════════════════════════════════════════════════
# ② 介质识别（显式优先 / 扩展名 / 认不出不猜）
# ══════════════════════════════════════════════════════════════════════════
def case_evidence_type() -> None:
    check("② 显式 type 优先于扩展名 ✓（.png 写 audio ⇒ audio ✓）",
          evidence_type_of({"path": "a.png", "type": "audio"}) == "audio", None)
    pairs = [
        ("a.png", "screenshot"), ("a.JPG", "screenshot"), ("a.gif", "gif"),
        ("a.mp4", "video"), ("a.mp3", "audio"), ("a.wav", "audio"),
        ("a.md", "markdown"), ("a.jsonl", "transcript"), ("a.json", "text"),
        ("a.log", "text"), ("a.html", "dom_snapshot"), ("a.txt", "text"),
    ]
    bad = [(path, want, evidence_type_of({"path": path})) for path, want in pairs
           if evidence_type_of({"path": path}) != want]
    check(f"② 扩展名 ⇒ 介质 12 组全对 ✓（.jsonl 是 transcript ✓ .json 是 text ✓）", not bad, bad)
    check("② Windows 反斜杠路径也认 ✓（证据常从 Windows 路径贴过来 ✓）",
          evidence_type_of({"path": r"out\ep001\char_a.PNG"}) == "screenshot", None)
    check("② 认不出的后缀返 None ✓（.bin **不许**猜成 text ✗ 猜了覆盖门禁就假绿 ✓）",
          evidence_type_of({"path": "a.bin"}) is None, evidence_type_of({"path": "a.bin"}))
    check("② 没有后缀返 None ✓（不猜 ✓）",
          evidence_type_of({"path": "evidence"}) is None, None)
    check("② 显式写了闭集外的值 ⇒ 原样返回 ✓（让上层报「不在闭集」✗ 不是「认不出」✓）",
          evidence_type_of({"path": "a.png", "type": "picture"}) == "picture", None)
    check("② 字符串条目等价于只给 path ✓（容忍简写 ✓）",
          evidence_type_of("a.mp4") == "video", None)


# ══════════════════════════════════════════════════════════════════════════
# ③ plan 校验（先声明的那一半）
# ══════════════════════════════════════════════════════════════════════════
def case_plan() -> None:
    ok = check_report(report())
    check("③ 一份最小合规报告 ⇒ ok ✓ 且无 problem ✓", ok.ok, ok.problems[:4])

    for field, label in (("title", "title"), ("expected", "expected"),
                         ("method", "method"), ("category", "category")):
        bad_item = item(**{field: ""})
        got = check_report(report(plan=[bad_item]))
        check(f"③ plan 缺 {label} ⇒ 报 problem ✓（先声明只做一半 ✓）",
              any(label in p for p in got.problems), got.problems[:3])

    got = check_report(report(plan=[item(verifier="robot")]))
    check("③ verifier 越界 ⇒ 报 problem 并列出闭集 ✓",
          any("verifier" in p and "program" in p for p in got.problems), got.problems[:3])

    got = check_report(report(plan=[item(required=())]))
    check("③ requiredEvidence 为空 ⇒ 报 problem ✓（参考要求 one or more ✓）",
          any("requiredEvidence" in p for p in got.problems), got.problems[:3])

    got = check_report(report(plan=[item(required=("image",))]))
    check("③ requiredEvidence 写了闭集外的介质 ⇒ 报 problem ✓",
          any("image" in p and "闭集" in p for p in got.problems), got.problems[:3])

    got = check_report(report(plan=[item("asset:a"), item("asset:a", title="另一条")]))
    check("③ plan id 重复 ⇒ 报 problem ✓（两条检查共用一个 id ⇒ 证据归属不明 ✗）",
          any("重复" in p for p in got.problems), got.problems[:3])

    got = check_report(report(plan=[{"id": "", "title": "没 id"}]))
    check("③ plan 缺 id ⇒ 报 problem ✓（缺的条目**留壳**点名 ✗ 不静默丢 ✓）",
          any("缺少 id" in p for p in got.problems), got.problems[:3])

    got = check_report({"plan": [], "cases": []})
    check("③ 没有 plan ⇒ 只 warning（事后补的验收）✓ verdict 仍 partial ✓",
          got.ok and any("事后补的" in w for w in got.warnings) and got.verdict == "partial",
          (got.warnings, got.verdict))


# ══════════════════════════════════════════════════════════════════════════
# ④ 证据（介质 / 说明 / 没有证据不许下结论）
# ══════════════════════════════════════════════════════════════════════════
def case_evidence() -> None:
    for desc, label in (("", "空"), ("JSON", "只写 JSON"), ("char_a.png", "只写文件名"),
                        ("证据", "只写「证据」"), ("日志", "只写「日志」")):
        got = check_report(report(cases=[case(evidence=[ev("out/char_a.png", desc)])]))
        check(f"④ description 是{label} ⇒ 报 problem ✓（拿文件名兜底不够 ✓）",
              any("description" in p for p in got.problems), got.problems[:3])

    got = check_report(report(cases=[case(evidence=[ev("out/char_a.png",
                                                       "瞳孔边缘清晰，左眉有一道短疤")])]))
    check("④ 正常 description ⇒ 不报 ✓", not any("description" in p for p in got.problems),
          got.problems[:3])

    got = check_report(report(cases=[case(evidence=[ev("", "看过了")])]))
    check("④ 证据没有 path ⇒ 报 problem ✓（点不开的证据不算证据 ✓）",
          any("path" in p for p in got.problems), got.problems[:3])

    got = check_report(report(cases=[case(evidence=[ev("out/a.bin", "看了二进制")])]))
    check("④ 介质认不出 ⇒ 报 problem ✓（不许静默当成 text ✗）",
          any("认不出介质" in p for p in got.problems), got.problems[:3])

    got = check_report(report(cases=[case(evidence=[ev("out/a.png", "看了", type="picture")])]))
    check("④ 显式介质越界 ⇒ 报「不在闭集」✓", any("不在闭集" in p for p in got.problems),
          got.problems[:3])

    got = check_report(report(cases=[case(status="pass", evidence=[])]))
    check("④ pass 却零证据 ⇒ 报 problem ✓", any("一条证据都没有" in p for p in got.problems),
          got.problems[:3])
    check("④ pass 却零证据 ⇒ **降级成 blocked** ✓（不许留 pass ✗ 否则 problem 与 verdict 打架 ✓）",
          got.cases[0]["status"] == "blocked" and got.cases[0].get("enforcedStatus") is True,
          got.cases[0])
    check("④ 降级后 verdict 是 partial ✓（不是 pass ✓）", got.verdict == "partial", got.verdict)

    got = check_report(report(cases=[case(status="fail", evidence=[])]))
    check("④ fail 却零证据 ⇒ 报 problem 但**保留 fail** ✓（保守方向正确 ✓）",
          any("一条证据都没有" in p for p in got.problems) and got.cases[0]["status"] == "fail",
          got.cases[0])

    got = check_report(report(cases=[case(status="pending")]))
    check("④ status 写「待定」⇒ 报 problem ✓ 且落 blocked ✓（绝不许落 pass ✓）",
          any("status" in p for p in got.problems) and got.cases[0]["status"] == "blocked",
          got.cases[0])

    got = check_report(report(cases=[case(status="pass", observation="")]))
    check("④ 判了 pass 却不写 observation ⇒ 报 problem ✓",
          any("observation" in p for p in got.problems), got.problems[:3])

    got = check_report(report(cases=[case(surface="kiosk")]))
    check("④ surface 越界 ⇒ 报 problem ✓", any("surface" in p for p in got.problems),
          got.problems[:3])


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 覆盖率门禁（整套东西的牙齿）
# ══════════════════════════════════════════════════════════════════════════
def case_coverage_gate() -> None:
    plan = [item(required=("audio",))]
    cases = [case(status="pass", evidence=[ev("out/transcript.txt", "逐字听写如下……")])]
    got = check_report(report(plan=plan, cases=cases))
    check("⑤ 声明要 audio 只交 text ⇒ 报 problem ✓（覆盖率门禁 ✓）",
          any("覆盖率门禁" in p for p in got.problems), got.problems[:3])
    check("⑤ 声明要 audio 只交 text ⇒ **改判 fail** ✓（不是 warning ✗）",
          got.cases[0]["status"] == "fail" and got.cases[0].get("enforcedStatus") is True,
          got.cases[0])
    check("⑤ 改判后 verdict=fail ✓（有失败不被「没验全」盖住 ✓）", got.verdict == "fail", got.verdict)

    cases = [case(status="pass", evidence=[ev("out/vo.mp3", "第三遍气息渐弱"),
                                           ev("out/char_a.png", "同一发型")])]
    got = check_report(report(plan=plan, cases=cases))
    check("⑤ 交齐声明要的介质 ⇒ 不改判 ✓ verdict=pass ✓",
          got.cases[0]["status"] == "pass" and got.verdict == "pass"
          and not any("覆盖率门禁" in p for p in got.problems), (got.cases[0], got.verdict))

    plan = [item(required=("audio",))]
    cases = [case(status="blocked", evidence=[ev("out/x.txt", "TTS 服务没起来，音频没生成")])]
    got = check_report(report(plan=plan, cases=cases))
    check("⑤ 拿不到音频就标 blocked ⇒ **只 warning 不改判** ✓（参考原话：别拿别的介质顶替 ✓）",
          got.cases[0]["status"] == "blocked"
          and any("blocked 是正确出口" in w for w in got.warnings), (got.cases[0], got.warnings[:3]))
    check("⑤ blocked 的 case 让 verdict 落 partial ✓（不是 pass ✓）", got.verdict == "partial",
          got.verdict)

    cases = [case(status="pass", evidence=[ev("out/a.png", "同一发型"),
                                           ev("out/b.txt", "对照准则第 2 条命中")])]
    got = check_report(report(plan=[item(required=("screenshot", "text"))], cases=cases))
    check("⑤ 要求两种介质、交了两种 ⇒ 不改判 ✓",
          got.cases[0]["status"] == "pass" and got.verdict == "pass", got.cases[0])


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 未执行 / verdict 现算
# ══════════════════════════════════════════════════════════════════════════
def case_verdict() -> None:
    plan = [item("asset:a"), item("asset:b")]
    got = check_report(report(plan=plan, cases=[case("asset:a")]))
    check("⑥ 计划 2 项只执行 1 项 ⇒ unexecuted 点名 ✓", got.summary["unexecuted"] == ["asset:b"],
          got.summary)
    check("⑥ 有计划项没执行 ⇒ verdict=partial ✓（不是 pass ✓）", got.verdict == "partial", got.verdict)
    check("⑥ 未执行不算 fail ✓（「没验到」≠「没过」✓）", got.summary["failed"] == 0, got.summary)

    got = check_report({"plan": [], "cases": []})
    check("⑥ 一条 case 都没有 ⇒ verdict=partial ✓（什么都没验不是通过 ✓）",
          derive_verdict([], []) == "partial" and got.verdict == "partial", got.verdict)

    got = check_report(report(plan=[item("asset:a"), item("asset:b")],
                              cases=[case("asset:a"), case("asset:b", evidence=[ev("out/b.png", "背面视角齐全")])]))
    check("⑥ 计划全执行 + 全通过 ⇒ verdict=pass ✓", got.verdict == "pass" and got.ok, got.problems[:3])

    got = check_report(report(summary={"verdict": "pass"}))
    check("⑥ 手写 verdict 与现算一致 ⇒ 不报 problem ✓", got.ok, got.problems[:3])

    got = check_report(report(cases=[case(status="fail", evidence=[ev("out/a.png", "缺左眉")])],
                              summary={"verdict": "pass"}))
    check("⑥ 手写 pass 但现算 fail ⇒ 报 problem ✓ 并采用现算 ✓（绿灯写不死 ✓）",
          any("现算" in p for p in got.problems) and got.verdict == "fail",
          (got.problems[:3], got.verdict))

    got = check_report(report(summary={"verdict": "awesome"}))
    check("⑥ 手写 verdict 越界 ⇒ 报 problem ✓",
          any("不在" in p and "awesome" in p for p in got.problems), got.problems[:3])

    check("⑥ derive_verdict 反验：有 fail + 有未执行 ⇒ fail（失败优先 ✓）",
          derive_verdict([{"status": "fail"}], ["x"]) == "fail", None)


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 清单 ⇒ 计划（本仓独有的桥）
# ══════════════════════════════════════════════════════════════════════════
def case_plan_from_manifest() -> None:
    manifest = {
        "assets": [
            {"asset_id": "char_a", "type": "character", "status": "to-generate",
             "target_path": "video/ep001/assets/char_a.png",
             "acceptance_criteria": ["同一发型", "同一服装", "脸部清晰"]},
            {"asset_id": "vo_01", "type": "audio", "status": "to-generate",
             "target_path": "audio/ep001/vo_01.mp3",
             "acceptance_criteria": ["无爆音"]},
            {"asset_id": "char_b", "type": "character", "status": "to-generate",
             "target_path": "video/ep001/assets/char_b.png"},
        ]
    }
    plan, notes = plan_from_manifest(manifest)
    check("⑦ 有准则的资产各成一条计划项 ✓（3 条准则不进 2 条 ✓ 逐条对照 ✓）",
          [entry["id"] for entry in plan] == ["asset:char_a", "asset:vo_01"], plan)
    check("⑦ expected 把三条准则逐条拼起来 ✓（不合并成「看起来没问题」✗）",
          plan[0]["expected"] == "同一发型；同一服装；脸部清晰", plan[0]["expected"])
    check("⑦ 图片资产要 screenshot ✓", plan[0]["requiredEvidence"] == ["screenshot"], plan[0])
    check("⑦ 音频资产按 target_path 认成 audio ✓（不硬编码 screenshot ✓）",
          plan[1]["requiredEvidence"] == ["audio"], plan[1])
    check("⑦ verifier 落 agent ✓（看图对照机器判不了 ✓）", plan[0]["verifier"] == "agent", plan[0])
    check("⑦ 没有准则的资产 ⇒ notes 点名 ✓（不许少一条检查还装作计划完整 ✓）",
          any("char_b" in note and "没有验收计划" in note for note in notes), notes)

    plan, notes = plan_from_manifest({"assets": []})
    check("⑦ 空清单 ⇒ 空计划 + notes 明说 ✓", plan == [] and notes, (plan, notes))

    plan, notes = plan_from_manifest({"assets": [{"asset_id": "x", "target_path": "blob",
                                                  "acceptance_criteria": ["能用"]}]})
    check("⑦ target_path 认不出介质 ⇒ 默认 screenshot **并 note 明说是猜的** ✓（不静默 ✓）",
          plan[0]["requiredEvidence"] == ["screenshot"] and any("认不出介质" in n for n in notes),
          notes)

    got = check_report({"cases": [case("asset:char_a")]}, manifest=manifest)
    check("⑦ 报告没 plan 但有清单 ⇒ 用清单现生成 ✓ planSource=manifest ✓",
          got.planSource == "manifest" and len(got.plan) == 2, (got.planSource, len(got.plan)))
    check("⑦ 清单生成的计划里 char_b 缺准则 ⇒ 那条缺口进 warnings ✓",
          any("char_b" in w for w in got.warnings), got.warnings[:3])


# ══════════════════════════════════════════════════════════════════════════
# ⑧ 归一化与渲染
# ══════════════════════════════════════════════════════════════════════════
def case_shape() -> None:
    check("⑧ plan 是单值对象也收 ✓（容忍简写 ✓）",
          len(coerce_plan(item())) == 1, None)
    check("⑧ plan 里混进字符串 ⇒ 留空壳 ✓（不静默丢 ✓）",
          len(coerce_plan(["坏的"])) == 1 and coerce_plan(["坏的"])[0]["id"] == "", None)
    check("⑧ coerce_plan 默认 verifier=agent ✓",
          coerce_plan([{"id": "x"}])[0]["verifier"] == "agent", None)
    check("⑧ 证据允许只给 path 字符串 ✓",
          coerce_cases([{"id": "x", "evidence": ["a.png"]}])[0]["evidence"][0]["path"] == "a.png", None)
    check("⑧ case 的 title 可当 name 用 ✓（清单/报告两种写法都能吃 ✓）",
          coerce_cases([{"id": "x", "title": "短名"}])[0]["name"] == "短名", None)

    got = check_report(report())
    text = render_report(got)
    check("⑧ render 有 verdict 与计数 ✓（前端/CLI 直接打印 ✓）",
          "verdict=" in text and "条（过" in text, text.splitlines()[:1])
    check("⑧ render 有结论句 ✓", "结论：" in text, text.splitlines()[-1])

    got = check_report({"plan": [], "cases": []})
    check("⑧ 空报告也渲染得出来 ✓（不抛异常 ✓）", "verdict=partial" in render_report(got), None)

    got = check_report("坏的")
    check("⑧ 报告不是对象 ⇒ 报 problem 不抛异常 ✓",
          got.problems and got.verdict == "partial", got.problems)


# ══════════════════════════════════════════════════════════════════════════
# ⑨ 路由
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    response = client.post("/api/v1/assets/acceptance/validate", json={"report": report()})
    data = response.json().get("data") or {}
    check("⑨ POST /assets/acceptance/validate 真能调用 ✓（200 + ok + verdict ✓）",
          response.status_code == 200 and data.get("ok") is True and data.get("verdict") == "pass",
          (response.status_code, data.get("problems")))

    manifest = {"assets": [{"asset_id": "char_a", "type": "character", "status": "to-generate",
                            "target_path": "video/char_a.png",
                            "acceptance_criteria": ["同一发型"]}]}
    response = client.post("/api/v1/assets/acceptance/validate",
                           json={"report": {"cases": []}, "manifest": manifest})
    data = response.json().get("data") or {}
    check("⑨ plan 为空 + 给了清单 ⇒ planSource=manifest ✓ 且未执行被点名 ✓",
          data.get("planSource") == "manifest" and data.get("summary", {}).get("unexecuted") == ["asset:char_a"],
          data.get("summary"))

    response = client.post("/api/v1/assets/acceptance/validate", json={"manifest": manifest})
    check("⑨ 缺 report ⇒ 400 ✓（不默认「通过」✗）", response.status_code == 400, response.status_code)

    coverage = report(plan=[item(required=("audio",))],
                      cases=[case(status="pass", evidence=[ev("out/a.txt", "逐字听写")])])
    data = client.post("/api/v1/assets/acceptance/validate",
                       json={"report": coverage}).json()["data"]
    check("⑨ 覆盖率门禁经 API 也改判 ✓（前端能直接画出哪条被改判 ✓）",
          data.get("verdict") == "fail" and data["cases"][0].get("enforcedStatus") is True,
          data.get("cases"))


def main() -> int:
    case_vocab()
    case_evidence_type()
    case_plan()
    case_evidence()
    case_coverage_gate()
    case_verdict()
    case_plan_from_manifest()
    case_shape()
    case_api()

    failures = [entry for entry in _RESULTS if not entry[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
