"""验收报告合同 —— 学自 lobehub ``acceptance/skills/acceptance``（``report-schema.md`` / ``evidence.md`` / ``report.md``）。

**为什么本仓需要它**：本仓的「验收」现在散成三处互不相认的形状 ——

* ``asset_manifest`` 的 ``acceptance_criteria``：只是一条 warning（"没有就凭感觉"），
  没人真去对照，更没有"交了什么证据"的位置；
* ``film_check`` 的 ``verdict``（ready／partial／blocked）：结论有了，但说不出凭哪几份证据；
* ``evaluator`` 的 ``score``：命中即分，与证据无关。

三处都回答不了同一句话：**这一件产物凭什么算过？** 参考实现把这句话固化成一份
可机检的报告（``plan`` 先声明、``cases`` 后执行、``evidence`` 逐条挂介质），
本题把它搬成纯函数层：不碰 DB、不调模型、毫秒级，能挡在**花钱之前**。

五条真机制（缺一条就退化成"写份总结"）：

1. **先声明再执行**：``plan[]`` 早于 ``cases[]``。每条检查先写 ``expected``（什么算过）
   与 ``requiredEvidence``（要什么介质），再去看产物。事后补证据很难"凭空过"——
   要过就必须指明介质，而介质是可以被机器核的。
2. **no evidence, no claim**：``pass``／``fail`` 的结论必须挂至少一条证据；
   真跑不了才是 ``blocked``。没有证据的"通过"在本模块直接不成立。
3. **覆盖率门禁**：声明要 ``audio`` 却只交了 ``text`` ⇒ **该条判 fail**（不是警告）。
   这条是整套东西的牙齿 —— 没有它，``requiredEvidence`` 只是装饰。
4. **证据要带能读懂的一句话**：``description`` 空、或只是文件名／``JSON``／``evidence``
   时**不算证据**（参考原话：拿文件名兜底不够）。
5. **未验到的条目明说，不许消失**：``plan`` 里没有对应 case 的条目记进
   ``unexecuted``，``verdict`` 至多 ``partial``。「没验到」≠「通过」≠「失败」。

⚠️ 两处**有意**与参考不同（免得被当 bug 改回去）：

* ``summary.verdict`` 取三值 ``pass``／``fail``／``partial``（参考 ``report.md`` 原文）；
  "没验到"落 ``partial`` 而**不是** ``pass``，与参考"允许 inconclusive"同义。
* ``verdict`` **以 cases 现算为准**：报告里手写的 ``summary.verdict`` 与现算不一致 ⇒
  报 problem 并采用现算值（参考 ingest 也 recount，但它只 warn；本仓要求"报告要能被
  人重新数一遍"，绿灯写不死）。
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

__all__ = [
    "DEFAULT_VERIFIER",
    "EVIDENCE_TYPES",
    "STATUSES",
    "SURFACES",
    "VERDICTS",
    "VERIFIERS",
    "ReportCheck",
    "check_report",
    "coerce_cases",
    "coerce_plan",
    "derive_verdict",
    "evidence_type_of",
    "plan_from_manifest",
    "render_report",
]

#: 谁给出这条结论 ✓（``program``＝机器判 ✓、``agent``＝智能体/人目视 ✓、``llm``＝语义判官 ✓）
VERIFIERS: tuple[str, ...] = ("program", "agent", "llm")
#: 参考实现的默认值就是 ``agent`` ✓（多数验收是"看一眼"）✓
DEFAULT_VERIFIER = "agent"

#: 证据介质闭集 ✓（参考 ``evidence.md`` 原表 ✓ 一个不多一个不少 ✓）
EVIDENCE_TYPES: tuple[str, ...] = (
    "screenshot",
    "gif",
    "video",
    "audio",
    "text",
    "markdown",
    "dom_snapshot",
    "transcript",
)

#: 单条 case 的状态 ✓（``blocked`` 是**合法出口** ✓：拿不到那份介质就标它 ✓ 别拿别的顶替 ✓）
STATUSES: tuple[str, ...] = ("pass", "fail", "blocked")
#: 结论面 ✓（``report.md`` 原文三值 ✓）
VERDICTS: tuple[str, ...] = ("pass", "fail", "partial")
#: 产物面 ✓（本仓用得上的是 cli／desktop／web ✓）
SURFACES: tuple[str, ...] = ("web", "desktop", "cli", "mobile", "bot")

#: 扩展名 ⇒ 介质 ✓。参考的 ingest 就是这么自动归类的（``.mp3`` ⇒ audio）✓
#: ⚠️ 它**不**从扩展名推断 provenance（``by``）✗ —— 谁交的必须自己写 ✓ 这里照此二分 ✓
_EVIDENCE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "screenshot": (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".avif"),
    "gif": (".gif",),
    "video": (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"),
    "audio": (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"),
    "dom_snapshot": (".html", ".htm", ".mhtml", ".dom"),
    "markdown": (".md", ".mdx"),
    "transcript": (".jsonl",),
    "text": (".txt", ".log", ".json", ".csv", ".tsv", ".yaml", ".yml", ".sql", ".diff", ".patch"),
}

#: 只写这些就等于没写 ✓（含中文同义）✓ —— 参考原话：filename fallback 不够 ✓
_WEAK_DESCRIPTIONS = frozenset(
    {
        "",
        "json",
        "text",
        "txt",
        "log",
        "evidence",
        "proof",
        "screenshot",
        "视频",
        "音频",
        "截图",
        "证据",
        "日志",
        "见文件",
        "见附件",
        "seefile",
        "seethelog",
        "thelog",
    }
)


def _as_list(value: Any) -> list[Any]:
    """容忍 ``None``／单值／序列 ✓（清单类字段经常从 JSON 里长成三种形状 ✓）。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def evidence_type_of(entry: Mapping[str, Any] | str) -> str | None:
    """一条证据的**介质类型** ✓（显式 ``type`` 优先 ✓ 否则按扩展名认 ✓）。

    认不出就返 ``None`` ✓ **不猜** ✓ —— 猜成 ``text`` 会让覆盖率门禁假绿 ✓
    （声明要 audio、交了个 ``.bin``，猜成 text 之后"覆盖了"，其实什么都没证明）✓。

    显式写了但不在闭集里 ⇒ **原样返回那个值** ✓，由上层报"不在闭集" ✓
    （这里吞掉它会让错误变成"认不出介质"，读起来像格式问题而不是拼错）✓。
    """
    if isinstance(entry, str):
        entry = {"path": entry, "type": ""}
    if not isinstance(entry, Mapping):
        return None
    explicit = str(entry.get("type") or "").strip().lower()
    if explicit:
        return explicit
    path = str(entry.get("path") or "").strip()
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    if not suffix:
        return None
    for kind, suffixes in _EVIDENCE_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    return None


def _is_weak_description(description: str, path: str) -> bool:
    """``description`` 是否**等于没写** ✓（空／纯介质名／只是文件名）✓。"""
    text = str(description or "").strip()
    if not text:
        return True
    key = re.sub(r"[\s_\-.:：、，,（）()\[\]]+", "", text).lower()
    if key in _WEAK_DESCRIPTIONS:
        return True
    name = PurePosixPath(str(path or "").replace("\\", "/")).name.lower()
    if name and text.lower() in {name, name.rsplit(".", 1)[0]}:
        return True
    return False


def coerce_plan(raw: Any) -> list[dict[str, Any]]:
    """把 ``plan`` 归一成条目数组 ✓（坏条目**留空壳**让校验器点名 ✗ 不静默丢 ✓）。

    丢掉坏条目是最坏的一种"干净"：报告里 5 条检查变 4 条，没人发现少了一条。
    """
    out: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if not isinstance(item, Mapping):
            out.append({"id": "", "title": "", "category": "", "verifier": "",
                        "method": "", "expected": "", "requiredEvidence": []})
            continue
        out.append(
            {
                "id": str(item.get("id") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "category": str(item.get("category") or "").strip(),
                "verifier": str(item.get("verifier") or DEFAULT_VERIFIER).strip().lower(),
                "method": str(item.get("method") or "").strip(),
                "expected": str(item.get("expected") or "").strip(),
                "requiredEvidence": [
                    str(x).strip().lower() for x in _as_list(item.get("requiredEvidence"))
                ],
            }
        )
    return out


def _coerce_evidence(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in _as_list(raw):
        if isinstance(entry, str):
            out.append({"path": entry.strip(), "type": "", "description": "", "by": ""})
        elif isinstance(entry, Mapping):
            out.append(
                {
                    "path": str(entry.get("path") or entry.get("file") or "").strip(),
                    "type": str(entry.get("type") or "").strip().lower(),
                    "description": str(entry.get("description") or "").strip(),
                    "by": str(entry.get("by") or "").strip(),
                }
            )
    return out


def coerce_cases(raw: Any) -> list[dict[str, Any]]:
    """把 ``cases`` 归一 ✓（``id`` 是与 ``plan`` 对齐的**唯一键** ✓ 别忘了抄 ✓）。"""
    out: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if not isinstance(item, Mapping):
            out.append(
                {
                    "id": "",
                    "name": "",
                    "category": "",
                    "surface": "",
                    "status": "",
                    "declaredStatus": "",
                    "observation": "",
                    "evidence": [],
                }
            )
            continue
        status = str(item.get("status") or "").strip().lower()
        out.append(
            {
                "id": str(item.get("id") or "").strip(),
                "name": str(item.get("name") or item.get("title") or "").strip(),
                "category": str(item.get("category") or "").strip(),
                "surface": str(item.get("surface") or "").strip().lower(),
                "status": status,
                "declaredStatus": status,
                "observation": str(item.get("observation") or "").strip(),
                "evidence": _coerce_evidence(item.get("evidence")),
            }
        )
    return out


def _asset_evidence_for(item: Mapping[str, Any]) -> tuple[str, str]:
    """一份资产**该交什么介质** ✓ ⇒ ``(介质, 说明)``（说明非空 ⇔ 介质是默认猜的 ✓）。"""
    kind = evidence_type_of({"path": str(item.get("target_path") or "")})
    if kind and kind in EVIDENCE_TYPES:
        return kind, ""
    return "screenshot", (
        f"{item.get('asset_id')} 的 target_path 认不出介质 ✓ ⇒ 计划先按 screenshot 要求 ✓"
        f"（实际是音频/视频就改这一格 ✗ 别让它一路假绿 ✗）"
    )


def plan_from_manifest(manifest: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """把资产清单的 ``acceptance_criteria`` 变成「先声明再执行」的验收计划 ✓。

    本仓独有的桥：``asset_manifest`` 收了 ``acceptance_criteria`` 却只当 warning ✓，
    于是"验收"一直是凭感觉 ✓。这里把它变成可判的 plan 条目 ✓ —— 每条资产的每条
    准则都会落进 ``expected``（逐条对照 ✓ 不合并成一句"看起来没问题" ✗）✓。

    返回 ``(plan, notes)``：``notes`` 是**计划写不出来的部分** ✓（哪些资产没有准则 ✓）——
    缺口必须明说 ✓ 不能因为"没有准则"就少一条检查还装作计划完整 ✓。
    """
    notes: list[str] = []
    plan: list[dict[str, Any]] = []
    assets = _as_list(manifest.get("assets")) if isinstance(manifest, Mapping) else []
    if not assets:
        notes.append("清单里没有 assets ⇒ 生成不出验收计划 ✓")
        return plan, notes
    for index, item in enumerate(assets):
        if not isinstance(item, Mapping):
            notes.append(f"assets[{index}] 不是对象 ⇒ 跳过 ✓（清单侧会一起报 ✓）")
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        if not asset_id:
            notes.append(f"assets[{index}] 没有 asset_id ⇒ 无法登记验收项 ✓")
            continue
        criteria = [str(x).strip() for x in _as_list(item.get("acceptance_criteria"))
                    if str(x).strip()]
        if not criteria:
            notes.append(
                f"{asset_id} 没有 acceptance_criteria ⇒ 这份资产**没有验收计划** ✓"
                f"（「凭什么算过」还空着 ✗ 清单侧同一条 warning ✓）"
            )
            continue
        medium, defaulted = _asset_evidence_for(item)
        if defaulted:
            notes.append(defaulted)
        plan.append(
            {
                "id": f"asset:{asset_id}",
                "title": f"{asset_id} 通过资产验收",
                "category": "资产验收",
                "verifier": "agent",  # 看图对照 ✓ 机器判不了"像不像那个人" ✓
                "method": f"对照 {len(criteria)} 条准则逐条看 {item.get('target_path') or '产物'}",
                "expected": "；".join(criteria),
                "requiredEvidence": [medium],
            }
        )
    return plan, notes


# ══════════════════════════════════════════════════════════════════════════
# 报告本体
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ReportCheck:
    """一份验收报告的机检结果 ✓（``ok=False`` ⇔ 有 problem ✓）。

    ``verdict`` **只用现算值** ✓；手写值与现算不一致会进 ``problems`` ✓，
    但**不采信手写值** ✗ —— 报告要能被任何人重新数一遍 ✓。
    """

    plan: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    planSource: str = "report"

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def verdict(self) -> str:
        return str(self.summary.get("verdict") or "partial")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "planSource": self.planSource,
            "plan": self.plan,
            "cases": self.cases,
            "summary": self.summary,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def derive_verdict(cases: Sequence[Mapping[str, Any]], unexecuted: Sequence[str]) -> str:
    """由**现算的 cases** 推结论 ✓（``fail`` 有失败 ✓、``partial`` 没验全 ✓、``pass`` 才叫过 ✓）。

    顺序是硬的：
    * 有一条 ``fail`` ⇒ ``fail``（有失败时"没验全"不该盖住失败 ✓）；
    * 有 ``blocked`` 或有计划项没执行 ⇒ ``partial``（**不许算过** ✓）；
    * 一条 case 都没有 ⇒ ``partial``（"什么都没验"不是通过 ✓ 这是最容易犯的假绿 ✗）；
    * 剩下才是 ``pass`` ✓。
    """
    if any(str(case.get("status") or "") == "fail" for case in cases):
        return "fail"
    if unexecuted or any(str(case.get("status") or "") == "blocked" for case in cases):
        return "partial"
    if not cases:
        return "partial"
    return "pass"


def _summarize(cases: Sequence[Mapping[str, Any]], unexecuted: Sequence[str]) -> dict[str, Any]:
    total = len(cases)
    passed = sum(1 for case in cases if case.get("status") == "pass")
    failed = sum(1 for case in cases if case.get("status") == "fail")
    blocked = sum(1 for case in cases if case.get("status") == "blocked")
    verdict = derive_verdict(cases, unexecuted)
    if verdict == "fail":
        conclusion = f"{failed} 条没过（共 {total} 条）⇒ 先修这些再谈通过"
    elif verdict == "partial":
        reasons: list[str] = []
        if blocked:
            reasons.append(f"{blocked} 条跑不了（blocked）")
        if unexecuted:
            reasons.append(f"{len(unexecuted)} 条计划项没执行")
        if not reasons:
            reasons.append("没有可数的 case")
        conclusion = "没有失败，但" + "、".join(reasons) + " ⇒ 这份报告还不能算通过"
    else:
        conclusion = f"{total} 条全部通过，证据齐备"
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "unexecuted": list(unexecuted),
        "verdict": verdict,
        "conclusion": conclusion,
    }


def check_report(payload: Any, *, manifest: Any = None) -> ReportCheck:
    """校验一份验收报告 ✓（不抛异常 ✓ 全线报进 ``problems`` ✓）。

    ``manifest`` 给了且 ``report.plan`` 为空 ⇒ 用清单的 ``acceptance_criteria`` 现生成计划 ✓
    （``planSource="manifest"`` ✓）✓。这是本仓的接法：清单里已经有准则，不必让调用方抄一遍 ✓。
    """
    result = ReportCheck()

    if not isinstance(payload, Mapping):
        result.problems.append("验收报告必须是一个对象（含 plan / cases / summary）")
        result.summary = _summarize([], [])
        return result

    plan = coerce_plan(payload.get("plan"))
    if not plan and manifest is not None:
        plan, notes = plan_from_manifest(manifest)
        result.planSource = "manifest"
        result.warnings.extend(notes)

    # ── ① plan 逐条校验（先声明的那一半）────────────────────────────────
    plan_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(plan):
        who = item["id"] or f"plan[{index}]"
        if not item["id"]:
            result.problems.append(f"{who} 缺少 id ⇒ 后面的 case 对不上它（覆盖门禁也就无从判）")
        elif item["id"] in plan_by_id:
            result.problems.append(f"plan 里 id 重复：{item['id']} ⇒ 两条检查共用一格证据")
        else:
            plan_by_id[item["id"]] = item
        for name, label in (
            ("title", "title（这条查什么）"),
            ("category", "category"),
            ("method", "method（怎么查）"),
            ("expected", "expected（什么算过）"),
        ):
            if not item[name]:
                result.problems.append(f"{who} 缺少 {label} ⇒ 「先声明再执行」只做了一半")
        if item["verifier"] not in VERIFIERS:
            result.problems.append(
                f"{who} 的 verifier={item['verifier']!r} 不在 {list(VERIFIERS)}"
                f"（默认 {DEFAULT_VERIFIER}）"
            )
        required = item["requiredEvidence"]
        if not required:
            result.problems.append(
                f"{who} 没声明 requiredEvidence ⇒ 这条检查判不了「声明要的证据有没有交」"
                f"（参考要求 one or more，别留空壳）"
            )
        else:
            bad = [x for x in required if x not in EVIDENCE_TYPES]
            if bad:
                result.problems.append(
                    f"{who} 的 requiredEvidence 里有 {bad} 不在闭集 {list(EVIDENCE_TYPES)}"
                )
    if not plan:
        result.warnings.append(
            "报告没有 plan ⇒ 这是一份**事后补的**验收，没有「先声明要验什么」（参考要求 plan 早于 cases）"
        )

    # ── ② cases 逐条校验（执行的那一半）────────────────────────────────
    cases = coerce_cases(payload.get("cases"))
    seen_ids: set[str] = set(case["id"] for case in cases if case["id"])
    unexecuted = [pid for pid in plan_by_id if pid not in seen_ids]

    for index, case in enumerate(cases):
        who = case["id"] or f"cases[{index}]"
        if not case["id"]:
            result.problems.append(f"{who} 缺少 id ⇒ 与 plan 对不上（对不上就是「没执行」）")
        if plan_by_id and case["id"] and case["id"] not in plan_by_id:
            result.warnings.append(
                f"{who} 不在 plan 里 ⇒ 这是**事后补的**检查（不算计划项，也不给它兜覆盖率）"
            )

        declared = case["declaredStatus"]
        if declared not in STATUSES:
            result.problems.append(
                f"{who} 的 status={declared!r} 不在 {list(STATUSES)}"
                f"（跑不了就写 blocked，别写「待定」）"
            )
        if case["surface"] and case["surface"] not in SURFACES:
            result.problems.append(
                f"{who} 的 surface={case['surface']!r} 不在 {list(SURFACES)}"
            )
        if declared in ("pass", "fail") and not case["observation"]:
            result.problems.append(f"{who} 判了 {declared} 却没写 observation ⇒ 结论没有依据可读")

        # 证据逐条核（介质 / 说明）
        kinds: set[str] = set()
        for evidence in case["evidence"]:
            path = evidence["path"]
            if not path:
                result.problems.append(f"{who} 有一条证据没有 path ⇒ 点不开的证据不算证据")
            kind = evidence_type_of(evidence)
            if kind is None:
                result.problems.append(
                    f"{who} 的证据 {path or '(无 path)'} 认不出介质"
                    f" ⇒ 显式写 type（{list(EVIDENCE_TYPES)}），别让覆盖门禁靠猜"
                )
            elif kind not in EVIDENCE_TYPES:
                result.problems.append(
                    f"{who} 的证据 {path or '(无 path)'} 写了 type={kind!r}"
                    f"，不在闭集 {list(EVIDENCE_TYPES)}"
                )
            else:
                kinds.add(kind)
            if _is_weak_description(evidence["description"], path):
                result.problems.append(
                    f"{who} 的证据 {path or '(无 path)'} 的 description 等于没写"
                    f"（写「看到什么、凭什么」；只写文件名/「JSON」不合格）"
                )

        # no evidence, no claim
        # ⚠️ pass 无证据要**降级成 blocked** ✗ 不能只报一条 problem 就留着 pass ✓ ——
        #    否则"有 problem 但 verdict=pass"这种自相矛盾的报告会出现 ✓（读数与结论打架 ✓）
        status = declared
        if declared in ("pass", "fail") and not case["evidence"]:
            result.problems.append(
                f"{who} 判了 {declared} 却一条证据都没有 ⇒ 结论不成立"
                f"（没有证据只能写 blocked）"
            )
            if declared == "pass":
                status = "blocked"
        # 状态写错 ⇒ 保守落"没跑成" ✓（至多 partial ✓ 绝不许落 pass ✓）
        if status not in STATUSES:
            status = "blocked"

        # 覆盖率门禁：声明要的介质没交 ⇒ 该条判 fail（跑不了时只提示）
        # ⚠️ 这是整套东西的牙齿 ✓：没有它 requiredEvidence 只是装饰 ✓
        if case["id"] in plan_by_id:
            required = set(plan_by_id[case["id"]]["requiredEvidence"])
            missing = sorted(required - kinds)
            if missing:
                if status == "blocked":
                    result.warnings.append(
                        f"{who} 声明要 {missing} 证据而没交 ⇒ blocked 是正确出口"
                        f"（参考原话：拿不到这份介质就标 blocked，别拿别的介质顶替），"
                        f"但 observation 要写清为什么拿不到"
                    )
                else:
                    status = "fail"
                    result.problems.append(
                        f"{who} 声明要 {missing} 证据，实际只交了 {sorted(kinds) or '没有'}"
                        f" ⇒ 判 fail（覆盖率门禁：缺这份介质就不能说它通过）"
                    )
        case["status"] = status
        if status != case["declaredStatus"]:
            case["enforcedStatus"] = True

        # CLI 类结论通常要"推理件 + 执行件"两份 text ✓（参考原话 normally ✓ ⇒ 只提示 ✓）
        if case["surface"] == "cli" and declared != "blocked":
            text_like = kinds & {"text", "markdown", "transcript"}
            if len(text_like) < 2 and declared == "pass":
                result.warnings.append(
                    f"{who} 是 CLI 类结论却只有 {len(text_like)} 份 text 证据"
                    f"（推理件 + 执行件各一份才算齐）"
                )

    # ── ③ 汇总（只用现算值）────────────────────────────────────────────
    result.plan = plan
    result.cases = cases
    result.summary = _summarize(cases, unexecuted)

    raw_summary = payload.get("summary")
    declared_verdict = ""
    if isinstance(raw_summary, Mapping):
        declared_verdict = str(raw_summary.get("verdict") or "").strip().lower()
    if declared_verdict and declared_verdict not in VERDICTS:
        result.problems.append(
            f"summary.verdict={declared_verdict!r} 不在 {list(VERDICTS)}"
        )
    elif declared_verdict and declared_verdict != result.summary["verdict"]:
        result.problems.append(
            f"summary.verdict 写的是 {declared_verdict}，按 cases 现算是 "
            f"{result.summary['verdict']} ⇒ 采用现算值"
            f"（手写绿灯不算数：报告要能被人重新数一遍）"
        )

    return result


def render_report(result: ReportCheck) -> str:
    """人读摘要 ✓（前端/CLI 直接打印 ✓ 与 ``to_dict()`` 同源 ✓）。"""
    summary = result.summary or _summarize([], [])
    lines = [
        f"验收报告 ｜ verdict={summary['verdict']} ｜ "
        f"{summary['total']} 条（过 {summary['passed']} / 未过 {summary['failed']} / "
        f"跑不了 {summary['blocked']}）｜ 计划 {len(result.plan)} 项，"
        f"未执行 {len(summary['unexecuted'])}"
    ]
    for case in result.cases:
        mark = {"pass": "✓", "fail": "✗", "blocked": "⏸"}.get(str(case["status"]), "?")
        enforced = "（覆盖门禁改判）" if case.get("enforcedStatus") else ""
        lines.append(
            f"  {mark} {case['id'] or '(无 id)'} {case['name']}{enforced} "
            f"｜ 证据 {len(case['evidence'])} 条"
        )
    if summary["unexecuted"]:
        lines.append("  未执行：" + "、".join(summary["unexecuted"]))
    for problem in result.problems:
        lines.append(f"  ✗ {problem}")
    for warning in result.warnings:
        lines.append(f"  ⚠ {warning}")
    lines.append(f"  结论：{summary['conclusion']}")
    return "\n".join(lines)
