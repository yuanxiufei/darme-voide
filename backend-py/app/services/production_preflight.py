"""**开跑前体检**（把四块已交付的能力**串成一条** ✓ 零依赖 ✓ 2026-09-18）。

## 它是什么、不是什么

**不是**新的判据 ✗ —— 它**只调用**已有的四个模块 ✓：

| 环节 | 调谁 | 回答什么 |
|---|---|---|
① 提示词落位 | :mod:`.shot_placeholders` | 占位符**解析成具体内容**了吗 ✓ 还有裸编号/XML 残留吗 ✗ |
② 质感层 | :mod:`.prompt_polish` | 提示词够硬吗 ✓（空话有没有物理锚点陪衬 ✓） |
③ 连续性 | :mod:`.continuity` | 这一镜接得上上一镜吗 ✓（道具时间线/越轴/提前暴露/转场动机 ✓） |
④ 验收门 | :mod:`.asset_manifest` | **付费生成**允许开始吗 ✓（未验收资产卡住了哪些镜头 ✗） |

**是**那条把四块连起来的线 ✓：前端要显示「本集生产准备度」时，
不必自己调四个端点再拼结论 ✗（那样每个页面都会写一遍合并逻辑 ✗）；
而且**顺序本身是有意义的** ✓ —— 提示词先落位 ✓、再验连续性 ✓、最后才谈钱 ✓（见 ``ORDER`` ✓）。

## ⭐ 输出的重点不是"红了几条"，而是**下一步做什么**

``nextActions`` 是一份**可执行顺序表** ✓（按"先改便宜的、后动花钱的"排 ✓）：
先补映射/动机这类**零成本**的 ✓，再验收资产 ✓，最后才生成 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import continuity as continuity_mod
from . import prompt_polish
from . import shot_placeholders as sp
from .asset_manifest import validate_manifest

__all__ = ["ORDER", "PreflightReport", "run_preflight"]

#: 体检环节的**执行顺序** ✓（从便宜到贵 ✓ —— 先改文字 ✓ 再改结构 ✓ 最后才谈钱 ✓）
ORDER: tuple[str, ...] = ("placeholders", "polish", "continuity", "assetGate")


@dataclass
class PreflightReport:
    """一份合并结论 ✓（``blockers`` 阻断开跑 ✗；``warnings`` 只是建议 ✓）。"""

    ready: bool = False
    blockers: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    nextActions: list[str] = field(default_factory=list)  # noqa: N815
    sections: dict[str, Any] = field(default_factory=dict)
    checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "blockers": self.blockers, "warnings": self.warnings,
                "nextActions": self.nextActions, "sections": self.sections,
                "checked": self.checked, "order": list(ORDER)}


def _block(report: PreflightReport, stage: str, message: str, **extra: Any) -> None:
    report.blockers.append({"stage": stage, "message": message, **extra})


def run_preflight(payload: Any) -> PreflightReport:
    """跑一遍体检 ✓ —— ``payload`` 里放**各自模块原本的形状** ✓（本函数不发明新字段 ✗）::

        {"promptText": "...", "maps": {...}, "polish": {...},
         "plan": {locations/characters/props/clues/shots}, "manifest": {assets}}

    任何一段缺省 ⇒ **跳过该段并在 ``checked`` 里如实标出** ✓（而不是假装检查过 ✗）。
    """
    report = PreflightReport()
    if not isinstance(payload, dict):
        _block(report, "input", "体检输入必须是 JSON 对象 ✗")
        return report

    # ── ① 占位符落位（零成本 ✓ 先做 ✓）──────────────────────────────────
    text = payload.get("promptText")
    if isinstance(text, str) and text.strip():
        resolved = sp.resolve_placeholders(text, sp.PlaceholderMaps(
            locations=dict((payload.get("maps") or {}).get("locations") or {}),
            roles=dict((payload.get("maps") or {}).get("roles") or {}),
            props=dict((payload.get("maps") or {}).get("props") or {}),
            clues=dict((payload.get("maps") or {}).get("clues") or {})))
        report.checked.append("placeholders")
        report.sections["placeholders"] = resolved.to_dict()
        for item in resolved.unresolved:
            _block(report, "placeholders",
                   f"编号 {item['value']}（{item['kind']}）没有映射 ✗ ⇒ 提示词里会留下"
                   f"**模型不认识的裸编号** ✓", value=item["value"])
        if resolved.residualTags:
            _block(report, "placeholders",
                   f"提示词里仍有类标签片段 {resolved.residualTags[:3]} ✗"
                   f"（写错闭合最常见 ✓ 扩散模型不认识它们 ✓）")

    # ── ② 质感层（零成本 ✓）──────────────────────────────────────────────
    polish_raw = payload.get("polish")
    if isinstance(polish_raw, dict) and polish_raw:
        polished = prompt_polish.build_prompt_layer(prompt_polish.PolishInputs(
            theme_tags=tuple(polish_raw.get("themeTags") or ()),
            character_scene=str(polish_raw.get("characterScene") or ""),
            lens=str(polish_raw.get("lens") or ""), palette=str(polish_raw.get("palette") or ""),
            texture=str(polish_raw.get("texture") or ""),
            shot_type=str(polish_raw.get("shotType") or ""),
            movement=str(polish_raw.get("movement") or ""),
            screen_direction=str(polish_raw.get("screenDirection") or ""),
            slice_text=str(polish_raw.get("slice") or ""),
            sounds=tuple(polish_raw.get("sounds") or ()),
            imperfections=tuple(polish_raw.get("imperfections") or ()),
            handheld=bool(polish_raw.get("handheld")),
            inner_monologue=bool(polish_raw.get("innerMonologue")),
            restrained_ending=bool(polish_raw.get("restrainedEnding"))))
        report.checked.append("polish")
        report.sections["polish"] = polished.to_dict()
        # ⚠️ 质感层的问题**不阻断** ✓（约束不够硬 ⇒ 画得差点 ✓ 不是接不上 ✓）
        report.warnings.extend({"stage": "polish", "message": item}
                               for item in polished.issues)

    # ── ③ 连续性（不花钱就能查 ✓ 但比前两段"贵"一点：要得改分镜 ✓）──────
    plan = payload.get("plan")
    if isinstance(plan, dict) and plan.get("shots"):
        checked = continuity_mod.check_continuity(plan)
        report.checked.append("continuity")
        report.sections["continuity"] = checked.to_dict()
        for item in checked.violations:
            _block(report, "continuity", item["message"],
                   shot=item.get("shot"), code=item.get("code"))
        report.warnings.extend({"stage": "continuity", "message": item.get("message")}
                               for item in checked.warnings)
        # §10 的「可删或可并」单独提示 ✓（它是**创作决定** ✓ 不阻断 ✓）
        if checked.deletable:
            report.warnings.append({
                "stage": "continuity",
                "message": f"有 {len(checked.deletable)} 个动作不推动任何东西 ✓"
                           f"（§10：可删或可并 ✓ —— 但删戏是创作决定，由人拍板 ✓）"})

    # ── ④ 验收门（**花钱前最后一道** ✓）──────────────────────────────────
    manifest = payload.get("manifest")
    if isinstance(manifest, dict) and manifest.get("assets"):
        gate = validate_manifest(manifest)
        report.checked.append("assetGate")
        report.sections["assetGate"] = gate.to_dict()
        if not gate.canGenerateVideo:
            for reason in gate.gateReasons:
                _block(report, "assetGate", reason)
        report.warnings.extend({"stage": "assetGate", "message": item}
                               for item in gate.warnings)

    if not report.checked:
        report.warnings.append({"stage": "input",
                                "message": "什么都没检查 ✓（既没给 promptText/polish ✓"
                                           "也没给 plan/manifest ✓）—— 结果不代表任何"})

    report.ready = not report.blockers
    report.nextActions = _next_actions(report)
    return report


def _next_actions(report: PreflightReport) -> list[str]:
    """把阻断项翻成**可执行顺序** ✓（先零成本的文字/结构 ✓ 再资产 ✓ 最后才是花钱 ✓）。"""
    actions: list[str] = []
    stages = {item.get("stage") for item in report.blockers}

    if "input" in stages:
        actions.append("先补上体检输入（promptText / polish / plan / manifest 至少一个 ✓）")
    if "placeholders" in stages:
        actions.append("① 先补占位符映射表 ✓（零成本：补 `maps` 即可 ✓ —— "
                       "**不要**让裸编号进提示词 ✗）")
    if "continuity" in stages:
        actions.append("② 再修连续性 ✓（零成本：改分镜/表 ✓ —— "
                       "道具状态要交代、转场要给动机、线索别提前露 ✓）")
    if "assetGate" in stages:
        gate = report.sections.get("assetGate") or {}
        order = gate.get("order") or []
        actions.append("③ 最后处理资产 ✓（要花钱 ✓）："
                       + (f"按生成顺序 {' → '.join(order[:5])} ✓；" if order else "")
                       + "未验收的资产**不得**进付费生成 ✗")
        cycles = gate.get("cycles") or []
        if cycles:
            actions.append(f"⚠️ 依赖成环 {cycles} ⇒ **先人工打断环** ✓（顺序无解 ✓）")
    if any(item.get("stage") == "assetGate" for item in report.blockers) and not actions:
        actions.append("按验收门逐条修 ✓")
    if not actions and not report.ready:
        actions.append("有阻断项但没有可执行建议 ⇒ 请看 ``blockers`` 原文 ✓")
    if report.ready:
        actions.append("✅ 可以开跑 ✓（注意：这只代表**事前**检查过了 ✓，"
                       "生成质量仍要看实际产出 ✓）")
    return actions
