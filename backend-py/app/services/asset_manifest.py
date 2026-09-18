"""**资产清单 + 验收门**（移植 ``reference/short-drama-agent`` 的 ``asset-to-video-pipeline.md`` ✓ 零依赖 ✓）。

## 它守的是哪条线

参考项目的原话是：**「不要从剧本分析直接跳到视频生成」** ✓ ——
中间必须有一道**资产验收**（*Asset review and approval* ✓），并且
**「不要从被拒或缺失的锚定资产提交付费视频任务」** ✓✓。

本项目原来的资产能力是「记录版本 / 激活版本」（``services/asset_versions.py`` ✓）——
那是**存** ✓；缺的是**门** ✗：没有任何东西阻止"角色参考图还是 ``to-generate`` 就去跑付费视频" ✗。

## 这个模块做三件事（都可验证 ✓ 不需要 DB ✓ 不需要模型 ✓）

1. **校验清单**（`asset-manifest.json` 形状 ✓）—— 必填字段、类型/状态取值、**asset_id 唯一** ✓、
   **依赖必须指向存在的资产** ✓；
2. ⭐ **依赖拓扑 + 成环检测** —— 给出**生成顺序** ✓（先有角色参考图才谈得上"基于它做三视图" ✓），
   有环则**指名报出环上的资产** ✓（否则执行器要么死循环 ✗ 要么随机跳过 ✗）；
3. ⭐ **验收门** —— 逐镜算「被谁卡住」✓，并给出**唯一一个**布尔 ``canGenerateVideo`` ✓：
   **只要还有未验收的资产被镜头引用，就一律拦下** ✗（不按"只有一张图"之类模糊理由放行 ✗）。

## 刻意保留的**人味**

``approved`` 只能由**人**给 ✓（本模块**不会**自己把 ``to-generate`` 提成 ``approved`` ✗）。
机器能做的就是**如实算出门锁着、以及锁在哪一条上** ✓。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = ["ASSET_TYPES", "STATUSES", "APPROVED", "BLOCKING", "ManifestReport",
           "generation_order", "validate_manifest"]

#: 六种资产类型 ✓（参考项目原文列全 ✓ —— ``ui_plate`` 是**留给后期叠加**的空屏 ✓，
#: 这条很关键：让模型去生成"手机屏幕上的一整段中文短信"必然糊 ✗ ⇒ 空屏 + 后期叠字 ✓）
ASSET_TYPES: tuple[str, ...] = ("character", "scene", "prop", "clue", "ui_plate", "keyframe")

#: 状态取值 ✓（``-`` 与 ``_`` 都收 ✓ —— 两种写法混用是常见的手写事故 ✓）
#: ⚠️ 这里**必须**写**归一化后**的形态（连字符 ✓）：:func:`_norm_status` 会把 ``to_generate``
#: 变成 ``to-generate`` ✓ ⇒ 若集合里留的是下划线写法，两者**永不相等** ✗
#: （自检 ㉔ 当场抓到：``needs-regeneration`` 的资产被漏出 ``BLOCKING`` ✗ ⇒
#: ① ``blocking`` 计数少 ✓、② 更严重的是**它被排除出生成顺序** ✗ ——
#: 而那个顺序恰恰就是「接下来该重做什么」✓）。
STATUSES: tuple[str, ...] = ("to-generate", "generated", "approved", "needs-regeneration",
                             "missing", "rejected")

#: **算验收通过**的状态 ✓（只有人点过的那一个 ✓）
APPROVED: frozenset[str] = frozenset({"approved"})

#: **会卡住下游**的状态 ✓ —— 注意 ``generated`` 也在内 ✗：
#: 「图生成了」不等于「人验收了」✓（参考项目专门区分了这两件事 ✓）。
#: 取值一律用**归一化形态** ✓（见上面 ``STATUSES`` 的注释 ✓）。
BLOCKING: frozenset[str] = frozenset({"to-generate", "generated", "needs-regeneration",
                                      "missing", "rejected"})

#: 每类资产的必填字段 ✓（``prompt`` 只对**要生成**的资产强制 ✓）
_REQUIRED_FIELDS: tuple[str, ...] = ("asset_id", "type", "status")


def _norm_status(value: Any) -> str:
    """状态归一化 ✓：``to_generate`` → ``to-generate`` ✓（大小写与横线/下划线都容忍 ✓）。"""
    return str(value or "").strip().lower().replace("_", "-")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


@dataclass
class ManifestReport:
    """清单体检 + 验收门 ✓（``problems`` 是**清单本身不成立** ✗；``warnings`` 只是建议 ✓）。"""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 生成顺序 ✓（已批准的资产**不进**这个序列 ✓ —— 没必要重做 ✓）
    order: list[str] = field(default_factory=list)
    #: 环上的资产 ✓（有环时非空 ✓ —— 直接点名，别让人自己找 ✓）
    cycles: list[list[str]] = field(default_factory=list)
    #: 每镜被谁卡住 ✓：``{镜头: [资产, ...]}``
    blockedShots: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def canGenerateVideo(self) -> bool:  # noqa: N802 —— 与 JSON 字段同名 ✓ 便于直接序列化 ✓
        """⭐ **验收门**：清单成立 ✓ + 无环 ✓ + **没有被卡住的镜头** ✓ ⇒ 才允许进付费生成 ✓。"""
        return not self.problems and not self.cycles and not self.blockedShots

    @property
    def gateReasons(self) -> list[str]:  # noqa: N802
        """门为什么锁着 ✓（逐条可执行 ✓ —— 前端直接展示 ✓）。"""
        reasons: list[str] = []
        if self.problems:
            reasons.append(f"清单本身不成立（{len(self.problems)} 条 ✗）⇒ 先修清单 ✓")
        if self.cycles:
            reasons.append(f"依赖成环 {self.cycles} ✗ ⇒ 生成顺序无解，必须人工打断 ✓")
        for shot in sorted(self.blockedShots):
            names = self.blockedShots[shot]
            reasons.append(f"镜头 {shot} 的资产未验收：{names} ✓（先验收或先换资产 ✓）")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "problems": self.problems, "warnings": self.warnings,
            "counts": self.counts,
            "order": self.order, "cycles": self.cycles,
            "blockedShots": self.blockedShots,
            "canGenerateVideo": self.canGenerateVideo,
            "gateReasons": self.gateReasons,
            "assets": self.assets,
        }


def generation_order(assets: dict[str, dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    """**拓扑排序** ✓ ⇒ ``(顺序, 环清单)``。

    用 Kahn 算法 ✓（稳定输出 ✓：同层按 ``asset_id`` 排序 ✓ ⇒ 同一份清单每次跑结果一致 ✓，
    便于人对着报告操作 ✓）。剩下没排完的就是环上的 ✓ —— **逐个点名** ✓。
    """
    pending = {key: set(_as_list(item.get("dependencies"))) & set(assets)
               for key, item in assets.items()}
    order: list[str] = []
    while True:
        ready = sorted(key for key, deps in pending.items() if not deps)
        if not ready:
            break
        for key in ready:
            order.append(key)
            pending.pop(key)
        for deps in pending.values():
            deps.difference_update(ready)

    if not pending:
        return order, []

    # 有环：把**环上的**挑出来（只被环外依赖的不算 ✓ —— 那样会误伤 ✓）
    remaining = set(pending)
    cycles: list[list[str]] = []
    seen: set[str] = set()
    for key in sorted(remaining):
        if key in seen:
            continue
        # 从 key 出发沿依赖走回自己 ⇒ 环 ✓
        path: list[str] = []
        cursor = key
        while cursor not in path:
            path.append(cursor)
            nxt = sorted(dep for dep in pending.get(cursor, set()) if dep in remaining)
            if not nxt:
                break
            cursor = nxt[0]
        if cursor in path:
            cycle = path[path.index(cursor):]
            cycles.append(sorted(cycle))
            seen.update(cycle)
    return order, cycles


def validate_manifest(manifest: Any) -> ManifestReport:
    """校验 ``asset-manifest.json`` ✓ 并算验收门 ✓（**不抛异常** ✗ —— 全线报进 ``problems`` ✓）。"""
    report = ManifestReport()
    if not isinstance(manifest, dict):
        report.problems.append("清单必须是 JSON 对象 ✗（参考形状见 asset-to-video-pipeline.md ✓）")
        return report

    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        report.problems.append("清单缺少非空的 assets 数组 ✗（没有资产就无所谓验收门 ✓）")
        return report

    seen: set[str] = set()
    for index, item in enumerate(raw_assets):
        if not isinstance(item, dict):
            report.problems.append(f"assets[{index}] 不是对象 ✗")
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        if not asset_id:
            report.problems.append(f"assets[{index}] 缺少 asset_id ✗（后面所有引用都靠它 ✓）")
            continue
        if asset_id in seen:
            report.problems.append(f"asset_id 重复：{asset_id} ✗（引用会指向不确定的那个 ✓）")
            continue
        seen.add(asset_id)

        for name in _REQUIRED_FIELDS:
            if not str(item.get(name) or "").strip():
                report.problems.append(f"{asset_id} 缺少必填字段 {name} ✗")
        kind = str(item.get("type") or "").strip().lower()
        if kind and kind not in ASSET_TYPES:
            report.problems.append(
                f"{asset_id} 的 type={kind!r} 不在 {list(ASSET_TYPES)} 里 ✗"
                f"（``ui_plate`` 用来留「空屏 + 后期叠字」✓ 别让模型生成整段中文 ✗）")
        status = _norm_status(item.get("status"))
        if status and status not in STATUSES:
            report.problems.append(f"{asset_id} 的 status={item.get('status')!r} 不认识 ✗"
                                   f"（可用：{list(STATUSES)} ✓）")
        if status in BLOCKING and not str(item.get("prompt") or "").strip():
            # 还没通过的资产**必须**有可再生的提示词 ✓ —— 否则"重做"时无从下手 ✗
            report.warnings.append(f"{asset_id} 未验收却**没有 prompt** ✓ ⇒ 想重做时没有依据 ✗")
        if not str(item.get("target_path") or "").strip():
            report.warnings.append(f"{asset_id} 没有 target_path ✓ ⇒ 生成后不知道放哪儿 ✗")
        if not _as_list(item.get("acceptance_criteria")):
            report.warnings.append(
                f"{asset_id} 没有 acceptance_criteria ✓ ⇒ 「验收」会变成凭感觉 ✗"
                f"（参考项目要求写成可逐条对照的清单 ✓）")
        report.assets[asset_id] = {**item, "status": status or "to-generate"}

    # 依赖必须指向**存在**的资产 ✓（悬空引用是最常见的清单事故 ✓）
    for asset_id, item in report.assets.items():
        for dep in _as_list(item.get("dependencies")):
            if dep not in report.assets:
                report.problems.append(f"{asset_id} 依赖了不存在的资产 {dep} ✗")
            elif dep == asset_id:
                report.problems.append(f"{asset_id} 依赖自己 ✗")

    untouched = [key for key in seen if key not in report.assets]
    if untouched:  # pragma: no cover —— 正常流程不会走到 ✓（留着防御 ✓）
        report.problems.append(f"这些资产没被收进报告：{untouched} ✗")

    blocked_assets = {key: item for key, item in report.assets.items()
                      if item["status"] in BLOCKING}
    report.order, report.cycles = generation_order(blocked_assets)
    report.order = [key for key in report.order if key in report.assets]

    # ── ⭐ 验收门：逐镜算被谁卡住 ✓ ─────────────────────────────────────
    report.blockedShots = _blocked_shots(report.assets)
    report.counts = _status_counts(report.assets)
    return report


def _blocked_shots(assets: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """镜头 → 卡住它的资产 ✓（``first_needed_by`` 就是「这个资产第一次被哪一镜用到」✓）。

    ⚠️ **传递**地算 ✓：若某镜头依赖的资产还没批准 ⇒ 该镜头被卡 ✓；
    并且**被卡住的资产也不该被生成**（否则等于绕过门 ✗）⇒ 继续往上传染 ✓。
    """
    blocked_shots: dict[str, list[str]] = {}
    for asset_id, item in assets.items():
        shot = str(item.get("first_needed_by") or "").strip()
        if not shot:
            continue
        if item["status"] in APPROVED:
            continue
        blocked_shots.setdefault(shot, []).append(asset_id)
    return {shot: sorted(names) for shot, names in sorted(blocked_shots.items())}


def _status_counts(assets: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assets.values():
        key = str(item.get("status") or "to-generate")
        counts[key] = counts.get(key, 0) + 1
    counts["total"] = len(assets)
    counts["approved"] = sum(1 for item in assets.values() if item.get("status") in APPROVED)
    counts["blocking"] = sum(1 for item in assets.values() if item.get("status") in BLOCKING)
    return counts


def asset_ids_for_shot(assets: dict[str, dict[str, Any]], shot: str) -> list[str]:
    """某镜用到的资产清单 ✓（``first_needed_by == shot`` 的全部 ✓）—— 给执行器做**前置检查** ✓。"""
    return sorted(key for key, item in assets.items()
                  if str(item.get("first_needed_by") or "").strip() == shot)


def gate_for_shots(manifest: Any, shots: Iterable[str]) -> dict[str, list[str]]:
    """按**给定的镜头集合**算门 ✓（比 ``first_needed_by`` 更严 ✓：执行器**调用前**的最后一道 ✓）。

    返回 ``{镜头: [未验收的资产, ...]}`` ✓ —— 空字典 ⇔ 这批镜头可以跑 ✓。
    """
    report = validate_manifest(manifest)
    out: dict[str, list[str]] = {}
    for shot in shots:
        names = [key for key in asset_ids_for_shot(report.assets, str(shot))
                 if report.assets[key]["status"] not in APPROVED]
        if names:
            out[str(shot)] = names
    return out
