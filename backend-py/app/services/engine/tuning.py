"""引擎参数覆盖层（P3）—— 把「跑次统计看出来的调整」变成**可回滚的一行配置**。

要解决的问题
============
``outcome_index`` 能算出「krea2 的 8 步档在夜戏里失败聚集 5 次」，
但结论写进台账之后**没有任何东西会去改行为** —— 台账是给人看的，参数在代码常量里。
于是每一条调优结论都得靠人改源码、跑回归、再提交，闭环断在最后一米。

本模块补的就是这一米：一个**外部 JSON**（``configs/engine/tuning.json``），
管线在组装 plan 之前读它，命中就换值并**把来源写进回执**。

四条克制（都是为了不变成"静默改行为"那种事故）
==============================================
1. **没有文件 = 行为逐字不变**。不是"开关关掉"，是"没有这条数据" ——
   所以本模块**不需要**环境变量开关，也不存在"忘记打开"的死代码。
2. **白名单 + 区间**：只有 :data:`TUNABLES` 里登记过的 (stage, field) 能改，
   且值必须落在 ``[low, high]``；未登记的字段一律 ``skipped`` 并给理由。
   ⚠️ 为什么不做成"任意字段任意值" ✗：本仓的参数语义**按族各不相同**
   （``steps`` 在 seedvr2 是 1 步蒸馏、在 wan22 是双专家 4 步、
   在 krea2 是 cfg=1 的正解 ...）—— 放开写就等于让一次统计上的偏差
   去改另一族的物理含义 ✓✗。
3. **只覆盖「调用方没表态」的值**（``request`` 的字段等于 :data:`TUNABLES` 的 ``default``）：
   用户显式传 ``steps=8`` 时必须按 8 跑，不能被"我们的调优"顶掉。
4. **写前备份 + 证据可核**：``set_field`` 要求 ``basis``（人话结论）与 ``evidence``
   （真实 run_id / 盘上文件），证据核不过**当场拒写**；每次写都把旧文件备份到
   ``configs/engine/_backups/``，``reset`` 一条命令回滚。

回执纪律
========
``apply_to_request`` 返回的 ``notes`` 会进 ``plan.warnings``（前端可见），
形如 ``参数来源: tuning:qwen_image.steps（30 → 24，依据 <id>）``。
**不写这条就等于偷偷改了参数** —— 出图不对时没人知道是配置在起作用。

建议（只读，另一半闭环）
========================
``suggest()`` 读 ``outcome_index`` 的跑次统计给出候选（失败聚集、降步数），
但它**一行都不写**：候选要人核过再用 :func:`set_field` 落盘。
为什么不让它自己应用 ✗：样本量只能挡住"一次运气"，挡不住"这批数据本身有偏"
（同一批夜戏、同一个 seed、同一台卡），影子验证必须在历史基准上回放才算数 ——
那是 ``_bench`` / 优化器那套的活，不是这里顺手能编的。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import hooks

__all__ = [
    "MIN_SAMPLES",
    "QC_STAGE",
    "TUNABLES",
    "Tunable",
    "apply_to_request",
    "clear_field",
    "load",
    "path",
    "problems",
    "report",
    "reset",
    "resolve",
    "set_field",
    "suggest",
]

#: 覆盖文件的默认位置（``ENGINE_TUNING_JSON`` 可整体改指向 —— 桌面端/测试要用）
_REPO_ROOT = Path(__file__).resolve().parents[4]


def path() -> Path:
    """覆盖文件的**唯一**路径来源（环境变量优先，其余现算 ⇒ 测试可整体隔离）。"""
    override = str(os.environ.get("ENGINE_TUNING_JSON") or "").strip()
    if override:
        return Path(override)
    return _REPO_ROOT / "configs" / "engine" / "tuning.json"


def _backup_dir() -> Path:
    return path().parent / "_backups"


@dataclass(frozen=True)
class Tunable:
    """一个允许被覆盖的字段。"""

    #: 「调用方没表态」时管线用的值 —— 只有请求里的值等于它时才会被覆盖换掉
    default: float
    low: float
    high: float
    note: str
    integer: bool = True


#: QC 伪档的 stage 名（``technical_qc.qc_thresholds()`` 读它）。
#: 它**不是引擎档**：``GenerationRequest`` 上没有这些字段名 ⇒ 永远换不进请求 ✓
QC_STAGE = "qc"

#: ⚠️ 逐族登记、**不要**做成通配：参数语义按族不同，通配就是"一次统计偏差改另一族的物理含义"
TUNABLES: dict[tuple[str, str], Tunable] = {
    ("sdxl", "steps"): Tunable(
        default=30, low=1, high=150,
        note="SDXL 单图。步数是纯成本项，低于 20 通常掉细节"),
    ("zimage", "steps"): Tunable(
        default=30, low=1, high=150,
        note="Z-Image（flow, shift=3）。8 步档可跑但夜景/文字容易糊"),
    ("qwen_image21", "steps"): Tunable(
        default=30, low=1, high=150,
        note="Qwen-Image 2.1（shift=0.69）。少于 20 步时脸容易崩"),
    ("qwen_image", "steps"): Tunable(
        default=30, low=1, high=150,
        note="Qwen-Image 2512（shift=3.1）。官方档 30 步"),
    ("krea2", "steps"): Tunable(
        default=30, low=1, high=150,
        note="Krea2。官方 8 步 turbos，但 cfg 必须 1.0；步数单改不解决过曝"),
    ("h3", "steps"): Tunable(
        default=30, low=1, high=150,
        note="H3 双流视频。步数与耗时近似线性"),

    # ——— QC 伪档（``stage="qc"``）：技术质检的判据阈值 ———
    #: 为什么塞进这张表而不是另起一套：白名单、区间、写前备份、**证据核验**
    #: （必须指向台账里真实存在的条目）、history、``problems()`` 全都现成。
    #: 而质检阈值恰恰**最需要**证据 —— 调低 ``blackminduration`` 就是让烂片过审、
    #: 调高就是批量误杀，凭感觉改一次，之后没人说得清"这批成片为什么突然全过/全挂"。
    #: ⚠️ 它**不是引擎档**：``apply_to_request`` 只认 ``GenerationRequest`` 上的字段名
    #: （``steps`` 等），这九个名字在请求对象上**一个都不存在** ⇒ 永远换不进请求 ✓
    #: （``tests/qc_thresholds_test.py`` 钉住）。
    #: ⚠️ 默认值必须与 ``technical_qc.TECH_QC_THRESHOLDS`` **逐值相等** —— 那份是 TS 镜像
    #: 常量（``route_parity_test`` 逐值比对），两处写法一漂就是两套判据 ✗
    #: （同样由那个测试核，这里**不 import** 它：``technical_qc`` 反过来要 import 本模块 ✓）。
    ("qc", "blackminduration"): Tunable(
        default=0.25, low=0.05, high=5.0, integer=False,
        note="黑场最小判定时长（s）。调低＝更易报黑场（误杀），调高＝放过短黑场"),
    ("qc", "blackpixthreshold"): Tunable(
        default=0.1, low=0.0, high=1.0, integer=False,
        note="blackdetect 的 pix_th：多暗才算黑（0＝纯黑，1＝全算黑）"),
    ("qc", "freezeminduration"): Tunable(
        default=1.0, low=0.1, high=10.0, integer=False,
        note="冻帧最小判定时长（s）。演出里的“静止”越长越容易误杀"),
    ("qc", "freezenoisethreshold"): Tunable(
        default=-60, low=-100, high=0, integer=False,
        note="freezedetect 的噪声阈值（dB）。越接近 0 越宽松"),
    ("qc", "truepeakdb"): Tunable(
        default=-1, low=-30, high=0, integer=False,
        note="音频真峰上限（dBTP）。超过即判削波风险"),
    ("qc", "loudnesstarget"): Tunable(
        default=-14, low=-40, high=0, integer=False,
        note="集成响度目标（LUFS）。平台口径不同（-14 是流媒体常用档）"),
    ("qc", "loudnesstolerance"): Tunable(
        default=1, low=0.1, high=10, integer=False,
        note="集成响度容差（LUFS）。调小＝同一批片更容易被判响度不合格"),
    ("qc", "fpsmin"): Tunable(
        default=15, low=1, high=60, integer=False,
        note="合理帧率下限。低于它判“帧率过低”（慢动作/抽帧的成片要按实需求调）"),
    ("qc", "durationtolerancesec"): Tunable(
        default=1, low=0.05, high=10, integer=False,
        note="时长偏差容差（s）。与分镜目标时长的允许差"),
}

#: 由 :data:`TUNABLES` 反推（不手抄）：加一个新字段只改那张表
_FIELDS = tuple(sorted({name for _, name in TUNABLES}))


def _empty() -> dict[str, Any]:
    return {"version": 1, "updatedAt": "", "overrides": {}, "history": []}


def load() -> dict[str, Any]:
    """读覆盖文件。**任何形态问题都降级为"没有覆盖"并记录问题**，绝不抛。"""
    file = path()
    if not file.exists():
        return _empty()
    try:
        # ⚠️ ``utf-8-sig`` 不是保险丝而是必需 ✓✗：Windows 上记事本「UTF-8」与 PowerShell 5.1
        #    的 ``Out-File``/``>`` **默认写 BOM** ✓，带 BOM 的文件用 ``utf-8`` 读出的首字符是
        #    ``\ufeff`` ⇒ ``json.loads`` 直接抛 ✓ ⇒ 落到下面的 ``_empty()`` ✓
        #    ⇒ 症状是「调优忽然全部失效、且一句话都不提示」✓✗（``diagnose()`` 也说不出原因 ✓）。
        data = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    overrides = data.get("overrides")
    data["overrides"] = overrides if isinstance(overrides, dict) else {}
    data.setdefault("version", 1)
    data.setdefault("history", [])
    return data


def _key(stage: str, name: str) -> str:
    return f"{str(stage or '').strip().lower()}.{str(name or '').strip().lower()}"


def resolve(stage: str, name: str, default: Any) -> tuple[Any, str]:
    """返回 ``(值, 来源)``。来源为 ``"default"`` 或 ``"override:<stage>.<field>"``。

    未登记 / 值非法 / 文件坏 —— 一律返回 ``(default, "default")``：
    **配置坏了按默认跑**比按一个读不出来的值跑安全得多（后者连报错都指不准）。
    """
    spec = TUNABLES.get((str(stage or "").strip().lower(), str(name or "").strip().lower()))
    if spec is None:
        return default, "default"
    raw = load()["overrides"].get(_key(stage, name))
    if not isinstance(raw, dict):
        return default, "default"
    value = _coerce(raw.get("value"), spec)
    if value is None:
        return default, "default"
    return value, f"override:{_key(stage, name)}"


def _coerce(value: Any, spec: Tunable) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (spec.low <= number <= spec.high):
        return None
    return int(round(number)) if spec.integer else number


def apply_to_request(stage: str, request: Any) -> tuple[Any, list[str]]:
    """把覆盖值换进 ``GenerationRequest``（不改调用方显式给的值）。

    返回 ``(新请求, notes)``；``notes`` 要进 ``plan.warnings`` 让前端看得见。
    为什么改请求而不是改 plan ✗：``plan`` 里的 σ / timesteps / costFactor **都是从
    ``request.steps`` 推出来的**，改 plan 会把「4 步的 σ」配上「30 步的 steps」——
    形状对、数不对，而且不会报错。
    """
    key = str(stage or "").strip().lower()
    notes: list[str] = []
    updated = request
    for name in _FIELDS:
        spec = TUNABLES.get((key, name))
        if spec is None or not hasattr(request, name):
            continue
        current = getattr(request, name)
        try:
            if float(current) != float(spec.default):
                continue  # 调用方显式表态 ⇒ 不覆盖
        except (TypeError, ValueError):
            continue
        value, source = resolve(key, name, spec.default)
        if source == "default":
            continue
        entry = load()["overrides"].get(_key(key, name)) or {}
        basis = str(entry.get("basis") or "").strip()
        updated = replace(updated, **{name: value})
        notes.append(
            f"参数来源: tuning:{_key(key, name)}（{spec.default:g} → {value:g}"
            + (f"，依据：{basis}" if basis else "") + "）"
        )
    return updated, notes


def problems() -> list[str]:
    """自查：登记表与文件互相核一遍（给 ``GET /tuning`` 与测试用）。"""
    found: list[str] = []
    file = path()
    if file.exists():
        try:
            # 与 :func:`load` 同一编码口径 ✓（两处不一致 ⇒ load 读进来了、这里却报"读不出来" ✓✗）
            json.loads(file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as err:
            found.append(f"覆盖文件读不出来（已按无覆盖处理）：{err}")
    for key, entry in (load()["overrides"] or {}).items():
        stage, _, name = key.partition(".")
        spec = TUNABLES.get((stage, name))
        if spec is None:
            found.append(f"{key}: 未登记的字段（不会被任何管线读取）")
            continue
        if not isinstance(entry, dict):
            found.append(f"{key}: 记录不是对象")
            continue
        if _coerce(entry.get("value"), spec) is None:
            found.append(f"{key}: 值 {entry.get('value')!r} 不在 [{spec.low:g}, {spec.high:g}] 或不是数")
        if not str(entry.get("basis") or "").strip():
            found.append(f"{key}: 缺 basis（人话结论）")
        if not (entry.get("evidence") or []):
            found.append(f"{key}: 缺 evidence（改了行为却说不清凭据）")
    return found


def report() -> dict[str, Any]:
    """当前有效值与来源一览（前端/CLI 用）。"""
    data = load()
    rows = []
    for (stage, name), spec in sorted(TUNABLES.items()):
        entry = data["overrides"].get(_key(stage, name)) or {}
        value, source = resolve(stage, name, spec.default)
        rows.append({
            "stage": stage, "field": name,
            "default": spec.default, "value": value,
            "source": source,
            "basis": entry.get("basis") or "",
            "evidence": entry.get("evidence") or [],
            "updatedAt": entry.get("updatedAt") or "",
            "note": spec.note,
        })
    return {"path": path().as_posix(), "exists": path().exists(),
            "fields": rows, "problems": problems()}


#: 「样本量不足只报不应用」的门槛（设计文档 §4.3 第一道闸门）。
#: ⚠️ 别调小到 10 那种"看着够用"：一次统计偏差就改默认参数，出图不对时没人能回溯 ✓✗
MIN_SAMPLES = 30

#: 一次建议最多扫多少条跑次（索引库可能有几万条；全拉进内存只为算一个均值不值得）
_SCAN_LIMIT = 500

#: 「同一族里最多拿几条跑次当凭据」—— 证据是给人点开核的，不是越长越好
_EVIDENCE_MAX = 8


def _index():
    """跑次索引的**注入源**（没登记 ⇒ ``None`` ✓；不 import 服务层 ✓）。"""
    return hooks.get("run_index")


def _scan(**filters: Any) -> tuple[list[dict[str, Any]], int, bool]:
    """取一批跑次：``(items, total, available)``。

    ``total`` 与 ``len(items)`` **必须分开**：扫到 500 条就说"共 500 条"会把
    「抽样了」读成「就这么多」，样本量闸门随即被绕过 ✓✗

    ⚠️ 没登记跑次索引 ⇒ 返回 ``([], 0, available=False)`` ✓✗（**不是**"没有失败跑次 ⇒ 一切正常" ✓✗）：
    ``available=False`` 会让上游 `suggest()` 如实报"没接入"，闸门按"样本量 0"处理 ✓。
    """
    index = _index()
    if index is None:
        return [], 0, False
    result = index.query(limit=_SCAN_LIMIT, **filters)
    return (result.get("items") or [], int(result.get("total") or 0),
            bool(result.get("available")))


def _gate(samples: int, *, field: str | None, min_samples: int) -> dict[str, Any]:
    """三道闸门各自的状态（缺一条就会变成"自我损坏"，所以逐条如实写）。"""
    return {
        #: ① 样本量
        "samples": "ok" if samples >= min_samples else "insufficient",
        #: ② 影子验证：本模块**不做**（要在历史基准上真回放 ⇒ 归 `_bench` / 优化器那套）
        "shadow": "pending" if field else "n/a",
        #: ③ 可回滚：`set_field` 写前备份 + `clear_field`/`reset`，所以引擎侧本来就是可回的
        "rollback": "ready" if field else "n/a",
    }


def _candidate(*, rule: str, stage: str, field: str | None, current: Any,
               proposed: Any, why: str, samples: int, evidence: list[str],
               min_samples: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    gate = _gate(samples, field=field, min_samples=min_samples)
    item = {
        "id": f"{rule}:{stage}" + (f":{field}" if field else ""),
        "rule": rule, "stage": stage, "field": field,
        "current": current, "proposed": proposed,
        "why": why, "samples": samples, "evidence": evidence[:_EVIDENCE_MAX],
        "gate": gate,
        #: 建议**从不**自动落盘：影子验证没做，闸门②就是 pending ⇒ 一律不许直接生效
        "canApply": bool(field) and gate["samples"] == "ok" and proposed != current
                     and gate["shadow"] != "pending",
    }
    if field:
        item["applyHint"] = ("人若已在同一批历史基准上复核过，可用 POST /tuning 写入"
                             "（写前自动备份、history 留痕，回执里会出现「参数来源: tuning:…」）")
    if extra:
        item.update(extra)
    return item


def _failure_candidates(*, min_samples: int,
                        stats: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """规则一：某族的 ``failed`` 跑次里 ``error_code`` 高度集中 ⇒ 先归因再动参数。

    两道 ``min_samples`` 检查是有意的冗余：``failed`` 来自索引给的计数，``len(items)`` 是
    真拿到的明细，两者不一致（被 ``_SCAN_LIMIT`` 截断）时**以明细为准**；它们同时保证
    ``counts`` 非空（``max()`` 空字典会直接 ValueError，表现成"建议接口炸了"而不是"没结论"）。
    """
    outcome_index = _index()
    notes: list[str] = []
    if outcome_index is None and stats is None:
        return [], ["跑次索引没接入（引擎可独立运行 ⇒ 由服务层启动时登记）"]
    stats = stats if stats is not None else outcome_index.stats(group_by="stage")
    if not stats.get("available"):
        return [], [str(stats.get("hint") or "索引库不可用")]
    found: list[dict[str, Any]] = []
    for group in stats.get("groups") or []:
        stage = str(group.get("key") or "")
        failed = int(group.get("failed") or 0)
        if not stage or failed < min_samples:
            continue
        items, total, _ = _scan(stage=stage, status="failed")
        if len(items) < min_samples:
            notes.append(f"{stage}: 失败 {failed} 条（只看得到 {len(items)} 条明细）⇒ 不计")
            continue
        counts: dict[str, int] = {}
        ids: dict[str, list[str]] = {}
        for item in items:
            code = str(item.get("error_code") or "(未写 error_code)")
            counts[code] = counts.get(code, 0) + 1
            rid = str(item.get("run_id") or "")
            if rid:
                ids.setdefault(code, []).append(rid)
        code, count = max(counts.items(), key=lambda kv: kv[1])
        share = count / len(items)
        #: 50% 是"高度集中"的下限：三条各占三分之一的失败说明是三类问题，不该归成一个结论
        if share < 0.5:
            continue
        found.append(_candidate(
            rule="failure-cluster", stage=stage, field=None, current=None, proposed=None,
            why=(f"{stage} 的失败 {len(items)} 条里 {count} 条是同一个错误"
                 f"（{share:.0%}）：`{code}`。先按这条错误定位（OOM ⇒ 降尺寸；"
                 f"尺寸不整除 ⇒ 改吸附步长），**别先动 steps** —— 参数改不动环境问题"),
            samples=len(items), evidence=ids.get(code) or [], min_samples=min_samples,
            extra={"errorCode": code, "share": round(share, 4),
                   "scanned": len(items), "failedTotal": failed,
                   "sampled": total > len(items)},
        ))
    if not found:
        notes.append(f"没有哪一族的失败在 ≥{min_samples} 条样本里集中到同一个 error_code")
    return found, notes


def _steps_candidates(*, min_samples: int) -> tuple[list[dict[str, Any]], list[str]]:
    """规则二：分数与 steps 的关系已经平了 ⇒ 降 steps（省时间不降质）。

    ⚠️ 只在 ``runs.meta`` **真的带 ``steps``** 时才算：meta 里没这格就没有"步数"这个维度，
    拿 ``elapsed`` 反推步数等于猜 —— 猜出来的建议比没有建议更坏 ✓✗
    """
    #: ⚠️ 这里**不**取 `_index()` ✓：本规则只经 `_scan()` 取数 ✓，而 `_scan` 已按"没登记"兜住 ✓
    #: （旧写法白拿一个可能为 ``None`` 的变量、再不用它 ⇒ 读的人以为有来源 ✓✗）。
    current_overrides = load()["overrides"]
    found: list[dict[str, Any]] = []
    notes: list[str] = []
    for (stage, name), spec in sorted(TUNABLES.items()):
        if name != "steps":
            continue
        items, total, available = _scan(stage=stage, status="ok")
        if not available:
            notes.append(f"{stage}: 索引库不可用，跳过")
            continue
        buckets: dict[int, list[float]] = {}
        ids: dict[int, list[str]] = {}
        for item in items:
            meta = item.get("meta")
            meta = meta if isinstance(meta, dict) else {}
            raw_steps, score = meta.get("steps"), item.get("score")
            try:
                steps = int(float(raw_steps))
                value = float(score)
            except (TypeError, ValueError):
                continue
            buckets.setdefault(steps, []).append(value)
            rid = str(item.get("run_id") or "")
            if rid:
                ids.setdefault(steps, []).append(rid)
        usable = {k: v for k, v in buckets.items() if len(v) >= max(3, min_samples // 10)}
        if sum(len(v) for v in usable.values()) < min_samples or len(usable) < 2:
            continue
        ordered = sorted(usable)
        high, second = ordered[-1], ordered[-2]
        avg_high = sum(usable[high]) / len(usable[high])
        avg_second = sum(usable[second]) / len(usable[second])
        #: 「多走那些步数买到多少分」—— 低于 0.01 就是白花时间（1 分制，0.01 已是肉眼可辨下限）
        gain = avg_high - avg_second
        if gain >= 0.01:
            continue
        #: 缺口 6：**降步数要过质检关** ✓ —— 「省时间不降质」的"质"不能只看引擎自报分数 ✗
        #: （那份分数在真机上**根本不存在** ✗：``runtime._index_run`` 从来不写 ``score`` ⇒
        #: 本规则在真机上一直靠"没有分数就没有候选"空转 ✓✗）。质检通过率是**有人/有判据
        #: 看过**的离散结论 ✓ ⇒ 低档通过率明显更差就当场拦下这条候选 ✓（并留下理由 ✓）。
        verdict = _qc_feedback().gate(stage, high, second)
        if verdict.get("verdict") == "blocked":
            notes.append(str(verdict.get("why") or f"{stage}: 降步数被质检闸门拦下"))
            continue
        value, source = resolve(stage, name, spec.default)
        if source != "default" and float(value) == float(second):
            continue  # 已经是这个值了，不必再提
        found.append(_candidate(
            rule="steps-diminishing", stage=stage, field=name,
            current=value, proposed=second,
            why=(f"{stage}：{high} 步均分 {avg_high:.3f}，{second} 步均分 {avg_second:.3f}"
                 f"（差 {gain:+.3f} < 0.01）⇒ 降回 {second} 步，省时间不降质；"
                 f"质检闸门 {verdict.get('verdict')}"),
            samples=len(usable[high]) + len(usable[second]),
            evidence=(ids.get(high) or []) + (ids.get(second) or []),
            min_samples=min_samples,
            extra={"avgScoreHigh": round(avg_high, 4), "avgScoreSecond": round(avg_second, 4),
                   "gain": round(gain, 4), "scanned": len(items),
                   "sampled": total > len(items),
                   "alreadyOverridden": source != "default",
                   "qcGate": verdict},
        ))
    notes.append(_qc_feedback().summary_line())
    if not found:
        notes.append("没有哪一族的「更高步数」比次高档多买到 ≥0.01 分（或样本/步数维度不足）")
    return found, notes


def _qc_reviews():
    """质检复核的**注入源**（没登记 ⇒ ``None`` ✓）。

    ⚠️ 为什么要注入而**不是** import ✗：`qc_reviews` → `technical_qc` → **本模块** ⇒ 模块级
    import 必成环 ✓✗（所以这里一直是 late import ✓）；而晚 import 仍然把方向定成
    「引擎 → 服务层」✓✗。改成服务层启动时登记，环与方向一起解开 ✓。
    """
    return hooks.get("qc_reviews")


def _qc_feedback():
    """延迟 import 质检反哺（它要读派生索引库；模块级 import 会把"读盘"提前到加载期 ✗）。"""
    from . import qc_feedback  # noqa: PLC0415

    return qc_feedback


def _next_threshold(current: float, spec: Tunable, *, up: bool) -> float | None:
    """把阈值朝 ``up`` 方向推一格，返回新值；**推不动**（已顶到区间边界）返回 ``None`` ✓。

    ⚠️ 「一格」是 **×1.25 / ×0.8 的起手值**，不是从数据里算出来的结论 ✓✗ —— 复核只告诉我们
    方向（误报多 ⇒ 放宽 ✓），并没有告诉我们幅度（那要拿误报样本的实测值分布来定 ✓，本轮没做 ✗）。
    所以：方向有据 ✓、幅度是**建议起手值** ✓，并在 ``why`` 里如实写明 ✓ —— 不把倍率说成证据 ✗。

    ⚠️ 负值区间乘法方向是**反的** ✓✗：``truePeakDb=-1`` 要"调高"（放宽）得 ×0.8 得 -0.8，
    而 ``-60 × 1.25 = -75`` 是**更严** ✗。别写成无脑 ``value * 1.25`` ✓。
    """
    if current == 0:
        #: 0 乘不动 ⇒ 退一步按区间的 2% 走（有明确上下界的字段才可能到这里 ✓）
        step = (spec.high - spec.low) * 0.02
        value = step if up else -step
    else:
        grow = 1.25 if current > 0 else 0.8
        shrink = 0.8 if current > 0 else 1.25
        value = current * (grow if up else shrink)
    value = max(spec.low, min(spec.high, value))
    value = round(value) if spec.integer else round(value, 4)
    return None if float(value) == float(current) else float(value)


def _qc_hit_rate_candidates(*, reviewed_min: int | None = None) -> tuple[
        list[dict[str, Any]], list[str], dict[str, Any] | None]:
    """规则三：按**人工复核**算出的阈值命中率 ⇒ 误报多就放宽、漏报多就收紧（§4.3 第四条）。

    这是 ``notEvaluated`` 里那条 ``qc-threshold-hit-rate`` 的正解 ✓：分母不再是"触发频次"
    （那是分子 ✗），而是人看过成片之后给的结论（``hit`` / ``false_alarm`` / ``missed`` ✓）。

    三条边界：
    1. **样本门槛比规则一二更严的执行口径**：复核是人工成本，``MIN_REVIEWS``（默认 10 条）
       以下的 code **一个候选都不出** ✓，只把"还差几条"写进 notes ✓。
    2. **只动主字段**：``technical_qc.ISSUE_THRESHOLDS[code]`` 给的 ``field`` ✓；
       配合字段（黑场的 pix_th、冻帧的噪声阈值）只提示不动 ✗ —— 两格同时动就说不清是哪一格起效 ✓。
    3. **无对应阈值的 code 不出候选** ✓（``spec.unreadable`` / ``spec.probe_failed`` 的"误报"
       是 ffprobe 环境问题，改阈值等于把探测失败藏起来 ✗✗）。
    """
    service = _qc_reviews()
    if service is None:
        why = "质检复核没接入（引擎可独立运行 ⇒ 由服务层启动时登记）⇒ 命中率没有分母"
        return [], [why], {"rule": "qc-threshold-hit-rate", "why": why}
    summary = service.hit_rate()
    notes: list[str] = []
    if not summary.get("available"):
        reason = str(summary.get("reason") or "复核表读不出来")
        return [], [f"质检复核不可用：{reason}"], {
            "rule": "qc-threshold-hit-rate",
            "why": f"质检复核表读不出来（{reason}）⇒ 命中率没有分母，本轮不出候选",
        }

    threshold = reviewed_min if reviewed_min is not None else int(summary.get("minReviews")
                                                                  or service.MIN_REVIEWS)
    found: list[dict[str, Any]] = []
    thin: list[str] = []
    for item in summary.get("codes") or []:
        code = str(item.get("code") or "")
        reviewed = int(item.get("reviewed") or 0)
        target = item.get("threshold")
        if not target:
            continue
        if reviewed < threshold:
            thin.append(f"{code}: 复核 {reviewed} 条 < 门槛 {threshold}")
            continue
        spec = TUNABLES.get((QC_STAGE, str(target.get("field"))))
        if spec is None:
            #: 映射表指向了白名单里没有的字段 ⇒ 说出来（不静默跳过，那会让"没候选"看起来像"没问题"）
            notes.append(f"{code}: 映射到未知阈值 `{target.get('field')}`（TUNABLES 里没有）⇒ 跳过")
            continue
        verdict = str(item.get("verdict") or "")
        if verdict == "false_alarm_heavy":
            relax, up = True, str(target.get("relax") or "raise") == "raise"
        elif verdict == "miss_heavy":
            #: 漏报多 ⇒ 收紧 = 放宽的反方向 ✓
            relax, up = False, str(target.get("relax") or "raise") != "raise"
        elif verdict == "insufficient":
            thin.append(f"{code}: 复核 {reviewed} 条 < 门槛 {threshold}")
            continue
        else:
            continue  # ``ok``：命中率与漏报率都在容忍范围内 ⇒ 不动它 ✓（"没结论"≠"没问题"✓）

        current, source = resolve(QC_STAGE, str(target["field"]), spec.default)
        proposed = _next_threshold(float(current), spec, up=up)
        if proposed is None:
            notes.append(f"{code}: `{target['field']}`={current} 已顶在 [{spec.low}, {spec.high}] "
                         f"边界，{'放宽' if relax else '收紧'}方向推不动")
            continue
        hit_rate = item.get("hitRate")
        miss_rate = item.get("missRate")
        samples = [f"check:{sample.get('checkId')}"
                   for sample in (item.get("sampleReviews") or [])
                   if sample.get("checkId") is not None]
        found.append(_candidate(
            rule="qc-threshold-hit-rate", stage=QC_STAGE, field=str(target["field"]),
            current=current, proposed=proposed,
            why=(f"`{code}`（{item.get('label')}）复核 {reviewed} 条："
                 f"判对 {item.get('hit')}／误报 {item.get('falseAlarm')}／漏报 {item.get('missed')}"
                 + (f"，命中率 {float(hit_rate):.0%}" if hit_rate is not None else "")
                 + (f"，漏报率 {float(miss_rate):.0%}" if miss_rate is not None else "")
                 + f" ⇒ {'放宽' if relax else '收紧'} `{target['field']}`"
                 f"（{current} → {proposed}，×1.25/×0.8 是**起手值**不是实测结论）"
                 + (f"；若还不对再看配合字段 `{target.get('aux')}`（本轮不动它）"
                    if target.get("aux") else "")),
            samples=reviewed, evidence=samples, min_samples=threshold,
            extra={"issueCode": code, "hitRate": hit_rate, "missRate": miss_rate,
                   "verdict": verdict, "direction": "relax" if relax else "tighten",
                   "alreadyOverridden": source != "default",
                   "auxField": target.get("aux")},
        ))

    if found:
        notes.append(f"命中率候选 {len(found)} 条（复核门槛 {threshold} 条；方向有据、幅度是起手值）")
    elif thin:
        notes.append("质检阈值暂无候选：" + "；".join(thin))
    else:
        notes.append("质检阈值的命中率与漏报率都在容忍范围内（或没有可映射的 code）")

    note = {
        "rule": "qc-threshold-hit-rate",
        "why": (f"已接入人工复核：复核 {summary.get('records', {}).get('reviewed', 0)} 条"
                f"／待复核 {summary.get('records', {}).get('unreviewed', 0)} 条 QC 记录；"
                f"代码门槛 {threshold} 条复核。本轮候选 {len(found)} 条 —— "
                f"候选只给方向与起手幅度，影子验证仍未做"),
    }
    return found, notes, note


def suggest(*, min_samples: int = MIN_SAMPLES) -> dict[str, Any]:
    """从跑次统计里给出**只读**的调参候选（设计文档 §4.3）。

    三条硬边界（改这里之前先读）：

    1. **本函数不写任何文件** —— 不碰 ``tuning.json``、不碰台账。候选要人核过用
       ``set_field`` 落盘（那一步要 ``basis`` + 可核 ``evidence``）。
    2. **样本不足只报不应用**（``gate.samples == "insufficient"``），
       并且 ``canApply`` 恒为 ``False`` —— 影子验证（历史基准回放）不在本模块，
       它没做完就给「可应用」等于把闸门②换成一个布尔值。
    3. **说不出的维度如实列进 ``notEvaluated``** —— 「没检查」不是「检查过没问题」。
    """
    index = _index()
    #: ⚠️ 没登记索引 ⇒ ``available=False`` ＋ 一句人话 ✓✗（**不是**"0 条候选 ⇒ 没问题" ✓✗）
    probe = ({"available": False, "groups": [],
              "hint": "跑次索引没接入（引擎可独立运行 ⇒ 由服务层启动时登记）"}
             if index is None else index.stats(group_by="stage"))
    failed, failed_notes = _failure_candidates(min_samples=min_samples, stats=probe)
    steps, steps_notes = _steps_candidates(min_samples=min_samples)
    #: ⚠️ 第三条**不跟** ``min_samples`` 走 ✓：它的分母是人工复核条数（``MIN_REVIEWS``），
    #:    两套门槛本来就不同源（30 条跑次 ≠ 30 次人工看片），强行共用一个数就是自欺 ✓✗。
    qc_found, qc_notes, qc_note = _qc_hit_rate_candidates()
    candidates = sorted(failed + steps + qc_found,
                        key=lambda c: (-(c.get("samples") or 0), c["id"]))
    #: ``available`` 的语义是「索引库读得出来」，不是「盘上有这个文件」——
    #: 空库（文件刚建、一条跑次都没写）也是 available=true，只是候选为空；
    #: 反过来库坏掉时文件可能还在，那时 available 必须是 false（否则 0 条会被当结论）
    available = bool(probe.get("available"))
    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minSamples": min_samples,
        "available": available,
        "db": index.db_path().as_posix() if index is not None else None,
        "candidates": candidates,
        "notes": failed_notes + steps_notes + qc_notes,
        #: 明确不做/做不了的四类：写出来是为了让下一个人知道"没结论"≠"没问题"
        "notEvaluated": [
            #: 这一条**本轮已实现** ✓（分母＝人工复核，见 ``qc_reviews``）—— 仍留在
            #: ``notEvaluated`` 里是因为它**每次都得如实报一句**：复核了几条、门槛差多少、
            #: 本轮出没出候选 ✓✗。删掉它就等于"实现了 = 检查过了"，而颗粒度、样本、
            #: 幅度三件事都还留着保留 ✓。
            qc_note or {"rule": "qc-threshold-hit-rate",
                        "why": "人工复核表不可用 ⇒ 命中率没有分母"},
            {"rule": "sampler-schedule-score",
             "why": ("换默认 sampler/schedule 等于换族语义，TUNABLES 里有意没有这两格；"
                     "且 runs.meta 未必带这两键 ⇒ 只报不猜")},
            {"rule": "shadow-validation",
             "why": "历史基准回放不在这里（`_bench` / 优化器那套），本模块只保证「候选 + 证据」可核"},
        ],
        "overrides": sorted(load()["overrides"]),
        "hint": ("候选只是候选：`set_field(stage, field, value, basis=…, evidence=[run_id…])` "
                 "才会真改行为，且写前自动备份、可 `clear_field` 回滚"),
    }


def _write(data: dict[str, Any]) -> None:
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        _backup_dir().mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        target = _backup_dir() / f"tuning.{stamp}.json"
        index = 1
        while target.exists():
            index += 1
            target = _backup_dir() / f"tuning.{stamp}-{index}.json"
        target.write_text(file.read_text(encoding="utf-8"), encoding="utf-8")
    #: 先写临时文件再替换：中途断电不会留下半个 JSON（读不出来就等于"没覆盖"，
    #: 症状是"调优忽然失效且没有任何提示"）
    temp = file.parent / (file.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(file)


def set_field(stage: str, name: str, value: Any, *, basis: str,
              evidence: Any = None, by: str = "human") -> dict[str, Any]:
    """写一条覆盖（**写前备份**）。证据核不过 / 未登记 / 越界 ⇒ 拒写。"""
    key = (str(stage or "").strip().lower(), str(name or "").strip().lower())
    spec = TUNABLES.get(key)
    if spec is None:
        return {"ok": False, "why": f"{key[0]}.{key[1]} 不在可调白名单里"
                                    f"（已登记：{sorted(f'{s}.{f}' for s, f in TUNABLES)}）"}
    coerced = _coerce(value, spec)
    if coerced is None:
        return {"ok": False, "why": f"值 {value!r} 非法：必须是 {spec.low:g}–{spec.high:g} 之间的"
                                    f"{'整数' if spec.integer else '数'}"}
    text = str(basis or "").strip()
    if not text:
        return {"ok": False, "why": "缺 basis：改了行为必须写清「为什么改」"}
    items = evidence if isinstance(evidence, list) else ([evidence] if evidence else [])
    items = [str(item).strip() for item in items if str(item).strip()]
    if not items:
        return {"ok": False, "why": "缺 evidence：没有凭据的调优就是拍脑袋"}
    #: ⚠️ 台账没登记 ⇒ **拒写** ✓✗（不是"没人核 ⇒ 放过" ✓✗）：这是护栏三的落点 ✓，
    #: 引擎单独跑起来时没有台账，就不该允许任何人拿字符串当凭据改行为 ✓。
    ledger = hooks.get("ledger")
    if ledger is None:
        return {"ok": False, "unavailable": True,
                "why": "知识台账没接入（引擎可独立运行 ⇒ 由服务层启动时登记）⇒ 凭据核不了，"
                       "按「核不过」处理"}
    checked = ledger.check_evidence(items)
    if not checked["ok"]:
        return {"ok": False, "why": f"证据核不过：{'、'.join(checked['bad'])}",
                "bad": checked["bad"]}

    data = load()
    previous = data["overrides"].get(_key(*key)) or {}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["overrides"][_key(*key)] = {
        "value": coerced, "default": spec.default, "basis": text,
        "evidence": items, "by": by, "updatedAt": stamp,
    }
    data.setdefault("history", []).append({
        "at": stamp, "key": _key(*key), "value": coerced,
        "previous": previous.get("value"), "basis": text, "by": by,
    })
    data["history"] = data["history"][-200:]
    data["updatedAt"] = stamp
    _write(data)
    return {"ok": True, "key": _key(*key), "value": coerced,
            "previous": previous.get("value"), "path": path().as_posix(),
            "hint": "只影响「调用方没显式给这一格」的请求；回执里会出现「参数来源: tuning:…」"}


def clear_field(stage: str, name: str) -> dict[str, Any]:
    """删一条覆盖（等价于回到默认；同样先备份）。"""
    key = _key(stage, name)
    data = load()
    if key not in data["overrides"]:
        return {"ok": False, "why": f"{key} 本来就没有覆盖"}
    previous = data["overrides"].pop(key)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    data.setdefault("history", []).append({
        "at": stamp, "key": key, "value": None,
        "previous": previous.get("value"), "basis": "cleared",
    })
    data["updatedAt"] = stamp
    _write(data)
    return {"ok": True, "key": key, "previous": previous.get("value"),
            "path": path().as_posix()}


def reset() -> dict[str, Any]:
    """删掉**全部**覆盖（回滚到纯默认）。"""
    data = load()
    count = len(data["overrides"])
    data["overrides"] = {}
    data["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(data)
    return {"ok": True, "cleared": count, "path": path().as_posix()}
