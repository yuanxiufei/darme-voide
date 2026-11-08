"""修复域的**闭环**：诊断快照落盘 ＋「应用之后那个问题还在不在」的判定（知识闭环缺口 9 ✓）。

缺口原文
========
``repair`` 提案的生命周期只有 ``pending → applied → rolled_back`` ✓ —— 它回答的是「这份
改动加上去了没有」，**不回答**「加完之后那个问题消失了没有」✗。于是修复域是全仓唯一一个
「有产出、没结论」的环节：``apply`` 回成功，谁也不知道那条工作流是不是真能跑了；而
``diagnose`` 每次现算现回、算完即丢 ✗，连「这一周缺口是在变少还是变多」都答不了。

三件东西补齐它
==============
1. **快照**（:func:`build_snapshot` / :func:`save_snapshot`）：一次诊断的**缺口键集合**
   落成 ``<dataRoot>/repair/diagnose/<id>.json``，只留最近 ``KEEP_SNAPSHOTS`` 份 ✓。
   ⚠️ 只存键与计数，**不存**全盘模型清单 —— 那是上千条、每次几 MB，存下来是日志不是证据。
2. **缺口键**（:func:`finding_key`）：``<kind>:<小写文件名>`` ✓。为什么不比条数：新增一条
   无关缺口会让 ``counts.missing`` 变大，但那和「我刚修掉的这条」没关系 ✗；键集合才能
   精确回答「这一条还在不在」✓。
3. **判定**（:func:`verdict` / :func:`verify`）：应用后重新诊断，看那条键还在不在 ✓。

四档结论，宁可说不知道
======================
| 结论 | 含义 |
|---|---|
| ``resolved`` | 应用后复诊，这条键不在缺口里了 ⇒ 真修好了 ✓ |
| ``still`` | 键还在 ⇒ 改动加上了但问题没走 ✗（提案得重做，或根本不是这个原因） |
| ``stale`` | 提案落单时这条键就不在诊断里 ⇒ 键抄错了／已被别人修掉，**不能算成功** ✗ |
| ``unverified`` | 复诊本身没跑成（读不到工作流目录／明细被截断／提案没挂键）✓ 如实说不知道 |

第五档 ``rolled_back`` 不是判定结果、是**终态** ✓：改动被撤销了，应用期的结论不再代表
盘面现状 ⇒ 旧跑次按 ``partial`` 覆盖（净收益为零），别留着一条 ``ok`` 让成功率虚高 ✗。

⚠️ **截断会造成假「消失」**：``repair_diagnose`` 的 ``missing`` 明细封顶 40 条，缺口多于
40 时快照里只有前 40 个键 ⇒「本次诊断被截断 ＋ 键不在集合里」一律判 ``unverified``，
不判 ``resolved`` ✓ —— 缺席可能只是没列出来，不是修好了。

与知识闭环的接口
================
判定结果按 ``kind="repair"`` 写进 ``outcome_index`` 跑次表 ✓（这一档 ``KINDS`` 里一直有、
从来没有写入方）。``run_id`` 固定 ``repair-<proposalId>`` ⇒ 同一份提案复诊几次都只算一条，
统计不因复诊次数虚增 ✓；``resolved→ok``／``still→failed``／``stale→partial``，
``unverified`` **不写**（没有事实可记，写进去只会稀释分母 ✗）。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from . import outcome_index

__all__ = [
    "KEEP_SNAPSHOTS",
    "RESULTS",
    "RESULT_LABELS",
    "build_snapshot",
    "finding_key",
    "findings_of",
    "fingerprint",
    "history",
    "latest",
    "latest_keys",
    "normalize_finding",
    "record",
    "rolled_back",
    "save_snapshot",
    "verdict",
    "verify",
]

#: 快照保留份数 ✓ —— 一次诊断一份，按天跑一个月也才 30 份；60 够看趋势又不占地方
KEEP_SNAPSHOTS = 60

#: 四档结论（见模块头表格）✓。键是落库值，值是给人看的话。
RESULT_LABELS: dict[str, str] = {
    "resolved": "问题已消失（复诊时这条缺口不在了）",
    "still": "问题还在（改动应用了，但缺口没消失）",
    "stale": "对不上（提案落单时这条缺口就不在诊断里）",
    "unverified": "没法判定（复诊没跑成或没挂缺口键）",
    "rolled_back": "改动已回滚（应用期结论作废）",
}
RESULTS: tuple[str, ...] = tuple(RESULT_LABELS)

#: 判缺口键时认的两种"问题" ✓ —— 与 ``repair_diagnose`` 的两个明细列表一一对应
FINDING_KINDS: tuple[str, ...] = ("missing", "ambiguous")

#: 报告里最多带几条「还在」的证据（别把 40 条塞回前端）
MAX_EVIDENCE = 5


def _iso_now() -> str:
    """本地时间 ``YYYY-MM-DDTHH:MM:SS`` ✓（与仓内其它落盘时间同格式，不带时区后缀）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _stamp(at: str) -> str:
    """把 ``at`` 压成文件名里能用的紧凑串 ✓（``2026-11-07T15:30:00`` → ``20261107T153000``）。"""
    return "".join(char for char in str(at) if char.isalnum())


# ---------------------------------------------------------------- 缺口键


def finding_key(kind: Any, file: Any) -> str:
    """一条缺口的**稳定键** ✓：``<kind>:<小写文件名>``。

    ⚠️ 只认 ``missing``／``ambiguous`` 两档；别的 kind 一律拒（返回空串由调用方判），
    免得「键写错了但看起来像对的」混进快照 —— 那会让 :func:`verdict` 永远判 ``stale`` ✗。
    """
    key_kind = str(kind or "").strip().lower()
    name = str(file or "").strip().replace("\\", "/").split("/")[-1].lower()
    if key_kind not in FINDING_KINDS or not name:
        return ""
    return f"{key_kind}:{name}"


def normalize_finding(raw: Any) -> str:
    """校验并归一调用方给的 ``finding`` ✓（``"missing:xxx.SafeTensors"`` → 小写形）。

    收三种写法：``"missing:xxx"``／``{"kind": "missing", "file": "xxx"}``／``"xxx"``（按
    ``missing`` 解释 —— 缺文件是最常见的修复对象）。归一不了返回空串，**不抛**：
    调用方拿到空串就当「没挂缺口」，比 500 好。
    """
    if isinstance(raw, dict):
        return finding_key(raw.get("kind") or "missing", raw.get("file") or raw.get("name"))
    text = str(raw or "").strip()
    if not text:
        return ""
    if ":" in text:
        kind, _, name = text.partition(":")
        return finding_key(kind, name)
    return finding_key("missing", text)


def findings_of(result: dict[str, Any]) -> list[dict[str, Any]]:
    """从一次 ``repair_diagnose.diagnose()`` 的结果里抽**可追踪的缺口** ✓。

    ``ambiguous``（同名多份）也算一条：它同样堵着工作流（不知道该用哪份），修法通常是
    删掉多余副本 ⇒ 复诊时这条键消失就该判 ``resolved`` ✓。
    """
    rows: list[dict[str, Any]] = []
    for kind in FINDING_KINDS:
        for item in result.get(kind) or []:
            if not isinstance(item, dict):
                continue
            key = finding_key(kind, item.get("file"))
            if not key:
                continue
            rows.append({
                "key": key,
                "kind": kind,
                "file": item.get("file"),
                "usedBy": [str(name) for name in (item.get("usedBy") or [])][:MAX_EVIDENCE],
                "nodes": [str(name) for name in (item.get("nodes") or [])][:MAX_EVIDENCE],
            })
    rows.sort(key=lambda item: item["key"])
    return rows


def fingerprint(findings: Iterable[dict[str, Any]]) -> str:
    """缺口集合的**指纹** ✓ —— 键排序后取 sha256 前 16 位（大写）。

    两个用途：① 两次诊断「是不是同一批问题」一眼可比；② 快照文件名与内容都能对账，
    手改过 json 的人会被指纹对不上抓出来 ⚠️。空集合也有指纹（不是空串），
    「零缺口」本身是一个有意义的状态 ✓。
    """
    keys = sorted(str(item.get("key") or "") for item in findings if item.get("key"))
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return digest[:16].upper()


# ---------------------------------------------------------------- 快照


def build_snapshot(result: dict[str, Any], *, at: str | None = None,
                   snapshot_id: str | None = None) -> dict[str, Any]:
    """把一次诊断结果收成**可落盘的快照** ✓（不写盘，纯函数）。

    ⚠️ ``truncated`` 必须原样带上：:func:`verdict` 靠它区分「真消失」与「没列出来」✗。
    """
    findings = findings_of(result)
    moment = at or _iso_now()
    return {
        "id": snapshot_id or f"d-{_stamp(moment)}-{uuid.uuid4().hex[:4]}",
        "at": moment,
        "fingerprint": fingerprint(findings),
        "counts": dict(result.get("counts") or {}),
        "workflowsDir": result.get("workflowsDir"),
        "truncated": bool(result.get("truncated")),
        "findings": findings,
        "keys": [item["key"] for item in findings],
    }


def save_snapshot(snapshot: dict[str, Any], directory: Path | str,
                  *, keep: int = KEEP_SNAPSHOTS) -> Path:
    """快照落盘并**只留最近 ``keep`` 份** ✓（按 mtime 淘汰最旧的）。

    ⚠️ 淘汰只删本目录里的 ``*.json``：这个目录归本模块独占，别把别的东西挂进来。
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{snapshot['id']}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")

    files = sorted(folder.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for stale in files[max(1, int(keep)):]:
        try:
            stale.unlink()
        except OSError:
            continue
    return path


def history(directory: Path | str, *, limit: int = 20) -> list[dict[str, Any]]:
    """快照**目录**（新→旧）✓ —— 只回摘要，不回 40 条明细（前端要的是趋势，不是全表）。"""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    rows: list[dict[str, Any]] = []
    for path in files[:max(1, int(limit))]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "id": data.get("id") or path.stem,
            "at": data.get("at"),
            "fingerprint": data.get("fingerprint"),
            "counts": data.get("counts") or {},
            "findings": len(data.get("keys") or data.get("findings") or []),
            "truncated": bool(data.get("truncated")),
        })
    return rows


def latest(directory: Path | str) -> dict[str, Any] | None:
    """最新一份快照的**全文** ✓（没有 / 全坏 ⇒ ``None``，不编空壳）。"""
    folder = Path(directory)
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("id"):
            return data
    return None


def latest_keys(directory: Path | str) -> tuple[set[str], bool] | None:
    """最新快照的**键集合 + 是否被截断** ✓；没有快照返回 ``None``。

    ⚠️ 截断标志要一起交出去：调用方据此决定「提案落单时这条缺口到底在不在」能不能判。
    """
    snapshot = latest(directory)
    if snapshot is None:
        return None
    keys = {str(key) for key in (snapshot.get("keys") or []) if key}
    return keys, bool(snapshot.get("truncated"))


# ---------------------------------------------------------------- 判定


def verdict(*, finding: Any, was_present: bool | None, after_keys: Iterable[str],
            after_truncated: bool = False,
            evidence: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """**应用之后，这条缺口还在不在** ✓（纯函数，四档见模块头）。

    ``was_present`` 是提案落单那一刻的事实（``None`` ＝ 当时没有可用快照 ⇒ 判不了）✗；
    ``after_keys`` 是应用后复诊的键集合。判定顺序不可换：先问「有没有可比对象」，
    再问「当时在不在」，最后才问「现在还在不在」—— 顺序换了会把 ``stale`` 报成 ``resolved`` ✗。
    """
    key = normalize_finding(finding)
    if not key:
        return {"status": "unverified", "finding": "", "reason": "提案没有挂缺口键（finding），没有可比的对象"}
    if was_present is None:
        return {"status": "unverified", "finding": key,
                "reason": "提案落单时没有可用的诊断快照（或那份快照被截断），无从比较"}
    if not was_present:
        return {"status": "stale", "finding": key,
                "reason": "提案落单时这条缺口就不在诊断里 —— 键可能抄错，或已被别处修掉"}
    current = {str(item) for item in after_keys}
    if key in current:
        return {"status": "still", "finding": key, "reason": "复诊时这条缺口仍在",
                "evidence": list(evidence)[:MAX_EVIDENCE]}
    if after_truncated:
        return {"status": "unverified", "finding": key,
                "reason": "本次诊断的缺口明细被截断，缺席可能只是没列出来，不判成功"}
    return {"status": "resolved", "finding": key,
            "reason": "复诊时这条缺口已不在诊断里"}


def verify(*, finding: Any, was_present: bool | None,
           diagnose_fn: Callable[[], dict[str, Any]], directory: Path | str,
           keep: int = KEEP_SNAPSHOTS) -> dict[str, Any]:
    """应用后**重新诊断一次**并给出结论 ✓（顺带落一份快照，不额外收参数）。

    复诊失败不抛：``diagnose`` 读不到工作流目录、扫描 IO 失败都会变成 ``unverified`` ＋
    ``reason`` —— 修复域的动作已经落盘了，不能因为「复诊没跑成」把已经成功的事报成失败 ✗。
    """
    started_ns = time.time_ns()
    try:
        result = diagnose_fn()
    except Exception as err:  # noqa: BLE001 —— 盘面扫描可能 IO 失败，如实报而不是 500
        return {"status": "unverified", "finding": normalize_finding(finding),
                "reason": f"复诊本身失败：{type(err).__name__}: {err}",
                "checkedAt": _iso_now(), "saved": False}
    if not result.get("ok"):
        return {"status": "unverified", "finding": normalize_finding(finding),
                "reason": f"复诊失败：{result.get('error') or '诊断没给出结果'}",
                "checkedAt": _iso_now(), "saved": False}

    snapshot = build_snapshot(result)
    saved = True
    try:
        save_snapshot(snapshot, directory, keep=keep)
    except OSError:
        #: 快照存不下不影响判定本身 ⇒ 照样算，只在回执里说实话
        saved = False
    outcome = verdict(
        finding=finding,
        was_present=was_present,
        after_keys=snapshot["keys"],
        after_truncated=snapshot["truncated"],
        evidence=[item for item in snapshot["findings"] if item["key"] == normalize_finding(finding)],
    )
    outcome.update({
        "snapshotId": snapshot["id"],
        "fingerprint": snapshot["fingerprint"],
        "counts": snapshot["counts"],
        "checkedAt": snapshot["at"],
        "startedNs": started_ns,
        "elapsedMs": (time.time_ns() - started_ns) // 1_000_000,
        "saved": saved,
    })
    return outcome


def rolled_back(finding: Any, *, previous: Any = None, at: str | None = None) -> dict[str, Any]:
    """提案**回滚**时把判定改成终态 ✓（旧结论作废，跑次按 ``partial`` 覆盖）。

    ⚠️ 回滚后不重新诊断：回滚的语义是「把这次改动撤掉」，此刻盘面回到应用前，
    再诊断一次得到的是**应用前**那个状态，拿它当「修复失败」的证据是误判 ✗。
    """
    return {
        "status": "rolled_back",
        "finding": normalize_finding(finding),
        "reason": "提案已回滚：改动被撤销，应用期的判定不再代表盘面现状",
        "previous": str(previous) if previous else None,
        "checkedAt": at or _iso_now(),
        "saved": False,
    }


# ---------------------------------------------------------------- 记入跑次


#: 结论 → 跑次状态 ✓（``unverified`` 不在表里：没有事实可记，写进去只会稀释分母 ✗）
_OUTCOME_STATUS: dict[str, str] = {
    "resolved": "ok", "still": "failed", "stale": "partial", "rolled_back": "partial",
}


def record(proposal_id: Any, verification: dict[str, Any], *, findings_dir: Path | str | None = None) -> bool:
    """把一次判定记进 ``outcome_index``（``kind="repair"``）✓；返回是否写成功。

    ``run_id`` 固定 ``repair-<proposalId>`` ⇒ 复诊覆盖同一条，统计不会因复诊次数虚增 ✓。
    ``findings_dir`` 只用于把「当时前后各有哪些缺口」写进 meta 方便回查，不影响判定。
    """
    status = _OUTCOME_STATUS.get(str(verification.get("status") or ""))
    pid = str(proposal_id or "").strip()
    if not status or not pid:
        return False
    snapshot = latest(findings_dir) if findings_dir is not None else None
    return outcome_index.record_run({
        "run_id": f"repair-{pid}",
        "kind": "repair",
        "stage": "repair",
        "status": status,
        "scope": verification.get("finding") or None,
        "elapsed_ms": verification.get("elapsedMs"),
        "started_ns": verification.get("startedNs"),
        "ended_ns": time.time_ns(),
        "meta": {
            "verdict": verification.get("status"),
            "reason": verification.get("reason"),
            "finding": verification.get("finding"),
            "snapshotId": verification.get("snapshotId"),
            "fingerprint": verification.get("fingerprint") or (snapshot or {}).get("fingerprint"),
            "counts": verification.get("counts") or (snapshot or {}).get("counts"),
        },
    })
