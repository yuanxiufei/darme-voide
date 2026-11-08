"""**画布计划草稿仓** —— 落盘草案，确认后才交出可执行 op。

## 它解决什么

前端画布状态只在浏览器里（localStorage），后端是纯函数。这意味着「草稿」不能靠后端
持有画布事务 —— 后端能持有的只有**一份被冻结的计划**：

::

    模型/用户产出计划 ──▶ 落盘为 draft（此时 ops 存在但**不下发**）
                              │
                              ▼
                      用户看一眼 preflight：要建 3 个、连 2 条、跑 1 次
                              │
                    ┌─────────┴─────────┐
                    │ 确认               │ 不要了
                    ▼                   ▼
            重新编译 → 下发 ops      草案就一直躺着（可查、可丢）

## 为什么确认时才重新编译

因为 ``snapshot`` 可能变了（用户在草案期间动了画布）。确认时用**当下**的快照重编译，
编译器会重新做交叉校验 —— 于是「计划里那条线要连的节点被删了」这类问题在确认这一步
被挡住，而不是落地后留下一根断线 ✗。

## 幂等

计划 id = ``canvas_plan.plan_digest``（计划内容的指纹）。同一份计划重复提交得到**同一个 id**，
覆盖的是同一份草案，不会长出第二条记录 ✓。重复确认返回**上一次缓存下来的 ops**，
并标 ``replayed: true`` —— 前端据此不会把同一批节点建两遍 ✓。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.config import get_data_root

__all__ = [
    "PLAN_ID_PATTERN",
    "PlanNotFound",
    "list_plans",
    "load_plan",
    "plans_dir",
    "save_plan",
]

#: 计划 id = 16 位十六进制指纹 ✓。白名单式校验，杜绝路径穿越 ✗
PLAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")

STATUS_DRAFT = "draft"
STATUS_CONFIRMED = "confirmed"


class PlanNotFound(KeyError):
    """计划不存在 / id 形状不对 ✓。两者对外都答「没有这份计划」，不泄露目录结构。"""


def plans_dir() -> Path:
    """草案落点 ✓（在数据根下，不在源码树里）。"""
    return Path(get_data_root()) / "canvas" / "plans"


def _path_of(plan_id: str) -> Path:
    if not PLAN_ID_PATTERN.match(str(plan_id or "")):
        raise PlanNotFound(plan_id)
    return plans_dir() / f"{plan_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_plan(plan_id: str) -> dict[str, Any]:
    """读一份草案 ✓；不存在抛 ``PlanNotFound``。"""
    path = _path_of(plan_id)
    if not path.exists():
        raise PlanNotFound(plan_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PlanNotFound(plan_id) from error


def save_plan(record: dict[str, Any]) -> dict[str, Any]:
    """整条记录原子落盘（临时文件 + ``replace``）✓。

    直接 ``write_text`` 到目标文件的话，进程在写到一半被杀就会留下半个 JSON，
    下次读它报的是「计划不存在」—— 用户以为草案丢了，其实只是写崩了 ✗。
    """
    directory = plans_dir()
    directory.mkdir(parents=True, exist_ok=True)
    plan_id = str(record.get("id") or "")
    path = _path_of(plan_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)
    return record


def draft_record(
    plan: dict[str, Any],
    compiled: dict[str, Any],
    *,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """组一条 draft 记录 ✓ —— 此时 ``ops`` 存着但**不下发**（下发在 confirm）。"""
    from . import canvas_plan  # 局部导入：避免与 canvas_plan 的循环依赖

    stamp = _now()
    return {
        "id": canvas_plan.plan_digest(plan),
        "kind": compiled.get("kind"),
        "goal": compiled.get("goal"),
        "status": STATUS_DRAFT,
        "createdAt": stamp,
        "updatedAt": stamp,
        "confirmedAt": None,
        "confirmCount": 0,
        "expectedRevision": expected_revision,
        "plan": plan,
        "ops": compiled.get("ops") or [],
        "refs": compiled.get("refs") or {},
        "preflight": compiled.get("preflight") or {},
        "layout": compiled.get("layout") or {},
        "warnings": compiled.get("warnings") or [],
    }


def public_view(record: dict[str, Any], *, with_ops: bool = False) -> dict[str, Any]:
    """对外视图 ✓ —— draft 阶段**默认不含 ops**。

    这不是藏东西：草案阶段下发 ops，前端一时手快就落了，等于没有确认环节 ✗。
    要看「将要发生什么」，``preflight`` 的数字比一串 op 更直接。
    """
    view = {
        "planId": record.get("id"),
        "kind": record.get("kind"),
        "goal": record.get("goal"),
        "status": record.get("status"),
        "createdAt": record.get("createdAt"),
        "updatedAt": record.get("updatedAt"),
        "confirmedAt": record.get("confirmedAt"),
        "confirmCount": record.get("confirmCount", 0),
        "expectedRevision": record.get("expectedRevision"),
        "preflight": record.get("preflight") or {},
        "layout": record.get("layout") or {},
        "warnings": record.get("warnings") or [],
        "requiresConfirmation": record.get("status") == STATUS_DRAFT,
    }
    if with_ops:
        view["ops"] = record.get("ops") or []
        view["refs"] = record.get("refs") or {}
    return view


def list_plans(limit: int = 20) -> list[dict[str, Any]]:
    """最近的草案（新的在前）✓；坏文件直接跳过，不让一条烂记录挡住列表。"""
    directory = plans_dir()
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return [public_view(record) for record in records[: max(1, limit)]]
