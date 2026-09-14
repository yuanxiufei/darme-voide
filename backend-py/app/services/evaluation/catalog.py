"""评测基准 case 目录（单一来源）—— 与 ``evaluation/catalog.ts``（49 行）对齐。

* ``AGENT_BY_KIND``：``case.kind`` → Agent 类型（CLI 与 HTTP 路由共用）；
* ``list_benchmark_cases`` / ``load_case_by_id``：运行时发现与按 id 加载（供 HTTP 路由使用）。

⚠️ **基准目录位置**：TS 是从模块位置推 ``backend/benchmarks``（与 cwd 解耦，兼容 Docker
``WORKDIR /app``）。**S7 已把 4 个 case JSON 搬到 ``PROJECT_ROOT/benchmarks``**（逐字节一致），
默认值随之改到新位置，仍可用 ``BENCHMARKS_DIR`` 环境变量覆盖 —— 这样删掉 ``backend/`` 之后
本域不再依赖它。history 目录同源（见 ``optimizer`` 的 ``historyDir`` 默认值）。

⚠️ 两处**有意差异**：

1. ``readdirSync`` 的顺序是文件系统顺序；这里**按文件名排序**（顺序决定「同 id 时谁胜出」，
   也决定列表展示顺序 —— 确定性更好，代价是极端情况下与 Node 的取胜者不同）；
2. ``json.loads`` 失败**会抛**（与 TS 的 ``JSON.parse`` 一致）：``GET /evaluation/cases``
   在原 TS 里没有 try ⇒ 坏 json 会变成 500，这里同样不在目录层兜底。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ...config import PROJECT_ROOT
from .types import AGENT_BY_KIND

__all__ = [
    "AGENT_BY_KIND",
    "benchmarks_dir",
    "list_benchmark_cases",
    "load_case_by_id",
    "scan_case_files",
]


def benchmarks_dir() -> Path:
    """基准目录：``BENCHMARKS_DIR`` 优先，否则 ``<项目根>/benchmarks``。

    ⚠️ 项目根是「与 cwd 解耦」的写法（Docker 里 cwd 可能是 ``/app``）；
    S7 起 case JSON 就在这一层，**不再指向 ``backend/benchmarks``**。
    """
    override = os.environ.get("BENCHMARKS_DIR")
    if override:
        return Path(override)
    return PROJECT_ROOT / "benchmarks"


def scan_case_files() -> list[dict[str, str]]:
    """扫描目录里的 ``*.json``，返回 ``{id, kind, agentType, file}``（**只留已知 kind**）。"""
    directory = benchmarks_dir()
    if not directory.exists():
        return []
    cases: list[dict[str, str]] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        payload = json.loads(entry.read_text(encoding="utf-8"))
        kind = payload.get("kind") if isinstance(payload, dict) else None
        agent_type = AGENT_BY_KIND.get(kind or "", "")
        if not agent_type:
            continue
        cases.append({
            "id": payload.get("id"),
            "kind": kind,
            "agentType": agent_type,
            "file": str(entry),
        })
    return cases


def list_benchmark_cases() -> list[dict[str, Any]]:
    """列出全部基准 case 的元信息（供 ``GET /evaluation/cases`` 展示）。"""
    return [
        {"id": case["id"], "kind": case["kind"], "agentType": case["agentType"]}
        for case in scan_case_files()
    ]


def load_case_by_id(case_id: Any) -> dict[str, Any] | None:
    """按 case id 加载**完整** case（全目录扫描；case 文件极少，无性能顾虑）。"""
    hit = next((case for case in scan_case_files() if case["id"] == case_id), None)
    if hit is None:
        return None
    return json.loads(Path(hit["file"]).read_text(encoding="utf-8"))
