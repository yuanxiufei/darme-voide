"""Trace 持久化存储（**只移植回放侧**）—— 对齐 ``backend/src/utils/trace-store.ts``。

对齐 PenguinHarness 第 9 章「Trace 写入与回放（崩溃恢复）」：
trace 是 **append-only JSONL**，回放是**结构保真、非字节保真** —— 逐行 ``json.loads``，
跳过空白与残缺/损坏行，**不因单条坏记录整体失败**（这正是崩溃可回放的关键）。

**写侧三条约束**（``append_trace_event``，随 Agent/媒体域一起搬进来了）：

1. ``O_APPEND`` 打开 → **单次 write 写完** → 关闭：避免大 payload 被拆成多次 write 后进程崩溃留下残缺记录
2. **尾部愈合**：append 前探测末字节，非 ``\\n`` 则补 ``\\n``，防止残缺末行粘连下一条新记录
3. **串行化**：原 TS 用全局 promise 链保证多异步生产者不交错；Python 侧改用 ``threading.Lock``
   （FastAPI 的同步端点跑在线程池、媒体域又多线程回调 ⇒ 互斥量是这里正确的原语，
   而 ``asyncio.Lock`` 无法跨线程工作）

⇒ **绞杀期两个后端会同时往同一批 jsonl 追加**：两边都是 O_APPEND + 单次小 write，
POSIX/Windows 的追加写对小记录是原子的，因此可共存（顺序不保证，但不会互相踩坏）。
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from ..core.config import get_data_root

#: 连续下划线折叠成单个（原 TS 是两次 replace，第二次带 g 标志）
_MULTI_UNDERSCORE = re.compile(r"_{2,}")
_UNSAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9_-]")


def get_traces_dir() -> str:
    """数据根目录下的 traces 子目录（跟随数据目录切换）。"""
    return os.path.join(get_data_root(), "traces")


def safe_segment(s: str) -> str:
    """scope / traceId 可能含路径分隔符或 ``..`` —— 统一净化，防路径遍历。"""
    return _MULTI_UNDERSCORE.sub("_", _UNSAFE_SEGMENT.sub("_", s))


def trace_file_path(scope: str, trace_id: str) -> str:
    return os.path.join(get_traces_dir(), safe_segment(scope), f"{safe_segment(trace_id)}.jsonl")


def read_trace(scope: str, trace_id: str) -> list[dict[str, Any]]:
    """结构保真回放：逐行解析，跳过空白与残缺行。"""
    file_path = trace_file_path(scope, trace_id)
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    for line in content.split("\n"):
        text = line.strip()
        if not text:
            continue
        try:
            records.append(json.loads(text))
        except (ValueError, TypeError):
            # 残缺末行（进程崩溃写一半）：跳过，保持结构保真
            continue
    return records


def _collect_jsonl(directory: str) -> list[str]:
    out: list[str] = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        raise
    for entry in entries:
        if entry.is_dir():
            out.extend(_collect_jsonl(entry.path))
        elif entry.is_file() and entry.name.endswith(".jsonl"):
            out.append(entry.path)
    return out


def list_traces(scope: str | None = None) -> list[dict[str, Any]]:
    """列出所有（或某 scope 下）的 trace 元信息，按修改时间倒序。"""
    traces_dir = get_traces_dir()
    base = os.path.join(traces_dir, safe_segment(scope)) if scope else traces_dir
    try:
        files = _collect_jsonl(base)
    except OSError:
        return []

    metas: list[dict[str, Any]] = []
    for file_path in files:
        try:
            stat = os.stat(file_path)
            rel = os.path.relpath(file_path, traces_dir)
            parts = rel.split(os.sep)
            trace_id = os.path.basename(file_path)
            if trace_id.endswith(".jsonl"):
                trace_id = trace_id[:-6]
            # 原 TS 用 '/' 拼接（不是 path.sep）
            scope_name = "/".join(parts[:-1]) if len(parts) > 1 else ""
            records = read_trace(scope_name, trace_id)
            metas.append(
                {
                    "scope": scope_name,
                    "traceId": trace_id,
                    "file": rel,
                    "size": stat.st_size,
                    # JS 的 mtimeMs 是毫秒（且带小数）
                    "mtimeMs": stat.st_mtime * 1000,
                    "count": len(records),
                    "firstTs": records[0].get("ts") if records else None,
                    "lastTs": records[-1].get("ts") if records else None,
                }
            )
        except OSError:
            continue  # 忽略单文件错误

    metas.sort(key=lambda m: m["mtimeMs"], reverse=True)
    return metas


# ---------------------------------------------------------------------------
# 写侧（Agent / 媒体域用）
# ---------------------------------------------------------------------------

#: 全局串行化写入锁：保证 append 顺序稳定、不交错（对齐原 TS 的全局 promise 链）
_write_lock = threading.Lock()


def append_trace_event(record: dict[str, Any]) -> None:
    """追加一条 trace 记录（**不抛错**：trace 落盘失败不该影响主流程）。

    ⚠️ 与读侧一样只做「结构保真」：写的是紧凑 JSON + ``\\n``，规模控制在单次 write。
    """
    try:
        file_path = trace_file_path(record.get("scope", ""), record.get("traceId", ""))
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _write_lock:
            _append_json_line(file_path, line)
    except Exception:  # noqa: BLE001 - 与原 TS 的 .catch(console.warn) 同义
        pass


def _append_json_line(file_path: str, line: str) -> None:
    """``O_APPEND`` 打开 → 尾部愈合 → 单次 write → 关闭。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 尾部愈合：探测末字节，非换行则补换行
    heal = ""
    try:
        if os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    heal = "\n"
    except OSError:
        pass  # 文件尚不存在，无需愈合

    with open(file_path, "ab") as fh:  # O_APPEND
        fh.write((heal + line).encode("utf-8"))
        fh.flush()
