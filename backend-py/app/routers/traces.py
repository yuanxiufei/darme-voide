"""``/api/v1/traces`` —— 与 ``backend/src/routes/traces.ts`` 对齐（3 端点）。

* ``GET /stats``：按 scope 聚合所有 Agent 调用的 token 用量
* ``GET /?scope=``：列出 trace 元信息（按修改时间倒序）
* ``GET /{scope}/{traceId}``：回放单个 trace 的完整事件序列（结构保真）

数据来自数据根目录下的 ``traces/**.jsonl``（**文件，不是数据库**），走
``services/trace_store.py``。**写入侧仍在 Node** —— Python 只读，无锁竞争风险。

⚠️ 路由顺序：``/stats`` 必须先于 ``/{scope}/{traceId}`` 注册（虽然在 FastAPI 里
段数不同不会真冲突，但保持这个顺序更稳，也便于日后加一层路径时不踩坑）。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..core.response import bad_request, success
from ..services.trace_store import list_traces, read_trace

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


def _meta_number(meta: dict[str, Any], camel: str, snake: str) -> float:
    """对齐 ``Number(m.inputTokens ?? m.input_tokens ?? 0)``（空值合并链）。

    已知有意偏差：JS 对非数值得到 ``NaN``；这里也照算 ``NaN``，但**出口归 0**。
    原因：``JSON.stringify(NaN)`` 是 ``null``，而 Python 的 ``json.dumps(NaN)`` 会输出
    非法 JSON 的 ``NaN`` 字面量 —— 两者都不好看，归 0 对展示更稳。
    真实数据里这几个字段恒为数值（由 Node 侧写入）。
    """
    value = meta.get(camel)
    if value is None:
        value = meta.get(snake)
    if value is None:
        value = 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _js_number_falsy(x: float) -> bool:
    """JS 里数字的真值判断：``0`` 与 ``NaN`` 为假，其余为真。"""
    return x == 0 or x != x


def _clean(x: float) -> float:
    """出口把 NaN 归 0（见 ``_meta_number`` 的说明）。"""
    return 0 if x != x else x


@router.get("/stats")
def trace_stats():
    metas = list_traces()
    by_scope: dict[str, dict[str, float]] = {}
    total_input = 0.0
    total_output = 0.0
    total_tokens = 0.0
    total_runs = 0

    for meta in metas:
        records = read_trace(meta.get("scope", ""), meta.get("traceId", ""))
        best: dict[str, float] | None = None
        for record in records:
            m = record.get("meta")
            if not isinstance(m, dict):
                continue
            input_t = _meta_number(m, "inputTokens", "input_tokens")
            output_t = _meta_number(m, "outputTokens", "output_tokens")
            total_t = _meta_number(m, "totalTokens", "total_tokens")
            # TS: `Number(m.totalTokens ?? m.total_tokens ?? input + output)`
            if m.get("totalTokens") is None and m.get("total_tokens") is None:
                total_t = input_t + output_t
            # TS: if (input || output || total) —— 三者全为假才跳过
            if not (
                _js_number_falsy(input_t)
                and _js_number_falsy(output_t)
                and _js_number_falsy(total_t)
            ):
                if best is None or total_t > best["totalTokens"]:
                    best = {
                        "inputTokens": input_t,
                        "outputTokens": output_t,
                        "totalTokens": total_t,
                    }
        if best is None:
            continue
        scope = meta.get("scope") or "unknown"
        cur = by_scope.setdefault(
            scope, {"runs": 0, "inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        )
        cur["runs"] += 1
        cur["inputTokens"] += best["inputTokens"]
        cur["outputTokens"] += best["outputTokens"]
        cur["totalTokens"] += best["totalTokens"]
        total_input += best["inputTokens"]
        total_output += best["outputTokens"]
        total_tokens += best["totalTokens"]
        total_runs += 1

    scopes = [
        {
            "scope": scope,
            "runs": int(v["runs"]),
            "inputTokens": _clean(v["inputTokens"]),
            "outputTokens": _clean(v["outputTokens"]),
            "totalTokens": _clean(v["totalTokens"]),
        }
        for scope, v in sorted(
            by_scope.items(), key=lambda kv: _clean(kv[1]["totalTokens"]), reverse=True
        )
    ]

    return success(
        {
            "totalInputTokens": _clean(total_input),
            "totalOutputTokens": _clean(total_output),
            "totalTokens": _clean(total_tokens),
            "runs": total_runs,
            "byScope": scopes,
        }
    )


@router.get("")
def list_all(request: Request):
    scope = request.query_params.get("scope")
    return success(list_traces(scope or None))


@router.get("/{scope}/{trace_id}")
def replay(scope: str, trace_id: str):
    if not scope or not trace_id:
        return bad_request("scope and traceId are required")
    records = read_trace(scope, trace_id)
    if not records:
        return success({"scope": scope, "traceId": trace_id, "records": []})
    return success({"scope": scope, "traceId": trace_id, "count": len(records), "records": records})
