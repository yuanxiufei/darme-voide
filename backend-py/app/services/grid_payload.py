"""宫格提示词载荷的抽取/归一（**单一真源**）。

原先这三个纯函数长在 ``routers/grid.py`` 里（``_extract_json_candidate`` /
``_normalize_grid_payload`` / ``_find_grid_payload``）。2026-11-07 评测面要评
「宫格提示词」这一类产出（知识闭环缺口 8），它必须用**和生产完全同一套**抽取口径 ——
否则评测会对着"评测自己定义的格式"打分，而线上走的是另一套。故上收为 service，
两端各自 import（router 保留原名别名，既有测试与内部调用零改动）。

口径（逐字沿用原实现，勿"优化"）：

* ``grid_prompt`` 必须有非空字符串，否则整条载荷作废（返回 ``None``）；
* ``cell_prompts`` 的键可以是 camelCase（``shotNumber`` / ``frameType``）；
* 空 ``prompt`` 的格子**直接丢弃**（不是留空占位）；
* 递归下降：dict 值、list 元素、字符串（先 JSON.parse，失败再抠 ```` ```json ```` 围栏或首个 ``{...}``）。
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract_json_candidate", "find_grid_payload", "normalize_grid_payload"]


def extract_json_candidate(text: str) -> str:
    """从模型正文里抠 JSON：优先 ```` ```json ```` 围栏，其次第一个 ``{...}``。"""
    fenced = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced and fenced.group(1):
        return fenced.group(1).strip()
    plain = re.search(r"\{[\s\S]*\}", text)
    return plain.group(0).strip() if plain else ""


def normalize_grid_payload(payload: Any) -> dict[str, Any] | None:
    """把「可能带 camelCase / 缺字段」的载荷归一成 ``{grid_prompt, cell_prompts}``。"""
    if not isinstance(payload, dict):
        return None

    raw_grid = payload.get("grid_prompt")
    if not isinstance(raw_grid, str):
        raw_grid = payload.get("gridPrompt")
    grid_prompt = raw_grid.strip() if isinstance(raw_grid, str) else ""

    raw_cells = payload.get("cell_prompts")
    if not isinstance(raw_cells, list):
        raw_cells = payload.get("cellPrompts")
    raw_cells = raw_cells if isinstance(raw_cells, list) else []

    cell_prompts = []
    for cell in raw_cells:
        cell = cell if isinstance(cell, dict) else {}
        # `Number(x ?? y ?? 0) || 0` —— NaN / 0 都归 0
        number = cell.get("shot_number")
        if number is None:
            number = cell.get("shotNumber")
        try:
            shot_number = int(float(number)) if number is not None else 0
        except (TypeError, ValueError):
            shot_number = 0
        frame_type = cell.get("frame_type")
        if frame_type is None:
            frame_type = cell.get("frameType")
        prompt = str(cell.get("prompt") or "").strip()
        entry = {
            "shot_number": shot_number,
            "frame_type": str(frame_type if frame_type is not None else "first_frame"),
            "prompt": prompt,
        }
        if entry["prompt"]:
            cell_prompts.append(entry)

    if not grid_prompt:
        return None
    return {"grid_prompt": grid_prompt, "cell_prompts": cell_prompts}


def find_grid_payload(value: Any) -> dict[str, Any] | None:
    """递归找宫格载荷（字符串会先尝试 JSON 解析 / 抠 JSON）。"""
    if value is None:
        return None

    normalized = normalize_grid_payload(value)
    if normalized:
        return normalized

    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed or trimmed == "null":
            return None
        try:
            return find_grid_payload(json.loads(trimmed))
        except (ValueError, TypeError):
            candidate = extract_json_candidate(trimmed)
            if not candidate:
                return None
            try:
                return find_grid_payload(json.loads(candidate))
            except (ValueError, TypeError):
                return None

    if isinstance(value, list):
        for item in value:
            found = find_grid_payload(item)
            if found:
                return found
        return None

    if isinstance(value, dict):
        for nested in value.values():
            found = find_grid_payload(nested)
            if found:
                return found

    return None
