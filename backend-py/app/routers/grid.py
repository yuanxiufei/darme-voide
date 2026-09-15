"""grid 域 —— 与 ``backend/src/routes/grid.ts``（347 行）对齐。

**4 个端点全部迁移**（前缀 ``/api/v1/grid``，Node 是 ``api.route('/grid', grid)``）：

* ``POST /prompt``  生成宫格 prompt：**先试 Agent，失败/不可用则回落本地确定性构建器**
  （``build_grid_prompt`` / ``build_grid_cell_prompts``）；
* ``POST /generate`` 提交宫格大图的图片生成（用**统一的**分镜负面词 + 画风收口）；
* ``POST /split``   按 ``assignments`` 把大图切成单格并回写分镜（首帧/尾帧/参考图）；
* ``GET  /status/{id}`` 查这次图片生成的状态。

⚠️ 三处保真点：

* ``/prompt`` 的 Agent 分支**只认工具结果或正文里的 JSON**（``grid_prompt`` 必须非空），
  拿不到就用本地构建器 —— 所以 **Agent 未移植不影响该端点可用**（``source`` 会显示
  ``fallback``）；
* ``/generate`` 的尺寸是 ``960*cols x 540*rows``（固定单元格 960x540，**不随画幅变**）；
* ``/split`` 的 ``reference`` 分支是把格子**追加**到分镜已有 ``reference_images`` 数组末尾，
  且该列与 Node 共用 ⇒ **必须紧凑 JSON**。

✅ ``runAgentWithRetry`` 等价物**已可用**（``services/agents/runtime.run_agent_with_retry``）⇒
  ``try_agent_grid_prompt`` 已按原 TS 接通（2026-09-15 校正，早先这里是「恒返回 None」的存根：
  端点此前**永远**走本地构建器，而 Node 会优先用 Agent ⇒ 曾是真行为缺口）。
  仍保留「Agent 不可用即回落」的语义：任何异常都吞掉返回 ``None`` ⇒ ``source='fallback'``。
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.models import image_generations, storyboards
from ..core.request_utils import read_json
from ..core.response import bad_request, not_found, now, parse_param_id, success
from ..services.grid_split import split_grid_image
from ..services.image_generation import generate_image
from ..services.prompt_utils import (
    build_grid_cell_prompts,
    build_grid_prompt,
    build_reference_legend,
    build_storyboard_negative_prompt,
    collect_grid_reference_assets,
    resolve_effective_art_style,
)
from ..services.task_logger import log_task_error, log_task_payload, log_task_progress

router = APIRouter(prefix="/api/v1/grid", tags=["grid"])

#: 宫格单元固定尺寸（实际画布 = ``960*cols x 540*rows``）
CELL_WIDTH = 960
CELL_HEIGHT = 540


def _extract_json_candidate(text: str) -> str:
    """从模型正文里抠 JSON：优先 ```` ```json ```` 围栏，其次第一个 ``{...}``。"""
    fenced = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced and fenced.group(1):
        return fenced.group(1).strip()
    plain = re.search(r"\{[\s\S]*\}", text)
    return plain.group(0).strip() if plain else ""


def _normalize_grid_payload(payload: Any) -> dict[str, Any] | None:
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


def _find_grid_payload(value: Any) -> dict[str, Any] | None:
    """递归找宫格载荷（字符串会先尝试 JSON 解析 / 抠 JSON）。"""
    if value is None:
        return None

    normalized = _normalize_grid_payload(value)
    if normalized:
        return normalized

    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed or trimmed == "null":
            return None
        try:
            return _find_grid_payload(json.loads(trimmed))
        except (ValueError, TypeError):
            candidate = _extract_json_candidate(trimmed)
            if not candidate:
                return None
            try:
                return _find_grid_payload(json.loads(candidate))
            except (ValueError, TypeError):
                return None

    if isinstance(value, list):
        for item in value:
            found = _find_grid_payload(item)
            if found:
                return found
        return None

    if isinstance(value, dict):
        for nested in value.values():
            found = _find_grid_payload(nested)
            if found:
                return found

    return None


async def try_agent_grid_prompt(
    conn: Connection,
    episode_id: int,
    drama_id: int,
    storyboard_ids: list[int],
    rows: int,
    cols: int,
    mode: str,
    reference_legend: str,
) -> dict[str, Any] | None:
    """让 Agent 生成宫格 prompt（``grid_prompt_generator``）—— 抠 ``{grid_prompt, cell_prompts}``。

    与原 TS 逐字对齐：消息由 9 段拼接（**空段丢弃**、``\\n`` 连接，镜头 ID 走紧凑 JSON）；
    先看**工具结果**再看**正文**；**任何异常都吞掉返回 ``None``** ⇒ 端点回落本地构建器。
    """
    message = "\n".join(part for part in (
        "请为宫格图生成提示词，并优先调用工具完成。",
        "选中镜头ID：" + json.dumps(storyboard_ids, ensure_ascii=False, separators=(",", ":")),
        f"行数：{rows}",
        f"列数：{cols}",
        f"模式：{mode}",
        f"参考图映射：{reference_legend}" if reference_legend else "",
        "当提示词涉及到某个角色或场景时，直接把对应的图片编号写进提示词，"
        "例如：图片1中的角色A站了起来，图片3中的房间场景。不要只写名字，不写图片编号。",
        f"必须严格按 {rows}x{cols} 生成，总共 exactly {rows * cols} visible panels。"
        f"不要合并格子，不要缺格。",
        '必须返回 JSON，结构为：{"grid_prompt":"...","cell_prompts":'
        '[{"shot_number":1,"frame_type":"first_frame","prompt":"..."}]}',
    ) if part)

    try:
        # 惰性导入：避免 routing → services.agents 的导入环（与其它路由同一手法）
        from app.agent.runtime import run_agent_with_retry  # noqa: PLC0415

        result = await run_agent_with_retry(
            conn, "grid_prompt_generator", episode_id, drama_id, message, {"maxSteps": 10})
        from_tools = _find_grid_payload(result.tool_results)
        if from_tools:
            return from_tools
        return _find_grid_payload(result.text)
    except Exception:  # noqa: BLE001 —— 与原 TS 的裸 catch 等价：Agent 失败即回落
        return None


def _fetch_storyboards(conn: Connection, storyboard_ids: list[Any]) -> list[Any]:
    """按传入顺序取分镜（缺失的跳过）。"""
    rows = []
    for storyboard_id in storyboard_ids:
        row = conn.execute(
            select(storyboards).where(storyboards.c.id == storyboard_id)
        ).first()
        if row is not None:
            rows.append(row)
    return rows


@router.post("/prompt")
async def grid_prompt(request: Request, conn: Connection = Depends(get_tx)):
    """生成宫格 prompt（Agent 优先，失败则本地确定性构建）。"""
    try:
        body = await read_json(request)
        storyboard_ids = body.get("storyboard_ids")
        drama_id = body.get("drama_id")
        episode_id = body.get("episode_id")
        rows = body.get("rows")
        cols = body.get("cols")
        mode = body.get("mode") or "first_frame"

        if not storyboard_ids:
            return bad_request("storyboard_ids required")
        if not rows or not cols:
            return bad_request("rows and cols required")

        storyboard_rows = _fetch_storyboards(conn, storyboard_ids)
        if not storyboard_rows:
            return bad_request("No storyboards found")

        drama_style = resolve_effective_art_style(conn, drama_id)
        actual_rows, actual_cols = rows, cols
        resolved_episode_id = int(episode_id or storyboard_rows[0].episode_id or 0)
        reference_assets = collect_grid_reference_assets(conn, storyboard_rows)
        reference_legend = build_reference_legend(reference_assets)

        if not resolved_episode_id:
            return bad_request("episode_id required")

        try:
            agent_payload = await try_agent_grid_prompt(
                conn, resolved_episode_id, int(drama_id or 0), storyboard_ids,
                actual_rows, actual_cols, mode, reference_legend,
            )
            if agent_payload and agent_payload.get("grid_prompt"):
                log_task_progress("GridPrompt", "agent-success", {
                    "episodeId": resolved_episode_id, "dramaId": drama_id, "mode": mode,
                    "rows": actual_rows, "cols": actual_cols,
                    "storyboardCount": len(storyboard_ids),
                })
                log_task_payload("GridPrompt", "agent-result", agent_payload)
                return success({
                    **agent_payload,
                    "source": "agent",
                    "grid": {"rows": actual_rows, "cols": actual_cols},
                    "storyboard_ids": storyboard_ids,
                    "mode": mode,
                })
        except Exception as err:  # noqa: BLE001 —— 与原实现一致：Agent 失败即回落
            log_task_error("GridPrompt", "agent-failed", {
                "episodeId": resolved_episode_id, "dramaId": drama_id, "error": str(err),
            })

        grid_prompt_text = build_grid_prompt(
            conn, mode, storyboard_rows, actual_rows, actual_cols, drama_style, reference_assets
        )
        cell_prompts = build_grid_cell_prompts(
            conn, mode, storyboard_rows, actual_rows, actual_cols, reference_assets
        )
        log_task_progress("GridPrompt", "fallback-used", {
            "episodeId": resolved_episode_id, "dramaId": drama_id, "mode": mode,
            "rows": actual_rows, "cols": actual_cols,
            "storyboardCount": len(storyboard_ids),
        })
        return success({
            "grid_prompt": grid_prompt_text,
            "cell_prompts": cell_prompts,
            "source": "fallback",
            "grid": {"rows": actual_rows, "cols": actual_cols},
            "storyboard_ids": storyboard_ids,
            "mode": mode,
        })
    except Exception as err:  # noqa: BLE001
        log_task_error("GridPrompt", "prompt", {"error": str(err)})
        return bad_request(str(err))


@router.post("/generate")
async def grid_generate(request: Request, conn: Connection = Depends(get_tx)):
    """提交宫格大图的图片生成（画布 = ``960*cols x 540*rows``）。"""
    try:
        body = await read_json(request)
        storyboard_ids = body.get("storyboard_ids")
        drama_id = body.get("drama_id")
        rows = body.get("rows")
        cols = body.get("cols")
        mode = body.get("mode") or "first_frame"
        custom_prompt = body.get("custom_prompt")

        if not storyboard_ids:
            return bad_request("storyboard_ids required")
        if not rows or not cols:
            return bad_request("rows and cols required")

        storyboard_rows = _fetch_storyboards(conn, storyboard_ids)
        if not storyboard_rows:
            return bad_request("No storyboards found")

        drama_style = resolve_effective_art_style(conn, drama_id)
        reference_assets = collect_grid_reference_assets(conn, storyboard_rows)
        prompt = custom_prompt or build_grid_prompt(
            conn, mode, storyboard_rows, rows, cols, drama_style, reference_assets
        )
        reference_images = [asset["path"] for asset in reference_assets]

        size = f"{CELL_WIDTH * cols}x{CELL_HEIGHT * rows}"

        try:
            generation_id = await generate_image(conn, {
                "dramaId": drama_id,
                "prompt": prompt,
                # 宫格图与分镜静态图共用同一套负面词（含画风对立词）
                "negativePrompt": build_storyboard_negative_prompt(drama_style),
                "size": size,
                "frameType": f"grid_{mode}_{rows}x{cols}",
                "referenceImages": reference_images,
            })
            log_task_progress("GridGenerate", "reference-images", {
                "dramaId": drama_id, "mode": mode, "rows": rows, "cols": cols,
                "referenceCount": len(reference_images),
            })
            return success({
                "image_generation_id": generation_id,
                "grid": {"rows": rows, "cols": cols},
                "mode": mode,
                "storyboard_ids": storyboard_ids,
                "prompt": prompt,
                "reference_images": reference_images,
            })
        except Exception as err:  # noqa: BLE001
            return bad_request(str(err))
    except Exception as err:  # noqa: BLE001
        return bad_request(str(err))


@router.post("/split")
async def grid_split(request: Request, conn: Connection = Depends(get_tx)):
    """把宫格大图切成单格，并按 ``assignments`` 回写分镜（首帧/尾帧/参考图）。"""
    try:
        body = await read_json(request)
        generation_id = body.get("image_generation_id")
        rows = body.get("rows")
        cols = body.get("cols")
        assignments = body.get("assignments")

        if not generation_id:
            return bad_request("image_generation_id required")
        if not rows or not cols:
            return bad_request("rows and cols required")
        if not assignments:
            return bad_request("assignments required")

        record = conn.execute(
            select(image_generations).where(image_generations.c.id == generation_id)
        ).first()
        if record is None:
            return bad_request("Image generation not found")
        if record.status != "completed":
            return bad_request(f"Image status: {record.status}")
        if not record.local_path:
            return bad_request("No local image file")

        try:
            cells = await split_grid_image(record.local_path, rows, cols)

            results: list[dict[str, Any]] = []
            for i in range(min(len(assignments), len(cells))):
                assignment = assignments[i] or {}
                storyboard_id = assignment.get("storyboard_id")
                frame_type = assignment.get("frame_type")
                cell = cells[i]
                if not storyboard_id:
                    continue

                values: dict[str, Any] = {"updated_at": now()}
                if frame_type == "first_frame":
                    values["first_frame_image"] = cell["localPath"]
                elif frame_type == "last_frame":
                    values["last_frame_image"] = cell["localPath"]
                elif frame_type == "reference":
                    row = conn.execute(
                        select(storyboards.c.reference_images)
                        .where(storyboards.c.id == storyboard_id)
                    ).first()
                    existing: list[Any] = []
                    if row is not None and row[0]:
                        try:
                            existing = json.loads(row[0])
                        except (ValueError, TypeError):
                            existing = []
                        if not isinstance(existing, list):
                            existing = []
                    existing.append(cell["localPath"])
                    # ⚠️ 该列与 Node 共用 ⇒ 紧凑分隔符（不是默认的 ", " / ": "）
                    values["reference_images"] = json.dumps(
                        existing, ensure_ascii=False, separators=(",", ":")
                    )

                conn.execute(
                    update(storyboards).where(storyboards.c.id == storyboard_id).values(**values)
                )
                results.append({
                    "storyboard_id": storyboard_id,
                    "frame_type": frame_type,
                    "local_path": cell["localPath"],
                })

            return success({"cells": results})
        except Exception as err:  # noqa: BLE001
            return bad_request(str(err))
    except Exception as err:  # noqa: BLE001
        return bad_request(str(err))


@router.get("/status/{generation_id}")
def grid_status(generation_id: str, conn: Connection = Depends(get_conn)):
    """查某次宫格图片生成的状态。"""
    try:
        gen_id = parse_param_id(generation_id)
        if gen_id is None:
            return not_found("Invalid image generation id")
        row = conn.execute(
            select(image_generations).where(image_generations.c.id == gen_id)
        ).first()
        if row is None:
            return not_found("Not found")
        return success({
            "id": row.id,
            "status": row.status,
            "local_path": row.local_path,
            "image_url": row.image_url,
            "error_msg": row.error_msg,
        })
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(err)})
