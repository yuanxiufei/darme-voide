"""宫格图提示词工具（移植自 ``agents/tools/grid-prompt-tools.ts``，92 行）。

宫格 prompt 的构建已统一收敛到 :mod:`app.services.prompt_utils`；本工具**只负责读镜头数据
并调用统一构建器**，避免与 ``routers/grid.py`` 重复实现、逻辑分叉 —— 也就是说
**agent 路径与路由 fallback 必须产出完全一致的 prompt**（画风也走同一解析链）。

两个工具：

* ``read_shots_for_grid``  读选中镜头详情（供 Agent 判断格数与内容）；
* ``generate_grid_prompt`` 按 ``rows``/``cols``/``mode`` 生成整体 prompt + 逐格 prompt。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from ....db import engine
from ....models import storyboards
from ....services.prompt_utils import (
    build_grid_cell_prompts,
    build_grid_prompt,
    collect_grid_reference_assets,
    resolve_effective_art_style,
)
from ..tool import Tool, array_of, json_number, json_string, object_schema

__all__ = ["create_grid_prompt_tools"]


def _col(row: Any, name: str) -> Any:
    return getattr(row, name, None) if not isinstance(row, dict) else row.get(name)


def create_grid_prompt_tools(episode_id: int, drama_id: int) -> dict[str, Tool]:
    """工厂（**与 Node 同形**：闭包只注入 ``episodeId`` / ``dramaId``）。

    ⚠️ 每个工具**自己开短事务**（Node 用的是全局 ``db``）。这样 agent 循环期间不会长期占着
    写事务（SQLite 单写者），也不会踩「连接已被调用方关闭」—— 这是 compose 那轮学到的同一条教训。
    """

    async def read_shots_for_grid(arguments: dict[str, Any]) -> dict[str, Any]:
        shot_ids = arguments.get("shot_ids") or []
        if not shot_ids:
            return {"shots": []}
        with engine.begin() as conn:
            rows = conn.execute(
            select(storyboards)
                .where(storyboards.c.episode_id == episode_id)
                .order_by(storyboards.c.storyboard_number)
            ).all()
        return {"shots": [
            {
                "shot_number": _col(row, "storyboard_number"),
                # `sb.description || sb.title || ''`：两者都空则空串
                "description": _col(row, "description") or _col(row, "title") or "",
                "shot_type": _col(row, "shot_type") or "",
                "dialogue": _col(row, "dialogue") or "",
                "location": _col(row, "location") or "",
                "time": _col(row, "time") or "",
                "movement": _col(row, "movement") or "",
                "first_frame_prompt": _col(row, "first_frame_prompt") or "",
                "last_frame_prompt": _col(row, "last_frame_prompt") or "",
            }
                for row in rows
                if _col(row, "id") in shot_ids
            ]}

    async def generate_grid_prompt(arguments: dict[str, Any]) -> dict[str, Any]:
        shots = arguments.get("shots") or []
        rows = arguments.get("rows")
        cols = arguments.get("cols")
        mode = arguments.get("mode")
        if not shots:
            return {"error": "No shots provided", "grid_prompt": "", "cell_prompts": []}

        # 反查 DB 拿完整分镜（含参考图/角色外观/场景字段），
        # 保证 agent 路径与路由 fallback 用的是**同一套**构建逻辑
        shot_numbers = [shot.get("shot_number") for shot in shots]
        with engine.begin() as conn:
            all_rows = conn.execute(
            select(storyboards)
                .where(storyboards.c.episode_id == episode_id)
                .order_by(storyboards.c.storyboard_number)
            ).all()
        ordered = []
        for number in shot_numbers:
            match = next(
                (row for row in all_rows if _col(row, "storyboard_number") == number), None
            )
            if match is not None:
                ordered.append(match)

        # 画风收口：与路由 fallback 走同一解析链
        with engine.begin() as conn:
            drama_style = resolve_effective_art_style(conn, drama_id)
            reference_assets = collect_grid_reference_assets(conn, ordered)
            grid_prompt = build_grid_prompt(
                conn, mode, ordered, rows, cols, drama_style, reference_assets
            )
            cell_prompts = build_grid_cell_prompts(
                conn, mode, ordered, rows, cols, reference_assets
            )

        return {
            "grid_prompt": grid_prompt,
            "cell_prompts": cell_prompts,
            "reference_assets": [
                {"image_label": asset["imageLabel"], "path": asset["path"], "label": asset["label"]}
                for asset in reference_assets
            ],
        }

    #: zod 的 ``z.object({...})`` 里只有 ``shot_number`` 是必填（其余 ``.optional()``）
    shot_schema = object_schema(
        {
            "shot_number": json_number(),
            "description": json_string(),
            "shot_type": json_string(),
            "dialogue": json_string(),
            "location": json_string(),
            "time": json_string(),
            "movement": json_string(),
            "first_frame_prompt": json_string(),
            "last_frame_prompt": json_string(),
        },
        required=["shot_number"],
    )

    return {
        "read_shots_for_grid": Tool(
            id="read_shots_for_grid",
            description="读取选中镜头的详细信息，用于生成宫格图提示词。",
            input_schema=object_schema({"shot_ids": array_of({"type": "number"})}),
            execute=read_shots_for_grid,
        ),
        "generate_grid_prompt": Tool(
            id="generate_grid_prompt",
            description=(
                "为宫格图生成整体画面描述和每个格子的独立提示词。"
                "遵循 grid-image-generator SKILL.md 的三种模式规范。"
            ),
            input_schema=object_schema(
                {
                    "shots": array_of(shot_schema),
                    "rows": json_number(),
                    "cols": json_number(),
                    # 'first_frame' | 'first_last' | 'multi_ref'
                    "mode": json_string(),
                },
                required=["shots", "rows", "cols", "mode"],
            ),
            execute=generate_grid_prompt,
        ),
    }
