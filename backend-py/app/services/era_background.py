"""剧集时代背景 —— 移植 ``backend/src/services/era-background.ts`` 的**纯函数部分**。

本模块只包含不依赖 LLM 的解析逻辑（``parseEraBackground``）。
``extractDramaEraBackground`` 依赖文本生成链路，属后续阶段，故未移植，
对应的 ``POST /dramas/:id/era-background/extract`` 也暂时不注册。

⚠️ 解析规则刻意与 TS 版**逐字对齐**（包括它比错误文案更宽松这一点）：
   实际判据是 ``summary || imageHint`` 非空，**不是**「era/summary/imageHint 三者齐全」，
   而路由层的 400 文案写的是后者。以代码为准，否则会拒掉 Node 侧本来接受的存量数据。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..models import dramas

_EraBackground = dict[str, str]


def _js_truthy(value: Any) -> bool:
    """JS 的 falsy 集合：undefined / null / '' / 0 / NaN / false。"""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _js_or(*values: Any) -> Any:
    """模拟 JS 的 ``a || b || c``：返回第一个 truthy 值，全 falsy 时返回最后一个。"""
    for v in values:
        if _js_truthy(v):
            return v
    return values[-1] if values else ""


def _js_str(value: Any) -> str:
    """模拟 JS 的 ``String(x)``。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        # TS 侧对对象会得到 "[object Object]"，但那属于脏输入；这里保留 JSON 便于排查。
        # 用紧凑分隔符，保持全仓 JSON 文本风格一致（Node 的 JSON.stringify 无空格）
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def parse_era_background(raw: str | None) -> _EraBackground | None:
    """解析 ``dramas.era_background``（JSON 文本）为结构化对象；空/非法返回 None。"""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    out = {
        "era": _js_str(_js_or(obj.get("era"), obj.get("summary"), "")).strip()[:80],
        "summary": _js_str(_js_or(obj.get("summary"), obj.get("raw"), obj.get("era"), "")).strip()[:2000],
        # image_style_en 是历史键，优先级高于新键 imageHint（保持向后兼容）
        "imageHint": _js_str(
            _js_or(obj.get("image_style_en"), obj.get("imageHint"), obj.get("summary"), "")
        ).strip()[:1500],
    }
    if not out["summary"] and not out["imageHint"]:
        return None
    return out


def era_background_to_json(norm: _EraBackground) -> str:
    """落库用：与 TS 的 ``JSON.stringify(norm)`` 等价（键序固定为 era/summary/imageHint）。"""
    return json.dumps(norm, ensure_ascii=False, separators=(",", ":"))


def get_era_background(conn: Connection, drama_id: Any = None) -> _EraBackground | None:
    """读某剧的时代背景（``dramas.era_background`` 解析后）。

    ``if (!dramaId) return null`` 是 **JS 假值判断**：0 / 空串 / None 都返回 None。
    """
    if not drama_id:
        return None
    row = conn.execute(
        select(dramas.c.era_background).where(dramas.c.id == drama_id)
    ).first()
    if row is None:
        return None
    return parse_era_background(row[0])


def apply_era_image_clause(conn: Connection, prompt: str, drama_id: Any = None) -> str:
    """时代背景注入：``imageHint`` 存在时作为「时代/环境画面指令」**追加到 prompt 尾部**。

    不改变调用方原有 prompt 结构；无背景时原样返回。
    末尾点的处理：已有句号就不再加（``hint.endsWith('.') ? hint : `${hint}.```
    ``），最终拼成 ``"<prompt>, <hint>."``。
    """
    if not drama_id:
        return prompt
    era = get_era_background(conn, drama_id)
    hint = ((era or {}).get("imageHint") or "").strip()
    if not hint:
        return prompt
    clause = hint if hint.endswith(".") else f"{hint}."
    return f"{prompt}, {clause}"
