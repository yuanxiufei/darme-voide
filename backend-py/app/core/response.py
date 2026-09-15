"""统一响应层 —— 与 ``backend/src/utils/response.ts`` 逐字段对齐。

实测契约（前端 ``frontend/app/composables/useApi.ts`` 的消费方式）：

* 成功：HTTP 200 + ``{"code": 200, "data": ..., "message": "success"}``
* 创建：HTTP 201 + ``{"code": 201, "data": ..., "message": "created"}``
* 失败：HTTP 4xx/5xx + ``{"code": <status>, "message": "..."}``  ← **没有 data 字段**

前端判据是 ``!resp.ok || (json.code && json.code >= 400)``，读 ``json.message`` 取错误文案，
返回 ``json.data ?? json``。故上表任何一项写错都会直接导致前端报错文案异常。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from fastapi.responses import JSONResponse

# Node 侧 table 名 → 本模块函数名，便于对照阅读
_OK = 200
_CREATED = 201
_BAD_REQUEST = 400
_NOT_FOUND = 404
_CONFLICT = 409
_SERVER_ERROR = 500


def success(data: Any = None) -> JSONResponse:
    """成功信封 —— **`data` 键永远存在**（即使为 ``null``）。

    ⚠️ 这里曾经误判过一次，值得记下来：TS 侧是

    ```ts
    export function success(c: Context, data: any = null) {
      return c.json({ code: 200, data, message: 'success' })
    }
    ```

    调用点常写成 ``success(c, 可能有值 ? { x } : undefined)``。当时我按「``JSON.stringify``
    会丢掉值为 ``undefined`` 的键」推断出「无值时 data 键缺失」，为此专门做了个
    ``success_without_data()``。**但那是错的**：JS 的**默认参数**会把显式传入的
    ``undefined`` 先换成 ``null``（默认参数在实参为 undefined 时生效），所以 data 仍是
    ``null``、键仍存在。⇒ 该 helper 已删除，相关路由改回 ``success()``。

    真正的「无 data 键」只出现在**错误信封**（``bad_request`` / ``not_found`` / ``conflict``），
    它们本来就返回 ``{code, message}``。
    """
    return JSONResponse(status_code=_OK, content={"code": _OK, "data": data, "message": "success"})


def created(data: Any = None) -> JSONResponse:
    return JSONResponse(status_code=_CREATED, content={"code": _CREATED, "data": data, "message": "created"})


def bad_request(message: str = "bad request") -> JSONResponse:
    return JSONResponse(status_code=_BAD_REQUEST, content={"code": _BAD_REQUEST, "message": message})


def not_found(message: str = "not found") -> JSONResponse:
    return JSONResponse(status_code=_NOT_FOUND, content={"code": _NOT_FOUND, "message": message})


def conflict(message: str = "conflict") -> JSONResponse:
    return JSONResponse(status_code=_CONFLICT, content={"code": _CONFLICT, "message": message})


def server_error(message: str = "internal error") -> JSONResponse:
    return JSONResponse(status_code=_SERVER_ERROR, content={"code": _SERVER_ERROR, "message": message})


def now() -> str:
    """与 JS ``new Date().toISOString()`` 同形：``2026-09-12T07:11:22.123Z``。"""
    ts = datetime.now(timezone.utc)
    return f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}.{ts.microsecond // 1000:03d}Z"


def parse_param_id(raw: Any, *, positive_only: bool = True) -> int | float | None:
    """安全解析路由参数 id，非法时返回 None。

    对齐 Node ``utils/response.ts`` 的 ``parseParamId``：``Number.isFinite(Number(raw)) && > 0``。
    ⚠️ 该判据**允许小数**（``"1.5"`` → 1.5 通过）⇒ 后续查库自然查不到，由调用方给出
    「xxx not found」而非「Invalid xxx id」。这与 props 的 ``parseId``（``Number.isInteger``）
    **不同**，别合并 —— 见 ``routers/props.py`` 的注释。

    已知有意偏差：``Number('0x10') === 16`` 而 Python 的 ``float()`` 不接受十六进制 ⇒ 返回 None。
    属极端脏输入，且两条路径最终都是 404。
    """
    if isinstance(raw, bool):  # bool 是 int 的子类，JS 里 true 会变 1，这里显式拒绝
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / Infinity
        return None
    if positive_only and value <= 0:
        return None
    return int(value) if value.is_integer() else value


# ---------------------------------------------------------------------------
# 行 → dict / 字段映射工具
# ---------------------------------------------------------------------------

def row_to_dict(row: Any) -> dict[str, Any]:
    """SQLAlchemy Row → dict。列名本就是 snake_case，等于契约要求的形状。"""
    return dict(row._mapping) if row is not None else {}


def rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in rows]


def camel_to_snake(key: str) -> str:
    """``episodeNumber`` → ``episode_number``（对齐 Node 的 key.replace(/[A-Z]/g, ...)）。"""
    out: list[str] = []
    for ch in key:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def js_number(raw: Any) -> int | float | None:
    """镜像 JS 的 ``Number(raw)``：非有限值（``NaN``/``Infinity``）返回 None。

    用于「``Number.isFinite(Number(x))`` 才放行」这类校验。注意与 ``parse_param_id`` 的区别：
    这里**不检查 ``> 0``**（负数、0、小数都算通过），因为原 TS 就是只判有限性。

    ⚠️ **整数值必须返回 ``int`` 而不是 ``float``**：JS 只有一种数字类型，
    ``JSON.stringify(Number('37'))`` 得 ``37``；而 Python 的 ``json.dumps(37.0)`` 得 ``37.0``。
    这个差异前端**用宽松比较看不出来**（``37 == 37.0`` 为真），但会出现在**响应体字节**里，
    一旦前端用 ``===`` 严格比较或把值拼进字符串（如 ``scope: "drama-37.0"``）就会暴露。
    （这个 bug 曾真实存在于所有「回显 query 里的数字」的端点。）

    安全性边界：只在 ``|x| <= 2^53``（JS 安全整数范围）内转 int —— 再大 JS 会用指数形式
    序列化（``1e+30``），而 Python 转 int 会吐一长串数字，反而更不像。
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return _normalize_js_number(raw)
    text = str(raw).strip()
    if text == "":
        return 0  # Number('') === 0
    try:
        value = float(text)
    except ValueError:
        return None  # JS 此处为 NaN
    return _normalize_js_number(value)


def _normalize_js_number(value: float) -> int | float | None:
    if not math.isfinite(value):
        return None
    if value.is_integer() and abs(value) <= 2**53:
        return int(value)
    return value


def js_round(value: float) -> int:
    """镜像 JS 的 ``Math.round``。

    ⚠️ **不能用 Python 内置的 ``round``**：Python 是银行家舍入（``round(2.5) == 2``），
    而 JS 的 ``Math.round`` 是「.5 向 +∞」（``Math.round(2.5) === 3``、``Math.round(-1.5) === -1``）。
    成本金额一直按 ``Math.round(x * 10000) / 10000`` 保留 4 位小数，用错会让金额差 1 个最小单位。
    """
    return math.floor(value + 0.5)


def js_truthy(value: Any) -> bool:
    """镜像 JS 的真值判断（用于 ``a || b`` 与 ``cond ? x : y`` 这类分支）。

    ⚠️ **最容易踩的一条**：**JS 里空数组 ``[]`` 与空对象 ``{}`` 都是「真值」**，
    而 Python 里是假值。``style-profiles`` 的 ``body.storytelling || {}``、
    ``input.preferences ? JSON.stringify(...) : null``、``agent-configs`` 的
    ``if (body.skills)`` 都依赖这一点 —— 直接写 Python 的 ``if value:`` 会把用户
    显式传入的 ``{}`` / ``[]`` 误判成「没传」而落成 null。
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == value and value != 0  # NaN 与 0 为假
    if isinstance(value, str):
        return value != ""
    return True  # list / dict（含空）/ 其他对象 → JS 为真


def js_nullish(value: Any, fallback: Any) -> Any:
    """镜像 JS 的 ``a ?? b``（**只看 null/undefined**，与 ``||`` 不同）。

    区别很关键：``model: body.model ?? existing.model`` 里传 ``''`` 会**保留空串**，
    而 ``name: body.name || existing.name`` 里传 ``''`` 会**回退旧值**。
    """
    return fallback if value is None else value


def internal_error(message: str = "internal error") -> JSONResponse:
    """对齐 Node 里散布的 ``c.json({code:500, data:null, message}, 500)``。"""
    return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": message})


def to_camel_case(key: str) -> str:
    """``episode_number`` → ``episodeNumber``。"""
    parts = key.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


#: 列名 → JS 属性名的**例外表**：该表的 drizzle 属性名并不等于列名的 camelCase。
#:
#: 由 `tmp/schema-pairs.mjs` 全量实测得出：`schema.ts` 共 **493** 个字段，
#: 「属性名 ≠ camelCase(列名)」的**只有 1 处** —— `prop_templates`：
#: TS 写的是 `customPrompt: text('image_prompt')`（列名是历史遗留的 image_prompt）。
#: 不做这层映射就会返回 `imagePrompt`，而前端与 props 路由用的都是 `customPrompt`。
_COLUMN_TO_JS_OVERRIDES: dict[str, dict[str, str]] = {
    "prop_templates": {"image_prompt": "customPrompt"},
}


def dict_to_camel(row: dict[str, Any], table: str | None = None) -> dict[str, Any]:
    """snake_case dict → Node 端的 JS 属性名。

    **为什么必须双向转换**：Node 侧两类端点返回的形状不同，是既成契约：
    * 显式调过 ``toSnakeCase`` 的（dramas / episodes 列表与详情）→ **snake_case**
    * 直接返回 drizzle 行的（characters / scenes / props）→ **camelCase**
      （drizzle 返回的是 TS 声明的属性名）
    前端也确实按这个差异写代码（episode 工作台里是 ``c.voice_style || c.voiceStyle``
    的防御式双读，而 ``CharacterEditor.vue`` 直接用 ``c.voiceStyle``）⇒ 不能「顺手统一」。
    """
    overrides = _COLUMN_TO_JS_OVERRIDES.get(table or "", {})
    return {overrides.get(k, to_camel_case(k)): v for k, v in row.items()}


def row_to_camel(row: Any, table: str | None = None) -> dict[str, Any]:
    """SQLAlchemy Row → camelCase dict（对齐 drizzle 行）。"""
    return dict_to_camel(row_to_dict(row), table)


def rows_to_camel(rows: Any, table: str | None = None) -> list[dict[str, Any]]:
    return [dict_to_camel(dict(r._mapping), table) for r in rows]


def pick_fields(src: dict[str, Any], snake_keys: list[str]) -> dict[str, Any]:
    """白名单取字段，**snake_case 优先、camelCase 兜底**。

    对齐 Node ``routes/dramas.ts`` 的 ``pickFields``：先看 snake 键是否存在，再看 camel 键。
    目的是兼容历史上前端双写两种命名的调用方，同时避免把未知列写进 SQL 触发 no such column。
    """
    out: dict[str, Any] = {}
    for key in snake_keys:
        camel = "".join(part.capitalize() if i else part for i, part in enumerate(key.split("_")))
        if key in src:
            out[key] = src[key]
        elif camel in src:
            out[key] = src[camel]
    return out


def parse_json_array(raw: Any) -> list[Any]:
    """``tags`` 这类以 JSON 文本落库的列 → list（解析失败回退空数组，和 Node 的语义一致）。"""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        import json

        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (ValueError, TypeError):
        return []
