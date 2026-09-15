"""资源库（角色/场景/兵器/服装模板）的**共享实现**。

Node 侧这四个文件（``characterLibrary.ts`` / ``sceneLibrary.ts`` / ``weaponLibrary.ts`` /
``costumeLibrary.ts``）是「同一套逻辑换列清单」的复制体：都走 ``queryHelper`` 的原生 SQL、
都有 list/categories/tags/get/create/update/delete/batch-delete，差别只在列名、搜索列、
排序白名单、额外过滤项、以及 apply/from-* 的去向。

与其抄 4 份（4 份就意味着 4 处可能漂移），这里实现**一份规格驱动**的构建器，
每个库只提供一份可审阅的 ``LibrarySpec``。

⚠️ 两个容易踩对的点：

1. **资源库用的是另一套响应信封**：成功 ``code: 0``（不是 200），错误 ``code: 400/404/500``
   且 **HTTP 仍是 200**（TS 是 ``c.json(obj)`` 不带状态码）。前端 ``useApi.ts`` 的判据
   ``!resp.ok || (json.code && json.code >= 400)`` 正好解释了为什么它要额外看 ``json.code``。
2. **列表/详情的 ``data`` 里没有 ``message`` 键**（``c.json({code:0, data})``），
   只有写操作才有。故 ``lib_ok`` 的 message 是可选的。

另外三个 JS 语义要对齐：``parseInt`` 的宽松解析（``"12abc"`` → 12）、``!v`` 的真值判断
（``[]`` 是 truthy，别写成 Python 的 ``not v``）、``safeStringify`` 对「非 JSON 字符串」
会返回**再编码一次的 JSON 字符串**（读回来能还原）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.request_utils import read_json
from ..core.response import now

#: ``source_drama_id`` 这类列在创建时用 ``|| null``（缺省写 NULL 而非 ''）
_NULL_ON_CREATE = ("source_drama_id",)

_PARSE_INT = re.compile(r"[+-]?\d+")


# ---------------------------------------------------------------------------
# 原生 SQL 薄封装（资源库这四个域在 Node 侧走的是 better-sqlite3 原生 SQL）
#
# ⚠️ 两个坑，缺一个都会报 `ArgumentError: List argument must consist only of dictionaries`：
#
# 1. **SQLAlchemy 的 `text()` 不支持 `?` 占位符** —— 它只认 `:named` 风格。把带 `?` 的
#    SQL 交给 `text()`，那些 `?` 不会被识别为绑定参数，运行时就会抛参数蒸馏错误。
#    正确做法是 `exec_driver_sql`：**绕过 SQLAlchemy 的编译**，把 qmark SQL 原样交给
#    底层 sqlite3（sqlite3 原生就支持 `?`）。
# 2. **即使是 `exec_driver_sql`，位置参数也必须传 tuple** —— 传 list 会被当成
#    「多组参数」（executemany 语义）。
#
# 本模块与 `routers/libraries.py` 的所有原生 SQL 都必须走这三个函数。
# ---------------------------------------------------------------------------

def q_all(conn: Connection, sql: str, params: Any = ()) -> list[Any]:
    return _exec(conn, sql, params).mappings().all()


def q_one(conn: Connection, sql: str, params: Any = ()) -> Any:
    return _exec(conn, sql, params).mappings().first()


def q_run(conn: Connection, sql: str, params: Any = ()) -> Any:
    return _exec(conn, sql, params)


def _exec(conn: Connection, sql: str, params: Any) -> Any:
    """无参时不传第二参数 —— 空 tuple 有被当成「0 组参数」而返回空结果的风险。"""
    values = tuple(params)
    return conn.exec_driver_sql(sql, values) if values else conn.exec_driver_sql(sql)


def js_parse_int(value: Any) -> int | None:
    """模拟 JS ``parseInt``：允许前导空白与正负号，遇到非数字即截断。

    ``parseInt("12abc") === 12``、``parseInt("abc") === NaN``、``parseInt("3.9") === 3``。
    资源库路由全用这个（**不是** ``parseParamId``），所以 ``/…/12abc`` 会命中 id=12。
    已知偏差：不处理 ``parseInt("0x10") === 16`` 的进制自动识别（极端脏输入）。
    """
    if value is None:
        return None
    m = _PARSE_INT.match(str(value).lstrip())
    if not m:
        return None
    return int(m.group(0))


def js_truthy(v: Any) -> bool:
    """JS 真值判断。⚠️ ``[]`` 与 ``{}`` 在 JS 里是 **truthy**，Python 的 ``not v`` 会判反。"""
    if v is None or v is False:
        return False
    if isinstance(v, str):
        return v != ""
    if isinstance(v, (int, float)):
        return v != 0
    return True


def safe_parse_json(v: Any) -> Any:
    """对齐 TS ``safeParseJson``：falsy → ``[]``（空串时）或 ``null``；对象原样；否则尝试解析。"""
    if not js_truthy(v):
        return [] if v == "" else None
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def safe_stringify(v: Any) -> str:
    """对齐 TS ``safeStringify``。

    注意字符串分支：**是合法 JSON 就原样返回，否则再编码一次**（``"abc"`` → ``'"abc"'``）。
    这样写回的列用 ``safeParseJson`` 读能还原成原字符串，不会丢数据。
    """
    if v is None:
        return ""
    if isinstance(v, str):
        try:
            json.loads(v)
            return v
        except (ValueError, TypeError):
            # ⚠️ 紧凑分隔符（Node 是 `JSON.stringify(v)`）
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def lib_ok(data: Any = None, message: str | None = None) -> JSONResponse:
    """资源库成功响应：``code: 0``；``message`` 仅写操作才有（列表/详情没有该键）。"""
    body: dict[str, Any] = {"code": 0, "data": data}
    if message is not None:
        body["message"] = message
    return JSONResponse(status_code=200, content=body)


def lib_err(code: int, message: str) -> JSONResponse:
    """资源库错误响应：**HTTP 仍是 200**，错误码在 body 的 ``code`` 里。"""
    return JSONResponse(status_code=200, content={"code": code, "data": None, "message": message})


def camel(key: str) -> str:
    parts = key.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


@dataclass(frozen=True)
class LibrarySpec:
    """一个资源库的全部差异点。其余行为由 ``build_library_router`` 统一实现。"""

    path: str                      # 路由前缀，如 /api/v1/character-library
    table: str                     # DB 表名
    label: str                     # 中文名，用于错误/成功文案（"角色模板"）
    search_columns: tuple[str, ...]
    allowed_sorts: tuple[str, ...]
    columns: tuple[str, ...]       # 可创建/可更新的列（不含 id / created_at / updated_at / usage_count）
    json_columns: tuple[str, ...]  # 需 safeStringify / safeParseJson 的列
    required: tuple[str, ...] = ("name",)
    required_message: str = "名称为必填项"
    exact_filters: tuple[tuple[str, str], ...] = ()      # (query 参数名, 列名)
    #: 非 JSON 列在创建时的默认值（TS 的 `x || default`）；其余非 JSON 列默认 `''`
    create_defaults: dict[str, Any] = field(default_factory=lambda: {"category": "通用"})
    #: 列表/详情里额外补一份 camelCase 别名的列（原 snake 列**保留**）
    out_aliases: tuple[str, ...] = ("reference_images",)
    #: 可选：filter-options 的构建器（返回 dict）
    filter_options: Callable[[Connection], dict[str, Any]] | None = None
    #: 可选：(query 参数名, 列名, 缺省列表) —— 用于 categories 的兜底
    categories_fallback: tuple[str, ...] | None = None
    #: 可选：apply 处理函数（POST /{id}/apply）
    apply: Callable[..., JSONResponse] | None = None
    #: 可选：(路由片段, 处理函数) —— from-* （如 /from-character/{character_id}）
    from_route: tuple[str, Callable[..., JSONResponse]] | None = None
    #: 可选：(路由片段, 处理函数) —— 额外的字面量 GET（如 weapon 无、scene 有 filter-options）
    extra_gets: tuple[tuple[str, Callable[..., JSONResponse]], ...] = ()


def _query_params(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.query_params.items()}


def _parse_library_query(q: dict[str, str]) -> dict[str, Any]:
    """对齐 TS ``parseLibraryQuery`` / ``parseQuery``。"""
    page = max(1, js_parse_int(q.get("page")) or 1)
    page_size = min(100, max(1, js_parse_int(q.get("pageSize")) or 20))
    raw_tags = q.get("tags") or ""
    return {
        "page": page,
        "page_size": page_size,
        "search": q.get("search") or "",
        "category": q.get("category") or "",
        "tags": [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else [],
        "sort_by": q.get("sortBy") or "updated_at",
        "sort_order": "asc" if (q.get("sortOrder") or "desc") == "asc" else "desc",
        "raw": q,
    }


def _decorate(row: dict[str, Any], spec: LibrarySpec) -> dict[str, Any]:
    """按 spec 解析 JSON 列并补 camelCase 别名（原地改 + 加别名，与原 TS 一致）。"""
    for col in spec.json_columns:
        if col in row:
            row[col] = safe_parse_json(row[col])
    for col in spec.out_aliases:
        if col in row:
            row[col if col in ("tags", "metadata") else camel(col)] = row[col]
    return row


def build_library_router(spec: LibrarySpec) -> APIRouter:
    """按 spec 生成一个资源库的 APIRouter（路由顺序即语义：字面量路径必须先注册）。"""
    router = APIRouter(prefix=spec.path, tags=[spec.table])

    # ------------------------------------------------------------------ GET /
    @router.get("")
    def list_items(request: Request, conn: Connection = Depends(get_conn)):
        try:
            q = _parse_library_query(_query_params(request))
            params: list[Any] = []
            where = "WHERE deleted_at IS NULL"

            if q["search"]:
                kw = f"%{q['search']}%"
                where += " AND (" + " OR ".join(f"{c} LIKE ?" for c in spec.search_columns) + ")"
                params.extend([kw] * len(spec.search_columns))

            for tag in q["tags"]:
                where += " AND tags LIKE ?"
                params.append(f"%{tag}%")

            if q["category"]:
                cats = [c for c in q["category"].split(",") if c]
                if cats:
                    where += " AND (" + " OR ".join("category = ?" for _ in cats) + ")"
                    params.extend(cats)

            for qkey, column in spec.exact_filters:
                value = q["raw"].get(qkey) or ""
                if value:
                    where += f" AND {column} = ?"
                    params.append(value)

            # 排序字段来自白名单，缺省 updated_at（同时挡掉了 SQL 注入）
            sort_field = q["sort_by"] if q["sort_by"] in spec.allowed_sorts else "updated_at"
            sort_dir = "ASC" if q["sort_order"] == "asc" else "DESC"

            total = q_one(conn, f"SELECT COUNT(*) AS total FROM {spec.table} {where}", params)
            total = (total["total"] if total else 0) or 0

            offset = (q["page"] - 1) * q["page_size"]
            rows = q_all(
                conn,
                f"SELECT * FROM {spec.table} {where} "
                f"ORDER BY {sort_field} {sort_dir} LIMIT ? OFFSET ?",
                (*params, q["page_size"], offset),
            )

            items = [_decorate(dict(r), spec) for r in rows]
            return lib_ok(
                {
                    "items": items,
                    "total": total,
                    "page": q["page"],
                    "pageSize": q["page_size"],
                    "totalPages": -(-total // q["page_size"]),  # 向上取整 = Math.ceil
                }
            )
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # ------------------------------------------------------------- GET /tags
    @router.get("/tags")
    def list_tags(conn: Connection = Depends(get_conn)):
        try:
            rows = q_all(
                conn,
                f"SELECT tags FROM {spec.table} WHERE deleted_at IS NULL AND tags IS NOT NULL",
            )
            tag_set: set[str] = set()
            for r in rows:
                parsed = safe_parse_json(r["tags"])
                if isinstance(parsed, list):
                    tag_set.update(str(t) for t in parsed)
            return lib_ok(sorted(tag_set))
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # ------------------------------------------------------- GET /categories
    @router.get("/categories")
    def list_categories(conn: Connection = Depends(get_conn)):
        try:
            rows = q_all(
                conn,
                f"SELECT DISTINCT category FROM {spec.table} "
                "WHERE deleted_at IS NULL ORDER BY category",
            )
            cats = [r["category"] for r in rows]
            # weapon 的 categories 有兜底，character/scene/costume 没有 —— 按 spec 区分
            if spec.categories_fallback and not cats:
                return lib_ok(list(spec.categories_fallback))
            return lib_ok(cats)
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # --------------------------------------------------- GET /filter-options
    if spec.filter_options is not None:
        builder = spec.filter_options

        @router.get("/filter-options")
        def filter_options(conn: Connection = Depends(get_conn)):
            try:
                return lib_ok(builder(conn))
            except Exception as exc:  # noqa: BLE001
                return lib_err(500, str(exc))

    # -------------------------------------------------------- POST /batch-delete
    # ⚠️ 必须注册在 GET/PUT/DELETE `/{item_id}` 之前（否则 batch-delete 会被当 id 吃掉）
    @router.post("/batch-delete")
    async def batch_delete(request: Request, conn: Connection = Depends(get_tx)):
        try:
            body = await read_json(request)
            ids = body.get("ids")
            if not isinstance(ids, list) or not ids:
                return lib_err(400, "请提供要删除的ID列表")
            placeholders = ",".join("?" for _ in ids)
            result = q_run(
                conn,
                f"UPDATE {spec.table} SET deleted_at = ? "
                f"WHERE id IN ({placeholders}) AND deleted_at IS NULL",
                (now(), *ids),
            )
            changes = result.rowcount or 0
            return lib_ok(
                {"deletedCount": changes}, f"成功删除 {changes} 个{spec.label}"
            )
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # ------------------------------------------------------------ apply / from-*
    if spec.apply is not None:
        apply_handler = spec.apply

        @router.post("/{item_id}/apply")
        async def apply_item(item_id: str, request: Request, conn: Connection = Depends(get_tx)):
            return apply_handler(conn, item_id, await read_json(request))

    if spec.from_route is not None:
        from_segment, from_handler = spec.from_route

        @router.post(from_segment)
        async def from_source(source_id: str, request: Request, conn: Connection = Depends(get_tx)):
            return from_handler(conn, source_id, await read_json(request))

    # --------------------------------------------------------------- GET /{id}
    @router.get("/{item_id}")
    def get_item(item_id: str, conn: Connection = Depends(get_conn)):
        try:
            iid = js_parse_int(item_id)
            if iid is None:  # parseInt 得 NaN ⇒ 400（注意不是 parseParamId 的语义）
                return lib_err(400, "无效的ID参数")
            row = q_one(
                conn, f"SELECT * FROM {spec.table} WHERE id = ? AND deleted_at IS NULL", (iid,)
            )
            if row is None:
                return lib_err(404, f"{spec.label}不存在")
            return lib_ok(_decorate(dict(row), spec))
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # ------------------------------------------------------------------ POST /
    @router.post("")
    async def create_item(request: Request, conn: Connection = Depends(get_tx)):
        try:
            body = await read_json(request)
            for key in spec.required:
                if not js_truthy(body.get(key)):
                    return lib_err(400, spec.required_message)

            values: dict[str, Any] = {}
            for column in spec.columns:
                key = camel(column)
                if column in spec.json_columns:
                    # TS 对 JSON 列不做 `||` 兜底，直接 safeStringify（undefined → ''）
                    values[column] = safe_stringify(body.get(key))
                elif column in _NULL_ON_CREATE:
                    values[column] = body.get(key) or None
                else:
                    values[column] = body.get(key) or spec.create_defaults.get(column, "")

            ts = now()
            values["created_at"] = ts
            values["updated_at"] = ts
            cols = ", ".join(values.keys())
            marks = ", ".join("?" for _ in values)
            result = q_run(
                conn, f"INSERT INTO {spec.table} ({cols}) VALUES ({marks})", tuple(values.values())
            )
            return lib_ok({"id": result.lastrowid}, "创建成功")
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # ------------------------------------------------------------- PUT /{id}
    @router.put("/{item_id}")
    async def update_item(item_id: str, request: Request, conn: Connection = Depends(get_tx)):
        try:
            iid = js_parse_int(item_id)
            if iid is None:
                return lib_err(400, "无效的ID参数")
            existing = q_one(
                conn, f"SELECT * FROM {spec.table} WHERE id = ? AND deleted_at IS NULL", (iid,)
            )
            if existing is None:
                return lib_err(404, f"{spec.label}不存在")

            body = await read_json(request)
            values: dict[str, Any] = {}
            for column in spec.columns:
                key = camel(column)
                if column in spec.json_columns:
                    # TS: `x !== undefined ? safeStringify(x) : existing` —— 传 null 会写 ''（清空）
                    values[column] = (
                        safe_stringify(body.get(key)) if key in body else existing[column]
                    )
                else:
                    # TS: `x ?? existing` —— 传 null 也保留原值
                    incoming = body.get(key)
                    values[column] = existing[column] if incoming is None else incoming
            values["updated_at"] = now()

            assignments = ", ".join(f"{c}=?" for c in values)
            q_run(
                conn,
                f"UPDATE {spec.table} SET {assignments} WHERE id=? AND deleted_at IS NULL",
                (*values.values(), iid),
            )
            return lib_ok({"id": iid}, "更新成功")
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    # ---------------------------------------------------------- DELETE /{id}
    @router.delete("/{item_id}")
    def delete_item(item_id: str, conn: Connection = Depends(get_tx)):
        try:
            iid = js_parse_int(item_id)
            if iid is None:
                return lib_err(400, "无效的ID参数")
            result = q_run(
                conn,
                f"UPDATE {spec.table} SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now(), iid),
            )
            if (result.rowcount or 0) == 0:
                return lib_err(404, f"{spec.label}不存在")
            return lib_ok({"id": iid}, "删除成功")
        except Exception as exc:  # noqa: BLE001
            return lib_err(500, str(exc))

    return router
