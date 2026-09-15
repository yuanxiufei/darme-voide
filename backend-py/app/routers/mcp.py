"""MCP 域 —— 与 ``backend/src/routes/mcp.ts``（27 行）对齐。**3 个端点**。

* ``GET  /api/v1/mcp/status``  已配置 / 已连接的 server 与工具清单；
* ``POST /api/v1/mcp/refresh`` 关连接并重新发现，返回刷新后的状态；
* ``POST /api/v1/mcp/test``    测试连接单个 server（临时连接，测完即关）。

⚠️ 三处保真点（都「看着像 bug，但就是原样」）：

1. **两个 POST 的错误信封不是本项目的错误信封**：它们是 ``c.json({code: 500, data: null,
   message})`` ⇒ **HTTP 500 且带 ``data: null``**（而 ``server_error`` 那种是 500 但**无 data 键**）。
   前端按这个结构解析，别"顺手统一"；
2. ``POST /test`` 的 body 用 ``await c.req.json()``（**在 try 内**）⇒ 坏 JSON 会走 catch
   变成 500（**不是** 400）—— 与 ``read_json`` 那套"宽容空 body"**刻意不同**；
3. ``POST /test`` 的 ``name`` 只判 **truthy**：``{"name": 123}`` **不会被 400 拒**，而是交给
   ``test_mcp_server`` 返回 ``{ok: false, error: 'server name is required'}`` 的 **200** 成功信封。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..core.response import bad_request, success
from app.mcp.client import get_mcp_status, refresh_mcp, test_mcp_server

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])


def _inner_error(message: str) -> JSONResponse:
    """``c.json({code: 500, data: null, message})`` —— **带 data: null 的 500**（见模块头第 1 点）。"""
    return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": message})


@router.get("/status")
def mcp_status():
    """查看已配置与已连接的 MCP server 及工具。"""
    return success(get_mcp_status())


@router.post("/refresh")
async def mcp_refresh():
    """关闭所有连接并重新发现。"""
    try:
        await refresh_mcp()
        return success(get_mcp_status())
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        return _inner_error(str(err))


@router.post("/test")
async def mcp_test(request: Request):
    """测试连接单个 server（独立临时连接，测完即关，不写缓存）。

    ⚠️ body 直接 ``request.json()``（不宽容）：坏 JSON ⇒ 走 catch ⇒ **500**。
    """
    try:
        body: Any = await request.json()
        if not isinstance(body, dict) or not body.get("name"):
            return bad_request("server name is required")
        return success(await test_mcp_server(body))
    except Exception as err:  # noqa: BLE001
        return _inner_error(str(err))
