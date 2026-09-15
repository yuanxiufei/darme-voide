"""MCP 客户端（S5 从后端服务里独立出来）—— 顶层包，与 `skills/` / `scripts/` 同级。

⚠️ **不要**把本目录命名为第三方 SDK 的包名（如官方 `mcp`）—— 这里就是 MCP 客户端本体：
手写 JSON-RPC over HTTP（:mod:`mcp.client`），不依赖任何 MCP SDK。若将来引入官方 SDK，
必须先解决同名遮蔽（本项目 `backend-py/` 在 sys.path 首位 ⇒ 本目录会**遮蔽**同名第三方包）。

依赖方向（单向链，由 ``tests/layering_test.py`` 机械守卫）：

    app/routers  →  agent  →  app/services  →  app/core
                        ↘  mcp  ↗（mcp 只依赖 app/core）
"""

from .client import discover_mcp_tools, get_mcp_status, refresh_mcp, test_mcp_server

__all__ = ["discover_mcp_tools", "get_mcp_status", "refresh_mcp", "test_mcp_server"]
