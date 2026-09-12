"""请求读取小工具（对齐 Hono 的 ``c.req.json()`` 容错行为）。"""

from __future__ import annotations

from typing import Any

from fastapi import Request


async def read_json(request: Request) -> dict[str, Any]:
    """宽容读取请求体：非法 JSON / 非对象一律返回空 dict。

    TS 侧 ``c.req.json()`` 会抛错并被各路由的 catch 转成 400；Python 侧走空 dict 后由
    业务校验（必填字段 / NOT NULL）给出 400，**最终状态码一致**，且不会把
    「客户端发了坏 JSON」升级成 500。
    """
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - 对齐 TS：任何解析异常都不该 500
        return {}
    return data if isinstance(data, dict) else {}
