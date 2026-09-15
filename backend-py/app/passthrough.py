"""绞杀者接缝的共享实现 —— 「未迁移的请求怎么处理」。

两种用法：

1. **兜底**（``main.py`` 的 ``/api/v1/{path}``、``/webhooks/{path}``）：整段前缀下方
   未被任何已迁移路由命中的请求走这里。
2. **显式委派**（各 router 里声明）：某些未迁移的路径**会被已注册的参数路由遮蔽**
   —— 典型例子 ``GET /agent-configs/defaults`` 会被 ``GET /agent-configs/{id}`` 吞掉，
   永远到不了兜底。这类路径必须在 router 里**显式、且注册在参数路由之前**声明委派。

⚠️ 这是本项目最容易静默出错的一类问题：被遮蔽的端点不会报错，只会返回
「Invalid xxx id」这类看起来合理的 404，从而**悄悄失去 Node 的实现**。
⇒ ``tests/route_parity_test.py`` 会系统性扫描全部 TS 路径来找这类遮蔽，别再靠肉眼。
"""

from __future__ import annotations

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .config import NODE_BACKEND_URL, PROXY_TO_NODE

#: 逐跳首部（RFC 7230）：转发时必须剥掉，否则会污染下游连接语义
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

#: 反代到 Node 的共享客户端（连接池复用；SSE 长连接也走它）
_proxy_client: httpx.AsyncClient | None = None


def init_proxy_client() -> None:
    """进程启动时调用（``main.py`` 的 lifespan）。"""
    global _proxy_client
    if PROXY_TO_NODE and _proxy_client is None:
        # read=None：SSE / 长轮询不能有读超时
        _proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))


async def close_proxy_client() -> None:
    global _proxy_client
    if _proxy_client is not None:
        await _proxy_client.aclose()
        _proxy_client = None


async def proxy_to_node(request: Request, path: str) -> Response:
    """把请求原样转发给 Node 后端（路径、查询串、方法、请求体、首部全透传）。"""
    assert _proxy_client is not None
    target = f"{NODE_BACKEND_URL}/{path}"
    headers = [
        (k, v)
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() != "host"
    ]
    try:
        upstream_request = _proxy_client.build_request(
            request.method,
            target,
            headers=headers,
            content=await request.body(),
            params=list(request.query_params.multi_items()),
        )
        upstream = await _proxy_client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "code": 502,
                "message": (
                    f"反代到上游后端失败（{NODE_BACKEND_URL}）：{exc}。"
                    "⚠️ Node 后端已于 2026-09-15 删除 ⇒ 除非你自己起了别的上游，"
                    "否则请把 PROXY_TO_NODE 设为 0：未实现的路径会回 501 并说明原因。"
                ),
            },
        )

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    # aiter_raw 保证 SSE 不被缓冲（管线进度流依赖），且透传压缩体与 content-encoding 一致
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=resp_headers,
        background=BackgroundTask(upstream.aclose),
    )


def not_migrated(path: str) -> JSONResponse:
    """**未实现**路径的明确回执（比静默 404 好排查）。

    ⚠️ 2026-09-15 改措辞：Node 后端**已删除** ⇒ 原来那句「设置 ``PROXY_TO_NODE=1`` 可反代到
    Node 保持可用」已**不可能成立**（照做只会去起一个不存在的服务）。现在如实说明「这条路径
    没有实现」，并指出该去哪儿补。
    """
    return JSONResponse(
        status_code=501,
        content={
            "code": 501,
            "message": (
                f"/{path} 未实现（Node 后端已于 2026-09-15 删除）。"
                "如需该能力，请在 backend-py/app/routers/ 下补实现。"
            ),
        },
    )


async def delegate(request: Request, path: str) -> Response:
    """router 里为「被遮蔽的未迁移路径」显式声明的委派入口。

    开启反代时转发给 Node；否则回 501 并说明未迁移。
    """
    if not PROXY_TO_NODE:
        return not_migrated(path)
    return await proxy_to_node(request, path)
