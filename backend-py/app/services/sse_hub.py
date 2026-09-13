"""SSE 进度推送事件总线 —— 与 ``utils/sse-hub.ts``（57 行）对齐。

对齐 PenguinHarness 第 16.4 节「SSE 端点的交付保证」，四条设计约束：

1. **内存 pub/sub，按 ``dramaId`` 分频道**（每个短剧生成任务一个频道）；
2. **发布端 fire-and-forget**：``publish_pipeline_event`` **永不抛**，订阅者 handler 的异常
   自吞 —— 「进度推送」这个簿记动作绝不能拖垮 auto-pipeline 主流程
   （对齐 16.3 节 `drive` 收尾不变量）；
3. **单进程单实例**，频道无订阅者时**自动清理**，避免内存泄漏；
4. **事件不落盘、不缓存**（历史状态由 ``GET /status`` 快照提供，SSE 只推增量 ——
   对齐 16.4 节「新订阅不回放 buffer」）。

⚠️ 结构性差异（Python 侧）：handler 仍是**同步**函数（与 TS 一致），SSE 路由把它桥到
``asyncio.Queue``；发布与路由在同一条事件循环上，故 ``put_nowait`` 安全。
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["publish_pipeline_event", "subscribe_pipeline"]

#: 事件处理器（同步）
Handler = Callable[[dict[str, Any]], None]

#: ``dramaId -> 订阅者集合``
_channels: dict[int, set[Handler]] = {}


def publish_pipeline_event(drama_id: int, event: dict[str, Any]) -> None:
    """发布进度事件到指定 drama 频道（**永不抛**，订阅者异常自吞）。

    ⚠️ 事件体里会补上 ``ts``（``new Date().toISOString()``），调用方只传 ``type`` 等业务字段。
    频道不存在 / 无订阅者时**直接返回**（不做任何缓存）。
    """
    subscribers = _channels.get(drama_id)
    if not subscribers:
        return
    from ..response import now as _now

    full = {**event, "ts": _now()}
    for handler in list(subscribers):
        try:
            handler(full)
        except Exception:  # noqa: BLE001 —— 单个订阅者异常不影响其他订阅者，也绝不抛回发布方
            pass


def subscribe_pipeline(drama_id: int, handler: Handler) -> Callable[[], None]:
    """订阅 drama 频道，返回**取消函数**（取消后若频道空了就删掉，防内存泄漏）。"""
    subscribers = _channels.get(drama_id)
    if subscribers is None:
        subscribers = set()
        _channels[drama_id] = subscribers
    subscribers.add(handler)

    def unsubscribe() -> None:
        subscribers.discard(handler)
        if not subscribers:
            _channels.pop(drama_id, None)

    return unsubscribe
