"""ComfyUI 进度 / 预览桥 —— 移植上游 ``comfyui-gateway/src/preview-hub.js``（253 行，MIT）的**语义**。

## 源到底解决了什么

ComfyUI 的 ``/ws`` 会推四类消息：``progress_state``（按节点上报 value/max）、
``kj_preview_override``（采样中途的预览帧，base64）、``execution_success/error``、``execution_interrupted``。
源把它们收拢成「一条任务的一个百分比 + 一张最新预览图」，前端只读这个结果 ✓。
没有它，前端只能对着 ``/history`` 干等——**采样到 90% 和卡死在 9% 看起来一模一样** ✗。

## 保真的三条（照搬，不许"优化"）

1. ⭐ **进度按节点 value/max 加权，不是按节点数平均** ✓：
   放大链 + 采样器两段并行时，按节点平均会出现「40% ⇒ 100% ⇒ 掉回 60%」✗。
2. ⭐ **预览帧只留最新一张 + TTL** ✓：预览帧是 base64（单帧几十到几百 KB），
   存历史会几百 MB 地涨 ✗；TTL 与源一致 10 分钟 ✓。
3. ⭐ **收尾必须显式三态** ``done`` / ``error`` / ``cancelled`` ✓：
   只推进度不收尾，前端会永远转圈（源里 ``execution_interrupted`` 也算收尾 ✓）。

## 三处适配（必须改，否则装不进本仓）

1. **浏览器 WebSocket → 服务端可选依赖**：源的 ``new WebSocket`` 在 Node/浏览器里是内置的，
   Python 侧要 ``websockets``，而本仓 ``requirements.txt`` **只有 httpx** ✓。
   ⇒ 用 ``importlib.util.find_spec`` 探测：有就订阅（真预览 ✓），没有就只推轮询状态，
   并在事件里如实写 ``progressAvailable=False``（**绝不假装有进度** ✗）。
2. **内存 Map 由 HTTP 拉 → 自建按 ``runId`` 的小总线**：源的 ``get(promptId)`` 靠前端轮询；
   本仓已有 ``sse_hub``，但它**按 dramaId 分频道**，而这里要按 runId ⇒ 另建一条，
   不动既有总线（改它会把 auto_pipeline 的订阅语义搅乱 ✗）。
3. **不推断 ``plan.steps``**：源的计划表来自它自己的 runner（``plan.steps`` / ``plan.absolute``）。
   本仓拿不到那份计划，直接用消息里的 ``value/max``——
   单采样器时与源的加权**完全等价**；多采样器时 ``sum(value)/sum(max)`` 反而比"按计划归一"更准 ✓。
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import time
from typing import Any, Callable

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "ProgressTracker",
    "parse_message",
    "percent_of",
    "publish",
    "snapshot",
    "subscribe",
    "watch_websocket",
    "websockets_available",
]

#: 与源 ``DEFAULT_TTL_MS`` 一致 ✓：预览帧与进度记录都留 10 分钟
DEFAULT_TTL_SECONDS = 10 * 60

#: 源的两类自定义消息（``kj_preview_override`` 来自 KJNodes 的预览扩展 ✓）
PREVIEW_EVENT = "kj_preview_override"
PROGRESS_EVENT = "progress_state"

Handler = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# 纯函数层（无 IO、可单测 ✓）
# ---------------------------------------------------------------------------

def percent_of(nodes: Any) -> tuple[int, int, float]:
    """把 ``progress_state.nodes`` 折成 ``(step, total, percent)`` ✓。

    * 只认**同时有 value 与 max（且 max > 0）**的节点 ✓ —— 缺 max 的节点（纯 IO 节点）不参与，
      否则 ``value/0`` 会让整条进度变成 ``inf`` ✗；
    * 多个节点**先各自累加再相除** ✓（按节点平均是错的，见模块头保真点 1）；
    * ``percent`` 保留 1 位小数 ✓（前端直接显示，不自己 round）。
    """
    total = 0
    step = 0
    if isinstance(nodes, dict):
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            try:
                value = float(node.get("value"))
                maximum = float(node.get("max"))
            except (TypeError, ValueError):
                continue
            if maximum <= 0 or value < 0:
                continue
            total += int(maximum)
            step += int(min(value, maximum))
    percent = round(step * 100.0 / total, 1) if total > 0 else 0.0
    return step, total, percent


def parse_message(raw: Any, run_id: int | None = None) -> dict[str, Any] | None:
    """ComfyUI ``/ws`` 的一条原始消息 → 归一化事件（不认识就返 ``None`` ✓）。"""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(raw, str):
        return None
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(message, dict):
        return None

    kind = str(message.get("type") or "")
    data = message.get("data") if isinstance(message.get("data"), dict) else {}
    prompt_id = str(data.get("prompt_id") or "") or None

    if kind == PROGRESS_EVENT:
        step, total, percent = percent_of(data.get("nodes"))
        return {
            "type": "progress", "runId": run_id, "promptId": prompt_id,
            "step": step, "total": total, "percent": percent,
        }

    if kind == PREVIEW_EVENT:
        image = data.get("image")
        if not image:
            return None
        mime = str(data.get("mime") or "image/jpeg")
        return {
            "type": "preview", "runId": run_id, "promptId": prompt_id,
            # 源的预览就是裸 base64；这里补成 data URL ✓（前端 `<img src>` 可直接吃 ✓）
            "image": f"data:{mime};base64,{image}",
            "mime": mime,
            "nodeId": data.get("node_id"),
        }

    if kind == "execution_success":
        return {"type": "done", "runId": run_id, "promptId": prompt_id}
    if kind == "execution_error":
        return {"type": "error", "runId": run_id, "promptId": prompt_id,
                "message": str(data.get("exception_message") or data.get("status_str") or "上游执行失败")}
    if kind == "execution_interrupted":
        return {"type": "cancelled", "runId": run_id, "promptId": prompt_id}

    # ``executing`` / ``status`` / ``execution_start`` 等：**不产生事件**
    # （它们的信息量已被 progress_state 覆盖；多推一条只会让前端多一次无意义重渲染 ✗）
    return None


def websockets_available() -> bool:
    """``websockets`` 装了没 ✓（本仓默认没装 ⇒ 预览降级，但**进度桥本身照常可用** ✓）。"""
    try:
        return importlib.util.find_spec("websockets") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 运行时：按 runId 的进度表 + 订阅总线
# ---------------------------------------------------------------------------

class ProgressTracker:
    """一条 run 的进度/预览快照 + 订阅者（**单进程内存** ✓，与 ``sse_hub`` 同口径 ✓）。"""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._state: dict[int, dict[str, Any]] = {}
        self._subscribers: dict[int, set[Handler]] = {}

    def apply(self, run_id: int, event: dict[str, Any]) -> dict[str, Any]:
        """合并一条事件进快照并**广播给订阅者**（订阅者异常自吞 ✓，绝不打断上游轮询 ✓）。"""
        record = self._state.get(run_id)
        if record is None:
            record = {"runId": run_id, "percent": 0.0, "step": 0, "total": 0,
                      "state": "queued", "preview": None, "updatedAt": 0.0}
            self._state[run_id] = record

        kind = event.get("type")
        if kind == "progress":
            record["percent"] = event.get("percent") or 0.0
            record["step"] = event.get("step") or 0
            record["total"] = event.get("total") or 0
            record["state"] = "running"
        elif kind == "preview":
            # 只留最新一帧 ✓（历史预览帧没有消费方，攒着只是涨内存 ✗）
            record["preview"] = {"image": event.get("image"), "nodeId": event.get("nodeId")}
            record["state"] = "running"
        elif kind in ("done", "error", "cancelled"):
            record["state"] = {"done": "succeeded", "error": "failed", "cancelled": "cancelled"}[kind]
            # 收尾时把进度拉满 ✓（差 2% 停在 98% 收尾，看起来像失败了 ✗）
            if kind == "done":
                record["percent"] = 100.0
                record["step"] = record.get("total") or record.get("step") or 0
        elif kind == "poll":
            record["pullStatus"] = event.get("status")
            if record["state"] in ("queued", "running"):
                record["state"] = "running" if event.get("status") == "processing" else record["state"]
        record["updatedAt"] = time.monotonic()

        full = dict(event)
        full.setdefault("runId", run_id)
        self._broadcast(run_id, full)
        return full

    def snapshot(self, run_id: int) -> dict[str, Any]:
        """当前快照（过期返 ``None`` 语义的 ``{}`` ✓）。"""
        record = self._state.get(run_id)
        if record is None:
            return {}
        if time.monotonic() - float(record.get("updatedAt") or 0) > self._ttl:
            self._state.pop(run_id, None)
            return {}
        return dict(record)

    def subscribe(self, run_id: int, handler: Handler) -> Callable[[], None]:
        handlers = self._subscribers.setdefault(run_id, set())
        handlers.add(handler)

        def unsubscribe() -> None:
            handlers.discard(handler)
            if not handlers:
                self._subscribers.pop(run_id, None)

        return unsubscribe

    # ⚠️ 刻意**不设**后台清理定时器（源用 ``setInterval(closeIdle)``）：
    #    本仓 ``sse_hub`` 也没有定时器 ✓，过期记录在 :meth:`snapshot` 读取时惰性清 ✓ ——
    #    加一个常驻 ``asyncio`` 任务只为清几个字典，是拿生命周期风险换内存（✗ 进程退出还要收尾）。

    def _broadcast(self, run_id: int, event: dict[str, Any]) -> None:
        for handler in list(self._subscribers.get(run_id) or ()):
            try:
                handler(event)
            except Exception:  # noqa: BLE001 —— 与 sse_hub 同口径：推送簿记不许拖垮主流程 ✓
                pass


#: 进程级单例（与 ``sse_hub`` 一样是模块级状态 ✓）
_tracker = ProgressTracker()


def publish(run_id: int, event: dict[str, Any]) -> dict[str, Any]:
    """把事件并进快照并广播 ✓（**永不抛** ✓）。"""
    try:
        return _tracker.apply(run_id, event)
    except Exception:  # noqa: BLE001
        return dict(event)


def snapshot(run_id: int) -> dict[str, Any]:
    return _tracker.snapshot(run_id)


def subscribe(run_id: int, handler: Handler) -> Callable[[], None]:
    return _tracker.subscribe(run_id, handler)


# ---------------------------------------------------------------------------
# 可选：ComfyUI WS 预览订阅
# ---------------------------------------------------------------------------

async def watch_websocket(
    ws_url: str,
    prompt_id: str | None,
    run_id: int,
    *,
    timeout_seconds: float = 1800,
) -> bool:
    """订阅 ComfyUI ``/ws`` 直到收尾或超时 ✓；返回「是否真的连上并按协议走完」✓。

    ⚠️ 三种情况都**不算失败**，返回 ``False`` 但**不抛** ✓：
    * 没装 ``websockets``（本仓默认）⇒ 调用方只推轮询状态 ✓；
    * 连不上（ComfyUI 在别处 / 端口不通）✓；
    * 上游不发 ``progress_state``（版本旧）✓。
    这样"预览"是**加分项**，不是运行的前置条件 ✓。
    """
    if not websockets_available():
        return False
    try:
        import websockets  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 —— find_spec 说有没有时，真 import 仍可能因版本崩 ✗
        return False

    client_id = f"darme-{run_id}-{int(time.time() * 1000)}"
    url = f"{ws_url.rstrip('/')}/ws?clientId={client_id}"
    deadline = time.monotonic() + timeout_seconds
    connected = False
    try:
        async with websockets.connect(url, max_size=32 * 1024 * 1024) as socket:
            connected = True
            # 源同口径：先声明支持预览元数据，KJNodes 才会带 mime/step ✓
            try:
                await socket.send(json.dumps(
                    {"type": "feature_flags", "data": {"supports_preview_metadata": True}},
                    separators=(",", ":")))  # 紧凑 ✓ 与 Node 的 JSON.stringify 字节一致（route_parity 在盯 ✓）
            except Exception:  # noqa: BLE001 —— 协商失败不影响进度 ✓
                pass
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=30)
                except asyncio.TimeoutError:
                    continue
                event = parse_message(raw, run_id)
                if event is None:
                    continue
                # 只认自己那条 prompt ✓（ComfyUI 的 /ws 是**全局广播**：
                # 不筛的话，别人的任务会把我们的进度覆盖成 3% ✗）
                if prompt_id and event.get("promptId") and event["promptId"] != prompt_id:
                    continue
                publish(run_id, event)
                if event.get("type") in ("done", "error", "cancelled"):
                    return connected
    except Exception:  # noqa: BLE001 —— 连不上/断开都属于"降级"，不是运行失败 ✓
        return connected
    return connected
