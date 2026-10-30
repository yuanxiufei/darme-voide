"""Ollama HTTP 客户端 —— 把 ``/api/*`` 收成一套**有类型的 Python 接口**（在线路径的唯一出口）。

⭐ 为什么在 ``ollama.py``（进程管理）与 ``ollama_store.py``（离线清单）之外再来一层：
那两块一个只会「拉/探进程」、一个只会「读盘」，**谁都不发业务请求**；而模型管理/推理的
每个调用点如果各自拼 ``httpx``，URL 前缀、超时、NDJSON 流解析、错误体解析就会各写一份
（本仓最忌讳的「一句话手抄多份」，见 ``run_all.py`` 里那条教训）。

⚠️ 三条硬约束：
1. **地址只从 :func:`app.services.ollama.resolve_ollama_url` 拿** —— 不在这里再写默认值 ✗。
   ``OLLAMA_HOST`` / 显式配置 / 默认值的优先级只有那一处真源。
2. **流式接口一律 NDJSON**（一行一个 JSON，最后一行 ``{"done": true}``）；中途一行坏了要
   **跳过这一行而不是整条流炸掉** ✓ —— 上游在 ``done_reason`` 之后偶尔会追加非 JSON 的
   stderr 片段，整体 ``json.loads`` 必失败 ✗。
3. **错误要带 HTTP 状态与上游原句**（``{"error": "..."}``）—— 把 400 变成
   ``OllamaError('model requires more system memory')`` 才能直接显示给用户 ✓。
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

from .ollama import resolve_ollama_url

__all__ = [
    "DEFAULT_TIMEOUT",
    "LONG_TIMEOUT",
    "OllamaError",
    "OllamaClient",
    "iter_ndjson",
    "normalize_show",
    "user_message",
]

#: 普通管理请求（列表/详情/删除）—— 本机 loopback，2 秒够；卡住就说明服务假死 ✓
DEFAULT_TIMEOUT = httpx.Timeout(5.0, connect=2.0)
#: 拉模型/推理 —— 可能几分钟到几十分钟，只限 connect（读不设上限）✓
LONG_TIMEOUT = httpx.Timeout(None, connect=5.0, read=None)


class OllamaError(RuntimeError):
    """Ollama 调用失败：``status`` 是 HTTP 状态（连接级失败为 ``None``），``detail`` 是上游原句。"""

    def __init__(self, message: str, *, status: int | None = None, detail: str = "", url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.url = url

    def as_dict(self) -> dict[str, Any]:
        return {"message": str(self), "status": self.status, "detail": self.detail, "url": self.url}


def _error_of(response: httpx.Response) -> OllamaError:
    """从失败响应里取 ``{"error": "..."}``；拿不到就用原文前 400 字（截断是为了日志不爆 ✓）。"""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, Mapping):
            detail = str(payload.get("error") or payload.get("message") or "")
    except ValueError:
        detail = ""
    if not detail:
        detail = (response.text or "")[:400]
    message = f"Ollama 返回 {response.status_code}" + (f"：{detail}" if detail else "")
    return OllamaError(message, status=response.status_code, detail=detail, url=str(response.url))


def iter_ndjson(line: str) -> Mapping[str, Any] | None:
    """解析**一行** NDJSON；空行/坏行/非对象 ⇒ ``None``（调用方跳过 ✓，不抛 ✗）。"""
    text = (line or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, Mapping) else None


class OllamaClient:
    """一个 Ollama 端点的客户端。**用完必须关**（``async with`` 或 ``await close()`` ✓）。

    ⚠️ 常驻一个 ``httpx.AsyncClient`` 而不是每次请求新建：拉模型时每行都新建连接会把
    ``localhost`` 打到 ``EADDRNOTAVAIL``（本机端口耗尽，实测过的老坑 ✓）。
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = resolve_ollama_url(base_url)
        self._client: httpx.AsyncClient | None = None

    # ── 生命周期 ──────────────────────────────────────────────────────────
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=DEFAULT_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    # ── 底层请求 ──────────────────────────────────────────────────────────
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: httpx.Timeout | None = None,
        raw_body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        client = self._http()
        try:
            response = await client.request(
                method,
                path,
                json=json_body,
                params=params,
                timeout=timeout or DEFAULT_TIMEOUT,
                content=raw_body,
                headers=headers,
            )
        except httpx.HTTPError as exc:                       # 连接级失败：服务没起/端口不对 ✓
            raise OllamaError(f"连不上 Ollama（{self.base_url}）：{exc}", url=self.url(path)) from exc
        if response.status_code >= 400:
            raise _error_of(response)
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {"raw": response.text}
        return payload if isinstance(payload, Mapping) else {"data": payload}

    async def _stream(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """NDJSON 流：逐行 yield，坏行跳过，``{"error": ...}`` 转异常 ✓。"""
        client = self._http()
        try:
            async with client.stream(
                "POST", path, json=dict(payload), timeout=timeout or LONG_TIMEOUT
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _error_of(response)
                async for line in response.aiter_lines():
                    item = iter_ndjson(line)
                    if item is None:
                        continue
                    if item.get("error"):
                        raise OllamaError(
                            f"Ollama 流内报错：{item.get('error')}",
                            status=response.status_code,
                            detail=str(item.get("error")),
                            url=self.url(path),
                        )
                    yield item
        except httpx.HTTPError as exc:
            raise OllamaError(f"连不上 Ollama（{self.base_url}）：{exc}", url=self.url(path)) from exc

    # ── 服务状态 ──────────────────────────────────────────────────────────
    async def version(self) -> dict[str, Any]:
        """``/api/version`` —— 顺带当「服务活着」的最轻探针（比 ``/api/tags`` 便宜 ✓）。"""
        return await self._request("GET", "/api/version")

    async def reachable(self) -> bool:
        try:
            await self.version()
            return True
        except OllamaError:
            return False

    # ── 模型管理 ──────────────────────────────────────────────────────────
    async def tags(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/tags")
        models = payload.get("models")
        return [m for m in models if isinstance(m, Mapping)] if isinstance(models, list) else []

    async def show(self, name: str) -> dict[str, Any]:
        """``/api/show`` —— 能力面/详情/模板/参数/投影器信息。**在线时是权威口径** ✓。"""
        return await self._request("POST", "/api/show", json_body={"model": name, "name": name})

    async def ps(self) -> list[dict[str, Any]]:
        """``/api/ps`` —— **已加载进显存**的模型（含 ``expires_at`` / ``size_vram``）✓。"""
        payload = await self._request("GET", "/api/ps")
        models = payload.get("models")
        return [m for m in models if isinstance(m, Mapping)] if isinstance(models, list) else []

    async def delete(self, name: str) -> None:
        await self._request("DELETE", "/api/delete", json_body={"model": name, "name": name})

    async def copy(self, source: str, destination: str) -> None:
        await self._request(
            "POST", "/api/copy", json_body={"source": source, "destination": destination}
        )

    async def pull(self, name: str, *, insecure: bool = False) -> AsyncIterator[Mapping[str, Any]]:
        """拉模型（流式进度：``status`` / ``completed`` / ``total``）✓。"""
        payload: dict[str, Any] = {"model": name, "name": name, "stream": True}
        if insecure:
            payload["insecure"] = True
        async for item in self._stream("/api/pull", payload, timeout=LONG_TIMEOUT):
            yield item

    async def create(self, name: str, modelfile: str) -> AsyncIterator[Mapping[str, Any]]:
        """按 Modelfile **建/改**一个模型（``ollama create`` 的 HTTP 形态，流式）✓。"""
        payload = {"model": name, "name": name, "modelfile": modelfile, "stream": True}
        async for item in self._stream("/api/create", payload, timeout=LONG_TIMEOUT):
            yield item

    # ── 推理 ──────────────────────────────────────────────────────────────
    async def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        format_schema: Mapping[str, Any] | str | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
    ) -> Any:
        """``/api/chat``。``stream=True`` 返回**异步生成器**，否则返回整包 dict ✓。

        ⚠️ 图片**不是**顶层字段：Ollama 只认 ``messages[i].images``（每条消息各自带图 ✓）。
        读视频＝抽帧后按 :func:`user_message` 组一条带多帧的用户消息塞进来 ——
        顶层放 ``images`` 会被服务端**静默忽略**（那一版实现看起来「跑通了」其实模型没看到图 ✗）。
        """
        payload: dict[str, Any] = {"model": model, "messages": list(messages), "stream": stream}
        if options:
            payload["options"] = dict(options)
        if tools:
            payload["tools"] = list(tools)
        if format_schema is not None:
            payload["format"] = format_schema
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think
        if not stream:
            return await self._request("POST", "/api/chat", json_body=payload, timeout=LONG_TIMEOUT)
        return self._stream("/api/chat", payload, timeout=LONG_TIMEOUT)

    async def embed(
        self,
        model: str,
        inputs: Sequence[str] | str,
        *,
        truncate: bool = True,
        dimensions: int | None = None,
        keep_alive: str | int | None = None,
    ) -> dict[str, Any]:
        """``/api/embed``（新式批量向量化；返回值里 ``embeddings`` 是二维列表 ✓）。"""
        payload: dict[str, Any] = {"model": model, "input": inputs, "truncate": truncate}
        if dimensions:
            payload["dimensions"] = dimensions
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return await self._request("POST", "/api/embed", json_body=payload, timeout=LONG_TIMEOUT)


def user_message(content: str, images: Sequence[str] | None = None) -> dict[str, Any]:
    """组一条用户消息；``images`` 是**裸 base64**（不带 ``data:`` 前缀，上游不认前缀 ✗）。

    ⚠️ 这个函数存在的唯一理由：图片必须挂在**消息**上。手写 ``{"images": [...]}`` 放在
    请求顶层是错的（见 :meth:`OllamaClient.chat` 的警告），而错的方式是**静默不生效** ——
    所以宁可把「对的那一种」做成唯一的构造函数 ✓。
    """
    message: dict[str, Any] = {"role": "user", "content": content}
    if images:
        message["images"] = [str(x) for x in images]
    return message


def normalize_show(payload: Mapping[str, Any]) -> dict[str, Any]:
    """把 ``/api/show`` 的响应压成前端要的那几项（**不改语义、只改形状** ✓）。

    ⚠️ 上游字段名有两套并存：新版 ``capabilities`` 是权威列表；老版只有 ``details.family``
    与 ``projector_info``。这里**以 ``capabilities`` 为准**，缺失时按 projector 在场补 vision
    （老版服务端就是这么判的 ✓），并把 ``details`` 摊平 —— 前端不该为了取一个参数量去猜层级。
    """
    details = payload.get("details") if isinstance(payload.get("details"), Mapping) else {}
    capabilities = payload.get("capabilities")
    caps = [str(c) for c in capabilities] if isinstance(capabilities, list) else []
    if not caps and payload.get("projector_info"):
        caps = ["vision"]
    model_info = payload.get("model_info") if isinstance(payload.get("model_info"), Mapping) else {}
    return {
        "capabilities": caps,
        "family": str(details.get("family") or ""),
        "families": list(details.get("families") or []) if isinstance(details.get("families"), list) else [],
        "parameterSize": str(details.get("parameter_size") or ""),
        "quantization": str(details.get("quantization_level") or ""),
        "format": str(details.get("format") or ""),
        "template": str(payload.get("template") or ""),
        "parameters": str(payload.get("parameters") or ""),
        "system": str(payload.get("system") or ""),
        "license": str(payload.get("license") or ""),
        "modifiedAt": str(payload.get("modified_at") or ""),
        "modelInfo": dict(model_info),
        "projector": dict(payload.get("projector_info") or {}) if isinstance(payload.get("projector_info"), Mapping) else {},
        "hasProjector": bool(payload.get("projector_info")) or "vision" in caps,
    }


def env_base_url() -> str:
    """当前生效的 Ollama 地址（**唯一真源**在 :func:`resolve_ollama_url`，这里只是别名）✓。"""
    return resolve_ollama_url(os.environ.get("OLLAMA_BASE_URL") or None)
