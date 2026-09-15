"""MCP（Model Context Protocol）工具接入层 —— 与 ``agents/mcp.ts``（285 行）对齐。

设计精髓（单一事实来源：外部工具生态接入，**单点故障绝不拖垮整个 Agent Session**）：

1. **三传输推断**：显式 ``transport`` → ``command`` 则 stdio → 否则 http；
2. **容错**：非法条目只告警跳过、连接失败只降级、**永不 throw** —— 外部工具挂了不影响主流程；
3. **命名隔离**：``mcp__<server>__<tool>`` 三段式扁平命名空间，与业务工具零冲突；
4. **懒连接 single-flight**：会话组装时才连接，所有 server **并行**连接，连接 + 握手 + 发现
   共用**一个** ``connectTimeoutMs`` 预算（单一计时器，不做多层超时叠加）；
5. **权限方向**：MCP 的 ``readOnlyHint`` 是**不可信 hint**，本层默认按更严格的 rw 处理（不据此放行）。

配置来源：``configs/mcp-servers.json``（JSON 数组，可用 ``MCP_SERVERS_PATH`` 覆盖）。
文件不存在或解析失败 ⇒ 空列表，对现有 Agent **零影响**。

⚠️ **刻意的实现差异（重要）**：Node 用官方 ``@modelcontextprotocol/sdk``，而 Python 侧
**没有装 ``mcp`` 包**（也刻意不新增依赖）⇒ 这里自写**极简 JSON-RPC 客户端**，只覆盖本层用到的
三个方法：``initialize`` / ``tools/list`` / ``tools/call``（外加握手后的
``notifications/initialized``）。已实现 **stdio** 与 **streamable HTTP**（含 ``Mcp-Session-Id``
回传）；**HTTP+SSE（旧传输）未移植**：显式声明 ``sse`` 时**显式报错**（不静默降级成 http）。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from app.core.config import PROJECT_ROOT
from app.agent.tool import Tool

__all__ = [
    "MCP_TOOL_PREFIX",
    "discover_mcp_tools",
    "get_mcp_status",
    "infer_transport",
    "load_mcp_servers",
    "refresh_mcp",
    "sanitize_segment",
    "serialize_tool_result",
    "test_mcp_server",
]

#: 命名隔离前缀（``mcp__<server>__<tool>``）
MCP_TOOL_PREFIX = "mcp__"

#: 连接 + 握手 + 发现共用的**单一**超时预算
DEFAULT_CONNECT_TIMEOUT_MS = 10_000

#: MCP 协议版本（写死即可：本层只做工具发现与调用）
_PROTOCOL_VERSION = "2024-11-05"

#: 客户端自报身份（与 TS 一致）
_CLIENT_INFO = {"name": "drama-studio", "version": "1.0.0"}


def mcp_config_path() -> Path:
    """配置路径：``MCP_SERVERS_PATH`` 优先，否则 ``<项目根>/configs/mcp-servers.json``。"""
    override = os.environ.get("MCP_SERVERS_PATH")
    if override:
        return Path(override)
    return PROJECT_ROOT / "configs" / "mcp-servers.json"


def load_mcp_servers() -> list[dict[str, Any]]:
    """读取并净化配置：**非法条目跳过**、``enabled === false`` 跳过、名字 trim。

    文件缺失 / 解析失败 / 非数组 ⇒ 空列表（⚠️ 不抛异常，这是「零影响」承诺的实现）。
    """
    try:
        raw = json.loads(mcp_config_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 与 TS 的 catch 等价
        return []
    if not isinstance(raw, list):
        return []
    servers: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) \
                or not item["name"].strip():
            print(f"[mcp] invalid server entry skipped (missing name): "
                  f"{json.dumps(item, ensure_ascii=False, separators=(',', ':'))}")
            continue
        if item.get("enabled") is False:
            continue
        servers.append({**item, "name": item["name"].strip()})
    return servers


def infer_transport(cfg: dict[str, Any]) -> str:
    """``transport`` 显式优先 → 有 ``command`` 走 stdio → 否则 http。"""
    if cfg.get("transport"):
        return str(cfg["transport"])
    if cfg.get("command"):
        return "stdio"
    return "http"


def sanitize_segment(s: str) -> str:
    """净化工具名/服务名，仅保留 ``[a-zA-Z0-9_-]``（其余换 ``_``），避免工具 id 非法。"""
    return "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "_-")) else "_" for ch in s)


def serialize_tool_result(result: Any) -> dict[str, Any]:
    """把 MCP 工具结果序列化成 LLM 可读结构（``structuredContent`` → ``content`` → ``raw``）。"""
    result = result if isinstance(result, dict) else {}
    is_error = bool(result.get("isError"))
    if result.get("structuredContent"):
        return {**result["structuredContent"], "isError": is_error}
    content = result.get("content")
    if isinstance(content, list):
        texts = [item.get("text") for item in content
                 if isinstance(item, dict) and item.get("type") == "text"
                 and isinstance(item.get("text"), str)]
        if texts:
            return {"text": "\n".join(texts), "isError": is_error}
    return {"raw": json.dumps(result if result else None,
                              ensure_ascii=False, separators=(",", ":")), "isError": is_error}


async def _with_timeout(awaitable: Any, ms: int, label: str) -> Any:
    """单一计时器超时（连接/握手/发现**共用**一个预算）。"""
    try:
        return await asyncio.wait_for(awaitable, timeout=ms / 1000)
    except asyncio.TimeoutError:
        raise ValueError(f"{label} timed out after {ms}ms") from None


class McpClient:
    """极简 JSON-RPC 客户端基类（见模块头：只覆盖三个方法）。"""

    async def connect(self) -> None:
        raise NotImplementedError

    async def list_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class StdioMcpClient(McpClient):
    """stdio 传输：子进程 + **按行分隔的 JSON-RPC**（MCP stdio 的线格式）。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0

    async def connect(self) -> None:
        command = self.cfg.get("command")
        if not command:
            raise ValueError(f"MCP \"{self.cfg['name']}\": stdio transport missing command")
        env = {**os.environ, **(self.cfg.get("env") or {})}
        self._proc = await asyncio.create_subprocess_exec(
            str(command), *[str(a) for a in (self.cfg.get("args") or [])],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # ⚠️ 与 TS 的 `stderr: 'ignore'` 一致：外部 server 的噪声不该污染本服务日志
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.cfg.get("cwd"),
            env=env,
        )
        await self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        })
        self._notify("notifications/initialized")

    def _notify(self, method: str) -> None:
        """通知（无 id、不等回复）。"""
        if self._proc is None or self._proc.stdin is None:
            return
        payload = {"jsonrpc": "2.0", "method": method}
        self._proc.stdin.write(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        )

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise ValueError("MCP stdio transport is not connected")
        self._next_id += 1
        request_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self._proc.stdin.write(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        )
        await self._proc.stdin.drain()

        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise ValueError(f"MCP \"{self.cfg['name']}\": stdio closed unexpectedly")
            try:
                message = json.loads(line)
            except ValueError:  # 非 JSON 行（噪声）跳过
                continue
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue  # 通知 / 他人响应
            if message.get("error"):
                raise ValueError(f"MCP \"{self.cfg['name']}\": {message['error']}")
            return message.get("result") or {}

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list", {})
        return list(result.get("tools") or [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._request("tools/call", {"name": name, "arguments": arguments})

    async def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.terminate()
        except Exception:  # noqa: BLE001 —— 关连接失败不该影响主流程
            pass
        self._proc = None


class HttpMcpClient(McpClient):
    """streamable HTTP 传输：POST JSON-RPC；``initialize`` 回的 ``Mcp-Session-Id`` 要带上。

    ⚠️ 用 ``httpx`` 直连（与 ``file_storage`` 同先例）：MCP 连接失败**只降级**，
    不需要厂商重试语义。
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._next_id = 0
        self._session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(self.cfg.get("headers") or {}),
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _post(self, payload: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
        url = self.cfg.get("url")
        if not url:
            raise ValueError(f"MCP \"{self.cfg['name']}\": http transport missing url")
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_ms / 1000)) as client:
            response = await client.post(
                str(url), headers=self._headers(),
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
        session = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
        if session:
            self._session_id = session
        if response.status_code >= 400:
            raise ValueError(f"MCP \"{self.cfg['name']}\": HTTP {response.status_code} "
                             f"{response.text[:200]}")
        return _parse_http_body(response.text)

    async def _call(self, method: str, params: dict[str, Any] | None = None,
                    *, notify: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._next_id += 1
            payload["id"] = self._next_id
        if params is not None:
            payload["params"] = params
        return await self._post(payload, DEFAULT_CONNECT_TIMEOUT_MS)

    async def connect(self) -> None:
        await self._call("initialize", {
            "protocolVersion": _PROTOCOL_VERSION, "capabilities": {}, "clientInfo": _CLIENT_INFO,
        })
        await self._call("notifications/initialized", None, notify=True)

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._call("tools/list", {})
        return list(result.get("tools") or [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._call("tools/call", {"name": name, "arguments": arguments})


def _parse_http_body(text: str) -> dict[str, Any]:
    """响应体可能是 JSON，也可能是 **SSE**（``data: {...}`` 行）—— 两种都认。"""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return {}
        return parsed.get("result") or {} if isinstance(parsed, dict) else {}
    for line in stripped.splitlines():
        if line.startswith("data:"):
            try:
                parsed = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed.get("result") or {}
    return {}


def _build_client(cfg: dict[str, Any]) -> McpClient:
    """按传输类型造客户端。⚠️ ``sse``（HTTP+SSE 旧传输）**未移植** ⇒ 显式报错。"""
    transport = infer_transport(cfg)
    if transport == "stdio":
        return StdioMcpClient(cfg)
    if transport == "sse":
        raise ValueError(
            f"MCP \"{cfg.get('name')}\": sse 传输（HTTP+SSE）未移植，请改用 stdio 或 streamable http"
        )
    return HttpMcpClient(cfg)


#: 已连接 server：``name -> {config, client, tools}``
_connected_servers: dict[str, dict[str, Any]] = {}

#: 当前配置指纹（检测配置变更失效缓存）
_cache_fingerprint: str = ""

#: single-flight：并发调用共享同一次发现
_discovery_in_flight: "asyncio.Task[dict[str, Tool]] | None" = None


def _to_tool(server_name: str, client: McpClient, mcp_tool: dict[str, Any]) -> Tool:
    """把一个 MCP tool 转成本层 ``Tool``（命名隔离 + **宽松入参透传**）。"""
    full_name = f"{MCP_TOOL_PREFIX}{sanitize_segment(server_name)}__" \
                f"{sanitize_segment(str(mcp_tool.get('name')))}"
    source_desc = (mcp_tool.get("description") or "").strip()
    description = (f"[MCP/{server_name}] {source_desc}" if source_desc
                   else f"[MCP/{server_name}] external tool {mcp_tool.get('name')}")

    async def _execute(args: dict[str, Any] | None = None) -> dict[str, Any]:
        result = await client.call_tool(str(mcp_tool.get("name")), (args or {}))
        return serialize_tool_result(result)

    return Tool(
        id=full_name,
        description=description,
        # ⚠️ **不逐字段还原 JSON Schema**（转换脆弱）：宽松对象，参数透传给 MCP server 校验
        input_schema={"type": "object", "additionalProperties": True},
        execute=_execute,
    )


async def _connect_server(cfg: dict[str, Any]) -> None:
    """连接单个 server 并发现工具（失败抛出，由 ``gather`` 兜底降级）。"""
    client = _build_client(cfg)
    timeout_ms = int(cfg.get("connectTimeoutMs") or DEFAULT_CONNECT_TIMEOUT_MS)
    name = cfg["name"]
    try:
        await _with_timeout(client.connect(), timeout_ms, f"MCP \"{name}\" connect")
        listed = await _with_timeout(client.list_tools(), timeout_ms, f"MCP \"{name}\" listTools")
        tools: dict[str, Tool] = {}
        for item in listed:
            if not isinstance(item, dict):
                continue
            tool = _to_tool(name, client, item)
            if tool.id in tools:
                print(f'[mcp] duplicate tool "{tool.id}" skipped')
                continue
            tools[tool.id] = tool
        _connected_servers[name] = {"config": cfg, "client": client, "tools": tools}
    except BaseException:
        # 连接失败：关闭残留连接，向上抛给调用方降级
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
        raise


def _merge_cached_tools() -> dict[str, Tool]:
    """合并所有已连接 server 的工具（**重名跳过后者**，与 TS 一致）。"""
    merged: dict[str, Tool] = {}
    for server in _connected_servers.values():
        for key, tool in server["tools"].items():
            if key in merged:
                print(f'[mcp] tool name collision "{key}" — latter skipped')
                continue
            merged[key] = tool
    return merged


async def _close_all_servers() -> None:
    for server in list(_connected_servers.values()):
        try:
            await server["client"].close()
        except Exception:  # noqa: BLE001
            pass
    _connected_servers.clear()


async def _do_discover() -> dict[str, Tool]:
    servers = load_mcp_servers()
    fingerprint = json.dumps(servers, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    global _cache_fingerprint
    # 配置未变且已连接过 → 直接复用缓存
    if fingerprint == _cache_fingerprint:
        return _merge_cached_tools()

    # 配置变更或首次：关闭旧连接，重建
    await _close_all_servers()
    _cache_fingerprint = fingerprint

    if not servers:
        return {}

    results = await asyncio.gather(
        *[_connect_server(server) for server in servers], return_exceptions=True
    )
    for item in results:
        if isinstance(item, BaseException):
            # ⚠️ 单点故障**只降级，绝不 throw** —— 外部工具挂了不影响主流程
            print(f"[mcp] server connect failed: {item}")
    return _merge_cached_tools()


async def discover_mcp_tools() -> dict[str, Tool]:
    """发现所有已配置 server 的工具（懒连接 + single-flight + 缓存，**绝不 throw**）。"""
    global _discovery_in_flight
    if _discovery_in_flight is not None:
        return await _discovery_in_flight
    _discovery_in_flight = asyncio.ensure_future(_do_discover())
    try:
        return await _discovery_in_flight
    finally:
        _discovery_in_flight = None


async def refresh_mcp() -> None:
    """手动刷新：关闭所有连接并重新发现（供路由触发）。"""
    global _cache_fingerprint
    await _close_all_servers()
    _cache_fingerprint = ""
    await _do_discover()


def get_mcp_status() -> dict[str, Any]:
    """当前 MCP 接入状态（``configured`` = 配置里的名字，``connected`` = 已连接明细）。"""
    return {
        "configured": [cfg["name"] for cfg in load_mcp_servers()],
        "connected": [
            {
                "server": server["config"]["name"],
                "transport": infer_transport(server["config"]),
                "toolCount": len(server["tools"]),
                "tools": list(server["tools"]),
            }
            for server in _connected_servers.values()
        ],
    }


async def test_mcp_server(cfg: dict[str, Any]) -> dict[str, Any]:
    """测试连接单个 server（**独立临时连接，测完即关，不写缓存**）。"""
    name = cfg.get("name") if isinstance(cfg, dict) else None
    if not name or not isinstance(name, str):
        return {"ok": False, "error": "server name is required"}
    client: McpClient | None = None
    try:
        client = _build_client(cfg)
        timeout_ms = int(cfg.get("connectTimeoutMs") or DEFAULT_CONNECT_TIMEOUT_MS)
        await _with_timeout(client.connect(), timeout_ms, f'MCP "{name}" connect')
        listed = await _with_timeout(client.list_tools(), timeout_ms, f'MCP "{name}" listTools')
        tools = [item.get("name") for item in listed if isinstance(item, dict)]
        return {"ok": True, "server": name, "transport": infer_transport(cfg),
                "toolCount": len(tools), "tools": tools}
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价：失败也是一种「结果」
        return {"ok": False, "server": name, "error": str(err)}
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
