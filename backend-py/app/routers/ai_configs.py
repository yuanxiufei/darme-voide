"""``/api/v1/ai-configs`` + ``/api/v1/ai-providers`` —— 对齐 ``routes/aiConfigs.ts``。

**已迁移 12 个端点**：列表 / 新建(201) / 一键配置 / 一键本地 / 详情 / 更新 / 删除 /
本地配置列表 / 服务商目录(``ai-providers``) + ``GET /runtime/health``（本地四大运行时探测）
+ **``GET /gpu/status``、``POST /gpu/release-all``（GPU 显存管理器，配 ``services/gpu_manager.py``）**
**未迁移 6 个**（全部依赖外部进程或网络）：``/ollama/*``(4)、``POST /models``、``POST /test``

⚠️ 四处易错点：

1. **路由顺序**：``GET /configs/local`` 必须排在 ``GET /{config_id}`` 之前（否则 ``configs``
   会被当 id 解析）。虽然两者段数不同、FastAPI 不会真的冲突，但保持这个顺序更稳。
2. **``DELETE`` 是硬删**（``db.delete``），不是软删 —— 与其余域不同，别照抄习惯。
3. **两种错误码**：``GET``/``DELETE`` 的 catch 走 **500** ``{code,data:null,message}``，
   而 ``POST``/``PUT`` 的 catch 走 **400**（无 ``data`` 键）。
4. ``GET /:id`` 与 ``PUT /:id`` 在**记录不存在**时回 404 ``'not found'``（默认文案），
   而 ``parseParamId`` 失败回 404 ``'Invalid config id'`` —— 两句话不一样。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import agent_configs, ai_service_configs, ai_service_providers
from ..response import (
    bad_request,
    created,
    internal_error,
    js_truthy,
    not_found,
    now,
    parse_param_id,
    row_to_dict,
    success,
)
from ..services.ai_configs import (
    LOCAL_PRESET_SERVICES,
    PRESET_AGENT_DEFAULTS,
    PRESET_AGENT_MODEL,
    PRESET_SERVICES,
    UNSET,
    build_settings,
    is_local_config,
    map_config_row,
)
from ..services.gpu_manager import get_nvidia_smi, gpu_manager
from ..services.task_logger import log_task_error
from ..services.ollama import (
    LOCAL_RUNTIMES,
    find_ollama_exe,
    format_bytes,
    is_ollama_reachable,
    normalize_ollama_url,
    probe_local_runtime,
    try_start_ollama,
)
from ..services.provider_probe import (
    build_list_models,
    build_probe,
    parse_models_safely,
    redact_url,
)

router = APIRouter(prefix="/api/v1/ai-configs", tags=["ai-configs"])
providers_router = APIRouter(prefix="/api/v1/ai-providers", tags=["ai-providers"])

#: 紧凑 JSON（对齐 JSON.stringify）
_JSON = {"ensure_ascii": False, "separators": (",", ":")}


# ---------------------------------------------------------------------------
# 列表 / 新建
# ---------------------------------------------------------------------------

@router.get("")
def list_configs(request: Request, conn: Connection = Depends(get_conn)):
    try:
        service_type = request.query_params.get("service_type")
        rows = conn.execute(select(ai_service_configs)).all()
        if js_truthy(service_type):
            rows = [r for r in rows if r.service_type == service_type]
        return success(
            [map_config_row(r, with_settings_fields=True, with_is_local=True) for r in rows]
        )
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.post("")
def create_config(body: dict[str, Any], conn: Connection = Depends(get_tx)):
    try:
        if not js_truthy(body.get("service_type")) or not js_truthy(body.get("provider")):
            return bad_request("service_type and provider are required")

        ts = now()
        provider = body.get("provider")
        service_type = body.get("service_type")
        result = conn.execute(
            ai_service_configs.insert().values(
                service_type=service_type,
                provider=provider,
                name=body.get("name") if js_truthy(body.get("name")) else f"{provider}-{service_type}",
                base_url=body.get("base_url") if js_truthy(body.get("base_url")) else "",
                api_key=body.get("api_key") if js_truthy(body.get("api_key")) else "",
                model=json.dumps(
                    body.get("model") if js_truthy(body.get("model")) else [], **_JSON
                ),
                priority=body.get("priority") if js_truthy(body.get("priority")) else 0,
                settings=build_settings(
                    negative_prompt=body.get("negative_prompt", UNSET),
                    checkpoint_map=body.get("checkpoint_map", UNSET),
                    settings=body.get("settings", UNSET),
                ),
                is_active=True,
                created_at=ts,
                updated_at=ts,
            )
        )
        new_id = int(result.inserted_primary_key[0])
        row = conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.id == new_id)
        ).first()
        return created(map_config_row(row))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# 一键配置（预设 / 本地）
# ---------------------------------------------------------------------------

def _upsert_preset(
    conn: Connection,
    presets: tuple[dict[str, Any], ...],
    *,
    api_key: str,
    name_prefix: str,
    ts: str,
) -> None:
    """按 (service_type, provider) 覆盖写入预设配置。

    ⚠️ 原 TS 是「先按 service_type 查、再在内存里 filter provider」—— 语义上等价于按
    (service_type, provider) 查找，照抄以免漏掉同类型多 provider 的情形。
    """
    for preset in presets:
        candidates = conn.execute(
            select(ai_service_configs).where(
                ai_service_configs.c.service_type == preset["service_type"]
            )
        ).all()
        existing = next((r for r in candidates if r.provider == preset["provider"]), None)

        values = {
            "service_type": preset["service_type"],
            "provider": preset["provider"],
            "name": f"{name_prefix}{preset['label']}服务",
            "base_url": preset["base_url"],
            "api_key": api_key,
            "model": json.dumps([preset["model"]], **_JSON),
            "priority": preset["priority"],
            "is_active": True,
            "updated_at": ts,
        }
        if existing is not None:
            conn.execute(
                ai_service_configs.update()
                .where(ai_service_configs.c.id == existing.id)
                .values(**values)
            )
        else:
            conn.execute(ai_service_configs.insert().values(**values, created_at=ts))


@router.post("/quick-preset")
def quick_preset(body: dict[str, Any], conn: Connection = Depends(get_tx)):
    try:
        api_key = str(body.get("api_key") or "").strip()
        if not api_key:
            return bad_request("api_key is required")

        ts = now()
        _upsert_preset(conn, PRESET_SERVICES, api_key=api_key, name_prefix="默认", ts=ts)

        # 同步把 5 个 Agent 指向预设模型（upsert by agent_type）
        for agent in PRESET_AGENT_DEFAULTS:
            existing = conn.execute(
                select(agent_configs).where(
                    agent_configs.c.agent_type == agent["agent_type"]
                )
            ).first()
            if existing is not None:
                conn.execute(
                    agent_configs.update()
                    .where(agent_configs.c.id == existing.id)
                    .values(
                        name=agent["name"],
                        model=PRESET_AGENT_MODEL,
                        is_active=True,
                        updated_at=ts,
                    )
                )
            else:
                conn.execute(
                    agent_configs.insert().values(
                        agent_type=agent["agent_type"],
                        description="",
                        model=PRESET_AGENT_MODEL,
                        name=agent["name"],
                        system_prompt="",
                        temperature=0.7,
                        max_tokens=4096,
                        max_iterations=10,
                        is_active=True,
                        created_at=ts,
                        updated_at=ts,
                    )
                )

        configs = [map_config_row(r) for r in conn.execute(select(ai_service_configs)).all()]
        agents = [row_to_dict(r) for r in conn.execute(select(agent_configs)).all()]
        return success({"configs": configs, "agents": agents, "agent_model": PRESET_AGENT_MODEL})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("/quick-local")
def quick_local(conn: Connection = Depends(get_tx)):
    try:
        ts = now()
        # 本地服务不需要真实 key，但字段不可为空 ⇒ 固定 'local'
        _upsert_preset(conn, LOCAL_PRESET_SERVICES, api_key="local", name_prefix="本地", ts=ts)
        configs = [map_config_row(r) for r in conn.execute(select(ai_service_configs)).all()]
        return success({"configs": configs})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# 本地配置（必须早于 /{config_id} 注册）
# ---------------------------------------------------------------------------

@router.get("/configs/local")
def list_local_configs(conn: Connection = Depends(get_conn)):
    try:
        rows = conn.execute(select(ai_service_configs)).all()
        return success(
            [
                map_config_row(r)
                for r in rows
                if is_local_config(r.base_url or "", r.provider or "")
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


# ---------------------------------------------------------------------------
# 详情 / 更新 / 删除
# ---------------------------------------------------------------------------

@router.get("/{config_id}")
def get_config(config_id: str, conn: Connection = Depends(get_conn)):
    try:
        cid = parse_param_id(config_id)
        if cid is None:
            return not_found("Invalid config id")
        row = conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.id == cid)
        ).first()
        if row is None:
            return not_found()  # 默认文案 'not found'
        return success(map_config_row(row, with_settings_fields=True))
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.put("/{config_id}")
def update_config(config_id: str, body: dict[str, Any], conn: Connection = Depends(get_tx)):
    try:
        cid = parse_param_id(config_id)
        if cid is None:
            return not_found("Invalid config id")
        row = conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.id == cid)
        ).first()
        if row is None:
            return not_found()

        updates: dict[str, Any] = {"updated_at": now()}
        # `'x' in body` ⇒ **键存在性**（显式传 null 会写入 null）
        for key, column in (
            ("provider", "provider"),
            ("name", "name"),
            ("base_url", "base_url"),
            ("api_key", "api_key"),
            ("priority", "priority"),
            ("is_active", "is_active"),
        ):
            if key in body:
                updates[column] = body[key]
        if "model" in body:
            updates["model"] = json.dumps(body["model"], **_JSON)
        # settings 是「合并写」：任一相关键出现即重算，保住既有扩展配置
        if any(k in body for k in ("negative_prompt", "checkpoint_map", "settings")):
            updates["settings"] = build_settings(
                existing=row.settings,
                negative_prompt=body.get("negative_prompt", UNSET),
                checkpoint_map=body.get("checkpoint_map", UNSET),
                settings=body.get("settings", UNSET),
            )

        conn.execute(
            ai_service_configs.update().where(ai_service_configs.c.id == cid).values(**updates)
        )
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.delete("/{config_id}")
def delete_config(config_id: str, conn: Connection = Depends(get_tx)):
    try:
        cid = parse_param_id(config_id)
        if cid is None:
            return not_found("Invalid config id")
        existing = conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.id == cid)
        ).first()
        if existing is None:
            return not_found("Config not found")
        # ⚠️ 硬删（这张表没有 deleted_at）
        conn.execute(delete(ai_service_configs).where(ai_service_configs.c.id == cid))
        return success()
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


# ---------------------------------------------------------------------------
# 服务商目录（TS 里作为独立 Hono app 导出，挂在 /api/v1/ai-providers）
# ---------------------------------------------------------------------------

@providers_router.get("")
def list_providers(conn: Connection = Depends(get_conn)):
    """原 TS **没有 try/catch**（异常走 Hono 默认 500），这里同样只做兜底。"""
    try:
        rows = conn.execute(select(ai_service_providers)).all()
        parsed = []
        for row in rows:
            d = row_to_dict(row)
            raw = d.get("preset_models")
            d["preset_models"] = json.loads(raw) if js_truthy(raw) else []
            parsed.append(d)
        return success(parsed)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


# ---------------------------------------------------------------------------
# Ollama 本地模型管理（子进程 + HTTP，Python 完全等价）
# ---------------------------------------------------------------------------

async def _lenient_json(request: Request) -> dict[str, Any]:
    """对齐 ``await c.req.json().catch(() => ({}))`` —— 解析失败一律当空对象，不报错。"""
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


@router.post("/ollama/status")
async def ollama_status(request: Request):
    """检测 Ollama 运行状态并列出本机所有已安装模型。"""
    try:
        body = await _lenient_json(request)
        base_url = normalize_ollama_url(body.get("base_url"))
        exe = find_ollama_exe()

        if not await is_ollama_reachable(base_url):
            return success(
                {
                    "running": False,
                    "base_url": base_url,
                    "models": [],
                    "exe": exe,
                    "message": f"无法连接 {base_url}（Ollama 服务未运行）",
                }
            )

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")

        if not resp.is_success:
            return success(
                {
                    "running": True,
                    "base_url": base_url,
                    "models": [],
                    "exe": exe,
                    "message": f"Ollama 响应异常 (HTTP {resp.status_code})",
                }
            )

        try:
            data = resp.json()
        except ValueError:
            data = {}
        raw_models = data.get("models") if isinstance(data, dict) else None
        models = [
            {
                "name": m.get("name"),
                "size": m.get("size") or 0,
                "size_label": format_bytes(m.get("size")),
                "modified_at": m.get("modified_at") or "",
                "digest": str(m.get("digest") or "")[:12],
            }
            for m in (raw_models or [])
            if isinstance(m, dict)
        ]
        models.sort(key=lambda m: m["name"] or "")
        return success(
            {
                "running": True,
                "base_url": base_url,
                "models": models,
                "exe": exe,
                "message": f"检测到 {len(models)} 个本地模型",
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("/ollama/start")
async def ollama_start():
    """启动本地 Ollama 服务（**最坏等 20 秒**，见 services/ollama.py 模块头）。"""
    try:
        return success(await try_start_ollama())
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("/ollama/pull")
async def ollama_pull(request: Request):
    """下载/拉取模型，**NDJSON 流式**返回进度（原 TS 用 hono 的 stream）。"""
    try:
        body = await _lenient_json(request)
        base_url = normalize_ollama_url(body.get("base_url"))
        name = str(body.get("name") or "").strip()
        if not name:
            return bad_request("缺少模型名，例如 qwen3:8b")
        if not await is_ollama_reachable(base_url):
            return bad_request(f"Ollama 服务未运行（{base_url}），请先在下方启动")

        # 拉模型可能很久（原 TS 给 1 小时超时）
        client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=3600.0))
        upstream_request = client.build_request(
            "POST", f"{base_url}/api/pull", json={"name": name, "stream": True}
        )
        upstream = await client.send(upstream_request, stream=True)

        if upstream.status_code >= 400:
            text = (await upstream.aread()).decode("utf-8", "replace")
            await upstream.aclose()
            await client.aclose()
            return success(
                {"ok": False, "error": (text or f"HTTP {upstream.status_code}")[:400]}
            )

        async def relay():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(relay(), media_type="application/x-ndjson; charset=utf-8")
    except Exception as exc:  # noqa: BLE001
        return success({"ok": False, "error": str(exc)})


@router.post("/ollama/delete")
async def ollama_delete(request: Request):
    """删除本地 Ollama 模型。"""
    try:
        body = await _lenient_json(request)
        base_url = normalize_ollama_url(body.get("base_url"))
        name = str(body.get("name") or "").strip()
        if not name:
            return bad_request("缺少模型名")
        if not await is_ollama_reachable(base_url):
            return bad_request(f"Ollama 服务未运行（{base_url}）")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                "DELETE", f"{base_url}/api/delete", json={"name": name}
            )

        if not resp.is_success:
            text = resp.text
            return bad_request((text or f"HTTP {resp.status_code}")[:400])
        return success({"ok": True, "deleted": name})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# 平台检测 + 模型列举
# ---------------------------------------------------------------------------

async def _list_provider_models(
    provider: str, base_url: str, api_key: str | None
) -> dict[str, Any]:
    """执行模型列举，返回 ``{listable, models, error}``。

    注意 ``listable`` 在**有描述符时恒为 true**（即使请求失败），只有「该平台不支持在线列举」
    才是 false —— 前端用它与 ``error`` 组合出提示文案。
    """
    desc = build_list_models(provider, base_url, api_key)
    if not desc:
        return {"listable": False, "models": [], "error": ""}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.request(desc["method"], desc["url"], headers=desc["headers"])
        if not resp.is_success:
            reason = resp.reason_phrase or ""
            return {
                "listable": True,
                "models": [],
                "error": f"HTTP {resp.status_code} {reason}".strip(),
            }
        try:
            data = resp.json()
        except ValueError:
            data = {}
        return {"listable": True, "models": parse_models_safely(desc["parse"], data), "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {"listable": True, "models": [], "error": str(exc)}


@router.post("/models")
async def list_models(request: Request):
    """检测平台连通性 + 列出可用模型 + 检测指定模型是否存在。"""
    try:
        body = await _lenient_json(request)
        provider = str(body.get("provider") or "").strip()
        base_url = str(body.get("base_url") or "").strip()
        if not provider:
            return bad_request("provider is required")

        api_key = str(body.get("api_key") or "")
        # 兼容旧的单 model 字符串与新的 models 数组
        source = body.get("models") if isinstance(body.get("models"), list) else None
        if source is None:
            source = [body.get("model")] if body.get("model") else []
        wanted_list = [w for w in (str(m or "").strip() for m in source) if w]

        p = provider.lower()
        result = await _list_provider_models(provider, base_url, api_key)
        listable, models, error = result["listable"], result["models"], result["error"]

        # 检测模型是否存在：精确匹配；Ollama 允许省略 tag（qwen3 匹配 qwen3:14b）
        model_checks = []
        for wanted in wanted_list:
            w = wanted.lower()
            exists = any(
                (str(mid).lower() == w)
                or (p == "ollama" and str(mid).lower().startswith(f"{w}:"))
                for mid in models
            )
            model_checks.append({"model": wanted, "exists": exists})

        first_missing = next((c for c in model_checks if not c["exists"]), None)
        if error:
            message = f"平台连接失败：{error}"
        elif not listable:
            message = "该平台暂不支持在线列举模型，可手动填写模型名（可用「测试配置」验证连通性）"
        elif len(model_checks) == 0:
            message = f"已检测到 {len(models)} 个可用模型"
        elif first_missing:
            message = f"模型 {first_missing['model']} 未出现在平台模型列表中，请核对模型名"
        elif len(model_checks) == 1:
            message = f"模型 {model_checks[0]['model']} 可用 ✓"
        else:
            message = f"{len(model_checks)} 个模型均可用 ✓"

        return success(
            {
                "provider": provider,
                "base_url": base_url,
                "listable": listable,
                # `listable ? !error : null`
                "reachable": (not error) if listable else None,
                "models": models[:500],
                "models_count": len(models),
                "model": model_checks[0]["model"] if model_checks else None,
                "model_exists": model_checks[0]["exists"] if model_checks else False,
                "model_checks": model_checks,
                "message": message,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("/test")
async def test_config(request: Request):
    """按厂商拼出「最小可判活请求」，发出去按状态码判定连通性。"""
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001 - 原 TS 此处无 try/catch ⇒ 解析失败是 500
        return internal_error(str(exc))

    if not body.get("service_type") or not body.get("provider") or not body.get("base_url"):
        return bad_request("service_type, provider and base_url are required")

    model = body["model"][0] if isinstance(body.get("model"), list) and body["model"] else (
        None if isinstance(body.get("model"), list) else body.get("model")
    )
    probe = build_probe(
        body.get("service_type"), body.get("provider"), body.get("base_url"), model, body.get("api_key")
    )
    probe_url = redact_url(probe["url"])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                probe["method"],
                probe["url"],
                headers=probe["headers"],
                # ⚠️ 判据必须是 `is not None`：`{}` 在 Python 里是**假值**，
                # 而 JS 里是真值（会真的发出 "{}" 这个 body）
                # ⚠️ 这是**发给厂商**的请求体：Node 是 `JSON.stringify(probe.body)`（紧凑）
            content=json.dumps(probe["body"], ensure_ascii=False, separators=(",", ":"))
            if probe["body"] is not None
            else None,
            )
        text = resp.text
        reachable = resp.status_code in (200, 204, 400, 401, 403)
        if reachable:
            message = (
                "端点可访问，认证与路径基本正常"
                if resp.is_success
                else "端点已响应，请根据状态码判断认证或路径是否正确"
            )
        else:
            message = "端点未按预期响应，请检查 Base URL 和代理前缀"
        return success(
            {
                "ok": resp.is_success,
                "reachable": reachable,
                "status": resp.status_code,
                "status_text": resp.reason_phrase,
                "method": probe["method"],
                "url": probe_url,
                "message": message,
                "response_preview": text[:240],
            }
        )
    except Exception as exc:  # noqa: BLE001
        return success(
            {
                "ok": False,
                "reachable": False,
                "method": probe["method"],
                "url": probe_url,
                "message": str(exc) or "请求失败",
                "response_preview": "",
            }
        )


# ─── GPU 显存监控 ──────────────────────────────────────────────

@router.get("/gpu/status")
async def gpu_status():
    """GPU 显存使用状态（管理器快照 + ``nvidia-smi`` 实时硬件数据）。

    ⚠️ 本端点**返回裸 JSON**（不是 ``{code,data,message}`` 信封）—— 与原 TS 的 ``c.json(result)``
    一致；失败时也是 HTTP 200 + ``{"code":500,...}``（``c.json`` 没带状态码）。
    """
    try:
        result = {
            **gpu_manager.get_status(),
            "isLocalMode": False,
            "hardware": None,
        }
        smi = await get_nvidia_smi()
        result["isLocalMode"] = smi is not None
        if smi:
            result["hardware"] = {
                "gpuName": smi["gpuName"],
                "totalMemoryMB": smi["totalMemoryMB"],
                "usedMemoryMB": smi["usedMemoryMB"],
                "freeMemoryMB": smi["freeMemoryMB"],
                "utilizationPercent": smi["utilizationPercent"],
                "temperatureC": smi["temperatureC"],
            }
        return JSONResponse(
            content=json.loads(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(content={"code": 500, "data": None, "message": str(exc)})


@router.post("/gpu/release-all")
async def gpu_release_all():
    """强制释放所有本地模型显存（调试用）。"""
    try:
        await gpu_manager.release_all()
        return success({"message": "All GPU models released", "status": gpu_manager.get_status()})
    except Exception as exc:  # noqa: BLE001
        log_task_error("AIConfig", "gpu-release-all", {"error": str(exc)})
        return bad_request(str(exc))


@router.get("/runtime/health")
async def runtime_health(conn: Connection = Depends(get_conn)):
    """统一探测本地四大运行时（文本/图像/视频/语音）。"""
    try:
        configs = conn.execute(select(ai_service_configs)).all()
        probes = await asyncio.gather(
            *(probe_local_runtime(rt["base_url"], rt["probe_path"]) for rt in LOCAL_RUNTIMES)
        )
        services = []
        for rt, probe in zip(LOCAL_RUNTIMES, probes):
            registered = [
                row
                for row in configs
                if is_local_config(row.base_url or "", row.provider or "")
                and row.service_type == rt["service_type"]
            ]
            services.append(
                {
                    "key": rt["key"],
                    "label": rt["label"],
                    "service_type": rt["service_type"],
                    "provider": rt["provider"],
                    "base_url": rt["base_url"],
                    "running": probe["reachable"],
                    "http_status": probe["httpStatus"],
                    "latency_ms": probe["latencyMs"],
                    "error": probe["error"],
                    "registered_count": len(registered),
                    "registered": [
                        {
                            "id": row.id,
                            "name": row.name,
                            "provider": row.provider,
                            "base_url": row.base_url,
                        }
                        for row in registered
                    ],
                }
            )
        running_count = sum(1 for s in services if s["running"])
        return success(
            {
                "services": services,
                "running_count": running_count,
                "total": len(services),
                "all_running": running_count == len(services),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))
