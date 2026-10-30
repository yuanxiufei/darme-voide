"""模型目录 —— 把「盘上的清单」与「服务端的能力面」合成一份**可判定的清单**。

它回答三个问题（前端/智能体都只问这一个模块）：

1. **有哪些模型、各自能干什么** —— 在线走 ``/api/show``（权威 ✓），离线走清单 + config
   层 + 模板层（:mod:`app.services.ollama_native.capabilities` 的五路信号 ✓）；
2. **哪些能读视频** —— 判据是 ``vision``（见 :func:`capabilities.is_video_reader` 的说明：
   Ollama 没有视频模型，读视频＝抽帧喂视觉模型 ✓）；
3. **装了哪些插件** —— 权重之外的附加层（adapter / projector / draft / template …），
   并指出**哪些插件改变了能力面**（projector ⇒ 能读图，adapter ⇒ 换了行为）。

⚠️ 在线/离线**必须给出同一个形状**：调用点不该分两套代码。差异只体现在每个模型的
``source``（``service`` / ``disk``）与 ``capabilitiesSource``（``service`` / ``manifest``）
两个字段上 ✓ —— 这也是「这条结论有多硬」的唯一说明处，别把它藏进日志 ✗。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from . import ollama_store
from .ollama import format_bytes, resolve_ollama_url
from .ollama_client import OllamaClient, OllamaError, normalize_show
from .ollama_native import capabilities as capability_rules
from .ollama_native import manifest as manifest_rules

__all__ = [
    "disk_inventory",
    "inventory",
    "plugin_summary",
    "video_generation_models",
    "video_readers",
]


def _read_json(path: str) -> Mapping[str, Any]:
    try:
        with open(path, "rb") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _read_text(path: str | None, limit: int = 262_144) -> str:
    """读模板/参数这类小文本层。**有上限**：一份坏清单可能把模板指到几 GB 的权重 blob 上 ✓。"""
    if not path:
        return ""
    try:
        with open(path, "rb") as handle:
            return handle.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _layer_digests(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(mediaType, digest)`` 列表，顺序照清单 ✓（同一种层可能有多份，比如多个 adapter ✓）。"""
    found: list[tuple[str, str]] = []
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        return found
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        digest = str(layer.get("digest") or "").strip()
        if digest:
            found.append((str(layer.get("mediaType") or ""), digest))
    return found


def _disk_model(item: Mapping[str, Any], root: str) -> dict[str, Any]:
    """单个盘上模型 → 统一形状（读清单 + config 层 + 模板层 ✓，读不到就按缺省判 ✓）。"""
    manifest = _read_json(str(item.get("path") or ""))
    layers = _layer_digests(manifest)
    media_types = [media_type for media_type, _ in layers]

    config_payload: Mapping[str, Any] = {}
    template = ""
    # ⚠️ config 层在清单里是**顶层 ``config`` 描述符**，不在 ``layers`` 里 ✗ ——
    # 只扫 layers 的实现会永远读不到能力自述，于是「有 config.capabilities 的模型」全被判空。
    config_desc = manifest.get("config")
    if isinstance(config_desc, Mapping):
        digest = str(config_desc.get("digest") or "").strip()
        if digest:
            config_payload = _read_json(ollama_store.blob_path(digest, root) or "")
    for media_type, digest in layers:
        if not config_payload and media_type in (
            manifest_rules.CONFIG_MEDIA_TYPE,
            manifest_rules.MEDIA_TYPE_CONFIG,
        ):
            config_payload = _read_json(ollama_store.blob_path(digest, root) or "")
        elif not template and media_type == manifest_rules.MEDIA_TYPE_TEMPLATE:
            template = _read_text(ollama_store.blob_path(digest, root))

    facts = capability_rules.capabilities_from(
        config=config_payload, media_types=media_types, template=template
    )
    capabilities = facts["capabilities"]
    return {
        "name": str(item.get("name") or ""),
        "size": int(item.get("size") or 0),
        "sizeLabel": str(item.get("size_label") or format_bytes(int(item.get("size") or 0))),
        "modifiedAt": str(item.get("modified_at") or ""),
        "digest": str(item.get("digest") or ""),
        "source": "disk",
        "capabilitiesSource": "manifest",
        "path": str(item.get("path") or ""),
        "missingBlobs": int(item.get("missingBlobs") or 0),
        "capabilities": capabilities,
        "capabilitySources": facts["sources"],
        "capabilityLabel": capability_rules.describe(capabilities),
        "plugins": facts["plugins"],
        "pluginKinds": sorted({p["kind"] for p in facts["plugins"]}),
        "videoReader": bool(facts["video_reader"]),
        "family": facts["model_family"],
        "format": facts["model_format"],
        "renderer": facts["renderer"],
        "parser": facts["parser"],
        "hasProjector": any(m == manifest_rules.MEDIA_TYPE_PROJECTOR for m in media_types),
    }


def disk_inventory(root: str | None = None) -> dict[str, Any]:
    """**离线**清点（只读盘、不打网络 ✓）。盘上没有模型库 ⇒ 空清单 + 一句原因（不抛错 ✓）。"""
    target = root if root is not None else ollama_store.models_root()
    items = ollama_store.list_models(target)
    models = [_disk_model(item, target or "") for item in items]
    models.sort(key=lambda m: m["name"])
    return {
        "online": False,
        "baseUrl": resolve_ollama_url(),
        "modelsRoot": target or "",
        "available": bool(items) or ollama_store.is_available(target),
        "models": models,
        "reason": "" if items else "盘上还没有 Ollama 模型库（或模型库为空）",
    }


async def inventory(
    base_url: str | None = None,
    *,
    include_details: bool = True,
    fallback_to_disk: bool = True,
) -> dict[str, Any]:
    """**在线优先**清点；连不上就回落盘上（``fallback_to_disk``）✓。

    ``include_details=True`` 会对每个模型再发一次 ``/api/show`` —— 模型多时这是 N+1 次本机
    请求（每次 ~1ms，可接受 ✓）；只要名字/大小时传 ``False`` 走纯 ``/api/tags``（省一半时间 ✓）。
    """
    client = OllamaClient(base_url)
    try:
        try:
            version = await client.version()
            listing = await client.tags()
        except OllamaError as exc:
            if not fallback_to_disk:
                raise
            report = disk_inventory()
            report["baseUrl"] = client.base_url
            report["reason"] = f"服务不可用（{exc}），已回落盘上清单"
            return report
        models: list[dict[str, Any]] = []
        for raw in listing:
            entry = _service_model(raw)
            if include_details:
                try:
                    shown = normalize_show(await client.show(entry["name"]))
                except OllamaError:
                    shown = {}
                if shown:
                    caps = shown["capabilities"] or entry["capabilities"]
                    entry.update({
                        "capabilities": capability_rules.normalize_capabilities(caps),
                        "capabilityLabel": capability_rules.describe(caps),
                        "videoReader": capability_rules.is_video_reader(caps),
                        "family": shown["family"] or entry["family"],
                        "format": shown["format"] or entry["format"],
                        "renderer": entry["renderer"],
                        "parser": entry["parser"],
                        "hasProjector": bool(shown["hasProjector"]),
                        "parameterSize": shown["parameterSize"],
                        "quantization": shown["quantization"],
                    })
            models.append(entry)
        models.sort(key=lambda m: m["name"])
        return {
            "online": True,
            "baseUrl": client.base_url,
            "version": str(version.get("version") or ""),
            "modelsRoot": ollama_store.models_root(),
            "available": True,
            "models": models,
            "reason": "",
        }
    finally:
        await client.close()


def _service_model(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``/api/tags`` 的单条 → 统一形状（**不带**能力面：那要 ``/api/show`` ✓）。"""
    name = str(raw.get("name") or raw.get("model") or "")
    details = raw.get("details") if isinstance(raw.get("details"), Mapping) else {}
    size = int(raw.get("size") or 0)
    return {
        "name": name,
        "size": size,
        "sizeLabel": format_bytes(size),
        "modifiedAt": str(raw.get("modified_at") or ""),
        "digest": str(raw.get("digest") or ""),
        "source": "service",
        "capabilitiesSource": "service",
        "path": "",
        "missingBlobs": 0,
        "capabilities": [],
        "capabilitySources": {},
        "capabilityLabel": "",
        "plugins": [],
        "pluginKinds": [],
        "videoReader": False,
        "family": str(details.get("family") or ""),
        "format": str(details.get("format") or ""),
        "renderer": "",
        "parser": "",
        "hasProjector": False,
        "parameterSize": str(details.get("parameter_size") or ""),
        "quantization": str(details.get("quantization_level") or ""),
    }


def video_readers(models: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """能读视频帧的模型（``vision``），按「越大越优先」排序 —— 视觉塔通常随参数量变强 ✓。

    ⚠️ 这里**不按名字猜**（``*-vl`` / ``llava`` 之类）：名字是可改的营销字段，``vision``
    是清单/服务端判出来的结构事实 ✓。名字只用来排序时做同分兜底 ✓。
    """
    readers = [dict(m) for m in models if m.get("videoReader")]
    readers.sort(key=lambda m: (-int(m.get("size") or 0), str(m.get("name") or "")))
    return readers


def plugin_summary(models: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """插件汇总：按种类计数 + 「谁装了哪种插件」+ **能改能力面的那些**单独标出 ✓。"""
    counts: dict[str, int] = {}
    by_model: list[dict[str, Any]] = []
    for model in models:
        plugins = [p for p in (model.get("plugins") or []) if isinstance(p, Mapping)]
        if not plugins:
            continue
        kinds = sorted({str(p.get("kind") or "") for p in plugins})
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
        by_model.append({
            "model": str(model.get("name") or ""),
            "plugins": [dict(p) for p in plugins],
            "kinds": kinds,
            "changesCapabilities": [
                k for k in kinds if k in capability_rules.CAPABILITY_BEARING_PLUGINS
            ],
        })
    return {
        "counts": dict(sorted(counts.items())),
        "models": by_model,
        "total": sum(counts.values()),
        "kinds": sorted(counts),
    }


def video_generation_models(limit: int = 40) -> dict[str, Any]:
    """**视频生成**模型在哪（Ollama **不含**这类，必须去 ComfyUI/H3 那条线找 ✓）。

    ⚠️ 这个函数存在的意义是**把「读视频」与「生成视频」分开**：用户说的「视频模型」往往
    指后者，而 Ollama 永远不会有它 —— 老实回答「生成类在 ComfyUI 侧」比给一个空列表有用 ✓。
    复用 ``local_model_scan.scan_local_models(kinds=["video"])``（扫描有深度/数量上限 ✓）；
    扫描成本不低，所以**只在显式要报告时调**，不进 ``inventory`` ✓。
    """
    try:
        from .local_model_scan import scan_local_models
    except Exception as exc:                      # noqa: BLE001 - 扫描器坏了不该连累目录 ✓
        return {"models": [], "count": 0, "reason": f"本地扫描不可用：{exc}"}

    try:
        report = scan_local_models({"kinds": ["video"], "maxFiles": 4000})
    except Exception as exc:                      # noqa: BLE001
        return {"models": [], "count": 0, "reason": f"扫描失败：{exc}"}

    found: list[dict[str, Any]] = []
    for item in report.get("models") or []:
        if not isinstance(item, Mapping):
            continue
        found.append({
            "name": str(item.get("filename") or ""),
            "path": str(item.get("path") or ""),
            "sizeLabel": str(item.get("sizeHuman") or ""),
            "ecosystem": str(item.get("ecosystem") or ""),
            "runtime": str(item.get("runtime") or ""),
        })
        if len(found) >= limit:
            break
    return {
        "models": found,
        "count": len(found),
        "truncated": bool(report.get("truncated")),
        "reason": "" if found else "没扫到视频生成模型（ComfyUI/H3 模型库为空或未配置）",
    }
