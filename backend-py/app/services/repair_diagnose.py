"""**画布修复诊断** —— 回答「现在到底缺什么、缺了堵住哪条工作流」（源：`assistant-runtime/diagnostics.mjs` ✓ 重写）。

为什么不照搬：源那份是**源项目网关的客户端**，8 个工具全打 ``127.0.0.1:8291`` 的
``/health`` ``/api/setup/status`` ``/api/resources/status`` ``/api/system/stats`` …
本仓没有这些端点 ✗，照抄过来每个工具都是 404 ⇒ 必须按本仓数据源**重写** ✓。

**照搬的是设计**（这部分是踩出来的，别丢）：

1. 每个能力都拆成「一个工具一个专门问句」✓ —— 别做 ``get_status()`` 返回一大坨，
   模型会挑不出重点 ✗。
2. **一律返回摘要 + 条数封顶** ✓（源 ``capList(list, 8)`` + ``TEXT_LIMIT 14000`` 截断）。
   全量资源清单 70+ 条塞进上下文，等于把注意力稀释掉 ✗。
3. **只问本机**（源 ``normalizeGatewayUrl`` 只放行 ``127.0.0.1/localhost/::1``）✓ ——
   诊断不该把仓库路径或模型清单发到外部。
4. 取不到就**如实报错**，不编默认值 ✓（源 ``collectSnapshot`` 失败给 ``ok:false`` 而不是抛）。

**本仓数据源**（都是现成的，不重造）：

| 问句 | 数据源 |
|---|---|
| 某条工作流要哪些模型文件 | 读 ``configs/comfyui/workflows/*.json`` 的 loader 节点（API 格式：顶层就是节点表） |
| 这些文件在不在盘上 | ``local_model_scan.scan_local_models()``（既有扫描器，**复用**不重写） |

⚠️ 扫描一律 ``fingerprint=False``：诊断只问「在不在」，不需要读每个文件头判真架构
（那是 2ms/个 × 上千文件；开着会让这个端点要十几秒 ✗）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import local_model_scan

#: 认作模型权重的扩展名 ✓（与源 ``.safetensors/.pt/.pth/.gguf/.onnx`` 同口径，补 ``.ckpt/.bin``）
MODEL_EXTENSIONS: tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".bin")

#: 摘要里最多列几条缺口（源 ``capList(problems, 8)``）
MAX_PROBLEMS = 8
#: 一次诊断最多回多少条缺口明细（超出只给计数，别把 70 条全吐给模型）
MAX_MISSING_ROWS = 40
#: registry.json 不是工作流，跳过 ✓
REGISTRY_FILE = "registry.json"


def _is_model_ref(value: Any) -> bool:
    """这串是不是模型文件名 ✓ —— 按扩展名判，不看节点类型（loader 的类名各式各样）。"""
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().lower().endswith(MODEL_EXTENSIONS)


def _node_inputs(node: Any) -> dict[str, Any]:
    return node.get("inputs") if isinstance(node, dict) and isinstance(node.get("inputs"), dict) else {}


def _iter_nodes(payload: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    """兼容两种格式 ✓ —— API 格式（顶层扁平 ``{nid: {class_type, inputs}}``）
    与 UI 格式（``{"nodes": [{"type", "widgets_values"}]}``）。"""
    if not isinstance(payload, dict):
        return
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        for index, node in enumerate(nodes):
            if isinstance(node, dict):
                yield str(node.get("id") or index), {
                    "class_type": node.get("type") or "",
                    "inputs": {},
                    "widgets_values": node.get("widgets_values") or [],
                }
        return
    for node_id, node in payload.items():
        if not isinstance(node, dict):
            continue
        yield str(node_id), {
            "class_type": node.get("class_type") or node.get("type") or "",
            "inputs": _node_inputs(node),
            "widgets_values": node.get("widgets_values") or [],
        }


def workflow_requirements(workflows_dir: Path) -> dict[str, Any]:
    """反推**每份工作流要哪些模型文件** ✓。

    ⚠️ 工作流里写的是 loader 的 ``widgets_values``/``inputs`` 里的**文件名字面量** ——
    这是「缺哪件就堵哪条工作流」的唯一准确依据；注册表 ``registry.json`` **不声明模型依赖** ✗。
    """
    directory = Path(workflows_dir)
    if not directory.is_dir():
        return {"ok": False, "error": f"工作流目录不存在：{directory}", "refs": {}, "workflows": 0}

    refs: dict[str, dict[str, Any]] = {}
    parsed = 0
    for path in sorted(directory.glob("*.json")):
        if path.name == REGISTRY_FILE:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed += 1
        for _node_id, node in _iter_nodes(payload):
            class_type = str(node.get("class_type") or "")
            is_lora = "Lora" in class_type
            values: list[Any] = list((node.get("inputs") or {}).values())
            values += list(node.get("widgets_values") or [])
            for value in values:
                if not _is_model_ref(value):
                    continue
                name = str(value).strip().replace("\\", "/").split("/")[-1]
                slot = refs.setdefault(name.lower(), {
                    "file": name,
                    "kind": "lora" if is_lora else "model",
                    "usedBy": [],
                    "nodes": [],
                })
                if path.stem not in slot["usedBy"]:
                    slot["usedBy"].append(path.stem)
                if class_type and class_type not in slot["nodes"]:
                    slot["nodes"].append(class_type)

    return {"ok": True, "workflowsDir": str(directory), "workflows": parsed, "refs": refs}


def _index_local_files(roots: list[str] | None = None) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """本机模型索引 ✓ —— 键是**小写文件名**，值是同名文件列表（同名多份要能报 ``ambiguous``）。"""
    scan = local_model_scan.scan_local_models({"fingerprint": False, "roots": roots} if roots else {"fingerprint": False})
    index: dict[str, list[dict[str, Any]]] = {}
    for item in scan.get("models") or []:
        name = str(item.get("filename") or "").strip()
        if not name:
            continue
        index.setdefault(name.lower(), []).append({
            "path": item.get("path"),
            "dirname": item.get("dirname"),
            "sizeBytes": item.get("sizeBytes"),
            "kind": item.get("kind"),
        })
    return index, list(scan.get("roots") or [])


def diagnose(workflows_dir: Path, roots: list[str] | None = None) -> dict[str, Any]:
    """**一次答完**：工作流要什么、盘上有没有、缺的堵了谁 ✓。

    ⚠️ 只回摘要：``missing`` 明细封顶 ``MAX_MISSING_ROWS`` 条，超出给计数并置 ``truncated``。
    """
    req = workflow_requirements(workflows_dir)
    if not req.get("ok"):
        return {"ok": False, "error": req.get("error") or "读不到工作流目录"}

    index, scanned_roots = _index_local_files(roots)
    missing: list[dict[str, Any]] = []
    present: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for key, slot in sorted(req["refs"].items()):
        hits = index.get(key) or []
        if not hits:
            missing.append({**slot, "present": False})
            continue
        if len(hits) > 1:
            # ⚠️ 同名多份 ⇒ 说清"不止一处"，别假装确定是哪份 ✗
            ambiguous.append({**slot, "present": True, "copies": hits[:MAX_PROBLEMS]})
        present.append({**slot, "present": True, "path": hits[0]["path"], "sizeBytes": hits[0]["sizeBytes"]})

    covered = sorted({name for item in missing for name in item["usedBy"]})
    total_workflows = int(req.get("workflows") or 0)
    return {
        "ok": True,
        "workflowsDir": req.get("workflowsDir"),
        "modelRoots": scanned_roots,
        "counts": {
            "workflows": total_workflows,
            "references": len(req["refs"]),
            "present": len(present),
            "missing": len(missing),
            "blockedWorkflows": len(covered),
        },
        "missing": missing[:MAX_MISSING_ROWS],
        "blockedWorkflows": covered[:MAX_MISSING_ROWS],
        "ambiguous": ambiguous[:MAX_PROBLEMS],
        "truncated": len(missing) > MAX_MISSING_ROWS,
        "note": (
            "⚠️ 按**文件名**核对，不按目录：模型常被放在别的子目录（如 Krea2 的 int8 在 diffusion_models/Krea2/）。"
            "同名多份会列进 ambiguous，不替你挑。"
        ),
    }


def summarize(result: dict[str, Any], limit: int = MAX_PROBLEMS) -> dict[str, Any]:
    """给模型看的**摘要** ✓（与源 ``summarizeProblems`` 同用途：别把全量表塞进上下文）。"""
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error") or "诊断失败"}
    counts = result.get("counts") or {}
    rows = []
    for item in (result.get("missing") or [])[:limit]:
        rows.append({
            "file": item.get("file"),
            "kind": item.get("kind"),
            "usedBy": (item.get("usedBy") or [])[:3],
        })
    return {
        "ok": True,
        "counts": counts,
        "missingTop": rows,
        "missingMore": max(0, int(counts.get("missing") or 0) - len(rows)),
        "blockedWorkflows": (result.get("blockedWorkflows") or [])[:limit],
        "ambiguous": len(result.get("ambiguous") or []),
    }
