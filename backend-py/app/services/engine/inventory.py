"""引擎的**权重体检 / 就绪报告**（零依赖 ✓）—— 跑之前先回答「这台机器能不能跑、缺什么」。

## 它解决什么真问题

本机的现实是：H3 那套要 **8 个组件、主权重单个 19.53 GiB** ✓，而且是「几十 GB 下载」级别 ✗。
没有这份体检时，用户只能**跑到一半**才发现某个组件根本没装 ✗（甚至更糟：装了个**截断的文件**，
加载时才炸 ✗）。所以本模块把三件事提前：

1. **在不在**（按 ``configs/models.json`` 的清单逐条对 ``models_dir`` ✓）；
2. **完不完整**（safetensors 头部自述的字节数与文件大小是否自洽 ⇒ 截断当场发现 ✓）；
3. **够不够显存**（权重字节是**精确值** ✓；激活是**估算**且明确标注 ✗ 不承诺 ✓）。

## 事实来源（不猜 ✓）

* 清单：``configs/models.json``（``models[]`` 每项含 ``kind`` / ``filename`` / ``file_path`` /
  ``size_gib`` / ``required`` ✓）；
* 目录：``configs/model-paths.json`` 的 ``models_dir`` ✓（解析优先级由
  :func:`app.services.local_model_scan.get_model_paths` 统一提供 ✓ 与本仓其它地方同一份来源 ✓）；
* 校验：:mod:`app.services.engine.safetensors`（纯 Python ✓）。

⚠️ **GGUF 只报「在/不在 + 大小」**✗（读取器未实现 ✓），并明确标注 ``verified: false`` ✓ ——
不假装验过 ✓。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..local_model_scan import get_model_paths
from . import safetensors as st

__all__ = [
    "DEFAULT_CAPACITY_GIB",
    "component_status",
    "estimate_vram",
    "load_catalog",
    "readiness",
]

#: 本机目标显卡（A5000 24GB ✓）—— 只用于算「余量」，可覆盖 ✓
DEFAULT_CAPACITY_GIB = 24.0

#: 各阶段的必需组件筛选规则（``category`` + 是否 ``required`` ✓）
STAGE_FILTERS: dict[str, dict[str, Any]] = {
    "h3": {"category": "video"},
    "image": {"category": "image"},
    "audio": {"category": "audio"},
    "text": {"category": "text"},
}


def catalog_path() -> Path:
    """``configs/models.json`` 的路径（与 ``scripts/model_manager.py`` 同一份 ✓）。"""
    return Path(__file__).resolve().parents[3].parent / "configs" / "models.json"


def load_catalog() -> dict[str, Any]:
    """读清单 ✓（缺失/非法 ⇒ 返回空表而不是抛错 ✓ —— 体检本身不该把服务搞崩 ✓）。"""
    path = catalog_path()
    if not path.exists():
        return {"models": [], "nodes": [], "error": f"清单不存在：{path}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        return {"models": [], "nodes": [], "error": f"清单读不了：{err}"}
    if not isinstance(data, dict):
        return {"models": [], "nodes": [], "error": "清单不是对象 ✗"}
    models = data.get("models")
    return {
        "models": [item for item in models if isinstance(item, dict)] if isinstance(models, list) else [],
        "nodes": data.get("nodes") if isinstance(data.get("nodes"), list) else [],
        "error": None,
    }


def models_dir() -> Path | None:
    """清单里声明的模型根目录（空 ⇒ ``None`` ✓）。"""
    value = str(get_model_paths().get("models_dir") or "").strip()
    return Path(value) if value else None


def component_path(entry: dict[str, Any], root: Path | None = None) -> Path | None:
    """按清单算出该组件的**期望落点** ✓：优先 ``file_path``（含类别子目录 ✓），否则 ``kind/filename`` ✓。"""
    root = root if root is not None else models_dir()
    if root is None:
        return None
    relative = str(entry.get("file_path") or "").strip()
    if not relative:
        filename = str(entry.get("filename") or "").strip()
        if not filename:
            return None
        kind = str(entry.get("kind") or "").strip()
        relative = f"{kind}/{filename}" if kind else filename
    return root / relative


def component_status(entry: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """单个组件的体检结论（不抛错 ✓ —— 缺文件也是**结论**的一种 ✓）。"""
    path = component_path(entry, root)
    expected_gib = entry.get("size_gib")
    status: dict[str, Any] = {
        "key": entry.get("key"),
        "name": entry.get("name"),
        "kind": entry.get("kind"),
        "category": entry.get("category"),
        "required": bool(entry.get("required")),
        "path": str(path) if path else None,
        "present": False,
        "bytes": 0,
        "expectedGiB": expected_gib,
        "verified": False,
        "problems": [],
    }
    if path is None:
        status["problems"].append("清单里没有可解析的路径（filename/file_path 皆空，或 models_dir 未配置 ✗）")
        return status
    if not path.exists():
        status["problems"].append("未安装 ✗")
        return status
    try:
        status["bytes"] = path.stat().st_size
    except OSError as err:
        status["problems"].append(f"读不到文件：{err}")
        return status
    status["present"] = True

    if path.suffix.lower() == ".gguf":
        status["problems"].append("GGUF：本读取器只认 safetensors ⇒ 仅校了「在/不在 + 大小」✗")
        status["verified"] = False
    else:
        info = st.inspect(path)
        status["problems"].extend(info.problems)
        status["verified"] = info.ok
        status["tensorCount"] = info.tensor_count
        status["dtypeCounts"] = info.dtype_counts
        status["biggest"] = [{"name": t.name, "shape": t.shape, "dtype": t.dtype}
                             for t in info.biggest(5)]

    actual_gib = round(status["bytes"] / 1024 ** 3, 2)
    status["actualGiB"] = actual_gib
    status["sizeMismatch"] = False
    if isinstance(expected_gib, (int, float)) and float(expected_gib) > 0:
        # 大小偏差 >5% ⇒ **可疑**（下载中断/下错文件都会这样 ✓）
        # ⚠️ 刻意与「结构损坏」分开：结构问题（截断/头部矛盾）会**阻断**就绪 ✗，
        #    大小偏差只进 `suspectRequired`（因为社区量化版/重新导出会让体积合法地变 ✓）。
        delta = abs(actual_gib - float(expected_gib)) / float(expected_gib)
        if delta > 0.05:
            status["sizeMismatch"] = True
            status["problems"].append(
                f"大小与清单不符：实测 {actual_gib} GiB vs 清单 {expected_gib} GiB"
                f"（偏差 {delta * 100:.1f}% ⇒ 可能没下完或下错 ⚠️）")
    return status


def readiness(stage: str = "h3", *, root: Path | None = None,
              capacity_gib: float = DEFAULT_CAPACITY_GIB) -> dict[str, Any]:
    """某阶段的**就绪报告** ✓：必需组件齐不齐、缺多少、显存够不够估。

    ``required`` 由清单的 ``required`` 字段决定 ✓；非必需组件也会列出来（可选件 ✓），
    但**不影响** ``ready`` 判定 ✓。
    """
    catalog = load_catalog()
    rule = STAGE_FILTERS.get(stage)
    entries = catalog["models"]
    if rule:
        entries = [item for item in entries if item.get("category") == rule["category"]]

    components = [component_status(entry, root) for entry in entries]
    required = [item for item in components if item["required"]]
    missing = [item for item in required if not item["present"]]
    #: **阻断**就绪：结构上不成立（截断 / 头部矛盾 / 读不了 ✓）
    #: ⚠️ 这里**不能**再排除 ``sizeMismatch`` ✗ —— 初版加了 `and not sizeMismatch`，
    #:    结果「截断的文件」因为同时「大小也不符」而被漏出断障名单 ✗（自检 ⑳ 当场抓到 ✓）。
    #:    正确语义：**结构坏 ⇒ 一律阻断**；只有**结构完好**的文件才可能"仅可疑" ✓。
    broken = [item for item in required if item["present"] and not item["verified"]]
    #: **不阻断**：结构完好但大小与清单不符（可疑 ⚠️ —— 社区量化版/重新导出会让体积合法地变 ✓）
    suspect = [item for item in required
               if item.get("sizeMismatch") and item["verified"]]
    optional_missing = [item["key"] for item in components
                        if not item["required"] and not item["present"]]

    weights_bytes = sum(int(item["bytes"]) for item in components if item["required"])
    weights_gib = round(weights_bytes / 1024 ** 3, 2)
    return {
        "stage": stage,
        "catalogError": catalog.get("error"),
        "modelsDir": str(root or models_dir() or ""),
        "ready": not missing and not broken,
        "components": components,
        "missingRequired": [item["key"] for item in missing],
        "brokenRequired": [item["key"] for item in broken],
        "suspectRequired": [item["key"] for item in suspect],
        "optionalMissing": optional_missing,
        "requiredWeightsGiB": weights_gib,
        "vram": estimate_vram(weights_bytes, capacity_gib=capacity_gib),
        "catalogNodes": len(catalog.get("nodes") or []),
    }


def estimate_vram(weights_bytes: int, *, activation_bytes: int = 0,
                  safety: float = 1.25, capacity_gib: float = DEFAULT_CAPACITY_GIB) -> dict[str, Any]:
    """显存粗估 ✓ —— **权重是精确值，激活是调用方给的，乘安全系数后才是"估"** ✓。

    ⚠️ 诚实边界：真正的峰值还取决于**实现细节**（offload 策略、注意力算法、分块解码 ✓）⇒
    这里给的是「**下界 + 安全系数**」的量级判断 ✓，**不是承诺** ✗。跑起来后用 ``nvidia-smi`` 实测为准 ✓。
    """
    total = int((int(weights_bytes) + max(0, int(activation_bytes))) * max(1.0, float(safety)))
    total_gib = round(total / 1024 ** 3, 2)
    return {
        "weightsGiB": round(int(weights_bytes) / 1024 ** 3, 2),
        "activationGiB": round(max(0, int(activation_bytes)) / 1024 ** 3, 2),
        "safety": safety,
        "estimatedGiB": total_gib,
        "capacityGiB": capacity_gib,
        "fits": total_gib <= capacity_gib,
        "headroomGiB": round(capacity_gib - total_gib, 2),
        "disclaimer": "估算：权重精确、激活由调用方给；峰值还取决于 offload/注意力/分块实现 ✗ ⇒ 实测为准 ✓",
    }
