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
* 校验：:mod:`app.services.engine.safetensors`（纯 Python ✓）与
  :mod:`app.services.engine.gguf`（纯 Python ✓，2026-09-20 起支持 ✓）。
* H3 键名核对：:mod:`app.services.engine.h3_keys`（纯 Python ✓ torch-free ✓
  2026-09-20 起支持 ✓）—— 文件形态是 H3 时（`h3_form.looks_like_h3_form` ✓），
  直接对照参考结构核「键全集 + 形状关系」✓（真权重到手**插上就验** ✓，
  不用先建 50 层模型 ✗）。

⚠️ GGUF 此前只报「在/不在 + 大小」✗（读取器未实现 ✗）—— **已补齐** ✓（2026-09-20 ✓）：
现在 ``.gguf`` 走 :mod:`.gguf` 真体检（张量表 / 截断 / 量化方案 ✓）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..local_model_scan import get_model_paths
from . import gguf as gguf_mod
from . import h3_form
from . import h3_keys
from . import safetensors as st
from . import tiers as tiers_mod

__all__ = [
    "DEFAULT_CAPACITY_GIB",
    "component_status",
    "estimate_vram",
    "load_catalog",
    "readiness",
    "upscale_status",
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


def _audit_h3_form(status: dict[str, Any], tensors: Mapping[str, Sequence[int] | None]) -> None:
    """文件形态是 H3 时（`h3_form.looks_like_h3_form` ✓）做**键名核对** ✓（2026-09-20 起）。

    好处：真权重到手那一刻就能答「这份 checkpoint 能不能装进 ``H3FormTrunk``」✓ ——
    不用先建 50 层模型再让 ``load_module_weights`` 报 missing ✗。
    核不通过（缺键 / 形状不符 / 结构矛盾）⇒ 进 ``problems`` ⇒ **阻断**就绪 ✓。
    """
    if not tensors or not h3_form.looks_like_h3_form(set(tensors)):
        return
    try:
        audit = h3_keys.audit_h3_checkpoint(tensors)
    except Exception as err:  # noqa: BLE001 —— 核对器自身炸了也是「结论」的一种 ✓
        status["problems"].append(f"H3 键名核对器异常：{err}")
        return
    status["h3Audit"] = audit.to_dict()
    status["problems"].extend(audit.problems)
    for item in audit.missing[:6]:
        status["problems"].append(f"H3 键名核对：缺 `{item}` ✗")
    for item in audit.shape_mismatch[:4]:
        status["problems"].append(
            f"H3 键名核对：`{item['key']}` 形状 {item['got']} ≠ 期望 {item['expected']} ✗")
    if not audit.ok:
        status["verified"] = False


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
        # ⚠️ 2026-09-20 起有**真读取器** ✓（此前只报「在/不在 + 大小」✗）⇒ 与 safetensors
        #    同一口径：结构坏 ⇒ 阻断 ✗；量化方案名来自 ``general.file_type`` ✓。
        info = gguf_mod.inspect(path)
        status["problems"].extend(info.problems)
        status["verified"] = info.ok
        status["tensorCount"] = info.tensor_count
        status["dtypeCounts"] = info.dtype_counts
        status["quantScheme"] = info.quant_scheme
        status["biggest"] = [{"name": t.name, "shape": t.shape, "dtype": t.type_name}
                             for t in info.biggest(5)]
        _audit_h3_form(status, {name: tensor.shape for name, tensor in info.tensors.items()})
    else:
        info = st.inspect(path)
        status["problems"].extend(info.problems)
        status["verified"] = info.ok
        status["tensorCount"] = info.tensor_count
        status["dtypeCounts"] = info.dtype_counts
        status["biggest"] = [{"name": t.name, "shape": t.shape, "dtype": t.dtype}
                             for t in info.biggest(5)]
        _audit_h3_form(status, {name: tensor.shape for name, tensor in info.tensors.items()})

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


def upscale_status(*, root: Path | None = None, key: str = "", path: str = "") -> dict[str, Any]:
    """**超清放大器**的可用性 ✓（2026-09-24 接 ✓）—— 让「超清到底能不能开」在**跑之前**就看得见 ✓✗。

    ⚠️ 三段都要**能说清「没查」** ✗✗（本仓铁律：**没查 ≠ 通过** ✓）：
    1. 没给 ``path`` / ``key`` ⇒ ``checked=False`` ✓ + 说明**怎么让它可查** ✓（不是通过 ✓✗）；
    2. 清单里没有这个组件 ⇒ ``checked=True`` + ``present=False`` ✓ + **期望的格式串** ✓
       （不然只能靠猜要什么权重 ✓✗）；
    3. 权重在盘上 ⇒ 读它的**内嵌契约** ✓：读不出来 ⇒ 给**具名**原因 ✓（回退普通模式是有理由的 ✓）。
    """
    from . import upscale as upscale_mod   # 局部 import ✓：只有这条查询路需要它 ✓（避免清单层拉着引擎层 ✓）

    resolved = str(path or "").strip()
    if not resolved and key:
        entry = next((item for item in load_catalog()["models"] if item.get("key") == key), None)
        if entry is not None:
            found = component_path(entry, root)
            resolved = "" if found is None else str(found)
    report: dict[str, Any] = {
        "checked": False, "present": False, "key": str(key or ""), "path": resolved or None,
        "contract": None, "contractError": None, "format": upscale_mod.CHECKPOINT_FORMAT,
        "requiresMaxTileAbove2x": True, "notes": [],
    }
    if not resolved:
        report["notes"].append(
            "**没查** ✗：没给放大器权重（``upscale.path`` / ``upscale.key`` 或清单里的组件 ✓）"
            "⇒ 超清计划会**回退普通模式**并给出理由 ✓ —— ⚠️ 别把这条读成「超清可用」✗✗")
        return report
    target = Path(resolved)
    if not target.exists():
        report.update({"checked": True, "notes": [
            f"清单/入参指向的放大器权重**不在盘上** ✗：{target} ⇒ 超清会回退普通模式 ✓（把文件放上去再查 ✓）"]})
        return report
    report["checked"] = True
    report["present"] = True
    try:
        metadata, _header = st.read_header(target)
        contract = upscale_mod.read_upscaler_contract(metadata)
    except Exception as err:  # noqa: BLE001 —— 读不了/契约不合法 ⇒ **具名拒绝** ✓ 不猜 ✓
        contract = None
        report["contractError"] = f"{type(err).__name__}: {err}"
    report["contract"] = contract
    if contract is None:
        report["notes"].append(
            "内嵌契约读不出来 ✗ ⇒ ``plan_upscale`` 会**回退普通模式**（理由就是上面那条 ✓）"
            "—— ⚠️ 不是「文件坏了」的笼统话：真因在 ``contractError`` 里 ✓✗")
    else:
        report["notes"].append(
            f"契约 OK ✓（``in_channels={contract['base_config']['in_channels']}`` ✓、"
            f"``width={contract['config']['width']}`` / ``heads={contract['config']['heads']}`` ✓）"
            f"⇒ 2× 可做 ✓；⚠️ **>2× 要调用方显式给 ``maxTile``** ✗（本仓不猜安全块大小 ✓）")
    report["bytes"] = target.stat().st_size
    return report


def readiness(stage: str = "h3", *, root: Path | None = None,
              capacity_gib: float = DEFAULT_CAPACITY_GIB,
              upscale_key: str = "", upscale_path: str = "") -> dict[str, Any]:
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
        # ⭐ 档位表随就绪报告一起给 ✓（2026-09-24 接的 ✓）：步数 / 分辨率 / 加速件 /
        #    显存建议 / **必备模型** ✓ —— ⚠️ 没给容量时各档的 advised 是「**没比**」✗ 不是通过 ✓。
        "tiers": tiers_mod.summary(gpu_gib=capacity_gib if capacity_gib else None),
        # ⭐ 超清放大器可用性 ✓（2026-09-24 接的 ✓）：⚠️ 没给 key/path ⇒ ``checked=False``
        #    **不是**「超清可用」✗✗（本仓铁律：没查 ≠ 通过 ✓）。
        "upscale": upscale_status(root=root, key=upscale_key, path=upscale_path),
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
