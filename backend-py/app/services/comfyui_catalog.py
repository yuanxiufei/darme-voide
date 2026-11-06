"""ComfyUI 工作流目录（``configs/comfyui/workflows``）的**唯一读取口**（自研重写 ✓）。

这一层只做「**读**」：注册表怎么读、图文件怎么取、变体怎么展开、步数怎么估。
**不碰**任何参数注入（那是 :mod:`app.services.comfyui_bindings` 的事 ✓）。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``
（MIT ✓）的 ``comfyui-gateway/src/registry.js`` 与 ``step-plan.js`` ⇒ 这里只搬**语义**
（API/UI 格式判定 ✓、变体展开 ✓、步数估算 ✓），代码**自己写** ✗（不拷源码 ✗）。

## 为什么必须有一个「目录层」

工作流图是**数据**，不是代码 ✓：上游 21 条 entry 共用 18 个图文件（3 个 ``video-h3-*``
共用 ``h3-ref2va-drama-vdn.json`` ✓），且 ``image-qwen21`` 一条 entry 带 4 个变体
（``t2i`` / ``edit`` / ``inpaint`` / ``outpaint`` ✓，各指向不同图 ✓）。
把这些散在调用方里 ⇒ 「同一份图被读成不同形状」「改了图忘了改引用」是必然 ✗。

## 关键判据（都验过 ✓）

1. **API 格式才能 POST** ✗：``{"3": {"class_type": ...}}`` 是 API 格式 ✓；
   ``{"nodes": [...], "links": [...]}`` 是 UI 格式 ✗（只能界面打开，**不能**提交 ✓）。
   两者**互不通用** ✗ ⇒ :func:`load_graph` 读到 UI 格式**必须报**，不许静默往下走 ✓。
2. **变体展开不是替换**（``variants`` 只覆盖自己声明的键 ✓）：
   ``{"file": "image-qwen21-edit-turbo.json", "label": "多图编辑"}`` ⇒ 合并后
   ``label`` 换掉、``bindings``/``parameters`` 仍是主 entry 的 ✓。
3. **步数估算是「多阶段并一条进度条」**（``estimate_step_plan`` ✓）：
   ``KSampler`` 取 ``steps`` ✓；``KSamplerAdvanced`` 取 ``end_at_step - start_at_step`` ✓；
   ``SamplerCustomAdvanced`` / ``SelfLiftH3Sampler`` 要**顺着连线**追到 ``ManualSigmas``
   （步数 = sigma 个数 − 1 ✓）或 ``BasicScheduler`` ✓；``H3SigmaRefiner`` 是
   「上游步数 + ``extra_steps``」✓ ⇒ 这些节点**不认**就等于进度条永远不动 ✗。
4. **递归要有环保护**（2026-11-02 加 ✓）：``H3SigmaRefiner.sigmas`` 是连线 ⇒
   顺着追可能绕回自己 ✗（源实现没有防护 ✓，图一旦成环就是死循环 ✗）⇒ 用 ``_seen`` 截断 ✓。
5. **不许路径穿越** ✗：图文件名来自注册表（可被编辑 ✓）⇒ :func:`graph_path` 拒绝
   含 ``..`` / 绝对路径的 ``file`` ✓。

运行::

    from app.services.comfyui_catalog import entries, load_graph
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .comfyui_ui_workflow import UiWorkflowError, convert_ui_workflow, is_ui_workflow

from ..core.config import PROJECT_ROOT

__all__ = [
    "CATALOG_DIR",
    "CatalogError",
    "REGISTRY_NAME",
    "entries",
    "estimate_step_plan",
    "get_entry",
    "graph_path",
    "is_api_format",
    "iter_parameter_keys",
    "load_graph",
    "load_registry",
    "resolve_preset",
    "resolve_variant",
    "variant_ids",
]

#: 工作流目录（注册表 + 图文件都在这一层 ✓）
CATALOG_DIR = PROJECT_ROOT / "configs" / "comfyui" / "workflows"

#: 注册表文件名
REGISTRY_NAME = "registry.json"

#: 质量档 → 预设键（``presets.quality`` 下的名字 ✓）
QUALITY_KEYS = ("low", "medium", "high")


class CatalogError(Exception):
    """目录层错误：图不存在 / 不是 API 格式 / 变体不在册 / 路径穿越 ✗。

    带 ``code`` 是为了让调用方（HTTP 层）能直接翻译成 4xx，不用再猜字符串 ✓。
    """

    def __init__(self, message: str, *, code: str = "catalog_error", details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """读注册表（每次都读盘 ✓ —— 注册表是**可编辑资产**，缓存会让人改完不生效 ✗）。"""
    target = Path(path) if path else CATALOG_DIR / REGISTRY_NAME
    if not target.is_file():
        raise CatalogError(f"工作流注册表不在盘上：{target}", code="registry_missing")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogError(f"注册表读不出来：{exc}", code="registry_unreadable") from exc
    if not isinstance(data, dict) or not isinstance(data.get("workflows"), list):
        raise CatalogError("注册表结构不对（缺 workflows 列表）", code="registry_malformed")
    return data


def is_api_format(graph: Any) -> bool:
    """是不是 **API 格式**（只有它能 ``POST /prompt`` ✓）。

    判据：是 dict ✓、**没有** ``nodes`` 键（UI 格式标志 ✗）、至少一项带 ``class_type`` ✓。
    """
    if not isinstance(graph, dict) or not graph:
        return False
    if "nodes" in graph:
        return False
    return any(isinstance(node, dict) and "class_type" in node for node in graph.values())


def entries(registry: dict[str, Any] | None = None, *, include_hidden: bool = False) -> list[dict[str, Any]]:
    """列出 entry（默认**排除** ``hidden`` ✓ —— 隐藏项不该出现在可选列表里 ✗）。"""
    data = registry if registry is not None else load_registry()
    rows = [row for row in data.get("workflows", []) if isinstance(row, dict)]
    if include_hidden:
        return rows
    return [row for row in rows if not row.get("hidden")]


def get_entry(workflow_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """按 id 取 entry（找不到返回 ``None`` ✓ —— 由调用方决定「报 404 还是回落」✓）。"""
    for row in entries(registry, include_hidden=True):
        if row.get("id") == workflow_id:
            return row
    return None


def variant_ids(entry: dict[str, Any]) -> tuple[str, ...]:
    """该 entry 可用的变体名（无 ``variants`` 则只有主形态 ✓）。"""
    variants = entry.get("variants")
    if not isinstance(variants, dict):
        return ()
    return tuple(str(key) for key in variants)


def resolve_variant(entry: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    """展开变体：**只覆盖变体自己声明的键** ✓（``bindings``/``parameters`` 沿用主 entry ✓）。

    ``variant`` 为空 ⇒ 返回主形态副本 ✓；变体不在册 ⇒ :class:`CatalogError` ✓
    （静默回落主形态会让「用户选了 inpaint 却跑成 t2i」这种错**不报错** ✗）。
    """
    if not variant:
        return dict(entry)
    variants = entry.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        raise CatalogError(
            f"工作流 {entry.get('id')} 没有变体 {variant}（可用：{', '.join(variant_ids(entry)) or '无'}）",
            code="unknown_variant",
        )
    overlay = variants[variant]
    if not isinstance(overlay, dict):
        raise CatalogError(f"变体 {variant} 不是对象", code="variant_malformed")
    merged = dict(entry)
    merged.update(overlay)
    return merged


def graph_path(entry: dict[str, Any], catalog_dir: str | Path | None = None) -> Path:
    """entry 指向的图文件绝对路径（**拒绝路径穿越** ✓）。"""
    root = Path(catalog_dir) if catalog_dir else CATALOG_DIR
    name = str(entry.get("file") or "").strip()
    if not name:
        raise CatalogError(f"工作流 {entry.get('id')} 没写 file", code="graph_unspecified")
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CatalogError(f"图文件名不合法（不许绝对路径或 ..）：{name}", code="graph_path_escape")
    return root / candidate


def load_graph(
    entry: dict[str, Any],
    variant: str | None = None,
    catalog_dir: str | Path | None = None,
) -> dict[str, Any]:
    """读图并**保证是 API 格式** ✓。

    UI 格式（ComfyUI 界面里「保存」出来的那种）会**就地转成 API 格式** ✓
    （走 :mod:`app.services.comfyui_ui_workflow`）；转不动才报 ``UiWorkflowError`` 的 code ✓。

    返回的图是**新对象**（每次读盘 ✓），调用方可随意改 ✓ —— 注入参数在副本上做，
    不会污染「下一次请求拿到的图」✓。
    """
    resolved = resolve_variant(entry, variant)
    target = graph_path(resolved, catalog_dir)
    if not target.is_file():
        raise CatalogError(f"工作流图不在盘上：{target}", code="graph_missing")
    try:
        graph = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogError(f"工作流图读不出来：{exc}", code="graph_unreadable") from exc
    # UI 格式 ⇒ **就地转换** ✓（从 ComfyUI 界面保存出来的图占大多数，一律拒等于废图 ✗）
    if is_ui_workflow(graph):
        try:
            graph, _warnings = convert_ui_workflow(graph)
        except UiWorkflowError as exc:
            raise CatalogError(str(exc), code=exc.code) from exc
    if not is_api_format(graph):
        raise CatalogError(f"{target.name} 既不是 API 格式也不是 UI 格式，认不出来", code="graph_format_unknown")
    return graph


def resolve_preset(entry: dict[str, Any], quality: str | None = None) -> dict[str, Any]:
    """取质量档预设（``presets.quality`` ✓）。``auto`` / 空 ⇒ ``{}`` ✓（不预设 ✗）。

    ``quality`` 一律 **lower + strip** ✓：上游写 ``"Auto"`` / ``" auto "`` 都要认 ✓。
    """
    key = str(quality or "auto").strip().lower() or "auto"
    presets = entry.get("presets")
    if not isinstance(presets, dict):
        return {}
    bucket = presets.get("quality")
    if not isinstance(bucket, dict):
        return {}
    value = bucket.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int:
    """JS ``positiveInt`` 的口径 ✓：非法/非正一律 ``0``（不是抛错 ✓）。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _linked_node(graph: dict[str, Any], value: Any) -> dict[str, Any] | None:
    """把 ``["7", 0]`` 这种连线解成上游节点对象 ✓（不是连线 ⇒ ``None`` ✓）。"""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    node = graph.get(str(value[0]))
    return node if isinstance(node, dict) else None


def _sigma_steps(graph: dict[str, Any], node: dict[str, Any] | None, node_id: str = "", seen: frozenset[str] = frozenset()) -> int:
    """顺着 sigma 连线数步数 ✓（``ManualSigmas`` / ``BasicScheduler`` / ``H3SigmaRefiner`` ✓）。

    ``node_id`` 是当前节点的 id（**环保护用** ✓）；``seen`` 是已走过的节点 id 集合 ✓。
    """
    if not node:
        return 0
    inputs = node.get("inputs") or {}
    class_type = node.get("class_type")
    if class_type == "ManualSigmas":
        raw = str(inputs.get("sigmas") or "")
        values = []
        for part in raw.replace(",", " ").split():
            try:
                values.append(float(part))
            except ValueError:
                continue
        return max(0, len(values) - 1)
    if class_type == "BasicScheduler":
        return _positive_int(inputs.get("steps"))
    if class_type == "H3SigmaRefiner":
        # 环保护：同一个节点不追第二次（源实现无防护 ⇒ 图成环即死循环 ✗）
        if node_id and node_id in seen:
            return 0
        link = inputs.get("sigmas")
        upstream_id = str(link[0]) if isinstance(link, (list, tuple)) and link else ""
        nxt = seen | {node_id} if node_id else seen
        upstream = _sigma_steps(graph, _linked_node(graph, link), upstream_id, nxt)
        return upstream + _positive_int(inputs.get("extra_steps"))
    return 0


def estimate_step_plan(graph: dict[str, Any]) -> dict[str, Any]:
    """估算一张图的**采样步数分布**，让多阶段工作流并成一条连续进度条 ✓。

    返回 ``{"total": int, "steps": {节点id: 步数}, "absolute": {节点id: True}}``：

    * ``steps`` —— 每个采样节点**自己**这一步要跑多少步 ✓；
    * ``absolute`` —— ``SelfLiftH3Sampler`` 的步数是**全局步号**（低分辨率阶段按自己的
      短总量报，高清阶段从 ``transition_step`` 续 ✓）⇒ 进度条**不能**把它们相加 ✗；
    * ``total`` —— 其余节点的合计（``absolute`` 的那份**已计入**上游 sigma ✓，不重复加 ✗）。
    """
    steps: dict[str, int] = {}
    absolute: dict[str, bool] = {}
    total = 0
    for node_id, node in (graph or {}).items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        class_type = node.get("class_type")
        value = 0
        if class_type == "KSampler":
            value = _positive_int(inputs.get("steps"))
        elif class_type == "KSamplerAdvanced":
            count = _positive_int(inputs.get("steps"))
            start = max(0, _positive_int(inputs.get("start_at_step")))
            end_input = inputs.get("end_at_step")
            end = min(count, _positive_int(end_input)) if end_input is not None else count
            value = max(0, end - start)
        elif class_type in ("SamplerCustomAdvanced", "SelfLiftH3Sampler"):
            link = inputs.get("sigmas")
            upstream_id = str(link[0]) if isinstance(link, (list, tuple)) and link else ""
            value = _sigma_steps(graph, _linked_node(graph, link), upstream_id)
            if class_type == "SelfLiftH3Sampler" and value > 0:
                absolute[str(node_id)] = True
        if value > 0:
            steps[str(node_id)] = value
            total += value
    return {"total": total, "steps": steps, "absolute": absolute}


def iter_parameter_keys(entry: dict[str, Any]) -> Iterable[str]:
    """entry 声明的**可调参数**名（``parameters[].key`` ✓），供前端画参数卡 ✓。"""
    for item in entry.get("parameters") or []:
        if isinstance(item, dict) and item.get("key"):
            yield str(item["key"])
