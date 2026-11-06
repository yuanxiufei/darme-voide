"""**画布插件契约** —— 插件清单的形状、校验、内置注册表。

源：``infinite-canvas/plugins/canvas/sdk/src/types.ts``（14 KB 的 React 契约）+
``{html,markdown,panorama,sticky-note,svg,template}/src/index.tsx``（6 个 JSON 插件）
⇒ **本文件按本仓口径重写** ✓。

## 重写了什么，没重写什么

* **契约形状 → Python**（本文件）：id / 节点 type 命名、尺寸界、metadata 归属、
  内置面板互斥、批量唯一性。这些是**机器能判的**，所以放后端、可单测、不依赖前端。
* **渲染组件 → 前端 ``components/canvas/plugins/``**：源的 ``Content`` 是 React 组件，
  本仓重写成 5 个 ``.vue``（HTML / Markdown / 全景 / 便利贴 / SVG），
  名字由清单的 ``contentComponent`` 指定，与 ``utils/canvasPlugins.ts`` 逐项对拍。
* ``template`` 插件**不是能力**：它是源项目给插件作者的起步示例（注释原文「改我」），
  迁进本仓只会变成一份没人用的样板 ✗ ⇒ 不登记，只在这里记一笔。

## 与源的四处必须改的口径

====================  ===============================  ======================================
项                    源                               本仓
====================  ===============================  ======================================
坐标/尺寸             ``position:{x,y}`` + ``width/height``  ``x/y/w/h`` 平铺（同 ``canvas_ops``）
内置节点类型          含 ``audio``                      ``text/image/video/audio/config/group``（2026-11-06 起齐 ✓）
节点数据              ``node.metadata.*`` 一整袋        本仓节点**平铺字段** + ``values``；
                                                       插件自定义键一律落 ``values``（字符串键值）
图标                  ``ReactNode``                    **字符串**（emoji 或图标名）—— Vue 侧不能吃 React 节点 ✗
运行时注入            ``React``/``jsx``/``Fragment``   **不注入**（本仓是 Vue，注 React 会装两份框架 ✗）
====================  ===============================  ======================================

## 宿主渲染槽（已接）

本仓 ``CanvasNodeType`` 原是**封闭联合**，画布节点渲染在 ``CanvasNodeCard.vue`` 里按类型分支。
接入时按顺序动了三处**画布核心**：

① 类型从封闭联合放开成「内置五类 ∪ ``<pluginId>:<name>``」的模板字面量类型
   （不是裸 ``string`` —— 那样 ``NODE_TYPE_LABEL[type]`` 会静默返回 ``undefined`` ✗）；
② 卡片正文加插件槽：``PluginNodeBody`` 按清单里的 ``contentComponent`` 取组件，
   三态（正常 / 登记了但缺组件 / 未安装）都必须画出来，不许留白；
③ 插件自定义键统一落 ``node.values``（不造 ``metadata`` 层），
   由本文件的 ``RESERVED_METADATA_KEYS`` 挡住「插件键占用平铺真字段」。

``registry()["hosted"]`` 因此如实为 ``True``；未登记的插件类型**保留节点数据**并显示
「未安装的插件」，导入别家画布时不会凭空吞节点 ✓。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from . import canvas_ops

__all__ = [
    "BUILTIN_PLUGINS",
    "PLUGIN_ID_PATTERN",
    "RESERVED_METADATA_KEYS",
    "SIZE_MAX",
    "SIZE_MIN",
    "registry",
    "validate_bundle",
    "validate_plugin",
]

#: 插件 id / 节点名都只准 kebab-case（与源建议一致，且能直接当 DOM data 属性 ✗ 不必转义）
PLUGIN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
#: 节点 type 必须是 ``<pluginId>:<name>``
NODE_TYPE_PATTERN = re.compile(r"^([a-z0-9][a-z0-9-]*):([a-z0-9][a-z0-9-]+)$")
#: 语义化版本（源 6 个插件全是 x.y.z）
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

#: 节点尺寸界：小于 120 放不下标题，大于 2000 会把画布挤爆
SIZE_MIN = 120.0
SIZE_MAX = 2000.0

#: **插件不许占用**的 metadata 键 —— 它们是本仓 ``CanvasNode`` 的**平铺真字段**，
#: 插件往这里写会与节点自身的状态打架（改 x 改不到、读 w 读到旧值 ✗）
RESERVED_METADATA_KEYS: frozenset[str] = frozenset({
    "id", "type", "title", "x", "y", "w", "h",
    "text", "url", "groupId", "workflowId", "variant",
    "values", "runMode", "engineStage",
})

#: 源 6 个插件里 5 个是真能力（``template`` 是起步样板，不迁）
BUILTIN_PLUGINS: tuple[dict, ...] = (
    {
        "id": "html",
        "name": "HTML 节点",
        "version": "1.2.0",
        "description": "沙箱 iframe 渲染 HTML，支持 {{input}} 注入上游文本",
        "nodes": [{
            "type": "html:render",
            "title": "HTML",
            "icon": "🌐",
            "description": "沙箱渲染 HTML",
            "defaultSize": {"w": 420, "h": 320},
            "defaultMetadata": {"content": ""},
            "minimapColor": "#ec4899",
            "hidePanel": True,
            "interactionToggle": True,
            "contentComponent": "HtmlNodeContent",
            "toolbar": ["html-toggle-edit"],
        }],
    },
    {
        "id": "markdown",
        "name": "Markdown 节点",
        "version": "1.1.0",
        "description": "在画布中编辑与渲染 Markdown",
        "nodes": [{
            "type": "markdown:doc",
            "title": "Markdown",
            "icon": "📝",
            "description": "编辑与渲染 Markdown",
            "defaultSize": {"w": 360, "h": 300},
            "defaultMetadata": {"content": ""},
            "minimapColor": "#6366f1",
            "hidePanel": True,
            "interactionToggle": True,
            "contentComponent": "MarkdownNodeContent",
            "toolbar": ["md-toggle-edit"],
        }],
    },
    {
        "id": "panorama",
        "name": "3D 全景节点",
        "version": "1.1.0",
        "description": "查看 360° 等距柱状全景图，支持上传与 AI 生成，可从上游图片节点取图",
        "nodes": [{
            "type": "panorama:viewer",
            "title": "3D 全景",
            "icon": "🌐",
            "description": "360° 全景查看器（上传 / AI 生成）",
            "defaultSize": {"w": 480, "h": 300},
            "defaultMetadata": {},
            "minimapColor": "#0ea5e9",
            "interactionToggle": True,
            # 复用宿主内置生成面板；本仓模式 text/image/video/audio（2026-11-06 起齐了 ✓）
            "builtinPanel": {"mode": "image", "writeBackToSelf": True},
            "contentComponent": "PanoramaNodeContent",
            "toolbar": ["panorama-generate", "panorama-reset"],
        }],
    },
    {
        "id": "sticky-note",
        "name": "便利贴节点",
        "version": "1.1.0",
        "description": "可自选颜色、双击编辑、拖动即可移动的便利贴",
        "nodes": [{
            "type": "sticky-note:note",
            "title": "便利贴",
            "icon": "📌",
            "description": "彩色便利贴",
            "defaultSize": {"w": 240, "h": 200},
            "defaultMetadata": {"content": "", "pluginColor": "#fde68a"},
            "minimapColor": "#f59e0b",
            "hidePanel": True,
            "contentComponent": "StickyNoteNodeContent",
            "toolbar": [],
        }],
    },
    {
        "id": "svg",
        "name": "SVG 节点",
        "version": "1.1.0",
        "description": "透明背景渲染 SVG 矢量图，可接收上游文本节点的 SVG 源码",
        "nodes": [{
            "type": "svg:vector",
            "title": "SVG",
            "icon": "🔷",
            "description": "渲染 SVG 矢量图",
            "defaultSize": {"w": 320, "h": 320},
            "defaultMetadata": {},
            "minimapColor": "#14b8a6",
            "transparentBackground": True,
            "contentComponent": "SvgNodeContent",
            "toolbar": [],
        }],
    },
)


def _check_size(node: dict, where: str, errors: list[str]) -> None:
    size = node.get("defaultSize")
    if not isinstance(size, dict):
        errors.append(f"{where}.defaultSize 要是对象 {{w, h}}")
        return
    # 源用 {width,height} ⇒ 搬到本仓必须改成 {w,h}；这里两种都收，但只认数值
    width = size.get("w", size.get("width"))
    height = size.get("h", size.get("height"))
    for label, value in (("w", width), ("h", height)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}.defaultSize.{label} 要是数字")
        elif not SIZE_MIN <= float(value) <= SIZE_MAX:
            errors.append(f"{where}.defaultSize.{label}={value} 越界（{SIZE_MIN:.0f}–{SIZE_MAX:.0f}）")


def validate_plugin(plugin: Any) -> list[str]:
    """校验**单个**插件清单，返回错误列表（空 = 合法）✓。

    每条检查都对应一个真会出问题的场景，不是凑数：
    """
    errors: list[str] = []
    if not isinstance(plugin, dict):
        return ["插件清单要是对象"]

    plugin_id = plugin.get("id")
    if not isinstance(plugin_id, str) or not PLUGIN_ID_PATTERN.match(plugin_id):
        errors.append(f"id 必须是 kebab-case：{plugin_id!r}")
        return errors  # id 不合法 ⇒ 后面每条都与它相关，不再叠加噪音

    # ① 插件 id 不能与内置节点类型同名：否则「text 插件」与内置文本节点在渲染与菜单里分不清 ✗
    if plugin_id in canvas_ops.NODE_TYPES:
        errors.append(f"id 与内置节点类型冲突：{plugin_id}")

    for field in ("name", "version"):
        value = plugin.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} 必填")
    version = plugin.get("version")
    if isinstance(version, str) and version.strip() and not SEMVER_PATTERN.match(version):
        errors.append(f"version 要是 x.y.z：{version!r}")

    nodes = plugin.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes 要是非空数组")
        return errors

    seen: set[str] = set()
    for index, node in enumerate(nodes):
        where = f"nodes[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{where} 不是对象")
            continue
        node_type = node.get("type")
        if not isinstance(node_type, str):
            errors.append(f"{where}.type 必填")
            continue
        # ② 必须带命名空间：两个插件抢同一个裸 type 时，后者会静默覆盖前者 ✗
        match = NODE_TYPE_PATTERN.match(node_type)
        if not match:
            errors.append(f"{where}.type 要写成 <pluginId>:<name>：{node_type!r}")
        elif match.group(1) != plugin_id:
            errors.append(f"{where}.type 的命名空间 {match.group(1)!r} ≠ 插件 id {plugin_id!r}")
        # ③ 不能撞本仓内置节点类型
        if node_type in canvas_ops.NODE_TYPES:
            errors.append(f"{where}.type 与内置节点类型冲突：{node_type}")
        if node_type in seen:
            errors.append(f"{where}.type 在同一个插件里重复：{node_type}")
        seen.add(node_type)

        if not isinstance(node.get("title"), str) or not node["title"].strip():
            errors.append(f"{where}.title 必填")
        # ④ 图标必须是字符串：源的 ReactNode 在 Vue 侧吃不下 ✗
        icon = node.get("icon")
        if not isinstance(icon, str) or not icon.strip():
            errors.append(f"{where}.icon 要是非空字符串（emoji 或图标名）")

        _check_size(node, where, errors)

        # ⑤ 默认 metadata 不能占本仓平铺字段：写了也读不到，还会与节点状态打架 ✗
        metadata = node.get("defaultMetadata") or {}
        if not isinstance(metadata, dict):
            errors.append(f"{where}.defaultMetadata 要是对象")
        else:
            grabbed = sorted(set(metadata) & RESERVED_METADATA_KEYS)
            if grabbed:
                errors.append(f"{where}.defaultMetadata 占了节点真字段：{', '.join(grabbed)}")

        # ⑥ 「不要面板」与「用内置面板」同时给 = 自相矛盾 ✗
        if node.get("hidePanel") and node.get("builtinPanel"):
            errors.append(f"{where} 同时声明 hidePanel 与 builtinPanel，语义矛盾")

        # ⑦ 内置面板的模式必须是本仓支持的（判据读 GENERATION_MODES ✓，别手抄名单 ✗）
        panel = node.get("builtinPanel")
        if panel is not None:
            if not isinstance(panel, dict):
                errors.append(f"{where}.builtinPanel 要是对象")
            else:
                mode = panel.get("mode")
                if mode not in canvas_ops.GENERATION_MODES:
                    errors.append(
                        f"{where}.builtinPanel.mode 不支持：{mode!r}"
                        f"（本仓只有 {', '.join(canvas_ops.GENERATION_MODES)}）"
                    )

        # ⑧ 没有内容组件就没有渲染物 ⇒ 节点是个空框 ✗
        if not isinstance(node.get("contentComponent"), str) or not node["contentComponent"].strip():
            errors.append(f"{where}.contentComponent 必填（Vue 组件名，前端按名注册）")

    return errors


def validate_bundle(plugins: Iterable[Any]) -> dict:
    """校验**一批**插件：单件校验 + 跨件唯一性（id / 节点 type 全局唯一）✓。"""
    errors: list[str] = []
    conflicts: list[str] = []
    ids: dict[str, int] = {}
    types: dict[str, str] = {}
    count = 0

    for index, plugin in enumerate(plugins):
        count += 1
        where = f"plugins[{index}]"
        own = validate_plugin(plugin)
        errors.extend(f"{where}: {item}" for item in own)
        if not isinstance(plugin, dict):
            continue
        plugin_id = plugin.get("id")
        if isinstance(plugin_id, str):
            ids[plugin_id] = ids.get(plugin_id, 0) + 1
        for node in plugin.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_type = node.get("type")
            if not isinstance(node_type, str):
                continue
            if node_type in types and types[node_type] != plugin_id:
                conflicts.append(f"节点类型 {node_type} 同时被 {types[node_type]} 与 {plugin_id} 声明")
            types[node_type] = str(plugin_id)

    duplicates = sorted(name for name, hits in ids.items() if hits > 1)
    if duplicates:
        conflicts.append(f"插件 id 重复：{', '.join(duplicates)}")

    return {"ok": not errors and not conflicts, "errors": errors, "conflicts": conflicts, "count": count}


def registry() -> dict:
    """内置注册表（已校验）—— 给前端/agent 读的插件清单一页 ✓。

    ``nodeTypes`` 是**全部可用的插件节点 type**，前端据此决定卡片渲染槽；
    与内置 ``canvas_ops.NODE_TYPES`` 一起构成画布能渲染的节点全集。
    """
    report = validate_bundle(BUILTIN_PLUGINS)
    node_types = [
        node["type"]
        for plugin in BUILTIN_PLUGINS
        for node in plugin.get("nodes") or []
        if isinstance(node, dict) and isinstance(node.get("type"), str)
    ]
    return {
        "plugins": list(BUILTIN_PLUGINS),
        "nodeTypes": node_types,
        "builtinNodeTypes": list(canvas_ops.NODE_TYPES),
        "validation": report,
        "hosted": True,
        "note": (
            "宿主渲染槽已接：`CanvasNodeCard.vue` 按 `contentComponent` 取 "
            "`components/canvas/plugins/` 里的组件，自定义键落 `node.values`。"
            "未登记的插件类型保留节点数据并显示「未安装的插件」。"
        ),
    }
