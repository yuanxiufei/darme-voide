"""**画布工具层** —— 把 LLM 的工具调用编译成本仓画布可执行的 op 列表。

源：``infinite-canvas/canvas-agent/src/canvas/{schemas,operations,tools}.ts`` ⇒
**本文件按本仓口径重写** ✓（不是翻译）。

## 为什么不搬

**源与本仓的画布数据模型不是一套**：

====================  ==============================  ================================================
项                    源                              本仓（``composables/useCanvas.ts``）
====================  ==============================  ================================================
节点类型              ``image/text/config/video/audio``  ``text/image/video/audio/config/group``（六类全齐 ✓）
坐标                  ``position:{x,y}`` + ``width/height``  ``x/y/w/h`` 平铺
视口                  ``{x,y,k}``                     ``{x,y,scale}``（夹 0.2–3）
正文                  ``metadata.content``            ``text`` 字段
配置参数              ``metadata.{model,size,...}``   ``values: Record<string,string>`` + ``workflowId``
生成通道              ``mode: text/image/video/audio``  ``runMode: engine/comfyui`` + ``engineStage``
====================  ==============================  ================================================

照抄**不报错，只静默写坏画布**：本仓 ``normalizeDoc`` 只认 ``w/h``，一个带 ``position`` 的
op 落地后宽高为 0 ⇒ 被当坏节点丢掉，用户看不见刚让 AI 建的东西 ✗。

## 三条与本仓对齐的口径

1. 尺寸与 ``utils/canvasNodeSize.ts::DEFAULT_NODE_SIZE`` 逐项一致（自己拍一套会让同一张图被压扁 ✗）。
2. 视口 ``scale`` 夹 0.2–3，与 ``useCanvas.zoomAt`` 同界。
3. **整批校验、整批拒**（与 ``importDoc(strict)`` 同口径）：半批落地比不执行更难收拾 ✗。

## 谁执行

本文件**只产出 op，不碰画布**。执行在前端 ``utils/canvasOps.ts``（要写 ``ref`` 触发 Vue 更新）；
后端若自己改 localStorage 等于绕开状态层 ✗。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

__all__ = [
    "DEFAULT_NODE_SIZE",
    "DROPPED_TOOLS",
    "GENERATION_MODES",
    "MAX_SCALE",
    "MIN_SCALE",
    "NODE_TYPES",
    "OP_TYPES",
    "PLUGIN_TYPE_PATTERN",
    "REFERENCE_PATTERN",
    "SCENES",
    "SCENE_TOOL_SETS",
    "TOOL_NAMES",
    "build_request",
    "compact_snapshot",
    "next_canvas_x",
    "node_type_problem",
    "registered_plugin_node_types",
    "scene_problems",
    "tool_schemas",
    "tools_for_scene",
    "validate_ops",
]

# ---------------------------------------------------------------- 与 useCanvas 同源的常量

#: 节点类型（**与 ``useCanvas.CanvasNodeType`` 逐项一致** ⚠️）
#:
#: 2026-11-06 补 ``audio`` ✓：自研引擎的 Music3 档（`stage="music3"`）产物是 **wav** ✓，
#: 而画布原来只认图片/视频 ⇒ 歌跑完了**没有卡能装它** ✓✗（前端按 `.wav` 判成 image ⇒
#: 一张裂图 ✗）。补这一类不是为了对齐源，是引擎真出了音频 ✓。
NODE_TYPES: tuple[str, ...] = ("text", "image", "video", "audio", "config", "group")
#: 生成模式（四类 ✓）。未知模式**如实拒绝**，不静默降级 ✗ —— 源把未知归成 image，
#: 那会让「跑音乐」变成「跑图」且不报错 ✗。
GENERATION_MODES: tuple[str, ...] = ("text", "image", "video", "audio")
#: op 类型（与前端 ``canvasOps.ts`` 执行器一一对应）
OP_TYPES: tuple[str, ...] = (
    "add_node",
    "update_node",
    "delete_nodes",
    "delete_edges",
    "connect_nodes",
    "select_nodes",
    "set_viewport",
    "run_generation",
)

#: 默认尺寸 —— **与 ``utils/canvasNodeSize.ts::DEFAULT_NODE_SIZE`` 逐项一致** ⚠️
DEFAULT_NODE_SIZE: dict[str, tuple[float, float]] = {
    "text": (340.0, 240.0),
    "image": (340.0, 240.0),
    "video": (420.0, 236.0),
    #: 音频没有画面 ⇒ 不吃 16:9 的高度 ✓；但播放器要放得下（控件条 ＋ 地址行 ＋ 按钮 ✓）
    "audio": (420.0, 180.0),
    "config": (340.0, 240.0),
    "group": (760.0, 480.0),
}

NODE_GAP = 80.0        # 新节点落在现有节点最右缘再留一道间距（源 nextCanvasX 同口径）
FLOW_DX = 420.0        # 提示词节点 → 配置节点 的水平间距
TEXT_ROW_H = 240.0     # 竖排行距基准
TEXT_COL_W = 340.0     # 横排列距基准
MIN_SCALE = 0.2        # 与 useCanvas.MIN_SCALE 一致
MAX_SCALE = 3.0        # 与 useCanvas.MAX_SCALE 一致
SNAPSHOT_TEXT_LIMIT = 240   # 快照里正文超过这个长度就截断
SNAPSHOT_TEXT_KEEP = 120    # 截断时保留的字符数
REFERENCE_PATTERN = re.compile(r"@\[node:([\w-]+)\]")
#: 插件节点 type 形态 —— **与 ``utils/canvasPlugins.ts::PLUGIN_TYPE_PATTERN`` 同规则** ⚠️
PLUGIN_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9-]+$")

# ---------------------------------------------------------------- 工具清单（源 34 → 本仓 22，另 12 条逐条写理由不迁）

_P_GENERATE = {
    "prompt": "提示词",
    "title": "提示词节点标题，可省",
    "x": "x",
    "y": "y",
    "referenceNodeIds": "参考节点 id 数组，可省",
    "workflowId": "工作流 id",
    "values": "工作流参数（键值一律字符串）",
}

TOOLS: dict[str, tuple[str, dict[str, str]]] = {
    "canvas_get_state": ("读取当前画布：节点（id/类型/标题/坐标/尺寸/正文摘要/连线）与视口。**动手前先读一次**。", {}),
    "canvas_get_selection": ("读取当前选中的节点 id。", {}),
    "canvas_export_snapshot": ("导出画布快照（压缩版），用于理解已有布局。", {}),
    "canvas_apply_ops": (
        "批量执行画布操作。ops 支持 add_node、update_node、delete_nodes、delete_edges、"
        "connect_nodes、select_nodes、set_viewport、run_generation。**整批校验，任一条不合法整批不执行**。",
        {"ops": "操作数组（见八种 type）"},
    ),
    "canvas_create_node": (
        "创建任意类型节点：text、image、video、audio、config、group，或已登记的插件节点"
        "（`<pluginId>:<name>`，如 sticky-note:note，清单见 /api/v1/canvas/plugins）。"
        "图片/视频/音频用 url 给资源地址；配置节点用 workflowId + values 指定工作流与参数。",
        {
            "nodeType": "text/image/video/audio/config/group 或 <pluginId>:<name>",
            "title": "标题，可省",
            "x": "世界坐标 x，可省（默认落在现有节点右侧）",
            "y": "世界坐标 y，可省（默认 0）",
            "w": "宽，可省（按类型取默认）",
            "h": "高，可省（按类型取默认）",
            "text": "text 节点的正文，可省",
            "url": "image/video/audio 的资源地址，可省",
        },
    ),
    "canvas_create_text_node": ("创建一个文本节点。省略 x 时自动落在画布右侧。",
                                {"text": "正文", "title": "标题", "x": "x", "y": "y", "w": "宽", "h": "高"}),
    "canvas_create_text_nodes": ("批量创建文本节点，按 row（横排）或 column（竖排）自动排版。",
                                 {"items": "文本节点数组（每项 text/title/x/y/w/h）", "x": "起始 x",
                                  "y": "起始 y", "gap": "间距，默认 40", "direction": "row 或 column，默认 row"}),
    "canvas_create_config_node": (
        "创建配置节点（工作流 + 参数）。给了 autoRun 就顺手跑一次生成。",
        {"prompt": "提示词（音乐档是曲风描述）", "mode": "text/image/video/audio，默认 image",
         "title": "标题", "x": "x", "y": "y",
         "w": "宽", "h": "高", "autoRun": "是否立即生成", "workflowId": "绑定的工作流 id",
         "values": "工作流参数（键值一律字符串；音乐档用 lyrics / seconds）",
         "engineStage": "自研引擎档位，可省", "runMode": "engine 或 comfyui，默认 engine"},
    ),
    "canvas_create_generation_flow": (
        "创建一次生成的完整链路：提示词节点 → 配置节点 → 连线，并选中配置节点。"
        "给了 referenceNodeIds 会把参考图也连上去。",
        {"prompt": "提示词", "mode": "text/image/video/audio，默认 image", "title": "提示词节点标题",
         "x": "x", "y": "y", "referenceNodeIds": "参考图节点 id 数组", "autoRun": "是否立即生成",
         "workflowId": "工作流 id", "values": "参数键值"},
    ),
    "canvas_generate_text": ("直接生成文本：建提示词节点 + 配置节点 + 立刻跑。", _P_GENERATE),
    "canvas_generate_image": ("直接生成图片：建提示词节点 + 配置节点 + 立刻跑。", _P_GENERATE),
    "canvas_generate_video": ("直接生成视频：建提示词节点 + 配置节点 + 立刻跑。", _P_GENERATE),
    #: 源有这一条 ✓（本仓 2026-11-06 补回：Music3 真出歌之后画布才有「音频生成」这条路 ✓）。
    "canvas_generate_audio": (
        "直接生成音频（音乐 / 演唱）：建提示词节点 + 配置节点 + 立刻跑。"
        "曲风描述写在 prompt，歌词与时长写在 values（`lyrics` / `seconds`）。"
        "产物是 wav ⇒ 落进画布的是**音频卡**，不是图片卡 ✗。",
        _P_GENERATE,
    ),
    "canvas_update_node": ("修改节点字段（patch 只覆盖给出的键）。改正文请用 canvas_update_node_text。",
                           {"id": "节点 id", "patch": "要覆盖的字段（x/y/w/h/title/url/workflowId/values…）"}),
    "canvas_update_node_text": ("改文本节点正文（可同时改标题）。",
                                {"id": "节点 id", "text": "新正文", "title": "新标题，可省"}),
    "canvas_move_nodes": ("移动节点。给 x/y 是**绝对位置**，给 dx/dy 是**相对位移**（同时给时 x/y 优先）。",
                          {"items": "数组，每项 {id, x?, y?, dx?, dy?}"}),
    "canvas_resize_node": ("调整节点尺寸（世界坐标单位）。", {"id": "节点 id", "w": "宽", "h": "高"}),
    "canvas_delete_nodes": ("删除节点（挂在它们身上的连线一并删除）。", {"ids": "节点 id 数组"}),
    "canvas_connect_nodes": ("连线。**不许自连、不许重复、不许连不存在的节点**，违反的连线会被丢弃并回报。",
                            {"connections": "数组，每项 {from, to}"}),
    "canvas_select_nodes": ("设置选中节点（给空数组即取消选中）。", {"ids": "节点 id 数组"}),
    "canvas_set_viewport": ("设置视口。scale 会被夹在 0.2–3。", {"x": "平移 x", "y": "平移 y", "scale": "缩放"}),
    "canvas_run_generation": ("让某个配置节点开始生成（节点必须已绑好工作流）。",
                              {"nodeId": "配置节点 id", "mode": "text/image/video/audio，可省",
                               "prompt": "覆盖用提示词，可省"}),
}

#: 源有、本仓**没有宿主**因而未迁移的工具 —— 逐条写理由，别让人以为漏了 ⚠️
DROPPED_TOOLS: dict[str, str] = {
    "site_navigate": "源靠它跳 /canvas/:id、/image、/video、/prompts、/assets；本仓画布是单例（localStorage 一份），"
                     "没有多画布页与工作台页 ⇒ 导航归前端，不该让 LLM 猜路径 ✗",
    "canvas_list_projects": "同上：本仓没有「画布项目」实体（画布不落后端、无 id 可分页）⇒ 无对象可列",
    "canvas_create_attachment_nodes": "源用「附件 id」间接换图；本仓图片节点直接吃 url ⇒ "
                                      "已由 canvas_create_node(nodeType=image, url=…) 覆盖",
    "canvas_create_image_prompt_flow": "与 canvas_create_generation_flow(mode=image) 完全重合 ⇒ 合并",
    "generation_get_status": "本仓任务状态在 /api/v1/generations 与本仓任务表里，属生成域不属画布 op",
    "workbench_image_get_config": "本仓没有独立「生图工作台」页；生成参数就是配置节点的 values ⇒ 读它 = canvas_get_state",
    "workbench_image_generate": "同上，等价于建配置节点 + run_generation",
    "workbench_video_get_config": "本仓没有独立「视频创作台」页",
    "workbench_video_generate": "同上，等价于建配置节点 + run_generation",
    "prompts_search": "本仓提示词库有专用后端（/api/v1/prompts）与专用工具，混进画布 op 会让一次调用跨两个域 ✗",
    "assets_list": "本仓资产库是 libraries 域（角色/场景/服装/道具/武器），有独立 API 与语义",
    "assets_add": "同上；源的 assets 是「素材」，本仓对应 library + images，不是同一个东西",
}

#: 给 LLM 的工具名（顺序稳定：读 → 写 → 生成）
TOOL_NAMES: tuple[str, ...] = tuple(TOOLS)

# ---------------------------------------------------------------- 场景工具面（收敛 + 守卫）

#: 场景名 ✓。**只列真实存在的宿主**：多一个就得多一处调用方，否则是死配置 ✗
SCENES: tuple[str, ...] = ("chat", "canvas")

#: 各场景能看见哪些工具 ✓ —— 与 ``canvas_agent.build_canvas_tools`` 的过滤口径是同一份。
#:
#: 为什么要收敛：22 个工具一次性摊给模型，它在「帮我把这句话变成一张图」时也会去翻
#: ``canvas_move_nodes`` / ``canvas_set_viewport`` 这些它压根不该动的旋钮 ⇒ 步数全花在试探上。
#: 对话场景只留「读 ＋ 整批落地」四件：要精细排版就该切到画布场景，而不是给对话放宽权限 ✓。
#: 画布场景给**全量**（= ``TOOL_NAMES``），因为它就是那张画布的编辑器 ✓。
SCENE_TOOL_SETS: dict[str, frozenset[str]] = {
    "chat": frozenset(
        {
            "canvas_get_state",
            "canvas_get_selection",
            "canvas_export_snapshot",
            "canvas_apply_ops",
        }
    ),
    "canvas": frozenset(TOOL_NAMES),
}


def scene_problems(scene: str) -> list[str]:
    """场景配置自身的毛病 ⇒ 错误文案列表；空表示健康 ✓。

    ⚠️ **必须查的两件事**（都是「静默归零」的前置条件，参考 dramaclaw 的教训）：

    1. 白名单里点名了工具表里**没有**的名字 ⇒ 配置漂移。它不会报错，只会让过滤结果
       悄悄少几条 —— 于是模型该有的能力凭空消失，而日志里一切正常 ✗；
    2. 白名单与工具表**交集为空** ⇒ 模型拿到 0 个工具，仍能正常回话、正常结束，
       表现是「AI 说它做不到」而不是「配置错了」✗。

    两条都当场报错，不允许用「反正返回空表也不崩」敷衍过去 ✓。
    """
    if scene not in SCENE_TOOL_SETS:
        return [f"未知场景：{scene!r}（允许 {', '.join(SCENES)}）"]
    allowed = SCENE_TOOL_SETS[scene]
    problems: list[str] = []
    unknown = sorted(allowed - set(TOOL_NAMES))
    if unknown:
        problems.append(
            f"场景 {scene} 的白名单点名了不存在的工具：{', '.join(unknown)}"
            f"（工具表只有 {len(TOOL_NAMES)} 个，键空间必须同族）"
        )
    if not (allowed & set(TOOL_NAMES)):
        problems.append(
            f"场景 {scene} 的白名单与工具表交集为空 ⇒ 模型会拿到 0 个工具且不报错"
            f"（白名单 {len(allowed)} 条，工具表 {len(TOOL_NAMES)} 条，命名族对不上）"
        )
    return problems


def tools_for_scene(scene: str) -> tuple[str, ...]:
    """场景可见的工具名（保持 ``TOOL_NAMES`` 的顺序）✓；配置有病直接抛 ``ValueError``。

    抛而不是返回空表：空工具表是个**合法但致命**的返回值，调用方无从分辨
    「这个场景就该没有工具」和「我把名字写错了」✗。
    """
    problems = scene_problems(scene)
    if problems:
        raise ValueError("；".join(problems))
    allowed = SCENE_TOOL_SETS[scene]
    return tuple(name for name in TOOL_NAMES if name in allowed)


#: 参数类型表 —— **单一真源**：`/canvas/tools` 与画布 Agent 的工具注册都读这里。
#:
#: 为什么要分类型而不是一律 `string`：模型看到 `ops: {type: "string"}` 会真的把数组
#: **序列化成字符串**再传（"\"[{\\\"type\\\":\\\"add_node\\\"}]\""），于是校验报「不是数组」✗。
#: 但也不能只按名字猜死：`title`/`text` 是字符串、`x` 是数字，都在表里点名。
_ARRAY_PARAMS: frozenset[str] = frozenset({"ops", "items", "ids", "connections", "referenceNodeIds"})
_NUMBER_PARAMS: frozenset[str] = frozenset({"x", "y", "w", "h", "dx", "dy", "gap", "scale"})
_BOOL_PARAMS: frozenset[str] = frozenset({"autoRun"})


def param_schema(key: str, description: str) -> dict[str, Any]:
    """单个参数的 JSON Schema（未登记的名字退回 string —— 与源工具的写法一致）。"""
    if key in _ARRAY_PARAMS:
        return {"type": "array", "description": description, "items": {"type": "object"}}
    if key in _NUMBER_PARAMS:
        return {"type": "number", "description": description}
    if key in _BOOL_PARAMS:
        return {"type": "boolean", "description": description}
    return {"type": "string", "description": description}


def tool_schemas(scene: str | None = None) -> list[dict]:
    """OpenAI/Anthropic 通用的 function-calling 形态（本仓不引 zod，直接给 JSON Schema）✓。

    ``scene`` 给了就只出该场景可见的那几条（顺序同 ``tools_for_scene``）✓；
    不给 = 全量（老调用方的口径不变，不破坏既有契约 ✓）。
    """
    names = list(TOOL_NAMES) if scene is None else list(tools_for_scene(scene))
    return [
        {
            "name": name,
            "description": TOOLS[name][0],
            "parameters": {
                "type": "object",
                "properties": {
                    key: param_schema(key, desc) for key, desc in TOOLS[name][1].items()
                },
            },
        }
        for name in names
    ]


# ---------------------------------------------------------------- 校验


def _num(value: Any, label: str, errors: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} 要是数字，收到 {type(value).__name__}")
        return None
    return float(value)


def _known_ids(snapshot: dict | None) -> set[str]:
    return {str(item["id"]) for item in (snapshot or {}).get("nodes") or [] if item.get("id")}


def registered_plugin_node_types() -> tuple[str, ...]:
    """已登记的插件节点 type ✓（``<pluginId>:<name>``，来自 ``canvas_plugins`` 注册表）。

    **延迟 import**：``canvas_plugins`` 反过来 import 本模块（要内置 ``NODE_TYPES``），
    在模块顶部 import 会成环 ✗。
    """
    from . import canvas_plugins

    return tuple(canvas_plugins.registry()["nodeTypes"])


def node_type_problem(node_type: Any) -> str | None:
    """节点 type 校验 ⇒ 错误文案；``None`` 表示放行 ✓。

    允许：内置五类 ＋ **已登记**的插件 ``<pluginId>:<name>`` ✓。
    形态对但没登记的插件类型**当场拒**：前端会把它画成「未安装的插件」，
    与其让模型编一个幽灵类型，不如报错并列出可用清单 ✓。
    """
    if node_type in NODE_TYPES:
        return None
    if not isinstance(node_type, str) or not PLUGIN_TYPE_PATTERN.match(node_type):
        return (
            f"nodeType 不认识：{node_type!r}"
            f"（允许内置 {', '.join(NODE_TYPES)}，或插件 type <pluginId>:<name>）"
        )
    registered = registered_plugin_node_types()
    if node_type not in registered:
        return f"nodeType 未登记的插件节点：{node_type!r}（已登记 {', '.join(registered) or '无'}）"
    return None


def validate_ops(ops: Any, snapshot: dict | None = None) -> tuple[list[dict], list[str]]:
    """**整批**校验：返回 ``(合法 op 列表, 错误列表)``；有错误调用方应整批拒绝 ✓。

    与前端 ``importDoc(strict)`` 同口径 —— 不执行半批。半批落地后用户看到
    「一半建了、一半没建」，比什么都没发生更难收拾 ✗。
    """
    errors: list[str] = []
    clean: list[dict] = []
    if not isinstance(ops, (list, tuple)):
        return [], ["ops 要是数组"]
    #: 哪些 id 算"存在"：快照里的 + **本批 add_node 刚建的**（按 op 顺序推进）。
    #:
    #: ⚠️ 曾经只算快照：`canvas_generate_image` 在**带快照**时会先建文本/配置节点、
    #: 再连线，而这两个 id 快照里当然没有 ⇒ 同一批自己的 op 被判"指向不存在的节点" ✗。
    #: 不带快照时一切正常，所以只有接了真实画布才会露头。
    known = _known_ids(snapshot)
    #: 只有调用方**给了**快照才做存在性校验：没给就不知道画布长什么样，不能乱判 ✗
    strict = snapshot is not None

    for index, raw in enumerate(ops):
        where = f"ops[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{where} 不是对象")
            continue
        kind = raw.get("type")
        if kind not in OP_TYPES:
            errors.append(f"{where}.type 不认识：{kind!r}（允许 {', '.join(OP_TYPES)}）")
            continue

        if kind == "add_node":
            node_type = raw.get("nodeType") or raw.get("node_type")
            # 内置五类 + 已登记插件（`<pluginId>:<name>`）✓，判据单一真源 `node_type_problem`
            problem = node_type_problem(node_type)
            if problem:
                errors.append(f"{where}.{problem}")
                continue
            item: dict[str, Any] = {"type": "add_node", "nodeType": node_type}
            for key in ("id", "title", "text", "url", "workflowId", "variant", "runMode", "engineStage"):
                if raw.get(key) is not None:
                    item[key] = str(raw[key])
            for key in ("x", "y", "w", "h"):
                number = _num(raw.get(key), f"{where}.{key}", errors)
                if number is not None:
                    item[key] = number
            values = raw.get("values")
            if values is not None:
                if not isinstance(values, dict):
                    errors.append(f"{where}.values 要是对象")
                    continue
                item["values"] = {str(k): str(v) for k, v in values.items()}
            # 本批新建的节点对**后面的** op 立刻可见（否则自己的连线会被判成幽灵）
            if item.get("id"):
                known.add(str(item["id"]))
            clean.append(item)

        elif kind == "update_node":
            if not raw.get("id"):
                errors.append(f"{where}.id 必填")
                continue
            patch = raw.get("patch")
            if patch is not None and not isinstance(patch, dict):
                errors.append(f"{where}.patch 要是对象")
                continue
            item = {"type": "update_node", "id": str(raw["id"])}
            if patch:
                fixed: dict[str, Any] = {}
                for key, value in patch.items():
                    if key in ("x", "y", "w", "h"):
                        number = _num(value, f"{where}.patch.{key}", errors)
                        if number is not None:
                            fixed[key] = number
                    elif key == "values" and isinstance(value, dict):
                        fixed[key] = {str(k): str(v) for k, v in value.items()}
                    else:
                        fixed[key] = value
                item["patch"] = fixed
            clean.append(item)

        elif kind in ("delete_nodes", "delete_edges"):
            ids = raw.get("ids") or ([raw["id"]] if raw.get("id") else [])
            if not ids:
                errors.append(f"{where}.ids 必填且不能为空")
                continue
            clean.append({"type": kind, "ids": [str(one) for one in ids]})

        elif kind == "connect_nodes":
            pairs_raw = raw.get("connections")
            if pairs_raw is None:
                pairs_raw = [raw]
            if not isinstance(pairs_raw, list) or not pairs_raw:
                errors.append(f"{where}.connections 要是非空数组")
                continue
            pairs: list[dict] = []
            for pair in pairs_raw:
                if not isinstance(pair, dict):
                    errors.append(f"{where}.connections 里有非对象")
                    continue
                source = pair.get("from") or pair.get("fromNodeId")
                target = pair.get("to") or pair.get("toNodeId")
                if not source or not target:
                    errors.append(f"{where}.connections 每项要有 from 与 to")
                    continue
                if str(source) == str(target):
                    # 自连一定会被前端拒 ⇒ 这里就报，别让整批白跑
                    errors.append(f"{where}.connections 出现自连：{source}")
                    continue
                if strict and (str(source) not in known or str(target) not in known):
                    errors.append(f"{where}.connections 指向不存在的节点：{source} → {target}")
                    continue
                pairs.append({"from": str(source), "to": str(target)})
            if pairs:
                clean.append({"type": "connect_nodes", "connections": pairs})

        elif kind == "select_nodes":
            ids = raw.get("ids")
            if not isinstance(ids, list):
                errors.append(f"{where}.ids 要是数组（可空）")
                continue
            clean.append({"type": "select_nodes", "ids": [str(one) for one in ids]})

        elif kind == "set_viewport":
            viewport = raw.get("viewport") if isinstance(raw.get("viewport"), dict) else raw
            x = _num(viewport.get("x", 0), f"{where}.x", errors) or 0.0
            y = _num(viewport.get("y", 0), f"{where}.y", errors) or 0.0
            scale = _num(viewport.get("scale", viewport.get("k", 1)), f"{where}.scale", errors)
            scale = 1.0 if scale is None else min(MAX_SCALE, max(MIN_SCALE, scale))
            clean.append({"type": "set_viewport", "viewport": {"x": x, "y": y, "scale": scale}})

        elif kind == "run_generation":
            if not raw.get("nodeId"):
                errors.append(f"{where}.nodeId 必填")
                continue
            mode = raw.get("mode")
            if mode is not None and mode not in GENERATION_MODES:
                # 明确拒：源把未知模式静默归成 image，那会让「跑视频」变成「跑图」且不报错 ✗
                errors.append(f"{where}.mode 不支持：{mode!r}（本仓只有 {', '.join(GENERATION_MODES)}）")
                continue
            item = {"type": "run_generation", "nodeId": str(raw["nodeId"])}
            if mode:
                item["mode"] = mode
            if raw.get("prompt") is not None:
                item["prompt"] = str(raw["prompt"])
            clean.append(item)

    return clean, errors


# ---------------------------------------------------------------- 工具调用 → ops


def next_canvas_x(snapshot: dict | None) -> float:
    """新节点默认横坐标 = 现有节点最右缘 + ``NODE_GAP`` ✓（与源 ``nextCanvasX`` 同口径）。"""
    nodes = (snapshot or {}).get("nodes") or []
    if not nodes:
        return 0.0
    right = 0.0
    for node in nodes:
        try:
            right = max(right, float(node.get("x") or 0) + float(node.get("w") or 0))
        except (TypeError, ValueError):
            continue
    return right + NODE_GAP


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _size(node_type: str, payload: dict) -> tuple[float | None, float | None]:
    """节点尺寸 (w, h)。

    插件节点在**前端清单**里带默认尺寸（后端没有这张表）⇒ 返回 ``None`` 表示「不指定」，
    由前端按 ``PLUGIN_NODES`` 填 ✓。这里猜一个 340×240 会把插件自己的默认尺寸顶掉 ✗。
    """
    base = DEFAULT_NODE_SIZE.get(node_type)
    if base is None:
        return _as_float(payload.get("w")), _as_float(payload.get("h"))
    base_w, base_h = base
    return _as_float(payload.get("w")) or base_w, _as_float(payload.get("h")) or base_h


def _mode(value: Any) -> str:
    return value if value in GENERATION_MODES else "image"


def _values(payload: dict) -> dict[str, str] | None:
    """配置卡参数**一律字符串**（与 ``useCanvas.normalizeValues`` 同口径）✓。"""
    raw = payload.get("values")
    if not isinstance(raw, dict):
        return None
    out = {str(key): str(value) for key, value in raw.items() if value is not None}
    return out or None


#: 音频档允许**写在顶层**的两格 ✓（与写进 `values` 等价 ✓，只是省一层）
TOP_LEVEL_VALUE_KEYS: tuple[str, ...] = ("lyrics", "seconds")


def _merge_top_values(payload: dict) -> dict[str, str] | None:
    """把音频档的顶层 ``lyrics``/``seconds`` 并进 ``values`` ✓。

    模型照 ``canvas_create_text_node`` 的 ``text`` 直觉写成
    ``{"prompt": …, "lyrics": …, "seconds": 20}`` 是很自然的事 ✓；若只认 ``values.lyrics``，
    这段歌词会**静默消失**（歌照跑、只是没词，任何环节都不报错 ✗）。
    **显式 ``values`` 里的同名键优先** ✓ —— 那是模型明确指定的一层，不该被顶层盖掉 ✓。
    """
    raw = payload.get("values")
    explicit = ({str(key): str(value) for key, value in raw.items() if value is not None}
                if isinstance(raw, dict) else {})
    top = {key: str(payload[key]) for key in TOP_LEVEL_VALUE_KEYS if payload.get(key) is not None}
    return {**top, **explicit} or None


def _add_node_op(payload: dict, *, node_type: str, x: float, y: float, title: str | None) -> dict:
    width, height = _size(node_type, payload)
    op: dict[str, Any] = {"type": "add_node", "nodeType": node_type, "x": x, "y": y}
    if width is not None:
        op["w"] = width
    if height is not None:
        op["h"] = height
    if title:
        op["title"] = title
    if payload.get("id"):
        op["id"] = str(payload["id"])
    for key in ("text", "url", "workflowId", "engineStage"):
        if payload.get(key) is not None:
            op[key] = str(payload[key])
    if payload.get("runMode") in ("engine", "comfyui"):
        op["runMode"] = payload["runMode"]
    values = _values(payload)
    if values:
        op["values"] = values
    return op


def _title_of(mode: str) -> str:
    """配置卡标题 ✓ —— 「音频生成」不叫「音乐生成」✗：本仓眼下只有 Music3 一档走它 ✓，
    将来 TTS 也归 audio ✓（标题写死成「音乐」就是把通道名改小了 ✗）。
    """
    return {"text": "文本生成", "video": "视频生成", "audio": "音频生成"}.get(mode, "图片生成")


def _stable_id(prefix: str, *parts: Any) -> str:
    """确定性 id（同一输入产出同一 id ⇒ 重放不会建出两套节点 ✓）。"""
    import hashlib

    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _generation_flow(payload: dict, snapshot: dict | None) -> list[dict]:
    """提示词节点 → 配置节点 → 连线 → 选中 →（可选）开跑。

    ⚠️ 与源的一处**语义修正**：源把 ``prompt`` 原样塞进配置卡、再另建文本节点放同样的话 ——
    同一句话在画布里存两份，改一处另一处不动 ✗。本仓配置卡的提示词就是 ``text`` 里的
    ``@[node:…]`` 引用串 ✓，所以引用是**真的**：改文本节点，配置卡跟着变。
    """
    mode = _mode(payload.get("mode"))
    prompt = str(payload.get("prompt") or "")
    x = _as_float(payload.get("x"))
    x = next_canvas_x(snapshot) if x is None else x
    y = _as_float(payload.get("y")) or 0.0
    text_id = str(payload.get("textId") or _stable_id("text", prompt))
    config_id = str(payload.get("configId") or _stable_id("config", mode, prompt, x, y))

    references = [str(one) for one in payload.get("referenceNodeIds") or [] if isinstance(one, str)]
    mentioned = REFERENCE_PATTERN.findall(prompt)
    # 提示词里只引用了已作为参考传进来的节点 ⇒ 不用再造文本节点（源同口径 ✓）
    reuse = (
        bool(references)
        and bool(mentioned)
        and all(one in references for one in mentioned)
        and not REFERENCE_PATTERN.sub("", prompt).strip()
    )
    tokens = references if reuse else [text_id, *references]
    prompt_expr = "\n".join(f"@[node:{one}]" for one in tokens)

    ops: list[dict] = []
    if not reuse:
        text_op = _add_node_op({**payload, "text": prompt}, node_type="text", x=x, y=y,
                               title=str(payload.get("title") or "提示词"))
        text_op["id"] = text_id
        ops.append(text_op)
    merged = _merge_top_values(payload)
    config_payload: dict[str, Any] = {**payload, "text": prompt_expr}
    if merged:
        config_payload["values"] = merged
    ops.append(_add_node_op(config_payload, node_type="config", x=x + FLOW_DX, y=y,
                            title=_title_of(mode)) | {"id": config_id})
    links = ([] if reuse else [{"from": text_id, "to": config_id}]) + [
        {"from": one, "to": config_id} for one in references
    ]
    if links:
        ops.append({"type": "connect_nodes", "connections": links})
    ops.append({"type": "select_nodes", "ids": [config_id]})
    if payload.get("autoRun"):
        ops.append({"type": "run_generation", "nodeId": config_id, "mode": mode, "prompt": prompt_expr})
    return ops


def _text_nodes(payload: dict, snapshot: dict | None) -> list[dict]:
    items = payload.get("items") or []
    base_x = _as_float(payload.get("x"))
    base_x = next_canvas_x(snapshot) if base_x is None else base_x
    base_y = _as_float(payload.get("y")) or 0.0
    gap = _as_float(payload.get("gap"))
    gap = 40.0 if gap is None else gap
    row = payload.get("direction") != "column"
    ops: list[dict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        x = _as_float(item.get("x"))
        y = _as_float(item.get("y"))
        x = (base_x + index * (TEXT_COL_W + gap) if row else base_x) if x is None else x
        y = (base_y if row else base_y + index * (TEXT_ROW_H + gap)) if y is None else y
        ops.append(_add_node_op(item, node_type="text", x=x, y=y, title=str(item.get("title") or "文本")))
    return ops


def _move_ops(payload: dict, snapshot: dict | None) -> list[dict]:
    """绝对位置优先、否则按相对位移累加（源同口径）。"""
    current = {str(node.get("id")): node for node in (snapshot or {}).get("nodes") or []}
    ops: list[dict] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        node_id = str(item["id"])
        base = current.get(node_id) or {}
        x = _as_float(item.get("x"))
        y = _as_float(item.get("y"))
        patch: dict[str, float] = {}
        if x is not None:
            patch["x"] = x
        elif item.get("dx") is not None:
            patch["x"] = float(base.get("x") or 0) + (_as_float(item.get("dx")) or 0.0)
        if y is not None:
            patch["y"] = y
        elif item.get("dy") is not None:
            patch["y"] = float(base.get("y") or 0) + (_as_float(item.get("dy")) or 0.0)
        if patch:
            ops.append({"type": "update_node", "id": node_id, "patch": patch})
    return ops


#: 只读工具：不产 op，由前端直接回快照
READ_TOOLS = ("canvas_get_state", "canvas_get_selection", "canvas_export_snapshot")


def build_request(name: str, payload: dict | None = None, snapshot: dict | None = None) -> dict:
    """把一个工具调用编译成 ``{"name": "canvas_apply_ops", "input": {"ops": [...]}}``。

    只读工具返回 ``ops=[]`` 且 ``needsSnapshot=True`` —— 让调用方知道「该去问前端要快照」，
    而不是把它当「什么都不用做」✗。
    """
    payload = payload or {}
    if name not in TOOLS:
        raise ValueError(f"未知画布工具：{name}（可用 {len(TOOLS)} 个）")

    if name in READ_TOOLS:
        return {"name": name, "input": {}, "ops": [], "needsSnapshot": True}

    if name == "canvas_apply_ops":
        ops, errors = validate_ops(payload.get("ops"), snapshot)
        if errors:
            return {"name": name, "input": payload, "ops": [], "errors": errors}
        return {"name": "canvas_apply_ops", "input": {"ops": ops}, "ops": ops, "errors": []}

    if name == "canvas_create_node":
        node_type = payload.get("nodeType") or payload.get("node_type")
        problem = node_type_problem(node_type)
        if problem:
            return {"name": name, "input": payload, "ops": [], "errors": [problem]}
        x = _as_float(payload.get("x"))
        x = next_canvas_x(snapshot) if x is None else x
        y = _as_float(payload.get("y")) or 0.0
        ops = [_add_node_op(payload, node_type=node_type, x=x, y=y, title=payload.get("title"))]
    elif name == "canvas_create_text_node":
        x = _as_float(payload.get("x"))
        x = next_canvas_x(snapshot) if x is None else x
        y = _as_float(payload.get("y")) or 0.0
        ops = [_add_node_op(payload, node_type="text", x=x, y=y, title=str(payload.get("title") or "文本"))]
    elif name == "canvas_create_text_nodes":
        ops = _text_nodes(payload, snapshot)
    elif name == "canvas_create_config_node":
        x = _as_float(payload.get("x"))
        x = next_canvas_x(snapshot) if x is None else x
        y = _as_float(payload.get("y")) or 0.0
        config_id = str(payload.get("id") or _stable_id("config", x, y, payload.get("prompt")))
        config_title = str(payload.get("title") or _title_of(_mode(payload.get("mode"))))
        merged = _merge_top_values(payload)
        ops = [_add_node_op({**payload, "id": config_id, **({"values": merged} if merged else {})},
                            node_type="config", x=x, y=y, title=config_title)]
        if payload.get("autoRun"):
            ops.append({"type": "run_generation", "nodeId": config_id,
                        "mode": _mode(payload.get("mode")), "prompt": str(payload.get("prompt") or "")})
    elif name == "canvas_create_generation_flow":
        ops = _generation_flow(payload, snapshot)
    elif name.startswith("canvas_generate_"):
        ops = _generation_flow({**payload, "mode": name.rsplit("_", 1)[-1], "autoRun": True}, snapshot)
    elif name == "canvas_update_node":
        op: dict[str, Any] = {"type": "update_node", "id": str(payload.get("id") or "")}
        patch = payload.get("patch")
        if isinstance(patch, dict) and patch:
            op["patch"] = patch
        ops = [op]
    elif name == "canvas_update_node_text":
        patch = {"text": str(payload.get("text") or "")}
        if payload.get("title"):
            patch["title"] = str(payload["title"])
        ops = [{"type": "update_node", "id": str(payload.get("id") or ""), "patch": patch}]
    elif name == "canvas_move_nodes":
        ops = _move_ops(payload, snapshot)
    elif name == "canvas_resize_node":
        width = _as_float(payload.get("w"))
        height = _as_float(payload.get("h"))
        if width is None or height is None:
            return {"name": name, "input": payload, "ops": [], "errors": ["w 与 h 都要是数字"]}
        ops = [{"type": "update_node", "id": str(payload.get("id") or ""), "patch": {"w": width, "h": height}}]
    elif name == "canvas_delete_nodes":
        ids = [str(one) for one in payload.get("ids") or []]
        ops = [{"type": "delete_nodes", "ids": ids}] if ids else []
        if not ids:
            return {"name": name, "input": payload, "ops": [], "errors": ["ids 不能为空"]}
    elif name == "canvas_connect_nodes":
        ops = [{"type": "connect_nodes", "connections": payload.get("connections") or []}]
    elif name == "canvas_select_nodes":
        ops = [{"type": "select_nodes", "ids": [str(one) for one in payload.get("ids") or []]}]
    elif name == "canvas_set_viewport":
        viewport = payload.get("viewport") if isinstance(payload.get("viewport"), dict) else payload
        scale = _as_float(viewport.get("scale"))
        scale = 1.0 if scale is None else min(MAX_SCALE, max(MIN_SCALE, scale))
        ops = [{"type": "set_viewport",
                "viewport": {"x": _as_float(viewport.get("x")) or 0.0,
                             "y": _as_float(viewport.get("y")) or 0.0,
                             "scale": scale}}]
    else:  # canvas_run_generation
        ops = [{"type": "run_generation", "nodeId": str(payload.get("nodeId") or ""),
                "mode": _mode(payload.get("mode")),
                **({"prompt": str(payload["prompt"])} if payload.get("prompt") is not None else {})}]

    ops, errors = validate_ops(ops, snapshot)
    return {"name": "canvas_apply_ops", "input": {"ops": ops}, "ops": ops, "errors": errors}


# ---------------------------------------------------------------- 快照压缩


def compact_snapshot(snapshot: dict | None) -> dict:
    """压缩画布快照再交给模型：正文截断、只留判断所需的字段 ✓。

    源只截 ``metadata.content``；本仓正文在 ``text`` 上，另外把 ``values`` 全部原样带出去 ——
    工作流参数可能很长，但**模型改参数必须看到它**，所以只截正文不截参数 ✓。
    """
    if not isinstance(snapshot, dict):
        raise ValueError("当前没有已连接画布")
    nodes: list[dict] = []
    for node in snapshot.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        item = {
            key: node[key]
            for key in ("id", "type", "title", "x", "y", "w", "h", "url", "groupId",
                        "workflowId", "variant", "runMode", "engineStage", "values")
            if node.get(key) is not None
        }
        text = node.get("text")
        if isinstance(text, str) and len(text) > SNAPSHOT_TEXT_LIMIT:
            item["text"] = text[:SNAPSHOT_TEXT_KEEP] + "…"
        elif text is not None:
            item["text"] = text
        nodes.append(item)
    return {
        "nodes": nodes,
        "edges": snapshot.get("edges") or [],
        "viewport": snapshot.get("viewport") or {"x": 0, "y": 0, "scale": 1},
        "selection": snapshot.get("selection") or [],
    }
