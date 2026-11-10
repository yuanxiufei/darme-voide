"""Agent 元信息登记表 —— **Agent 层的唯一成员来源**。

## 为什么是这个形态（2026-09-25 收敛）

沿用 ``agents/index.ts`` 的镜像注释已经完成使命：``backend/`` 已于 2026-09-15 删除，
本文件即成为唯一来源（漂移守卫 ``tests/route_parity_test.py::_registry_drift`` 仍在，
但它的作用已退化为「锁定 Python 侧取值不再变大」）。

收敛前的真实问题**不是字段重复**（每个字段确实只有一处），而是**成员集合被复述了 6 遍**：

* ``AGENT_PHASES`` / ``AGENT_DEFAULT_NAMES`` / ``VALID_AGENT_TYPES``（本文件）
* ``agent_prompts.DEFAULT_INSTRUCTIONS``（键集合 = 成员集合）
* ``runtime.STYLE_PROFILE_TYPES`` / ``runtime.create_agent_tools`` 的 if/elif 分派
* ``subagent.SUBAGENT_REGISTRY``

⇒ 新增一个 Agent 要在 6~8 处同步改，漏一处**静默失效**（分派漏改 ⇒ 工具为空、
``get_agent_defaults`` 少一行、orchestrator 派不到它）。

现在改为：**``AGENT_SPECS`` 声明一次，其余 6 个视图全部派生**（见文件末）。
新增 Agent = 在 ``_STRUCTURE`` 加一条 + 在 ``agent_prompts`` 写提示词，
``runtime`` / ``subagent`` **一行都不用改**。

## 单一权威的边界（刻意保留的两处，各有守卫）

* **提示词正文** 仍住在 ``agent_prompts.py``（「只放 instructions」，内容与结构分居）。
  两个键集合必须逐项同序 —— 由下面 ``_assert_same_membership()`` **导入期硬失败**兜底，
  而不是等某个测试偶然发现。
* **``HOST_TOOL_NAMES``** 仍是静态清单：取值来源是 Mastra ``tool.id``（snake_case）。
  但对照物已换 —— ``backend/``（Node）删除后，冻结的 TS 快照只能作历史参考；
  权威改成 ``live_host_tool_names()``（真实 ``create_*`` 工厂产出），两者必须严格相等。
  2026-11-06 再收一步：live 取的是 ``Tool.id`` 而非工厂 dict 的**键** —— 驱动侧的函数名
  就是 ``Tool.id``（``ToolRegistry.as_openai_tools`` 逐项用它），键只是工厂内检索标签；
  两者约定等同，但一旦某工厂沿用 TS 的 camelCase 键，只读键的 live 会拿一个**模型永远
  调不到的名字**去当「真实工具」，于是清单守卫与 ``missingTools`` 会一起说错。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, NamedTuple

from .agent_prompts import DEFAULT_INSTRUCTIONS

__all__ = [
    "AGENT_DEFAULT_NAMES",
    "AGENT_PHASES",
    "AGENT_SPECS",
    "HOST_TOOL_NAMES",
    "STYLE_PROFILE_TYPES",
    "SUBAGENT_REGISTRY",
    "VALID_AGENT_TYPES",
    "AgentSpec",
    "get_agent_spec",
    "get_default_name",
    "get_host_tool_names",
    "live_host_tool_names",
    "live_tool_objects",
    "SUBAGENT_ONLY_TOOL_NAMES",
]


class AgentSpec(NamedTuple):
    """一个 Agent 的全部声明式属性。

    * ``type`` —— 稳定标识（= DB ``agent_configs.agent_type`` = ``run_subagent`` 的入参）；
    * ``default_name`` —— 出厂显示名（对齐 ``DEFAULT_PROMPTS[type].name``）；
    * ``instructions`` —— 出厂提示词正文（来自 :mod:`app.services.agent_prompts`）；
    * ``phase`` —— 工作流制作阶段名（``None`` = 不进阶段表，如 orchestrator）；
    * ``tool_set`` —— 工具集标识，由 ``runtime._TOOL_FACTORIES`` 解析（``None`` = 不可运行）；
    * ``style_profile`` —— 是否注入项目风格 Profile（house style）；
    * ``subagent_*`` —— 供 orchestrator 调度用的能力清单条目（``None`` = 不可被委托）。
    """

    type: str
    default_name: str
    instructions: str
    phase: str | None = None
    tool_set: str | None = None
    style_profile: bool = False
    subagent_name: str | None = None
    subagent_capability: str | None = None


class _Structure(NamedTuple):
    """Agent 的**结构性**属性（与提示词正文分居两个文件）。"""

    default_name: str
    phase: str | None = None
    tool_set: str | None = None
    style_profile: bool = False
    #: ``(name, capability)``；``None`` = 不可被 orchestrator 委托
    subagent: tuple[str, str] | None = None


#: 合法 Agent 的结构声明。**插件的顺序即声明序**（= ``DEFAULT_PROMPTS`` 顺序）。
#: ⚠️ 注意 ``subagent.name`` 与 ``default_name`` **不是**同一个字段：``grid_prompt_generator``
#: 出厂显示名是「宫格图提示词生成」，而给 orchestrator 看的能力清单写「图片提示词生成」——
#: 两者分别有漂移守卫，不能合并（曾经差点被「顺手统一」掉）。
_STRUCTURE: dict[str, _Structure] = {
    "script_rewriter": _Structure(
        "剧本改写", "剧本编写", "script", True,
        ("剧本改写", "将小说/原始内容改写为格式化短剧剧本并保存到当前集"),
    ),
    "extractor": _Structure(
        "角色场景提取", "资产提取", "extract", True,
        ("角色场景提取", "从剧本提取角色与场景（同名/同地点智能去重）并保存"),
    ),
    "storyboard_breaker": _Structure(
        "分镜拆解", "分镜拆解", "storyboard", True,
        ("分镜拆解", "将剧本拆解为带完整字段的分镜方案并保存"),
    ),
    "voice_assigner": _Structure(
        "角色音色分配", "音色分配", "voice", False,
        ("角色音色分配", "为每个角色匹配最合适的音色并保存"),
    ),
    "grid_prompt_generator": _Structure(
        "宫格图提示词生成", "画面提示词", "grid", False,
        ("图片提示词生成", "生成角色/场景/宫格图的英文提示词"),
    ),
    # orchestrator（总调度）：不在阶段表里（不参与工作流阶段展示），也不可被自己委托
    "orchestrator": _Structure(
        "短剧制作总调度", None, "subagent", False, None,
    ),
}


def _assert_same_membership() -> None:
    """成员集合必须与提示词键集合**逐项同序**（不一致即导入期硬失败）。

    ⚠️ 这里刻意**不用**「差集报警」式的软检查：成员表少一条的后果是
    ``VALID_AGENT_TYPES`` 少一个 Agent、``create_agent_tools`` 返回 ``None`` ⇒
    该 Agent **静默不可运行**，只在调用时才暴露。宁可启动即崩。
    """
    declared = tuple(_STRUCTURE)
    prompts = tuple(DEFAULT_INSTRUCTIONS)
    if declared != prompts:
        missing = [t for t in prompts if t not in declared]
        extra = [t for t in declared if t not in prompts]
        raise RuntimeError(
            "Agent 成员表与提示词键集合不一致（顺序也须一致）。"
            f"缺少结构声明={missing}；多余结构声明={extra}"
        )


_assert_same_membership()


#: 全部 Agent 的声明式登记表（**顺序即声明序**）。
AGENT_SPECS: tuple[AgentSpec, ...] = tuple(
    AgentSpec(
        type=agent_type,
        default_name=structure.default_name,
        instructions=DEFAULT_INSTRUCTIONS[agent_type],
        phase=structure.phase,
        tool_set=structure.tool_set,
        style_profile=structure.style_profile,
        subagent_name=structure.subagent[0] if structure.subagent else None,
        subagent_capability=structure.subagent[1] if structure.subagent else None,
    )
    for agent_type, structure in _STRUCTURE.items()
)


# ===========================================================================
# 以下 6 个视图**全部派生**自 AGENT_SPECS —— 不在别处复述成员集合
# ===========================================================================

_SPEC_BY_TYPE: dict[str, AgentSpec] = {spec.type: spec for spec in AGENT_SPECS}

#: 合法 Agent 类型 —— 对齐 ``validAgentTypes``（= ``Object.keys(DEFAULT_PROMPTS)``）。
#: **顺序即声明序**，守卫会逐项比对。
VALID_AGENT_TYPES: tuple[str, ...] = tuple(spec.type for spec in AGENT_SPECS)

#: Agent → 出厂显示名（对齐 ``DEFAULT_PROMPTS[type].name``，即 ``getDefaultName`` 的取值）。
AGENT_DEFAULT_NAMES: dict[str, str] = {spec.type: spec.default_name for spec in AGENT_SPECS}

#: Agent → 工作流制作阶段名（对齐 ``AGENT_PHASES``）。
#: 注意只有 **5 个**：``orchestrator``（总调度）不在阶段表里，它不参与工作流阶段展示。
AGENT_PHASES: dict[str, str] = {
    spec.type: spec.phase for spec in AGENT_SPECS if spec.phase is not None
}

#: 会被注入风格 Profile 的 Agent（其余直接透传）。
STYLE_PROFILE_TYPES: tuple[str, ...] = tuple(
    spec.type for spec in AGENT_SPECS if spec.style_profile
)

#: 领域专家 Agent 能力清单（供 orchestrator 决策调度）——逐字对齐 TS 的 ``SUBAGENT_REGISTRY``。
SUBAGENT_REGISTRY: tuple[dict[str, str], ...] = tuple(
    {
        "type": spec.type,
        "name": spec.subagent_name or "",
        "capability": spec.subagent_capability or "",
    }
    for spec in AGENT_SPECS
    if spec.subagent_capability is not None
)


#: 宿主（本项目）已注册的**工具名**集合 —— 对齐 ``getHostToolNames()``。
#:
#: ⚠️ 取值来源是 Mastra 的 ``tool.id``（snake_case），**不是**工厂返回对象里的 camelCase 键：
#: 真正发给模型的是 ``tool.id``，与各 Agent 提示词里「调用 read_script_for_extraction」的写法一致。
#: 取错会静默把所有工具判成「未提供」，让整个诊断失效。
#:
#: Node 侧是懒计算 + 缓存（遍历 ``validAgentTypes`` 调 ``createAgentTools`` 收集 ``tool.id``）——
#: 那需要 Mastra 工厂，Python 侧无法执行 ⇒ 改为静态清单，并用漂移守卫保证与 TS 源码一致。
#:
#: ⚠️ 2026-11-06：只对着**冻结的 TS 快照**比对是不够的 —— ``backend/``（Node）已删除，快照
#: 定格在删除那一刻，Python 侧**此后新增的工具永远不会被它发现**。实测漏了三个真实工具
#: （``search_script_library`` 于 2026-11-04 补的按需通路、``list_skill_refs``／``read_skill_ref``
#: 同批），于是「宿主有没有这个工具」这件事在本文件里会**向两个方向说谎**：
#:   * 真实工具不在清单里 ⇒ 有 skill 声明它时，``/skills`` 的 ``missingTools`` 把它报成「缺」，
#:     徽标写着「绑定后它会教 Agent 调用不存在的工具」，而那个工具其实好得很；
#:   * 已删工具留在清单里 ⇒ 徽标不亮，绑定后调用落空。
#: ⇒ 现以 ``live_host_tool_names()``（真实工厂产出）为权威，TS 快照降级为历史对照，
#: 两道守卫分别钉住「真实工具不能缺席」与「清单里的名字必须有出处」。
HOST_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "assign_voice",
        "edit_script",
        "generate_grid_prompt",
        "get_characters",
        "list_available_agents",
        "list_skill_refs",
        "list_voices",
        "read_episode_script",
        "read_existing_characters",
        "read_existing_props",
        "read_existing_scenes",
        "read_script_for_extraction",
        "read_shots_for_grid",
        "read_skill_ref",
        "read_storyboard_context",
        "rewrite_to_screenplay",
        "run_subagent",
        "save_continuity_states",
        "save_dedup_characters",
        "save_dedup_props",
        "save_dedup_scenes",
        "save_script",
        "save_storyboards",
        "search_reference_prompts",
        "search_script_library",
        "update_storyboard",
    }
)

#: 只能由 ``subagent.py`` 提供的两个工具（``create_run_subagent_tools`` 产出）。它的 ``conn``
#: 只供**执行**委托时真跑一次 Agent run，构造期不查库 ⇒ 现与其它 8 个工厂一样被真实调用取名，
#: 本常量降级为「必须存在的两个」的**声明**，由 ``skills_test`` 钉住（``live ⊇ 本集合``）。
#: 旧写法是把这串手抄名**并入** live 结果 —— 工厂里 ``Tool.id`` 一改名，全仓唯一该知道这件事的
#: 地方（live）反倒最不可能知道。
SUBAGENT_ONLY_TOOL_NAMES: frozenset[str] = frozenset({"list_available_agents", "run_subagent"})


#: :func:`live_tool_objects` 的取数口：``(模块路径, 工厂名, 构造参数)``。
#: 存成模块级常量而不是写在函数体里，是为了让测试拿**同一份**清单去交叉核对
#: 「``Tool.id`` == 驱动发给模型的 ``function.name``」—— 测试里抄第二份，就是把
#: 「手抄清单」这个病根换个地方重新种下。
_LIVE_TOOL_FACTORIES: tuple[tuple[str, str, tuple[object, ...]], ...] = (
    ("app.agent.tools.corpus_tools", "create_corpus_tools", ()),
    ("app.agent.tools.library_tools", "create_library_tools", ()),
    ("app.agent.tools.skill_ref_tools", "create_skill_ref_tools", ()),
    ("app.agent.tools.extract_tools", "create_extract_tools", (0, 0)),
    ("app.agent.tools.grid_prompt_tools", "create_grid_prompt_tools", (0, 0)),
    ("app.agent.tools.script_tools", "create_script_tools", (0,)),
    ("app.agent.tools.storyboard_tools", "create_storyboard_tools", (0, 0)),
    ("app.agent.tools.voice_tools", "create_voice_tools", (0, 0)),
    ("app.agent.subagent", "create_run_subagent_tools", (0, 0, 0, "")),
)


def live_tool_objects() -> list[Any]:
    """按 :data:`_LIVE_TOOL_FACTORIES` **真实构造**全部宿主工具对象。

    工厂在函数内 import（``services`` ←→ ``agent`` 互引，模块级 import 会成环）；
    构造期不查库、不联网（真正的 IO 都在 ``execute`` 里）⇒ 拿 ``0`` 当占位形参安全。
    """
    out: list[Any] = []
    for module_name, factory_name, args in _LIVE_TOOL_FACTORIES:
        factory = getattr(import_module(module_name), factory_name)
        out.extend(factory(*args).values())
    return out


def live_host_tool_names() -> frozenset[str]:
    """**真实**宿主工具集：逐个调 ``app/agent/tools/*.py`` 与 ``subagent.py`` 的工厂取 ``Tool.id``。

    这是 ``HOST_TOOL_NAMES`` 的权威对照物 —— 冻结的 TS 快照只记到 Node 后端被删除那一刻，
    之后 Python 侧新增的工具它一概看不见（2026-11-06 实测漏 3 个，见上面 ⚠️）。

    取 ``Tool.id`` 而不是工厂返回 dict 的**键**：驱动侧拿到的函数名就是 ``Tool.id``
    （``ToolRegistry.as_openai_tools`` 逐项用它组 ``function.name``），键只是工厂内的检索标签。
    只读键的写法在「键 = id」时毫无差别，**一旦某个工厂用了 camelCase 键**（TS 迁移常见形态，
    ``ToolRegistry.get`` 已为此按 id 兜底）就会拿一个模型永远调不到的名字当「真实工具」——
    ``HOST_TOOL_NAMES`` 守卫随即误红、``missingTools`` 也跟着说错。
    """
    return frozenset(tool.id for tool in live_tool_objects())


def get_agent_spec(agent_type: str) -> AgentSpec | None:
    """取 Agent 的结构声明（未知类型返回 ``None``）。"""
    return _SPEC_BY_TYPE.get(agent_type)


def get_default_name(agent_type: str) -> str:
    """``DEFAULT_PROMPTS[type]?.name || type``。"""
    return AGENT_DEFAULT_NAMES.get(agent_type) or agent_type


def get_host_tool_names() -> frozenset[str]:
    return HOST_TOOL_NAMES
