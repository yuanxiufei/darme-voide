"""Agent 元信息登记表 —— 镜像 ``agents/index.ts`` 里与「非 Agent 运行」相关的常量。

⚠️ **这是同一份数据的第二处副本**，必须正视：Node 侧的单一事实来源是
``agents/index.ts`` 的 ``DEFAULT_PROMPTS`` / ``AGENT_PHASES`` 与 ``tools/*.ts`` 的 ``tool.id``。
绞杀期两边并存 ⇒ 存在漂移风险，所以：

* 这里**只镜像静态度量**（阶段名、显示名、宿主工具名集合）；**提示词正文不在本文件**，已单独落在
  ``services/agent_prompts.py``（2026-09-12 决策变更：它成了 ``runtime`` 的真回落值、
  ``evaluation`` 的基准、且 ``GET /agent-configs/defaults`` 原先是**委托回 Node** 的，
  详见该模块头；两者各有逐字漂移守卫）；
* ``tests/route_parity_test.py`` 里有**漂移守卫**：直接从 TS 源码抽取这些值与本文件比对，
  一旦 Node 侧改名/加工具而没同步，测试立刻红。

⇒ 等 Node 下线，这里就成为唯一来源，届时漂移守卫自动失效（可直接删掉那段）。
"""

from __future__ import annotations

#: Agent → 工作流制作阶段名（对齐 ``AGENT_PHASES``）。
#: 注意只有 **5 个**：``orchestrator``（总调度）不在阶段表里，它不参与工作流阶段展示。
AGENT_PHASES: dict[str, str] = {
    "script_rewriter": "剧本编写",
    "extractor": "资产提取",
    "voice_assigner": "音色分配",
    "storyboard_breaker": "分镜拆解",
    "grid_prompt_generator": "画面提示词",
}

#: Agent → 出厂显示名（对齐 ``DEFAULT_PROMPTS[type].name``，即 ``getDefaultName`` 的取值）。
#: 含 ``orchestrator`` —— ``validAgentTypes`` 是 6 个键，比 ``AGENT_PHASES`` 多一个。
AGENT_DEFAULT_NAMES: dict[str, str] = {
    "script_rewriter": "剧本改写",
    "extractor": "角色场景提取",
    "storyboard_breaker": "分镜拆解",
    "voice_assigner": "角色音色分配",
    "grid_prompt_generator": "宫格图提示词生成",
    "orchestrator": "短剧制作总调度",
}

#: 宿主（本项目）已注册的**工具名**集合 —— 对齐 ``getHostToolNames()``。
#:
#: ⚠️ 取值来源是 Mastra 的 ``tool.id``（snake_case），**不是**工厂返回对象里的 camelCase 键：
#: 真正发给模型的是 ``tool.id``，与各 Agent 提示词里「调用 read_script_for_extraction」的写法一致。
#: 取错会静默把所有工具判成「未提供」，让整个诊断失效。
#:
#: Node 侧是懒计算 + 缓存（遍历 ``validAgentTypes`` 调 ``createAgentTools`` 收集 ``tool.id``）——
#: 那需要 Mastra 工厂，Python 侧无法执行 ⇒ 改为静态清单，并用漂移守卫保证与 TS 源码一致。
HOST_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "assign_voice",
        "generate_grid_prompt",
        "get_characters",
        "list_available_agents",
        "list_voices",
        "read_episode_script",
        "read_existing_characters",
        "read_existing_props",
        "read_existing_scenes",
        "read_script_for_extraction",
        "read_shots_for_grid",
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
        "update_storyboard",
    }
)


#: 合法 Agent 类型 —— 对齐 ``validAgentTypes``（= ``Object.keys(DEFAULT_PROMPTS)``）。
#: **顺序即 TS 声明序**（``dict`` 保序），守卫会逐项比对。
VALID_AGENT_TYPES: tuple[str, ...] = tuple(AGENT_DEFAULT_NAMES)


def get_default_name(agent_type: str) -> str:
    """``DEFAULT_PROMPTS[type]?.name || type``。"""
    return AGENT_DEFAULT_NAMES.get(agent_type) or agent_type


def get_host_tool_names() -> frozenset[str]:
    return HOST_TOOL_NAMES
