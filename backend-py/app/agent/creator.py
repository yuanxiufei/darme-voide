"""Agent 创建器 —— 与 ``agents/creator.ts``（169 行）对齐。

对齐 PenguinHarness 自进化闭环第一环 ``agent-creation``：

**一句话需求 → Agent 配置（name / description / systemPrompt / skills）**

前置契约（缺失即报错）：

* ``agent_type`` 必须属于 ``VALID_AGENT_TYPES``；
* ``requirement`` 非空。

协议化输出：LLM 必须返回**严格 JSON**（字段名固定），解析失败即报错；生成的 systemPrompt
保持原工作流程与工具调用语义，只针对需求强化。

⚠️ **防作弊隔离**：生成器看不到评测 rubric，只看默认提示词 + 用户需求（与 optimizer 一致）。

⚠️ 三处保真点：

1. ``AVAILABLE_SKILL_IDS`` 是**模块加载时的快照**（原 TS 是顶层 ``const``）—— 运行期新增
   skill 不会进这个列表；自检需要时可替换该模块属性；
2. 生成用**单发** LLM 调用（无工具、无模型 fallback，取 ``textConfig.model``）—— Python 侧复用
   ``runtime.default_generate``（空工具集），即 Mastra ``agent.generate`` 的等价物；
3. ``persist_agent_config`` 的 upsert 按 ``agent_type`` 取**第一条**（不挑 is_active），
   更新时把 ``deleted_at`` **清回 null**（复活软删行）✓。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from app.core.db import engine
from app.core.models import agent_configs
from app.core.response import now
from app.services.ai_providers import get_text_config
from app.services.agent_prompts import get_default_instructions
from app.services.agent_registry import VALID_AGENT_TYPES, get_default_name
from app.agent.runtime import default_generate
from app.agent.skills import list_core_skill_ids, resolve_default_skills
from app.agent.tool import ToolRegistry

__all__ = [
    "AVAILABLE_SKILL_IDS",
    "CREATOR_INSTRUCTIONS",
    "GeneratedAgentConfig",
    "extract_json",
    "generate_agent_config",
    "persist_agent_config",
]

#: creator 可推荐给 Agent 的 skill 集合 = **项目自有 skill**（``skills/`` 顶层含 SKILL.md 的目录）。
#: 不含外部技能库（vendor）：那些是「按触发词独立启动」的会话式技能，绑给常驻 Agent
#: 只会注入半截指令（其正文引用的 ``references/*.md`` 加载器不读）。⚠️ **模块加载时快照**。
AVAILABLE_SKILL_IDS: list[str] = list_core_skill_ids()

CREATOR_INSTRUCTIONS = """你是一名 AI Agent 配置生成专家。根据用户的一句话需求，为指定类型的 Agent 生成优化后的配置。

约束：
1. 保持原有工作流程与工具调用语义不变（不修改工具名、不改变调用顺序）
2. 只针对用户需求做针对性强化（补充规则、示例、强调易错点）
3. 输出必须是严格 JSON，不要输出任何解释、前言，也不要包 markdown 代码块围栏"""

#: 整段被 ``` 围栏包住（``^```lang\n...\n```$``）
_FENCE = re.compile(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$")


@dataclass
class GeneratedAgentConfig:
    """生成结果（**尚未落库**）。"""

    agent_type: str
    name: str
    description: str
    system_prompt: str
    #: 标准化后的 skill 绑定（只含合法 id、enabled=True）
    skills: list[dict[str, Any]] = field(default_factory=list)


def extract_json(text: str) -> Any:
    """健壮 JSON 提取：剥 markdown 围栏 + 取**首个 ``{`` 到末个 ``}``**。"""
    stripped = (text or "").strip()
    fence = _FENCE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("生成器未返回有效 JSON")
    return json.loads(stripped[start:end + 1])


async def generate_agent_config(
    conn: Connection, agent_type: str, requirement: str
) -> GeneratedAgentConfig:
    """一句话需求 → 生成 Agent 配置（**不落库**）。"""
    if agent_type not in VALID_AGENT_TYPES:
        raise ValueError(
            f"未知 Agent 类型：{agent_type}（可用：{', '.join(VALID_AGENT_TYPES)}）"
        )
    if not (requirement or "").strip():
        raise ValueError("requirement 不能为空")

    default_instructions = get_default_instructions(agent_type)
    text_config = get_text_config(conn)

    user_message = "\n".join([
        f"Agent 类型：{agent_type}",
        "当前默认提示词：",
        "--- 开始 ---",
        default_instructions,
        "--- 结束 ---",
        "",
        f"可用 skill 集合：{', '.join(AVAILABLE_SKILL_IDS)}",
        "",
        f"用户需求：{requirement}",
        "",
        "请严格输出 JSON（字段名固定）：",
        '{"name": "简短中文显示名", "description": "一句话职责", '
        '"system_prompt": "完整系统提示词（可直接落库）", '
        '"skills": ["从可用集合选出的 skill id 列表"]}',
    ])

    result = await default_generate(
        config=text_config,
        model=str(text_config.get("model") or ""),
        instructions=CREATOR_INSTRUCTIONS,
        tools=ToolRegistry({}),  # 单发：不给工具
        message=user_message,
        max_steps=1,
        model_settings={},
    )
    parsed = extract_json(result.get("text") or "")
    if not isinstance(parsed, dict):
        raise ValueError("生成器未返回有效 JSON")

    system_prompt = str(parsed.get("system_prompt") or "").strip()
    if len(system_prompt) < 50:
        raise ValueError(f"生成器产出无效 systemPrompt（length={len(system_prompt)}）")

    # skills 过滤：只保留合法 id，非法**静默丢弃**；空则回退该 Agent 的默认绑定（frontmatter 自描述）
    raw_skills = parsed.get("skills")
    requested = ([str(item) for item in raw_skills] if isinstance(raw_skills, list) else [])
    requested = [item for item in requested if item in AVAILABLE_SKILL_IDS]
    final_ids = requested or resolve_default_skills(agent_type)
    skills = [{"id": skill_id, "enabled": True, "priority": index + 1}
              for index, skill_id in enumerate(final_ids)]

    return GeneratedAgentConfig(
        agent_type=agent_type,
        name=str(parsed.get("name") or "").strip() or get_default_name(agent_type),
        description=str(parsed.get("description") or "").strip(),
        system_prompt=system_prompt,
        skills=skills,
    )


def persist_agent_config(candidate: GeneratedAgentConfig) -> dict[str, Any]:
    """把生成的配置 upsert 到 ``agent_configs``（**保留 model/temperature 等既有值**），返回落库行。

    ⚠️ 自管短事务（对齐 Node 的逐语句自动提交）：调用方（optimizer / 路由）不该把这次写入
    卷进自己的事务 —— 否则失败会连累调用方的回滚语义。
    """
    ts = now()
    skills_str = json.dumps(candidate.skills, ensure_ascii=False, separators=(",", ":"))

    with engine.begin() as conn:
        existing = conn.execute(
            select(agent_configs).where(agent_configs.c.agent_type == candidate.agent_type)
        ).first()

        if existing is not None:
            conn.execute(
                agent_configs.update().where(agent_configs.c.id == existing.id).values(
                    name=candidate.name,
                    description=candidate.description,
                    system_prompt=candidate.system_prompt,
                    skills=skills_str,
                    is_active=True,
                    # ⚠️ 复活软删行（原 TS 显式写 deletedAt: null）
                    deleted_at=None,
                    updated_at=ts,
                )
            )
            row = conn.execute(
                select(agent_configs).where(agent_configs.c.id == existing.id)
            ).first()
            return row

        new_id = int(conn.execute(agent_configs.insert().values(
            agent_type=candidate.agent_type,
            name=candidate.name,
            description=candidate.description,
            model="",
            system_prompt=candidate.system_prompt,
            temperature=0.7,
            max_tokens=4096,
            max_iterations=10,
            skills=skills_str,
            is_active=True,
            created_at=ts,
            updated_at=ts,
        )).lastrowid)
        row = conn.execute(
            select(agent_configs).where(agent_configs.c.id == new_id)
        ).first()
        return row
