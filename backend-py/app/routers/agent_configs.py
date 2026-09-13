"""``/api/v1/agent-configs`` —— 与 ``backend/src/routes/agentConfigs.ts`` 对齐（5 / 7 端点）。

**已迁移 5 个**：``GET /``、``GET /:id``、``POST /``（按 agent_type upsert）、``PUT /:id``、``DELETE /:id``
**未迁移 2 个**：

* ``GET /defaults`` —— 依赖 ``agents/index.ts`` 的 ``DEFAULT_PROMPTS``（5 个 Agent 的完整提示词）
  与 ``skills.ts`` 解析各 SKILL.md frontmatter 的 ``agents:`` 绑定 ⇒ 属**提示词/skills 资产域**，
  随那一批一起做（那里是「提示词多头维护」的单一事实来源，不能在这里另存一份副本）。
* ``POST /generate`` —— 调 LLM 生成 Agent 配置（``agents/creator.js``）。

⚠️ 返回形状是 **snake_case**（原 TS 显式 ``toSnakeCase``），且**错误码两种**：
``GET /`` 与 ``GET /:id``、``DELETE`` 的 catch 走 **500** ``{code,data:null,message}``，
而 ``POST`` / ``PUT`` 的 catch 走 **400** ``{code,message}``（无 ``data`` 键）—— 别统一。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import agent_configs
from ..request_utils import read_json
from ..services.agent_registry import VALID_AGENT_TYPES
from ..services.agents.creator import generate_agent_config, persist_agent_config
from ..services.agents.runtime import get_agent_defaults
from ..services.task_logger import log_task_error
from ..response import (
    bad_request,
    internal_error,
    js_nullish,
    js_truthy,
    not_found,
    now,
    parse_param_id,
    row_to_dict,
    success,
)

router = APIRouter(prefix="/api/v1/agent-configs", tags=["agent-configs"])


def _validate_skills(value: Any) -> tuple[str | None, str | None]:
    """校验 skills 并规范化成紧凑 JSON。返回 ``(skills_json, error_message)``。

    对齐原 TS 的三条错误：``skills is not valid JSON`` / ``skills must be a JSON array`` /
    ``each skill must have an id field``。注意原 TS 会先 ``JSON.parse`` 再
    ``JSON.stringify`` 回写 —— 所以最终落库的一定是**紧凑格式**（无多余空格）。
    """
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return None, "skills is not valid JSON"
    if not isinstance(parsed, list):
        return None, "skills must be a JSON array"
    for item in parsed:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item.get("id"):
            return None, "each skill must have an id field"
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), None


@router.get("")
def list_configs(conn: Connection = Depends(get_conn)):
    try:
        rows = conn.execute(
            select(agent_configs).where(agent_configs.c.deleted_at.is_(None))
        ).all()
        return success([row_to_dict(r) for r in rows])
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.get("/defaults")
def get_defaults():
    """Agent「出厂默认配置」= 提示词 + 默认 Skill 绑定。

    **已本地化**（2026-09-12）：原先这里**委托回 Node**，因为 ``DEFAULT_PROMPTS`` 与
    ``skills.ts`` 都没迁。现在两者都已落地（``services/agent_prompts.py`` +
    ``agents/skills.py``，各带逐字/一致性守卫），本端点**不再依赖 Node**。

    ⚠️ 同时去掉了委托期的 ``include_in_schema=False``：那只是「shim 别出现在 API 文档里」的
    临时标记，但 ``tests/route_parity_test.py`` 判定「Python 已注册路径」的口径正是
    **OpenAPI schema** ⇒ 留着它会让本端点被当成**未迁移端点**去探活，进而误报 SHADOW。

    ⚠️ 这个声明**必须排在 ``GET /{config_id}`` 之前**：否则 ``defaults`` 会被当成 id 解析，
    返回 404「Invalid agent config id」—— 看起来合理，实则**静默丢掉了实现**。
    （这一类遮蔽由 ``tests/route_parity_test.py`` 系统性兜住。）
    """
    return success(get_agent_defaults())


@router.post("/generate")
async def generate_config(request: Request, conn: Connection = Depends(get_conn)):
    """一句话需求 → Agent 配置（``dry_run=true`` 只预览不落库）——**已本地化**（2026-09-13）。

    ⚠️ 路由层先自己校验一遍（与 ``creator`` 内的校验**重复**，但**文案不同**）：
    ``agent_type required`` / ``requirement required`` 是路由的；而 creator 抛的是
    ``未知 Agent 类型：…`` / ``requirement 不能为空``。别"统一"这两套文案。

    ⚠️ 声明在 ``POST ""`` 之前只为可读性：``POST /generate`` 是字面量段，不会被它遮蔽。
    """
    try:
        body = await read_json(request)
        agent_type = body.get("agent_type")
        requirement = body.get("requirement")
        dry_run = body.get("dry_run") is True

        if not agent_type:
            return bad_request("agent_type required")
        if agent_type not in VALID_AGENT_TYPES:
            return bad_request(
                f"未知 Agent 类型：{agent_type}（可用：{', '.join(VALID_AGENT_TYPES)}）"
            )
        if not (isinstance(requirement, str) and requirement.strip()):
            return bad_request("requirement required")

        candidate = await generate_agent_config(conn, str(agent_type), str(requirement))
        payload_candidate = {
            "agentType": candidate.agent_type,
            "name": candidate.name,
            "description": candidate.description,
            "systemPrompt": candidate.system_prompt,
            "skills": candidate.skills,
        }
        if dry_run:
            return success({"candidate": payload_candidate})

        saved = persist_agent_config(candidate)
        return success({"candidate": payload_candidate, "saved": row_to_dict(saved)})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("AgentConfigsAPI", "generate", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.get("/{config_id}")
def get_config(config_id: str, conn: Connection = Depends(get_conn)):
    try:
        cid = parse_param_id(config_id)
        if cid is None:
            return not_found("Invalid agent config id")
        row = conn.execute(
            select(agent_configs).where(
                (agent_configs.c.id == cid) & (agent_configs.c.deleted_at.is_(None))
            )
        ).first()
        if row is None:
            return not_found("Not found")
        return success(row_to_dict(row))
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.post("")
def upsert_config(body: dict[str, Any], conn: Connection = Depends(get_tx)):
    """按 ``agent_type`` upsert（软删过的也会被复用并复活）。"""
    try:
        agent_type = body.get("agent_type")
        if not js_truthy(agent_type):
            return bad_request("agent_type required")

        skills_str: str | None = None
        # ⚠️ `if (body.skills)` 是真值判断 ⇒ 空数组 [] 在 JS 里为**真**，必须仍走校验
        if js_truthy(body.get("skills")):
            skills_str, err = _validate_skills(body["skills"])
            if err:
                return bad_request(err)

        ts = now()
        existing = conn.execute(
            select(agent_configs).where(agent_configs.c.agent_type == agent_type)
        ).first()

        if existing is not None:
            # ⚠️ name 用 `||`（空串回退旧值），其余用 `??`（空串保留）—— 语义不同，别统一
            values = {
                "name": body.get("name") if js_truthy(body.get("name")) else existing.name,
                "model": js_nullish(body.get("model"), existing.model),
                "system_prompt": js_nullish(body.get("system_prompt"), existing.system_prompt),
                "temperature": js_nullish(body.get("temperature"), existing.temperature),
                "max_tokens": js_nullish(body.get("max_tokens"), existing.max_tokens),
                "max_iterations": js_nullish(body.get("max_iterations"), existing.max_iterations),
                "skills": skills_str if skills_str is not None else existing.skills,
                "is_active": js_nullish(body.get("is_active"), True),
                "deleted_at": None,  # 复活软删记录
                "updated_at": ts,
            }
            conn.execute(
                agent_configs.update()
                .where(agent_configs.c.id == existing.id)
                .values(**values)
            )
            row = conn.execute(
                select(agent_configs).where(agent_configs.c.id == existing.id)
            ).first()
            return success(row_to_dict(row))

        result = conn.execute(
            agent_configs.insert().values(
                agent_type=agent_type,
                name=body.get("name") if js_truthy(body.get("name")) else "",
                description=body.get("description") if js_truthy(body.get("description")) else "",
                model=body.get("model") if js_truthy(body.get("model")) else "",
                system_prompt=(
                    body.get("system_prompt") if js_truthy(body.get("system_prompt")) else ""
                ),
                temperature=js_nullish(body.get("temperature"), 0.7),
                max_tokens=js_nullish(body.get("max_tokens"), 4096),
                max_iterations=js_nullish(body.get("max_iterations"), 10),
                skills=skills_str if js_truthy(skills_str) else None,
                is_active=js_nullish(body.get("is_active"), True),
                created_at=ts,
                updated_at=ts,
            )
        )
        new_id = int(result.inserted_primary_key[0])
        row = conn.execute(select(agent_configs).where(agent_configs.c.id == new_id)).first()
        return success(row_to_dict(row))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.put("/{config_id}")
def update_config(config_id: str, body: dict[str, Any], conn: Connection = Depends(get_tx)):
    try:
        cid = parse_param_id(config_id)
        if cid is None:
            return not_found("Invalid agent config id")

        # ⚠️ 原 TS 用 `'model' in body`（**键存在性**，不是真值）⇒ 显式传 null 会写入 null
        field_map = {
            "model": "model",
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "max_iterations": "max_iterations",
            "is_active": "is_active",
            "system_prompt": "system_prompt",
            "name": "name",
            "description": "description",
        }
        updates: dict[str, Any] = {"updated_at": now()}
        for key, column in field_map.items():
            if key in body:
                updates[column] = body[key]

        if "skills" in body:
            skills_str, err = _validate_skills(body["skills"])
            if err:
                return bad_request(err)
            updates["skills"] = skills_str

        conn.execute(agent_configs.update().where(agent_configs.c.id == cid).values(**updates))
        row = conn.execute(select(agent_configs).where(agent_configs.c.id == cid)).first()
        # ⚠️ 对不存在的 id：UPDATE 影响 0 行，随后 toSnakeCase(undefined) 在 JS 里返回 {}，
        # 于是这里会回 200 + data:{}（**不是 404**）。row_to_dict(None) 同样是 {} ✓
        return success(row_to_dict(row))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.delete("/{config_id}")
def delete_config(config_id: str, conn: Connection = Depends(get_tx)):
    try:
        cid = parse_param_id(config_id)
        if cid is None:
            return not_found("Invalid agent config id")
        conn.execute(
            agent_configs.update()
            .where((agent_configs.c.id == cid) & (agent_configs.c.deleted_at.is_(None)))
            .values(deleted_at=now())
        )
        # 原 TS 是 `success(c)` ⇒ `data` 取默认值 null ⇒ 返回 `{"code":200,"data":null,...}`
        return success()
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
