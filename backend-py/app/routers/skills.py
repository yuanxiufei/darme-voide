"""``/api/v1/skills`` —— 与 ``backend/src/routes/skills.ts`` 对齐（6 端点，全部迁移）。

列表 / 侧栏元信息 / 读取 / 保存 / 新建 / 删除。

**这是纯文件系统域**（``skills/`` 目录，不是数据库）—— 绞杀期两个后端读同一份目录，无冲突。

⚠️ 五个照抄的行为：

1. **路由顺序**：``GET /meta`` 必须早于 ``GET /{skill_id}``（同段数，否则 ``meta`` 被当 skill id）。
2. **通配路径**：skill id 允许含 ``/``（嵌套 skill）⇒ 用 FastAPI 的 ``/{skill_id:path}``。
3. **删除保护**（删除闸）：顶层目录 **且** frontmatter ``agents:`` 非空的 skill 拒删 ——
   判据两项都满足才算保护对象；``agents`` 读不到（解析失败）时**保守按资产处理**。
   ``agents`` 传给 ``None`` 与 ``[]`` 语义完全不同：前者=解析失败→保护，后者=空声明→可删。
4. **``GET /skills`` 的扫描会**同时**记录当前目录的 skill 并继续递归**（子目录里可能还有 skill）。
5. 成功但无 warning 时返回 ``{"code":200,"data":null,...}``（``success(c, undefined)`` 走默认参数）。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.db import get_conn
from ..core.models import agent_configs
from ..core.response import bad_request, internal_error, js_truthy, success
from ..services.agent_registry import AGENT_PHASES, get_default_name, get_host_tool_names
from ..services.skill_parser import ParsedSkill, parse_skill, render_skill
from ..services.skills import (
    SKILL_CHAR_BUDGET,
    SKILLS_DIR,
    list_skill_ids,
    resolve_default_skills,
)

router = APIRouter(prefix="/api/v1/skills", tags=["skills"])

#: 外部技能库声明文件名：顶层目录含此文件 = 一个技能库（vendor）
LIBRARY_FILE = "library.yaml"

#: 允许的 skill id 字符集（再配合「无 `..`、非绝对路径、解析后仍在 SKILLS_DIR 内」三重校验）
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-./]+$")


# ---------------------------------------------------------------------------
# 技能库声明 / 分类
# ---------------------------------------------------------------------------

def _read_library(dir_name: str) -> dict[str, str]:
    """读取 ``skills/<lib>/library.yaml``。

    **库靠显式声明识别，不靠目录名、也不靠层级猜测** ⇒ 「加库 / 换库 / 改展示名」都是纯文件操作。
    声明损坏时回退「目录名即库名」，不让整个列表接口失败。
    """
    fallback = {"id": dir_name, "label": dir_name, "description": ""}
    file = SKILLS_DIR / dir_name / LIBRARY_FILE
    if not file.exists():
        return fallback
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return fallback
        name, label, description = raw.get("name"), raw.get("label"), raw.get("description")
        lib_id = name.strip() if isinstance(name, str) and name.strip() else dir_name
        lib_label = label.strip() if isinstance(label, str) and label.strip() else lib_id
        return {
            "id": lib_id,
            "label": lib_label,
            "description": description.strip() if isinstance(description, str) else "",
        }
    except Exception:  # noqa: BLE001 - 声明损坏不该让列表接口失败
        return fallback


def _list_libraries() -> list[dict[str, str]]:
    """扫描 ``skills/`` 顶层，列出全部外部技能库（按库标识排序）。"""
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        (
            _read_library(entry.name)
            for entry in SKILLS_DIR.iterdir()
            if entry.is_dir() and (entry / LIBRARY_FILE).exists()
        ),
        key=lambda lib: lib["id"],
    )


def _classify(skill_id: str) -> tuple[str, str | None]:
    """推导 skill 来源（换库/加库/改库名零代码改动）。

    ``core``   —— 顶层目录自身含 SKILL.md，即项目自有 skill（与工作流、工具名、字段契约对齐）
    ``vendor`` —— 顶层目录含 library.yaml，即外部技能库；source = 库标识（来自声明，可与目录名不同）
    """
    if "/" not in skill_id:
        return "core", None
    top = skill_id.split("/")[0]
    # 顶层目录自身含 SKILL.md → 自有 skill 的子目录，其子项仍按自有处理
    if (SKILLS_DIR / top / "SKILL.md").exists():
        return "core", None
    if (SKILLS_DIR / top / LIBRARY_FILE).exists():
        return "vendor", _read_library(top)["id"]
    # 兜底：无声明文件的容器目录，退化为「目录名即库名」
    return "vendor", top


def _validate_skill_id(raw_id: str) -> str | None:
    """校验 skill id 安全，**防路径遍历**。返回净化后的 id 或 None。"""
    sanitized = re.sub(r"[\0\r\n]", "", raw_id)
    if not _SAFE_ID_RE.match(sanitized):
        return None
    if sanitized.startswith("/") or sanitized.startswith("\\"):
        return None
    if ".." in sanitized:
        return None
    resolved = (SKILLS_DIR / sanitized).resolve()
    # 与原 TS 同形：既允许严格在 SKILLS_DIR 之下，也允许恰好等于它
    if not (str(resolved).startswith(str(SKILLS_DIR) + "/") or str(resolved).startswith(str(SKILLS_DIR) + "\\") or str(resolved) == str(SKILLS_DIR)):
        return None
    return sanitized


def _safe_skill_path(skill_id: str) -> Path:
    return SKILLS_DIR / skill_id / "SKILL.md"


def _safe_skill_dir(skill_id: str) -> Path:
    return SKILLS_DIR / skill_id


def _is_protected_skill(skill_id: str, agents: list[str] | None) -> bool:
    """删除保护判定（删除闸）。

    保护对象 = **被默认注入的项目资产**，判据两项都要满足：

    ① 顶层目录（id 不含 ``/``）—— 「Agent 根 skill」或「提示词库」；
    ② frontmatter ``agents:`` 声明非空 —— 确实参与默认注入，删除会直接改变 Agent 行为，
       其中未纳入版本控制的部分（如 prompt-style-library）删掉即**永久丢失**。

    用户自建 skill（``skills/<agent>/<name>/``，id 含 ``/``）不受保护；**顶层但未声明 agents**
    （手工建目录 / 接口直连创建，不参与默认注入）同样可删 —— 否则合法删除会被永久误拒。
    ``agents`` 传 ``None``（文件读不到 / 解析失败）时保守按资产处理。
    """
    if "/" in skill_id:
        return False
    if agents is None:
        return True
    return len(agents) > 0


def _count_reference_files(skill_dir: Path) -> int:
    """统计 ``references/`` 内的参考文件数（注入时不会展开 ⇒ 前端据此提示「引用不可达」）。"""
    ref_dir = skill_dir / "references"
    if not ref_dir.exists():
        return 0
    count = 0

    def walk(directory: Path) -> None:
        nonlocal count
        for entry in directory.iterdir():
            if entry.is_dir():
                walk(entry)
            else:
                count += 1

    try:
        walk(ref_dir)
    except OSError:
        pass  # 读取失败按已统计的部分返回，不阻断列表
    return count


def _injected_chars(parsed: ParsedSkill) -> int:
    """真实注入体量（字符）= ``render_skill`` 后的长度，与注入时的预算闸**同口径**。

    不能直接用文件字节数：前置契约 / 协议字段段的拼接同样计入注入。
    """
    try:
        return len(render_skill(parsed))
    except Exception:  # noqa: BLE001
        return 0


def _injected_chars_of(skill_id: str) -> int:
    file = _safe_skill_path(skill_id)
    if not file.exists():
        return 0
    try:
        return _injected_chars(parse_skill(file.read_text(encoding="utf-8"), skill_id))
    except OSError:
        return 0


def _load_skill_bindings(conn: Connection) -> dict[str, list[str]]:
    """反向查询 skill 绑定关系：``skillId → agent_type`` 列表（含已禁用的）。

    优先 DB ``agent_configs.skills``（用户显式配置）；DB 未配置（null）的 agent 回退该 skill
    frontmatter ``agents:`` 里的自描述声明。
    """
    bindings: dict[str, list[str]] = {}

    def add(skill_id: str, agent: str) -> None:
        current = bindings.setdefault(skill_id, [])
        if agent not in current:
            current.append(agent)

    for agent in AGENT_PHASES:
        for skill_id in resolve_default_skills(agent):
            add(skill_id, agent)

    rows = conn.execute(
        select(agent_configs).where(agent_configs.c.deleted_at.is_(None))
    ).all()
    for row in rows:
        if not row.skills:
            continue  # 未配置 → 保留默认映射
        try:
            arr = json.loads(row.skills)
        except (ValueError, TypeError):
            continue  # 忽略损坏的 skills JSON
        if not isinstance(arr, list):
            continue
        # 用户显式配置：先移除该 agent 的默认绑定，再按用户配置重建
        for agents in bindings.values():
            if row.agent_type in agents:
                agents.remove(row.agent_type)
        for item in arr:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            add(item["id"], row.agent_type)
    return bindings


# ---------------------------------------------------------------------------
# GET /skills
# ---------------------------------------------------------------------------

@router.get("")
def list_skills(conn: Connection = Depends(get_conn)):
    try:
        bindings = _load_skill_bindings(conn)
        library_labels = {lib["id"]: lib["label"] for lib in _list_libraries()}
        host_tools = get_host_tool_names()
        skills: list[dict[str, Any]] = []

        if not SKILLS_DIR.is_dir():
            return success(skills)

        def scan(directory: Path, prefix: str = "") -> None:
            # 排序仅为**确定性**（原 TS 用 readdirSync 的 OS 顺序，属任意序）
            for entry in sorted(directory.iterdir(), key=lambda p: p.name):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                skill_id = f"{prefix}/{entry.name}" if prefix else entry.name
                if skill_file.exists():
                    parsed = parse_skill(skill_file.read_text(encoding="utf-8"), entry.name)
                    bound_agents = bindings.get(skill_id, [])
                    category, source = _classify(skill_id)
                    allowed_tools = parsed.metadata.allowed_tools
                    # 真实依赖 = frontmatter 声明 ∪ 正文引用（二者都可能指向宿主未注册的工具）
                    required_tools = list(
                        dict.fromkeys([*allowed_tools, *parsed.foreign_tool_refs])
                    )
                    skills.append(
                        {
                            "id": skill_id,
                            "name": parsed.metadata.name,
                            "description": parsed.metadata.description,
                            "preconditions": parsed.metadata.preconditions,
                            "protocol": parsed.metadata.protocol,
                            "category": category,
                            "source": source,
                            "sourceLabel": (library_labels.get(source) or source) if source else None,
                            "workflows": parsed.metadata.workflows,
                            "agents": parsed.metadata.agents,
                            "priority": parsed.metadata.priority,
                            "allowedTools": allowed_tools,
                            "foreignToolRefs": parsed.foreign_tool_refs,
                            "missingTools": [t for t in required_tools if t not in host_tools],
                            "charCount": _injected_chars(parsed),
                            "referenceCount": _count_reference_files(entry),
                            "protected": _is_protected_skill(skill_id, parsed.metadata.agents),
                            "boundAgents": bound_agents,
                            "phases": list(
                                dict.fromkeys(
                                    [
                                        *(AGENT_PHASES[a] for a in bound_agents if a in AGENT_PHASES),
                                        *parsed.metadata.workflows,
                                    ]
                                )
                            ),
                        }
                    )
                # Always recurse —— 子目录里可能还有嵌套 skill
                scan(entry, skill_id)

        scan(SKILLS_DIR)
        return success(skills)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


# ---------------------------------------------------------------------------
# GET /skills/meta（**必须早于 /{skill_id}**）
# ---------------------------------------------------------------------------

@router.get("/meta")
def skills_meta():
    """侧栏元信息：主流程 Agent + 外部技能库（全部由目录结构推导，前端不再硬编码）。"""
    try:
        declared = _list_libraries()
        by_id: dict[str, dict[str, str]] = {lib["id"]: lib for lib in declared}
        core_count = 0
        counters: dict[str, int] = {}

        for skill_id in list_skill_ids():
            category, source = _classify(skill_id)
            if category == "vendor" and source:
                counters[source] = counters.get(source, 0) + 1
                # 兜底库（顶层目录无 library.yaml）：补一个库条目，保证「core + 各库 = 总数」
                # 自洽，且这些 skill 在侧栏有入口（否则会从计数里消失、UI 也不可达）
                if source not in by_id:
                    by_id[source] = {"id": source, "label": source, "description": ""}
            else:
                core_count += 1

        agents = []
        for agent_type, phase in AGENT_PHASES.items():
            ids = resolve_default_skills(agent_type)
            agents.append(
                {
                    "type": agent_type,
                    "label": get_default_name(agent_type),
                    "phase": phase,
                    "skillCount": len(ids),
                    # 默认绑定的实际注入体量合计（与 charBudget 对比即可看出余量/超预算风险）
                    "charCount": sum(_injected_chars_of(i) for i in ids),
                }
            )

        sources = [
            {
                "id": lib["id"],
                "label": lib["label"],
                "description": lib["description"],
                "skillCount": counters.get(lib["id"], 0),
                # 是否带 library.yaml 声明（false = 无声明的兜底库，展示名即目录名）
                "declared": any(d["id"] == lib["id"] for d in declared),
            }
            for lib in by_id.values()
        ]
        sources.sort(key=lambda lib: lib["id"])

        return success(
            {
                "agents": agents,
                "charBudget": SKILL_CHAR_BUDGET,
                "sources": sources,
                "coreCount": core_count,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


# ---------------------------------------------------------------------------
# GET / PUT / DELETE /skills/{id}（id 可含 /）
# ---------------------------------------------------------------------------

@router.get("/{skill_id:path}")
def get_skill(skill_id: str):
    try:
        sid = _validate_skill_id(skill_id)
        if sid is None:
            return bad_request("Invalid skill id")
        file = _safe_skill_path(sid)
        if not file.exists():
            return bad_request("Skill not found")
        return success({"id": sid, "content": file.read_text(encoding="utf-8")})
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.put("/{skill_id:path}")
def save_skill(skill_id: str, body: dict[str, Any]):
    try:
        sid = _validate_skill_id(skill_id)
        if sid is None:
            return bad_request("Invalid skill id")

        content = body.get("content")
        if not isinstance(content, str):
            # 原 TS 会因 writeFileSync(undefined) 抛错 → 500
            raise TypeError("content is required and must be a string")

        skill_dir = _safe_skill_dir(sid)
        skill_dir.mkdir(parents=True, exist_ok=True)
        _safe_skill_path(sid).write_text(content, encoding="utf-8")

        # 保存后校验 frontmatter：正文头部被误删/改坏会让 name/description 丢失、agents 声明失效
        # （该 Skill 静默退出默认注入）—— 这类「静默行为变更」必须回告前端
        warning: str | None = None
        if not re.match(r"^\s*---", content):
            warning = "内容缺少 frontmatter（--- 包裹的头部），name / description / agents 声明将无法识别。"
        elif len(parse_skill(content, sid).metadata.agents) == 0:
            warning = "frontmatter 的 agents 声明为空，该 Skill 不再默认注入任何 Agent（可在「Agent 配置 → 绑定 Skills」中手动启用）。"

        return success({"warning": warning}) if warning else success()
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.post("")
def create_skill(body: dict[str, Any]):
    try:
        skill_id = body.get("id")
        if not js_truthy(skill_id):
            return bad_request("Skill id is required")
        if not isinstance(skill_id, str):
            # 原 TS：非字符串会在 validateSkillId 的 .replace 上抛错 → 500
            raise TypeError("id must be a string")

        valid_id = _validate_skill_id(skill_id)
        if valid_id is None:
            return bad_request("Invalid skill id")

        skill_dir = _safe_skill_dir(valid_id)
        if skill_dir.exists():
            return bad_request("Skill already exists")

        name = body.get("name")
        description = body.get("description")
        display = name if js_truthy(name) else valid_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        _safe_skill_path(valid_id).write_text(
            f"""---
name: {display}
description: {description if js_truthy(description) else ''}
preconditions: []
protocol: []
# agents: 声明默认注入哪些 Agent（本项目为 script_rewriter / extractor / storyboard_breaker /
#         voice_assigner / grid_prompt_generator）；留空 = 不默认注入，仅在前端按需手动绑定
agents: []
# priority: 注入顺序，越小越靠前（缺省 100）
priority: 100
---

# {display}

Write your skill content here.
""",
            encoding="utf-8",
        )
        return success(
            {
                "id": valid_id,
                "name": display,
                "description": description if js_truthy(description) else "",
            }
        )
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


@router.delete("/{skill_id:path}")
def delete_skill(skill_id: str):
    try:
        sid = _validate_skill_id(skill_id)
        if sid is None:
            return bad_request("Invalid skill id")
        skill_dir = _safe_skill_dir(sid)
        if not skill_dir.exists():
            return bad_request("Skill not found")

        # 受保护 = 顶层且被 frontmatter `agents:` 默认注入的资产；解析失败 → 保守保护
        declared_agents: list[str] | None
        try:
            declared_agents = parse_skill(
                _safe_skill_path(sid).read_text(encoding="utf-8"), sid
            ).metadata.agents
        except OSError:
            declared_agents = None

        if _is_protected_skill(sid, declared_agents):
            return bad_request(
                f"「{sid}」是项目自有 Skill（由 frontmatter agents 声明默认注入，受删除保护），"
                "不支持删除。如需停用，请在「Agent 配置 → 绑定 Skills」中取消勾选。"
            )
        shutil.rmtree(skill_dir, ignore_errors=True)
        return success()
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
