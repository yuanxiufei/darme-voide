"""**修复域** —— 提案 / 应用 / 回滚（源：`assistant-runtime/server.mjs` 的 proposals 那一段 ✓）。

本域只做**文件级**修复：读源码、搜源码、把「找一段换一段」编译成提案、确认后落盘、
不想要就回滚。它**不调用任何模型** ✓ —— 谁生成提案（人、脚本、或 `agent` 域里的对话）
不在本域的职责里，这样修复内核可以单独测、单独审计。

⚠️ **提案为什么不走网络回传**：``propose`` 的返回里**不带** ``oldContent``（它是应用时
判断"方案过期"的基准 ✓✗），提案以 ``<dataRoot>/repair/proposals/<id>.json`` 落盘，
``apply`` 只收 ``proposalId``。让前端拿着整份旧源码来回传，等于把校验基准交给客户端 ✗。

⚠️ 权能边界：默认可改范围只有 ``frontend/app/``（与源 ``infinite-canvas/web/`` 同口径）。
需要放开时用 ``REPAIR_ALLOWED_PREFIXES``（逗号分隔）显式追加，**别写成空串**（空 = 不限 ✗）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from ..core.config import PROJECT_ROOT, get_data_root
from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services import repair_diagnose as repair_diag
from ..services import repair_skill as repair_skill
from ..services import repair_tools as repair

router = APIRouter(prefix="/api/v1/repair", tags=["repair"])


def _workspace_root() -> Path:
    """修复工作区根 ✓ —— 默认仓库根；可用 ``REPAIR_WORKSPACE_ROOT`` 指到别处。"""
    raw = (os.environ.get("REPAIR_WORKSPACE_ROOT") or "").strip()
    return Path(raw).resolve() if raw else Path(PROJECT_ROOT).resolve()


def _repair_data_dir() -> Path:
    """提案与备份的落点 ✓（在数据根下，**不在**源码树里 —— 否则会被自己扫到）。"""
    return Path(get_data_root()) / "repair"


def _proposal_dir() -> Path:
    return _repair_data_dir() / "proposals"


def _backup_dir() -> Path:
    return _repair_data_dir() / "backups"


def _allowed_prefixes() -> tuple[str, ...]:
    """可改目录白名单 ✓（默认只前端；显式配置才放开）。"""
    raw = (os.environ.get("REPAIR_ALLOWED_PREFIXES") or "").strip()
    if not raw:
        return repair.DEFAULT_ALLOWED_PREFIXES
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _proposal_file(proposal_id: str) -> Path:
    safe = "".join(char for char in str(proposal_id) if char.isalnum())
    if not safe:
        raise repair.RepairError("proposalId 不合法")
    return _proposal_dir() / f"{safe}.json"


def _load_proposal(proposal_id: Any) -> dict[str, Any]:
    path = _proposal_file(str(proposal_id))
    if not path.exists():
        raise repair.RepairError(f"提案不存在：{proposal_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_proposal(proposal: dict[str, Any]) -> None:
    directory = _proposal_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = _proposal_file(str(proposal["id"]))
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=1), encoding="utf-8")


def _public_view(proposal: dict[str, Any]) -> dict[str, Any]:
    """给前端看的提案 ✓ —— **剥掉 oldContent/newContent**（可能上万字符，且是校验基准）。"""
    return {
        "id": proposal.get("id"),
        "summary": proposal.get("summary"),
        "status": proposal.get("status"),
        "edits": [
            {
                "path": edit.get("path"),
                "isCreate": bool(edit.get("isCreate")),
                "oldPreview": edit.get("oldPreview", ""),
                "newPreview": edit.get("newPreview", ""),
            }
            for edit in proposal.get("edits") or []
        ],
    }


# ---------------------------------------------------------------- 只读


@router.get("/workspace")
def workspace(path: str = ".", depth: int = 2) -> object:
    """列工作区源码树 ✓（不进 node_modules/.git/data/models/output）。"""
    try:
        return success(repair.list_workspace(_workspace_root(), path, depth))
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))


@router.get("/source")
def source(path: str, start_line: int = 1, max_lines: int = 400) -> object:
    """读一段源码 ✓（带行号、限制行数）。"""
    try:
        return success(repair.read_source(_workspace_root(), path, start_line=start_line, max_lines=max_lines))
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))


@router.get("/search")
def search(q: str = Query(default=""), path: str = ".", max_results: int = repair.MAX_SEARCH_RESULTS) -> object:
    """全文搜索 ✓（纯字符串包含，不做正则）。"""
    try:
        return success(repair.search_source(_workspace_root(), q, relative=path, max_results=max_results))
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))


# ---------------------------------------------------------------- 提案 / 应用 / 回滚


@router.post("/propose")
async def propose(request: Request) -> object:
    """生成提案 ✓ —— 每条 ``{path, search, replace}``（新建用 ``{path, create: true, replace}``）。"""
    body = await read_json(request)
    edits = body.get("edits")
    if not isinstance(edits, list) or not edits:
        return bad_request("edits 必须是非空数组")
    try:
        proposal = repair.make_proposal(
            _workspace_root(), body.get("summary"), edits,
            allowed_prefixes=_allowed_prefixes(),
        )
        _save_proposal(proposal)
        return success(_public_view(proposal))
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))


@router.get("/proposals")
def proposals() -> object:
    """列出提案（新→旧）✓。"""
    directory = _proposal_dir()
    if not directory.exists():
        return success({"items": []})
    items: list[dict[str, Any]] = []
    files = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in files[:50]:
        try:
            items.append(_public_view(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return success({"items": items})


@router.post("/apply")
async def apply(request: Request) -> object:
    """应用提案 ✓ —— 先校验（方案过期就整单拒）、再备份、最后写；写中途失败就地回滚。"""
    body = await read_json(request)
    try:
        proposal = _load_proposal(body.get("proposalId"))
        if proposal.get("status") not in ("pending", "rolled_back"):
            return bad_request(f"提案状态是 {proposal.get('status')}，不能重复应用")
        result = repair.apply_proposal(proposal, _workspace_root(), _backup_dir())
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))
    proposal["status"] = "applied"
    _save_proposal(proposal)
    return success({**result, "status": "applied"})


@router.post("/rollback")
async def rollback(request: Request) -> object:
    """回滚提案 ✓ —— 改过的文件退回备份，新建的文件删除。"""
    body = await read_json(request)
    try:
        proposal = _load_proposal(body.get("proposalId"))
        result = repair.rollback_proposal(proposal, _workspace_root(), _backup_dir())
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))
    proposal["status"] = "rolled_back"
    _save_proposal(proposal)
    return success({**result, "status": "rolled_back"})


# ---------------------------------------------------------------- 诊断（源 diagnostics.mjs 重写）


def _workflows_dir() -> Path:
    """工作流目录 ✓ —— 默认仓库带的 ``configs/comfyui/workflows``，可用环境变量指到别处。"""
    raw = (os.environ.get("REPAIR_WORKFLOWS_DIR") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(PROJECT_ROOT).resolve() / "configs" / "comfyui" / "workflows"


@router.get("/diagnose")
def diagnose(limit: int = repair_diag.MAX_PROBLEMS, full: bool = False) -> object:
    """**现在缺什么、缺了堵住哪条工作流** ✓。

    为什么要有这个：``propose/apply`` 只能改，**发现**得靠人工跑比对脚本 ——
    源 ``diagnostics.mjs`` 的 8 个工具全打源网关端点，本仓没有，故按本仓数据源重写。

    ``full=false``（默认）回摘要（条数封顶，给模型看）；``full=true`` 回全量明细。
    """
    try:
        result = repair_diag.diagnose(_workflows_dir())
    except Exception as err:  # noqa: BLE001 —— 扫描盘面可能 IO 失败，给用户话而不是 500
        return bad_request(f"诊断失败：{err}")
    if not result.get("ok"):
        return bad_request(str(result.get("error") or "诊断失败"))
    return success(result if full else repair_diag.summarize(result, max(1, limit)))


# ---------------------------------------------------------------- 技能说明（源 skill.mjs 重写）


@router.get("/skill")
def skill(context: str | None = None) -> object:
    """修复助手的**系统提示词素材**（L0–L4 权能边界）✓。

    为什么要有这个：源 ``skill.mjs`` 是「一段 Markdown + 一个拼装函数」，本仓没有对应文件；
    但修复域的护栏（可改范围、拒绝段、提案上限）**得让模型先知道**，否则它会提出
    一个注定被 ``apply`` 拒掉的方案 ✗。这里按本仓实现**现算**正文（清单从
    ``repair_tools`` / ``repair_diagnose`` 取值），所以不可能与实现漂移。

    ``context`` 可给一段 JSON 字符串（当前画布/选中节点等），原样嵌进提示词；非法 JSON 忽略。
    """
    parsed: Any = None
    raw = (context or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    text = repair_skill.build_system_prompt(
        context=parsed,
        workspace_root=str(_workspace_root()),
        allowed_prefixes=_allowed_prefixes(),
    )
    return success({
        "name": repair_skill.SKILL_NAME,
        "version": repair_skill.SKILL_VERSION,
        "markdown": text,
    })
