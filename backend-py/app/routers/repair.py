"""**修复域** —— 提案 / 应用 / 回滚（源：`assistant-runtime/server.mjs` 的 proposals 那一段 ✓）。

本域只做**文件级**修复：读源码、搜源码、把「找一段换一段」编译成提案、确认后落盘、
不想要就回滚。它**不调用任何模型** ✓ —— 谁生成提案（人、脚本、或 `agent` 域里的对话）
不在本域的职责里，这样修复内核可以单独测、单独审计。

⚠️ **提案为什么不走网络回传**：``propose`` 的返回里**不带** ``oldContent``（它是应用时
判断"方案过期"的基准 ✓✗），提案以 ``<dataRoot>/repair/proposals/<id>.json`` 落盘，
``apply`` 只收 ``proposalId``。让前端拿着整份旧源码来回传，等于把校验基准交给客户端 ✗。

⚠️ 权能边界：默认可改范围只有 ``frontend/app/``（与源 ``infinite-canvas/web/`` 同口径）。
需要放开时用 ``REPAIR_ALLOWED_PREFIXES``（逗号分隔）显式追加，**别写成空串**（空 = 不限 ✗）。

⚠️ **闭环**（知识闭环缺口 9）：``apply`` 之后必须回答「那个问题走了没有」，光回
``status=applied`` 只是说「改动写进去了」✗。所以提案可挂一枚缺口键（``finding``，
形如 ``missing:xxx.safetensors``，来自 ``/diagnose`` 或 ``/diagnose/history``）：应用当场
复诊、判 ``resolved``／``still``／``stale``／``unverified`` 四档并记进 ``outcome_index``；
改动落地慢（比如人手动下模型）时用 ``POST /verify`` 事后补判。判定口径与快照格式见
``services/repair_closure.py`` 模块头。
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
from ..services import repair_closure
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


def _diagnose_dir() -> Path:
    """诊断快照落点 ✓ —— 与提案分开存：快照是**事实层**（谁都不能改），提案是待确认的动作。"""
    return _repair_data_dir() / "diagnose"


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


def _attach_finding(proposal: dict[str, Any], raw: Any) -> None:
    """把「这条提案修的是哪条缺口」挂到提案上 ✓ —— 挂不上就**如实不挂**（不猜）。

    ``findingBefore.present`` 是**提案落单那一刻**的事实：应用后再看「还在不在」才有基准 ✓。
    当时没有快照 ⇒ 存 ``None``（判不了），而不是存 ``False`` —— 存 ``False`` 会让应用后
    一律判 ``stale``，等于把「没基准」说成「你抄错了」✗。
    """
    key = repair_closure.normalize_finding(raw)
    if not key:
        return
    proposal["finding"] = key
    snapshot = repair_closure.latest(_diagnose_dir())
    if snapshot is None:
        proposal["findingBefore"] = {
            "present": None, "snapshotId": None, "truncated": False,
            "note": "落单时还没有诊断快照 ⇒ 应用后只能判 unverified",
        }
        return
    truncated = bool(snapshot.get("truncated"))
    keys = {str(item) for item in (snapshot.get("keys") or []) if item}
    proposal["findingBefore"] = {
        "present": None if truncated else key in keys,
        "snapshotId": snapshot.get("id"),
        "truncated": truncated,
    }


def _verify_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """应用后**复诊**「这条缺口还在不在」✓ —— 顺带落快照、记跑次；复诊失败不改提案结果。"""
    key = proposal.get("finding")
    if not key:
        return {
            "status": "unverified",
            "finding": "",
            "reason": "提案没有挂缺口键（finding），应用后无对象可复诊",
            "checkedAt": None,
        }
    before = proposal.get("findingBefore") if isinstance(proposal.get("findingBefore"), dict) else {}
    was_present = before.get("present") if isinstance(before.get("present"), bool) else None
    outcome = repair_closure.verify(
        finding=key,
        was_present=was_present,
        diagnose_fn=lambda: repair_diag.diagnose(_workflows_dir()),
        directory=_diagnose_dir(),
    )
    outcome["wasPresent"] = was_present
    outcome["before"] = {name: before.get(name) for name in ("present", "snapshotId", "truncated")}
    proposal["verification"] = outcome
    _save_proposal(proposal)
    #: 写不进去只说明统计少一条，**不许**因此让已经落盘的应用报失败 ✓
    repair_closure.record(proposal.get("id"), outcome, findings_dir=_diagnose_dir())
    return outcome


def _persist_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """落一份诊断快照 ✓ —— 存不下不挡诊断（回执里如实写 ``saved=false``）。"""
    snapshot = repair_closure.build_snapshot(result)
    meta: dict[str, Any] = {
        "id": snapshot["id"],
        "at": snapshot["at"],
        "fingerprint": snapshot["fingerprint"],
        "counts": snapshot["counts"],
        "truncated": snapshot["truncated"],
        "saved": True,
    }
    try:
        repair_closure.save_snapshot(snapshot, _diagnose_dir())
    except OSError:
        meta["saved"] = False
    return meta


def _public_view(proposal: dict[str, Any]) -> dict[str, Any]:
    """给前端看的提案 ✓ —— **剥掉 oldContent/newContent**（可能上万字符，且是校验基准）。"""
    return {
        "id": proposal.get("id"),
        "summary": proposal.get("summary"),
        "status": proposal.get("status"),
        #: 修的是哪条缺口、以及「应用后它走没走」✓（没挂缺口就是 ``None``，别编一个 ``unverified``）
        "finding": proposal.get("finding"),
        "verification": proposal.get("verification"),
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
    """列工作区源码树 ✓（不进 node_modules/.git/data/models/indexes/corpus/output）。"""
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
    """生成提案 ✓ —— 每条 ``{path, search, replace}``（新建用 ``{path, create: true, replace}``）。

    可选 ``finding``：这条提案要修的那条缺口（``"missing:xxx.safetensors"``，从 ``/diagnose``
    的 ``missing`` 明细里抄键，或直接给 ``"xxx.safetensors"`` 按缺失处理）✓。挂了才会在
    ``apply`` 之后复诊；不挂也能用，只是应用后没有「问题走没走」可判 ⚠️。
    """
    body = await read_json(request)
    edits = body.get("edits")
    if not isinstance(edits, list) or not edits:
        return bad_request("edits 必须是非空数组")
    try:
        proposal = repair.make_proposal(
            _workspace_root(), body.get("summary"), edits,
            allowed_prefixes=_allowed_prefixes(),
        )
        _attach_finding(proposal, body.get("finding"))
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
    """应用提案 ✓ —— 先校验（方案过期就整单拒）、再备份、最后写；写中途失败就地回滚。

    挂了 ``finding`` 的提案在写盘成功后**当场复诊**一次，回执里带 ``verification`` ✓
    （``resolved`` 才是真修好；``still`` 说明改动加上了但问题没走 ✗）。
    """
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
    return success({**result, "status": "applied", "verification": _verify_proposal(proposal)})


@router.post("/rollback")
async def rollback(request: Request) -> object:
    """回滚提案 ✓ —— 改过的文件退回备份，新建的文件删除。

    回滚后把判定改成终态 ``rolled_back`` 并按 ``partial`` 覆盖跑次 ✓ —— 留着应用期的
    ``resolved/ok`` 会让修复成功率虚高（改动都没了，问题自然回来了）✗。
    """
    body = await read_json(request)
    try:
        proposal = _load_proposal(body.get("proposalId"))
        result = repair.rollback_proposal(proposal, _workspace_root(), _backup_dir())
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))
    proposal["status"] = "rolled_back"
    if proposal.get("finding"):
        outcome = repair_closure.rolled_back(
            proposal.get("finding"),
            previous=(proposal.get("verification") or {}).get("status"),
        )
        proposal["verification"] = outcome
        repair_closure.record(proposal.get("id"), outcome, findings_dir=_diagnose_dir())
    _save_proposal(proposal)
    return success({**result, "status": "rolled_back", "verification": proposal.get("verification")})


# ---------------------------------------------------------------- 诊断（源 diagnostics.mjs 重写）


def _workflows_dir() -> Path:
    """工作流目录 ✓ —— 默认仓库带的 ``configs/comfyui/workflows``，可用环境变量指到别处。"""
    raw = (os.environ.get("REPAIR_WORKFLOWS_DIR") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(PROJECT_ROOT).resolve() / "configs" / "comfyui" / "workflows"


@router.get("/diagnose")
def diagnose(limit: int = repair_diag.MAX_PROBLEMS, full: bool = False,
             persist: bool = True) -> object:
    """**现在缺什么、缺了堵住哪条工作流** ✓。

    为什么要有这个：``propose/apply`` 只能改，**发现**得靠人工跑比对脚本 ——
    源 ``diagnostics.mjs`` 的 8 个工具全打源网关端点，本仓没有，故按本仓数据源重写。

    ``full=false``（默认）回摘要（条数封顶，给模型看）；``full=true`` 回全量明细。

    ``persist=true``（默认）**顺带落一份快照** ✓（``/diagnose/history`` 看趋势、``/propose``
    的 ``findingBefore`` 也读它）；纯查询不想留痕时传 ``persist=false`` —— 快照是证据，
    不是缓存，只有真要「可回溯」时才写。
    """
    try:
        result = repair_diag.diagnose(_workflows_dir())
    except Exception as err:  # noqa: BLE001 —— 扫描盘面可能 IO 失败，给用户话而不是 500
        return bad_request(f"诊断失败：{err}")
    if not result.get("ok"):
        return bad_request(str(result.get("error") or "诊断失败"))
    payload = result if full else repair_diag.summarize(result, max(1, limit))
    if persist:
        payload["snapshot"] = _persist_snapshot(result)
    return success(payload)


@router.get("/diagnose/history")
def diagnose_history(limit: int = 20) -> object:
    """诊断快照的历史（新→旧）✓ —— 回答「缺口是在变少还是变多」。

    只回摘要（``id``／``at``／``fingerprint``／``counts``／``findings`` 条数／是否截断），
    不回每条明细：要看明细去 ``/diagnose`` 现诊。``fingerprint`` 相同＝同一批问题 ✓。
    """
    return success({
        "items": repair_closure.history(_diagnose_dir(), limit=max(1, int(limit))),
        "keep": repair_closure.KEEP_SNAPSHOTS,
        "dir": str(_diagnose_dir()),
    })


@router.post("/verify")
async def verify(request: Request) -> object:
    """**事后复诊**一份已应用的提案 ✓ —— 那些「改动加了、但修的东西要人去下／去换」的场合。

    例：提案改的是工作流里的模型文件名，可模型文件本身得人手动放盘上 ⇒ ``apply`` 那一刻
    复诊必然 ``still``。等文件到位后再打一次这个端点，结论变成 ``resolved`` 并覆盖同一条跑次 ✓。
    """
    body = await read_json(request)
    try:
        proposal = _load_proposal(body.get("proposalId"))
    except repair.RepairError as err:
        return bad_request(repair.repair_error_message(err))
    if proposal.get("status") != "applied":
        return bad_request(f"提案状态是 {proposal.get('status')}，只有 applied 才有「应用后」可复诊")
    return success(_verify_proposal(proposal))


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
