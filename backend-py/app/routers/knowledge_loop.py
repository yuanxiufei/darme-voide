"""``/api/v1/knowledge-loop`` —— 知识闭环四个模块各自的读写口。

为什么单开一个前缀，而不是塞进 ``/api/v1/knowledge``
====================================================
``/knowledge`` 是**外部剧本资料**的读写口（5403 份 PDF 的索引库 + 覆盖层），
本文件是**本仓自己跑出来的经验**（runs / entries / 投喂）。两者共用一张表会让
「删语料」与「删经验」的权限、备份、重建策略纠缠在一起 —— 分开更省事。

四个模块在这张路由上的位置
==========================
* 知识积累 —— ``POST /entries``（人写）＋ ``POST /distill``（从跑次蒸馏）
  ＋ ``GET /scheduler``（定时复盘的状态：**默认关**，关着也照实答 ``enabled=false``）
* Agent 自主学习 —— 走 ``app/agent/tools/knowledge_tools.py``（同一套 service，
  只是暴露给模型而不是 HTTP）；本文件提供 ``GET /search`` 供前端/调试用
* 引擎自我完善 —— ``GET /runs/stats``（同一份聚合）＋ ``GET /tuning/suggest``（只读候选）
  ＋ ``/tuning`` 的读/写/删（写入要 basis + 可核 evidence，写前自动备份）
* 模型投喂 —— ``GET /feed/preview``（L1 编译预览）＋ ``POST /feed/export``（L3 数据集）
  ＋ ``GET /datasets``（已导出的数据集 = 提交训练时 ``datasetRef`` 的可选值）
  ＋ ``GET /training``（训练任务与产物登记，重启后仍可查）

回执纪律（沿用 ``/knowledge`` 的三条，另加一条）
=============================================
1. ``available=false``（库没建/坏了）与「查到 0 条」必须分开报；
2. 被拒的写入必须带 ``why``（"证据核不过"与"标题为空"不是一回事）；
3. ``ranking`` 必须透出（``bm25``/``like``/``recent`` 三种排序不可比）；
4. **``dryRun`` 必须真的不写库**，且回执里写明"未写入"—— 预览与落库走同一个函数，
   靠参数分叉，不写第二套逻辑（否则预览永远比实际好看）。
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.response import (
    bad_request,
    conflict,
    forbidden,
    not_found,
    server_error,
    success,
    unprocessable,
)
from ..services import (
    knowledge_distill,
    knowledge_feed,
    knowledge_ledger,
    knowledge_scheduler,
    optimization_log,
    outcome_index,
    qc_reviews,
    training_registry,
)
from ..services.engine import tuning

router = APIRouter(prefix="/api/v1/knowledge-loop", tags=["knowledge-loop"])

#: 一天多少纳秒（蒸馏/统计的时间窗换算）
_DAY_NS = 86_400_000_000_000


def _since_ns(days: Any) -> int | None:
    """``sinceDays`` → 纳秒。**非数字直接 None**（不猜成 0 天 = 全量）。"""
    if days is None or days == "":
        return None
    try:
        return time.time_ns() - int(float(days) * _DAY_NS)
    except (TypeError, ValueError):
        return None


@router.get("/runs")
def get_runs(
    kind: str | None = Query(None, description="agent / image / video / tts / qc / train / repair"),
    drama_id: str | None = Query(None, alias="dramaId"),
    episode_id: str | None = Query(None, alias="episodeId"),
    storyboard_id: str | None = Query(None, alias="storyboardId"),
    shot_id: str | None = Query(None, alias="shotId"),
    status: str | None = Query(None, description="ok / failed / partial"),
    stage: str | None = Query(None),
    agent_type: str | None = Query(None, alias="agentType"),
    since_ns: int | None = Query(None, alias="sinceNs"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Any:
    """按剧/集/镜/阶段/状态查跑次（新到旧）。

    ``total`` 与 ``items`` 必须分开读：``total=3`` 是"就 3 条"，
    ``total=900, items=100`` 是"只给了前 100 条"。
    """
    return success(outcome_index.query(
        kind=kind, drama_id=drama_id, episode_id=episode_id,
        storyboard_id=storyboard_id, shot_id=shot_id, status=status,
        stage=stage, agent_type=agent_type, since_ns=since_ns,
        limit=limit, offset=offset,
    ))


@router.get("/runs/stats")
def get_runs_stats(
    group_by: str = Query("stage", alias="groupBy",
                          description="stage / status / kind / model / agent_type / drama_id …"),
    kind: str | None = Query(None),
    since_days: float | None = Query(None, alias="sinceDays",
                                     description="只看最近 N 天（不传=全量）"),
) -> Any:
    """聚合：条数／失败率／平均分／耗时 P50·P95，供看板与参数建议共用。"""
    try:
        return success(outcome_index.stats(
            group_by=group_by, kind=kind, since_ns=_since_ns(since_days)))
    except ValueError as exc:  # groupBy 非法 —— 明确 400，别静默回全量
        return bad_request(str(exc))


@router.post("/runs/rebuild")
def post_runs_rebuild(payload: dict[str, Any] | None = None) -> Any:
    """从 ``traces/`` 增量重建事实层（按 (mtime,size) 指纹跳过未变的文件）。

    这是**幂等**操作：同一批 trace 重建两次，``runs`` 内容一致（``run_id`` 覆盖写）。
    """
    body = payload or {}
    outcome = outcome_index.rebuild(
        since_ns=_since_ns(body.get("sinceDays")),
        dry_run=bool(body.get("dryRun")),
        limit=int(body["limit"]) if body.get("limit") else None,
    )
    return success(outcome)


@router.get("/entries")
def get_entries(
    kind: str | None = Query(None, description="fact/work/recipe/lesson/failure/threshold/prompt_patch"),
    scope: str | None = Query(None, description="真实字面形：global / EP12 / DRAMA3 / stage:krea2 / shot:SHOT-012"),
    status: str | None = Query(None, description="proposed / accepted / deprecated / rejected"),
    tags: str | None = Query(None, description="逗号分隔，**AND 语义**（多标签＝同时具备）"),
    include_deleted: bool = Query(False, alias="includeDeleted",
                                  description="回收站视角：连同已软删的一起列"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Any:
    """条目清单（分页）。**默认不筛状态** ⇒ 提案与已采纳混在一起，前端必须显式带上。"""
    return success(knowledge_ledger.list_entries(
        kind=kind, scope=scope, status=status, tags=tags,
        include_deleted=include_deleted, limit=limit, offset=offset))


@router.get("/entries/{entry_id}")
def get_entry(
    entry_id: str,
    include_deleted: bool = Query(False, alias="includeDeleted",
                                  description="回收站里看一条已删条目时必须显式带上"),
) -> Any:
    """条目详情 ＋ 挂载关系 ＋ 被注入过的记录。

    已删条目默认当「不存在」（404）：否则这条端点就是绕过软删的后门 ——
    回收站要看得显式 ``?includeDeleted=true``。
    """
    item = knowledge_ledger.detail(entry_id, include_deleted=include_deleted)
    if item is None:
        return not_found(f"没有这条知识：{entry_id}")
    return success(item)


@router.get("/entries/{entry_id}/related")
def get_entry_related(
    entry_id: str,
    limit: int = Query(10, ge=1, le=50),
    include_deleted: bool = Query(False, alias="includeDeleted",
                                  description="回收站里看邻居时必须显式带上"),
) -> Any:
    """一条知识的邻居：**共用同一 stage/shot 的兄弟条目** ＋ **正文相似的条目**。

    条目不存在不报 404 —— 回 ``found=false`` 而不是异常，因为这是「详情面板的附加信息」，
    为它把整页打成错误不值；但**也不假装成功**（``found`` 字段如实说）。
    已软删的条目默认按「不存在」对待（与详情一致：否则就是绕过软删的后门）。
    """
    return success(knowledge_ledger.related(entry_id, limit=limit,
                                            include_deleted=include_deleted))


@router.get("/context")
def get_context(
    stage: str | None = Query(None, description="阶段名，如 image / video / h3（蒸馏写的是这几个字面值）"),
    shot: str | None = Query(None, description="镜头 id，如 SHOT-001"),
    status: str | None = Query(None, description="只取该状态，如 accepted"),
    include_deleted: bool = Query(False, alias="includeDeleted"),
    limit: int = Query(10, ge=1, le=50),
) -> Any:
    """按「阶段 / 镜头」反查经验（读 ``entry_links`` 反向索引）。

    ``stage``/``shot`` 都不给 ⇒ **空结果 + 人话**，不返回全量：把"没点名上下文"
    当成"全部都是上下文命中"，调用方会以为拿到了针对性经验。
    """
    return success(knowledge_ledger.for_context(
        stage=stage, shot=shot, status=status,
        include_deleted=include_deleted, limit=limit))


@router.get("/context/refs")
def get_context_refs(
    ref_kind: str | None = Query(None, alias="refKind",
                                 description="stage / shot（不传＝两种都给）"),
    entry: str | None = Query(None, description="只看这一条挂的 ref"),
    limit: int = Query(200, ge=1, le=500),
) -> Any:
    """台账里**真被挂过**的 ref 清单（附条目数）。

    没有它前端只能猜阶段名（``video``? ``h3``?），猜错的表现是"查了没结果"，
    与"确实没有经验"永远分不开。
    """
    return success(knowledge_ledger.context_refs(
        ref_kind=ref_kind, entry=entry, limit=limit))


@router.post("/entries")
def post_entry(payload: dict[str, Any]) -> Any:
    """人工写一条（默认 ``proposed``；人可以直接写 ``accepted``——审的人就是作者）。

    正文走 JSON body：一条经验动辄几百字，走 query 会被代理截断，
    症状是「保存成功但后半段丢了」。
    """
    outcome = knowledge_ledger.upsert(
        kind=str(payload.get("kind") or "").strip(),
        scope=str(payload.get("scope") or "global").strip(),
        title=str(payload.get("title") or "").strip(),
        body=str(payload.get("body") or "").strip(),
        evidence=payload.get("evidence"),
        tags=str(payload.get("tags") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        status=str(payload.get("status") or "proposed").strip(),
        source_kind="human",
        source_ref=str(payload.get("sourceRef") or ""),
        operator=str(payload.get("operator") or payload.get("by") or ""),
        role=str(payload.get("role") or ""),
    )
    if not outcome.get("ok"):
        return _write_error(outcome)
    return success(outcome)


@router.post("/entries/{entry_id}/status")
def post_entry_status(entry_id: str, payload: dict[str, Any]) -> Any:
    """改状态（人审：采纳／驳回／废弃／复审）。非法迁移明确 400，不静默改。"""
    status = str(payload.get("status") or "").strip()
    if not status:
        return bad_request("缺少 status")
    role = str(payload.get("role") or "")
    gate = _role_gate(role, "editor")
    if gate is not None:
        return gate
    outcome = knowledge_ledger.set_status(
        entry_id, status, reason=str(payload.get("reason") or ""),
        operator=str(payload.get("operator") or payload.get("by") or ""), role=role)
    if not outcome.get("ok"):
        return _write_error(outcome)
    return success(outcome)


# ── 管理面：编辑 / 删除 / 恢复 / 版本 / 标签 / 分类 / 审计 ────────────────────
#
# 与上面 POST /entries 的分工必须写清（否则两条路会互相打架）：
#   * ``POST /entries``      ＝「又有一份证据支持它」⇒ support+1，正文通常不变；
#   * ``PATCH /entries/{id}`` ＝「这条写错了」⇒ 正文变、support 不动、version+1。
# 把编辑做成叠加，会让「改了三次错别字」看起来像「有三份独立证据」，那是灌水。
#
# 错误码分派**按服务层的机器码**（``result["code"]``），不在中文文案里找关键字：
# 文案改一个字就换 HTTP 码的判据，是最难查的那类回归。
# 完整契约见 ``docs/knowledge-base-contract.md``。

#: 角色等级。``ROLES`` 的顺序即权限序（viewer < editor < owner）。
_ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}

#: 机器码 → HTTP 状态。未列出的按 400（参数/业务判据不成立）。
_CODE_STATUS = {
    "not_found": 404,
    "conflict": 409,
    "deleted": 409,      # 「与当前状态冲突」：先 restore 再改，不是参数写错
    "not_deleted": 409,  # 硬删前置条件不满足
    "evidence": 422,     # 格式都对、内容站不住（补证据，不是改请求形状）
    "unavailable": 500,  # 库不可写：这是服务端的事，不该让前端去改参数
}


def _write_error(result: dict[str, Any]) -> Any:
    """服务层 ``ok=False`` → 与机器码一致的 HTTP 响应。**必须带 why 原文**。"""
    why = str(result.get("why") or "写入被拒")
    status = _CODE_STATUS.get(str(result.get("code") or ""), 400)
    if status == 404:
        return not_found(why)
    if status == 409:
        return conflict(why)
    if status == 422:
        return unprocessable(why)
    if status == 500:
        return server_error(why)
    return bad_request(why)


def _role_gate(role: str, need: str) -> Any:
    """声明式角色闸门，返回 ``None`` 表示放行。

    ⚠️ 这**不是认证**：本仓没有 Authorization 中间件，``role`` 由调用方自己在请求里声明，
    谁都能写 ``owner``。它解决的是「审计行里的角色与被执行的动作自相矛盾」这件事，
    不解决「有人冒充」。要真鉴权得先有登录态（见契约文档 §4 的「未做」清单）。

    ``role`` 为空＝**未声明** ⇒ 不判（向后兼容旧前端与 Agent 工具）；
    声明了就必须合法且够格，否则 400 / 403 —— 宁可拒，也不"猜你想干什么"。
    """
    name = str(role or "").strip()
    if not name:
        return None
    if name not in knowledge_ledger.ROLES:
        return bad_request(f"未知角色 {name!r}（可选：{'/'.join(knowledge_ledger.ROLES)}）")
    if _ROLE_RANK[name] < _ROLE_RANK[need]:
        return forbidden(f"角色 {name} 不能做这件事（需要 {need} 及以上）")
    return None


def _opt_int(value: Any, name: str) -> tuple[int | None, Any]:
    """可选的整数参数 → ``(值, 错误响应)``。空/缺省给 ``None``，非数字给 400。"""
    if value is None or value == "":
        return None, None
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, bad_request(f"{name} 必须是整数")


@router.patch("/entries/{entry_id}")
def patch_entry(entry_id: str, payload: dict[str, Any] | None = None) -> Any:
    """编辑一条知识（**部分字段**，未出现的字段不动）。

    只提供 PATCH、**有意不提供 PUT**：PUT 的语义是「整条替换」，一个忘了带 ``body``
    的调用就会把正文清空 —— 这是知识库里最贵的一类事故（正文是几百字的经验）。
    ``null`` 与「不传」同义（都不改），要清空标签请显式传 ``"tags": ""``。

    三种失败要分开：404（不在了）／409（版本冲突或已删除）／422（证据核不过）。
    """
    body = payload or {}
    role = str(body.get("role") or "")
    gate = _role_gate(role, "editor")
    if gate is not None:
        return gate
    confidence: float | None = None
    if body.get("confidence") is not None:
        try:
            confidence = float(body["confidence"])
        except (TypeError, ValueError):
            return bad_request("confidence 必须是数字")
    expected, error = _opt_int(body.get("expectedVersion"), "expectedVersion")
    if error is not None:
        return error
    result = knowledge_ledger.update_entry(
        entry_id,
        title=body.get("title"),
        body=body.get("body"),
        tags=body.get("tags"),
        evidence=body.get("evidence"),
        confidence=confidence,
        reason=str(body.get("reason") or ""),
        expected_version=expected,
        operator=str(body.get("operator") or body.get("by") or ""),
        role=role,
    )
    if not result.get("ok"):
        return _write_error(result)
    return success(result)


@router.delete("/entries/{entry_id}")
def delete_entry(
    entry_id: str,
    hard: bool = Query(False, description="true＝物理删除（**必须先软删**，这是防手滑的第二道闸）"),
    reason: str = Query("", description="删除原因（进审计 detail）"),
    operator: str = Query("", description="操作者（进审计与被删行）"),
    role: str = Query("", description="声明式角色；hard=true 需要 owner"),
) -> Any:
    """删除：默认软删（可恢复），``?hard=true`` 才是物理删除。

    硬删的后置事实：``entries``/``entry_links``/``entry_usage``/``entry_versions``/
    FTS 全清，**``entry_audit`` 保留** —— 「谁在什么时候删了什么」不能跟着内容一起蒸发。
    """
    gate = _role_gate(role, "owner" if hard else "editor")
    if gate is not None:
        return gate
    if hard:
        result = knowledge_ledger.purge(entry_id, operator=operator, role=role)
    else:
        result = knowledge_ledger.soft_delete(entry_id, reason=reason,
                                              operator=operator, role=role)
    if not result.get("ok"):
        return _write_error(result)
    return success(result)


@router.post("/entries/{entry_id}/restore")
def post_entry_restore(entry_id: str, payload: dict[str, Any] | None = None) -> Any:
    """撤销软删（幂等：本来没删就回 ``changed=false``，不报错）。"""
    body = payload or {}
    role = str(body.get("role") or "")
    gate = _role_gate(role, "editor")
    if gate is not None:
        return gate
    result = knowledge_ledger.restore(
        entry_id, reason=str(body.get("reason") or ""),
        operator=str(body.get("operator") or body.get("by") or ""), role=role)
    if not result.get("ok"):
        return _write_error(result)
    return success(result)


@router.get("/entries/{entry_id}/versions")
def get_entry_versions(entry_id: str) -> Any:
    """版本列表（只有元信息，**不含正文**）。

    ``current`` 是当前版本号，``items`` 是 1..current-1 的历史 —— 当前值永远在
    ``GET /entries/{id}`` 里。要正文请按版本号取 ``/versions/{v}``（含当前版）。
    """
    result = knowledge_ledger.versions(entry_id)
    if not result.get("available"):
        return success(result)  # 读失败降级：如实回 available=false + hint
    if not result.get("current"):
        return not_found(f"没有这条知识：{entry_id}")
    return success(result)


@router.get("/entries/{entry_id}/versions/{version}")
def get_entry_version(entry_id: str, version: int) -> Any:
    """某个版本的全文（``isCurrent=true`` 表示它就是当前值）。"""
    item = knowledge_ledger.version_detail(entry_id, version)
    if item is None:
        return not_found(f"{entry_id} 没有 v{version}")
    return success(item)


@router.post("/entries/{entry_id}/revert")
def post_entry_revert(entry_id: str, payload: dict[str, Any] | None = None) -> Any:
    """回滚到某个历史版本：**是一次普通编辑**（version+1 + 快照 + 审计），不删后续版本。"""
    body = payload or {}
    role = str(body.get("role") or "")
    gate = _role_gate(role, "editor")
    if gate is not None:
        return gate
    version, error = _opt_int(body.get("version"), "version")
    if error is not None:
        return error
    if version is None:
        return bad_request("缺 version（要回滚到哪一版）")
    result = knowledge_ledger.revert(
        entry_id, version, reason=str(body.get("reason") or ""),
        operator=str(body.get("operator") or body.get("by") or ""), role=role)
    if not result.get("ok"):
        return _write_error(result)
    return success(result)


@router.get("/tags/facets")
def get_tag_facets(
    include_deleted: bool = Query(False, alias="includeDeleted"),
    limit: int = Query(200, ge=1, le=1000),
) -> Any:
    """标签 facet（每个标签命中多少条，按条数倒序）。

    计数与列表用**同一套词边界匹配**（``',' || tags || ','``）：否则页面显示「节奏 5 条」，
    点进去却因为 `节奏慢` 不算命中而只剩 3 条 —— 这类对不上的账会被当成筛选坏了。
    """
    return success(knowledge_ledger.tag_facets(
        include_deleted=include_deleted, limit=limit))


@router.get("/taxonomy")
def get_taxonomy(
    include_deleted: bool = Query(False, alias="includeDeleted"),
    role: str = Query(""),
) -> Any:
    """分类导航：``kind``（闭集＋中文名＋计数）／``scope``／``status``／标签。

    这是「分类管理」的**唯一入口** —— 前端不该自己写死中文名，否则服务端加一类，
    前端就多出一串英文 id。
    """
    gate = _role_gate(role, "viewer")
    if gate is not None:
        return gate
    return success(knowledge_ledger.taxonomy(include_deleted=include_deleted))


@router.get("/audit")
def get_audit(
    entry: str | None = Query(None, alias="entryId", description="只看这一条的审计"),
    action: str | None = Query(None, description="/".join(knowledge_ledger.ACTIONS)),
    operator: str | None = Query(None),
    role: str = Query(""),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Any:
    """操作审计（倒序）。``total`` 是**全量条数**，不是本页条数。"""
    gate = _role_gate(role, "viewer")
    if gate is not None:
        return gate
    if action and action not in knowledge_ledger.ACTIONS:
        return bad_request(
            f"未知动作 {action!r}（可选：{'/'.join(knowledge_ledger.ACTIONS)}）")
    return success(knowledge_ledger.audit_log(
        entry=entry, action=action, operator=operator, limit=limit, offset=offset))


@router.get("/search")
def get_search(
    q: str = Query("", description="检索词；≥3 字走 bm25，1–2 字回落 LIKE"),
    kind: str | None = Query(None),
    scope: str | None = Query(None),
    status: str | None = Query(None),
    tags: str | None = Query(None, description="逗号分隔，AND 语义"),
    include_deleted: bool = Query(False, alias="includeDeleted"),
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
) -> Any:
    """检索知识。``ranking`` 原样透出（三种排序不可比）。

    ``total`` 是**全量命中数**（不是本页条数）——分页 UI 靠它算总页数；
    候选集有 500 条上限，超了回 ``totalCapped=true`` 并说明「至少 500」。
    """
    return success(knowledge_ledger.search(
        q, kind=kind, scope=scope, status=status, tags=tags,
        include_deleted=include_deleted, limit=limit, offset=offset))


@router.post("/distill")
async def post_distill(payload: dict[str, Any],
                       conn: Connection = Depends(get_conn)) -> Any:
    """从跑次里蒸馏候选并写成 ``proposed`` 条目。

    ``useLlm=false``（默认）走确定性模板 —— 不花一次模型调用，离线也能跑；
    ``useLlm=true`` 时逐条 ``await generate_text``，单条失败只跳过那一条。
    """
    body = payload or {}
    limit = max(1, min(int(body.get("limit") or 3), 20))
    dry_run = bool(body.get("dryRun"))
    pool = knowledge_distill.candidates(since_ns=_since_ns(body.get("sinceDays")))[:limit]
    if not pool:
        return success({"candidates": 0, "written": 0, "rejected": 0,
                        "items": [], "hint": "没有达到阈值的统计信号（样本不足或确实没有异常）"})

    if not body.get("useLlm"):
        return success(knowledge_distill.distill(
            candidates_list=pool, limit=limit, dry_run=dry_run))

    from ..services.text_generation import generate_text

    results: list[dict[str, Any]] = []
    for candidate in pool:
        try:
            answer = await generate_text(
                conn, knowledge_distill.prompt_for(candidate),
                {"temperature": 0.2, "maxTokens": 500})
        except Exception as exc:  # noqa: BLE001 - 模型不可用只影响这一条
            results.append({"key": candidate.get("key"), "ok": False,
                            "why": f"模型不可用：{exc}"})
            continue
        results.append(knowledge_distill.distill(
            candidates_list=[candidate], limit=1, dry_run=dry_run,
            llm=lambda _prompt, text=answer: text))
    return success({"mode": "llm", "candidates": len(pool), "results": results})


@router.post("/lift/refresh")
def post_lift_refresh(payload: dict[str, Any] | None = None) -> Any:
    """重算 accepted 条目的 ``lift``（注入后成功率 − 同期基线），负资产自动降级。

    样本不足的写 ``lift=None``（"还没结论"），**不写 0**（0 会被读成"没影响"）。
    """
    body = payload or {}
    days = body.get("windowDays") or 30
    try:
        window_ns = int(float(days) * _DAY_NS)
    except (TypeError, ValueError):
        return bad_request("windowDays 必须是数字")
    return success(knowledge_distill.refresh_lift(
        window_ns=window_ns,
        min_settled=int(body.get("minSettled") or 10),
        deprecate_below=float(body.get("deprecateBelow") or -0.05),
    ))


@router.get("/scheduler")
def get_scheduler() -> Any:
    """定时复盘的状态（``enabled``/``running``/``lastRunAt``/``nextRunAt``/``lastResult``）。

    ⚠️ 这里**只读**：定时器是进程级后台任务，跑没跑起来由 ``main.lifespan`` 决定。
    手动跑一轮请用 ``POST /distill``（那条路径可以带 ``useLlm``，且会当场回结果）——
    两处都调同一个 ``knowledge_distill``，不存在"预览版"与"真跑版"。
    """
    return success(knowledge_scheduler.state())


@router.get("/feed/preview")
def get_feed_preview(
    agent_type: str | None = Query(None, alias="agentType"),
    episode_id: int | None = Query(None, alias="episodeId"),
    drama_id: int | None = Query(None, alias="dramaId"),
    scopes: str | None = Query(None, description="额外作用域，逗号分隔，如 stage:krea2,shot:SHOT-012"),
    budget_chars: int = Query(knowledge_feed.DEFAULT_BUDGET, alias="budgetChars",
                              ge=0, le=8000),
    limit: int = Query(8, ge=1, le=30),
) -> Any:
    """L1 编译预览：将被塞进 system prompt「## Knowledge」段的那几行。

    ``entries`` 是真正被写进去的 id；生产侧拿它去 ``record_injection`` 记 usage，
    没有这一步 ``lift`` 永远算不出来。

    ⚠️ 预览必须能**按集预览**（``episodeId``），否则这里看不到真实注入：
    生产侧由 ``assemble_full_instructions`` 传 ``EP<id>``/``DRAMA<id>``，
    不带这两个参数预览出来的是「只剩全局成法」的另一种结果，两者差别恰好就是
    此前那个静默漏掉的那批条目。``blocked`` 才是「有知识但进不来」的那个数。
    """
    extra = [x.strip() for x in str(scopes or "").split(",") if x.strip()]
    extra += knowledge_feed.scopes_for(episode_id=episode_id, drama_id=drama_id)
    return success(knowledge_feed.compile_for_prompt(
        agent_type, scopes=extra, budget_chars=budget_chars, limit=limit))


@router.post("/feed/export")
def post_feed_export(payload: dict[str, Any] | None = None) -> Any:
    """L3 导出数据集（``items.jsonl`` + ``manifest.json``）。

    ``target=runs`` 导高分跑次（需要 meta 里有提示词，缺的跳过并计数）；
    ``target=knowledge`` 导已采纳知识的「指令 → 做法」对。
    """
    body = payload or {}
    target = str(body.get("target") or "knowledge").strip()
    out_dir = body.get("outDir") or None
    if target == "runs":
        return success(knowledge_feed.export_runs_dataset(
            out_dir,
            kind=body.get("kind") or "video",
            stage=body.get("stage") or None,
            min_score=body.get("minScore"),
            max_score=body.get("maxScore"),
            since_ns=_since_ns(body.get("sinceDays")),
            limit=int(body.get("limit") or 500),
            include_low=bool(body.get("includeLow")),
        ))
    if target == "knowledge":
        return success(knowledge_feed.export_knowledge_dataset(
            out_dir, limit=int(body.get("limit") or 500)))
    return bad_request("target 只能是 runs / knowledge")


@router.get("/datasets")
def get_datasets(limit: int = Query(50, ge=1, le=200)) -> Any:
    """已经导出的数据集（= 提交训练时 ``datasetRef`` 的**可选值**）。

    只读 manifest 摘要、不逐条核样本：列举要轻，真核在提交训练那一下
    （``knowledge_feed.resolve_dataset``）。``itemsExists=false`` 的行照样列出来 ——
    导出集被清掉这件事本身要可见，藏起来只会让下一次提交撞上一个说不清的 400。
    """
    return success(knowledge_feed.list_datasets(limit=limit))


@router.get("/datasets/resolve")
def get_dataset_resolve(ref: str = Query(..., description="数据集目录 / manifest.json / 数据集名")) -> Any:
    """核一个 ``datasetRef``（与训练提交口**同一个判据**）。

    前端「先看一眼这批数据能不能用」不该靠猜：核不过就带 ``why`` 回 400，
    而不是等到点开跑才在训练入口吃一个 400。
    """
    result = knowledge_feed.resolve_dataset(ref)
    return success(result) if result.get("ok") else bad_request(str(result.get("why") or "核不过"))


@router.get("/report")
def get_report(
    include_candidates: bool = Query(False, alias="includeCandidates",
                                     description="是否顺带算一次蒸馏候选（会扫最近 5000 条跑次）"),
    since_days: float | None = Query(None, alias="sinceDays"),
) -> Any:
    """闭环总览：事实层多大、台账各状态多少、下一批候选是什么。"""
    payload: dict[str, Any] = {
        "runs": outcome_index.stats(group_by="stage", since_ns=_since_ns(since_days)),
        "ledger": knowledge_ledger.stats(),
        "runsTotal": outcome_index.count(),
    }
    if include_candidates:
        payload["candidates"] = knowledge_distill.candidates(
            since_ns=_since_ns(since_days))[:10]
    return success(payload)


# ── 优化史（P1 续 / 缺口 3）：每一轮优化到底提升了多少 ──────────────────────
#
# 起因是一条回答不出来的问题：「这次改动提升了多少」。优化器的分数只落在
# ``benchmarks/history/<时间戳>.json``（同名 case 重跑即覆盖 ✗），而生效提示词落在
# ``agent_configs``（覆盖式，上一次生效的分数当场就没了 ✗）—— 两边都不成对。
# 所以这张表**只读不写**：写入只发生在优化器内部（``optimize_agent_prompt``），
# 不允许 HTTP 手写一行 —— 手写的"提升了多少"就是编的（也没法核）。

@router.get("/optimizations")
def get_optimizations(
    agent_type: str | None = Query(None, alias="agentType"),
    case_id: str | None = Query(None, alias="caseId"),
    limit: int = Query(100, ge=1, le=optimization_log.MAX_LIMIT),
) -> Any:
    """优化史（倒序）＋ 汇总：跑过几轮、采纳几次、落库几次、最好的一次涨了多少。

    三个数必须分开看 ✓：``rounds``（试了几次）／``accepted``（分数更高的几次）／
    ``persisted``（真的变成默认值的几次）—— 合成一个「成功率」会把
    「试了 5 轮只成 1 轮」与「成了 1 轮但没落库」糊成同一个数字 ✗。
    """
    result = optimization_log.list_records(agent_type=agent_type, case_id=case_id, limit=limit)
    return success(result)


# ── 引擎自我完善（P3）：参数覆盖层 ──────────────────────────────────────────

@router.get("/tuning")
def get_tuning() -> Any:
    """当前有效参数与来源（``configs/engine/tuning.json``）。

    每行都带 ``source``：``default`` 或 ``override:<stage>.<field>``。
    ``problems`` 非空说明「文件里写了但没人会读」或「改了行为却缺凭据」。
    """
    return success(tuning.report())


@router.get("/tuning/suggest")
def get_tuning_suggest(
    min_samples: int = Query(tuning.MIN_SAMPLES, alias="minSamples", ge=1, le=200,
                             description="样本量低于它就只报不应用（默认 30）"),
) -> Any:
    """调参**候选**（只读）：从跑次统计里找「失败聚集」与「步数已到平台期」。

    ⚠️ 这个接口**不改任何东西** —— 候选的 ``canApply`` 恒为 ``false``（影子验证还没做），
    人核过之后才用 ``POST /tuning`` 落盘（那一步要 basis + 可核 evidence）。
    回执里的 ``notEvaluated`` 是如实列出「算不了的维度」：没检查 ≠ 检查过没问题。
    """
    return success(tuning.suggest(min_samples=min_samples))


@router.post("/tuning")
def post_tuning(payload: dict[str, Any] | None = None) -> Any:
    """写一条覆盖（写前自动备份；证据核不过整条拒）。

    必须给 ``basis``（为什么改）与 ``evidence``（真实 run_id / 盘上文件）——
    没有凭据的调优就是拍脑袋，会让下一次出图错误无从归因。
    """
    body = payload or {}
    result = tuning.set_field(
        str(body.get("stage") or ""), str(body.get("field") or "steps"),
        body.get("value"), basis=str(body.get("basis") or ""),
        evidence=body.get("evidence"), by=str(body.get("by") or "human"))
    return success(result) if result.get("ok") else bad_request(
        str(result.get("why") or "写入被拒"))


@router.delete("/tuning")
def delete_tuning(
    stage: str = Query("", alias="stage"),
    field: str = Query("steps", alias="field"),
    reset_all: bool = Query(False, alias="all",
                            description="为真时清空全部覆盖（回到纯默认）"),
) -> Any:
    """删一条覆盖，或 ``?all=1`` 全清。两条路径都先备份。"""
    if reset_all:
        return success(tuning.reset())
    result = tuning.clear_field(stage, field)
    return success(result) if result.get("ok") else bad_request(
        str(result.get("why") or "没有这条覆盖"))


# ── 引擎自我完善（P3 续）：质检阈值的**人工复核**与命中率 ────────────────────
#
# 这一节补的是 ``notEvaluated`` 里那条 ``qc-threshold-hit-rate``：光有「触发频次」算不出
# 命中率（那是分子），分母必须是**人看过成片之后的结论**。所以这里有写入口 ✓ ——
# 与整条闭环其它端点「只读候选」的口径不同：复核结论是**事实**（人看到的东西），
# 由人录；而由事实推出的参数改动仍然只能走 ``POST /tuning``（要 basis + evidence）✓。

@router.get("/qc/hit-rate")
def get_qc_hit_rate() -> Any:
    """按 issue ``code`` 的命中率 / 漏报率（分母＝人工复核条数）。

    ``available=false`` 与「复核 0 条」严格分开 ✓：前者是表读不出来，后者是"还没人核"——
    两者都会让 ``hitRate`` 为 ``null``，但一个要修库、一个要安排人看片 ✗。
    """
    return success(qc_reviews.hit_rate())


@router.get("/qc/pending")
def get_qc_pending(
    limit: int = Query(20, ge=1, le=200),
    conn: Connection = Depends(get_conn),
) -> Any:
    """**待复核**清单：最近几条 QC 记录 ＋ 每条里每个 code 的当前判定（未判为 ``null``）。

    这条端点是"命中率能不能有分母"的入口 ✓：复核表建好了但没人知道该核哪一条的话，
    命中率会永远停在 ``null`` ✗ —— 所以主动把最近报了缺陷的片子摆出来。
    """
    items = qc_reviews.pending(conn, limit=limit)
    return success({
        "items": items,
        "unreviewed": len([item for item in items if not item.get("reviewed")]),
        "verdicts": qc_reviews.VERDICTS,
        "minReviews": qc_reviews.MIN_REVIEWS,
    })


@router.get("/qc/reviews")
def get_qc_reviews(
    limit: int = Query(50, ge=1, le=500),
    check_id: int | None = Query(None, alias="checkId"),
    code: str | None = Query(None),
    verdict: str | None = Query(None, description="hit / false_alarm / missed"),
    conn: Connection = Depends(get_conn),
) -> Any:
    """已录的复核结论（按更新时间倒序）。"""
    return success({
        "items": qc_reviews.list_reviews(conn, limit=limit, check_id=check_id,
                                        code=code, verdict=verdict),
        "verdicts": qc_reviews.VERDICTS,
    })


@router.post("/qc/reviews")
def post_qc_review(
    payload: dict[str, Any] | None = None,
    conn: Connection = Depends(get_tx),
) -> Any:
    """录一条复核结论：``checkId`` + ``code`` + ``verdict``（``hit``/``false_alarm``/``missed``）。

    ``(checkId, code)`` 已存在就**改判**（覆盖，不追加）✓ —— 同一个人看完放大帧改变主意是
    改判，不是第二条样本 ✗（否则命中率会被自己的犹豫稀释 ✓✗）。
    ``verdict`` 给错或 ``code`` 不在闭集里一律 400 并带 ``why`` ✓。
    """
    body = payload or {}
    try:
        check_id = int(body.get("checkId") or body.get("check_id") or 0)
    except (TypeError, ValueError):
        return bad_request("checkId 必须是整数")
    if not check_id:
        return bad_request("缺 checkId（要复核的是哪一条质检记录）")
    try:
        item = qc_reviews.record_review(
            conn, check_id=check_id, code=str(body.get("code") or ""),
            verdict=str(body.get("verdict") or ""), note=str(body.get("note") or ""),
            reviewer=str(body.get("reviewer") or body.get("by") or ""),
            storyboard_id=_int_or_none(body.get("storyboardId")),
            video_generation_id=_int_or_none(body.get("videoGenerationId")),
        )
    except ValueError as err:
        return bad_request(str(err))
    return success({"review": item, "hitRate": qc_reviews.hit_rate(conn)})


@router.delete("/qc/reviews/{review_id}")
def delete_qc_review(review_id: int, conn: Connection = Depends(get_tx)) -> Any:
    """撤销一条复核（打错码 / 选错片子）。"""
    if not qc_reviews.delete_review(conn, review_id):
        return not_found(f"没有 id={review_id} 的复核记录")
    return success({"deleted": review_id})


def _int_or_none(value: Any) -> int | None:
    """``"3"``/``3`` → ``3``；空/非数字 → ``None``（不编 0 ✓）。"""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── 模型投喂（P4）：训练任务与产物登记 ──────────────────────────────────────

@router.get("/training")
def get_training(
    limit: int = Query(20, ge=1, le=200),
    status: str | None = Query(None, description="产物状态：candidate / accepted / deprecated"),
) -> Any:
    """训练任务与训练产物（**重启后仍可查** —— 这是它存在的唯一理由）。

    ``problems`` 非空说明登记表在报自己的账不平：默认档的文件不见了、
    或某个任务永远停在 ``running``（进程死掉留下的僵尸）。
    """
    return success({
        "path": training_registry.db_path().as_posix(),
        "jobs": training_registry.list_jobs(limit=limit),
        "models": training_registry.list_models(status=status, limit=limit),
        "stats": training_registry.stats(),
        "problems": training_registry.problems(),
        "hint": "candidate 不会自动成为默认；要生效请 POST /training/models/{id}/promote 并给出依据与凭据",
    })


@router.post("/training/scan")
def post_training_scan(payload: dict[str, Any] | None = None) -> Any:
    """按 taskId 重扫产物目录（训练跑完时已自动扫过一次，这里是手工补扫口）。"""
    body = payload or {}
    task_id = str(body.get("taskId") or "").strip()
    if not task_id:
        return bad_request("缺 taskId")
    job = training_registry.get_job(task_id)
    if job is None:
        return not_found(f"没有这个训练任务：{task_id}")
    found = training_registry.scan_artifacts(job)
    return success({
        "taskId": task_id, "found": found, "count": len(found),
        "hint": "扫到 0 个不等于出错：请确认 values.outputDir 下的 *.safetensors 是否已生成",
    })


@router.post("/training/models/{model_id}/promote")
def post_training_promote(model_id: str,
                          payload: dict[str, Any] | None = None) -> Any:
    """把产物定为该类的默认档（**唯一能改变出图行为的入口**）。

    必须给 ``basis``（为什么）与 ``evidence``（真实 run_id / 盘上文件）——
    没有凭据的"我觉得这个好"会让下次风格变化无从归因。
    """
    body = payload or {}
    result = training_registry.promote(
        model_id, basis=str(body.get("basis") or ""),
        evidence=body.get("evidence"), by=str(body.get("by") or "human"))
    return success(result) if result.get("ok") else bad_request(
        str(result.get("why") or "提升被拒"))


@router.post("/training/models/{model_id}/demote")
def post_training_demote(model_id: str,
                         payload: dict[str, Any] | None = None) -> Any:
    """撤掉默认档（文件不删、记录不删 —— 「不用它」不等于「它没存在过」）。"""
    body = payload or {}
    result = training_registry.demote(model_id, why=str(body.get("why") or ""))
    return success(result) if result.get("ok") else bad_request(
        str(result.get("why") or "撤销失败"))
