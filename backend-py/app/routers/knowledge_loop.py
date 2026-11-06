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

from ..core.db import get_conn
from ..core.response import bad_request, not_found, success
from ..services import (
    knowledge_distill,
    knowledge_feed,
    knowledge_ledger,
    knowledge_scheduler,
    outcome_index,
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
    scope: str | None = Query(None, description="global / drama:<id> / stage:<name> / agent:<type>"),
    status: str | None = Query(None, description="proposed / accepted / deprecated / rejected"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Any:
    """条目清单（分页）。**默认不筛状态** ⇒ 提案与已采纳混在一起，前端必须显式带上。"""
    return success(knowledge_ledger.list_entries(
        kind=kind, scope=scope, status=status, limit=limit, offset=offset))


@router.get("/entries/{entry_id}")
def get_entry(entry_id: str) -> Any:
    """条目详情 ＋ 挂载关系 ＋ 被注入过的记录。"""
    item = knowledge_ledger.detail(entry_id)
    if item is None:
        return not_found(f"没有这条知识：{entry_id}")
    return success(item)


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
    )
    if not outcome.get("ok"):
        return bad_request(outcome.get("why") or "写入失败")
    return success(outcome)


@router.post("/entries/{entry_id}/status")
def post_entry_status(entry_id: str, payload: dict[str, Any]) -> Any:
    """改状态（人审：采纳／驳回／废弃／复审）。非法迁移明确 400，不静默改。"""
    status = str(payload.get("status") or "").strip()
    if not status:
        return bad_request("缺少 status")
    outcome = knowledge_ledger.set_status(entry_id, status,
                                         reason=str(payload.get("reason") or ""))
    if not outcome.get("ok"):
        return bad_request(outcome.get("why") or "状态未变")
    return success(outcome)


@router.get("/search")
def get_search(
    q: str = Query("", description="检索词；≥3 字走 bm25，1–2 字回落 LIKE"),
    kind: str | None = Query(None),
    scope: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
) -> Any:
    """检索知识。``ranking`` 原样透出（三种排序不可比）。"""
    return success(knowledge_ledger.search(
        q, kind=kind, scope=scope, status=status, limit=limit))


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
    budget_chars: int = Query(knowledge_feed.DEFAULT_BUDGET, alias="budgetChars",
                              ge=0, le=8000),
    limit: int = Query(8, ge=1, le=30),
) -> Any:
    """L1 编译预览：将被塞进 system prompt「## Knowledge」段的那几行。

    ``entries`` 是真正被写进去的 id；生产侧拿它去 ``record_injection`` 记 usage，
    没有这一步 ``lift`` 永远算不出来。
    """
    return success(knowledge_feed.compile_for_prompt(
        agent_type, budget_chars=budget_chars, limit=limit))


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
