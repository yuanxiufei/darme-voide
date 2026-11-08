"""知识闭环守卫（2026-11-06 立）—— 事实层 / 台账 / 蒸馏 / 投喂 / 训练登记 / 数据集引用。

为什么要有这一套
================
这四个模块的失效方式**全都不会报错**，只会安静地给出一个好看的数字 ——
这正是本仓最需要机器判据的那类缺陷：

* 索引重建不幂等 ⇒ 每天重扫一遍，``runs`` 里同一份 trace 出现两次，失败率凭空减半；
* ``dryRun`` 照写库 ⇒ 预览里"会写 3 条"、真跑却写了 6 条，谁也不知道差在哪；
* 2 字中文在 trigram 下**零命中且不报错** ⇒ 用户搜「过曝」得 0 条，结论是"库里没这条经验"；
* 证据核验退化成橡皮图章 ⇒ Agent 编一个 ``run_id`` 就能把自己的猜测写成"经验"，
  然后被 L1 注进下一次的 system prompt —— 幻觉自我强化。

八组判据
========
① 事实层：重建幂等、指纹增量、``dryRun`` 不落库、状态归一、``total``/``items`` 分离；
② 台账：证据核不过整条拒、``support`` 叠加、状态机单向、三种 ``ranking`` 如实透出；
③ 蒸馏：阈值下的候选、``dryRun`` 不写、模板路径可离线跑、无证据候选显式跳过；
④ 投喂：只取 accepted、预算截断、``record_injection`` 可算 lift、导出带 ``rights``、
   低分样本不进训练集；
⑤ 接线：端点挂上、Agent 工具**写不进 accepted**、``sinceDays`` 非数字不猜成 0；
⑥ 参数覆盖（P3）：无 ``tuning.json`` 时零改动、显式值不被覆盖、``plan`` 里标来源；
⑦ 训练登记（P4）：候选不自动升、``promote`` 四门全拒、重启后仍可查、僵尸与丢档点名；
⑧ 数据集引用（P4 的另一半）：导出集**可核**（指纹·条数·角色·权利）、被手改/缺 rights/
   空集/地址不存在**全拒**、核过的来路随任务与产物落库、样本被清 ⇒ ``problems`` 点名。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/knowledge_loop_test.py
"""
from __future__ import annotations

import asyncio
import ast
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"
sys.path.insert(0, str(BP))
os.environ.setdefault("PROXY_TO_NODE", "0")

#: 夹具库与真实库**必须分开**：这套会大量写库，跑在真数据根上就是把测试的假经验
#: 注进创作者的 system prompt（L1 只认 accepted，但 proposed 的堆积同样污染检索）。
TMP = Path(tempfile.mkdtemp(prefix="kb-loop-"))
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome-index.sqlite").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "knowledge-ledger.sqlite").as_posix()

from app.services import (  # noqa: E402
    knowledge_distill,
    knowledge_feed,
    knowledge_ledger,
    outcome_index,
    trace_store,
)

TRACES = TMP / "traces"
#: 两个模块都各自 import 了一份 ``get_traces_dir`` ⇒ 两处都要换，只换一处等于没换
outcome_index.get_traces_dir = lambda: TRACES.as_posix()
trace_store.get_traces_dir = lambda: TRACES.as_posix()
#: ``check_evidence`` 会拿数据根拼一次路径，同样换成夹具目录（防真数据根上有同名文件）
knowledge_ledger.get_data_root = lambda: TMP.as_posix()

from app.services.engine import hooks  # noqa: E402

#: ⚠️ 2026-11-07：引擎读业务数据已改成**注入口**（`hooks.get("ledger"/"run_index")` ✓，
#: 生产由 `app/main.py` 启动时登记 ✓）—— 本套必须补同一步 ✗：否则 `tuning.set_field()`
#: 会因「知识台账没接入」**全部拒写**（护栏三的可见症状 ✓），覆盖层那段随后断在
#: 「tuning.json 不存在」而整套崩（不是 FAIL，更容易被当成环境问题 ✓✗）。
hooks.provide(ledger=knowledge_ledger, run_index=outcome_index)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write_trace(name: str, records: list[dict], scope: str = "agent") -> None:
    directory = TRACES / scope
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")


def _run(run_id: str, **fields) -> bool:
    payload = {"run_id": run_id, "kind": "image", "status": "ok", **fields}
    return outcome_index.record_run(payload)


def main() -> int:
    # ================================================================== ① 事实层
    empty = outcome_index.rebuild()
    check("事实层: 空目录重建 → 扫描 0、写入 0、可用",
          empty["scanned"] == 0 and empty["written"] == 0 and empty["failed"] == 0, empty)

    _write_trace("t1", [
        {"ts": "2026-11-05T10:00:00Z", "level": "INFO", "action": "START script",
         "meta": {"dramaId": 1, "episodeId": 2, "agentType": "script",
                  "model": "probe-model", "stage": "agent", "stage_kind": "probe"}},
        {"ts": "2026-11-05T10:00:05Z", "level": "ERROR", "action": "LLM_TIMEOUT"},
        {"ts": "2026-11-05T10:00:09Z", "level": "SUCCESS", "action": "DONE ok",
         "meta": {"totalMs": 9000}},
    ])
    _write_trace("t2", [
        {"ts": "2026-11-05T11:00:00Z", "level": "INFO", "action": "START script",
         "meta": {"dramaId": 1, "episodeId": 2, "agentType": "script",
                  "model": "probe-model", "stage": "agent", "score": 0.9,
                  "cost": 0.02, "prompt": "一个雨夜的巷口"}},
        {"ts": "2026-11-05T11:00:05Z", "level": "ERROR", "action": "MODEL_OOM"},
    ])
    first = outcome_index.rebuild()
    check("事实层: 首轮重建扫到 2 份、写入 2 条",
          first["scanned"] == 2 and first["written"] == 2 and first["failed"] == 0, first)

    check("事实层: ERROR 之后有 DONE ⇒ ok（恢复跑次不算失败）",
          (outcome_index.query(limit=50)["items"]
           and any(r["run_id"] == "trace:agent/t1" and r["status"] == "ok"
                   for r in outcome_index.query(limit=50)["items"])))
    check("事实层: ERROR 之后没有 DONE ⇒ failed，且带 error_code",
          any(r["run_id"] == "trace:agent/t2" and r["status"] == "failed"
              and r["error_code"] == "MODEL_OOM"
              for r in outcome_index.query(limit=50)["items"]))

    second = outcome_index.rebuild()
    check("事实层: ⭐ 二次重建幂等（指纹跳过，不重写）",
          second["written"] == 0 and second["skipped"] == 2 and second["changed"] == 0, second)
    check("事实层: 幂等后总条数不变（t1/t2 各一条，未重复插入）",
          outcome_index.count() == 2, outcome_index.count())

    _write_trace("t3", [
        {"ts": "2026-11-05T12:00:00Z", "level": "INFO", "action": "START script",
         "meta": {"stage": "agent"}},
        {"ts": "2026-11-05T12:00:01Z", "level": "SUCCESS", "action": "DONE ok"},
    ])
    dry = outcome_index.rebuild(dry_run=True)
    check("事实层: ⭐ dryRun 如实回 changed/written 但**不落库**",
          dry["changed"] == 1 and dry["written"] == 1 and dry["dryRun"] is True
          and not outcome_index.exists("trace:agent/t3"), dry)
    real = outcome_index.rebuild()
    check("事实层: dryRun 之后真重建仍能补上（指纹未被提前记账）",
          real["written"] == 1 and outcome_index.exists("trace:agent/t3"), real)

    check("事实层: record_run 拒收未知 kind / 缺 run_id / 空状态",
          outcome_index.record_run({"run_id": "r1", "kind": "nope", "status": "ok"}) is False
          and outcome_index.record_run({"kind": "image", "status": "ok"}) is False
          and outcome_index.record_run({"run_id": "r2", "kind": "image"}) is False)

    _run("cancelled-1", kind="agent", status="cancelled", stage="cancel-probe")
    stats = outcome_index.stats(group_by="stage")
    cancel_group = next((g for g in stats["groups"] if g["key"] == "cancel-probe"), None)
    check("事实层: cancelled 归 partial（不悄悄算成失败）",
          cancel_group is not None and cancel_group["partial"] == 1
          and cancel_group["failed"] == 0, cancel_group)

    agent_group = next((g for g in stats["groups"] if g["key"] == "agent"), None)
    check("事实层: 失败率分母含全部状态（ok 2 / failed 1 ⇒ 0.3333）",
          agent_group is not None and agent_group["failRate"] == round(1 / 3, 4), agent_group)
    check("事实层: avgScore 只按有分的算",
          agent_group is not None and agent_group["avgScore"] == 0.9, agent_group)

    page = outcome_index.query(limit=1)
    check("事实层: ⭐ total 与 items 分离（限制 1 条但总数照报）",
          len(page["items"]) == 1 and page["total"] == outcome_index.count() >= 4, page["total"])

    raised = False
    try:
        outcome_index.stats(group_by="body")
    except ValueError:
        raised = True
    check("事实层: groupBy 白名单外的列当场 ValueError（不拼进 SQL）", raised)

    # ================================================================== ② 台账
    no_ev = knowledge_ledger.upsert(kind="lesson", scope="EP1", title="无证据的经验",
                                    body="随便写")
    check("台账: ⭐ 非外部类型空证据整条拒（不是先存再标记可疑）",
          no_ev["ok"] is False and "必须带证据" in no_ev["why"], no_ev)

    fake = knowledge_ledger.upsert(kind="lesson", scope="EP1", title="编一条经验",
                                   body="随手写的", evidence=["trace:agent/ghost"])
    check("台账: ⭐ 假 run_id 整条拒，why 点名单号",
          fake["ok"] is False and "证据核不过" in fake["why"]
          and "trace:agent/ghost" in fake["why"], fake)

    fact = knowledge_ledger.upsert(kind="fact", scope="global", title="短剧单集时长惯例",
                                   body="竖屏短剧单集 60–120 秒是常态。", tags="事实")
    check("台账: fact/work 允许空证据（外部知识的证据是来源）", fact["ok"] is True, fact)

    good = knowledge_ledger.upsert(
        kind="lesson", scope="EP1", title="夜景 8 步会过曝",
        body="Krea2 8 步档在夜戏里高光溢出；改 12 步或补负向更稳。",
        evidence=["trace:agent/t2"], tags="夜景,过曝,krea2", confidence=0.7)
    check("台账: 真 run_id 可写，created=True 且 support=1",
          good["ok"] is True and good["created"] is True and good["support"] == 1, good)

    dup = knowledge_ledger.upsert(kind="lesson", scope="EP1", title="夜景 8 步会过曝",
                                 body="补一句：12 步比 8 步稳。", evidence=["trace:agent/t2"])
    check("台账: ⭐ 同一件事再提只累加 support（不新建）",
          dup["ok"] is True and dup["created"] is False and dup["support"] == 2, dup)
    check("台账: 二次写入不会把 accepted 打回 proposed（本条仍 proposed）",
          knowledge_ledger.detail(good["id"])["status"] == "proposed")

    other_scope = knowledge_ledger.upsert(kind="lesson", scope="EP2", title="夜景 8 步会过曝",
                                          body="另一集的同样结论。", evidence=["trace:agent/t2"])
    check("台账: 同标题不同作用域是两条（scope 进 id）",
          other_scope["id"] != good["id"], (other_scope["id"], good["id"]))

    moved = knowledge_ledger.set_status(good["id"], "accepted", reason="人审通过")
    check("台账: proposed → accepted 允许",
          moved["ok"] is True and moved["to"] == "accepted", moved)
    reverted = knowledge_ledger.set_status(good["id"], "proposed")
    check("台账: ⭐ accepted → proposed 被拒（状态机单向）",
          reverted["ok"] is False and "不允许" in reverted["why"], reverted)
    check("台账: 同状态幂等成功（不报错也不改）",
          knowledge_ledger.set_status(good["id"], "accepted")["ok"] is True)

    knowledge_ledger.upsert(kind="lesson", scope="global", title="过曝先降步数",
                            body="8 步档过曝时先回到 12 步，再考虑换模型。",
                            evidence=["trace:agent/t1"], status="accepted", confidence=0.6)

    short = knowledge_ledger.search("过曝")
    check("台账: ⭐ 2 字中文能搜到，且 ranking 如实为 like",
          short["total"] >= 1 and short["ranking"] == "like", short["ranking"])

    conn = knowledge_ledger._connect()
    has_fts = knowledge_ledger._has_fts(conn)
    conn.close()
    #: ⚠️ trigram 是**逐 3 字符切片**建索引的 ⇒ 查询必须是正文里的**连续子串**。
    #: 拿"夜景8步会过曝"去查"夜景 8 步会过曝"（带空格）会 0 命中 —— 不是 bug，
    #: 是分词口径；判据要挑一段真正连着的原文。
    long_q = knowledge_ledger.search("先降步数")
    check("台账: ≥3 字走 bm25（该 SQLite 编了 trigram 时）；否则如实退 like",
          long_q["total"] >= 1
          and long_q["ranking"] == ("bm25" if has_fts else "like"), long_q["ranking"])

    recent = knowledge_ledger.search("")
    check("台账: 空查询 ranking=recent（与相关性无关，必须写明）",
          recent["ranking"] == "recent" and recent["total"] >= 1, recent["ranking"])

    check("台账: LIKE 字面量转义（% _ \\ 都要挡，否则静默匹配到别的东西）",
          knowledge_ledger._like_literal("50%_a\\b") == "50\\%\\_a\\\\b",
          knowledge_ledger._like_literal("50%_a\\b"))

    empty_q = knowledge_ledger.search("绝不可能存在的词组xyzzy")
    check("台账: 冷查询 total=0 但 available=True（「没有」与「库坏」分开）",
          empty_q["available"] is True and empty_q["total"] == 0, empty_q)

    knowledge_ledger.set_lift(good["id"], None)
    check("台账: ★ lift=None 是「还没结论」，不是 0（0 会被读成没影响）",
          knowledge_ledger.detail(good["id"])["lift"] is None)
    knowledge_ledger.set_lift(good["id"], 0.12)
    check("台账: lift 写入可回读", knowledge_ledger.detail(good["id"])["lift"] == 0.12)

    knowledge_feed.record_injection([good["id"]], run_id="trace:agent/t1", outcome="ok")
    knowledge_feed.record_injection([good["id"]], run_id="trace:agent/t2", outcome="failed")
    usage = knowledge_ledger.usage_summary(good["id"])
    check("台账: record_usage 是 lift 的唯一数据源（settled=2 / okRate=0.5）",
          usage["settled"] == 2 and usage["okRate"] == 0.5, usage)

    # ================================================================== ③ 蒸馏
    for index in range(5):
        _run(f"probe-fail-{index}", kind="video", stage="distill-probe", status="failed",
             error_code="OOM", elapsed_ms=1000 + index)
    pool = [c for c in knowledge_distill.candidates() if c["key"] == "distill-probe"]
    check("蒸馏: 5 条同阶段失败 ⇒ 产出 failure_cluster 候选",
          bool(pool) and pool[0]["pattern"] == "failure_cluster", pool)
    check("蒸馏: 候选自带可核证据（_evidence_of 逐条查过索引）",
          bool(pool) and len(pool[0]["evidence"]) == 5, pool[0]["evidence"] if pool else None)

    before = knowledge_ledger.list_entries(limit=200)["total"]
    preview = knowledge_distill.distill(candidates_list=pool, limit=5, dry_run=True)
    after = knowledge_ledger.list_entries(limit=200)["total"]
    check("蒸馏: ⭐ dryRun 真的一条都没写库",
          preview["written"] == 0 and before == after
          and all("dry-run" in (item.get("why") or "") for item in preview["items"]), preview)

    written = knowledge_distill.distill(candidates_list=pool, limit=5)
    entry_id = next((item["id"] for item in written["items"] if item["ok"]), "")
    entry = knowledge_ledger.detail(entry_id) if entry_id else None
    check("蒸馏: 无 LLM 时走模板也能落库（proposed + sourceKind=distill-rule）",
          written["written"] == 1 and entry is not None
          and entry["status"] == "proposed" and entry["source_kind"] == "distill-rule", written)
    check("蒸馏: 模板条目 confidence 封顶 0.5（机器看出的现象还不是人的结论）",
          entry is not None and float(entry["confidence"]) <= 0.5, entry and entry["confidence"])

    no_evidence = knowledge_distill.distill(candidates_list=[{
        "pattern": "failure_cluster", "key": "ghost-stage", "kind": "failure",
        "score": 9.0, "stats": {"stage": "ghost-stage", "total": 9, "failed": 9,
                                "failRate": 1.0, "topErrors": []},
        "evidence": [],
    }], limit=1)
    check("蒸馏: 没有可核证据的候选显式跳过（不写空证据条目）",
          no_evidence["written"] == 0 and no_evidence["rejected"] == 1
          and "没有一条证据" in no_evidence["items"][0]["why"], no_evidence)

    # ============================================================ ③b 关联读侧（2026-11-08）
    # 这批判据盯的是「entry_links 只写不读」：蒸馏一直在挂 stage/shot，但没有任何
    # 反向查询 —— 于是"这条经验挂在哪个阶段上"这个**事实**等于没存。
    refs = knowledge_ledger.context_refs(ref_kind="stage")
    probe_ref = next((r for r in refs["items"] if r["refId"] == "distill-probe"), None)
    check("关联: ⭐ 蒸馏写的 stage ref 可反查（原先这张表只写不读）",
          refs["available"] is True and probe_ref is not None and probe_ref["entries"] >= 1,
          refs)

    scoped = knowledge_ledger.for_context(stage="distill-probe")
    check("关联: 按阶段反查拿到该阶段的经验（顺序按 confidence 降序）",
          scoped["total"] >= 1
          and any(item["entry"]["id"] == entry_id for item in scoped["items"])
          and all(item["refKind"] == "stage" for item in scoped["items"]), scoped)
    check("关联: 反查回执自带 refs，调用方能确认查的是哪个阶段",
          scoped["refs"] == [{"refKind": "stage", "refId": "distill-probe"}], scoped["refs"])

    blank = knowledge_ledger.for_context()
    check("关联: ⭐ stage/shot 都不给 ⇒ 空结果 + 人话（不把全库当上下文命中）",
          blank["available"] is True and blank["total"] == 0
          and blank["items"] == [] and "必须点名" in blank["hint"], blank)

    ghost = knowledge_ledger.for_context(stage="ghost-stage-xyzzy")
    check("关联: 没挂过条目的阶段 ⇒ total=0 且 hint 说明是没有而不是坏了",
          ghost["available"] is True and ghost["total"] == 0 and bool(ghost["hint"]), ghost)

    sibling = knowledge_ledger.upsert(
        kind="lesson", scope="EP1", title="同阶段另一条经验",
        body="同一个阶段下的另一条结论，用来验证反向关联。",
        evidence=["probe-fail-0"], links=[("stage", "distill-probe")])
    check("关联: 同 ref 的第二条条目写库成功（反向边的前提）", sibling["ok"] is True, sibling)

    neighbours = knowledge_ledger.related(entry_id)
    shared_ids = [item["entry"]["id"] for item in neighbours["items"] if item["via"] == "shared"]
    check("关联: ⭐ 共用同一 stage 的兄弟条目被反向查出（这才是 entry_links 的用处）",
          sibling["id"] in shared_ids
          and neighbours["counts"]["shared"] >= 1, neighbours)
    check("关联: 共享档不带分数（有事实依据，不该伪造成相似度）",
          all(item["score"] is None for item in neighbours["items"] if item["via"] == "shared"),
          neighbours["items"])
    check("关联: 本条自己的 ref 原样带回（前端要显示挂在哪个阶段上）",
          {"refKind": "stage", "refId": "distill-probe"} in neighbours["links"],
          neighbours["links"])

    near = knowledge_ledger.upsert(
        kind="fact", scope="global", title="Krea2 夜景低步数会过曝",
        body="低步数在夜景里容易过曝，先降步数再看。")
    near_dup = knowledge_ledger.upsert(
        kind="fact", scope="global", title="Krea2 夜景低步数会过曝，先降步数",
        body="低步数在夜景里容易过曝，先降步数再看。")
    unrelated = knowledge_ledger.upsert(
        kind="fact", scope="global", title="口型与音轨对不上",
        body="成片的唇形与音轨差半拍。")
    cross_kind = knowledge_ledger.upsert(
        kind="lesson", scope="EP3", title="Krea2 夜景低步数会过曝",
        body="低步数在夜景里容易过曝，先降步数再看。", evidence=["probe-fail-1"])
    fact_neighbours = knowledge_ledger.related(near["id"])
    similar_ids = [item["entry"]["id"] for item in fact_neighbours["items"]
                   if item["via"] == "similar"]
    check("关联: 正文近乎同源的条目进相似档且带分数（≥ SIMILAR_RATIO）",
          near_dup["id"] in similar_ids
          and all(float(item["score"] or 0) >= knowledge_ledger.SIMILAR_RATIO
                  for item in fact_neighbours["items"] if item["via"] == "similar"),
          fact_neighbours["items"])
    check("关联: ⭐ 不相关条目**不**进相似档（否则邻居列表就是噪声）",
          unrelated["id"] not in similar_ids, similar_ids)
    check("关联: ⭐ 跨 kind 不做相似（fact 与 lesson 字面再像也不是一件事）",
          cross_kind["id"] not in similar_ids, similar_ids)
    check("关联: 空文本相似度是 0 而不是 1（否则空条目会跟谁都像）",
          knowledge_ledger._similarity("", "夜景过曝") == 0.0)

    knowledge_ledger.soft_delete(near_dup["id"])
    deleted_neighbours = knowledge_ledger.related(near["id"])
    check("关联: 已软删的邻居默认不出现（回收站里的条目不该混进正常邻居）",
          near_dup["id"] not in [i["entry"]["id"] for i in deleted_neighbours["items"]])
    revived = knowledge_ledger.related(near["id"], include_deleted=True)
    check("关联: includeDeleted=True 时才把已删邻居带回来",
          near_dup["id"] in [i["entry"]["id"] for i in revived["items"]])

    knowledge_ledger.soft_delete(near["id"])
    dead = knowledge_ledger.related(near["id"])
    check("关联: 条目已软删 ⇒ found=True/deleted=True（不是不存在）",
          dead["found"] is True and dead["deleted"] is True and dead["items"] == [], dead)
    missing = knowledge_ledger.related("no-such-entry-xyzzy")
    check("关联: 条目不存在 ⇒ found=False 且 available=True（不抛异常、也不假装成功）",
          missing["found"] is False and missing["available"] is True, missing)

    # 路由层：服务层绿 ≠ 端点通 —— camelCase 别名只接一半就 422，而前端传的正是驼峰
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import knowledge_loop as knowledge_loop_router

    probe_app = FastAPI()
    probe_app.include_router(knowledge_loop_router.router)
    with TestClient(probe_app) as client:
        base = "/api/v1/knowledge-loop"
        resp = client.get(f"{base}/entries/{near['id']}/related",
                          params={"limit": 5, "includeDeleted": "true"})
        payload = resp.json()
        check("关联: GET /entries/{id}/related 通，includeDeleted 别名可收（驼峰不是 422）",
              resp.status_code == 200 and payload.get("data", {}).get("found") is True
              and payload.get("data", {}).get("deleted") is True, payload)

        resp = client.get(f"{base}/context", params={"stage": "distill-probe", "limit": 5})
        payload = resp.json()
        check("关联: GET /context 按阶段反查通，且回 data 包装",
              resp.status_code == 200 and payload.get("data", {}).get("total", 0) >= 1
              and payload["data"]["refs"][0]["refId"] == "distill-probe", payload)

        resp = client.get(f"{base}/context")
        payload = resp.json()
        check("关联: GET /context 不给参数 ⇒ 200 + total=0 + hint（不是 422，也不返回全量）",
              resp.status_code == 200 and payload["data"]["total"] == 0
              and bool(payload["data"]["hint"]), payload)

        resp = client.get(f"{base}/context/refs", params={"refKind": "stage"})
        payload = resp.json()
        check("关联: GET /context/refs 只回 stage 档，且条目数是正数",
              resp.status_code == 200
              and all(i["refKind"] == "stage" for i in payload["data"]["items"])
              and any(i["refId"] == "distill-probe" and i["entries"] >= 1
                      for i in payload["data"]["items"]), payload)

    lift = knowledge_distill.refresh_lift()
    check("蒸馏: 样本不足的 accepted 条目写 lift=None 并计入 skipped",
          lift["skipped"] >= 1 and all(i["lift"] is None for i in lift["items"]
                                       if i["why"]), lift)

    # ================================================================== ④ 投喂
    knowledge_ledger.upsert(kind="lesson", scope="EP1", title="不属于 global 的经验",
                            body="这条在 agent 级作用域里，不该进全局提示词。",
                            evidence=["trace:agent/t1"], status="accepted")

    compiled = knowledge_feed.compile_for_prompt("script")
    check("投喂: ⭐ 只取 accepted（proposed 不进 system prompt）",
          compiled["available"] is True and compiled["text"] != ""
          and all(i in {e["id"] for e in knowledge_ledger.list_entries(
              status="accepted", limit=200)["items"]} for i in compiled["entries"]), compiled)
    check("投喂: 非 global / 非 agent:script 作用域的条目被排除",
          "不属于 global 的经验" not in compiled["text"], compiled["text"])

    tight = knowledge_feed.compile_for_prompt("script", budget_chars=20)
    check("投喂: 预算不够时 truncated=True 且 used 不超预算（截断如实报）",
          tight["truncated"] is True and tight["used"] <= 20, tight)

    empty_feed = knowledge_feed.compile_for_prompt("script", kinds=("recipe",))
    check("投喂: ⭐ 无可用知识时 text 为空但 available=True（「没好料」不是「库坏了」）",
          empty_feed["text"] == "" and empty_feed["available"] is True
          and "accepted" in empty_feed["hint"], empty_feed)

    #: 写入口给的作用域是 EP<id>/DRAMA<id>（knowledge_tools）与 stage:/shot:（knowledge_distill），
    #: 而读入口原先只认 global 与「全仓没人写」的 agent:<type> ⇒ 同一集里刚采纳的经验
    #: 在下一次同集跑次里注入不进去，回执还只说「还没有知识」。这一组钉住键域对齐。
    broad = knowledge_feed.compile_for_prompt("script")
    ep_feed = knowledge_feed.compile_for_prompt("script", scopes=["EP1"])
    check("投喂: ⭐ 传 scopes=EP1 才带出 EP1 作用域条目（写读键域对齐，不是靠 agent:<type>）",
          "不属于 global 的经验" in ep_feed["text"] and "EP1" in ep_feed["scopes"], ep_feed)
    check("投喂: 不传 scopes 时同一条仍被排除，且 blocked 如实多算（默认不猜作用域）",
          "不属于 global 的经验" not in broad["text"]
          and broad["blocked"] >= 1 and broad["blocked"] > ep_feed["blocked"],
          {"broadBlocked": broad["blocked"], "epBlocked": ep_feed["blocked"]})

    check("投喂: scopes_for 与写入口字面形一致（EP12／DRAMA3），非数字不编",
          knowledge_feed.scopes_for(episode_id=12, drama_id=3) == ["EP12", "DRAMA3"]
          and knowledge_feed.scopes_for() == []
          and knowledge_feed.scopes_for(episode_id="x") == [],
          knowledge_feed.scopes_for(episode_id=12, drama_id=3))

    knowledge_ledger.upsert(kind="threshold", scope="EP9", title="阈值只在第九集成立",
                            body="第九集的过曝阈值比其他集低一档。",
                            evidence=["trace:agent/t1"], status="accepted")
    blocked_feed = knowledge_feed.compile_for_prompt("script", kinds=("threshold",))
    check("投喂: ⭐ 有知识但作用域不匹配 ⇒ text 空 + blocked>0 + hint 点名作用域"
          "（不是「还没积累」）",
          blocked_feed["text"] == "" and blocked_feed["blocked"] >= 1
          and "作用域" in blocked_feed["hint"], blocked_feed)

    #: 准入打通了还不够：若排序仍只看 weight，一条全局高置信成法就会把
    #: 本集的具体经验挤出 limit 之外，定向通道等于开了一半。这组钉住「定向优先占预算」。
    #: 造数用 `work` —— 它不在 DEFAULT_KINDS 里，不会污染其它判据的默认投喂；
    #: 证据照挂真实 run（假 run_id 会被 check_evidence 拒收，且拒收是静默的，故逐条核 ok）。
    for index in range(9):
        written = knowledge_ledger.upsert(
            kind="work", scope="global", title=f"通用成法{index}",
            body=f"通用第{index}条正文。", evidence=["trace:agent/t1"],
            confidence=0.9, status="accepted")
        check(f"投喂: 造数 {index} 写入成功（静默拒收是造数最常踩的坑）",
              written.get("ok") is True, written)
    knowledge_ledger.upsert(kind="work", scope="EP1", title="只在第一集成立的写法",
                            body="第一集的外景只用这一种布光。",
                            evidence=["trace:agent/t1"], confidence=0.1,
                            status="accepted")
    knowledge_ledger.upsert(kind="work", scope="EP1", title="第一集特有的机位",
                            body="第一集的过肩镜一律从右后方起。",
                            evidence=["trace:agent/t1"], confidence=0.05,
                            status="accepted")
    ranked = knowledge_feed.compile_for_prompt("script", scopes=["EP1"], limit=8,
                                               kinds=("work",))
    check("投喂: ⭐ 定向条目挤进预算（9 条全局 0.9 vs 2 条本集 0.1/0.05，limit=8），"
          "且档内按 weight 排而不是按插入顺序",
          "只在第一集成立的写法" in ranked["text"]
          and "第一集特有的机位" in ranked["text"]
          and ranked["targeted"] == 2
          and ranked["text"].index("只在第一集成立的写法")
          < ranked["text"].index("第一集特有的机位"), ranked)

    plain_feed = knowledge_feed.compile_for_prompt("script", limit=8, kinds=("work",))
    check("投喂: 不给 scopes 时两条仍在预算外，targeted=0、blocked=2"
          "（定向没开就不假装开）",
          "只在第一集成立的写法" not in plain_feed["text"]
          and "第一集特有的机位" not in plain_feed["text"]
          and plain_feed["targeted"] == 0 and plain_feed["blocked"] == 2, plain_feed)

    _run("vid-good", kind="video", status="ok", stage="video-probe", score=0.9,
         meta={"prompt": "雨夜巷口，逆光", "steps": 8, "cfg": 1.0, "seed": 777})
    _run("vid-bad", kind="video", status="ok", stage="video-probe", score=0.2,
         meta={"prompt": "糊成一团"})
    _run("vid-noprompt", kind="video", status="ok", stage="video-probe", score=0.85, meta={})

    dataset = knowledge_feed.export_runs_dataset(TMP / "ds-runs", kind="video",
                                                min_score=0.5, limit=50)
    items_path = Path(dataset["itemsPath"])
    lines = [json.loads(line) for line in
             items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check("投喂: 导出 runs 只留过线样本，且带 rights=self-produced",
          dataset["count"] == 1 and len(lines) == 1
          and lines[0]["rights"] == "self-produced" and lines[0]["role"] == "train",
          dataset)
    check("投喂: 没提示词的跑次跳过并计数（不编空串充数）",
          dataset["noPrompt"] >= 1, dataset["noPrompt"])
    check("投喂: 预算外的低分样本记为 skippedLowScore",
          dataset["skippedLowScore"] >= 1, dataset["skippedLowScore"])

    eval_set = knowledge_feed.export_runs_dataset(TMP / "ds-eval", kind="video",
                                                  max_score=0.3, include_low=True, limit=50)
    check("投喂: ⭐ include_low 的低分样本 role=eval（负样本不进训练集）",
          eval_set["count"] == 1 and eval_set["roles"].get("eval") == 1, eval_set["roles"])

    kb_set = knowledge_feed.export_knowledge_dataset(TMP / "ds-kb")
    check("投喂: 导出知识数据集（instruction/response 对，accepted 才有）",
          kb_set["count"] >= 1
          and (Path(kb_set["itemsPath"]).exists() and (TMP / "ds-kb" / "manifest.json").exists()),
          kb_set)
    check("投喂: manifest 带指纹与 role/rights 分布",
          len(kb_set["fingerprint"]) == 16 and kb_set["rights"].get("self-produced") == kb_set["count"],
          kb_set["fingerprint"])

    # ================================================================== ⑤ 接线
    from app.main import app  # noqa: PLC0415

    paths = app.openapi()["paths"]
    wanted = ["/runs", "/runs/stats", "/runs/rebuild", "/entries", "/search", "/distill",
              "/lift/refresh", "/feed/preview", "/feed/export", "/report",
              "/datasets", "/datasets/resolve", "/context", "/context/refs"]
    missing = [p for p in wanted if f"/api/v1/knowledge-loop{p}" not in paths]
    check("接线: 知识闭环路由已挂到 /api/v1/knowledge-loop", not missing, missing)
    check("接线: 关联读侧两个口也挂上了（/context 与 /context/refs）",
          all(f"/api/v1/knowledge-loop{p}" in paths for p in ["/context", "/context/refs"]))
    check("接线: 条目邻居口是路径参数形态（/entries/{entry_id}/related）",
          any(k.endswith("/entries/{entry_id}/related") for k in paths), list(paths)[:3])
    check("接线: 条目详情与状态口在（路径参数形态）",
          all(any(k.startswith(f"/api/v1/knowledge-loop{p}") for k in paths)
              for p in ["/entries/{entry_id}", "/entries/{entry_id}/status"]))

    from app.agent.tools.knowledge_tools import create_knowledge_tools  # noqa: PLC0415

    tools = create_knowledge_tools(episode_id=1, drama_id=1)
    propose = tools["propose_knowledge"]
    check("接线: ★ 提案工具的参数表里没有 status（模型不能自己拍板）",
          "status" not in propose.input_schema["properties"],
          list(propose.input_schema["properties"]))
    check("接线: 提案工具要求 kind/title/body",
          set(propose.input_schema["required"]) == {"kind", "title", "body"},
          propose.input_schema["required"])

    rejected = asyncio.run(propose.run({
        "kind": "lesson", "title": "模型编的经验", "body": "证据是编的",
        "evidence": ["run:made-up-999"], "status": "accepted",
        "confidence": 0.99,
    }))
    check("接线: ⭐ 工具侧假证据整条拒（幻觉进不了台账）",
          rejected.get("ok") is False and "拒收" in rejected.get("hint", ""), rejected)

    accepted = asyncio.run(propose.run({
        "kind": "lesson", "title": "模型观察到的真现象", "body": "从 trace:agent/t1 看出来的",
        "evidence": ["trace:agent/t1"], "status": "accepted", "confidence": 0.99,
    }))
    stored = knowledge_ledger.detail(accepted.get("id", "")) if accepted.get("ok") else None
    check("接线: ⭐ 即使传 status=accepted 也只落 proposed（只能提议）",
          stored is not None and stored["status"] == "proposed", stored and stored["status"])
    check("接线: 工具写入的 confidence 封顶 0.6",
          stored is not None and float(stored["confidence"]) <= 0.6,
          stored and stored["confidence"])
    check("接线: 工具默认作用域取注入的 episodeId", stored is not None
          and stored["scope"] == "EP1", stored and stored["scope"])

    searched = asyncio.run(tools["search_knowledge"].run({"query": "过曝"}))
    check("接线: 检索工具透出 ranking（不粉饰）",
          searched.get("available") is True and searched.get("ranking") == "like", searched)

    contextual = asyncio.run(tools["context_knowledge"].run({"stage": "distill-probe"}))
    check("接线: ⭐ 上下文工具按 stage 反查到该阶段经验（关键词检索答不了这一问）",
          contextual.get("available") is True and contextual.get("total", 0) >= 1
          and any(i["refId"] == "distill-probe" for i in contextual.get("items", [])),
          contextual)

    at_shot = asyncio.run(tools["context_knowledge"].run({"shot": "ghost-shot-xyzzy"}))
    check("接线: 上下文工具查不存在的镜头 ⇒ 空 + hint（不编经验）",
          at_shot.get("total") == 0 and bool(at_shot.get("hint")), at_shot)

    discover = asyncio.run(tools["context_knowledge"].run({}))
    check("接线: ⭐ 三个参数都不给 ⇒ 回 ref 清单（让模型别猜阶段名）",
          discover.get("available") is True and discover.get("items")
          and "distill-probe" in discover.get("hint", ""), discover)

    neighbour_tool = asyncio.run(tools["context_knowledge"].run({"entryId": sibling["id"]}))
    check("接线: 上下文工具取邻居带回本条挂载的 ref（links 原样给模型看）",
          neighbour_tool.get("found") is True
          and {"refKind": "stage", "refId": "distill-probe"} in neighbour_tool.get("links", []),
          neighbour_tool)
    dead_tool = asyncio.run(tools["context_knowledge"].run({"entryId": near["id"]}))
    check("接线: 已软删条目在工具侧也不给邻居（与详情一致，不留后门）",
          dead_tool.get("deleted") is True and dead_tool.get("items") == [], dead_tool)

    bad_kind = asyncio.run(tools["search_knowledge"].run({"query": "过曝", "kind": "nope"}))
    check("接线: 工具给错类型不静默回空（available=False + hint 列可选项）",
          bad_kind.get("available") is False and "可选" in bad_kind.get("hint", ""), bad_kind)

    from app.routers.knowledge_loop import _since_ns  # noqa: PLC0415

    check("接线: sinceDays 非数字 → None（不猜成 0 天 = 全量）",
          _since_ns("abc") is None and _since_ns(None) is None and _since_ns("") is None)
    week_ago = _since_ns("7")
    check("接线: sinceDays 转纳秒且落在过去（不是 0、不是未来）",
          isinstance(week_ago, int) and 0 < week_ago < time.time_ns(), week_ago)

    from fastapi.testclient import TestClient  # noqa: PLC0415

    client = TestClient(app)
    runs_resp = client.get("/api/v1/knowledge-loop/runs", params={"limit": 2})
    body = runs_resp.json()
    check("接线: GET /runs 走统一信封（code=200 + data.total/items）",
          runs_resp.status_code == 200 and body["code"] == 200
          and body["data"]["total"] >= 2 and len(body["data"]["items"]) == 2, body.get("code"))

    feed_resp = client.get("/api/v1/knowledge-loop/feed/preview",
                           params={"agentType": "script", "budgetChars": 40})
    feed_body = feed_resp.json()
    check("接线: GET /feed/preview 的 budgetChars 别名生效",
          feed_resp.status_code == 200 and feed_body["data"]["budget"] == 40, feed_body.get("data"))

    bad_post = client.post("/api/v1/knowledge-loop/entries", json={
        "kind": "lesson", "title": "接口层假证据", "body": "x", "evidence": ["trace:agent/none"]})
    #: 2026-11-07 契约调整：证据核不过是「格式都对、内容站不住」⇒ **422**（旧断言是 400）。
    #: 400 留给「请求形状不成立」（缺字段/未知枚举/类型错），两者前端动作不同：
    #: 400 改请求、422 补证据。分派真源 = 服务层 ``result["code"]``，见
    #: ``docs/knowledge-base-contract.md`` §1.1。
    check("接线: POST /entries 证据核不过 → 422 + message 带原因",
          bad_post.status_code == 422 and "证据核不过" in bad_post.json()["message"],
          bad_post.json())

    stats_resp = client.get("/api/v1/knowledge-loop/runs/stats", params={"groupBy": "body"})
    check("接线: 非法 groupBy → 400（不静默回全量）", stats_resp.status_code == 400)

    report = client.get("/api/v1/knowledge-loop/report").json()
    check("接线: GET /report 同时给事实层与台账两本账",
          report["code"] == 200 and "runs" in report["data"] and "ledger" in report["data"],
          list(report.get("data") or {}))

    # ================================================================== ⑥ L1 注入
    from app.agent import runtime as agent_runtime  # noqa: PLC0415

    os.environ.pop(agent_runtime.KNOWLEDGE_FEED_ENV, None)
    check("注入: ⭐ 开关默认关 ⇒ system prompt 逐字不变（加闭环不改既有产出）",
          agent_runtime.knowledge_feed_enabled() is False
          and agent_runtime.append_knowledge_feed("script", "BASE") == "BASE")

    os.environ[agent_runtime.KNOWLEDGE_FEED_ENV] = "1"
    fed: list[str] = []
    injected_text = agent_runtime.append_knowledge_feed("script", "BASE", fed)
    check("注入: 打开后追加分节，且标题声明「仅供参考、冲突以上文为准」",
          injected_text.startswith("BASE")
          and agent_runtime.KNOWLEDGE_FEED_HEADER in injected_text
          and "过曝先降步数" in injected_text, injected_text)
    #: id 形态是 ``<kind>-<16 位 hash>``（kind 前缀让「光看 id 就知道是什么类」）
    check("注入: 真正写进去的条目 id 被回传（lift 的唯一数据源）",
          len(fed) >= 1
          and all(isinstance(i, str) and len(i.rsplit("-", 1)[-1]) == 16 for i in fed), fed)

    fed_ep: list[str] = []
    ep_text = agent_runtime.append_knowledge_feed("script", "BASE", fed_ep, scopes=["EP1"])
    check("注入: ⭐ scopes 真透传进编译（同一集的经验这次进得了 system prompt）",
          "不属于 global 的经验" in ep_text, ep_text)

    original_compile = knowledge_feed.compile_for_prompt
    knowledge_feed.compile_for_prompt = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        check("注入: 台账炸了只跳过（不让统计坏了阻断创作）",
              agent_runtime.append_knowledge_feed("script", "BASE") == "BASE")
    finally:
        knowledge_feed.compile_for_prompt = original_compile

    os.environ.pop(agent_runtime.KNOWLEDGE_FEED_ENV, None)

    preview = client.get("/api/v1/knowledge-loop/feed/preview",
                         params={"agentType": "script", "episodeId": 1}).json()
    check("投喂: ⭐ GET /feed/preview 带 episodeId 时与生产同口径（预览得到本集条目）",
          "不属于 global 的经验" in (preview["data"]["text"] or "")
          and "EP1" in preview["data"]["scopes"], preview.get("data"))

    preview_bare = client.get("/api/v1/knowledge-loop/feed/preview",
                              params={"agentType": "script"}).json()
    check("投喂: 预览不带 episodeId ⇒ blocked>0（把漏掉的那些如实数出来）",
          preview_bare["data"]["blocked"] >= 1, preview_bare.get("data"))

    source = (BP / "app" / "agent" / "runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "assemble_full_instructions"]
    calls_feed = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "append_knowledge_feed"
        for node in ast.walk(nodes[0])
    ) if nodes else False
    check("注入: ⭐ 注入挂在**唯一组装点**里（否则两个入口会分叉，历史缺陷重演）",
          calls_feed, [n.name for n in nodes])

    feed_calls = [n for n in ast.walk(nodes[0])
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "append_knowledge_feed"] if nodes else []
    scoped_in_assembler = any(
        any(kw.arg == "scopes"
            and isinstance(kw.value, ast.Call)
            and isinstance(kw.value.func, ast.Attribute)
            and kw.value.func.attr == "scopes_for"
            for kw in call.keywords)
        for call in feed_calls)
    check("注入: ⭐ 组装点自己从 episode_id/drama_id 算作用域（不指望各调用方记得传）",
          scoped_in_assembler, [kw.arg for c in feed_calls for kw in c.keywords])

    check("注入: record_knowledge_usage 记 failed 也照记（失败样本比成功的更有用）",
          agent_runtime.record_knowledge_usage(fed, type("T", (), {"trace_id": "abc12345"})(),
                                               "failed") is None
          and knowledge_ledger.usage_summary(fed[0])["settled"] >= 1)

    # ================================================================== ⑦ 参数覆盖层
    from app.services.engine import pipeline as engine_pipeline  # noqa: PLC0415
    from app.services.engine import tuning  # noqa: PLC0415

    tuning_file = TMP / "engine" / "tuning.json"
    os.environ["ENGINE_TUNING_JSON"] = tuning_file.as_posix()

    probe = engine_pipeline.GenerationRequest(prompt="测试用提示词", stage="qwen_image")
    plain, no_notes = tuning.apply_to_request("qwen_image", probe)
    check("覆盖层: ⭐ 没有 tuning.json = 行为逐字不变（不是「开关关掉」）",
          plain is probe and no_notes == [], no_notes)

    for name, result in (
        ("缺 basis", tuning.set_field("qwen_image", "steps", 24, basis="",
                                      evidence=["trace:agent/t1"])),
        ("缺 evidence", tuning.set_field("qwen_image", "steps", 24, basis="夜戏过曝",
                                         evidence=[])),
        ("假证据", tuning.set_field("qwen_image", "steps", 24, basis="夜戏过曝",
                                    evidence=["trace:agent/ghost"])),
        ("未登记字段", tuning.set_field("wan22", "steps", 4, basis="x",
                                        evidence=["trace:agent/t1"])),
        ("越界值", tuning.set_field("qwen_image", "steps", 9999, basis="x",
                                    evidence=["trace:agent/t1"])),
    ):
        check(f"覆盖层: {name} 拒写（附理由，不夹到边界也不硬写）",
              result["ok"] is False and bool(result["why"]), result)

    allowed = tuning.set_field("qwen_image", "steps", 24,
                               basis="夜景 8 步过曝，回到 24 步更稳",
                               evidence=["trace:agent/t1"])
    check("覆盖层: 证据可核即可写，回执带 previous",
          allowed["ok"] is True and allowed["value"] == 24 and allowed["previous"] is None,
          allowed)

    tuning.set_field("qwen_image", "steps", 28, basis="再抬一点", evidence=["trace:agent/t1"])
    backups = sorted((tuning_file.parent / "_backups").glob("tuning.*.json"))
    check("覆盖层: ⭐ 每次改都先备份（一条命令能回滚）",
          len(backups) >= 1, [p.name for p in backups])

    tuned, tuned_notes = tuning.apply_to_request("qwen_image", probe)
    check("覆盖层: ⭐ 30 → 28 且回执写明来源与依据（不偷偷改参数）",
          tuned.steps == 28 and tuned is not probe
          and any("参数来源: tuning:qwen_image.steps" in note and "28" in note
                  for note in tuned_notes), tuned_notes)

    explicit = engine_pipeline.GenerationRequest(prompt="测试用提示词", stage="qwen_image",
                                                steps=8)
    kept, kept_notes = tuning.apply_to_request("qwen_image", explicit)
    check("覆盖层: ⭐ 调用方显式给 steps=8 ⇒ 不覆盖（用户的表态优先）",
          kept is explicit and kept.steps == 8 and kept_notes == [], kept_notes)

    sibling, sibling_notes = tuning.apply_to_request(
        "zimage", engine_pipeline.GenerationRequest(prompt="测试用提示词", stage="zimage"))
    check("覆盖层: 别的 stage 不受影响（逐族登记，不是全局开关）",
          sibling.steps == 30 and sibling_notes == [], sibling_notes)

    plan = engine_pipeline.build_plan(
        engine_pipeline.GenerationRequest(prompt="测试用提示词", stage="qwen_image"))
    check("覆盖层: ⭐ 真跑 build_plan 生效，且 σ 与步数同源（不是事后改 plan.steps）",
          plan.steps == 28 and len(plan.sigmas) == plan.steps + 1
          and any("参数来源" in warn for warn in plan.warnings),
          (plan.steps, len(plan.sigmas), plan.warnings))

    check("覆盖层: report 如实标 source/override",
          any(row["stage"] == "qwen_image" and row["field"] == "steps"
              and row["source"] == "override:qwen_image.steps"
              for row in tuning.report()["fields"]), tuning.report()["fields"][:2])

    tuning_file.write_text("{ 这不是 JSON", encoding="utf-8")
    broken, broken_notes = tuning.apply_to_request("qwen_image", probe)
    check("覆盖层: ⭐ 文件坏了按默认跑 + problems 如实报（不静默失效）",
          broken.steps == 30 and broken_notes == [] and bool(tuning.problems()),
          tuning.problems())

    reset = tuning.reset()
    check("覆盖层: reset 全清后回到默认（回滚闭环）",
          reset["ok"] is True and tuning.resolve("qwen_image", "steps", 30) == (30, "default"),
          reset)

    os.environ.pop("ENGINE_TUNING_JSON", None)

    pipeline_source = (BP / "app" / "services" / "engine" / "pipeline.py").read_text(
        encoding="utf-8")
    pipeline_tree = ast.parse(pipeline_source)
    build_nodes = [n for n in ast.walk(pipeline_tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "build_plan"]
    wired = False
    if build_nodes:
        wired = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_to_request"
            for node in ast.walk(build_nodes[0])
        )
    check("覆盖层: ⭐ 覆盖层接在 build_plan 里（不是死代码：查得到、改得动）",
          wired, [n.name for n in build_nodes])

    # ================================================================== ⑧ 训练登记表
    from app.services import training_registry  # noqa: PLC0415
    from app.services.lora_train.runtime import LoraTrainRuntime  # noqa: PLC0415

    registry_file = TMP / "training-registry.sqlite"
    os.environ["TRAINING_REGISTRY_DB"] = registry_file.as_posix()

    check("登记: 空库不崩，且如实报 0 条（不是 500、也不是假装有库）",
          training_registry.stats()["totalJobs"] == 0
          and training_registry.list_jobs() == []
          and training_registry.list_models() == [])

    ws = TMP / "tasks" / "lora-test0001"
    out_dir = TMP / "lora-out"
    job = {"id": "lora-test0001", "modelKey": "wan22", "status": "running",
           "stage": "训练中", "workspace": ws.as_posix(),
           "logPath": (ws / "train.log").as_posix(),
           "createdAt": 1_700_000_000_000, "startedAt": 1_700_000_001_000,
           "values": {"output_dir": out_dir.as_posix()},
           "plan": {"train": ["python", "-m", "train.py"]}}
    check("登记: 任务一提交就落表（进程没了这次提交也查得到）",
          training_registry.record_job(job) is True
          and training_registry.get_job("lora-test0001")["status"] == "running")
    training_registry.record_job({**job, "status": "succeeded",
                                 "finishedAt": 1_700_000_900_000})
    rows = training_registry.list_jobs()
    check("登记: 同 taskId 幂等 upsert（一条、状态与收尾时间已更新）",
          len(rows) == 1 and rows[0]["status"] == "succeeded"
          and rows[0]["finishedAt"] == 1_700_000_900_000, rows)
    check("登记: skip_ids 生效（合并视图靠它去重）",
          training_registry.list_jobs(skip_ids=["lora-test0001"]) == [])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wan22_lora_ep01.safetensors").write_bytes(b"fake-lora-weights")
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "intermediate.safetensors").write_bytes(b"half-done")
    found = training_registry.scan_artifacts(training_registry.get_job("lora-test0001"))
    models = training_registry.list_models()
    check("登记: ⭐ 扫产物得 2 个文件，一律 candidate 且非默认（候选不自动成默认）",
          len(found) == 2 and len(models) == 2
          and all(m["status"] == "candidate" and m["isDefault"] is False for m in models),
          [(m["path"], m["status"], m["isDefault"]) for m in models])

    target = next(m for m in models if m["path"].endswith("wan22_lora_ep01.safetensors"))
    other = next(m for m in models if m["modelId"] != target["modelId"])
    no_basis = training_registry.promote(target["modelId"], basis="",
                                        evidence=[registry_file.as_posix()])
    check("登记: 提升缺 basis ⇒ 拒", no_basis["ok"] is False and "basis" in no_basis["why"],
          no_basis)
    fake = training_registry.promote(target["modelId"], basis="夜景细节更好",
                                     evidence=["trace:agent/ghost"])
    check("登记: 提升用假证据 ⇒ 拒（与台账同一把尺，不另开豁免）",
          fake["ok"] is False and "证据核不过" in fake["why"], fake)
    ghost = training_registry.record_artifact("lora-test0001", TMP / "nope.safetensors")
    missing = training_registry.promote(ghost["modelId"], basis="x",
                                        evidence=[registry_file.as_posix()])
    check("登记: 文件不在盘上 ⇒ 拒（设成默认会让下次出图崩在装载）",
          missing["ok"] is False and "不在盘上" in missing["why"], missing)

    first = training_registry.promote(
        target["modelId"], basis="夜景细节明显更稳，EP001 复跑 5 张全过",
        evidence=[registry_file.as_posix()])
    states = {m["modelId"]: m for m in training_registry.list_models()}
    check("登记: 证据可核 ⇒ accepted + 默认档",
          first["ok"] is True and states[target["modelId"]]["status"] == "accepted"
          and states[target["modelId"]]["isDefault"] is True, first)

    second = training_registry.promote(other["modelId"], basis="另一个也用着顺手",
                                       evidence=[registry_file.as_posix()])
    states = {m["modelId"]: m for m in training_registry.list_models()}
    check("登记: ⭐ 同类只留一个默认档（旧的自动 deprecated，文件不删）",
          second["ok"] is True and target["modelId"] in second["demoted"]
          and states[other["modelId"]]["isDefault"] is True
          and states[target["modelId"]]["isDefault"] is False
          and states[target["modelId"]]["status"] == "deprecated", second)

    training_registry.scan_artifacts(training_registry.get_job("lora-test0001"))
    states = {m["modelId"]: m for m in training_registry.list_models()}
    check("登记: ⭐ 重扫盘不许把人工结论打回 candidate",
          states[target["modelId"]]["status"] == "deprecated"
          and states[other["modelId"]]["isDefault"] is True, states[target["modelId"]])

    check("登记: demote 撤掉默认档但记录不删（「不用它」≠「它没存在过」）",
          training_registry.demote(other["modelId"], why="换回上一版")["ok"] is True
          and training_registry.stats()["totalModels"] == 3
          and training_registry.stats()["defaults"] == [])

    training_registry.promote(target["modelId"], basis="复跑通过，定回来",
                              evidence=[registry_file.as_posix()])
    Path(target["path"]).unlink()
    issues = training_registry.problems()
    check("登记: ⭐ 默认档文件被删 ⇒ problems 点名（不许静静用着不存在的文件）",
          any("不在盘上" in item for item in issues), issues)

    training_registry.record_job({"id": "lora-zombie", "status": "running",
                                  "createdAt": 1_700_000_000_000,
                                  "startedAt": 1_700_000_000_000})
    check("登记: running 僵尸被点名（重启后内存表消失，这行永远 running）",
          any("lora-zombie" in item and "running" in item
              for item in training_registry.problems()))

    fresh = LoraTrainRuntime()
    listed = fresh.list_tasks(limit=10)
    check("登记: ⭐ 新实例（=重启后）list_tasks 仍看得到历史，且标 restored",
          len(listed) >= 2 and all(item.get("restored") for item in listed)
          and any(item["id"] == "lora-test0001" and item["terminal"] is True
                  for item in listed), listed[:2])
    check("登记: 重启后没有残留的忙闲状态（没跑就是没跑，不假装还在跑）",
          fresh.busy() is False)

    os.environ.pop("TRAINING_REGISTRY_DB", None)

    runtime_source = (BP / "app" / "services" / "lora_train" / "runtime.py").read_text(
        encoding="utf-8")
    runtime_tree = ast.parse(runtime_source)
    record_calls = [
        node for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_record"
    ]
    check("登记: ⭐ 收尾登记真的接在运行时的两条出口上（否则又是死代码）",
          len(record_calls) >= 3, len(record_calls))

    # ============================================================ ⑨ 数据集引用贯通
    from app.services.lora_train.errors import LoraTrainConfigError  # noqa: PLC0415
    from app.services.lora_train.runtime import check_dataset_ref  # noqa: PLC0415

    #: 默认落点也要指向夹具（否则「只给数据集名」那条解析会去真数据根上找）
    knowledge_feed.get_data_root = lambda: TMP.as_posix()
    by_name = knowledge_feed.export_runs_dataset(None, kind="video", min_score=0.5, limit=50)
    check("数据集: 不给目录时落在 <数据根>/lora-train/datasets/runs",
          Path(by_name["itemsPath"]).parent == TMP / "lora-train" / "datasets" / "runs",
          by_name["itemsPath"])

    good = knowledge_feed.resolve_dataset(TMP / "ds-runs")
    check("数据集: ⭐ 导出集的引用可核（指纹/条数/角色/权利/字节都拿得出）",
          good["ok"] is True and good["dataset"]["count"] == 1
          and len(str(good["dataset"]["fingerprint"])) == 16
          and len(str(good["dataset"]["contentSha256"])) == 64
          and good["dataset"]["integrity"] == "sha256"
          and good["dataset"]["roles"] == {"train": 1}
          and good["dataset"]["rights"] == {"self-produced": 1}, good)

    check("数据集: 只给名字（runs）也能解析到默认落点那份",
          knowledge_feed.resolve_dataset("runs")["ok"] is True)
    listed = knowledge_feed.list_datasets()
    check("数据集: ⭐ 列举能让人选 ref（带 itemsExists，不逐条核样本）",
          listed["available"] is True and len(listed["items"]) == 1
          and listed["items"][0]["ref"] == "runs"
          and listed["items"][0]["itemsExists"] is True, listed)

    ghost = knowledge_feed.resolve_dataset(TMP / "no-such-dataset")
    check("数据集: 地址不存在 ⇒ 拒（不猜、不返回空引用）",
          ghost["ok"] is False and "找不到导出集" in ghost["why"], ghost)

    #: HTTP 面：前端「选一批数据 / 先看一眼能不能用」就走这两个端点 ✓ ——
    #: 判据与提交口必须是**同一把尺**，所以这里真发请求、不只看路由表里有没有它。
    ds_list = client.get("/api/v1/knowledge-loop/datasets").json()
    check("数据集: GET /datasets 列出可选 ref（带 itemsExists，走统一信封）",
          ds_list.get("code") == 200 and ds_list["data"]["available"] is True
          and [item["ref"] for item in ds_list["data"]["items"]] == ["runs"],
          ds_list.get("data"))

    ds_ok = client.get("/api/v1/knowledge-loop/datasets/resolve",
                       params={"ref": str(TMP / "ds-runs")})
    check("数据集: GET /datasets/resolve 核得过 ⇒ 200 + 与提交口同一份事实",
          ds_ok.status_code == 200 and ds_ok.json()["data"]["ok"] is True
          and ds_ok.json()["data"]["dataset"]["integrity"] == "sha256", ds_ok.json())

    ds_bad = client.get("/api/v1/knowledge-loop/datasets/resolve",
                        params={"ref": str(TMP / "no-such-dataset")})
    check("数据集: GET /datasets/resolve 核不过 ⇒ 400 + why（不是 200 里塞 ok=false）",
          ds_bad.status_code == 400 and "找不到导出集" in ds_bad.json()["message"],
          ds_bad.json())

    #: 缺参数走的是全局 ``RequestValidationError`` 处理器 ⇒ **400 带 message**（本仓口径：
    #: 参数错一律 400；FastAPI 默认的 422 会被统一信封吃成 400）—— 关键是别静默回空结果。
    ds_missing = client.get("/api/v1/knowledge-loop/datasets/resolve")
    check("数据集: 缺 ref 参数 ⇒ 400 + 点名 ref（真参数校验，不是默默回个空结果）",
          ds_missing.status_code == 400 and "ref" in ds_missing.json()["message"],
          ds_missing.json())

    #: ✎ 手改导出集：manifest 说 1 条、items 里却塞了 2 行 —— count 对不上就必须拒。
    #: 这条是「假血统」的典型来源：有人为了让训练多吃一点，直接往 jsonl 里加行。
    edited = TMP / "ds-edited"
    edited.mkdir(parents=True, exist_ok=True)
    source_items = Path(good["dataset"]["itemsPath"]).read_text(encoding="utf-8").strip()
    (edited / "items.jsonl").write_text(source_items + "\n" + source_items + "\n",
                                       encoding="utf-8")
    manifest = json.loads((TMP / "ds-runs" / "manifest.json").read_text(encoding="utf-8"))
    manifest["itemsPath"] = (edited / "items.jsonl").as_posix()
    (edited / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                          encoding="utf-8")
    tampered = knowledge_feed.resolve_dataset(edited)
    check("数据集: ⭐ manifest 与 items.jsonl 条数对不上 ⇒ 拒（被手改过）",
          tampered["ok"] is False and "手改" in tampered["why"], tampered)

    rights_less = TMP / "ds-norights"
    rights_less.mkdir(parents=True, exist_ok=True)
    row = json.loads(source_items.splitlines()[0])
    row["rights"] = ""
    (rights_less / "items.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    bad_manifest = dict(manifest, itemsPath=(rights_less / "items.jsonl").as_posix(),
                        rights={})
    (rights_less / "manifest.json").write_text(
        json.dumps(bad_manifest, ensure_ascii=False), encoding="utf-8")
    no_rights = knowledge_feed.resolve_dataset(rights_less)
    check("数据集: ⭐ 样本没有 rights ⇒ 拒（权利不明的样本不进训练集）",
          no_rights["ok"] is False and "rights" in no_rights["why"], no_rights)

    empty_set = TMP / "ds-empty"
    empty_set.mkdir(parents=True, exist_ok=True)
    (empty_set / "items.jsonl").write_text("", encoding="utf-8")
    (empty_set / "manifest.json").write_text(json.dumps(
        dict(manifest, itemsPath=(empty_set / "items.jsonl").as_posix(), count=0,
             roles={}, rights={}), ensure_ascii=False), encoding="utf-8")
    check("数据集: 空集 ⇒ 拒（0 条样本训出来的是「把底模又训一遍」）",
          knowledge_feed.resolve_dataset(empty_set)["ok"] is False)

    #: ✎ 更阴的手改：条数、角色、权利、指纹**全都对得上**，只把某条样本的正文换了。
    #: 指纹只对 source 取值 ⇒ 光靠它这条会一路绿灯。所以第 4 条判据按字节核。
    content_edit = TMP / "ds-content"
    content_edit.mkdir(parents=True, exist_ok=True)
    tweaked = json.loads(source_items.splitlines()[0])
    tweaked["prompt"] = "手改过的一句提示词"
    (content_edit / "items.jsonl").write_text(json.dumps(tweaked, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    (content_edit / "manifest.json").write_text(json.dumps(
        dict(manifest, itemsPath=(content_edit / "items.jsonl").as_posix()),
        ensure_ascii=False), encoding="utf-8")
    changed = knowledge_feed.resolve_dataset(content_edit)
    check("数据集: ⭐ 只改正文（条数/角色/权利/指纹全对得上）⇒ 仍拒：按字节核 sha256",
          changed["ok"] is False and "sha256" in changed["why"], changed)

    legacy = TMP / "ds-legacy"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "items.jsonl").write_text(source_items + "\n", encoding="utf-8")
    legacy_manifest = {key: value for key, value in manifest.items()
                       if key != "contentSha256"}
    legacy_manifest["itemsPath"] = (legacy / "items.jsonl").as_posix()
    (legacy / "manifest.json").write_text(json.dumps(legacy_manifest, ensure_ascii=False),
                                          encoding="utf-8")
    old_set = knowledge_feed.resolve_dataset(legacy)
    check("数据集: ⭐ 老 manifest 没有 contentSha256 ⇒ 退到重算指纹，integrity 如实标低一档",
          old_set["ok"] is True and old_set["dataset"]["integrity"] == "fingerprint",
          old_set.get("dataset"))

    resolved = check_dataset_ref(TMP / "ds-runs")
    #: ``ref`` 原样保留调用方写的那串（便于人对账），其余字段才是归一化事实
    check("数据集: 提交口核引用走同一把尺（返回的是归一化事实，不是原字符串）",
          isinstance(resolved, dict) and resolved["ref"] == str(TMP / "ds-runs")
          and resolved["count"] == 1 and resolved["itemsPath"].endswith("items.jsonl"),
          resolved)
    check("数据集: 没给 datasetRef ⇒ None（不编一个「未知数据集」出来）",
          check_dataset_ref("") is None and check_dataset_ref(None) is None)
    try:
        check_dataset_ref(TMP / "no-such-dataset")
        raised = ""
    except LoraTrainConfigError as err:
        raised = str(err)
    check("数据集: ⭐ 引用核不过 ⇒ 当场抛配置错（400，任务不入队）",
          "datasetRef 核不过" in raised, raised)

    #: ① 登记表这边：任务的 dataset 块要能存下来、重启后仍能读出来（= 血统）
    registry_file2 = TMP / "training-registry-ds.sqlite"
    os.environ["TRAINING_REGISTRY_DB"] = registry_file2.as_posix()
    ds_out = TMP / "ds-out"
    ds_out.mkdir(parents=True, exist_ok=True)
    (ds_out / "with_data.safetensors").write_bytes(b"weights-with-lineage")
    check("数据集: 带 dataset 的任务落登记表",
          training_registry.record_job({
              "id": "lora-ds0001", "modelKey": "wan22", "status": "succeeded",
              "createdAt": 1_700_000_000_000,
              "workspace": (TMP / "tasks" / "lora-ds0001").as_posix(),
              "values": {"output_dir": ds_out.as_posix()},
              "dataset": resolved}) is True)
    restored_job = training_registry.get_job("lora-ds0001")
    check("数据集: ⭐ 重启后再读，仍说得出「这批数据」的指纹与条数",
          (restored_job.get("dataset") or {}).get("fingerprint")
          == resolved["fingerprint"]
          and (restored_job.get("dataset") or {}).get("count") == 1, restored_job.get("dataset"))

    training_registry.scan_artifacts(restored_job)
    with_ds = [m for m in training_registry.list_models() if m.get("dataset")]
    check("数据集: ⭐ 产物也记住了来路（人是从产物反查，不会先记得 taskId）",
          len(with_ds) == 1
          and with_ds[0]["dataset"]["ref"] == resolved["ref"], with_ds)
    check("数据集: stats 报得出「说得出数据来源」的覆盖率",
          training_registry.stats()["jobsWithDataset"] == 1
          and training_registry.stats()["modelsWithDataset"] == 1,
          training_registry.stats())

    ds_listed = LoraTrainRuntime().list_tasks(limit=10)
    check("数据集: ⭐ 合并视图的 restored 行也带 datasetRef（形状与内存行对齐）",
          any(item.get("datasetRef") == resolved["ref"] for item in ds_listed), ds_listed)

    Path(resolved["itemsPath"]).unlink()
    check("数据集: ⭐ 样本文件被清掉 ⇒ problems 点名（这时血统已经断了）",
          any("数据集样本文件不在盘上" in item
              for item in training_registry.problems()), training_registry.problems())
    os.environ.pop("TRAINING_REGISTRY_DB", None)

    runtime_tree = ast.parse((BP / "app" / "services" / "lora_train" / "runtime.py")
                             .read_text(encoding="utf-8"))
    submit_fn = next((node for node in ast.walk(runtime_tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "submit"), None)
    submit_calls = {node.func.id for node in ast.walk(submit_fn)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)} \
        if submit_fn is not None else set()
    submit_kwargs = {kw.arg for node in ast.walk(submit_fn)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == "LoraTrainTask"
                     for kw in node.keywords} if submit_fn is not None else set()
    check("数据集: ⭐ datasetRef 真的接在提交口上（核验 + 写进任务，不是死代码）",
          "check_dataset_ref" in submit_calls and "dataset" in submit_kwargs,
          (sorted(submit_calls), sorted(submit_kwargs)))

    from app.routers.lora_train import _dataset_ref_from  # noqa: PLC0415

    check("数据集: HTTP 两种拼法都认（datasetRef / dataset_ref），空串当没给",
          _dataset_ref_from({"datasetRef": " runs "}) == "runs"
          and _dataset_ref_from({"dataset_ref": "runs"}) == "runs"
          and _dataset_ref_from({"datasetRef": "   "}) is None
          and _dataset_ref_from({}) is None)

    router_tree = ast.parse((BP / "app" / "routers" / "lora_train.py").read_text(encoding="utf-8"))
    router_kwargs = [kw.arg for node in ast.walk(router_tree)
                     if isinstance(node, ast.Call)
                     for kw in node.keywords]
    check("数据集: ⭐ HTTP 口收 datasetRef 并喂给运行时（/train 与 /preview 都收）",
          router_kwargs.count("dataset_ref") == 1
          and "_dataset_ref_from" in (BP / "app" / "routers" / "lora_train.py")
          .read_text(encoding="utf-8"), router_kwargs)

    # ================================================================== 收尾
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name
              + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
