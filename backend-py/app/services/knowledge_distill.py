"""知识蒸馏 —— 从跑次明细里挑出「值得写下来的一件事」，并写成一条知识。

两段式，故意的
==============
**第一段（确定性）**：在 ``outcome-index`` 的明细上做统计，找出四类信号 ——
失败聚集、分数高分离、返工热点、成本离群。这一段**不花一次 LLM 调用**，
所以它永远不可能幻觉出"某处有个问题"。

**第二段（归纳）**：把候选的**摘要 + 允许引用的 run_id 白名单**交给模型，
让它写一句人话（现象 → 根因 → 修法）。模型只能引用白名单里的 id；
写库前 ``knowledge_ledger`` 会再逐条核一遍 —— 核不过整条拒。

没接 LLM 也能跑
===============
``llm=None`` 时用确定性模板产出条目（``source_kind="distill-rule"``），
这样 ① 离线/测试环境可验证全链路；② 模型不可用时不至于让知识积累彻底停摆。
模板条目的 ``confidence`` 上限压到 0.5：它是"机器看出来的现象"，还不是"人的结论"。
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from app.services import knowledge_ledger, outcome_index

__all__ = ["candidates", "distill", "refresh_lift", "prompt_for"]

#: 四类信号的最小样本量（低于这个数只报不写 —— 3 条跑次的"规律"就是噪声）
MIN_SAMPLE = 5

#: 分数高分离的判据：前后 10% 的平均分差
SPLIT_GAP = 0.25
SPLIT_SAMPLE = 10

#: 成本离群：P95 超过 P50 这么多倍
COST_RATIO = 2.0
COST_SAMPLE = 20

#: 返工热点：同一镜头的视频跑次条数
REWORK_COUNT = 3

#: ``lift`` 判负的阈值（低于它就说明这条知识在拖后腿）
LIFT_FLOOR = -0.05
LIFT_MIN_SETTLED = 10


def _load(*, since_ns: int | None = None, limit: int = 5000) -> list[dict[str, Any]]:
    return outcome_index.samples(since_ns=since_ns, limit=limit)


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _evidence_of(rows: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """取最能说明问题的那几条 run（失败优先、分数极端优先），并**逐条核存在**。"""
    ordered = sorted(
        rows,
        key=lambda r: (0 if r.get("status") == "failed" else 1,
                       r.get("score") if r.get("score") is not None else 0.5),
    )
    picked: list[str] = []
    for row in ordered:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in picked:
            continue
        if not outcome_index.exists(run_id):
            continue
        picked.append(run_id)
        if len(picked) >= limit:
            break
    return picked


def candidates(*, since_ns: int | None = None,
               min_sample: int = MIN_SAMPLE) -> list[dict[str, Any]]:
    """四类确定性候选（按信号强度排序）。"""
    rows = _load(since_ns=since_ns)
    found: list[dict[str, Any]] = []

    by_stage: dict[str, list[dict[str, Any]]] = {}
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage = str(row.get("stage") or row.get("scope") or row.get("kind") or "(未知)")
        by_stage.setdefault(stage, []).append(row)
        shot = row.get("shot_id") or row.get("storyboard_id")
        if shot and row.get("kind") == "video":
            by_shot.setdefault(str(shot), []).append(row)

    # ① 失败聚集
    for stage, group in by_stage.items():
        failed = [r for r in group if r.get("status") == "failed"]
        if len(group) < min_sample or len(failed) < min_sample:
            continue
        rate = len(failed) / len(group)
        if rate <= 0.4:
            continue
        codes: dict[str, int] = {}
        for row in failed:
            code = str(row.get("error_code") or "(无错误码)")
            codes[code] = codes.get(code, 0) + 1
        top = sorted(codes.items(), key=lambda kv: -kv[1])[:3]
        found.append({
            "pattern": "failure_cluster",
            "key": stage,
            "kind": "failure",
            "score": rate * len(failed),
            "stats": {"stage": stage, "total": len(group), "failed": len(failed),
                      "failRate": round(rate, 4), "topErrors": top},
            "evidence": _evidence_of(failed),
        })

    # ② 分数高分离
    for stage, group in by_stage.items():
        scored = [float(r["score"]) for r in group if r.get("score") is not None]
        if len(scored) < SPLIT_SAMPLE * 2:
            continue
        low, high = _percentile(scored, 0.1), _percentile(scored, 0.9)
        if low is None or high is None or (high - low) <= SPLIT_GAP:
            continue
        best = [r for r in group if r.get("score") is not None
                and float(r["score"]) >= high][:3]
        found.append({
            "pattern": "score_split",
            "key": stage,
            "kind": "recipe",
            "score": (high - low) * len(scored) / 100,
            "stats": {"stage": stage, "samples": len(scored),
                      "p10": round(low, 4), "p90": round(high, 4),
                      "gap": round(high - low, 4),
                      "bestModels": sorted({str(r.get("model") or "") for r in best if r.get("model")})},
            "evidence": _evidence_of(best, limit=3),
        })

    # ③ 返工热点
    for shot, group in by_shot.items():
        if len(group) < REWORK_COUNT:
            continue
        found.append({
            "pattern": "rework_hotspot",
            "key": shot,
            "kind": "lesson",
            "score": len(group),
            "stats": {"shotId": shot, "runs": len(group),
                      "stages": sorted({str(r.get("stage") or "") for r in group if r.get("stage")}),
                      "scores": [r.get("score") for r in group if r.get("score") is not None]},
            "evidence": _evidence_of(group),
        })

    # ④ 成本离群
    for stage, group in by_stage.items():
        elapsed = [float(r["elapsed_ms"]) for r in group if r.get("elapsed_ms")]
        if len(elapsed) < COST_SAMPLE:
            continue
        p50, p95 = _percentile(elapsed, 0.5), _percentile(elapsed, 0.95)
        if not p50 or not p95 or p95 <= p50 * COST_RATIO:
            continue
        slow = sorted(group, key=lambda r: -(r.get("elapsed_ms") or 0))[:3]
        found.append({
            "pattern": "cost_outlier",
            "key": stage,
            "kind": "lesson",
            "score": (p95 / p50) * len(elapsed) / 100,
            "stats": {"stage": stage, "samples": len(elapsed),
                      "p50Ms": int(p50), "p95Ms": int(p95),
                      "ratio": round(p95 / p50, 2)},
            "evidence": _evidence_of(slow, limit=3),
        })

    found.sort(key=lambda item: -float(item["score"]))
    return found


# ---------------------------------------------------------------------------
# 归纳
# ---------------------------------------------------------------------------

def prompt_for(candidate: dict[str, Any]) -> str:
    """给模型的输入：**只给摘要与白名单**，不给原始日志（日志里有提示注入风险）。"""
    return (
        "你是短剧生产线的复盘员。下面是一条**已由统计确认**的现象（不是猜测）：\n"
        f"现象类型：{candidate['pattern']}\n"
        f"对象：{candidate['key']}\n"
        f"统计：{json.dumps(candidate['stats'], ensure_ascii=False)}\n"
        f"可引用的证据 id（只能引用这些，不要编造）：{candidate['evidence']}\n\n"
        "请写一条**可复用**的经验，只输出一个 JSON 对象：\n"
        '{"title": "一句话标题（≤30 字，写现象不写结论）", '
        '"body": "现象 → 可能根因 → 下一次该怎么做（≤200 字，具体到参数或步骤）", '
        '"tags": "逗号分隔的 2-3 个标签", "confidence": 0.0-1.0}\n'
        "要求：不确定就写不确定，不要编造数字与文件名；不要复述统计。"
    )


def _parse_json_block(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个完整 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _template_entry(candidate: dict[str, Any]) -> dict[str, Any]:
    """没接模型时的确定性模板。**故意写得比模型保守**（confidence 封顶 0.5）。"""
    stats = candidate["stats"]
    pattern = candidate["pattern"]
    if pattern == "failure_cluster":
        errors = ", ".join(f"{code}×{count}" for code, count in stats.get("topErrors", []))
        title = f"{stats['stage']} 失败率偏高（{stats['failed']}/{stats['total']}）"
        body = (f"近期待统计：{stats['stage']} 共 {stats['total']} 次，失败 {stats['failed']} 次"
                f"（{stats['failRate']:.0%}）。集中错误：{errors or '未归因'}。"
                "下一次执行该阶段前先核对最集中的那个错误码对应的前置条件（参数/尺寸/模型是否就绪），"
                "不要盲目重试。")
        tags = "失败率,自动蒸馏"
    elif pattern == "score_split":
        title = f"{stats['stage']} 分数分离明显（P10 {stats['p10']} / P90 {stats['p90']}）"
        body = (f"同样走 {stats['stage']}，好的与差的分差 {stats['gap']}（样本 {stats['samples']}）。"
                "把高分那批的共同参数（模型/步数/尺寸）单独记下来做默认配方，"
                "低分那批不要混进训练数据。")
        tags = "配方,分数,自动蒸馏"
    elif pattern == "rework_hotspot":
        stages = ", ".join(stats.get("stages") or [])
        title = f"镜头 {stats['shotId']} 返工 {stats['runs']} 次"
        body = (f"该镜头已跑 {stats['runs']} 次（涉及 {stages or '未知阶段'}），"
                f"分数序列 {stats.get('scores')}。"
                "先判断是提示词/参考图问题还是生成参数问题，别继续同参数硬重试；"
                "同参数重试不会让分数变好。")
        tags = "返工,镜头,自动蒸馏"
    elif pattern == "cost_outlier":
        title = f"{stats['stage']} 耗时离群（P95 是 P50 的 {stats['ratio']} 倍）"
        body = (f"{stats['stage']} 的 P50 为 {stats['p50Ms']}ms、P95 为 {stats['p95Ms']}ms"
                f"（{stats['samples']} 样本）。长尾不是模型慢就是重试叠加，"
                "先看长尾那几条的 error/retry 再决定是降步数还是加超时。")
        tags = "耗时,长尾,自动蒸馏"
    else:
        title = f"{candidate['key']} 待复盘"
        body = f"统计信号：{json.dumps(stats, ensure_ascii=False)}"
        tags = "自动蒸馏"
    return {"title": title, "body": body, "tags": tags, "confidence": 0.5}


def distill(*, candidates_list: list[dict[str, Any]] | None = None,
            llm: Callable[[str], str] | None = None,
            since_ns: int | None = None, limit: int = 5,
            dry_run: bool = False,
            writer: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """把候选写进台账（``status=proposed``，等人审或后续 support 达标）。

    ``llm`` 缺省时走模板；``llm`` 抛错只影响那一条，不中断整批。
    """
    pool = candidates_list if candidates_list is not None else candidates(since_ns=since_ns)
    pool = pool[:max(0, int(limit))]
    write = writer or knowledge_ledger.upsert

    written = rejected = failed_llm = 0
    items: list[dict[str, Any]] = []
    for candidate in pool:
        if not candidate.get("evidence"):
            items.append({"pattern": candidate.get("pattern"), "key": candidate.get("key"),
                          "ok": False, "why": "没有一条证据能在索引里核到，跳过"})
            rejected += 1
            continue

        payload: dict[str, Any] | None = None
        source_kind = "distill-rule"
        if llm is not None:
            try:
                answer = llm(prompt_for(candidate))
                payload = _parse_json_block(answer or "")
                if payload is not None:
                    source_kind = "distill-llm"
            except Exception as exc:  # noqa: BLE001 - 单条失败不影响整批
                failed_llm += 1
                payload = None
                items.append({"pattern": candidate.get("pattern"), "key": candidate.get("key"),
                              "ok": False, "why": f"模型调用失败：{exc}"})
                continue

        if payload is None:
            payload = _template_entry(candidate)
            #: 模型给了但解析不出 ⇒ 这条要如实标回模板（不假装是模型的结论）
            source_kind = "distill-rule"

        if dry_run:
            items.append({"pattern": candidate.get("pattern"), "key": candidate.get("key"),
                          "ok": True, "id": "", "why": "dry-run 未写入",
                          "sourceKind": source_kind, "preview": payload})
            continue

        result = write(
            kind=str(payload.get("kind") or candidate.get("kind") or "lesson"),
            scope=f"stage:{candidate['key']}" if candidate.get("pattern") != "rework_hotspot"
            else f"shot:{candidate['key']}",
            title=str(payload.get("title") or ""),
            body=str(payload.get("body") or ""),
            evidence=candidate["evidence"],
            tags=str(payload.get("tags") or ""),
            confidence=min(1.0, max(0.0, float(payload.get("confidence") or 0.4))),
            status="proposed",
            source_kind=source_kind,
            source_ref=str(candidate.get("pattern")),
            links=[("stage", str(candidate.get("key")))] if candidate.get("pattern") != "rework_hotspot"
            else [("shot", str(candidate.get("key")))],
            require_evidence=True,
        )
        items.append({"pattern": candidate.get("pattern"), "key": candidate.get("key"),
                      "ok": bool(result.get("ok")), "id": result.get("id"),
                      "why": result.get("why") or "", "sourceKind": source_kind})
        if result.get("ok"):
            written += 1
        else:
            rejected += 1

    return {
        "candidates": len(pool),
        "written": written,
        "rejected": rejected,
        "llmFailed": failed_llm,
        "dryRun": dry_run,
        "items": items,
    }


# ---------------------------------------------------------------------------
# lift：知识到底有没有用
# ---------------------------------------------------------------------------

def refresh_lift(*, window_ns: int = 30 * 86_400 * 1_000_000_000,
                 min_settled: int = LIFT_MIN_SETTLED,
                 deprecate_below: float = LIFT_FLOOR) -> dict[str, Any]:
    """按「注入后结果 vs 同期基线」重算每条 accepted 知识的 ``lift``。

    ⚠️ 基线必须取**同期**（同一 ``window_ns`` 内）全部 agent 跑次的成功率，
    不能拿"上个月"比 —— 题材/画风/模型一换，季节差异会被误读成"知识有效"。

    样本不足 ``min_settled`` 的条目只写 ``lift=None``（**报得出"还没结论"，
    而不是报一个 0.0 冒充"没影响"**）。
    """
    window_start = time.time_ns() - int(window_ns)
    total_agent = outcome_index.count(kind="agent", since_ns=window_start)
    ok_agent = outcome_index.count(kind="agent", status="ok", since_ns=window_start)
    baseline = (ok_agent / total_agent) if total_agent else None

    listing = knowledge_ledger.list_entries(status="accepted", limit=200)
    judged = skipped = deprecated = 0
    items: list[dict[str, Any]] = []
    for entry in listing.get("items", []):
        summary = knowledge_ledger.usage_summary(entry["id"])
        if baseline is None or not summary.get("settled") or summary["settled"] < min_settled:
            knowledge_ledger.set_lift(entry["id"], None)
            skipped += 1
            items.append({"id": entry["id"], "title": entry["title"], "lift": None,
                          "why": "样本不足或同期无基线"})
            continue
        lift = round(float(summary["okRate"]) - baseline, 4)
        knowledge_ledger.set_lift(entry["id"], lift)
        judged += 1
        action = "keep"
        if lift < deprecate_below:
            knowledge_ledger.set_status(
                entry["id"], "deprecated",
                reason=f"lift {lift} < {deprecate_below}（样本 {summary['settled']}）")
            deprecated += 1
            action = "deprecated"
        items.append({"id": entry["id"], "title": entry["title"], "lift": lift,
                      "settled": summary["settled"], "action": action})
    return {"baseline": round(baseline, 4) if baseline is not None else None,
            "agentRuns": total_agent, "judged": judged, "skipped": skipped,
            "deprecated": deprecated, "items": items}
