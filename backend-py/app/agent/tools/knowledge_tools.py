"""知识台账的 Agent 工具（读一条、写一条，2026-11-06）。

**只做两件事**，全部口径在 ``services/knowledge_ledger.py``（单一真源），本文件不重复实现：

* ``search_knowledge`` —— 读。给 Agent 一个「本仓自己跑出来的经验」的出口：
  这个题材哪个参数配得好、这类失败上次是怎么修的、哪个镜头写法被证伪过。
  与 ``search_script_library``（外部 5403 份资料，只读）**不是一回事**：
  那边是别人的书，这边是本仓自己的账。
* ``propose_knowledge`` —— 写，且**只能写 ``proposed``**。
  Agent 可以提议，不能自己拍板 —— 状态机 ``proposed → accepted → …`` 由人审推进
  （``POST /api/v1/knowledge-loop/entries/{id}/status``）。

三条硬约束都落在 service 里，工具侧只负责如实转述：

1. **证据必须可核**：``evidence`` 里每个 id 要么是 ``outcome_index`` 里真实存在的 run_id，
   要么是盘上真实存在的路径。核不过 ⇒ 整条拒收（``ok=false`` + ``why``），
   **不是**「先存下来再标记可疑」。这比在提示词里写「不要编造」有效得多。
2. **同一件事只一条**：``id = kind×scope×归一标题`` 的 hash。二次提议只 ``support+1``。
   所以模型重复提同一件事是安全的，但换个说法提同一件事会变成两条 ——
   ``title`` 要写「一次运行的结论」，不要写成一本书的目录。
3. **检索三态不可混**：``ranking`` 为 ``bm25``（≥3 字，越小越相关）/ ``like``（1–2 字中文只能走这条，
   只是包含匹配）/ ``recent``（空查询，与相关性无关）。回执原样带出，不粉饰。
"""
from __future__ import annotations

from typing import Any

from app.agent.tool import Tool, array_of, json_number, json_string, object_schema
from app.services import knowledge_ledger

__all__ = ["create_knowledge_tools"]


def create_knowledge_tools(episode_id: int | None = None,
                           drama_id: int | None = None) -> dict[str, Tool]:
    """知识台账读写工具。

    ``episode_id`` / ``drama_id`` 只用于给 ``scope`` 一个默认值（``EP<id>`` / ``DRAMA<id>``）——
    写进台账的经验天然属于某一集；不传则落 ``global``（跨项目通用成法）。
    """

    default_scope = ""
    if episode_id:
        default_scope = f"EP{episode_id}"
    elif drama_id:
        default_scope = f"DRAMA{drama_id}"

    async def search_knowledge(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        kind = str(arguments.get("kind") or "").strip().lower()
        if kind and kind not in knowledge_ledger.KINDS:
            #: 给错类型不静默忽略（那会让人以为"库里没这条经验"）：如实报可选项。
            return {"available": False, "total": 0, "items": [], "ranking": "none",
                    "hint": f"未知知识类型 {kind!r}（可选：{'/'.join(knowledge_ledger.KINDS)}）"}
        status = str(arguments.get("status") or "").strip().lower()
        if status and status not in knowledge_ledger.STATUSES:
            return {"available": False, "total": 0, "items": [], "ranking": "none",
                    "hint": f"未知状态 {status!r}（可选：{'/'.join(knowledge_ledger.STATUSES)}）"}
        try:
            limit = int(arguments.get("limit")) if arguments.get("limit") is not None else 6
        except (TypeError, ValueError):
            limit = 6
        scope = str(arguments.get("scope") or "").strip() or None

        result = knowledge_ledger.search(
            query,
            kind=kind or None,
            #: ⚠️ 不给 ``status`` 时**不预筛**：``proposed`` 的条目对 Agent 也有参考价值，
            #: 而「只在 accepted 里搜」会让冷启动阶段（一条都还没人审）变成空库。
            status=status or None,
            scope=scope,
            limit=limit,
        )
        #: bm25 分数与 like 的"包含"不同量纲 ⇒ 只回排名，不回分数（``ranking`` 原样带出）。
        return result

    async def propose_knowledge(arguments: dict[str, Any]) -> dict[str, Any]:
        kind = str(arguments.get("kind") or "").strip().lower()
        title = str(arguments.get("title") or "").strip()
        body = str(arguments.get("body") or "").strip()
        if not title or not body:
            return {"ok": False, "why": "title 与 body 都不能为空"}

        raw_evidence = arguments.get("evidence")
        if isinstance(raw_evidence, str):
            evidence: list[str] = [x.strip() for x in raw_evidence.split(",") if x.strip()]
        elif isinstance(raw_evidence, (list, tuple)):
            evidence = [str(x).strip() for x in raw_evidence if str(x).strip()]
        elif raw_evidence is None:
            evidence = []
        else:
            evidence = [str(raw_evidence).strip()]

        try:
            confidence = float(arguments.get("confidence")) \
                if arguments.get("confidence") is not None else 0.5
        except (TypeError, ValueError):
            confidence = 0.5
        #: Agent 的提议最高只到 0.6 —— 一条没经过人审、也没跑出 lift 的经验，
        #: 不该在检索里压过 ``accepted`` 的条目。
        confidence = max(0.0, min(confidence, 0.6))

        scope = str(arguments.get("scope") or "").strip() or default_scope or "global"

        result = knowledge_ledger.upsert(
            kind=kind,
            scope=scope,
            title=title,
            body=body,
            evidence=evidence,
            tags=str(arguments.get("tags") or "").strip(),
            confidence=confidence,
            #: 硬写 ``proposed``：不接受模型传状态（传了也无视）。
            status="proposed",
            source_kind="agent",
            source_ref=f"agent-tool:EP{episode_id}" if episode_id else "agent-tool",
        )
        if result.get("ok"):
            result["hint"] = (
                "已记为 proposed（待人工审）。同一件事再提一次只会 support+1，不会新建。"
                if result.get("created")
                else "同一件事已有条目，本次只累加 support。"
            )
        else:
            result["hint"] = (
                "被拒收：这条经验的可信度取决于证据。补上真实 run_id"
                "（GET /api/v1/knowledge-loop/runs 可查）或盘上真实文件路径后重提。"
            )
        return result

    return {
        "search_knowledge": Tool(
            id="search_knowledge",
            description=(
                "查**本仓自己跑出来的经验台账**（内部知识，与 search_script_library 的外部资料不同）。"
                "七类：fact（行业/作品事实）/ work（作品资料）/ recipe（成法，如某题材该用什么参数）"
                " / lesson（经验）/ failure（失败样本：现象→根因→修法）/ threshold（阈值建议）"
                " / prompt_patch（提示词补丁）。"
                "开工前先查一次，避免重复踩已知坑；hallucinate 前也先查一次，本仓可能已经有定论。"
                "回执带 ranking：bm25（≥3 字，越小越相关）/ like（1–2 字中文只能包含匹配）"
                " / recent（空查询，与相关性无关）—— 三者不可比。"
            ),
            input_schema=object_schema(
                {
                    "query": json_string(
                        "检索词。建议 3 字以上具体术语（如「口型 对不上」「8 步 过曝」）；"
                        "1–2 字只能全表包含匹配，精度差。留空＝按最近更新倒序"
                    ),
                    "kind": json_string(
                        "可选：只查某一类 —— fact/work/recipe/lesson/failure/threshold/prompt_patch"
                    ),
                    "status": json_string(
                        "可选：proposed（待审）/ accepted（已采纳，最可信）/ deprecated / rejected。"
                        "不填＝全部"
                    ),
                    "scope": json_string(
                        "可选：作用域，如 EP12 / DRAMA3 / global（global 条目在任何作用域下都会被带出）"
                    ),
                    "limit": json_number("返回条数，默认 6，最多 50"),
                },
                required=["query"],
            ),
            execute=search_knowledge,
        ),
        "propose_knowledge": Tool(
            id="propose_knowledge",
            description=(
                "把本轮跑出来的**可复用经验**记进台账（写入状态固定为 proposed，待人工审）。"
                "什么时候该提：你确认了一个可复现的因果关系 —— 某参数/某写法/某步骤会稳定导致某结果。"
                "什么时候不该提：一次性的偶然现象、你自己的推测、别处读来的二手说法"
                "（后者该走 fact/work 且要带来源）。"
                "**必须带证据**：真实 run_id 或盘上真实文件路径，核不过整条拒收。"
                "反例也别浪费 —— 失败样本（现象→根因→修法）是台账里最值钱的一类。"
            ),
            input_schema=object_schema(
                {
                    "kind": json_string(
                        "知识类型：fact / work / recipe / lesson / failure / threshold / prompt_patch"
                    ),
                    "title": json_string(
                        "一句话结论（会被归一成 id，所以同义不同说法会变成两条）——"
                        "写「8 步档在夜景会过曝」，不要写「关于参数的思考」"
                    ),
                    "body": json_string(
                        "正文：现象／条件／根因／怎么用。要能被下一个人直接执行，不要写成感想"
                    ),
                    "evidence": array_of(
                        json_string("一条 run_id 或盘上路径"),
                        "证据列表（真 run_id 或真实文件路径）。fact/work 可留空（外部来源）",
                    ),
                    "tags": json_string("可选：逗号分隔标签，如「夜景,过曝,krea2」"),
                    "confidence": json_number(
                        "可选：0–0.6（工具侧封顶 0.6，未经人审的经验不该压过 accepted 条目）"
                    ),
                    "scope": json_string(
                        f"可选：作用域，留空默认 {default_scope or 'global'}"
                    ),
                },
                required=["kind", "title", "body"],
            ),
            execute=propose_knowledge,
        ),
    }
