"""语料检索工具（移植自 ``agents/tools/corpus-tools.ts``，148 行）—— 让 Agent 能「找相似镜头」。

数据源：``data/prompt-corpus/_normalized/prompts.jsonl``（由 ``backend-py/app/scripts/corpus/`` 产出）。

⚠️ 该目录**已 gitignore**（第三方语料不进仓库）⇒ 新克隆的仓库里文件不存在，所以本工具
**必须在语料缺失时优雅降级**：返回 ``available=false`` + 提示，而不是抛错 ——
否则「没采语料」会直接变成「Agent 跑不动」。

⚠️ 打分口径与 ``backend-py/app/scripts/corpus/search.py`` **保持一致**（2-gram + 位置加权）：改这里要同步改那边，
否则「命令行验证有效」与「Agent 实际效果」会分叉。

⚠️ 返回的 ``prompt_excerpt`` **截断到 400 字符**且压平空白：参考是给模型看的，过长会挤占上下文；
要全文自行按 id 到语料里取。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.agent.tool import Tool, json_number, json_string, object_schema

__all__ = ["CORPUS_JSONL", "load_corpus", "tokenize", "create_corpus_tools"]

#: 语料文件（与 Node 同法：**基于仓库根**解析，兼容任意 cwd / Docker）
CORPUS_JSONL = PROJECT_ROOT / "data" / "prompt-corpus" / "_normalized" / "prompts.jsonl"

#: 拉丁词（≥2 字符，含 . _ -）
_LATIN_WORD = re.compile(r"[a-z0-9][a-z0-9._-]+")
#: 非 CJK 一律当分隔符
_NON_CJK = re.compile(r"[^\u4e00-\u9fff]+")
_CJK_RUN = re.compile(r"\u4e00-\u9fff")

#: 懒加载 + 进程内缓存：语料 ~2.5MB 文本，启动时读一次即可（缺失则缓存 None，不反复 stat）
_cache: list[dict[str, Any]] | None | bool = False  # False = 未加载


def load_corpus() -> list[dict[str, Any]] | None:
    """读语料（缺失/解析失败返回 None，并**缓存**该结果）。"""
    global _cache
    if _cache is not False:
        return _cache  # type: ignore[return-value]
    if not Path(CORPUS_JSONL).exists():
        _cache = None
        return None
    try:
        rows = [
            json.loads(line)
            for line in Path(CORPUS_JSONL).read_text(encoding="utf-8").split("\n")
            if line
        ]
        _cache = rows
    except Exception:  # noqa: BLE001 —— 与 TS 的 catch 等价
        _cache = None
    return _cache  # type: ignore[return-value]


def tokenize(text: str) -> list[str]:
    """查询分词：拉丁词按词边界；**CJK 取 2-gram**（中文无空格，2-gram 最省事且够用）。

    去重保序（JS 的 ``Set``），单字 CJK 段也保留。
    """
    out: dict[str, None] = {}
    lower = str(text).lower()
    for word in _LATIN_WORD.findall(lower):
        out[word] = None
    for segment in _NON_CJK.sub(" ", lower).split():
        if not segment:
            continue
        if len(segment) == 1:
            out[segment] = None
        for i in range(len(segment) - 1):
            out[segment[i:i + 2]] = None
    return list(out)


_NOT_AVAILABLE_HINT = (
    "本地语料未安装（data/prompt-corpus/ 已 gitignore，不在仓库内）。"
    "请先执行 python backend-py/app/scripts/corpus/fetch_raw.py 采集、python backend-py/app/scripts/corpus/normalize.py 归一化。"
    "在语料就绪前，请改用 SKILL 里已有的词表与锚点，不要假设能检索到参考。"
)

_RESULT_NOTE = (
    "以上为**参考写法**，不是可直接照抄的剧本内容：请只借鉴镜头/光线/结构手法，"
    "不要搬运原文（合规红线见 skills/video-prompt-library 6.9）。"
)


def create_corpus_tools() -> dict[str, Tool]:
    """工厂（本工具**不依赖 episodeId/dramaId**）。"""

    async def search_reference_prompts(arguments: dict[str, Any]) -> dict[str, Any]:
        rows = load_corpus()
        if rows is None:
            return {"available": False, "results": [], "hint": _NOT_AVAILABLE_HINT}

        tokens = tokenize(arguments.get("query") or "")
        if not tokens:
            return {"available": True, "results": [], "hint": "检索词为空。"}

        top = arguments.get("top")
        limit = min(max(top if top is not None else 3, 1), 5)

        scored: list[tuple[dict[str, Any], int, float]] = []
        for row in rows:
            fields = (
                (5, (row.get("title") or "").lower()),
                (3, " ".join(row.get("tags") or []).lower()),
                (2, (row.get("category") or "").lower()),
                (2, (row.get("summary_zh") or "").lower()),
                (1, (row.get("prompt") or "").lower()),
            )
            score = 0
            hits = 0
            for token in tokens:
                got = 0
                for weight, haystack in fields:
                    if token in haystack:
                        got += weight
                if got:
                    hits += 1
                score += got
            if score > 0:
                scored.append((row, score, hits / len(tokens)))

        scored.sort(key=lambda item: -item[1])  # JS 与 Python 的 sort 都是稳定排序

        return {
            "available": True,
            "total_matched": len(scored),
            "results": [
                {
                    "score": score,
                    "coverage": round(coverage, 2),  # `Number(cover.toFixed(2))`
                    "source": row.get("source"),
                    "license": row.get("license"),
                    "lang": row.get("lang"),
                    "title": row.get("title"),
                    "category": row.get("category"),
                    "tags": (row.get("tags") or [])[:8],
                    "mode": row.get("mode"),
                    "duration": row.get("duration"),
                    "aspect": row.get("aspect"),
                    # 截断 + 压平空白（原 TS：replace(/\s+/g,' ').slice(0,400)）
                    "prompt_excerpt": re.sub(r"\s+", " ", row.get("prompt") or "")[:400],
                }
                for row, score, coverage in scored[:limit]
            ],
            "note": _RESULT_NOTE,
        }

    return {
        "search_reference_prompts": Tool(
            id="search_reference_prompts",
            description=(
                "在本地提示词语料库（约 9000 条真实视频提示词，含 Seedance 短剧/电影级写法）中"
                "检索与当前镜头相似的参考写法。当你需要具体可抄的运镜/光线/一致性写法，"
                "或想让画面描述更专业时调用；返回若干条参考提示词（已截断）。"
            ),
            input_schema=object_schema(
                {
                    "query": json_string("检索词，建议用中文关键词组合，例如「雨夜 霓虹 街道 特写 手持」"),
                    "top": json_number("返回条数，默认 3，最多 5"),
                },
                required=["query"],
            ),
            execute=search_reference_prompts,
        ),
    }
