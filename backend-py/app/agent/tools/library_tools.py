"""外部剧本/编剧知识库的 Agent 工具（2026-11-04）。

**只做一件事**：把 ``services/script_library.py`` 的检索结果按模型的用量裁好递过去 ——
分词、打分、路径重映射、盘点全部在 service 里（单一真源），本文件不重复实现任何口径。

为什么挂在编剧与分镜两个 Agent 上：

* 编剧 Agent 改台词/结构时，要能查到「三幕结构怎么落」「这类桥段别人怎么写」——
  ``methods`` 那边是 5403 部方法书与类型片剧本正文（含《故事》这类方法论原书）。
* 分镜 Agent 定镜头语言、表演、打戏节奏时，``playbook`` 那边是 137 份专项（表演/微表情、
  打戏、镜头语法），比让它自由发挥稳定得多。

回执里带了 ``sources``（逐源 available/ranking/matched/count）与 ``corpus``（库多大）：
**「命中 33 段、给你 3 段」和「库里就 3 段」必须能被模型自己区分开**，
否则它会拿一条孤例当通行写法（与 ``corpus_tools`` 的语料盘点同一条纪律）。
"""
from __future__ import annotations

from typing import Any

from app.agent.tool import Tool, json_number, json_string, object_schema
from app.services import script_library

__all__ = ["create_library_tools"]


def create_library_tools() -> dict[str, Tool]:
    """知识库检索工具（编剧/分镜 Agent 共用）。"""

    async def search_script_library(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        source = str(arguments.get("source") or "all").strip().lower()
        if source not in ("all", "methods", "playbook"):
            source = "all"
        limit = arguments.get("limit")
        try:
            limit = int(limit) if limit is not None else 4
        except (TypeError, ValueError):
            limit = 4
        limit = max(1, min(limit, 8))
        path_contains = arguments.get("path_contains")
        path_contains = str(path_contains).strip() if path_contains else None

        #: 目录维度（国家/类型/奖项/权限）只对 methods 生效；空值一律不进 filters，
        #: 免得把「没填」当成「筛这个」→ 结果被静默筛空。
        filters = {}
        for key in ("country", "genre", "category", "docType", "permission", "festival"):
            value = arguments.get(key)
            value = str(value).strip() if value else ""
            if value:
                filters[key] = value

        #: 反之 ``skill`` 只筛 playbook（skills/<名>/ 那一棵）；给错源不会静默忽略 ——
        #: service 会把该源标成 filtersApplied=false 并附一句 hint。
        skill = arguments.get("skill")
        skill = str(skill).strip() if skill else ""

        return script_library.search(
            query,
            source=source,
            limit=limit,
            path_contains=path_contains,
            filters=filters or None,
            skill=skill or None,
        )

    return {
        "search_script_library": Tool(
            id="search_script_library",
            description=(
                "查本地剧本/编剧知识库（第三方资料，只读）。两源："
                "methods=5403 部编剧方法书与类型片剧本正文（《故事》等原书、爆款短剧合集），"
                "playbook=137 份专项方法 2332 小节（打戏招式库 77、大招九系属性 14、"
                "镜头语法与运镜、造型库、特效库、情绪实例）。"
                "写或改剧本时查「结构怎么落、桥段怎么写」，定镜头语言/表演时查专项写法，"
                "写人物/场景/道具提示词时查造型库与提示词库（path_contains=\"library/general\"）。"
                "返回片段已截断到 400 字，并附 corpus 盘点与 sources 计数——"
                "参考是写法与结构，**不要搬运原文句子**进剧本（合规红线）。"
            ),
            input_schema=object_schema(
                {
                    "query": json_string(
                        "检索词，建议 3 字以上、用具体术语，如「激励事件」「三幕结构 中点反转」"
                        "「微表情 克制 近景」；1–2 字词只能靠全表模糊匹配，精度差"
                    ),
                    "source": json_string(
                        "来源：all（默认，两源都查）/ methods（方法书与剧本正文）"
                        " / playbook（专项方法文本）"
                    ),
                    "limit": json_number("每个来源返回几条，默认 4，最多 8"),
                    "path_contains": json_string(
                        "可选：只在这些路径下找，如「08-编剧方法书籍」「library/combat」"
                        "（打戏 77）「library/ultimate」（大招 14）「library/general」"
                    ),
                    "country": json_string("可选：国家地区，如「韩国」「美国,日本」（只作用于 methods）"),
                    "genre": json_string("可选：类型/二级分类，如「犯罪」「科幻」"),
                    "category": json_string("可选：一级分类，如「类型片」「编剧方法」"),
                    "docType": json_string("可选：资料类型，如「剧本」「方法书」"),
                    "permission": json_string("可选：权限状态，如「可引用」"),
                    "festival": json_string("可选：电影节，如「戛纳」「柏林」（含关系标签）"),
                    "skill": json_string(
                        "可选：素材包名（只作用于 playbook）—— 现存两个：film-cinedance"
                        "（镜头调度与运镜）/ film-expert-bridge（打戏＋大招＋通用方法论）。"
                        "只留一个名字；不确定就留空（传别的名字会查不到，不是资料没有）"
                    ),
                },
                required=["query"],
            ),
            execute=search_script_library,
        ),
    }
