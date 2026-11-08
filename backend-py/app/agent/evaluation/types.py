"""评测闭环的数据契约 —— 与 ``evaluation/types.ts``（151 行）对齐。

⚠️ **有意差异**：``types.ts`` **只有类型、没有运行期值**（TS 的 interface 编译后不存在）。
Python 没有对应物，所以本模块把那些 interface **记录成文档 + 键集合**，供两处使用：

* ``catalog`` / ``scorer`` 读取基准 JSON 时知道字段名；
* 自检可以拿键集合**校验 4 个基准 case 文件**的形状（TS 侧靠类型系统，运行期没人校验）。

设计思想（对齐 PenguinHarness 的 benchmark-design）：

* ``statement``（**公开**）：喂给 Agent 的输入（剧本 + 角色 + 场景）；
* ``rubric``（**私密**）：金标准答案 + 确定性评分规则，**优化器不得读取**。
"""
from __future__ import annotations

__all__ = [
    "AGENT_BY_KIND",
    "CASE_KEYS",
    "LITERAL_KINDS",
    "RUBRIC_KEYS_BY_KIND",
    "STATEMENT_KEYS_BY_KIND",
]

#: ``case.kind`` → Agent 类型（CLI 与 HTTP 路由共用，避免重复映射）
AGENT_BY_KIND: dict[str, str] = {
    "storyboard": "storyboard_breaker",
    "extractor": "extractor",
    "script_rewriter": "script_rewriter",
    "voice_assigner": "voice_assigner",
    #: 2026-11-07 补（知识闭环缺口 8）：第 5 种可评对象 —— **图片提示词层**。
    #: 它正好卡在"产出层"与"成片层"之间：提示词写砸了，成片评测那一维会替它挨罚，
    #: 但那时已经花掉一次视频生成的钱。所以这一层必须自己有基准。
    "grid_prompt": "grid_prompt_generator",
}

#: ``kind`` 的**字面量**联合（TS 里是 ``'storyboard' | 'extractor' | ...``）。
#: 与 ``AGENT_BY_KIND`` 的键**必须一致**（自检会比对，防止加了一种忘了另一种）。
LITERAL_KINDS: tuple[str, ...] = (
    "storyboard",
    "extractor",
    "script_rewriter",
    "voice_assigner",
    "grid_prompt",
)

#: 每个 case 的顶层键（``BenchmarkCase`` 的公共部分）。
CASE_KEYS: tuple[str, ...] = ("id", "kind", "statement", "rubric")

#: ``statement``（**公开**输入）的键 —— 按 kind 分。
STATEMENT_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "storyboard": ("script", "characters", "scenes"),
    "extractor": ("script",),
    "script_rewriter": ("content",),
    "voice_assigner": ("script", "characters"),
    #: ``shots`` 是**要送进宫格的分镜**（``[{number,title,scene,action,dialogue}]``），
    #: 由 ``seed_case`` 落成真实 ``storyboards`` 行再把 id 交给 Agent —— 宫格 Agent 的
    #: 工具（``read_shots_for_grid``）只认 id，不认 JSON 片段。
    "grid_prompt": ("shots", "rows", "cols", "mode"),
}

#: ``rubric``（**私密**金标准）的键 —— 按 kind 分。
RUBRIC_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "storyboard": ("minShots", "maxShots", "requiredFields", "videoPromptTags",
                   "durationRange", "titleLengthRange"),
    "extractor": ("goldenCharacters", "goldenScenes", "minAppearanceLength",
                  "minPromptLength"),
    "script_rewriter": ("minScenes", "forbiddenCameraWords"),
    "voice_assigner": ("legalVoiceIds", "requireReason"),
    #: 宫格的金标准全是**可数/可测**的：格数对不对、有没有中文（英文提示词硬要求）、
    #: 有没有模板残留（``[...]`` / ``TODO`` / ``<...>``）、格间是否互相复制、正文够不够长。
    "grid_prompt": ("rows", "cols", "minGridPromptLength", "minCellPromptLength",
                    "forbiddenTags"),
}

#: ``ScoreReport.dimensions[]`` 的键。
SCORE_DIMENSION_KEYS: tuple[str, ...] = ("name", "score", "max", "detail")

#: ``ScoreReport`` 的键。
SCORE_REPORT_KEYS: tuple[str, ...] = ("caseId", "kind", "dimensions", "total")
