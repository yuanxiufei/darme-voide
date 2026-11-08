"""确定性评分器 —— 与 ``evaluation/scorer.ts``（327 行）对齐。**纯函数，不依赖 LLM / DB**。

把「分镜拆得好不好」「角色提取全不全」变成可计算的分数。所有维度都来自 Agent 提示词里
**明确要求**的规则 ⇒ 改提示词，分数就跟着动，因此分数能客观反映提示词质量。

⚠️ 五处**必须位精对齐**的地方（差一点分数就不可比）：

1. ``round(n) = Math.round(n * 10) / 10`` —— 用 **``js_round``**（JS 的 ``Math.round`` 是
   「半数向 +∞」），Python 内置 ``round`` 是**银行家舍入**（0.5 会往偶数靠）；
2. ``toFixed(0)`` 的百分比同理：JS 半数**远离零**，Python 的 ``f"{x:.0f}"`` 是半数取偶
   ⇒ 这里先用 ``js_round`` 再转整；
3. ``isFilled`` 的**五分类**：``null/undefined`` 假、字符串看 ``trim()`` 后非空、
   数字看 ``Number.isFinite``、数组看**长度非零**、其余（对象）**一律算填了**；
4. 五个评分器返回的 ``caseId`` 都是**空串**（由 ``evaluator`` 事后填）—— 别"顺手"塞 case id；
   第五个 ``score_grid_prompts``（宫格提示词）是本仓新增，TS 侧没有对拍对象：它的判据不求"像 TS"，
   只求"能判死真问题"（长度 / 格数 / 中文占比 / 占位符 / 格间复制，五维全硬判据）；
5. ``scoreVoiceAssignment`` 的「角色 ID 合法无重复」会算出**可能为负**的比率
   （``in_legal - dup`` 可小于 0）⇒ 必须 ``max(0, min(1, rate))`` **夹紧**。

维度名与 ``detail`` 文案都是**用户可见**的评测报告内容，逐字保真。
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from app.core.response import js_round

__all__ = [
    "score_extraction",
    "score_grid_prompts",
    "score_script_rewrite",
    "score_storyboards",
    "score_voice_assignment",
]

#: 宫格评分里"中文"的判据（CJK 汉字 + 假名）。英文提示词是硬要求 ——
#: 混进中文会让出图侧的中文渲染变成乱码字块，而 Agent 自己不会报错。
_CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

#: 中文占比阈值：宫格 Agent 的系统提示词**自己规定**要把位置写成「格1」、参考图写成「图片1」、
#: 骨架里还有「首帧 / 尾帧」这类中文标签 ⇒ 按「含中文即扣」判，会把**完全合规**的产出判成不合格。
#: 标签在一条 60 字符的提示词里占个位数百分比，而真写成中文散文一定在 40% 以上 ⇒
#: 判据用**占比**而不是"有没有"（15% 这条线留了三倍余量，两边都不擦边）。
_CJK_RATIO_MAX = 0.15


def _cjk_ratio(text: str) -> float:
    """CJK 字符占比（空文本按 0 —— 空不是"中文太多"，长度维会单独罚它）。"""
    body = str(text or "")
    if not body:
        return 0.0
    return len(_CJK.findall(body)) / len(body)


def _round1(value: float) -> float:
    """``Math.round(n * 10) / 10``（一位小数，半数向 +∞）。"""
    return js_round(value * 10) / 10


def _pct(ratio: float) -> str:
    """``(ratio * 100).toFixed(0)`` —— 取整后转字符串（半数**远离零**）。"""
    return str(js_round(ratio * 100))


def _is_filled(value: Any) -> bool:
    """字段是否「填了有意义的值」（对齐 TS ``isFilled``）。"""
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, bool):  # ⚠️ bool 是 int 的子类，先拦掉（TS 里 typeof true !== 'number'）
        return True
    if isinstance(value, (int, float)):
        return _is_finite(value)
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _scores(dims: list[dict[str, Any]]) -> float:
    return _round1(sum(dim["score"] for dim in dims))


def _dim(name: str, score: float, maximum: int, detail: str) -> dict[str, Any]:
    return {"name": name, "score": score, "max": maximum, "detail": detail}


def _report(kind: str, dims: list[dict[str, Any]]) -> dict[str, Any]:
    """⚠️ ``caseId`` 恒为**空串**（由 evaluator 填）。"""
    return {"caseId": "", "kind": kind, "dimensions": dims, "total": _scores(dims)}


# ===========================================================================
# 分镜评分（6 维 / 100 分）
# ===========================================================================


def score_storyboards(
    shots: Sequence[dict[str, Any]],
    rubric: dict[str, Any],
    legal: dict[str, set],
) -> dict[str, Any]:
    """分镜评分。``legal`` = seed 时产生的合法 ID 集合（用于校验绑定是否凭空捏造）。"""
    dims: list[dict[str, Any]] = []
    count = len(shots)
    required: list[str] = list(rubric.get("requiredFields") or [])
    tags: list[str] = list(rubric.get("videoPromptTags") or [])
    duration_range = rubric.get("durationRange") or [0, 0]
    title_range = rubric.get("titleLengthRange") or [0, 0]

    # 1. 镜头数量（10 分）
    in_range = rubric["minShots"] <= count <= rubric["maxShots"]
    dims.append(_dim(
        "镜头数量合理", 0 if count == 0 else (10 if in_range else 5), 10,
        f"{count} 个镜头（期望 {rubric['minShots']}~{rubric['maxShots']} 之间）",
    ))

    # 2. 核心字段完整率（30 分）
    if required:
        per_shot = [len([f for f in required if _is_filled(shot.get(f))]) / len(required)
                    for shot in shots]
    else:
        per_shot = []
    avg = (sum(per_shot) / count) if count else 0
    missing = [f for f in required if any(not _is_filled(shot.get(f)) for shot in shots)]
    missing_text = f"，常缺字段：{'/'.join(missing)}" if missing else ""
    dims.append(_dim(
        "核心字段完整率", _round1(avg * 30), 30,
        f"平均覆盖率 {_pct(avg)}%{missing_text}" if count else "无分镜",
    ))

    # 3. video_prompt 标记完整（20 分）
    if tags:
        per_shot_tags = [
            len([t for t in tags if t in (shot.get("video_prompt") or "")]) / len(tags)
            for shot in shots
        ]
    else:
        per_shot_tags = []
    tag_avg = (sum(per_shot_tags) / count) if count else 0
    dims.append(_dim(
        "video_prompt 标记", _round1(tag_avg * 20), 20,
        f"标记 {' '.join(tags)} 平均覆盖 {_pct(tag_avg)}%",
    ))

    # 4. duration 范围（10 分）
    ok_duration = len([
        shot for shot in shots
        if shot.get("duration") is not None
        and duration_range[0] <= shot["duration"] <= duration_range[1]
    ])
    dims.append(_dim(
        "duration 范围", _round1((ok_duration / count) * 10) if count else 0, 10,
        f"{ok_duration}/{count} 个镜头在 {duration_range[0]}~{duration_range[1]} 秒",
    ))

    # 5. title 长度（10 分）
    ok_title = len([
        shot for shot in shots
        if (shot.get("title") or "").strip()
        and title_range[0] <= len((shot["title"] or "").strip()) <= title_range[1]
    ])
    dims.append(_dim(
        "title 长度", _round1((ok_title / count) * 10) if count else 0, 10,
        f"{ok_title}/{count} 个镜头标题在 {title_range[0]}~{title_range[1]} 字",
    ))

    # 6. 角色/场景绑定合法（20 分）—— 防止凭空捏造 ID
    legal_scene_ids = legal.get("sceneIds") or set()
    legal_character_ids = legal.get("characterIds") or set()
    ok_binding = 0
    for shot in shots:
        scene_ok = shot.get("scene_id") is None or shot["scene_id"] in legal_scene_ids
        character_ids = shot.get("character_ids") or []
        chars_ok = (not character_ids
                    or all(cid in legal_character_ids for cid in character_ids))
        if scene_ok and chars_ok:
            ok_binding += 1
    dims.append(_dim(
        "角色/场景绑定合法", _round1((ok_binding / count) * 20) if count else 0, 20,
        f"{ok_binding}/{count} 个镜头 scene_id/character_ids 均来自上下文合法集合",
    ))

    return _report("storyboard", dims)


# ===========================================================================
# 提取评分（5 维 / 100 分）
# ===========================================================================


def _unique_trimmed(values: Iterable[Any]) -> set[str]:
    """``new Set(list.map(x => (x||'').trim()).filter(Boolean))`` —— 去重 + 去空。"""
    return {str(value).strip() for value in values if str(value or "").strip()}


def score_extraction(
    characters: Sequence[dict[str, Any]],
    scenes: Sequence[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """提取评分（角色/场景的召回率与精确率各占一半，外加外貌完整度）。"""
    dims: list[dict[str, Any]] = []

    golden_chars: list[str] = list(rubric.get("goldenCharacters") or [])
    extracted_names = _unique_trimmed(c.get("name") for c in characters)
    extracted_locs = _unique_trimmed(s.get("location") for s in scenes)
    golden_locs = [s["location"] for s in (rubric.get("goldenScenes") or [])]

    # 1. 角色召回率（35 分）
    recalled = len([g for g in golden_chars if g in extracted_names])
    rate = (recalled / len(golden_chars)) if golden_chars else 0
    missed = "、".join([g for g in golden_chars if g not in extracted_names]) or "无"
    dims.append(_dim(
        "角色召回率", _round1(rate * 35), 35,
        f"{recalled}/{len(golden_chars)}（漏提：{missed}）",
    ))

    # 2. 角色精确率（25 分）—— 防止多提/幻觉角色
    precise = len([n for n in extracted_names if n in golden_chars])
    rate = (precise / len(extracted_names)) if extracted_names else 0
    extra = "、".join([n for n in extracted_names if n not in golden_chars]) or "无"
    dims.append(_dim(
        "角色精确率", _round1(rate * 25), 25,
        f"{precise}/{len(extracted_names)}（多提：{extra}）",
    ))

    # 3. 场景召回率（15 分）
    recalled_scene = len([g for g in golden_locs if g in extracted_locs])
    rate = (recalled_scene / len(golden_locs)) if golden_locs else 0
    missed_scene = "、".join([g for g in golden_locs if g not in extracted_locs]) or "无"
    dims.append(_dim(
        "场景召回率", _round1(rate * 15), 15,
        f"{recalled_scene}/{len(golden_locs)}（漏提：{missed_scene}）",
    ))

    # 4. 场景精确率（10 分）
    precise_scene = len([loc for loc in extracted_locs if loc in golden_locs])
    rate = (precise_scene / len(extracted_locs)) if extracted_locs else 0
    extra_scene = "、".join([loc for loc in extracted_locs if loc not in golden_locs]) or "无"
    dims.append(_dim(
        "场景精确率", _round1(rate * 10), 10,
        f"{precise_scene}/{len(extracted_locs)}（多提：{extra_scene}）",
    ))

    # 5. 外貌描述完整（15 分）—— 分母是**角色条数**（不是去重后的名字数）
    min_len = rubric.get("minAppearanceLength") or 0
    ok_appearance = len([
        c for c in characters if len(str(c.get("appearance") or "").strip()) >= min_len
    ])
    rate = (ok_appearance / len(characters)) if characters else 0
    dims.append(_dim(
        "外貌描述完整", _round1(rate * 15), 15,
        f"{ok_appearance}/{len(characters)} 个角色外貌描述 ≥ {min_len} 字",
    ))

    return _report("extractor", dims)


# ===========================================================================
# 剧本改写评分（5 维 / 100 分）
# ===========================================================================

#: 场景头：``## S01 | 内景 · 地点 | 时间段``（编号 + 内外景·地点 + 时间段三段）
_SCENE_HEADER = re.compile(r"##\s*S(\d+)\s*\|\s*[^|\n]*\|\s*[^|\n]*")

#: 场景头里的编号（``S01`` → 1）
_SCENE_HEADER_NO = re.compile(r"S(\d+)", re.IGNORECASE)

#: 对白行：``角色名：（动作）台词``
_DIALOGUE_LINE = re.compile(r"^[^\n#|]{1,12}[：:]\s*(?:（[^）]*）)?[^\n]", re.MULTILINE)


def score_script_rewrite(content: str, rubric: dict[str, Any]) -> dict[str, Any]:
    """剧本改写评分（格式规范 / 编号连续 / 对白格式 / 禁用镜头语言）。"""
    dims: list[dict[str, Any]] = []
    text = content or ""

    headers = [match.group(0) for match in _SCENE_HEADER.finditer(text)]
    scene_numbers: list[int] = []
    for header in headers:
        found = _SCENE_HEADER_NO.search(header)
        if found is None:
            continue
        number = int(found.group(1))
        scene_numbers.append(number)

    # 1. 剧本已保存（20 分）
    length = len(text.strip())
    dims.append(_dim(
        "剧本已保存", 0 if length == 0 else (20 if length >= 100 else 10), 20,
        f"剧本长度 {length} 字",
    ))

    # 2. 场景头格式规范（25 分）
    min_scenes = rubric.get("minScenes") or 0
    rate = 1 if len(headers) >= min_scenes else len(headers) / max(min_scenes, 1)
    dims.append(_dim(
        "场景头格式规范", _round1(rate * 25), 25,
        f"{len(headers)} 个场景头（期望 ≥ {min_scenes}）",
    ))

    # 3. 场景编号连续（15 分）—— 排序后与 1..N 逐一比对
    ordered = sorted(scene_numbers)
    consecutive = len([n for index, n in enumerate(ordered) if n == index + 1])
    rate = (consecutive / len(ordered)) if ordered else 0
    dims.append(_dim(
        "场景编号连续", _round1(rate * 15), 15,
        f"{consecutive}/{len(ordered)} 个场景编号从 S01 连续递增",
    ))

    # 4. 对白格式（20 分）
    dialogue_lines = [match.group(0) for match in _DIALOGUE_LINE.finditer(text)]
    dims.append(_dim(
        "对白格式", 20 if dialogue_lines else 0, 20,
        f"{len(dialogue_lines)} 行对白",
    ))

    # 5. 无镜头语言（20 分）—— 景别/运镜等属于分镜步骤，不应出现在剧本
    hits = [word for word in (rubric.get("forbiddenCameraWords") or []) if word in text]
    dims.append(_dim(
        "无镜头语言", 0 if hits else 20, 20,
        f"出现违禁词：{'/'.join(hits)}" if hits else "无景别/运镜等镜头语言",
    ))

    return _report("script_rewriter", dims)


# ===========================================================================
# 音色分配评分（4 维 / 100 分）
# ===========================================================================


def score_voice_assignment(
    assignments: Sequence[dict[str, Any]],
    rubric: dict[str, Any],
    legal: dict[str, set],
) -> dict[str, Any]:
    """音色分配评分（覆盖 / 音色合法 / 理由 / 角色 ID 合法无重复）。"""
    dims: list[dict[str, Any]] = []
    character_ids = legal.get("characterIds") or set()
    total = len(character_ids)
    assigned_ids = {a.get("character_id") for a in assignments
                    if a.get("character_id") is not None}

    # 1. 角色覆盖（40 分）
    covered = len([cid for cid in character_ids if cid in assigned_ids])
    rate = (covered / total) if total else 0
    dims.append(_dim(
        "角色覆盖", _round1(rate * 40), 40, f"{covered}/{total} 个角色已分配音色",
    ))

    # 2. 音色 ID 合法（30 分）—— 必须来自 list_voices 的音色库
    legal_voices = set(rubric.get("legalVoiceIds") or [])
    ok_voice = len([a for a in assignments if a.get("voice_id") and a["voice_id"] in legal_voices])
    rate = (ok_voice / len(assignments)) if assignments else 0
    dims.append(_dim(
        "音色 ID 合法", _round1(rate * 30), 30,
        f"{ok_voice}/{len(assignments)} 条分配的 voice_id 在音色库中",
    ))

    # 3. 分配理由说明（15 分）—— 不要求就直接给满分
    if not rubric.get("requireReason"):
        dims.append(_dim("分配理由说明", 15, 15, "不要求理由（跳过）"))
    else:
        with_reason = len([a for a in assignments if str(a.get("reason") or "").strip()])
        rate = (with_reason / len(assignments)) if assignments else 0
        dims.append(_dim(
            "分配理由说明", _round1(rate * 15), 15,
            f"{with_reason}/{len(assignments)} 条分配附理由",
        ))

    # 4. 角色 ID 合法无重复（15 分）—— ⚠️ 比率可能为负 ⇒ 必须夹紧到 [0, 1]
    in_legal = len([a for a in assignments
                    if a.get("character_id") is not None and a["character_id"] in character_ids])
    dup_count = len(assignments) - len(assigned_ids)
    rate = ((in_legal - dup_count) / len(assignments)) if assignments else 0
    clamped = max(0.0, min(1.0, rate))
    dims.append(_dim(
        "角色 ID 合法无重复", _round1(clamped * 15), 15,
        f"{in_legal}/{len(assignments)} 条 character_id 合法，重复 {dup_count} 条",
    ))

    return _report("voice_assigner", dims)


# ===========================================================================
# 宫格提示词评分（5 维 / 100 分，2026-11-07 补 —— 知识闭环缺口 8）
# ===========================================================================


def score_grid_prompts(payload: dict[str, Any] | None, rubric: dict[str, Any]) -> dict[str, Any]:
    """宫格提示词评分。

    ⚠️ 五个维度**全部是硬判据**（长度 / 格数 / 中文占比 / 是否残留模板 / 格间是否复制），
    没有一条是"看着像不像好提示词"✨ —— 因为那类判据在本仓没有金标准，
    打出来的分会看着很专业、其实不可复核（缺口 8 的教训就是这么来的）。

    ``payload`` 用的是**生产同一套**解析结果（``services/grid_payload.find_grid_payload``）；
    给 ``None`` ＝ Agent 压根没产出可识别载荷 ⇒ 五维全 0（不是"跳过"✓）。
    """
    payload = payload if isinstance(payload, dict) else {}
    grid_prompt = str(payload.get("grid_prompt") or "")
    cells = [item for item in (payload.get("cell_prompts") or []) if isinstance(item, dict)]
    cell_texts = [str(item.get("prompt") or "") for item in cells]

    dims: list[dict[str, Any]] = []

    # 1. 提示词长度（25 分）—— 整体与单格各占一半份额；不足按比例给（不四舍五入到 0/满）
    min_grid = int(rubric.get("minGridPromptLength") or 120)
    min_cell = int(rubric.get("minCellPromptLength") or 40)
    grid_len = len(grid_prompt.strip())
    grid_rate = 1.0 if grid_len >= min_grid else (grid_len / min_grid if min_grid else 0.0)
    short_cells = [text for text in cell_texts if len(text.strip()) < min_cell]
    cell_rate = ((len(cell_texts) - len(short_cells)) / len(cell_texts)) if cell_texts else 0.0
    rate = (max(0.0, min(1.0, grid_rate)) + max(0.0, min(1.0, cell_rate))) / 2
    dims.append(_dim(
        "提示词长度", _round1(rate * 25), 25,
        f"grid_prompt {grid_len} 字符（期望 ≥ {min_grid}）；"
        f"{len(cell_texts) - len(short_cells)}/{len(cell_texts)} 格 ≥ {min_cell} 字符",
    ))

    # 2. 格数吻合（25 分）—— 多了少了都扣（多余格意味着模型在凑数）
    rows = int(rubric.get("rows") or 0)
    cols = int(rubric.get("cols") or 0)
    want = rows * cols
    got = len(cells)
    matched = (min(got, want) / max(got, want, 1)) if want else 0.0
    dims.append(_dim(
        "格数吻合", _round1(matched * 25), 25,
        f"{got} 格（期望 {rows}×{cols}={want}）",
    ))

    # 3. 英文提示词（20 分）—— 中文占比超阈值即算不合格（阈值含义见 ``_CJK_RATIO_MAX``）
    texts = cell_texts + [grid_prompt]
    chinese = [index for index, text in enumerate(texts) if _cjk_ratio(text) > _CJK_RATIO_MAX]
    rate = ((len(texts) - len(chinese)) / len(texts)) if cell_texts else 0.0
    dims.append(_dim(
        "英文提示词", _round1(rate * 20), 20,
        (f"{len(chinese)}/{len(texts)} 处中文占比超 {_CJK_RATIO_MAX:.0%}（末位是整体提示词）"
         if chinese else f"中文占比均 < {_CJK_RATIO_MAX:.0%}（只有格号/图片号这类合规标签）"),
    ))

    # 4. 无模板残留（15 分）—— `[...]` / `TODO` / `xxx` / `<...>` 这类占位符
    # ⚠️ 分母是**全部待判文本、含空的**，分子只数「非空且干净」：空产出它不是"没有占位符"，
    # 否则 payload=None 时这一维会白送 15 分（空串里当然搜不到 `[`），总分成了一条假绿。
    tags = [str(tag) for tag in (rubric.get("forbiddenTags") or ("[", "<", "TODO", "xxx"))]
    all_texts = cell_texts + [grid_prompt]
    clean = [text for text in all_texts
             if text.strip() and not any(tag and tag in text for tag in tags)]
    dirty = [index for index, text in enumerate(all_texts)
             if any(tag and tag in text for tag in tags)]
    rate = (len(clean) / len(all_texts)) if any(text.strip() for text in all_texts) else 0.0
    dims.append(_dim(
        "无模板残留", _round1(rate * 15), 15,
        f"{len(dirty)} 处残留占位符（{'/'.join(tags)}）" if dirty else "无占位符残留",
    ))

    # 5. 格间不重复（15 分）—— 同一句复制 N 遍会让每格出几乎一样的图
    unique = len(set(text.strip() for text in cell_texts if text.strip()))
    rate = (unique / len(cell_texts)) if cell_texts else 0.0
    dims.append(_dim(
        "格间不重复", _round1(rate * 15), 15,
        f"{unique}/{len(cell_texts)} 格提示词互不相同",
    ))

    return _report("grid_prompt", dims)
