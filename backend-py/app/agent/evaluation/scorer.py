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
4. 四个评分器返回的 ``caseId`` 都是**空串**（由 ``evaluator`` 事后填）—— 别"顺手"塞 case id；
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
    "score_script_rewrite",
    "score_storyboards",
    "score_voice_assignment",
]


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
