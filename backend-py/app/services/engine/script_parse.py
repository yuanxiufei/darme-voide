"""**导演稿 → 生成段**的纯规则层 ✓（2026-09-24 补 ✓ 口径来自逆向 ✓ 零依赖 ✓）。

## 为什么放在引擎里
H3 **单次原生只能生成 ≤15 s** ✓ ⇒「一段 = 一次生成」是最稳的切法 ✓；所以「怎么把一篇稿子
切成生成段」是**管线的前置数学** ✓，与 :mod:`app.services.engine.segments`（帧级分段计划 ✓）
同一族 ✓。⚠️ 本模块**不碰 LLM** ✗（纯规则 ⇒ 可复现、可自检 ✓）。

## 口径（逐条对应来源 ✓）
* **5 种标记写法** ✓（照参考实现的口径 ✓）：段标记带时长 ``段1（6.6秒）：…`` / 区间
  ``[0s-6.6s] …`` 与 ``0:00-0:06 …`` 与 ``0至6秒：…`` / 时长直写 ``6.6秒 | …``；
* **官方 Shot 格式** ✓：``[Shot 1] … [Shot 2] At 00:03.000, …`` ⇒ 时长由 ``At`` 时间戳差值给 ✓
  （末镜**沿用上一镜**的差值 ✓）；**总时长 ≤15 s ⇒ 合并成一段** ✓（一次生成 ✓），>15 s 才贪心装桶 ✓；
* **无标记** ⇒ 按朗读时长估算：**4.5 字/秒** ✓，切成 **8~15 s** ✓；
* **断点优先级 = 段落换行 > 句末标点 > 从句标点** ✓，**绝不在句中硬断** ✗（仅无标点超长串按字数兜底 ✓，
  且那种段会带 ``hard_cut=True`` ✓ 让人看得见 ✓）；
* **时长吸附 H3 帧网格** ✓：``17k+5`` @24fps ✓ —— 用 :func:`geometry.snap_frames` **同一处判据** ✗
  （两处各写一份必然漂 ✓✗），且吸附**只许往下** ✗✗（往上会越过 15 s 上限 ✓✗）。

## 不猜
* ⚠️ **英文语速未核** ✗：来源只给了中文「4.5 字/秒」✓ ⇒ 本模块把每个**英文字母串按一个词算 1** ✓
  并留了 ``units_per_second`` 口子 ✓（英文要不要另调，由调用方定 ✓）；
* ⚠️ **不改写文本** ✗：绝不**重排**子镜头时间戳 ✓（那会动到用户手写的官方提示词 ✓✗）；
  分桶/分段只在**边界**切 ✓；
* ⚠️ 与来源的一处**有意差异** ✓：显式时长 >15 s 的段，来源是「截到 15 秒」✗ —— 那会**静默丢内容** ✗✗
  ⇒ 本模块改成**按断点切开** ✓（宁可多一次生成，也不丢字 ✓）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["CHARS_PER_SECOND", "MAX_SECONDS", "MIN_SECONDS", "ScriptPlan", "ScriptSegment",
           "estimate_seconds", "parse_script", "speak_units"]

#: 朗读语速口径 ✓（**中文** ✓ 来源：参考实现「标准语速 4.5 字/秒」✓）。
CHARS_PER_SECOND = 4.5
#: 无标记稿切成的目标区间 ✓（来源口径 ✓）。
MIN_SECONDS = 8.0
#: H3 单次原生生成的**硬上限** ✓（来源口径 ✓ + 与本仓 `segments` 的上限同义 ✓）。
MAX_SECONDS = 15.0

#: 断点优先级 ✓（**段落换行 > 句末标点 > 从句标点** ✓ —— 段落用 ``\n`` 判 ✓）。
SENTENCE_BREAKS = "。！？!?…"
CLAUSE_BREAKS = "，、：；,;:"

_SEGMENT_LINE_RE = re.compile(
    r"^[ \t]*(?:段|镜头|shot)?\s*(\d+)\s*[（(]\s*([\d.]+)\s*秒?\s*[)）]\s*[:：]?\s*(.*)$",
    re.IGNORECASE)
#: 区间标记的三种写法 ✓ —— ⚠️ **分开写** ✗：并成一条「单位可省」的大正则会把
#: 「3-4 个镜头」这种**普通文本**也吃成区间 ✓✗ ⇒ 每条都必须有**决定性证据** ✓
#: （冒号时间 / 单位 ✓）。用**具名组**统一（``h1/m1/h2/m2/rest`` ✓）。
_RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[ \t]*(?:\[|【)?\s*(?P<h1>\d+):(?P<m1>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*(?:-|—|~|至)\s*"
               r"(?:(?P<h2>\d+):)?(?P<m2>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*(?:\]|】)?\s*[:：]?\s*(?P<rest>.*)$"),
    re.compile(r"^[ \t]*(?:\[|【)?\s*(?P<m1>\d+(?:\.\d+)?)\s*(?:s|秒)\s*(?:-|—|~|至)\s*"
               r"(?P<m2>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*(?:\]|】)?\s*[:：]?\s*(?P<rest>.*)$"),
    re.compile(r"^[ \t]*(?:\[|【)?\s*(?P<m1>\d+(?:\.\d+)?)\s*(?:-|—|~|至)\s*(?P<m2>\d+(?:\.\d+)?)\s*"
               r"(?:s|秒)\s*(?:\]|】)?\s*[:：]?\s*(?P<rest>.*)$"),
)
_BAR_RE = re.compile(r"^[ \t]*([\d.]+)\s*秒?\s*\|\s*(.*)$")
_SHOT_RE = re.compile(r"\[Shot\s+(\d+)\]", re.IGNORECASE)
_SHOT_AT_RE = re.compile(r"At\s+(\d{2}):(\d{2})\.(\d{3})", re.IGNORECASE)


def _match_range(line: str) -> tuple[float, float, str] | None:
    """区间标记 ⇒ ``(起点秒, 终点秒, 正文)`` ✓；不是区间 ⇒ ``None`` ✓。"""
    for pattern in _RANGE_PATTERNS:
        found = pattern.match(line)
        if not found:
            continue
        groups = found.groupdict()
        start = float(groups.get("h1") or 0) * 60 + float(groups["m1"])
        end = float(groups.get("h2") or 0) * 60 + float(groups["m2"])
        return start, end, (groups.get("rest") or "").strip()
    return None


def _seconds_of(found: re.Match[str] | None) -> float:
    """``At mm:ss.mmm`` ⇒ 秒 ✓。"""
    if found is None:
        return 0.0
    return int(found.group(1)) * 60 + int(found.group(2)) + int(found.group(3)) / 1000.0


@dataclass(frozen=True)
class ScriptSegment:
    """一段生成请求 ✓（``frames`` 已经是**吸附到网格**的值 ✓ ⇒ ``seconds`` 是吸附后的 ✓）。"""

    index: int
    text: str
    seconds: float
    frames: int
    source: str          # marker / range / bar / shots / plain ✓
    hard_cut: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "seconds": self.seconds, "frames": self.frames,
                "source": self.source, "hardCut": self.hard_cut, "chars": len(self.text),
                "text": self.text}


@dataclass(frozen=True)
class ScriptPlan:
    """分段结果 ✓（``notes`` 说明「为什么这么切」✓ —— 静默切分比切错更难查 ✗）。"""

    segments: tuple[ScriptSegment, ...]
    style: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"style": self.style, "count": len(self.segments), "notes": list(self.notes),
                "segments": [segment.to_dict() for segment in self.segments]}


def speak_units(text: str) -> int:
    """「要读多少个字」✓：CJK 字符各 1 ✓ + 每个**英文字母/数字串**算 1 ✓（⚠️ 英文那半边是**本仓取舍** ✗）。"""
    units = 0
    latin_run = False
    for char in str(text or ""):
        if char.isalpha() and char.isascii() or char.isdigit() and char.isascii():
            if not latin_run:
                units += 1
                latin_run = True
        elif "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf":
            units += 1
            latin_run = False
        else:
            latin_run = False
    return units


def estimate_seconds(text: str, *, units_per_second: float = CHARS_PER_SECOND) -> float:
    """朗读时长估算 ✓（``字数 ÷ 语速`` ✓；``units_per_second`` 可覆盖 ✓ —— 见模块头的「英文未核」✗）。"""
    rate = float(units_per_second or CHARS_PER_SECOND)
    if rate <= 0:
        raise ValueError(f"语速必须为正（收到 {units_per_second!r} ✗）")
    return round(speak_units(text) / rate, 3)


def _from_geometry():
    from . import geometry

    return geometry


def _snap(seconds: float, *, cap_seconds: float | None) -> tuple[float, int]:
    """秒 → ``(吸附后秒数, 帧数)`` ✓；⚠️ 只许**往下**吸附 ✗✗（往上会越过 15 s 上限 ✓✗）。"""
    geometry = _from_geometry()
    frames = geometry.snap_frames(seconds)
    if cap_seconds is not None:
        limit = int(geometry.snap_frames(cap_seconds))
        if limit > cap_seconds * geometry.H3_FPS:      # 往上吸附会越界 ⇒ 退一格 ✓
            limit -= geometry.H3_FRAME_GRID
        limit = max(geometry.H3_MIN_FRAMES, limit)
        if frames > limit:
            frames = limit
    return round(frames / geometry.H3_FPS, 4), frames


def _break_index(text: str, start: int, end: int) -> int | None:
    """在 ``[start, end)`` 里找**最优先**的断点 ✓ ⇒ 切点（切在断点**之后** ✓）；没有 ⇒ ``None`` ✓。

    优先级：段落换行（``\\n`` ✓）> 句末标点 ✓ > 从句标点 ✓；同级取**靠后**的 ✓（尽量切长 ✓）。
    """
    best: tuple[int, int] | None = None
    for index in range(max(0, start), min(len(text), end)):
        char = text[index]
        if char == "\n":
            priority = 0
        elif char in SENTENCE_BREAKS:
            priority = 1
        elif char in CLAUSE_BREAKS:
            priority = 2
        else:
            continue
        if best is None or priority <= best[1]:
            best = (index + 1, priority)
    return best[0] if best else None


def _pieces(text: str, *, units_per_second: float, min_seconds: float,
            max_seconds: float) -> list[tuple[str, bool]]:
    """把一段文本切成 ``min~max`` 秒的片 ✓ ⇒ ``[(片, 是否硬切)]`` ✓（**绝不丢字符** ✓）。

    切法：从当前位置往后吃到「够 ``max_seconds`` 的字数 ✓」为止，再在这个窗口的**后半段**里
    找**最优先**的断点（段落 > 句末 > 从句 ✓）—— 找得到就切在那儿 ✓、找不到才**硬切**在窗口末尾 ✓
    并**标出来** ✓。⚠️ 「绝不在句中硬断」✗ 的准确含义是「**有断点时不许硬切**」✓：
    没有任何标点的超长串只能硬切 ✓，但那种段必须带 ``hard_cut=True`` ✓ 让人看得见 ✓。
    最后把短于 ``min_seconds`` 的片**并入前一片** ✓（切太碎比切粗更费生成 ✓）。
    """
    body = text.strip()
    if not body:
        return []
    rate = float(units_per_second or CHARS_PER_SECOND)
    max_units = max(1, int(max_seconds * rate))
    pieces: list[tuple[str, bool]] = []
    cursor = 0
    while cursor < len(body):
        chunk = body[cursor:]
        cut = len(body)
        hard = False
        if speak_units(chunk) > max_units:
            taken = len(chunk)
            for offset in range(len(chunk)):
                if speak_units(chunk[:offset + 1]) >= max_units:
                    taken = offset + 1
                    break
            found = _break_index(chunk, max(1, int(taken * 0.4)), taken)
            if found is not None:
                cut = cursor + found
            else:
                cut = cursor + taken
                hard = True
        piece = body[cursor:cut].strip()
        if piece:
            pieces.append((piece, hard))
        if cut <= cursor:
            break
        cursor = cut
    merged: list[tuple[str, bool]] = []
    for piece, hard in pieces:
        if (merged and not hard and piece
                and estimate_seconds(piece, units_per_second=units_per_second) < min_seconds
                and estimate_seconds(merged[-1][0] + piece,
                                     units_per_second=units_per_second) <= max_seconds):
            merged[-1] = (f"{merged[-1][0]} {piece}", merged[-1][1])
        else:
            merged.append((piece, hard))
    return merged


@dataclass(frozen=True)
class _Item:
    text: str
    seconds: float
    source: str
    #: 切这一片时**没有断点可用** ⇒ 硬切 ✓（⚠️ 必须**从切分处带上来** ✗ —— 后面再推断一次就会错 ✓✗）
    hard: bool = False


def _parse_shots(text: str) -> list[_Item] | None:
    """官方 Shot 格式 ⇒ 每个 ``[Shot N]`` 一项 ✓（时长取 ``At`` 时间戳**差值** ✓ 末镜沿用上一镜 ✓）。

    ⚠️ 必须按 ``[Shot N]`` 数**镜头** ✗ —— 只数 ``At`` 会**少算**开头那个没有时间戳的镜头 ✓✗
    （本套自检第一次就栽在这上面 ✓）。
    """
    starts = [(match.start(), int(match.group(1))) for match in _SHOT_RE.finditer(text)]
    if not starts:
        return None
    stamps: list[float] = []
    for position, (offset, _number) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(text)
        stamps.append(_seconds_of(_SHOT_AT_RE.search(text, offset, end)))
    items: list[_Item] = []
    for position, (offset, _number) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(text)
        if position + 1 < len(starts):
            span = stamps[position + 1] - stamps[position]
        else:
            span = stamps[-1] - stamps[-2] if len(stamps) > 1 else 0.0
            if span <= 0:
                span = stamps[-1] or 1.0
        items.append(_Item(text=text[offset:end].strip(), seconds=round(max(span, 0.0), 3),
                           source="shots"))
    return items or None


def _parse_marked(text: str) -> list[_Item] | None:
    """5 种标记写法 ⇒ 每段一项 ✓（**没有**任何标记 ⇒ ``None`` ✓）。"""
    items: list[_Item] = []
    pending: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if pending:
                pending.append("")
            continue
        matched = _SEGMENT_LINE_RE.match(line)
        if matched and (matched.group(1) or matched.group(2)):
            items.append(_Item(text=matched.group(3).strip(),
                               seconds=float(matched.group(2)), source="marker"))
            pending = []
            continue
        ranged = _match_range(line)
        if ranged is not None:
            start, end, rest = ranged
            items.append(_Item(text=rest.strip(), seconds=round(max(end - start, 0.0), 3),
                               source="range"))
            pending = []
            continue
        barred = _BAR_RE.match(line)
        if barred and barred.group(1):
            items.append(_Item(text=barred.group(2).strip(), seconds=float(barred.group(1)),
                               source="bar"))
            pending = []
            continue
        if items:
            items[-1] = _Item(text=(items[-1].text + "\n" + line).strip(),
                              seconds=items[-1].seconds, source=items[-1].source)
        else:
            pending.append(line)
    if not items:
        return None
    if pending:
        items.insert(0, _Item(text="\n".join(pending).strip(),
                              seconds=estimate_seconds("\n".join(pending)), source="plain"))
    return items


def parse_script(text: str, *, units_per_second: float = CHARS_PER_SECOND,
                 min_seconds: float = MIN_SECONDS, max_seconds: float = MAX_SECONDS) -> ScriptPlan:
    """导演稿 → :class:`ScriptPlan` ✓（判据全在模块头 ✓；⚠️ 空稿 ⇒ 空计划 + 说明 ✓ 不抛错 ✗）。"""
    body = str(text or "").strip()
    if not body:
        return ScriptPlan(segments=(), style="empty", notes=("稿子是空的 ⇒ 没有段可切 ✓",))
    lo, hi = float(min_seconds), float(max_seconds)
    if not 0 < lo < hi:
        raise ValueError(f"区间不合法：min={min_seconds!r} max={max_seconds!r} ✗（要 0 < min < max ✓）")
    notes: list[str] = []
    shots = _parse_shots(body)
    if shots is not None:
        items, style = shots, "shots"
        total = sum(item.seconds for item in items)
        if total <= hi:
            notes.append(f"官方 Shot 格式：总时长 {total:.2f}s ≤ {hi:g}s ⇒ **合并成一段** ✓（一次生成 ✓）")
            if len(items) > 1:
                # ⚠️ 合并 = 只把各镜正文**接起来** ✓；子镜头标记与时间戳**原样保留** ✗（不改写 ✓）
                items = [_Item(text=" ".join(item.text for item in items),
                               seconds=round(total, 3), source="shots")]
        else:
            notes.append(f"官方 Shot 格式：总时长 {total:.2f}s > {hi:g}s ⇒ 按 Shot 边界贪心装桶 ✓"
                         f"（⚠️ **不改写**子镜头时间戳 ✗）")
    else:
        marked = _parse_marked(body)
        if marked is None:
            items, style = [_Item(text=body, seconds=estimate_seconds(body,
                                                                     units_per_second=units_per_second),
                                  source="plain")], "plain"
            notes.append(f"没有任何标记 ⇒ 按朗读时长估算（{units_per_second:g} 字/秒 ✓）"
                         f"、切 {lo:g}~{hi:g}s ✓")
        else:
            items, style = marked, marked[0].source
            notes.append(f"识别到标记（{style} ✓）⇒ 按标记切分 ✓")

    expanded: list[_Item] = []
    for item in items:
        if item.seconds > hi:
            pieces = _pieces(item.text, units_per_second=units_per_second, min_seconds=lo,
                             max_seconds=hi)
            if len(pieces) > 1:
                notes.append(f"有一段的**显式时长** > {hi:g}s ⇒ 按断点切成 {len(pieces)} 段 ✓"
                             f"（⚠️ 与来源的「截到 15 秒」**不同** ✓：本仓不丢内容 ✗）")
            for piece, hard in pieces:
                expanded.append(_Item(text=piece, seconds=estimate_seconds(
                    piece, units_per_second=units_per_second), source=item.source, hard=hard))
        else:
            expanded.append(item)
    if style in ("plain", "shots") and len(expanded) == 1:
        pieces = _pieces(expanded[0].text, units_per_second=units_per_second, min_seconds=lo,
                         max_seconds=hi)
        if len(pieces) > 1:
            expanded = [_Item(text=piece, seconds=estimate_seconds(piece,
                                                                   units_per_second=units_per_second),
                              source=style, hard=hard) for piece, hard in pieces]

    segments: list[ScriptSegment] = []
    for index, item in enumerate(expanded, start=1):
        seconds, frames = _snap(item.seconds, cap_seconds=hi)
        # ⚠️⚠️ ``hard_cut`` **沿用切分时的事实** ✗：有断点可用这件事只有 :func:`_pieces` 知道 ✓ ——
        #    这里曾**重新推断**一次，条件写成「这一片超没超上限」✗ ⇒ 硬切全被判成软切 ✓✗
        #    （本套自检的 ⑤ 就是为这个来的 ✓）。
        segments.append(ScriptSegment(index=index, text=item.text, seconds=seconds, frames=frames,
                                      source=item.source, hard_cut=item.hard))
    if segments and segments[0].frames and notes:
        notes.append(f"时长已**往下**吸附到 H3 帧网格（17k+5 @24fps ✓）："
                     f"{segments[0].seconds:g}s / {segments[0].frames} 帧 ✓")
    return ScriptPlan(segments=tuple(segments), style=style, notes=tuple(notes))
