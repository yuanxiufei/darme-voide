"""场次解析 —— 把剧本正文切成"场"（纯函数：无 IO、无 LLM、无随机）。

OCR 管线只把像素变成字，``chunks`` 的语义字段只有 ``page`` / ``chunk_no`` ⇒ 库里 5403 份
剧本**只能被搜到、不能被"按场取"**。而"第 12 场讲了什么""某角色出场几场"需要稳定边界。

这一层**不调模型**（services 层不依赖 LLM），场次边界是形式特征、正则判得出来。判不出来的
**如实标注**而不猜：``no_source`` 三档、``confidence``、``strategy``；``core_event`` /
``scene_tail`` 一律 None —— 那是语义判断，宁空着也不许拿首句动作糊弄。

识别路线（降序，先命中先用）：``md-heading``（``## S01 | 外景 · 街道 | 黄昏``，0.95）＞
``cn-numbered``（``场景1：``/``第 3 场`` 0.9、裸数字序号 0.85）＞ ``ep-scene``（``1-1 空地 日/外``
集-场号 0.85）＞ ``hash-scene``（``S#1. 치킨 집 앞 (밤)`` 韩语／英语的 ``S#`` 写法 0.85）＞
``slug``（``INT. X - DAY`` 0.9）＞ ``place-time``（``二层客厅 日内`` 裸地点+时段 0.6）＞
``none``（整篇一场，如实标 missing）。

⚠️ ``ep-scene`` / ``hash-scene`` / ``place-time`` 是 2026-11-06 补的：库内短剧剧本用的正是
``集-场号``（``1-0 夜外荒野`` / ``1-1a日外焉州-康郡``）、``地点+时段``（``纯黑环境 夜内``）与
韩语剧本的 ``S#1.`` 这几种写法，此前一律落到 ``none``、整篇被记成"一场"——**那是解析器不认识，
不是这些剧本没有场次**。

内外景（``INT./EXT.``）有三条来源：``INT.``/``内景`` 字面、``夜内``/``日外`` 这类「时段+内外」粘合词、
以及挂在地点**尾字**的 ``内``/``外``（``民政局内 日``）。第三条对 ``郊外``/``城内`` 这类地名不截字。

⚠️ **纯空行不切**：中文剧本空行到处都有，按空行切会得到几十个假场次，假结构比没结构更坏。
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["SCENE_STRATEGIES", "SCENES_SCHEMA_VERSION", "extract_cast", "parse_scenes"]

#: 解析器输出的形状版本（**不是** artifact-state.json 的 schema_version）
SCENES_SCHEMA_VERSION = 1
#: 全部可能的 ``strategy`` 返回值。``inline`` 是"正文被压平、按行内标记重切"这一档 ——
#: 它**必须**在册，否则"返回值不在枚举里"这类漂移永远查不出来（`parse_scenes` 真会返回它）。
SCENE_STRATEGIES = (
    "md-heading", "cn-numbered", "ep-scene", "hash-scene", "inline", "slug", "place-time", "none",
)

NO_SOURCE_AUTHORED = "authored"
NO_SOURCE_ASSIGNED = "production-assigned"
NO_SOURCE_MISSING = "missing"

MAX_HEADING_CHARS = 80

_MD_PREFIX = re.compile(r"^#{1,6}\s*")
_FENCE = re.compile(r"^\s{0,3}(?:```|~~~)")
#: 行内出现的围栏符号（判断"这一行开/闭了几次"）
_FENCE_TOKEN = re.compile(r"```|~~~")
#: ``场12 客厅 日内`` 也算强场号；但 ``场1-1``（集-场）必须留给 ``_EP_SCENE_NO`` ——
#: 否则 ``场1`` 先命中、剩下 ``-1 金銮殿`` 被当成地点（2026-11-06 实测的脏值）。
_STRONG_NO = re.compile(
    r"^(?:S(\d{1,4})|场景\s*(\d{1,4})(?!\s*[-—–·]\s*\d)|第\s*(\d{1,4})\s*场"
    r"|场\s*(\d{1,4})(?!\s*[-—–·]\s*\d))\b",
    re.I,
)
_WEAK_NO = re.compile(r"^(\d{1,4})\s*[.、．]\s*")
#: ``集-场号``：``1-1 空地 日/外``、``1-1a日外焉州-康郡``、``12-3 夜 内 客厅``。
#: 与 ``1.`` / ``1、`` 的区别是**两段数字**（集 + 场），场号每集重置 ⇒ ``scene_no`` 只取场号
#: 那一段，集号留在 ``heading`` 原文里（schema 没有 episode 列，不为了这个改表）。
#: 段数限 ``\d{1,3}``：4 位会被 ``2024-1-1`` 这类日期行命中。
_EP_SCENE_NO = re.compile(
    r"^(?:场(?:景)?)?\s*(?P<ep>\d{1,3})\s*[-—–·]\s*(?P<no>\d{1,3}[a-zA-Z]?)\s*"
)
#: 韩语／英语剧本常见的 ``S#1. 치킨 집 앞 (밤)``：``S`` 是 scene 的首字母，``#`` 只是分隔符
#: （与 ``S1`` 的唯一区别就是那个 ``#``，所以 ``_STRONG_NO`` 认不到）。``#`` **单独**起头的
#: ``#1`` 不认：markdown 标题、脚注、页码都是 ``#``，误判面太大。
_HASH_SCENE_NO = re.compile(r"^(?P<sig>[Ss])\s*#\s*(?P<no>\d{1,3}[a-zA-Z]?)[.．、):：]?\s*")
_INT_EXT = re.compile(
    r"(内外景|内/外景|内景/外景|内景|外景|(?<![A-Za-z])(?:INT|EXT)(?![A-Za-z])\.?)", re.I
)
_TIME_WORDS = (
    "凌晨", "清晨", "早晨", "早上", "上午", "白天", "正午", "中午", "午后", "下午",
    "黄昏", "傍晚", "日暮", "夜晚", "深夜", "午夜", "夜里", "夜间", "夜", "日",
    "CONTINUOUS", "AFTERNOON", "EVENING", "MORNING", "MIDNIGHT", "SUNSET", "NIGHT",
    "DAWN", "DUSK", "DAY", "LATER",
)
#: 时段前的**相对日期限定词**。``次日中午`` 是一个时段，不是"地点=大厅 次日、时段=中午" ——
#: 不认它，``location`` 会多出"次日"这个尾巴，拿去和 ``scenes.location`` 比时
#: ``东家商号大厅 次日`` 与 ``东家商号大厅`` 只能靠子串兜住，换个写法就漏（2026-11-04 实测）。
_TIME_QUALIFIERS = (
    "次日", "翌日", "隔日", "第二天", "第二日", "当日", "当天", "当晚", "当夜", "同夜", "次晨",
)
_TAIL_TIME = re.compile(
    r"[\s|｜·・\-—:：,，、]*(?:(?P<qual>"
    + "|".join(sorted(_TIME_QUALIFIERS, key=len, reverse=True))
    + r")\s*)?(?P<time>"
    + "|".join(sorted(_TIME_WORDS, key=len, reverse=True))
    + r")\s*$",
    re.I,
)
#: 时段与内外景**粘成一个词**：``夜内`` / ``日外`` / ``日/外`` / ``日 内`` / ``夜外``。
#: 只在①已确认场号（``ep-scene``）②``place-time`` 兜底且整行像场次头 这两种场合使用 ——
#: 裸正文里「日内瓦」「窗外」「室外」都会误命中，不能无条件当内外景标记。
_GLUED_INT_EXT = re.compile(
    r"(?P<time>" + "|".join(sorted(_TIME_WORDS, key=len, reverse=True))
    + r")\s*[/／]?\s*(?P<ie>内|外)"
)
#: ``place-time`` 的排除面：整行只要沾上台词/动作/特效标记，就不是场次头。
_PLACE_TIME_BAD = re.compile(r"[：:。！？；!?▲△※→【\[（(]")
#: 以这些词开头的行是**镜头/接续注记**，不是新场次：``接8-1 叶婉家客厅 日/内``（接上文那一场）、
#: ``插入镜头肖庆元办公室 日/内``（插镜头）。把它们当场次头会让场数虚增、并切断前一场的正文。
_PLACE_TIME_DENY = re.compile(
    r"^(?:接|插入镜头|插入|闪回|切至|跳切|叠化|淡入|淡出|画外|旁白|字幕|音效|音乐|特写|远景|近景|全景|中景|空镜)"
)
#: ``place-time`` 的整行长度上限（场次头本来就短，长了必然是正文句）
_PLACE_TIME_MAX = 24
_SEPARATORS = re.compile(r"[|｜·・\-—–:：,，、\s]+")
#: 行内场次标记：OCR 把整篇压成一段时（``_flatten`` 会吃掉换行），只能靠这些标记**
#: 在正文里的位置**重新切分。偏移对原文直接成立，不需要"插换行再重算偏移"那种脆做法。
_INLINE_MARK = re.compile(
    r"(?:场景\s*\d{1,4}|第\s*\d{1,4}\s*场|(?<![A-Za-z])(?:INT|EXT)\.|S\d{1,4}\s*[|｜]"
    #: ``集-场号`` 埋在行中间（OCR 一页一行时必然如此）：``…1-1b 日内康郡-乔府…``。
    #: 必须紧跟时段词才认 —— 光凭 ``\d-\d`` 会把「第1-2集」「1-1条」这类正文数字也切了。
    r"|(?<![\d-])\d{1,3}\s*-\s*\d{1,3}[a-zA-Z]?\s{0,2}(?=(?:凌晨|清晨|早晨|早上|上午|白天"
    r"|正午|中午|午后|下午|黄昏|傍晚|日暮|夜晚|深夜|午夜|夜里|夜间|夜|日)))", re.I
)
#: 任何 markdown 标题（含**不是**场次的小标题，如 ``## 人物设定``）：用来截断上一场
_ANY_MD_HEADING = re.compile(r"^#{1,6}\s+\S")
_DIALOGUE = re.compile(
    r"^\s*(?:[-—•·*]\s*)?([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9·]{0,11})"
    r"\s*(?:[（(][^）)\n]{0,14}[）)])?\s*[：:]"
)
_ENTER = re.compile(r"^\s*[（(]\s*([\u4e00-\u9fa5A-Za-z·]{1,10})\s*(?:上|入|进|冲进|推门进)")
_NOT_CAST = frozenset({
    "旁白", "画外音", "字幕", "内景", "外景", "日内", "日外", "时间", "地点", "人物", "场景",
    "镜头", "画面", "音效", "音乐", "备注", "注", "闪回", "切至", "淡入", "淡出", "转场",
    "全剧终", "未完待续", "OS", "VO", "V.O.", "O.S.", "INT", "EXT",
})
_INT_EXT_MAP = {
    "内景": "INT", "int": "INT", "int.": "INT",
    "外景": "EXT", "ext": "EXT", "ext.": "EXT",
    "内外景": "INT/EXT", "内/外景": "INT/EXT", "内景/外景": "INT/EXT",
}


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    """``[(start, end, body)]`` —— 每行在**原文**里的精确偏移（``end`` 不含换行符）。

    不能用 ``splitlines()`` 拼回去算偏移：``\\r\\n`` 会让整体错位，而 ``char_start`` /
    ``char_end`` 是"按场取原文"的唯一依据，错一个字就跨场。
    """
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        spans.append((offset, offset + len(body), body))
        offset += len(raw)
    return spans


def _fence_count(line: str) -> int:
    """这一行里出现了几个围栏符号。"""
    return len(_FENCE_TOKEN.findall(line or ""))


def _content_lines(lines: list[str]) -> list[str]:
    """去掉围栏标记行与围栏**内部**的行。

    "围栏内不算正文"这条判据出现在三处（场次边界、角色提取、首句引用）。各写一遍必然
    漏一处 —— 收在这里，三处都走它。
    """
    out: list[str] = []
    inside = False
    for line in lines:
        count = _fence_count(line)
        if count:
            if count % 2:
                inside = not inside
            continue  # 围栏标记行本身不是内容
        if inside:
            continue
        out.append(line)
    return out


def _fenced_lines(
    spans: list[tuple[int, int, str]],
) -> tuple[set[int], list[tuple[int, int]]]:
    """围栏代码块**内部**的行号集合 ＋ 对应的字符区间（都含起止两行）。

    ``_parse_heading`` 里那句 ``_FENCE`` 只能挡"这一行本身是 ```"——挡不住**围栏里**的
    内容。而说明文档里贴一段 ``## S01 | 内景 …`` 的示例写法是常事，误当成真场次会让整篇
    解析结果从"2 场"变成"3 场"且不报错。围栏要么整体算、要么不算，只判单行等于没判。

    返回**两份**是有原因的：走行结构的解析按行号排除，走 ``inline`` 的压平文本没有行号，
    只能按字符区间排除。同一件事两处都要挡，不挡的那处就是漏点。
    """
    inside = False
    marked: set[int] = set()
    regions: list[tuple[int, int]] = []
    region_start = 0
    for index, (start, end, body) in enumerate(spans):
        tokens = list(_FENCE_TOKEN.finditer(body))
        if tokens:
            marked.add(index)
            #: 同一行里成对出现的围栏符号（```示例```）配对成行内区间；
            #: 落单的最后那个才切换"跨行围栏"状态。
            for first, second in zip(tokens[0::2], tokens[1::2]):
                regions.append((start + first.start(), start + second.end()))
            if len(tokens) % 2:
                last = tokens[-1]
                if not inside:
                    region_start = start + last.start()
                else:
                    regions.append((region_start, start + last.end()))
                inside = not inside
            continue
        if inside:
            marked.add(index)
    if inside:  # 未闭合：一直到文末都算围栏（宁可少切，不可切在示例代码里）
        regions.append((region_start, spans[-1][1] if spans else 0))
    return marked, regions


def _place_time_ok(line: str) -> bool:
    """整行像不像「地点 + 时段」的场次头（``纯黑环境 夜内`` / ``二层客厅 日内``）。

    这是全套判据里最弱的一档，所以先按**排除法**收紧：短、无句读、无台词冒号、无动作/特效
    标记、去掉时段后还剩至少 2 个字的地点。即便全过，``parse_scenes`` 还要求这类头成规模
    （≥3 个）才采纳 —— 单行的「他走进客厅 日内」不许把整篇切出一场假结构。
    """
    body = line.strip()
    if (not body or len(body) > _PLACE_TIME_MAX
            or _PLACE_TIME_BAD.search(body) or _PLACE_TIME_DENY.match(body)):
        return False
    rest = _GLUED_INT_EXT.sub(" ", body, count=1)
    location = _SEPARATORS.sub("", rest).strip()
    return len(location) >= 2


def _parse_heading(line: str) -> dict[str, Any] | None:
    """行 → 场次头；不是场次头返回 None（``## 人物设定`` 不许被当成一场戏）。"""
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS or _FENCE.match(stripped):
        return None

    has_md = bool(_MD_PREFIX.match(stripped))
    text = _MD_PREFIX.sub("", stripped).strip()
    if not text:
        return None

    #: ``【``/``[`` 起是动作或场景描述（``1-1b 日内康郡-乔府-乔圭寝屋【小乔看着乔圭身下的陈年虎皮…``），
    #: 场次头到那里就结束了。OCR 把整页压成一行时，不截断会把整段正文算进 ``location``，
    #: ``heading`` 也会变成几百字。
    head_cut = min([pos for pos in (text.find(ch) for ch in "【[") if pos > 0], default=-1)
    if head_cut > 0:
        text = text[:head_cut].rstrip()
        stripped = text
        if not text:
            return None

    scene_no: int | None = None
    no_source = NO_SOURCE_MISSING
    strong = weak = ep_scene = hash_scene = False
    strong_match = _STRONG_NO.match(text)
    if strong_match:
        digits = next((g for g in strong_match.groups() if g), None)
        scene_no = int(digits) if digits else None
        no_source, strong = NO_SOURCE_AUTHORED, True
        text = text[strong_match.end():]
    else:
        ep_match = _EP_SCENE_NO.match(text)
        if ep_match:
            #: 只取**场号**那一段（``1-1a`` ⇒ 1）；集号每集重置、schema 无 episode 列，
            #: 所以集号不出现在 ``scene_no`` 里 —— 原文整串由 ``heading`` 原样保留。
            scene_no = int(re.match(r"\d+", ep_match.group("no")).group())
            no_source, ep_scene = NO_SOURCE_AUTHORED, True
            text = text[ep_match.end():]
        else:
            hash_match = _HASH_SCENE_NO.match(text)
            if hash_match:
                scene_no = int(re.match(r"\d+", hash_match.group("no")).group())
                no_source, hash_scene = NO_SOURCE_AUTHORED, True
                text = text[hash_match.end():]
            else:
                weak_match = _WEAK_NO.match(text)
                if weak_match:
                    scene_no, no_source, weak = int(weak_match.group(1)), NO_SOURCE_AUTHORED, True
                    text = text[weak_match.end():]

    int_ext_raw = ""
    int_match = _INT_EXT.search(text)
    if int_match:
        int_ext_raw = int_match.group(1)
        text = f"{text[:int_match.start()]} {text[int_match.end():]}"

    #: ``夜内``/``日外``/``日/外``：时段与内外景粘成一个词的写法。有场号锚定（``ep-scene``）
    #: 时可以放心剥；无场号时只能留给 ``place-time`` 兜底分支，且还要过整行像不像场次头。
    glued_time = glued_ie = ""
    if not int_match:
        glued_match = _GLUED_INT_EXT.search(text)
        if glued_match and (ep_scene or hash_scene or strong or weak
                            or _place_time_ok(stripped)):
            glued_time, glued_ie = glued_match.group("time"), glued_match.group("ie")
            text = f"{text[:glued_match.start()]} {text[glued_match.end():]}"

    time_value = ""
    tail_match = _TAIL_TIME.search(text)
    if tail_match:
        #: 限定词与时段词合并成一个值：``次日`` + ``中午`` ⇒ ``次日中午``（两者都描述打光时刻）
        time_value = f"{tail_match.group('qual') or ''}{tail_match.group('time')}"
        text = text[:tail_match.start()]
    elif glued_time:
        time_value = glued_time

    if strong:
        strategy, confidence = ("md-heading", 0.95) if has_md else ("cn-numbered", 0.9)
    elif ep_scene:
        strategy, confidence = "ep-scene", 0.85
    elif hash_scene:
        strategy, confidence = "hash-scene", 0.85
    elif weak and (int_ext_raw or time_value):
        strategy, confidence = "cn-numbered", 0.85
    elif int_match:
        strategy, confidence = "slug", 0.9
    elif glued_ie and _place_time_ok(stripped):
        #: 无场号、无 INT/EXT 字面，只有「地点 + 时段+内外景」—— 判据最弱，confidence 给 0.6，
        #: 并且在 ``parse_scenes`` 里还要过「至少 3 个这样的头」的门槛，单行不认。
        strategy, confidence = "place-time", 0.6
    else:
        return None

    location = _SEPARATORS.sub(" ", text).strip()
    #: 压平文本（inline 模式）会把正文接在标题后面 ⇒ 地点只取第一个句号之前，否则
    #: ``location`` 会变成"东家大厅 午夜 凌云进门"这种把台词也算进去的脏值。
    if location:
        location = re.split(r"[。！？；]", location, maxsplit=1)[0].strip()

    #: 中文剧本还把内外景写在**地点尾巴**上（``民政局内 日``、``（回忆）客厅内``）——既不是
    #: ``INT.`` 字面、也不是 ``夜内`` 那种「时段+内外」粘合词，前两支都漏。只认行尾单字
    #: 「内／外」；``郊外``／``城内`` 这类地名**不截字**（截完只剩一个「郊」，更难读），
    #: 只补内外景。
    if not int_ext_raw and not glued_ie and location:
        tail_ie = location[-1]
        if tail_ie in ("内", "外") and len(location) >= 2:
            glued_ie = tail_ie
            if len(location) >= 4:
                location = location[:-1].strip()

    return {
        "scene_no": scene_no,
        "no_source": no_source,
        "int_ext": (
            _INT_EXT_MAP.get(int_ext_raw.lower())
            or ({"内": "INT", "外": "EXT"}.get(glued_ie) if glued_ie else None)
            or None
        ),
        "location": location or None,
        "time": time_value or None,
        "heading": stripped,
        "strategy": strategy,
        "confidence": confidence,
    }


def _inline_heads(
    text: str, fenced_regions: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, dict[str, Any]]]:
    """无换行正文 → 场次头列表（按标记在原文里的位置切，偏移天然正确）。

    只在**行结构缺失**时启用：OCRed 的剧本常被压成一整段，此时逐行扫描必然一场都认不出。
    ``fenced_regions``：围栏代码块占的字符区间，落在里面的标记一律不算（与行结构解析
    同一口径，见 :func:`_fenced_lines`）。
    """
    def _fenced_at(pos: int) -> bool:
        return any(start <= pos < end for start, end in (fenced_regions or ()))

    marks = [match.start() for match in _INLINE_MARK.finditer(text) if not _fenced_at(match.start())]
    spans: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, start in enumerate(marks):
        stop = marks[index + 1] if index + 1 < len(marks) else len(text)
        #: 压平文本里"标题"的尽头是第一个句读 —— 再往后就是这一场的正文了。
        #: 不这么截，标题会把后面几场的字一起吞进来（``int_ext`` 会误判成``内景``、
        #: 时段词也找不到，因为它已不在"行尾"）。
        cuts = [pos for pos in (text.find(char, start) for char in "。！？；\n") if 0 <= pos < stop]
        head_end = min(cuts) + 1 if cuts else min(stop, start + MAX_HEADING_CHARS)
        head_text = text[start:head_end].rstrip("。！？；\n ")
        parsed = _parse_heading(head_text)
        if parsed is None:
            continue
        spans.append((text.count("\n", 0, start), start, head_end, parsed))
    return spans


_SENTENCE_CUT = re.compile(r"(?<=[。！？；])")


def _soft_lines(text: str) -> list[str]:
    """压平文本 → 伪行：按换行与句读切分（**保留**句尾标点）。

    只在 ``inline`` 模式下用。正文被 OCR 压成一整段时 ``splitlines()`` 只有一行，
    ``名字：台词`` 都在行中间 ⇒ ``extract_cast`` 一个都提不到、``opening`` 会把整段
    （含台词）吞进去。按句读切不改变任何偏移，只是让"按行看"的启发式重新有了粒度。
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        out.extend(part for part in _SENTENCE_CUT.split(line) if part.strip())
    return out


def extract_cast(body: str, known: Any = None, *, soft: bool = False) -> list[str]:
    """本场出场角色名（首次出现顺序、去重）。

    只认两种**有语法标记**的写法：``名字：台词`` / ``名字（情绪）：台词``，以及
    ``（名字 上）`` 这类进出场提示。``名字\\n台词`` 的无冒号老格式**提不出来** ——
    它和"两行叙述"无法区分。宁可漏，不可错：``cast`` 会被用来统计出场与做角色一致性核对。

    ``known`` 是**已知角色名集合**（B 侧能拿到项目角色表）。它只用来**放宽**名字长度上限，
    不用来过滤 —— 剧本里出现一个还没登记的新角色时，丢掉它比留下它更糟（"这场没人出场"
    这种幻觉会顺着 cast 传到分镜和配音）。

    ``soft=True``（仅 ``inline`` 模式）：按句读当行切，见 :func:`_soft_lines`。

    ⚠️ 中文名几乎不超过 4 字，而 ``云姐摇了摇头：`` 这种"动作+冒号"的行会被
    ``名{1,12}`` 误捕成 7 字"名字"。没有词典就只能用长度卡：>4 字一律不认（除非在 known 里）。
    """
    known_set = {str(name).strip() for name in (known or ()) if str(name).strip()}
    seen: dict[str, None] = {}
    lines = _content_lines(_soft_lines(body) if soft else (body or "").splitlines())
    for line in lines:
        if not line.strip():
            continue
        match = _DIALOGUE.match(line)
        name = match.group(1).strip() if match else ""
        if not name:
            enter = _ENTER.match(line)
            name = enter.group(1).strip() if enter else ""
        if not name or name in _NOT_CAST:
            continue
        if len(name) > 4 and name not in known_set:
            continue
        seen.setdefault(name, None)
    return list(seen)


def _opening_line(body: str, *, soft: bool = False) -> str | None:
    """本场第一条非对白行（截 120 字）。这是**引用**不是概括，原文可核。

    围栏里的行不算：否则 ``` 本身会成为 ``opening``（示例代码块的围栏行）。
    """
    lines = _content_lines(_soft_lines(body) if soft else (body or "").splitlines())
    for line in lines:
        text = line.strip()
        if text and not _DIALOGUE.match(text):
            return text[:120]
    return None


def parse_scenes(text: str, *, known_cast: Any = None) -> dict[str, Any]:
    """整篇 → ``{strategy, confidence, scenes, notes, preamble_chars, schema_version}``。

    每场带 ``char_start`` / ``char_end``（原文字符偏移，可直接切片取全文）与
    ``line_start`` / ``line_end``（便于对着原文核）。**偏移一律对原文成立**，调用方
    不要再 strip 正文，否则偏移与内容错位。
    """
    body_text = text or ""
    spans = _line_spans(body_text)
    fenced, fenced_regions = _fenced_lines(spans)
    heads = [
        (index, start, end, parsed)
        for index, (start, end, body) in enumerate(spans)
        if index not in fenced and (parsed := _parse_heading(body)) is not None
    ]
    notes: list[str] = []

    #: ``place-time`` 是弱判据（没场号、没 INT/EXT 字面）：**单条不认**。
    #: 真剧本这类头一定成规模（一部戏几十场）；只有一两条必然是正文里的偶然单行，
    #: 采信它会把整篇切成「(第一场) + 一句废话 + (第三场)」这种假结构。
    place_time_heads = [h for h in heads if h[3]["strategy"] == "place-time"]
    if place_time_heads and len(place_time_heads) < 3:
        heads = [h for h in heads if h[3]["strategy"] != "place-time"]

    #: 非场次的小标题（``## 人物设定`` / ``## 故事梗概``）：它们**终止**当前场。
    #: 否则附录会被算进最后一场的正文里，``opening`` 还会写成 ``## 人物设定`` 这种废话。
    stops = [
        start for index, (start, _end, body) in enumerate(spans)
        if index not in fenced
        and _ANY_MD_HEADING.match(body.strip()) and _parse_heading(body) is None
    ]

    #: 行结构缺失（整篇被压成一两行，OCR 常见）⇒ 按行内标记重切。
    #: 触发条件不能只写 ``not heads``：被压平的文本里第一个 ``场景1`` 命中了，整行就成了
    #: "唯一的一整场"，后面的 ``场景2``/``场景3`` 全被吞进去 —— 必须靠标记数量识破。
    inline_mode = False
    if len(body_text) > 40:
        inline_heads = _inline_heads(body_text, fenced_regions)
        #: 两种要救的情形：①整篇压成一两行（``len(heads) <= 1``）；②OCR 按页断行，
        #: 一页一行、每行中间还压着好几场（折腰：7 个行首头 vs 数百个行内头）。
        #: 正常剧本的行首头数 ≈ 行内标记数，``> 2 倍`` 这个门槛不会误切。
        if len(inline_heads) >= 2 and (len(heads) <= 1 or len(inline_heads) > len(heads) * 2):
            heads = inline_heads
            inline_mode = True
            notes.append(
                f"正文 {len(spans)} 行里场次标记大多不在行首（OCR 按页断行），改为按行内标记"
                f"切分得到 {len(heads)} 场（``strategy=inline``）。"
                "``char_start``/``char_end`` 仍对原文成立。"
            )

    if not heads:
        stripped = body_text.strip()
        if not stripped:
            return {
                "strategy": "none", "confidence": 0.0, "scenes": [],
                "notes": ["正文为空（可能还没 OCR 出文字，或只有图片页）。"],
                "preamble_chars": 0, "schema_version": SCENES_SCHEMA_VERSION,
            }
        char_start = len(body_text) - len(body_text.lstrip())
        return {
            "strategy": "none", "confidence": 0.0,
            "scenes": [{
                "scene_no": None, "no_source": NO_SOURCE_MISSING, "int_ext": None,
                "location": None, "time": None, "heading": None,
                "cast": extract_cast(stripped, known_cast), "opening": _opening_line(stripped),
                "core_event": None, "scene_tail": None,
                "char_start": char_start, "char_end": char_start + len(stripped),
                "chars": len(stripped), "confidence": 0.0, "extractor": "none",
                "line_start": 0, "line_end": len(spans),
            }],
            "notes": [
                "没认出任何场次边界（试过：``## S01`` 小标题、``场景N``、``第N场``、"
                "``INT./EXT.``、集-场号如 ``1-1``、成规模的地点+时段行）"
                "⇒ 整篇记为一场、``no_source=missing``。**不要**据此断言这部剧本只有一场。"
            ],
            "preamble_chars": 0, "schema_version": SCENES_SCHEMA_VERSION,
        }

    preamble_chars = len(body_text[:heads[0][1]].strip())
    preamble_scene: dict[str, Any] | None = None
    if preamble_chars:
        if preamble_chars >= 200:
            #: 首个场次头之前塞了半部戏 ⇒ **不能丢**。单列为一段未分场正文：
            #: ``scene_no=None`` + ``no_source=missing``（不假装它是"第 0 场"，那样统计会失真）。
            head_segment = body_text[:heads[0][1]]
            notes.append(
                f"首个场次头之前有 {preamble_chars} 字，单列为一段未分场正文"
                "（``sceneNo=null``、``noSource=missing``）—— 不丢，也不假装它是第 0 场。"
            )
            preamble_scene = {
                "scene_no": None, "no_source": NO_SOURCE_MISSING, "int_ext": None,
                "location": None, "time": None, "heading": None,
                "cast": extract_cast(head_segment, known_cast),
                "opening": _opening_line(head_segment),
                "core_event": None, "scene_tail": None,
                "char_start": 0, "char_end": heads[0][1],
                "chars": len(head_segment.rstrip()),
                "confidence": 0.0, "extractor": "preamble",
                "line_start": 0, "line_end": heads[0][0],
            }
        else:
            notes.append(f"首个场次头之前有 {preamble_chars} 字（片名/人物表），未计入任何场次。")
    if any(item["no_source"] == NO_SOURCE_MISSING for *_rest, item in heads):
        notes.append("部分场次头没有场号 ⇒ 按出现顺序补生产编号（``no_source=production-assigned``），"
                     "这不是剧本原有编号。⚠️ 若同一篇里另有序号（如集-场号），两套编号各自从 1 起，"
                     "**场号可能重复**，须按 ``noSource`` 区分。")
    if any(item["strategy"] == "ep-scene" for *_rest, item in heads):
        #: 集-场号（``1-1``）的场号**每集重置**，而 schema 没有 episode 列、``scene_no`` 只有一个数。
        #: 不说清楚，用户会以为"第 1 场"重复出现是解析错了。
        notes.append("场次号取自「集-场」写法的**场**段（``1-1`` ⇒ 第 1 场），集号每集重置、"
                     "未进 ``sceneNo``；原文完整标记见每场 ``heading``。")
    if any(item["strategy"] == "hash-scene" for *_rest, item in heads):
        notes.append("场次号取自 ``S#N`` 的 ``N``（韩语／英语剧本写法）；一份里含多集时场号同样会"
                     "从 1 重来，原文标记见每场 ``heading``。")
    if any(item["strategy"] == "place-time" for *_rest, item in heads):
        notes.append("部分场次头只有「地点 + 时段+内外景」、没有场号 ⇒ 按出现顺序补生产编号"
                     "（``strategy=place-time``，confidence 0.6，是位置判断不是原文编号）。")

    scenes: list[dict[str, Any]] = [preamble_scene] if preamble_scene else []
    assigned = 0
    for position, (line_index, _head_start, head_end, parsed) in enumerate(heads):
        next_head_start = heads[position + 1][1] if position + 1 < len(heads) else len(body_text)
        #: 场次之间夹着非场次小标题（人物表/附录）⇒ 在那里截断：附录不是这一场的正文
        inner_stops = [stop for stop in stops if head_end <= stop < next_head_start]
        segment_end = min(inner_stops) if inner_stops else next_head_start
        segment = body_text[head_end:segment_end]
        if parsed["no_source"] == NO_SOURCE_MISSING:
            assigned += 1
            scene_no, no_source = assigned, NO_SOURCE_ASSIGNED
        else:
            scene_no, no_source = parsed["scene_no"], parsed["no_source"]
        next_line = heads[position + 1][0] if position + 1 < len(heads) else len(spans)
        scenes.append({
            "scene_no": scene_no,
            "no_source": no_source,
            "int_ext": parsed["int_ext"],
            "location": parsed["location"],
            "time": parsed["time"],
            "heading": parsed["heading"],
            #: ``inline`` 模式下 segment 是一整段压平正文（没有换行）⇒ 按句读当行看，
            #: 否则 cast 恒为空、opening 会把台词一起吞进去。
            "cast": extract_cast(segment, known_cast, soft=inline_mode),
            "opening": _opening_line(segment, soft=inline_mode),
            "core_event": None,
            "scene_tail": None,
            "char_start": head_end,
            "char_end": segment_end,
            "chars": len(segment.rstrip()),
            "confidence": parsed["confidence"],
            "extractor": "heading",
            "line_start": line_index,
            "line_end": next_line,
        })

    if stops:
        notes.append(f"正文里有 {len(stops)} 处非场次小标题（如 ``## 人物设定``），"
                     "它们之后的内容不计入任何场次，否则附录会被算进最后一场。")

    #: 整篇置信度取最低那一场（不含 preamble）：一篇里有一场没把握，就不该说"这篇解析得很好"。
    #: preamble 的 0.0 不能算进来 —— 否则片头几百字人物表会把 173 场的 0.9 拉成 0.0。
    overall = min((item["confidence"] for item in scenes if item["extractor"] != "preamble"),
                  default=0.0)
    return {
        #: 走"行内切分"时路线本身就是 inline；否则用首个场次头的识别路线
        "strategy": "inline" if inline_mode else heads[0][3]["strategy"],
        "confidence": overall,
        "scenes": scenes,
        "notes": notes,
        "preamble_chars": preamble_chars,
        "schema_version": SCENES_SCHEMA_VERSION,
    }
