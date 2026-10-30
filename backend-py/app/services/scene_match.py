"""剧本场景 ↔ ``scenes`` 表地点/时段的匹配规则（提取去重与陈旧关联清理共用）。

为什么需要这个模块（2026-10-30 实测）：``extract_tools.save_dedup_scenes`` 的去重键是
**location + time 逐字相等**，而模型每次重跑给的地点/时段都会漂一点点：

* 地点：``民国上海街头`` / ``民国上海街道``、``东家大厅`` / ``东家商号大厅``；
* 时段：``夜晚`` / ``深夜``。

于是一集重跑三次就长出 8 个场景行（真实只有 3 个地点），而且**旧行仍挂在本集上** ——
分镜阶段读到的是全部 8 个，镜头就绑到了上一版剧本的场景（实测：剧本 S01/S02/S03 已换成
场景 7/8/9，镜头却绑 5/6）。地点与时段都按**同义归一**后再比，才既有去重又有清理。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

__all__ = [
    "LOCATION_MATCH_RATIO",
    "locations_match",
    "normalize_location",
    "normalize_time",
    "same_scene",
    "script_scene_headings",
]

#: 判定「同一个地点」的相似度阈值。实测同义写法（街头/街道、东家大厅/东家商号大厅）
#: ≥ 0.82；而只共享前缀的不同地点（东家商号大厅 / 东家商号书房）≈ 0.67 —— 留足间隔。
LOCATION_MATCH_RATIO = 0.82

#: 归一化时要丢掉的标点/分隔符
_PUNCT = re.compile(r"[\s·・,，。、:：;；|｜\-—_/\\()（）\[\]【】<>《》\"'“”‘’]")

#: 场景小标题行（``## S01 | 外景 · 民国上海街道 | 黄昏``）
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*(.+)$")

#: 小标题里的「内景/外景」前缀
_INDOOR_OUTDOOR = re.compile(r"^(内景|外景|日景|夜景|日|夜)\s*[·・.、:：]?\s*")

#: 小标题开头的场景编号（``S01 `` / ``场景2：``）
_HEADING_ID = re.compile(r"^(S\d+|场景\s*\d+|第\s*\d+\s*场)\s*[·・:：.、]?\s*")

#: 时段同义归一：同一「打光时段」写成不同词不该算两个场景
_TIME_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("凌晨", ("凌晨", "拂晓", "黎明前", "后半夜")),
    ("清晨", ("清晨", "黎明", "早晨", "早上", "日出")),
    ("上午", ("上午",)),
    ("白天", ("白天", "日间", "正午", "中午", "午间", "午后", "下午", "白日")),
    ("黄昏", ("黄昏", "傍晚", "日暮", "日落", "擦黑", "暮色")),
    ("夜晚", ("夜晚", "夜里", "夜间", "晚上", "深夜", "午夜", "半夜", "夜色", "入夜", "夜")),
)


#: 能被认出来的「时段桶名」（解析小标题时判断末尾那个词是不是时段）
_TIME_BUCKET_NAMES = frozenset(bucket for bucket, _ in _TIME_BUCKETS)


def normalize_location(value: str | None) -> str:
    """地点归一：去标点空白、小写（中文无大小写，主要去掉符号差异）。"""
    return _PUNCT.sub("", (value or "").strip().lower())


def normalize_time(value: str | None) -> str:
    """时段归一：同义词折进同一个「打光时段」桶；认不出来就返回原串。"""
    text = (value or "").strip()
    if not text:
        return ""
    for bucket, words in _TIME_BUCKETS:
        if any(word in text for word in words):
            return bucket
    return text


def locations_match(left: str | None, right: str | None,
                    threshold: float = LOCATION_MATCH_RATIO) -> bool:
    """两个地点写法是否指同一个地方。"""
    a = normalize_location(left)
    b = normalize_location(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def same_scene(location_a: str | None, time_a: str | None,
               location_b: str | None, time_b: str | None) -> bool:
    """地点同义 **且** 时段同桶 —— 才算同一条场景（同地点不同时段仍是两个场景）。"""
    if not locations_match(location_a, location_b):
        return False
    bucket_a = normalize_time(time_a)
    bucket_b = normalize_time(time_b)
    if not bucket_a or not bucket_b:
        # 有一边没写时段：不用时段否决（地点已经对上，重复建行更糟）
        return True
    return bucket_a == bucket_b


def script_scene_headings(script: str) -> list[dict[str, str]]:
    """把剧本里的场景小标题拆成 ``{raw, location, time}``。

    认的是平台剧本格式 ``## S01 | 外景 · 民国上海街道 | 黄昏``；也兼容
    ``## 场景1：东家大厅 夜`` 这种没有竖线的写法（整段当地点，时段留空）。
    """
    headings: list[dict[str, str]] = []
    for line in (script or "").splitlines():
        match = _HEADING.match(line)
        if match is None:
            continue
        text = match.group(1).strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"[|｜]", text) if part.strip()]
        if len(parts) >= 2:
            location = parts[1]
            time_part = parts[2] if len(parts) >= 3 else ""
        else:
            # 无竖线写法：``S01 · 东家大厅 夜`` / ``场景2：东家大厅 夜``
            residual = _HEADING_ID.sub("", parts[0] if parts else "").strip()
            head, _, tail = residual.rpartition(" ")
            # 只有末尾那个词**确实是时段词**（夜/黄昏/清晨…）才把它当时段
            if head.strip() and tail.strip() and normalize_time(tail) in _TIME_BUCKET_NAMES:
                location, time_part = head.strip(), tail.strip()
            else:
                location, time_part = residual, ""
        location = _INDOOR_OUTDOOR.sub("", location).strip()
        headings.append({"raw": text, "location": location, "time": time_part})
    return headings
