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

from app.services.scene_parser import parse_scenes

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

#: md 小标题的行前缀。识别本身已统一到 ``scene_parser``，这里只用来还原 ``raw``
#: （老调用方拿 ``raw`` 做"字面是否出现在标题里"的判断，不想要 ``## ``）
_MD_PREFIX = re.compile(r"^\s{0,3}#{1,6}\s*")

#: 时段同义归一：同一「打光时段」写成不同词不该算两个场景
_TIME_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("凌晨", ("凌晨", "拂晓", "黎明前", "后半夜")),
    ("清晨", ("清晨", "黎明", "早晨", "早上", "日出")),
    ("上午", ("上午",)),
    ("白天", ("白天", "日间", "正午", "中午", "午间", "午后", "下午", "白日")),
    ("黄昏", ("黄昏", "傍晚", "日暮", "日落", "擦黑", "暮色")),
    ("夜晚", ("夜晚", "夜里", "夜间", "晚上", "深夜", "午夜", "半夜", "夜色", "入夜", "夜")),
)





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

    **识别不在这里**：统一走 :func:`app.services.scene_parser.parse_scenes`（五条路线
    ``md-heading`` / ``cn-numbered`` / ``inline`` / ``slug``，外加围栏排除与 preamble 单列）。
    本函数只做转换：把 ``{heading, location, time}`` 还原成老调用方要的形状，并丢掉"地点与
    时段都读不出"的光杆标题。

    为什么必须统一（2026-11-04 实测）：旧实现把**任何** ``##`` 标题都当场景，于是
    ``## 人物设定`` 会变成 ``location="人物设定"`` 的假场景 —— ``_missing_script_scenes``
    把它当"剧本里有、场景表里没有"回灌给模型，要求"为这一场保存场景"，可剧本里根本没有这一场。
    同时旧实现认不出 ``1. 内景 客厅 夜``、也认不出被 OCR 压平的正文 ⇒ 判据**过宽与过窄同时存在**。
    两套实现并存时，改好一套另一套不跟，就是下一次"看起来能用"的坑。
    """
    headings: list[dict[str, str]] = []
    for item in parse_scenes(script)["scenes"]:
        if item["extractor"] == "preamble":
            continue  # 首个场次头之前的说明文字不是一场戏
        location = (item["location"] or "").strip()
        time_part = (item["time"] or "").strip()
        if not location and not time_part:
            continue  # 「场景覆盖」比不了光杆标题（``## S03``），留着只会制造假缺口
        raw = _MD_PREFIX.sub("", (item["heading"] or "").strip()).strip()
        headings.append({"raw": raw, "location": location, "time": time_part})
    return headings
