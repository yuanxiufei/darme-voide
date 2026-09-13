"""视觉图谱（Visual Graph，移植自 ``backend/src/shared/visual-graph.ts``，215 行）。

对齐参考项目「视觉图谱：景别/构图/运镜/灯光，由 Adviser 图谱驱动提示词」。

四类图谱（知识节点，共 41 个）：

1. ``shot_size``   景别：大远景/远景/全景/中景/中近景/近景/特写/大特写
2. ``composition`` 构图：三分法/对称/引导线/框架/中心/负空间/对角线/前景遮挡/纵深
3. ``movement``    运镜：固定/推/拉/摇/移/跟/升降/环绕/手持/主观/慢动作/滑动变焦/斜角
4. ``lighting``    灯光：自然光/硬光/柔光/逆光/侧光/顶光/低光/暖光/冷光/黄昏/夜景

两类能力：

* :func:`resolve_visual_term` —— 把分镜里的**中文**景别/机位/运镜/灯光映射成**英文电影术语**，
  供 image/video prompt 构造（避免中文术语直接混入英文 prompt 导致生图偏差）；
* :func:`build_visual_graph_guidance` —— 按剧风格生成一段「视觉图谱引导」，注入
  ``storyboard_breaker`` instructions，让拆镜时主动使用图谱。

⚠️ **这张表是数据契约**：``zh`` 别名要与 DB 里的实际取值对得上、``en`` 会进 prompt 文本。
漂移守卫（``tests/route_parity_test.py``）会逐条比对四类的 ``zh``/``en``。
"""
from __future__ import annotations

import re
from typing import Any

__all__ = [
    "VISUAL_GRAPH",
    "build_visual_graph_guidance",
    "get_visual_graph",
    "list_visual_terms",
    "match_visual_term",
    "resolve_visual_term",
    "resolve_visual_terms",
]

#: 四类图谱：每项是 ``{zh: [...别名], en: 英文术语, usage: 使用场景, pairsWith: [...]}``
VISUAL_GRAPH: dict[str, list[dict[str, Any]]] = {
    "shot_size": [
        {"zh": ["大远景", "极远景", "超远景"], "en": "extreme wide shot",
         "usage": "建立空间关系，交代环境与人物位置，适合开场/转场",
         "pairsWith": ["establishing", "symmetrical"]},
        {"zh": ["远景", "远景镜头"], "en": "wide shot",
         "usage": "人物全身入画，展示动作与环境的互动", "pairsWith": ["leading lines"]},
        {"zh": ["全景", "全景镜头"], "en": "full shot",
         "usage": "人物从头到脚完整入画，交代动作起点", "pairsWith": ["rule of thirds"]},
        {"zh": ["中景", "中景镜头"], "en": "medium shot",
         "usage": "膝盖以上入画，对话场景主力景别，兼顾动作与表情", "pairsWith": ["rule of thirds"]},
        {"zh": ["中近景"], "en": "medium close-up",
         "usage": "腰部以上入画，对话微表情与手势", "pairsWith": ["rule of thirds"]},
        {"zh": ["近景", "近景镜头"], "en": "close-up",
         "usage": "肩部以上入画，聚焦表情与情绪", "pairsWith": ["centered", "negative space"]},
        {"zh": ["特写", "特写镜头"], "en": "big close-up",
         "usage": "面部/局部细节，强调关键信息（眼神/手部/道具）", "pairsWith": ["centered"]},
        {"zh": ["大特写", "极特写"], "en": "extreme close-up",
         "usage": "单一细节充满画面（瞳孔/戒指/裂缝），高张力时刻", "pairsWith": ["centered"]},
    ],
    "composition": [
        {"zh": ["三分法", "三分之一"], "en": "rule of thirds",
         "usage": "主体置于画面三分线上，通用且最稳的构图"},
        {"zh": ["对称构图", "居中对称"], "en": "symmetrical composition",
         "usage": "庄严/仪式感/压迫感，适合权力场面与正面冲突"},
        {"zh": ["引导线", "引导线构图", "透视引导"], "en": "leading lines",
         "usage": "用道路/栏杆/视线引导观众目光到主体"},
        {"zh": ["框架构图", "框中框"], "en": "frame within frame",
         "usage": "门框/窗户/栏杆框住主体，增加纵深与窥视感"},
        {"zh": ["中心构图", "居中"], "en": "centered composition",
         "usage": "主体居中，直给与注目感，适合特写"},
        {"zh": ["负空间", "留白"], "en": "negative space",
         "usage": "大面积空白衬托孤独/渺小/呼吸感"},
        {"zh": ["对角线构图", "斜线构图"], "en": "diagonal composition",
         "usage": "动势与不稳定感，适合追逐/冲突"},
        {"zh": ["前景遮挡", "前景虚化"], "en": "foreground obstruction",
         "usage": "前景物体虚化遮挡，营造窥视与空间层次"},
        {"zh": ["纵深构图", "景深分层"], "en": "deep staging",
         "usage": "前景/主体/背景三层景深，交代空间关系"},
    ],
    "movement": [
        {"zh": ["固定", "固定镜头", "静止", "静态", "固定机位"], "en": "static locked-off shot",
         "usage": "机位稳定，适合对白与情绪静观", "pairsWith": ["centered"]},
        {"zh": ["推镜", "推近", "推进", "推镜头", "前推"], "en": "dolly in",
         "usage": "镜头向主体推进，聚焦注意力/施加压力", "pairsWith": ["close-up"]},
        {"zh": ["拉镜", "拉远", "拉出", "拉镜头", "后拉"], "en": "dolly out",
         "usage": "镜头远离主体，揭示环境/孤独感", "pairsWith": ["wide shot"]},
        {"zh": ["摇镜", "摇", "横摇", "摇拍", "摇镜头"], "en": "pan",
         "usage": "机位不动镜头水平转动，扫描环境或追踪横向运动"},
        {"zh": ["横移", "平移", "横向移动", "左移", "右移", "轨道横移"], "en": "lateral tracking",
         "usage": "镜头与主体平行移动，营造旁观感或展示空间连续"},
        {"zh": ["跟拍", "跟镜", "跟随", "跟随镜头", "跟踪"], "en": "tracking shot",
         "usage": "镜头跟随主体移动，身临其境"},
        {"zh": ["升降", "升镜", "降镜", "升降镜头"], "en": "crane shot",
         "usage": "镜头垂直升降，揭示空间尺度或权力关系"},
        {"zh": ["环绕", "环绕拍摄", "环拍", "360环绕", "旋转"], "en": "orbiting shot",
         "usage": "镜头环绕主体，360度审视，揭示全貌"},
        {"zh": ["手持", "手持跟拍", "手持镜头", "手持摄影"], "en": "handheld",
         "usage": "画面轻微晃动，纪实感/慌乱感"},
        {"zh": ["主观视角", "主观镜头", "第一人称"], "en": "point-of-view shot",
         "usage": "角色第一人称视野，代入感强"},
        {"zh": ["慢动作", "升格", "慢镜", "慢镜头"], "en": "slow motion",
         "usage": "升格拍摄，放大情绪/细节/关键瞬间"},
        {"zh": ["滑动变焦", "希区柯克变焦"], "en": "dolly zoom",
         "usage": "主体不动背景压缩/拉伸，眩晕与不安感"},
        {"zh": ["斜角", "荷兰角", "倾斜", "斜拍"], "en": "dutch angle",
         "usage": "地平线倾斜，不安/失衡/精神错乱感"},
    ],
    "lighting": [
        {"zh": ["自然光", "自然光线", "日光"], "en": "natural light",
         "usage": "真实自然光，纪录片感/日常感"},
        {"zh": ["硬光", "强光"], "en": "hard light",
         "usage": "强对比硬阴影，戏剧感/悬疑感"},
        {"zh": ["柔光", "柔光箱", "柔和光"], "en": "soft light",
         "usage": "均匀柔和，商业感/温情画面"},
        {"zh": ["逆光", "背光", "轮廓光", "逆光剪影"], "en": "backlight / rim light",
         "usage": "主体轮廓发光，神圣/神秘/剪影", "pairsWith": ["silhouette"]},
        {"zh": ["侧光", "伦勃朗光", "侧逆光"], "en": "side lighting",
         "usage": "明暗各半，塑造立体感与人物深度", "pairsWith": ["portrait"]},
        {"zh": ["顶光", "顶部光"], "en": "top light",
         "usage": "头顶直射，压抑/审讯感/悬疑"},
        {"zh": ["低光", "暗调", "低调光", "低照度"], "en": "low-key lighting",
         "usage": "大面积阴影，黑色电影/紧张氛围"},
        {"zh": ["暖光", "暖色调", "暖色光"], "en": "warm lighting",
         "usage": "橙黄色温，怀旧/温馨/暧昧"},
        {"zh": ["冷光", "冷色调", "冷色光"], "en": "cool lighting",
         "usage": "蓝青色温，疏离/科技感/夜色"},
        {"zh": ["黄昏", "金色时刻", "黄金时刻"], "en": "golden hour light",
         "usage": "日落前暖橙光线，浪漫/告别/高光时刻"},
        {"zh": ["夜景", "夜晚光线"], "en": "night scene lighting",
         "usage": "夜色照明，城市霓虹或月光，静谧/危险"},
    ],
}

#: 扁平化英文术语表：中文 → (英文, 类别)，供 :func:`match_visual_term` 快速查表。
#: ⚠️ **插入顺序即包含匹配的优先级**（同名别名不重复，先到先得）。
EN_BY_ZH: dict[str, tuple[str, str]] = {}
for _category, _nodes in VISUAL_GRAPH.items():
    for _node in _nodes:
        for _zh in _node["zh"]:
            EN_BY_ZH.setdefault(_zh, (_node["en"], _category))


def match_visual_term(zh_text: str | None) -> dict[str, str] | None:
    """匹配中文视觉术语 → ``{en, category}``；未命中返回 None。

    匹配策略（与 ``camera-movement-guides`` 一致）：

    1. **精确匹配优先**（含别名）；
    2. **包含匹配兜底**（如「快速横移」包含「横移」），且**术语长度 > 1**
       （单字术语太容易误命中）。
    """
    text = (zh_text or "").strip()
    if not text:
        return None

    if text in EN_BY_ZH:
        en, category = EN_BY_ZH[text]
        return {"en": en, "category": category}

    for zh, (en, category) in EN_BY_ZH.items():
        if len(zh) > 1 and zh in text:
            return {"en": en, "category": category}
    return None


def resolve_visual_term(zh_text: str | None) -> str | None:
    """把中文视觉术语映射为英文电影术语（**不考虑类别**，适合单字段翻译）；未命中返回 None。"""
    match = match_visual_term(zh_text)
    return match["en"] if match else None


def resolve_visual_terms(category: str, values: list[str | None]) -> list[str]:
    """按类别批量翻译：**仅当命中术语属于目标类别时**输出英文，否则保留原文。"""
    out: list[str] = []
    for value in values:
        if not value:
            continue
        match = match_visual_term(value)
        if match and match["category"] == category:
            out.append(match["en"])
        else:
            out.append(value)
    return out


def list_visual_terms(category: str) -> list[str]:
    """获取某类图谱的全部英文术语列表（供 prompt 枚举）。"""
    return [node["en"] for node in VISUAL_GRAPH[category]]


def get_visual_graph(category: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """获取图谱某个类别的全部节点（供前端展示图谱库 / 生成图谱文档）。"""
    if category:
        return {category: VISUAL_GRAPH[category]}
    return VISUAL_GRAPH


#: 风格特征词 → 推荐图谱节点（顺序即优先级）
_STYLE_HINTS: tuple[tuple[str, dict[str, list[str]]], ...] = (
    (
        r"(悬疑|惊悚|犯罪|推理|暗黑|黑帮)",
        {"size": ["近景/特写", "中近景"], "comp": ["框架构图"],
         "light": ["低光", "硬光", "侧光", "顶光"], "move": ["固定", "缓慢推镜", "手持"]},
    ),
    (
        r"(爱情|甜宠|温情|治愈|家庭)",
        {"size": ["中景", "中近景", "近景"], "comp": ["三分法", "留白/负空间"],
         "light": ["柔光", "暖光", "黄昏/金色时刻"], "move": ["固定", "缓慢拉镜"]},
    ),
    (
        r"(玄幻|武侠|古装|仙侠|奇幻|神话)",
        {"size": ["全景", "远景"], "comp": ["对称构图", "引导线", "纵深构图"],
         "light": ["逆光", "冷光", "黄昏/金色时刻"], "move": ["环绕", "升降", "慢动作"]},
    ),
    (
        r"(都市|职场|现实|商战|行业)",
        {"size": ["中景", "近景", "全景"], "comp": ["三分法", "引导线", "框架构图"],
         "light": ["自然光", "侧光", "硬光"], "move": ["固定", "跟拍", "横移"]},
    ),
    (
        r"(科幻|末世|末日|未来|赛博)",
        {"size": ["大远景", "特写"], "comp": ["对称构图", "引导线", "负空间"],
         "light": ["冷光", "低光", "硬光"], "move": ["环绕", "升降", "滑动变焦"]},
    ),
    (
        r"(动作|警匪|打斗|复仇|战斗)",
        {"size": ["中景", "全景", "特写"], "comp": ["对角线构图", "前景遮挡"],
         "light": ["硬光", "低光", "冷光"], "move": ["手持", "快速横移", "跟拍", "慢动作"]},
    ),
)


def build_visual_graph_guidance(genre: str | None = None, style: str | None = None) -> str:
    """按剧风格生成「视觉图谱引导」文本（注入 ``storyboard_breaker`` instructions）。

    依据风格选择强调的图谱节点，并给出景别递进/构图节奏/运镜动机/灯光氛围的通用创作指导。
    ⚠️ 命中的风格**可以叠加**（例如「悬疑+都市」两组提示都会出现），顺序按上表固定。
    """
    combined = f"{(genre or '').lower()}{(style or '').lower()}"

    size_hints: list[str] = []
    comp_hints: list[str] = []
    move_hints: list[str] = []
    light_hints: list[str] = []
    for pattern, hints in _STYLE_HINTS:
        if re.search(pattern, combined):
            size_hints.extend(hints["size"])
            comp_hints.extend(hints["comp"])
            light_hints.extend(hints["light"])
            move_hints.extend(hints["move"])

    parts: list[str] = []
    genre_note = f"（{genre}）" if genre else ""
    style_note = f" / 风格（{style}）" if style else ""
    parts.append(f"【视觉图谱引导】根据本剧类型{genre_note}{style_note}，拆镜时主动使用以下视觉图谱：")
    parts.append("")
    parts.append("1. 景别递进：")
    parts.append("   - 开场/转场优先远景或全景建立空间；对话主力中景/中近景；情绪高潮推进到近景/特写；")
    parts.append("   - 相邻镜头避免连续同景别（如需同景别必须换机位或换构图，否则画面跳剪感强）；")
    if size_hints:
        parts.append(f"   - 本剧推荐景别：{' / '.join(size_hints)}。")
    parts.append("")
    parts.append("2. 构图法则（每个镜头至少明确一种）：")
    parts.append("   - 三分法/对称/引导线/框架/中心/负空间/对角线/前景遮挡/纵深，按剧情意图选择；")
    parts.append("   - 构图意图要写进 image_prompt 与 first_frame_prompt/last_frame_prompt 的构图描述。")
    if comp_hints:
        parts.append(f"   - 本剧推荐构图：{' / '.join(comp_hints)}。")
    parts.append("")
    parts.append("3. 运镜动机（每个镜头运镜必须有叙事理由）：")
    parts.append("   - 推镜=聚焦/施压，拉镜=揭示/孤独，摇镜=扫描/追踪，跟拍=身临其境，环绕=审视，手持=慌乱纪实；")
    parts.append("   - movement 字段写中文运镜名（如「缓慢推镜」「360环绕」），不要写英文。")
    if move_hints:
        parts.append(f"   - 本剧推荐运镜：{' / '.join(move_hints)}。")
    parts.append("")
    parts.append("4. 灯光氛围（atmosphere 必须包含光线描述，体现「主光方向+光质+色温」）：")
    parts.append("   - 自然光/硬光/柔光/逆光/侧光/顶光/低光/暖光/冷光/黄昏/夜景，按情绪选择；")
    parts.append("   - 逆光配剪影制造神秘，侧光塑造立体，低光营造紧张，暖光传递温情。")
    if light_hints:
        parts.append(f"   - 本剧推荐灯光：{' / '.join(light_hints)}。")
    parts.append("")
    parts.append(
        "5. 一致性与安全区：以上图谱约束连同安全区/质感层/Mx-Shell 规则一并写入各镜 prompt，保持全剧视觉风格统一。"
    )

    return "\n".join(parts)
