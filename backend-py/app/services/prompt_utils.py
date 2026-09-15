"""统一视觉 Prompt 工具 —— 移植 ``shared/prompt-utils.ts``（**分块进行，本文件为第 1 块**）。

解决的原始问题：全项目 prompt 风格碎片化（角色图一套风格词、场景图另一套、宫格图第三套、视频无风格），
本模块把风格预设与 prompt 构建器统一到一处。

提示词分层契约（**改前先读**）：越靠上越通用、越靠下越具体::

    L1 身份/工作流层  agents/index.ts 的 DEFAULT_PROMPTS（agent 是谁、按什么流程干活）
    L2 画风层         ART_STYLE_CATALOG + DRAMA/EQUIP_ART_STYLE_MAP（10 种画风的四段词表）
    L3 画质收口层     QUALITY_TAIL（中性基准，静帧/视频各持一个引用常量）
    L4 瑕疵锚点层     IMPERFECTION_ANCHORS（仅写实向白名单注入）
    L5 负面层         NEGATIVE_*（各链路负面词，统一拼 NEGATIVE_BASE）

硬性规则（**这三条是历史事故换来的，别"顺手优化"**）：

1. **命中画风用 ``art, ART_STYLE_TAIL``；未命中才回退 ``VISUAL_STYLE_MASTER`` —— 二者二选一，禁止叠加**。
   MASTER 含 ``cinematic illustration style``，叠加会把水墨/动漫拉回写实插画。
2. **视频链路只用 ``VIDEO_*`` 家族**，不得引用 ``ART_STYLE_*`` / ``VISUAL_STYLE_MASTER``。
3. 各路由与服务**不得自建画风副本**；唯一入口是 ``resolve_effective_art_style`` + ``build_*_art_style_suffix``。

────────── 移植进度（原文件 1629 行）──────────
✅ 第 1 块：全域风格常量 + 画风目录/三张映射表 + 画风解析 + 各链路正/负面词后缀（原 L1–480）
✅ 第 2 块：预设 prompt + 角色/装备/单品/表情/物品/场景构建器（原 L482–900，**无外部依赖**）
✅ 第 3 块：分镜静帧/分镜视频 prompt 构建器（原 L902–1020；**依赖 `visual_graph.resolve_visual_term`**）
✅ 第 4 块：DB 读取辅助（分镜出场角色/场景描述/参考图/参考音频）（原 L1029–1195）
✅ 第 5 块：台词角色一致性校验 + 视频 prompt 标签剥离（原 L1207–1304）
✅ 第 6 块：宫格 prompt（参考资产收集 / 角度表 / buildGridPrompt / buildGridCellPrompts）
   （原 L1314–1628；**依赖 `camera_movement_guides`**）—— 至此 ``prompt-utils.ts`` **全量迁完**

⚠️ **本模块的值全部要与 TS 源逐字一致** —— 词表改一个字就会改变生成结果。
``tests/route_parity_test.py`` 里有**漂移守卫**从 ``prompt-utils.ts`` 抽取字符串常量逐一比对。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection

from ..core.models import (
    app_settings,
    characters,
    dramas,
    prop_templates,
    scenes,
    storyboard_characters,
    storyboard_props,
    storyboards,
)
from ..core.response import js_truthy
from .camera_movement_guides import get_camera_movement_composition
from .storyboard_helpers import parse_dialogue_for_tts
from .visual_graph import resolve_visual_term


def _col(row: Any, name: str) -> Any:
    """取行字段（不存在时 None）—— 与 ``image_generation._col`` 同一套容错写法。"""
    return getattr(row, name, None)

# ===========================================================================
# L1 前置：全域风格常量
# ===========================================================================

#: 主视觉风格标签 —— 所有角色/场景/分镜图像生成共用
VISUAL_STYLE_MASTER = "cinematic illustration style, consistent art style, soft cinematic lighting, high quality, no text, no watermark"

#: 角色立绘专用风格标签
VISUAL_STYLE_CHARACTER = "stylized character design, front-facing character portrait, clean white background, illustration style, not photorealistic"

#: 场景背景专用风格标签
VISUAL_STYLE_SCENE = "highly detailed cinematic environment, atmospheric lighting, movie quality composition, depth of field"

#: 视频生成风格引导前缀。
#: 对齐 Mx-Shell 质感层规范：真实镜头锚点（35mm/50mm、浅景深）、现实瑕疵锚点、克制结尾（不搞爆炸/胜利姿势）、
#: 声音策略（画面内不含 BGM，后期统一配乐）。
VISUAL_STYLE_VIDEO = (
    "cinematic motion, smooth camera movement, consistent character design, lighting continuity, "
    "real lens feel with natural depth of field (35mm/50mm cinematic lens), subtle film grain, "
    "grounded realistic texture with small natural imperfections, restrained ending (no explosion, no victory pose, no text overlay)"
)

#: UI 屏幕留白规则：画面中的屏幕/UI 元素只保留留白，禁止生成可读文字（一律后期叠加），防 AI 生成乱码文字。
UI_OVERLAY_RULE = "screens and displays (phone/computer/news/SMS/surveillance/map/clock) show only blank or neutral panels: no readable text, no legible characters, no app icons with text; all on-screen text is added in post-production"

#: 屏幕/UI 元素关键词（用于检测分镜描述是否含屏幕画面）
_SCREEN_ELEMENT_PATTERN = re.compile(
    r"手机|短信|屏幕|显示器|监控|地图|电脑|平板|文件|时间码|新闻|聊天|打字|发消息|"
    r"phone|screen|display|monitor|surveillance|map|computer|tablet|text message|news|sms|chat",
    re.IGNORECASE,
)

#: 屏幕留白图类别名（props 物品库用）
UI_PLATE_CATEGORY = "屏幕留白"


def contains_screen_element(text: str | None) -> bool:
    """检测文本中是否包含屏幕/UI 元素。"""
    if not text:
        return False
    return _SCREEN_ELEMENT_PATTERN.search(text) is not None


def build_ui_plate_image_prompt(plate: dict[str, Any]) -> str:
    """构建屏幕留白图（ui_plate）prompt：供后期叠加文字。"""
    plate_type = plate.get("type") or "phone screen"
    parts = [f"blank {plate_type} plate, no text, no readable characters, no icons with text"]
    if plate.get("context"):
        parts.append(f"context: {plate['context']}")
    parts.append(
        "flat neutral screen surface ready for post-production text overlay, empty UI layout"
    )
    return f"{', '.join(parts)}, {VISUAL_STYLE_SCENE}, {VISUAL_STYLE_MASTER}"


# ===========================================================================
# L5 负面层基础
# ===========================================================================

#: 通用负面基础（所有生成类型共用）
NEGATIVE_BASE = "text, watermark, signature, logo, subtitles, low quality, blurry, out of focus, pixelated, jpeg artifacts, oversaturated, distorted, deformed, disfigured, bad anatomy, extra limbs, extra fingers, mutated hands, bad proportions, duplicated elements"

#: 角色立绘负面提示词（排除写实照片质感、杂乱/多人背景、角色重复与裁切）
CHARACTER_IMAGE_NEGATIVE = f"{NEGATIVE_BASE}, photorealistic, realistic photo, 3d render, cluttered background, busy background, multiple characters, multiple people, duplicated character, inconsistent character, cropped head, cut off face"


# ===========================================================================
# L2 画风层：目录 / 映射表 / 解析
# ===========================================================================

#: 画风目录（**画风的单一事实来源**：key + 中文名 + 说明 + 分组）。
#:
#: ⚠️ 后端本文件为事实来源，前端 ``frontend/app/utils/artStyles.ts`` 为对齐镜像。
#: 新增画风必须同时改三处：本目录、``DRAMA_ART_STYLE_MAP``、前端镜像文件，
#: 否则前端下拉会出现「选了没效果」或「有词但选不到」。
ART_STYLE_CATALOG: list[dict[str, str]] = [
    {"key": "realistic", "label": "写实电影", "shortLabel": "写实", "desc": "真人质感、电影级镜头语言与胶片调色", "group": "实拍质感"},
    {"key": "cinematic", "label": "电影感", "shortLabel": "电影感", "desc": "商业电影帧、强氛围与色彩分级", "group": "实拍质感"},
    {"key": "noir", "label": "黑色电影", "shortLabel": "黑色", "desc": "黑白高反差、硬光影、1940s 侦探片质感", "group": "实拍质感"},
    {"key": "anime", "label": "日式动漫", "shortLabel": "动漫", "desc": "赛璐璐上色、鲜明线条、动画 key visual", "group": "绘画插画"},
    {"key": "ghibli", "label": "吉卜力", "shortLabel": "吉卜力", "desc": "手绘质感、温暖配色、治愈系", "group": "绘画插画"},
    {"key": "ink-wash", "label": "国风水墨", "shortLabel": "水墨", "desc": "水墨晕染、留白意境、宣纸质感", "group": "绘画插画"},
    {"key": "watercolor", "label": "水彩", "shortLabel": "水彩", "desc": "水彩晕染、柔和过渡、纸面纹理", "group": "绘画插画"},
    {"key": "comic", "label": "美漫漫画", "shortLabel": "漫画", "desc": "美式漫画、粗线条、网点阴影", "group": "三维与漫画"},
    {"key": "cyberpunk", "label": "赛博朋克", "shortLabel": "赛博", "desc": "霓虹雨夜、高饱和洋红青、全息光斑", "group": "三维与漫画"},
    {"key": "pixar3d", "label": "三维动画", "shortLabel": "3D", "desc": "皮克斯式三维渲染、圆润造型、柔和全局光", "group": "三维与漫画"},
]

#: 全部合法画风 key（校验用）
ART_STYLE_KEYS: list[str] = [option["key"] for option in ART_STYLE_CATALOG]


def is_valid_art_style(style: str | None) -> bool:
    """画风是否合法（**含空值放行** —— 空值表示「跟随继承」）。"""
    if not style:
        return True
    return style in ART_STYLE_KEYS


#: 戏剧视觉风格 → 英文画风描述。
#:
#: 每条按「画风核心 + 镜头/光线 + 调色/质感 + 画质」四段组织，而非单个形容词 ——
#: 模型对镜头规格、光位、胶片调色的响应远强于空泛的风格名。写实向靠镜头/胶片锚点，
#: 动漫画向靠渲染方式锚点（cel shading、flat color）。
DRAMA_ART_STYLE_MAP: dict[str, str] = {
    "realistic": "photorealistic cinematic film still, 35mm anamorphic lens, shallow depth of field, natural skin texture, practical light sources, subtle rim light, teal and orange color grading, fine film grain, Kodak film emulation",
    "cinematic": "cinematic film still style, dramatic moody lighting, shallow depth of field, anamorphic lens flare, high contrast low-key lighting, blockbuster color grading, volumetric atmosphere, letterbox composition",
    "noir": "classic film noir style, black and white high contrast, hard chiaroscuro lighting, venetian blind shadow patterns, deep blacks with silver highlights, 1940s cinematography, cigarette smoke haze, tense dutch angle framing",
    "anime": "2D anime illustration style, clean line art, cel shading, flat color, vibrant saturated palette, detailed hair highlights, expressive eyes, Japanese animation key visual quality",
    "ghibli": "ghibli-inspired hand-drawn animation style, soft watercolor background, gentle rounded character design, warm nostalgic palette, hand-painted texture, natural daylight, quiet slice-of-life atmosphere",
    "ink-wash": "Chinese ink wash painting style, expressive brush strokes, flowing ink diffusion, generous negative space, muted mineral pigments, rice paper texture, traditional gongbi-and-freestyle fusion",
    "watercolor": "watercolor painting style, soft brush strokes, translucent washes, delicate pastel palette, visible paper grain, bleeding edges, light washes with dry brush accents",
    "comic": "comic book illustration style, bold ink line work, halftone shading, dynamic comic character design, high contrast primary colors, cross-hatching texture, American superhero comic inking",
    "cyberpunk": "cyberpunk aesthetic, neon-lit rainy night, high saturation magenta and cyan glow, holographic signage bokeh, wet reflective streets, strong contrast, dystopian metropolitan atmosphere",
    "pixar3d": "stylized 3D animation render, Pixar-inspired character design, subsurface scattering skin, soft global illumination, rounded appealing proportions, vibrant colors, shallow depth of field",
}

#: 各画风对应的**对立风格负面词**（防止模型混用动漫/真人/三维，是画风稳定的关键）
_DRAMA_ART_NEGATIVE_MAP: dict[str, str] = {
    "realistic": "anime style, cartoon, illustration, cel shading, line art, 3d render",
    "cinematic": "anime style, cartoon, illustration, flat coloring",
    "noir": "color, vibrant colors, saturated palette, anime, cartoon, flat coloring",
    "anime": "photorealistic, realistic photo, live action, 3d render",
    "ghibli": "photorealistic, realistic photo, live action, 3d render",
    "ink-wash": "photorealistic, realistic photo, 3d render, harsh digital outline, oil painting, neon colors",
    "watercolor": "photorealistic, realistic photo, live action, harsh outline",
    "comic": "photorealistic, realistic photo, live action",
    "cyberpunk": "anime style, cartoon, flat 2d illustration, dull desaturated colors, daylight",
    "pixar3d": "photorealistic, realistic photo, live action, 2d anime, cel shading, flat coloring",
}

#: 装备/服装/武器/首饰图的画风尾（**无人物语义**）。
#: ``DRAMA_ART_STYLE_MAP`` 是为角色像准备的（含 character design / skin texture），
#: 直接用于「纯物品」三视图会**诱导模型生成人物或穿着人体的服装**，故单独定义。
_EQUIP_ART_STYLE_MAP: dict[str, str] = {
    "realistic": "professional product photography, studio softbox lighting, clean neutral background, sharp fabric and metal texture detail",
    "cinematic": "cinematic concept art style, dramatic moody lighting, shallow depth of field",
    "noir": "black and white product photography, hard directional lighting, deep shadows, high contrast silver highlights",
    "anime": "2D anime prop illustration style, clean line art, cel shading, vibrant colors",
    "ghibli": "ghibli-inspired hand-drawn art style, soft colors, gentle shading",
    "ink-wash": "Chinese ink wash painting of a single object, expressive brush strokes, rice paper texture, muted mineral pigments",
    "watercolor": "watercolor painting style, soft brush strokes, translucent washes, delicate pastel palette",
    "comic": "comic book illustration style, bold ink line work, halftone shading",
    "cyberpunk": "cyberpunk product shot, neon magenta and cyan rim lighting, wet reflective surface, dark background",
    "pixar3d": "stylized 3D product render, soft global illumination, rounded appealing form, clean studio background",
}

# ===========================================================================
# L3/L4 收口层
# ===========================================================================

#: 通用画质收口基准词（中性、不含任何画风倾向，避免与具体绘画/实拍画风互斥）
QUALITY_TAIL = "consistent art style, high quality, no text, no watermark"

#: 静帧画风层尾部收口。与 ``VIDEO_STYLE_TAIL`` 当前取值相同，但**语义独立**：
#: 静帧与视频允许各自演进（改视频不得影响静帧），故保留两个常量名。
ART_STYLE_TAIL = QUALITY_TAIL

#: 视频通用运动层：全画风共用，**刻意不含 lens / film grain 等写实专属词**
VIDEO_MOTION_BASE = "cinematic motion, smooth camera movement, consistent character design, lighting continuity"

#: 视频画风层尾部收口（写实/绘画通用，不含写实专属词）
VIDEO_STYLE_TAIL = QUALITY_TAIL

#: 反 AI 感锚点（**仅写实/实拍向画风注入**）。
#: 来源：Seedance2 语料实测高频词族 ——「可控的现实缺陷」（构图不完美 / 对焦不完美 / 环境瑕疵 /
#: 轻微手持不稳 / 自然高感颗粒）比堆砌 ``8k, ultra detailed`` 更能压掉塑料感。
#: 绘画向画风（动漫/水墨/水彩/漫画/三维）注入这些词会**直接破坏风格**，故按白名单注入。
IMPERFECTION_ANCHORS = "natural imperfect composition, slightly imperfect autofocus, believable environmental imperfections, subtle handheld micro-shake, natural high-ISO grain"

#: 需要注入镜头/胶片/现实瑕疵锚点的写实向画风
PHOTOREAL_ART_STYLES = ["realistic", "cinematic", "noir", "cyberpunk"]


# ===========================================================================
# 画风解析（全链路唯一入口）
# ===========================================================================

def get_global_art_style(conn: Connection) -> str | None:
    """读取全局默认画风（``app_settings.art_style``），未设置返回 None。"""
    row = conn.execute(
        select(app_settings).where(app_settings.c.key == "art_style")
    ).first()
    if row is None:
        return None
    return row.value or None


def get_drama_art_style(conn: Connection, drama_id: int | None = None) -> str | None:
    """读取剧集视觉风格（``dramas.style``），剧集不存在或未设置返回 None。"""
    if not drama_id:
        return None
    row = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
    if row is None:
        return None
    return row.style or None


def resolve_art_style_key(*candidates: str | None) -> str:
    """画风候选链解析（纯函数）：返回第一个**合法画风 key**。

    非法/未知 key 被**跳过而非透传** —— 避免脏值静默打断整条 prompt 的画风收口。
    """
    for candidate in candidates:
        if candidate and candidate in DRAMA_ART_STYLE_MAP:
            return candidate
    return "realistic"


def resolve_effective_art_style(
    conn: Connection,
    drama_id: int | None = None,
    char_style: str | None = None,
    drama_style: str | None = None,
) -> str:
    """全链路画风解析入口（DB 感知）。

    优先级：角色 style → 剧集 style → 全局默认（``app_settings.art_style``）→ ``realistic``。

    ⚠️ 各路由/生成服务**统一调用本函数**，不要再各自复制解析逻辑
    （历史上有 ``dramas.ts`` / ``characters.ts`` 两处重复实现）。

    参数 ``drama_style`` 是「已知的剧集画风」，用于避免重复查库（``?? `` 语义：传了就不查）。
    """
    resolved_drama = drama_style if drama_style is not None else get_drama_art_style(conn, drama_id)
    return resolve_art_style_key(char_style, resolved_drama, get_global_art_style(conn))


# ===========================================================================
# 各链路画风后缀（正词）
# ===========================================================================

def build_character_art_style_suffix(drama_style: str | None = None) -> str:
    """角色图统一画风后缀（追加在 prompt 末尾）。

    未指定 ``drama_style`` → 维持默认 stylized illustration（not photorealistic）；
    指定 → 优先使用映射画风，同剧强制一致。
    """
    art = (drama_style and DRAMA_ART_STYLE_MAP.get(drama_style)) or ""
    if not art:
        return f", {VISUAL_STYLE_CHARACTER}, {VISUAL_STYLE_MASTER}"
    return (
        f", {art}, cinematic illustration style, consistent art style, "
        "soft cinematic lighting, high quality, no text, no watermark"
    )


def build_equip_art_style_suffix(drama_style: str | None = None) -> str:
    """装备图统一画风/无人物后缀（追加在 equip prompt 末尾）。"""
    art = (drama_style and _EQUIP_ART_STYLE_MAP.get(drama_style)) or ""
    base = (
        ", isolated product shot, single object, empty scene, no people, no person, no model, "
        "no mannequin body, no body parts, no skin"
    )
    if not art:
        return f"{base}, stylized concept art, high quality, no text, no watermark"
    return f"{base}, {art}, high quality, no text, no watermark"


def build_scene_art_style_suffix(drama_style: str | None = None) -> str:
    """场景图（无人物环境）画风尾：与角色立绘**共用同一份画风词表**，
    保证同剧的「人」与「景」不会各自跑偏。"""
    art = (drama_style and DRAMA_ART_STYLE_MAP.get(drama_style)) or ""
    if not art:
        return f", {VISUAL_STYLE_SCENE}, {VISUAL_STYLE_MASTER}"
    return f", {art}, {ART_STYLE_TAIL}"


def build_storyboard_art_style_suffix(drama_style: str | None = None) -> str:
    """分镜静帧画风尾（含人物，与角色/场景链路同源）。"""
    art = (drama_style and DRAMA_ART_STYLE_MAP.get(drama_style)) or ""
    if not art:
        return f", {VISUAL_STYLE_MASTER}"
    return f", {art}, {ART_STYLE_TAIL}"


def build_video_art_style_suffix(drama_style: str | None = None) -> str:
    """视频画风层（画风核心 + 运动层 + 现实瑕疵锚点 + 克制结尾）。

    视频不能复用静帧的 ``cinematic illustration style`` 收口，也不能无条件叠加写实镜头词：
    绘画向画风只保留「画风核心 + 运动层」；写实向额外注入镜头/胶片/瑕疵锚点。
    未指定画风时回退 ``VISUAL_STYLE_VIDEO``（保持历史行为）。
    """
    art = (drama_style and DRAMA_ART_STYLE_MAP.get(drama_style)) or ""
    if not art:
        return f", {VISUAL_STYLE_VIDEO}"
    parts = [art, VIDEO_MOTION_BASE]
    if drama_style in PHOTOREAL_ART_STYLES:
        parts.append(IMPERFECTION_ANCHORS)
    parts.append("restrained ending (no explosion, no victory pose, no text overlay)")
    parts.append(VIDEO_STYLE_TAIL)
    return f", {', '.join(parts)}"


# ===========================================================================
# 各链路负面词
# ===========================================================================

#: 场景背景负面提示词（排除人物角色、特写人脸与空洞扁平构图）
SCENE_IMAGE_NEGATIVE = f"{NEGATIVE_BASE}, people, person, human figure, characters, portrait, close-up face, face, flat composition, empty boring layout"

#: 分镜图片负面提示词（排除角色不一致、多余人物与时代/现代元素穿帮）
STORYBOARD_IMAGE_NEGATIVE = f"{NEGATIVE_BASE}, inconsistent character, character mismatch, extra characters, wrong number of people, anachronism, modern objects, modern clothing"

#: 视频负面提示词（分镜视频 / 预设视频 / 手动视频共用）
VIDEO_NEGATIVE = f"{NEGATIVE_BASE}, motion blur, jittery, flickering, flicker, warping, morphing, melting, distorted face, inconsistent character, character drift, frame inconsistency, jump cuts, camera shake, static image, frozen frame"

#: 预设图片负面提示词（兼容旧引用）
PRESET_IMAGE_NEGATIVE = f"{NEGATIVE_BASE}, mutated body parts"

#: 预设视频负面提示词
PRESET_VIDEO_NEGATIVE = VIDEO_NEGATIVE


def build_character_negative_prompt(drama_style: str | None = None) -> str:
    """角色图默认负面提示词（按画风排除对立风格）。

    未指定 ``drama_style`` 或映射缺失时**回退** ``CHARACTER_IMAGE_NEGATIVE``（保持向后兼容）。
    """
    art_negative = (drama_style and _DRAMA_ART_NEGATIVE_MAP.get(drama_style)) or ""
    if not art_negative:
        return CHARACTER_IMAGE_NEGATIVE
    return (
        f"{NEGATIVE_BASE}, {art_negative}, cluttered background, busy background, "
        "multiple characters, multiple people, duplicated character, inconsistent character, "
        "cropped head, cut off face"
    )


def build_video_negative_prompt(drama_style: str | None = None) -> str:
    """视频负面词：在通用运动瑕疵基础上追加画风对立词。

    与图像链路（``build_character_negative_prompt``）同源，是防止
    「同一剧里动漫/真人/三维混用」的关键。
    """
    art_negative = (drama_style and _DRAMA_ART_NEGATIVE_MAP.get(drama_style)) or ""
    if not art_negative:
        return VIDEO_NEGATIVE
    return f"{VIDEO_NEGATIVE}, {art_negative}"


def build_scene_negative_prompt(drama_style: str | None = None) -> str:
    """场景图负面词：通用 + 画风对立词（与 ``build_scene_art_style_suffix`` 成对使用）。"""
    art_negative = (drama_style and _DRAMA_ART_NEGATIVE_MAP.get(drama_style)) or ""
    if not art_negative:
        return SCENE_IMAGE_NEGATIVE
    return f"{SCENE_IMAGE_NEGATIVE}, {art_negative}"


def build_storyboard_negative_prompt(drama_style: str | None = None) -> str:
    """分镜静帧负面词：通用 + 画风对立词（与 ``build_storyboard_art_style_suffix`` 成对使用）。"""
    art_negative = (drama_style and _DRAMA_ART_NEGATIVE_MAP.get(drama_style)) or ""
    if not art_negative:
        return STORYBOARD_IMAGE_NEGATIVE
    return f"{STORYBOARD_IMAGE_NEGATIVE}, {art_negative}"


# ===========================================================================
# 第 2 块：预设 / 角色 / 装备 / 单品 / 表情 / 物品 / 场景 prompt 构建器
# ===========================================================================

#: ``PresetVariationCardShot`` 的字段顺序 —— **顺序就是 prompt 里的词序**，不能重排。
_PRESET_SHOT_KEYS = (
    "themeFamily",
    "compositionPattern",
    "spaceType",
    "foregroundFrame",
    "mainFocalPoint",
    "thematicClue",
    "activity",
    "characterLayout",
    "lightStructure",
    "windDirection",
    "cameraPosition",
)

#: ``PresetVariationCardShot`` 的字段顺序（视频版，与图片版**不同**）
_PRESET_VIDEO_SHOT_KEYS = (
    "mainFocalPoint",
    "activity",
    "compositionPattern",
    "spaceType",
    "characterLayout",
)


def _join_parts(parts: list[str]) -> str:
    """``parts.join(', ')`` —— 与 JS 同形：缺失值占位为空串，**分隔符照留**。

    ⚠️ 别"顺手过滤空值"：JS 的 ``[undefined, 'x'].join(', ')`` 得到 ``, x``，
    过滤掉空元素会改变 prompt（词序与逗号位置都变）。
    """
    return ", ".join(parts)


def _part(value: Any) -> str:
    """取值并归一为字符串（``undefined``/``null`` 在 JS 的 join 里就是空串）。"""
    return "" if value is None else str(value)


def build_preset_image_prompt(shot: dict[str, Any]) -> str:
    """根据 Shot 配置构建预设图片生成 prompt。"""
    parts = [_part(shot.get(key)) for key in _PRESET_SHOT_KEYS]
    if js_truthy(shot.get("livingElement")):
        parts.append(_part(shot.get("livingElement")))
    parts.append(VISUAL_STYLE_MASTER)
    return _join_parts(parts)


def build_preset_video_prompt(shot: dict[str, Any], camera_move: str) -> str:
    """根据 Shot 配置 + 运镜方式构建预设视频生成 prompt。"""
    parts = [_part(shot.get(key)) for key in _PRESET_VIDEO_SHOT_KEYS]
    parts.append(_part(camera_move))
    parts.append(_part(shot.get("lightStructure")))
    parts.append(_part(shot.get("windDirection")))
    parts.append(VISUAL_STYLE_VIDEO)
    parts.append(VISUAL_STYLE_MASTER)
    return _join_parts(parts)


# ---------------------------------------------------------------------------
# 角色视觉 Prompt 构建
# ---------------------------------------------------------------------------

def _parse_string_array(raw: Any) -> list[str]:
    """JSON 数组字符串 → 非空字符串数组（损坏/非数组一律空）。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [s for s in (x.strip() for x in parsed if isinstance(x, str)) if s]


def parse_core_features(core_features: Any = None) -> list[str]:
    """解析角色核心特征标签。"""
    return _parse_string_array(core_features)


def parse_costumes(costumes: Any = None) -> list[str]:
    """解析角色多套服装变体。"""
    return _parse_string_array(costumes)


def _first(items: list[str]) -> str | None:
    """``arr[0]`` —— 空数组时 JS 得到 ``undefined``，这里得到 None。"""
    return items[0] if items else None


def format_equip(text: Any = None) -> str:
    """归一化装备/首饰文本：JSON 数组 → 逗号拼接；纯文本 → 去引号并折叠空白。"""
    if not text:
        return ""
    arr = parse_core_features(text)
    if arr:
        return ", ".join(arr)
    return re.sub(r"\s+", " ", re.sub(r"[\[\]\"']", "", str(text))).strip()


def parse_variations(variations: Any = None) -> list[dict[str, Any]]:
    """解析角色多变体立绘（``{name, imageUrl}`` 数组；只保留有字符串 name 的项）。"""
    if not variations:
        return []
    try:
        parsed = json.loads(variations)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict) and isinstance(x.get("name"), str)]


def _resolve_costume(char: dict[str, Any]) -> str | None:
    """服装取值链：``costume || clothing || costumes[0]``（JS 真值链）。"""
    return (
        (char.get("costume") if js_truthy(char.get("costume")) else None)
        or (char.get("clothing") if js_truthy(char.get("clothing")) else None)
        or _first(parse_costumes(char.get("costumes")))
    )


def build_character_visuals_clause(char: dict[str, Any]) -> str:
    """构建角色「服装/武器/首饰」正向子句。

    供**自定义 prompt** 场景强制注入：即使走了用户手写 prompt，
    智能拆分出的三个视觉字段也始终进入生图 prompt，避免信息被丢弃。
    """
    parts: list[str] = []
    costume = _resolve_costume(char)
    if costume:
        parts.append(f"wearing {costume}")
    weapons = format_equip(char.get("weapons"))
    if weapons:
        parts.append(f"armed with {weapons}")
    accessories = format_equip(char.get("accessories"))
    if accessories:
        parts.append(f"wearing accessories: {accessories}")
    return ", ".join(parts)


#: 三视图组合排版：一张横向长图内含正面/侧面/背面三个视角，同一个人物三次出现
THREE_VIEW_COMBINED_LAYOUT = "full body three-view turnaround, one single wide horizontal image with the same character shown three times side by side: front view on the left, side view in the middle, back view on the right, identical outfit and appearance across all three views, consistent character, head to toe"

#: 三视图画布尺寸：超宽横向比例（≈2.29:1），保证三个全身视角横向排开不挤压
THREE_VIEW_SIZE = "2048x896"


def build_three_view_negative(drama_style: str | None = None) -> str:
    """三视图负面词：在画风负面基础上排除分屏拼接/文字/边框/多余人。"""
    base = build_character_negative_prompt(drama_style)
    return (
        f"{base}, split image, split panel, grid layout, frame border, panel divider, "
        "side-by-side separate images, multiple different characters, different people, clone, "
        "extra person, text, watermark, logo, label"
    )


def build_character_image_prompt(char: dict[str, Any]) -> str:
    """构建角色图片生成 prompt（核心特征 + 外貌 + 视觉子句 + 性格氛围 + 画风收口）。"""
    parts: list[str] = [_part(char.get("name"))]
    core = parse_core_features(char.get("coreFeatures"))
    if core:
        parts.append(f"core features: {', '.join(core)}")
    if js_truthy(char.get("appearance")):
        parts.append(_part(char.get("appearance")))
    visuals = build_character_visuals_clause(char)
    if visuals:
        parts.append(visuals)
    # ⚠️ `description !== appearance` 的去重：两者相同时不重复灌入
    if js_truthy(char.get("description")) and char.get("description") != char.get("appearance"):
        parts.append(_part(char.get("description")))
    if js_truthy(char.get("personality")):
        parts.append(f"{char['personality']} expression and mannerisms")
    return f"{', '.join(parts)}{build_character_art_style_suffix(char.get('dramaStyle'))}"


def build_character_appearance_text(char: dict[str, Any]) -> str:
    """构建角色外观描述文本（用于分镜/视频 prompt 中引用），**不含风格标签**。

    ⚠️ 与 ``build_character_visuals_clause`` 有三处细节差异，别合并：
    ① 核心特征**不带** ``core features:`` 前缀；
    ② 首饰用 ``accessories:`` 而非 ``wearing accessories:``；
    ③ 最后用 ``': '`` 而非 ``', '`` 拼接。
    """
    parts: list[str] = [_part(char.get("name"))]
    core = parse_core_features(char.get("coreFeatures"))
    if core:
        parts.append(", ".join(core))
    if js_truthy(char.get("appearance")):
        parts.append(_part(char.get("appearance")))
    costume = _resolve_costume(char)
    if costume:
        parts.append(f"wearing {costume}")
    weapons = format_equip(char.get("weapons"))
    if weapons:
        parts.append(f"armed with {weapons}")
    accessories = format_equip(char.get("accessories"))
    if accessories:
        parts.append(f"accessories: {accessories}")
    if js_truthy(char.get("description")) and char.get("description") != char.get("appearance"):
        parts.append(_part(char.get("description")))
    return ": ".join(parts)


def build_equip_image_prompt(equip_type: str, char: dict[str, Any]) -> str:
    """构建角色「装备/服饰特写图」prompt（服装/武器/首饰独立**三视图设定图**）。

    设计原则：三类对象是「不同的东西」，提示词必须聚焦各自对象本体 ——
    服装看剪裁/面料/配色（无人物）、武器看造型/金属质感/雕刻（无手持）、首饰看宝石/工艺（无佩戴）。
    **不再把整段角色外貌、核心特征、角色名灌入**（那是角色立绘的职责），
    否则服装/武器/首饰提示词会与立绘雷同、彼此雷同。
    """
    parts: list[str] = []
    if equip_type == "clothing":
        value = format_equip(char.get("clothing"))
        parts.append(f"detailed costume three-view design of {value}" if value else "detailed costume three-view design")
        parts.append("one wide horizontal image with the same garment shown three times side by side: front view, back view, side view")
        parts.append("ghost mannequin, silhouette, fabric texture and color scheme, no person wearing it, no body")
    elif equip_type == "weapon":
        value = format_equip(char.get("weapons"))
        parts.append(f"detailed weapon three-view concept art of {value}" if value else "detailed weapon three-view concept art")
        parts.append("one wide horizontal image with the same weapon shown three times side by side: front view, side view, top view")
        parts.append("metal texture, engravings and material details, isolated on clean background, no hand, no character holding it")
    else:
        value = format_equip(char.get("accessories"))
        parts.append(f"detailed jewelry three-view design of {value}" if value else "detailed jewelry three-view design")
        parts.append("one wide horizontal image with the same piece of jewelry shown three times side by side: front view, side view, top view")
        parts.append("gemstone and metalwork details, isolated on clean background, no person wearing it")
    parts.append("object focused concept sheet, clean studio lighting, consistent item across all three views, no split panel borders, no grid frames, no text, no labels, no watermark")
    # 画风收口：装备图使用「无人物」画风尾，绝不引入 character design / skin 等人物语义词
    return f"{', '.join(parts)}{build_equip_art_style_suffix(char.get('dramaStyle'))}"


def build_equip_negative(drama_style: str | None = None) -> str:
    """装备三视图负面词：排除人物/手持/不同物体/边框文字。

    ⚠️ **不排除** ``side-by-side`` / 三视图布局 —— 否则模型会把三视角画成单个物体。
    """
    base = build_character_negative_prompt(drama_style)
    return (
        f"{base}, person, human figure, character, hands, body parts, multiple different items, "
        "different objects, duplicate inconsistent object, frame border, grid lines, panel divider, "
        "section divider, text, watermark, logo, label, extra floating parts, cropped object, "
        "cut off object"
    )


# ---------------------------------------------------------------------------
# 单件高清道具图（区别于三视图设定图：单件纯物品图，可直接进道具库/分镜）
# ---------------------------------------------------------------------------

#: 单件道具图尺寸：方形高清，便于道具库卡片与分镜引用
ITEM_IMAGE_SIZE = "1024x1024"


def build_item_image_prompt(equip_type: str, char: dict[str, Any]) -> str:
    """构建单件高清道具图 prompt（单品居中展示、纯物品、无人物）。

    与 ``build_equip_image_prompt``（三视角并排设定图）的区别就在此处：
    单品图用于**入库**与分镜画面中的物品质感呈现。
    """
    parts: list[str] = []
    if equip_type == "clothing":
        value = format_equip(char.get("clothing"))
        parts.append(f"hero product shot of {value}" if value else "hero product shot of a single costume")
        parts.append("one garment displayed alone and centered, full view, fabric texture, color and tailoring details, studio product lighting, clean soft gradient background, no person, no mannequin, no body")
    elif equip_type == "weapon":
        value = format_equip(char.get("weapons"))
        parts.append(f"hero product shot of {value}" if value else "hero product shot of a single weapon")
        parts.append("one weapon displayed alone and centered, full view, metal texture, engravings and material details, studio product lighting, clean soft gradient background, no hand, no character holding it")
    else:
        value = format_equip(char.get("accessories"))
        parts.append(f"hero product shot of {value}" if value else "hero product shot of a single piece of jewelry")
        parts.append("one piece of jewelry displayed alone and centered, full view, gemstone and metalwork details, studio product lighting, clean soft gradient background, no person wearing it")
    parts.append("single object only, high-detail prop asset, isolated, no text, no labels, no watermark, no logo, not a three-view, not side by side")
    return f"{', '.join(parts)}{build_equip_art_style_suffix(char.get('dramaStyle'))}"


def build_item_negative(drama_style: str | None = None) -> str:
    """单件道具图负面词：排除人物/手持/三视角拼接/多物品/文字（单品入库基准）。"""
    base = build_character_negative_prompt(drama_style)
    return (
        f"{base}, person, human figure, character, hands, body parts, mannequin, ghost mannequin, "
        "side by side, three views, multi view, turnaround, multiple items, different objects, "
        "duplicate, frame border, grid lines, panel divider, section divider, text, watermark, "
        "logo, label, cropped object, cut off object"
    )


# ---------------------------------------------------------------------------
# 角色表情头像特写组
# ---------------------------------------------------------------------------

#: 角色表情预设（9 个）：表情演出用头像特写组。
#: 后端在生成/回填/校验时与前端共用同一份常量，**key 必须对齐**
#: （漂移守卫会比对 key 集合，见 ``tests/route_parity_test.py``）。
EXPRESSION_PRESETS: list[dict[str, str]] = [
    {"key": "smile", "label": "微笑", "en": "gentle warm smile, softly curved lips, kind happy eyes"},
    {"key": "laugh", "label": "大笑", "en": "hearty open laugh, bright joyful eyes, wide laughing mouth"},
    {"key": "mischievous", "label": "俏皮", "en": "playful mischievous grin, slightly raised eyebrow, cheerful teasing look"},
    {"key": "angry", "label": "愤怒", "en": "fierce angry glare, furrowed brows, tight frowning mouth"},
    {"key": "sad", "label": "悲伤", "en": "sad melancholy face, downcast eyes, drooping mouth corners"},
    {"key": "surprised", "label": "惊讶", "en": "surprised wide eyes, raised eyebrows, slightly open mouth"},
    {"key": "tearful", "label": "泪目", "en": "teary glistening eyes about to cry, reddened rims, sorrowful trembling lips"},
    {"key": "serious", "label": "严肃", "en": "serious stern neutral face, calm unreadable gaze, pressed lips"},
    {"key": "sobbing", "label": "大哭", "en": "crying hard, eyes squeezed shut, tears streaming down cheeks, open wailing mouth"},
]


def find_expression_preset(key: str) -> dict[str, str] | None:
    """按 key 查找表情预设；未知 key 返回 None（调用方需自行兜底）。"""
    for preset in EXPRESSION_PRESETS:
        if preset["key"] == key:
            return preset
    return None


def build_expression_image_prompt(
    char: dict[str, Any], expression_key: str, drama_style: str | None = None
) -> str:
    """构建单个表情头像特写 prompt（头肩特写 + 表情 + 服装一致性）。

    构图必须单一表情、单一人物、头肩取景 —— 避免生成整身/多人/表情拼接；
    同时保留角色核心五官与服装，保证与立绘/三视图是**同一个人**。

    ⚠️ 服装链这里只有 ``clothing || costumes[0]``（**没有** ``costume``），
    与 ``build_character_visuals_clause`` 不同 —— 原 TS 如此。
    """
    preset = find_expression_preset(expression_key)
    parts: list[str] = [f"close-up portrait headshot of {_part(char.get('name'))}"]
    core = parse_core_features(char.get("coreFeatures"))
    if core:
        parts.append(f"core features: {', '.join(core)}")
    if js_truthy(char.get("appearance")):
        parts.append(_part(char.get("appearance")))
    costume = (
        (char.get("clothing") if js_truthy(char.get("clothing")) else None)
        or _first(parse_costumes(char.get("costumes")))
    )
    if costume:
        parts.append(f"wearing {costume}")
    parts.append(f"facial expression: {(preset['en'] if preset else None) or 'neutral calm expression'}")
    parts.append("looking directly at viewer, head and shoulders framing, single face only")
    return f"{', '.join(parts)}{build_character_art_style_suffix(drama_style)}"


def build_expression_negative(drama_style: str | None = None) -> str:
    """表情头像特写负面词：排除多人/多表情/拼接/整身（一张图只有单个头肩、单一表情）。"""
    base = build_character_negative_prompt(drama_style)
    return (
        f"{base}, multiple faces, multiple expressions, two or more different expressions, "
        "extra person, duplicate character, group shot, collage, split image, grid layout, "
        "side-by-side panels, body below shoulders, full body shot, whole figure, text, "
        "watermark, logo, label"
    )


# ---------------------------------------------------------------------------
# 物品视觉 Prompt 构建（连续性状态机 v3 物品库）
# ---------------------------------------------------------------------------

def build_prop_image_prompt(prop: dict[str, Any]) -> str:
    """构建物品（道具/信物/线索/法器）设定图 prompt。"""
    parts: list[str] = []
    # ⚠️ 分类用的是**全角括号** ``（道具）``，别改成半角
    parts.append(f"detailed prop design sheet of {_part(prop.get('name'))}（{prop.get('category') or '道具'}）")
    if js_truthy(prop.get("appearance")):
        parts.append(f"appearance: {prop['appearance']}")
    if js_truthy(prop.get("color")):
        parts.append(f"color: {prop['color']}")
    if js_truthy(prop.get("sizeHint")):
        parts.append(f"size: {prop['sizeHint']}")
    if js_truthy(prop.get("description")):
        parts.append(f"description: {prop['description']}")
    if js_truthy(prop.get("holder")):
        parts.append(f"used/hold by: {prop['holder']}")
    parts.append("single object centered, clean background, faithful to the drama art style")
    return f"{', '.join(parts)}, {VISUAL_STYLE_SCENE}, {VISUAL_STYLE_MASTER}"


# ---------------------------------------------------------------------------
# 场景视觉 Prompt 构建
# ---------------------------------------------------------------------------

def build_scene_image_prompt(options: dict[str, Any]) -> str:
    """构建场景图片生成 prompt（地点 + 时间氛围 **或** 直接给定的描述）。

    画风收口：统一走 ``DRAMA_ART_STYLE_MAP`` 四段词表
    —— 此前把落库 key 直接拼成 ``<key> visual style``，等于画风体系在场景图链路**完全没生效**。
    """
    parts: list[str] = []
    if js_truthy(options.get("prompt")):
        parts.append(_part(options.get("prompt")))
    else:
        parts.append(_part(options.get("location")))
        if js_truthy(options.get("time")):
            parts.append(f"{options['time']} lighting and atmosphere")
    return f"{', '.join(parts)}{build_scene_art_style_suffix(options.get('dramaStyle'))}"


# ===========================================================================
# 第 5 块：台词角色一致性校验 + 视频 prompt 标签剥离
# ===========================================================================

#: 对话抽取：`<说话人 1-10 字>:<台词 ≥8 字>`。**惰性量词**很重要 ——
#: 贪婪版会把前面整段当成说话人。
_DIALOGUE_SPEAKER_RE = re.compile(r"([^\n:：]{1,10}?)[:：]([^:：\n]{8,})")

#: 旁白/画外音不参与一致性校验
_NARRATOR_RE = re.compile(r"(旁白|画外音|narrator)", re.IGNORECASE)


def validate_dialogue_character_consistency(
    conn: Connection, dialogue: str | None, storyboard_id: int
) -> dict[str, Any]:
    """校验台词里的说话人是否都是**该分镜关联的角色**。

    返回 ``{mismatches, matchCount, allMatch}``（**camelCase** —— 它会直接进 API 响应）。

    两步取说话人：先用正则抽 ``名字：台词``；**正则没抽到**时才回退
    :func:`parse_dialogue_for_tts`（等价于 TS 里那份「避免循环依赖」的本地副本）。
    旁白 / 画外音 / narrator 一律跳过，不计入 matchCount 也不计入 mismatches。
    """
    if not dialogue:
        return {"mismatches": [], "matchCount": 0, "allMatch": True}

    # `new Set` 的语义：保序去重
    speaker_names: list[str] = []
    for match in _DIALOGUE_SPEAKER_RE.finditer(dialogue):
        name = match.group(1).strip()
        if name not in speaker_names:
            speaker_names.append(name)

    if not speaker_names:
        parsed = parse_dialogue_for_tts(dialogue)
        if parsed.get("speaker") and not parsed.get("ignorable"):
            speaker_names.append(parsed["speaker"])

    if not speaker_names:
        return {"mismatches": [], "matchCount": 0, "allMatch": True}

    # 关联角色（用 WHERE 查询替代全表扫描；只认未软删的角色）
    links = conn.execute(
        select(storyboard_characters.c.character_id).where(
            storyboard_characters.c.storyboard_id == storyboard_id
        )
    ).all()
    character_ids = [row[0] for row in links]
    char_names: set[str] = set()
    if character_ids:
        rows = conn.execute(
            select(characters.c.name).where(
                and_(
                    characters.c.id.in_(character_ids),
                    characters.c.deleted_at.is_(None),
                )
            )
        ).all()
        char_names = {row[0] for row in rows}

    mismatches: list[str] = []
    match_count = 0
    for name in speaker_names:
        if _NARRATOR_RE.fullmatch(name):
            continue
        if name in char_names:
            match_count += 1
        else:
            mismatches.append(name)
    return {"mismatches": mismatches, "matchCount": match_count, "allMatch": len(mismatches) == 0}


#: 视频提示词里的结构化标签 ``<location>`` / ``<role>`` / ``<voice>``
_VIDEO_TAG_RE = re.compile(r"</?(?:location|role|voice)>", re.IGNORECASE)
#: 时间段分隔符 ``<n>`` / ``<n/>`` / ``<n />``
_VIDEO_BREAK_RE = re.compile(r"<n\s*/?>", re.IGNORECASE)


def strip_video_prompt_tags(prompt: str) -> str:
    """剥离视频提示词中的结构化标记标签，只保留标签内的自然语言内容。

    背景：分镜 agent 生成的 ``video_prompt`` 用 ``<location>``/``<role>``/``<voice>``/``<n>``
    作为结构化 DSL，供程序解析（分段、角色绑定、场景提取）使用。但**视频扩散模型不识别
    这些 XML 标签**，原样保留只会成为文本噪声，占用注意力并可能干扰语义理解。

    规则（**顺序敏感**）：

    1. ``<location>…</location>`` / ``<role>`` / ``<voice>`` → 去掉开闭标签，**保留中间内容**；
    2. ``<n>``（时间段分隔符）→ 换成**换行**，让每个时间段独立成段；
    3. 压缩行内空白 → 去掉换行两侧空白 → 首尾 strip。
    """
    if not prompt:
        return prompt
    text = _VIDEO_TAG_RE.sub("", prompt)
    text = _VIDEO_BREAK_RE.sub("\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


# ===========================================================================
# 第 3 块：分镜视觉 prompt 构建
# ===========================================================================

def has_art_style_key(drama_style: str | None = None) -> bool:
    """``!!(dramaStyle && DRAMA_ART_STYLE_MAP[dramaStyle])`` —— **命中四段词表**才算有画风。

    与 `build_video_art_style_suffix` 里的判断同源：它决定视频 prompt 末尾挂
    「画风层」还是退回历史默认视觉层。
    """
    return bool(drama_style and DRAMA_ART_STYLE_MAP.get(drama_style))


def build_storyboard_image_prompt(options: dict[str, Any]) -> str:
    """构建分镜**叙事图片** prompt（融合角色 + 场景 + 镜头描述）。

    核心修复（原注释）：之前分镜图片生成只传 ``description``，没有角色外观和场景信息，
    导致角色在不同分镜中视觉不一致。

    分句规则：各段用 ``'. '`` 连接，**末尾统一挂画风后缀**；一段都没有时退化成
    ``cinematic shot`` + 画风后缀。
    """
    parts: list[str] = []

    # 1. 场景环境（`sceneDescription` 优先于 `location`）
    if js_truthy(options.get("sceneDescription")):
        parts.append(f"Scene: {options['sceneDescription']}")
    elif js_truthy(options.get("location")):
        parts.append(f"Location: {options['location']}")

    # 2. 角色外观
    if js_truthy(options.get("characterDescription")):
        parts.append(f"Characters: {options['characterDescription']}")

    # 3. 镜头描述（景别/机位经**视觉图谱**翻译为英文电影术语，避免中文混入英文 prompt）
    cinematic_hints: list[str] = []
    shot_en = resolve_visual_term(options.get("shotType"))
    if js_truthy(options.get("shotType")):
        cinematic_hints.append(shot_en or options["shotType"])
    angle_en = resolve_visual_term(options.get("cameraAngle"))
    if js_truthy(options.get("cameraAngle")):
        cinematic_hints.append(angle_en or options["cameraAngle"])
    if cinematic_hints:
        parts.append(f"Camera: {', '.join(cinematic_hints)}")

    # 4. 叙事内容（`storyboardDescription` 优先于 `description`）
    if js_truthy(options.get("storyboardDescription")):
        parts.append(str(options["storyboardDescription"]))
    elif js_truthy(options.get("description")):
        parts.append(str(options["description"]))

    # 底线
    if len(parts) == 0:
        return f"cinematic shot{build_storyboard_art_style_suffix(options.get('dramaStyle'))}"

    # 5. 画风收口：统一走 DRAMA_ART_STYLE_MAP 四段词表（与角色/场景链路同源）
    return f"{'. '.join(parts)}{build_storyboard_art_style_suffix(options.get('dramaStyle'))}"


#: 声音策略（对齐 Mx-Shell sound policy：画面内不混音，音乐后期叠加）
_SOUND_POLICY = (
    "Sound: production audio only, no background music in frame "
    "(music is mixed in post-production)"
)


def build_storyboard_video_prompt(options: dict[str, Any]) -> str:
    """构建分镜**视频**生成 prompt（融合角色 + 场景 + 叙事 + 动作）。

    核心修复（原注释）：之前视频生成只用 ``storyboardDescription``，完全没有角色外观上下文，
    导致视频中角色形象突变。

    ⚠️ 末尾拼装顺序**很讲究**，一眼看不出但会直接改变 prompt：

    ``<各段以 '. ' 连接>, <声音策略>.<UI 留白规则><环境音标记> <画风层>``

    * UI 留白规则自带**前置逗号**（命中才拼）；
    * 环境音标记自带**前置空格**；
    * 画风层：命中画风时用画风后缀（**去掉自带的逗号前缀**，改用空格衔接，保持历史分句形态）；
      未命中时**保持历史默认视觉层** ``VISUAL_STYLE_VIDEO + VISUAL_STYLE_MASTER``。
    """
    parts: list[str] = []

    # 1. 角色一致性 —— 最关键的部分
    character_appearances = options.get("characterAppearances")
    if character_appearances and len(character_appearances) > 0:
        parts.append(
            "Characters (maintain strict visual consistency): "
            + "; ".join(character_appearances)
        )

    # 2. 场景
    if js_truthy(options.get("scenePrompt")):
        parts.append(f"Setting: {options['scenePrompt']}")

    # 3. 叙事
    if js_truthy(options.get("storyboardDescription")):
        parts.append(str(options["storyboardDescription"]))
    elif js_truthy(options.get("description")):
        parts.append(str(options["description"]))

    # 4. 动作与运镜（运镜经视觉图谱翻译为英文电影术语）
    if js_truthy(options.get("action")):
        parts.append(f"Action: {options['action']}")
    if js_truthy(options.get("movement")):
        move_en = resolve_visual_term(options.get("movement"))
        parts.append(f"Camera movement: {move_en or options['movement']}")

    # 5. UI 屏幕留白规则（ui_plate 约束）：分镜含屏幕/UI 元素时强制只留白、文字后期叠加
    has_screen = any(
        contains_screen_element(options.get(key))
        for key in ("description", "storyboardDescription", "action")
    )
    ui_rule = f", {UI_OVERLAY_RULE}" if has_screen else ""

    # 6. H3 原生场景声标记 [background_audio]：H3 会在生成视频时同步合成环境音
    background_audio = (options.get("backgroundAudio") or "").strip()
    bg_audio = f" [background_audio] {background_audio}" if background_audio else ""

    # 画风层与上一句之间用空格衔接（去掉后缀自带的逗号前缀）
    if has_art_style_key(options.get("dramaStyle")):
        style_layer = re.sub(r"^,\s*", "", build_video_art_style_suffix(options.get("dramaStyle")))
    else:
        style_layer = f"{VISUAL_STYLE_VIDEO}, {VISUAL_STYLE_MASTER}"

    return f"{'. '.join(parts)}, {_SOUND_POLICY}.{ui_rule}{bg_audio} {style_layer}"


# ===========================================================================
# 第 4 块：DB 读取辅助（分镜上下文 / 参考图 / 参考音频）
# ===========================================================================

#: 参考音频 URL 的默认基址。
#: ⚠️ 原 TS 硬编码 Node 的端口 5789；本进程默认跑在 5790 ⇒ **单独跑 Python 后端时
#: 必须设 ``PUBLIC_BASE_URL``**，否则本地 H3 服务按 5789 拉参考音频会 404。
_DEFAULT_PUBLIC_BASE_URL = "http://localhost:5789"
_ABSOLUTE_URL_RE = re.compile(r"^(?:https?:|data:)")
_LEADING_SLASHES_RE = re.compile(r"^/+")


def to_public_media_url(path: str) -> str:
    """把媒体相对路径转成**可被本地/远程服务拉取**的绝对 URL（本地 H3 需能拉参考音频）。"""
    if _ABSOLUTE_URL_RE.match(path or ""):
        return path
    base = re.sub(r"/+$", "", os.environ.get("PUBLIC_BASE_URL") or _DEFAULT_PUBLIC_BASE_URL)
    return f"{base}/{_LEADING_SLASHES_RE.sub('', path)}"


def _costume_by_character(links: list[Any]) -> dict[Any, str]:
    """本镜头的「角色 → 服装变体」映射（只收有 costume 的关联）。"""
    mapping: dict[Any, str] = {}
    for link in links:
        if _col(link, "costume"):
            mapping[_col(link, "character_id")] = _col(link, "costume")
    return mapping


def _live_characters(conn: Connection, character_ids: list[Any]) -> list[Any]:
    """取未软删的角色行（**用 WHERE 查询**而不是内存过滤）。"""
    if not character_ids:
        return []
    return conn.execute(
        select(characters).where(
            and_(
                characters.c.id.in_(character_ids),
                characters.c.deleted_at.is_(None),
            )
        )
    ).all()


def get_storyboard_character_appearances(conn: Connection, storyboard_id: int) -> list[str]:
    """根据分镜 ID 获取关联角色的**外表描述**列表（供视频/图片 prompt 注入角色一致性）。"""
    links = conn.execute(
        select(storyboard_characters).where(
            storyboard_characters.c.storyboard_id == storyboard_id
        )
    ).all()
    if not links:
        return []

    costume_by_char_id = _costume_by_character(links)
    rows = _live_characters(conn, [_col(link, "character_id") for link in links])

    return [
        build_character_appearance_text({
            "name": _col(row, "name"),
            "appearance": _col(row, "appearance"),
            "description": _col(row, "description"),
            "coreFeatures": _col(row, "core_features"),
            "clothing": _col(row, "clothing"),
            "costume": costume_by_char_id.get(_col(row, "id")),
            "costumes": _col(row, "costumes"),
        })
        for row in rows
    ]


def get_storyboard_scene_description(conn: Connection, storyboard_id: int) -> str | None:
    """根据分镜 ID 获取关联场景的**视觉描述**（走 ``build_scene_image_prompt``）。"""
    sb_row = conn.execute(
        select(storyboards.c.scene_id).where(storyboards.c.id == storyboard_id)
    ).first()
    if sb_row is None or not sb_row[0]:
        return None

    scene = conn.execute(
        select(scenes.c.location, scenes.c.time, scenes.c.prompt).where(scenes.c.id == sb_row[0])
    ).first()
    if scene is None:
        return None

    return build_scene_image_prompt({
        "location": scene[0],
        "time": scene[1],
        "prompt": scene[2],
    })


def get_character_image_urls(conn: Connection, character_ids: list[int]) -> list[str]:
    """根据角色 ID 列表取**有立绘**的图片 URL（用于视频/图片 reference_images）。"""
    if not character_ids:
        return []
    rows = _live_characters(conn, character_ids)
    return [_col(row, "image_url") for row in rows if _col(row, "image_url")]


def get_storyboard_character_image_urls(conn: Connection, storyboard_id: int) -> list[str]:
    """根据分镜 ID 取关联角色的图片 URL。

    ⚠️ 镜头级服装变体优先：角色指定了变体**且变体有立绘**时用变体图，否则回退主图。
    """
    links = conn.execute(
        select(storyboard_characters).where(
            storyboard_characters.c.storyboard_id == storyboard_id
        )
    ).all()
    if not links:
        return []

    costume_by_char_id = _costume_by_character(links)
    rows = _live_characters(conn, [_col(link, "character_id") for link in links])

    urls: list[str] = []
    for row in rows:
        costume = costume_by_char_id.get(_col(row, "id"))
        if costume:
            variation = next(
                (
                    item
                    for item in parse_variations(_col(row, "variations"))
                    if item.get("name") == costume
                ),
                None,
            )
            if variation is not None and variation.get("imageUrl"):
                urls.append(variation["imageUrl"])
                continue
        if _col(row, "image_url"):
            urls.append(_col(row, "image_url"))
    return urls


def get_storyboard_reference_images(conn: Connection, storyboard_id: int) -> list[str]:
    """分镜参考图 = **角色立绘 + 场景图 + 本镜物品设定图**（跨集一致性用）。"""
    urls = get_storyboard_character_image_urls(conn, storyboard_id)

    sb_row = conn.execute(
        select(storyboards.c.scene_id).where(storyboards.c.id == storyboard_id)
    ).first()
    if sb_row is not None and sb_row[0]:
        scene = conn.execute(
            select(scenes.c.image_url).where(scenes.c.id == sb_row[0])
        ).first()
        if scene is not None and scene[0]:
            urls.append(scene[0])

    # 物品参考图（连续性状态机 v3 物品库）：本镜关联物品的设定图
    prop_links = conn.execute(
        select(storyboard_props.c.prop_id).where(
            storyboard_props.c.storyboard_id == storyboard_id
        )
    ).all()
    if prop_links:
        rows = conn.execute(
            select(prop_templates.c.image_url).where(
                and_(
                    prop_templates.c.id.in_([row[0] for row in prop_links]),
                    prop_templates.c.deleted_at.is_(None),
                )
            )
        ).all()
        for row in rows:
            if row[0]:
                urls.append(row[0])

    return urls


def get_storyboard_reference_audio_urls(conn: Connection, storyboard_id: int) -> list[str]:
    """收集分镜出场角色的**声线样本**音频 URL（H3 Ref2VA 参考音频，**最多 3 条**）。"""
    links = conn.execute(
        select(storyboard_characters.c.character_id).where(
            storyboard_characters.c.storyboard_id == storyboard_id
        )
    ).all()
    if not links:
        return []

    rows = _live_characters(conn, [row[0] for row in links])
    urls = [
        str(_col(row, "voice_sample_url")).strip()
        for row in rows
        if _col(row, "voice_sample_url") and str(_col(row, "voice_sample_url")).strip()
    ]
    return [to_public_media_url(url) for url in urls][:3]


# ===========================================================================
# 第 6 块：宫格图 prompt（参考资产收集 / 角度表 / buildGridPrompt / buildGridCellPrompts）
# ===========================================================================

#: 宫格「同场景多角度」用的角度表（25 条；`i % len` 取用，**顺序即画面节奏**）
_GRID_ANGLES = (
    "wide establishing shot", "medium shot character focus",
    "close-up detail", "dramatic low angle", "over-the-shoulder view",
    "bird eye view", "side profile", "atmospheric detail",
    "extreme close-up", "dutch angle", "silhouette shot",
    "depth of field focus", "symmetrical composition", "leading lines",
    "negative space", "high angle looking down", "ground level",
    "panoramic wide", "intimate two-shot", "reflection shot",
    "shadow play", "backlit silhouette", "macro detail",
    "split lighting", "rim light portrait",
)

#: 宫格链路只关心这几个分镜字段：(**蛇形列名**, camelCase 字段名)。
#: ``_sb_view`` 同时兼容 Row（属性）与 dict（两种命名），路由层传哪种都行。
_SB_KEY_MAP = (
    ("id", "id"),
    ("storyboard_number", "storyboardNumber"),
    ("first_frame_image", "firstFrameImage"),
    ("last_frame_image", "lastFrameImage"),
    ("composed_image", "composedImage"),
    ("reference_images", "referenceImages"),
    ("scene_id", "sceneId"),
    ("image_prompt", "imagePrompt"),
    ("description", "description"),
    ("title", "title"),
    ("action", "action"),
    ("movement", "movement"),
    ("location", "location"),
    ("shot_type", "shotType"),
)


def _sb_view(sb: Any) -> dict[str, Any]:
    """把分镜（Row / dict）归一成蛇形 key 字典（缺字段为 None）。"""
    out: dict[str, Any] = {}
    for snake, camel in _SB_KEY_MAP:
        if isinstance(sb, dict):
            out[snake] = sb[snake] if snake in sb else sb.get(camel)
        else:
            out[snake] = getattr(sb, snake, None)
    return out


def _js_tpl(value: Any) -> str:
    """JS 模板字面量里的取值：``null`` → ``"null"``、``undefined`` → ``"undefined"``。

    ⚠️ DB 行取到的空值是 SQL ``NULL``，在 JS 侧是 ``null`` ⇒ 这里映射成 ``"null"``
    （而不是 Python 的 ``"None"``）。宫格标签里会用到 ``镜头${sb.storyboardNumber}首帧``。
    """
    return "null" if value is None else str(value)


def _safe_parse_json_array(value: Any) -> list[Any]:
    """``JSON.parse`` 后取真值项（损坏 / 非数组一律空）。"""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if js_truthy(item)]


def _pos_label(i: int, rows: int, cols: int) -> str:
    """``row {r+1} col {c+1}``（⚠️ ``rows`` 参数**未使用**，原 TS 如此，别删参数）。"""
    return f"row {i // cols + 1} col {i % cols + 1}"


def _cell_label(i: int, rows: int, cols: int) -> str:
    """``格{i+1}（row R col C）``（**全角括号**）。"""
    return f"格{i + 1}（{_pos_label(i, rows, cols)}）"


def _get_storyboard_character_ids(conn: Connection, storyboard_ids: list[Any]) -> dict[int, list[int]]:
    """分镜 ID → 关联角色 ID 列表（顺序 = 关联表插入顺序）。"""
    if not storyboard_ids:
        return {}
    # ⚠️ ``storyboard_characters`` 是**复合主键**（无 ``id`` 列），按复合键排序保证确定性；
    # 顺序会影响角色外观在 ``Characters: a; b`` 里的先后（即 prompt 文本）
    rows = conn.execute(
        select(storyboard_characters.c.storyboard_id, storyboard_characters.c.character_id)
        .where(storyboard_characters.c.storyboard_id.in_(storyboard_ids))
        .order_by(storyboard_characters.c.storyboard_id, storyboard_characters.c.character_id)
    ).all()
    mapping: dict[int, list[int]] = {}
    for storyboard_id, character_id in rows:
        mapping.setdefault(storyboard_id, []).append(character_id)
    return mapping


def _build_character_appearance_map(
    conn: Connection, storyboard_character_ids: dict[int, list[int]]
) -> dict[int, str]:
    """角色 ID → 外观描述文本（**只取未软删**的角色；顺序 = 主键序）。"""
    all_ids = {cid for ids in storyboard_character_ids.values() for cid in ids}
    if not all_ids:
        return {}
    rows = _live_characters(conn, list(all_ids))
    mapping: dict[int, str] = {}
    for row in rows:
        char = dict(zip(row._mapping.keys(), row._mapping.values())) if hasattr(row, "_mapping") else row
        mapping[int(char["id"])] = build_character_appearance_text(char)
    return mapping


def _build_storyboard_reference_hints(
    sb: dict[str, Any],
    reference_assets: list[dict[str, Any]],
    storyboard_character_ids: dict[int, list[int]],
) -> list[str]:
    """分镜关联的参考图提示（角色立绘 + 场景图 + 已有分镜图），去重后**最多 4 条**。"""
    hints: list[str] = []
    char_ids = storyboard_character_ids.get(sb["id"]) or []

    for asset in reference_assets:
        kind = asset.get("kind")
        if kind == "scene" and sb["scene_id"] and asset.get("sceneId") == sb["scene_id"]:
            hints.append(f"{asset['imageLabel']}（{asset['label']}）")
        if kind == "character":
            if asset.get("characterId") and asset["characterId"] in char_ids:
                hints.append(f"{asset['imageLabel']}（{asset['label']}）")
        if kind == "storyboard" and asset.get("storyboardId") == sb["id"]:
            hints.append(f"{asset['imageLabel']}（{asset['label']}）")

    # 去重保序（JS 的 `[...new Set(x)]`）后截断
    seen: list[str] = []
    for hint in hints:
        if hint not in seen:
            seen.append(hint)
    return seen[:4]


def _build_enriched_cell_description(
    sb: dict[str, Any],
    index: int,
    storyboard_character_ids: dict[int, list[int]],
    char_appearance_map: dict[int, str],
    reference_assets: list[dict[str, Any]],
) -> str:
    """增强的 cell 描述：``Characters: …`` + ``参考…`` + 分镜描述（``'. '`` 连接）。"""
    parts: list[str] = []

    appearances = [
        char_appearance_map[cid]
        for cid in (storyboard_character_ids.get(sb["id"]) or [])
        if char_appearance_map.get(cid)
    ]
    if appearances:
        parts.append(f"Characters: {'; '.join(appearances)}")

    refs = _build_storyboard_reference_hints(sb, reference_assets, storyboard_character_ids)
    if refs:
        parts.append(f"参考{'、'.join(refs)}")

    desc = sb["image_prompt"] or sb["description"] or sb["title"] or f"shot {index + 1}"
    parts.append(desc)

    return ". ".join(parts)


def collect_grid_reference_assets(conn: Connection, storyboards: list[Any]) -> list[dict[str, Any]]:
    """收集宫格图涉及的参考图（首尾帧/镜头图/参考图 + 场景图 + 角色立绘，**上限 6 张**）。

    ⚠️ 顺序即「图片 N」编号，**且上限是 6** ⇒ 收集顺序会影响哪些图能进编号。
    顺序：逐个分镜（首帧 → 尾帧 → 镜头图 → 参考图）→ 场景图 → 角色立绘。
    """
    views = [_sb_view(sb) for sb in storyboards]
    storyboard_character_ids = _get_storyboard_character_ids(conn, [view["id"] for view in views])

    # `[...new Set(...filter(Boolean))]`：去重保序 + 去掉空值
    scene_ids: list[Any] = []
    for view in views:
        if view["scene_id"] and view["scene_id"] not in scene_ids:
            scene_ids.append(view["scene_id"])
    character_ids: list[Any] = []
    for ids in storyboard_character_ids.values():
        for cid in ids:
            if cid and cid not in character_ids:
                character_ids.append(cid)

    scene_rows = (
        conn.execute(
            select(scenes).where(scenes.c.id.in_(scene_ids)).order_by(scenes.c.id)
        ).all()
        if scene_ids
        else []
    )
    chars = _live_characters(conn, character_ids) if character_ids else []

    assets: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push_asset(path: Any, label: str, kind: str, **extra: Any) -> None:
        if not path or path in seen or len(assets) >= 6:
            return
        seen.add(path)
        assets.append({"path": path, "label": label, "kind": kind, **extra})

    for view in views:
        number = _js_tpl(view["storyboard_number"])
        push_asset(view["first_frame_image"], f"镜头{number}首帧", "storyboard",
                   storyboardId=view["id"])
        push_asset(view["last_frame_image"], f"镜头{number}尾帧", "storyboard",
                   storyboardId=view["id"])
        push_asset(view["composed_image"], f"镜头{number}镜头图", "storyboard",
                   storyboardId=view["id"])
        for ref in _safe_parse_json_array(view["reference_images"]):
            push_asset(ref, f"镜头{number}参考图", "storyboard", storyboardId=view["id"])
    for scene in scene_rows:
        location = _col(scene, "location")
        time = _col(scene, "time")
        push_asset(
            _col(scene, "image_url"),
            f"{location}{f'（{time}）' if time else ''}场景",
            "scene",
            sceneId=_col(scene, "id"),
        )
    for char in chars:
        char_dict = dict(char._mapping) if hasattr(char, "_mapping") else char
        push_asset(
            char_dict.get("image_url"),
            f"{char_dict.get('name')}角色",
            "character",
            characterId=char_dict.get("id"),
        )

    return [
        {**asset, "imageIndex": index + 1, "imageLabel": f"图片{index + 1}"}
        for index, asset in enumerate(assets)
    ]


def build_reference_legend(reference_assets: list[dict[str, Any]]) -> str:
    """参考图映射说明：``图片1=…；图片2=…``（**全角分号**）。"""
    if not reference_assets:
        return ""
    return "；".join(f"{asset['imageLabel']}={asset['label']}" for asset in reference_assets)


def build_grid_prompt(
    conn: Connection,
    mode: str,
    storyboards: list[Any],
    rows: int,
    cols: int,
    drama_style: str | None,
    reference_assets: list[dict[str, Any]],
) -> str:
    """构建宫格图整体 prompt（三种模式：``first_frame`` / ``first_last`` / ``multi_ref``）。

    统一注入：风格、参考图映射、角色外观、运镜构图。

    ⚠️ 画风回退是**字面量** ``cinematic illustration style``（**不是** ``VISUAL_STYLE_MASTER``）——
    与其它链路的回退不同，别"统一"掉。
    ⚠️ 末尾用 ``filter(Boolean).join('\\n')``：**空串会被丢掉**（与第 2 块的 ``, `` 拼接相反）。
    """
    style = (drama_style and DRAMA_ART_STYLE_MAP.get(drama_style)) or "cinematic illustration style"
    legend = build_reference_legend(reference_assets)
    views = [_sb_view(sb) for sb in storyboards]
    storyboard_character_ids = _get_storyboard_character_ids(conn, [view["id"] for view in views])
    char_appearance_map = _build_character_appearance_map(conn, storyboard_character_ids)

    if mode == "first_frame":
        cells = []
        for i, view in enumerate(views):
            desc = _build_enriched_cell_description(
                view, i, storyboard_character_ids, char_appearance_map, reference_assets
            )
            cells.append(f"{_cell_label(i, rows, cols)}: {desc}")
        return _join_lines([
            f"{rows}x{cols} grid layout, consistent art style, {style},",
            f"参考图映射：{legend}" if legend else "",
            "当画面涉及角色或场景时，优先使用对应的图片编号来约束一致性。",
            *cells,
            "high quality, cinematic lighting, no text, no watermark",
        ])

    if mode == "first_last":
        total_cells = rows * cols
        cells = []
        for i in range(total_cells):
            view = views[i % len(views)]  # ⚠️ 空列表会抛错（原 TS 同样是崩，不额外兜底）
            desc = view["image_prompt"] or view["description"] or view["title"] or f"shot {i + 1}"
            action = view["action"] or view["movement"] or ""
            refs = _build_storyboard_reference_hints(view, reference_assets, storyboard_character_ids)
            is_first = i % 2 == 0
            composition = get_camera_movement_composition(
                view["movement"] or "", "start" if is_first else "end"
            )
            comp = f", {composition}" if composition else ""
            frame_hint = (
                "opening moment"
                if is_first
                else f"{f'{action}, ' if action else ''}closing moment, subtle motion change"
            )
            refs_part = f"参考{'、'.join(refs)}，" if refs else ""
            cells.append(f"{_cell_label(i, rows, cols)}: {refs_part}{desc}, {frame_hint}{comp}")
        return _join_lines([
            f"{rows}x{cols} grid layout, consistent art style, {style},",
            f"参考图映射：{legend}" if legend else "",
            "first/last frame visual rhythm, alternating opening and closing beats across the grid,",
            *cells,
            "continuous motion implied between left and right, high quality, no text",
        ])

    if mode == "multi_ref":
        first = views[0]
        desc = first["image_prompt"] or first["description"] or first["title"] or "scene"
        total_cells = rows * cols
        cells = [
            f"{_cell_label(i, rows, cols)}: "
            f"{f'参考{legend}，' if legend else ''}{desc}, {_GRID_ANGLES[i % len(_GRID_ANGLES)]}"
            for i in range(total_cells)
        ]
        return _join_lines([
            f"{rows}x{cols} grid layout, same scene different angles and compositions, {style},",
            f"参考图映射：{legend}" if legend else "",
            f"main scene: {desc},",
            *cells,
            "consistent lighting and color palette, high quality, no text",
        ])

    return f"{rows}x{cols} grid, {style}, storyboard frames, high quality"


def _join_lines(parts: list[str]) -> str:
    """``parts.filter(Boolean).join('\\n')`` —— **空串被丢掉**（与 ``, `` 拼接不同）。"""
    return "\n".join(part for part in parts if js_truthy(part))


def build_grid_cell_prompts(
    conn: Connection,
    mode: str,
    storyboards: list[Any],
    rows: int,
    cols: int,
    reference_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建宫格图**逐格** prompt（与 ``build_grid_prompt`` 同源，供 split 回写/逐格生成）。

    ⚠️ 三个分支的**标点不同**（别统一）：``multi_ref`` 用半角 ``': '``；
    ``first_last`` 用全角 ``，首帧：`` / ``，尾帧：``；默认分支用全角 ``：``。
    """
    if not storyboards:
        return []
    views = [_sb_view(sb) for sb in storyboards]
    storyboard_character_ids = _get_storyboard_character_ids(conn, [view["id"] for view in views])

    if mode == "multi_ref":
        first = views[0]
        desc = first["image_prompt"] or first["description"] or first["title"] or "scene"
        out: list[dict[str, Any]] = []
        for i in range(rows * cols):
            refs = _build_storyboard_reference_hints(first, reference_assets, storyboard_character_ids)
            refs_part = f"参考{'、'.join(refs)}，" if refs else ""
            out.append({
                "shot_number": first["storyboard_number"],
                "frame_type": "reference",
                "prompt": f"{_cell_label(i, rows, cols)}: {refs_part}{desc}, "
                          f"{_GRID_ANGLES[i % len(_GRID_ANGLES)]}",
            })
        return out

    if mode == "first_last":
        out = []
        for i in range(rows * cols):
            view = views[i % len(views)]
            desc = (view["image_prompt"] or view["description"] or view["title"]
                    or f"shot {view['storyboard_number'] or ''}")
            motion = view["action"] or view["movement"] or ""
            refs = _build_storyboard_reference_hints(view, reference_assets, storyboard_character_ids)
            is_first = i % 2 == 0
            composition = get_camera_movement_composition(
                view["movement"] or "", "start" if is_first else "end"
            )
            comp = f", {composition}" if composition else ""
            refs_part = f"参考{'、'.join(refs)}，" if refs else ""
            location = f", {view['location']}" if view["location"] else ""
            shot_type = f", {view['shot_type']}" if view["shot_type"] else ""
            label = _cell_label(i, rows, cols)
            if is_first:
                prompt = f"{label}，首帧：{refs_part}{desc}{location}{shot_type}{comp}"
            else:
                prompt = (f"{label}，尾帧：{refs_part}{desc}"
                          f"{f', {motion}' if motion else ''}{location}{shot_type}{comp}")
            out.append({
                "shot_number": view["storyboard_number"],
                "frame_type": "first_frame" if is_first else "last_frame",
                "prompt": prompt,
            })
        return out

    out = []
    for index, view in enumerate(views[:rows * cols]):
        desc = (view["image_prompt"] or view["description"] or view["title"]
                or f"shot {view['storyboard_number'] or ''}")
        refs = _build_storyboard_reference_hints(view, reference_assets, storyboard_character_ids)
        composition = get_camera_movement_composition(view["movement"] or "", "start")
        comp = f", {composition}" if composition else ""
        refs_part = f"参考{'、'.join(refs)}，" if refs else ""
        location = f", {view['location']}" if view["location"] else ""
        shot_type = f", {view['shot_type']}" if view["shot_type"] else ""
        out.append({
            "shot_number": view["storyboard_number"],
            "frame_type": "first_frame",
            "prompt": (f"{_cell_label(index, rows, cols)}：{refs_part}{desc}{location}{shot_type}"
                       f", opening scene{comp}"),
        })
    return out
