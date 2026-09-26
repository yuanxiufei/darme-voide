"""**打标提示词表** ✓ —— 原样搬自 LoRAMaster ✓（本仓唯一"照抄数据"的地方 ✓）。

出处（参考实现 ✓）：``reference/lora/LoRAMaster/dataset_manager/AutoCaptioning.py``
第 19~141 行 ✓：

* :data:`CAPTION_TYPE_MAP` —— 打标类型 → 三档模板（**第 0 行 = 不限定 ✓、
  第 1 行 = 限字数 ✓、第 2 行 = 限长度短语 ✓**）✓；
* :data:`EXTRA_OPTION_MAP` —— 那些开关 → 一句补充要求 ✓；
* :func:`build_prompt` —— 选模板 + 拼补充要求 + ``str.format`` 填参 ✓（同一函数第 219~244 行 ✓）。

⚠️ 三处**必须照抄、不许"顺手改好"**的地方 ✗（改了就和参考实现的出图风格对不上了 ✓✗）：
1. 模板文案**一个字都不动** ✓（包括 ``Straightforward`` 里那个弯引号 ``“This image is…”`` ✓）；
2. :data:`EXTRA_OPTION_MAP` 的**中文措辞**不动 ✓（这是给模型看的 ✓，不是给 UI 看的 ✓）；
3. :func:`build_prompt` 里 ``length`` 与 ``word_count`` **填的是同一个值** ✓ ——
   看着像笔误 ✗，但参考实现就是这么写的 ✓（``caption_length`` 既当"长度短语"又当"字数"✓）；
   ⚠️ 若取值是 ``"long"`` ✓，第 2 档模板会变成 "Write a long detailed description…" ✓ 正确 ✓；
   若是 ``"80"`` ✓，第 1 档模板会说 "within 80 words" ✓ 也正确 ✓ —— 靠**选档**而不是靠值本身区分 ✓。
"""
from __future__ import annotations

from typing import Sequence

#: 打标类型 → 三档模板 ✓（原样照抄 ✓，见模块头 ✓）
CAPTION_TYPE_MAP: dict[str, list[str]] = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

#: 「打标长度」下拉的取值 ✓（原样照抄第 19 行 ✓）—— 注意**全是字符串** ✓（含数字那一段 ✓）
CAPTION_LENGTHS: list[str] = ["any", "very short", "short", "medium-length", "long", "very long"] + [
    str(i) for i in range(20, 261, 10)
]

#: 补充开关 → 一句话要求 ✓（原样照抄第 113~141 行 ✓）
EXTRA_OPTION_MAP: dict[str, str] = {
    "refer_character_name": "如果图像中有人物/角色，必须以 {name} 来称呼他们。",
    "exclude_people_info": "不要包含无法更改的人物/角色信息（如种族、性别等），但可以包含可更改的属性（如发型）。",
    "include_lighting": "包含光照信息。",
    "include_camera_angle": "包含相机角度信息。",
    "include_watermark": "包含是否有水印的信息。",
    "include_JPEG_artifacts": "包含是否存在 JPEG 压缩痕迹的信息。",
    "include_exif": "如果是照片，必须包含可能使用的相机及其参数信息，如光圈、快门速度、ISO 等。",
    "exclude_sexual": "不要包含任何性内容；保持 PG 级别。",
    "exclude_image_resolution": "不要提及图像分辨率。",
    "include_aesthetic_quality": "必须包含图像主观美学质量的信息，从低到非常高。",
    "include_composition_style": "包含图像构图风格信息，如引导线、三分法或对称性。",
    "exclude_text": "不要提及图像中的任何文字。",
    "specify_depth_field": "指定景深，并说明背景是否清晰或模糊。",
    "specify_lighting_sources": "如适用，说明可能使用的人工或自然光源。",
    "do_not_use_ambiguous_language": "不要使用任何模糊语言。",
    "include_nsfw": "包含图像是否为 SFW、暗示性或 NSFW 的信息。",
    "only_describe_most_important_elements": "只描述图像中最重要的元素。",
    "do_not_include_artist_name_or_title": "如果是艺术作品，不包含艺术家姓名或作品标题。",
    "identify_image_orientation": "识别图像方向（纵向、横向或正方形）及明显的长宽比。",
    "use_vulgar_slang_and_profanity": "使用粗俗俚语和脏话，例如（但不限于）“fucking”，“slut”，“cock”等。",
    "do_not_use_polite_euphemisms": "不要使用礼貌的委婉语——采用直接、随意的表达方式。",
    "include_character_age": "如适用，包含人物/角色的年龄信息。",
    "include_camera_shot_type": "说明图像拍摄类型，如极近景、近景、中近景、中景、牛仔镜头、中远景、远景或极远景。",
    "exclude_mood_feeling": "不要描述图像的情绪/感觉等。",
    "include_camera_vantage_height": "明确指定拍摄高度（视平线、低角度、虫眼视角、鸟瞰、无人机、屋顶等）。",
    "mention_watermark": "如果有水印，必须提及。",
    "avoid_meta_descriptive_phrases": "你的回复将用于文本生成图像模型，因此避免无用的元描述短语，例如“这张图片显示…”，“你看到的是…”，等等。",
}

#: 打标模型 ✓（参考实现第 250 行**固定**写死 ✓ —— 本仓保持同一默认 ✓，
#: 但允许用环境变量 ``VOIDE_CAPTION_MODEL`` 换成**本地已下好的**同一个模型的另一份 ✓，
#: 免得每台机器都去联网拉 8B 权重 ✓）
DEFAULT_CAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"

#: 量化档 ✓（照抄第 100~111 行 ✓；值里的 ``torch.dtype`` 在 caption.py 里现取 ✓ —— 这里只放形状 ✓）
QUANTIZATION_MODES: tuple[str, ...] = ("nf4", "int8", "bf16")

#: 参考实现写死的 system 提示 ✓（第 392 行 ✓）
SYSTEM_PROMPT = "You are a helpful image captioner."


def build_extra_options(selected: Sequence[str], character_name: str = "Huluwa") -> list[str]:
    """把勾选的开关翻成要求句 ✓（照抄第 212~217 行 ✓，含 ``"Huluwa"`` 这个默认角色名 ✓）。

    ⚠️ 不认识的键 ⇒ **跳过** ✓（参考实现也是跳过 ✓）—— 这里不是兜底 ✗：
    键来自本文件自己的 :data:`EXTRA_OPTION_MAP` ✓，跳过的只可能是拼错 ✓，
    所以另外用 :func:`unknown_extra_options` 让调用方**能查** ✓（不静默 ✗）。
    """
    return [EXTRA_OPTION_MAP[key].format(name=character_name) for key in selected if key in EXTRA_OPTION_MAP]


def unknown_extra_options(selected: Sequence[str]) -> list[str]:
    """勾选里有哪些**不认识**的键 ✓（给校验用 ✓ —— 空列表 = 全认识 ✓）。"""
    return [key for key in selected if key not in EXTRA_OPTION_MAP]


def build_prompt(caption_type: str, caption_length: str, extra_options: Sequence[str] | None = None,
                 user_prompt: str = "", character_name: str = "Huluwa") -> str:
    """拼打标提示词 ✓（照抄第 219~244 行 ✓，但**把两处隐含用法显式化** ✓）。

    ⚠️ 与参考实现的四点差别 ✓（前三处是**修它一个真 bug** ✓，最后一处是"不静默" ✓）：
    1. ``extra_options`` 在本仓收的是**开关键列表** ✓（就是 ``run_caption()`` 里
       ``extra_options.append(key)`` 攒出来的那个列表 ✓）；参考实现在函数体里写的是
       ``extra, name_input = extra_options`` ✗✗ —— 那是把入参当**二元组**拆 ✓，
       而调用方传的是**任意长度的键列表** ✓✗ ⇒ 只有恰好勾中 2 个开关时才不炸 ✓✗。
       本仓由本函数自己调 :func:`build_extra_options` 翻译 ✓，0 个、3 个、27 个都对 ✓。
    2. ``caption_type`` 不认识 ⇒ **报错点名** ✓（参考实现直接 ``KeyError`` ✗，消息里看不出
       "是你这个类型没配" ✓✗）；
    3. 开关键不认识 ⇒ **报错点名** ✓（参考实现静默跳过 ✗）；
    4. ``str.format`` 失败（自定义提示词里有裸 ``{`` ✓，例如贴了段 JSON ✗）⇒
       **报错说清是自定义提示词的问题** ✓（参考实现直接把
       ``KeyError``/``IndexError`` 冒出来 ✗，看不出是谁的锅 ✓✗）。
    """
    selected = list(extra_options or [])
    if caption_type not in CAPTION_TYPE_MAP:
        raise KeyError(
            f"不认识的打标类型 ✗：{caption_type!r}；合法值：{sorted(CAPTION_TYPE_MAP)}"
        )
    unknown = unknown_extra_options(selected)
    if unknown:
        raise KeyError(f"不认识的打标开关 ✗：{unknown}；合法值：{sorted(EXTRA_OPTION_MAP)}")

    if user_prompt and user_prompt.strip():
        prompt = user_prompt.strip()
    else:
        # 选档：any → 第 0 档 ✓；纯数字 → 第 1 档（限字数）✓；其余 → 第 2 档（限长度短语）✓
        if caption_length == "any":
            map_index = 0
        elif isinstance(caption_length, str) and caption_length.isdigit():
            map_index = 1
        else:
            map_index = 2
        prompt = CAPTION_TYPE_MAP[caption_type][map_index]

    if selected:
        prompt += " " + " ".join(build_extra_options(selected, character_name))
    try:
        return prompt.format(name=character_name, length=caption_length, word_count=caption_length)
    except (KeyError, IndexError, ValueError) as err:
        raise KeyError(
            f"打标提示词里有个填不了的占位符 ✗（只认 {{name}} / {{length}} / {{word_count}} ✓）："
            f"{err}；出问题的提示词：{prompt[:200]!r}"
        ) from err
