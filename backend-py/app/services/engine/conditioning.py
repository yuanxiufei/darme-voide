"""**条件生成**（图生视频的"首帧/参考图"那一半 ✓）—— 零依赖 ✓ 只算**掩码**。

## 分工（这是本模块唯一的设计决定 ✓）

图生视频要「首帧来自图片、后面交给模型」✗ —— 但**"怎么把图片潜变量和噪声拼起来"是实现相关的**
✗：有的在通道维拼接 ✓、有的在时间维替换 ✓、有的走 mask 输入 ✓，形状布局各家不同 ✗。
⇒ **引擎算掩码（与模型无关 ✓），后端把掩码套到自己的布局上** ✓。
引擎若去猜布局，只会在真机上错得莫名其妙 ✗（与 :mod:`app.services.engine.geometry`
「不猜时间压缩比」是同一条纪律 ✓）。

## 掩码语义（纯数学 ✓ 公开做法 ✓）

设潜空间帧数 ``L``（由 ``(帧数-1)//压缩比 + 1`` 得到 ✓ 见 :func:`geometry.latent_frames` ✓），
每帧一个权重 ✓：

```
权重 = strength ×  clamp(1 − (i − keep + 1)/fade)
```

* ``keep`` 帧完全来自首帧（权重 1 ✓，默认 1 ✓）；
* 随后 ``fade`` 帧线性过渡到 0 ✓（``fade = 0`` ⇒ 硬切 ✓）；
* 其余为 0 ✓（纯模型生成 ✓）。

⚠️ **不猜的部分**：``keep``/``fade``/``strength`` 的**好取值取决于模型与权重** ✗ ⇒
本模块只提供默认值（``keep=1, fade=0, strength=1`` ✓ = 标准首帧条件 ✓），
别的一律由调用方显式给 ✓。

## H3 参考素材的 **prompt 契约**（2026-09-24 补 ✓）

H3 的参考素材**不是「挂上就生效」** ✗：模型要不要「用」参考音频/视频，取决于提示词里的
**结构化声明** ✓ —— 三行协议键 ``subject_definitions`` / ``retention_analysis`` /
``detailed_description`` ✓，加关系协议词 ``fully_copy`` / ``partially_copy`` / ``reference`` ✓。

⚠️ **实测坑** ✗（参考实现 v1.14.1 的根因 ✓）：只写 ``<Audio 1>`` 这类**绑定句不算声明** ✗ ⇒
若照「看见标签就跳过」的写法，自动声明会被吞掉 ⇒ 成片音轨与配音**相关性≈0** ✓✗。
故跳过判据必须是「**用户真的写了保留声明**」✓（见 :func:`declaration_plan` ✓）。

⚠️ **本仓与参考实现的边界**（2026-09-24 定 ✓）：
* **协议键/协议词逐字照抄** ✓（那是模型接口 ✓，写错等于没声明 ✗）；
* **声明句子由本仓自己写** ✓（语义对齐官方 R2V 指南 ✓，但**不逐字搬**参考实现的文案/代码 ✗）；
* ⚠️ 区块**题注**（``AUDIO_BLOCK_HEADER`` ✓）**未核出**是协议的一部分 ✗ ⇒ 调用方可覆盖 ✓、别当协议 ✗；
* ⚠️ **不猜**：声明**能不能真生效**只能有真权重时验 ✗ ⇒ 本模块只保证**结构正确** ✓，
  以及把「该报的错」在**提交前**报出来 ✓（见 :func:`validate_markup` / :func:`select_h3_task` ✓）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["AUDIO_BLOCK_HEADER", "AUDIO_RELATIONSHIPS", "AudioReference", "ConditioningConfig",
           "ConditioningError", "DECLARATION_KEYS", "DeclarationPlan", "H3TaskError",
           "MAX_REFERENCE_VIDEOS", "VIDEO_BLOCK_HEADER", "apply_declarations", "audio_declaration",
           "audio_slots", "declaration_plan", "first_frame_mask", "h3_markup",
           "latent_frames_for", "markup_references", "picture_slots", "select_h3_task",
           "validate_markup", "video_declaration"]


@dataclass(frozen=True)
class ConditioningConfig:
    """首帧条件配置（不可变 ✓）。"""

    #: 完全取自首帧的**潜帧数** ✓（默认 1 ✓ —— 标准首帧条件 ✓）
    keep: int = 1
    #: 过渡潜帧数 ✓（``0`` = 硬切 ✓）
    fade: int = 0
    #: 整体强度 ✓（``0..1``；``1`` = 首帧完全生效 ✓）
    strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"keep": self.keep, "fade": self.fade, "strength": self.strength}


class ConditioningError(ValueError):
    """条件无法构造 ✓（**明确报错**比悄悄退化成"文生视频"好得多 ✗ —— 见 :func:`latent_frames_for` ✓）。"""


class H3TaskError(ConditioningError):
    """本段该走哪条 H3 条件路径**判不出来 / 装不上** ✓ ⇒ 宁可当场拒 ✗（见 :func:`select_h3_task` ✓）。"""


#: 参考关系的**协议词** ✓ —— 逐字 ✓（模型按它决定「怎么用」这路参考素材 ✓；写错等于没声明 ✗）。
#: * ``fully_copy``：整轨 1:1 复用（配音当最终音轨 ✓ 画面对口型 ✓）；
#: * ``partially_copy``：只复用**对话层** ✓，环境音/音效/配乐由模型新生成 ✓；
#: * ``reference``：只学**音色与语气** ✓，台词按提示词重新生成 ✓。
AUDIO_RELATIONSHIPS: tuple[str, ...] = ("fully_copy", "partially_copy", "reference")

#: 三行结构化声明的**协议键** ✓ —— 逐字 ✓（模型按它解析 ✓）。
DECLARATION_KEYS: tuple[str, ...] = ("subject_definitions", "retention_analysis",
                                    "detailed_description")

#: 参考视频路数上限 ✓（口径：H3 原生最多 3 路、单路 2~15 s ✓ —— 时长那半边本模块看不到 ✗，
#: 只能在提交前由调用方核 ✓）。
MAX_REFERENCE_VIDEOS = 3

#: 声明区块的**题注** ✓。⚠️ **未核出**它是模型协议的一部分 ✗（协议部分是那三行键 ✓）⇒
#: 调用方可以覆盖成自己的写法 ✓；别把它当协议依赖 ✗。
AUDIO_BLOCK_HEADER = "[reference generation + audio reference]"
VIDEO_BLOCK_HEADER = "[reference generation + video reference]"


@dataclass(frozen=True)
class AudioReference:
    """一路参考音频 ✓ 与它的保留关系 ✓。

    ``index`` 是 ``<Audio index>`` 里的编号 ✓（从 1 起 ✓）；``speaker`` 是提示词里
    ``(S<speaker>)`` 的编号 ✓（多角色对话时每路可绑不同说话人 ✓）。
    """

    index: int
    relation: str
    speaker: int = 1

    def tag(self) -> str:
        return f"<Audio {self.index}>"

    @property
    def speaker_tag(self) -> str:
        return f"(S{self.speaker})"

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "relation": self.relation, "speaker": self.speaker}


def _check_audio_refs(refs: Sequence[AudioReference]) -> list[AudioReference]:
    if not refs:
        raise ConditioningError(
            "没有任何参考音频 ⇒ 不该调 audio_declaration ✓（空声明只会往提示词里塞噪声 ✗）")
    seen: set[int] = set()
    for ref in refs:
        if ref.relation not in AUDIO_RELATIONSHIPS:
            raise ConditioningError(
                f"参考关系 {ref.relation!r} 不在协议词里 ✗ ⇒ 只能是 "
                + " / ".join(AUDIO_RELATIONSHIPS)
                + " ✓（协议词写错等于**没声明** ✗ —— 模型会忽略这路参考 ✓✗）")
        if ref.index < 1 or ref.speaker < 1:
            raise ConditioningError(
                f"编号必须从 1 起（收到 index={ref.index}, speaker={ref.speaker} ✗）"
                f" —— ``<Audio 0>`` 不是合法标签 ✓✗")
        if ref.index in seen:
            raise ConditioningError(
                f"<Audio {ref.index}> 声明了两次 ✗ ⇒ 同一路参考只能出现一次 ✓"
                f"（两路不同音频请各用各的编号 ✓）")
        seen.add(ref.index)
    return list(refs)


def _declared(header: str | None, keys: Sequence[tuple[str, Sequence[str]]]) -> str:
    lines = [f"{key}: {' '.join(parts)}" for key, parts in keys if parts]
    if not lines:
        return ""
    body = "\n".join(lines)
    return f"{header}\n{body}" if header else body


def audio_declaration(refs: Sequence[AudioReference], *,
                      header: str | None = AUDIO_BLOCK_HEADER) -> str:
    """一路或多路参考音频 → **结构化声明区块** ✓（三行协议键 ✓；不含题注除非给了 ``header`` ✓）。

    ⚠️ 这是「让模型真的用参考音频」的那一步 ✗：只写 ``<Audio 1>`` 绑定句**不算声明** ✗
    ⇒ 模型会自由发挥、不复用你的配音 ✓✗。⚠️ 只会**追加** ✓，不会改写你已有的段落 ✓。
    """
    definitions: list[str] = []
    retentions: list[str] = []
    details: list[str] = []
    for ref in _check_audio_refs(refs):
        tag, speaker = ref.tag(), ref.speaker_tag
        if ref.relation == "fully_copy":
            definitions.append(f"{tag} is the dialogue and voice reference for speaker {speaker}.")
            retentions.append(f"{tag}: fully_copy - {tag} is reused 1:1 as the complete final "
                              f"audio track of the target video.")
            details.append(f"Speaker {speaker} performs exactly the lines of {tag}, with lip "
                           f"movements precisely synchronized to {tag}.")
        elif ref.relation == "partially_copy":
            definitions.append(f"{tag} is the dialogue and voice reference for speaker {speaker}.")
            retentions.append(f"{tag}: partially_copy - the dialogue layer of {tag} is reused 1:1 "
                              f"as the dialogue track of the target video, while ambience, sound "
                              f"effects and music are generated anew around it.")
            details.append(f"Speaker {speaker} performs exactly the lines of {tag} with lip "
                           f"movements precisely synchronized, while ambience and effects are "
                           f"generated naturally.")
        else:  # reference：只学音色 ✓
            definitions.append(f"{tag} is the voice-timbre reference for speaker {speaker}.")
            retentions.append(f"{tag}: reference - the target speaker follows the voice timbre "
                              f"and delivery of {tag} without copying the original signal.")
            details.append(f"Speaker {speaker} speaks the lines described above using the voice "
                           f"timbre referenced from {tag}.")
    return _declared(header, (("subject_definitions", definitions),
                              ("retention_analysis", retentions),
                              ("detailed_description", details)))


def video_declaration(count: int, *, header: str | None = VIDEO_BLOCK_HEADER) -> str:
    """参考视频 → **结构化声明区块** ✓：运镜/动作/走位/节奏 1:1 复刻 ✓ + 主体**全部替换**为参考图主体 ✓。

    ⚠️ 官方关系是 ``reference`` ✓（动作跟视频、外观跟参考图 ✓）；⚠️ 路数上限 :data:`MAX_REFERENCE_VIDEOS` ✓，
    单路时长（2~15 s ✓）本模块看不到 ✗ ⇒ 由调用方在提交前核 ✓。
    """
    total = int(count)
    if total < 1:
        raise ConditioningError(f"参考视频路数必须 ≥1（收到 {count} ✗）—— 没有就别声明 ✓")
    if total > MAX_REFERENCE_VIDEOS:
        raise ConditioningError(
            f"参考视频 {total} 路超过上限 {MAX_REFERENCE_VIDEOS} 路 ✗ ⇒ H3 原生只接 3 路 ✓"
            f"（多出来的会被忽略 ✓✗）")
    tags = [f"<Video {index}>" for index in range(1, total + 1)]
    joined = " and ".join(tags)
    return _declared(header, (
        ("subject_definitions", [f"{tag} is a source video." for tag in tags]
         + ["The reference images supply the subjects' visual appearance (look, style, scene)."]),
        ("retention_analysis", [
            f"{tag}: reference - the camera trajectory, subject motion, blocking and rhythm of "
            f"{tag} are recreated 1:1, and every subject in {tag} is replaced by the subjects "
            f"from the reference images." for tag in tags]),
        ("detailed_description", [
            f"One continuous smooth take following the camera path and pacing of {joined}; all "
            f"appearances come from the reference images and the prompt. No cuts, no flicker, "
            f"no transitions of any kind."]),
    ))


@dataclass(frozen=True)
class DeclarationPlan:
    """要不要追加声明 ✓ 与**为什么** ✓（跳过不是「无事发生」✗ —— 得能说清是谁写的 ✓）。"""

    audio_needed: bool
    video_needed: bool
    audio_skipped: str | None = None
    video_skipped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"audioNeeded": self.audio_needed, "videoNeeded": self.video_needed,
                "audioSkipped": self.audio_skipped, "videoSkipped": self.video_skipped}


def declaration_plan(prompt: str, *, audio_refs: Sequence[AudioReference] = (),
                     video_count: int = 0) -> DeclarationPlan:
    """判断这两块声明**要不要**自动追加 ✓。

    跳过判据（⚠️ 这里就是那个坑的落点 ✗）：
    * **音频**：提示词里已有 ``retention_analysis`` 或 ``fully_copy`` ⇒ 认为**用户自己写了** ⇒ 跳过 ✓；
      ⚠️ **只出现 ``<Audio N>`` 标签不算** ✗✗（照标签跳 ⇒ 自动声明被吞 ⇒ 音轨与配音相关性≈0 ✓✗）；
    * **视频**：``retention_analysis`` **和** ``<Video`` **同时**出现 ⇒ 跳过 ✓（只写 ``<Video 1>``
      引用句不算 ✗ —— 同上，否则模型不跟视频 ✓✗）。
    """
    text = str(prompt or "")
    if not audio_refs:
        plan_audio, audio_reason = False, None
    elif "retention_analysis" in text or "fully_copy" in text:
        plan_audio, audio_reason = False, "提示词里已有保留声明（retention_analysis / fully_copy ✓）⇒ 不重复追加 ✓"
    else:
        plan_audio, audio_reason = True, None
    if int(video_count) <= 0:
        plan_video, video_reason = False, None
    elif "retention_analysis" in text and "<Video" in text:
        plan_video, video_reason = False, "提示词里同时有 retention_analysis 与 <Video ✓ ⇒ 视为用户已写 ✓"
    else:
        plan_video, video_reason = True, None
    return DeclarationPlan(audio_needed=plan_audio, video_needed=plan_video,
                           audio_skipped=audio_reason, video_skipped=video_reason)


def apply_declarations(prompt: str, *, audio_refs: Sequence[AudioReference] = (),
                       video_count: int = 0, audio_header: str | None = AUDIO_BLOCK_HEADER,
                       video_header: str | None = VIDEO_BLOCK_HEADER) -> tuple[str, DeclarationPlan]:
    """把该追加的声明**追加**到提示词尾部 ✓ ⇒ ``(最终提示词, 计划 ✓)``。

    ⚠️ 只追加 ✓（不重排、不改写原文 ✓）；⚠️ 判据见 :func:`declaration_plan` ✓（标签不算声明 ✗）。
    """
    plan = declaration_plan(prompt, audio_refs=audio_refs, video_count=video_count)
    text = str(prompt or "").rstrip()
    blocks: list[str] = []
    if plan.audio_needed:
        blocks.append(audio_declaration(audio_refs, header=audio_header))
    if plan.video_needed:
        blocks.append(video_declaration(video_count, header=video_header))
    if blocks:
        text = text + "\n\n" + "\n\n".join(blocks)
    return text, plan


#: 文本写法 → ``<Picture N>`` ✓（口径：``@图1`` / ``@image1`` / ``@img1`` / ``【图1】`` ✓；
#: ⚠️ 本仓的容错：允许中间有空格 ✓、英文大小写不敏感 ✓）。
_PICTURE_RE = re.compile(r"@\s*(?:图|image|img)\s*(\d+)|[【\[]\s*图\s*(\d+)\s*[】\]]", re.IGNORECASE)
#: 文本写法 → ``<Audio N>`` ✓（``@音1`` ✓；``@audio1`` 是本仓加的英文对称写法 ✓）。
_AUDIO_RE = re.compile(r"@\s*(?:音|audio)\s*(\d+)", re.IGNORECASE)
#: 已经写好的标签 ✓（用来做**越界校验** ✓）。
_TAG_RE = re.compile(r"<(Picture|Audio|Video)\s+(\d+)>")


def h3_markup(text: str) -> str:
    """把 ``@图N`` / ``【图N】`` / ``@音N`` 转成 H3 认的 ``<Picture N>`` / ``<Audio N>`` ✓。

    ⚠️ 只转**文本写法** ✓；已经是 ``<Picture N>`` 的原样留着 ✓（幂等 ✓）；
    ⚠️ 编号 0 会**报错** ✗（``<Picture 0>`` 不是合法标签 ✓）。
    """
    def picture(match: re.Match[str]) -> str:
        number = int(match.group(1) or match.group(2))
        if number < 1:
            raise ConditioningError(
                f"参考图编号必须从 1 起（收到 {match.group(0)!r} ✗）—— ``<Picture 0>`` 不是合法标签 ✓✗")
        return f"<Picture {number}>"

    def audio(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number < 1:
            raise ConditioningError(
                f"参考音频编号必须从 1 起（收到 {match.group(0)!r} ✗）—— ``<Audio 0>`` 不是合法标签 ✓✗")
        return f"<Audio {number}>"

    return _AUDIO_RE.sub(audio, _PICTURE_RE.sub(picture, str(text or "")))


def markup_references(text: str) -> dict[str, list[int]]:
    """列出文本里已经出现的 H3 标签编号 ✓（``{\"Picture\": [1, 2], ...}`` ✓）。"""
    found: dict[str, list[int]] = {}
    for kind, number in _TAG_RE.findall(str(text or "")):
        found.setdefault(kind, []).append(int(number))
    return {kind: sorted(set(numbers)) for kind, numbers in found.items()}


def validate_markup(prompt: str, *, pictures: int = 0, audios: int = 0, videos: int = 0) -> None:
    """**提交前**核一遍：提示词引用的编号有没有对应的素材 ✓（编号错位是最常见的错 ✗）。

    ⚠️ 只报「引用了但没有」那半边 ✓；「有素材却没引用」不报 ✗（旁白类素材很常见 ✓）。
    """
    available = {"Picture": int(pictures), "Audio": int(audios), "Video": int(videos)}
    for kind, numbers in markup_references(prompt).items():
        limit = available.get(kind, 0)
        too_big = [number for number in numbers if number > limit]
        if too_big:
            raise ConditioningError(
                f"提示词引用了 <{kind} {too_big[0]}> ✗，但本段只给了 {limit} 个 {kind} 素材 ✓"
                f" —— 编号错位是最常见的错 ✓（⚠️ 续接尾帧**排最前** ✓、占 <Picture 1> ✓，"
                f"换场景/换人物后编号会**前移** ✗ ⇒ 建议用 :func:`picture_slots` 对一遍 ✓）")


def picture_slots(*, tail: bool, images: int) -> list[tuple[int, str]]:
    """参考图**编号对照表** ✓ ⇒ ``[(1, \"续接尾帧\"), (2, \"上传图 1\"), ...]``。

    ⚠️ 口径：**续接尾帧排最前** ✓，然后才是本段上传的图 ✓（参考实现口径 ✓）——
    所以「取消续接尾帧」会让后面所有编号**前移一格** ✗（提示词里手写的 ``@图N`` 会指错 ✓✗）。
    """
    count = int(images)
    if count < 0:
        raise ConditioningError(f"上传图数量不能为负（收到 {images} ✗）")
    table: list[tuple[int, str]] = []
    if tail:
        table.append((1, "续接尾帧"))
    for index in range(1, count + 1):
        table.append((len(table) + 1, f"上传图 {index}"))
    return table


def audio_slots(*, segment_audio: bool, voices: int) -> list[tuple[int, str]]:
    """参考音频**编号对照表** ✓ ⇒ ``[(1, \"本段配音（参考驱动）\"), (2, \"音色槽 A1\"), ...]``。

    ⚠️ 口径：本段音频的「参考音频驱动」占第 1 路时，**音色槽编号自动从 A2 起** ✓（参考实现口径 ✓）。
    """
    count = int(voices)
    if count < 0:
        raise ConditioningError(f"音色槽数量不能为负（收到 {voices} ✗）")
    table: list[tuple[int, str]] = []
    if segment_audio:
        table.append((1, "本段配音（参考驱动）"))
    for index in range(1, count + 1):
        table.append((len(table) + 1, f"音色槽 A{index}"))
    return table


def select_h3_task(*, primary_model_kind: str, has_optional_fl2va: bool = False,
                   has_references: bool = False, has_first_frame: bool = False,
                   prefer_fl2va: bool = False) -> str:
    """本段走 **Ref2VA** 还是 **FL2VA** ✓ ⇒ ``\"ref2va\"`` / ``\"fl2va\"``。

    规则（口径来自参考实现 ✓；判据与错因由本仓自己写 ✓）：
    1. 主模型是 FL2VA **且本段有参考素材** ⇒ **拒** ✗✗（FL2VA 单模型通道不支持参考 ⇒
       强行生成会**丢素材或直接失败** ✓✗）—— 这是**硬约束** ✓，不许「悄悄忽略参考图」✗；
    2. 本段需要**硬首帧**续接 ⇒ 必须有 FL2VA ✓（主模型是 Ref2VA 又没接第二模型 ⇒ 拒，并给替代方案 ✓）；
    3. 有参考素材 ⇒ Ref2VA ✓；
    4. 其余：``prefer_fl2va`` 且可用 ⇒ FL2VA ✓；否则 Ref2VA ✓。

    ⚠️ 判不出模型形态（不是 ``ref2va``/``fl2va``）⇒ **拒** ✗ 不猜 ✓。
    """
    kind = str(primary_model_kind or "").strip().lower()
    if kind not in ("ref2va", "fl2va"):
        raise H3TaskError(
            f"模型形态 {primary_model_kind!r} 判不出来 ✗ ⇒ 本引擎**不猜**它支持什么条件 ✓"
            f"（猜错会装出一个形状对、能力不对的模型 ✗✗）。请给 ``ref2va`` 或 ``fl2va`` ✓，"
            f"或先按权重文件名/键名把形态定下来 ✓（见 :mod:`app.services.engine.h3_keys` ✓）")
    if kind == "fl2va" and has_references:
        raise H3TaskError(
            "主模型是 FL2VA，而本段带参考素材（参考图/音频/视频 ✗）⇒ **直接拒** ✗✗："
            "参考素材只能由 Ref2VA 生成 ✓（FL2VA 单模型通道不支持参考 ⇒ 强行生成会**丢素材或失败** ✓✗）。"
            "解法：① 本机已装 Ref2VA 时，改走 Ref2VA 重试 ✓；"
            "② 没装就把 ``minimax_h3_ref2va_*`` 放到 ComfyUI 的 ``models/diffusion_models`` 并重启 ✓；"
            "③ 本段确实只有文本 ⇒ 去掉参考素材 ✓。")
    fl2va_available = bool(has_optional_fl2va) or kind == "fl2va"
    if has_first_frame:
        if fl2va_available:
            return "fl2va"
        raise H3TaskError(
            "本段要**硬首帧**续接（FL2VA ✓），但当前没有可用的 FL2VA ✗"
            "（主模型是 Ref2VA 且没接第二模型 ✓）⇒ 二选一："
            "① 接上 FL2VA 第二模型 ✓；② 续接方式改成「软参考 Ref2VA（保人物 ✓）」✓。")
    if has_references:
        return "ref2va"
    if (prefer_fl2va and fl2va_available) or kind == "fl2va":
        return "fl2va"
    return "ref2va"


def latent_frames_for(frames: int, temporal_compression: int | None) -> int:
    """帧数 → 潜帧数 ✓；**没给压缩比就报错** ✗（不猜 ✓ —— 猜错会让首帧落在错误的帧上 ✗）。"""
    if not temporal_compression:
        raise ConditioningError(
            "首帧条件需要 temporalCompression（潜空间时间压缩比 ✓）—— 本引擎**不猜**这个值 ✗："
            "猜错会把首帧放到错误的潜帧上 ✗。请在请求里显式给出 ✓。")
    from . import geometry

    return geometry.latent_frames(int(frames), temporal_compression=int(temporal_compression))


def first_frame_mask(latent_frames: int, config: ConditioningConfig | None = None) -> list[float]:
    """算出**每个潜帧的权重** ✓（长度 = ``latent_frames`` ✓；越界取值会**报错**而不是截断 ✗）。"""
    config = config or ConditioningConfig()
    total = max(1, int(latent_frames))
    keep = int(config.keep)
    fade = max(0, int(config.fade))
    strength = float(config.strength)
    if keep < 0 or fade < 0:
        raise ConditioningError(f"keep/fade 不能为负（收到 keep={keep}, fade={fade} ✗）")
    if not 0.0 <= strength <= 1.0:
        raise ConditioningError(f"strength 必须落在 0..1（收到 {strength} ✗）")
    if keep > total:
        raise ConditioningError(
            f"keep={keep} 超过潜帧数 {total} ✗ ⇒ 这等于「整段都给首帧」✗，几乎肯定是参数写错了 ✓")
    if keep + fade > total:
        raise ConditioningError(
            f"keep+fade = {keep + fade} 超过潜帧数 {total} ✗ ⇒ 过渡区放不下 ✓")

    mask: list[float] = []
    for index in range(total):
        if index < keep:
            weight = 1.0
        elif fade and index < keep + fade:
            # 第 keep 帧起线性淡出到 0 ✓（在 keep+fade 处正好为 0 ✓）
            weight = max(0.0, 1.0 - (index - keep + 1) / fade)
        else:
            weight = 0.0
        mask.append(round(weight * strength, 6))
    return mask
