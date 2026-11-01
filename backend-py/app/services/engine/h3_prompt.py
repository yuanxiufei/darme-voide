"""H3 文本侧的**呈现格式**（presentation）+ **token 标签**（`minimax_token_tags`）✓ —— 纯 Python ✓ 零依赖 ✓。

## 为什么单独一个模块

H3 的文本条件**不是** chat 模板 ✓（`:mod:`.qwen3_te` 那种 ``<|im_start|>user\\n…`` 在这里**一个都不许加** ✗）——
参考 `comfy/text_encoders/minimax.py` 抬头逐字写着：

```
t2va:   <prompt>
fl2va:  "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>
ref2va: per condition in request order (1-based ordinals per type):
          image -> "<Picture i>: " <vision block>
          audio -> "<Audio j>: "              (audio never enters Qwen)
          video -> "<Video k>: " then per 2-frame temporal block
                   "<T.T seconds>" <vision block(2 frames)>
        then <prompt>
```

⇒ 「标签文本 + 视觉块」是**交错**的 ✓，而视觉块要占用若干个 **文本序列位置** ✓ ⇒ 两者必须**一起**算出来
（先算文本、再拼视觉 ⇒ 位置全错 ✓ 且不报错 ✗）。

## 三条会静默出错的硬事实

1. **逐段分词** ✗ 不是整串分词：参考对每个 ``add_text(s)`` **单独**调 ``tokenize_with_weights`` ✓
   ⇒ ``"<Picture 1>: "`` 与后面的 prompt 之间的 BPE 边界**断开** ✓。整串连起来分词结果**不同** ✓
   （不报错、只是标签 token 的 id 排布变了 ✓✗）⇒ 本模块输出的是**段序列**（:func:`segments` ✓），
   每个文本段必须**各自**分词 ✓。
2. **序号按类型独立计数** ✓，不是全局序号 ✗：``<Picture 1> <Audio 1> <Picture 2>`` 里第二张图是
   ``<Picture 2>`` ✓ 不是 ``<Picture 3>`` ✗。
3. **视觉块的时间戳是「两帧的中点」** ✓ 且格式写死 ``"<%.1f seconds>"`` ✓：
   ``<0.5 seconds>`` ✓ / ``<1.0 seconds>`` ✓（⚠️ ``1`` 会写成 ``1.0`` ✓ —— ``%.1f`` 补一位小数 ✓）。

## token 标签（DiT 的 adaLN 口径 ✓）

参考 ``token_tags_from_embeds_info`` 逐字：

```python
tags = torch.ones(seq_len, dtype=torch.long)
for e in embeds_info:
    if e.get("type") == "image":
        tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
```

⇒ **视觉块整段（含两侧的 ``<|vision_start|>`` / ``<|vision_end|>``）= 0** ✓、其余（正文、``<Picture 1>: ``、
``<Audio 1>: ``、``<T.T seconds>`` 这些**纯文本标签**）= 1 ✓。

⚠️ 与 :mod:`.h3_form` 的 ``mod_row`` 不是一回事 ✗✗：那边是**主干**里「唯一时间戳序号 × 3 + 模态」
（text=1 / video=0 / audio=2 ✓）；这边的 0/1 是**文本侧**「这个位置是视觉特征还是文本」✓。
两者数值上恰好都把 0 给了「视觉/video」✓、但**判据完全不同** ✓ —— 别混用 ✗。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "EXTRA_TOKENS", "FRAMES_PER_BLOCK", "IMAGE_PAD", "PAD_TOKEN_ID", "TAG_TEXT", "TAG_VISION",
    "VIDEO_FPS", "VISION_END", "VISION_START",
    "check_pieces", "presentation", "segments", "text_of", "token_tags", "video_blocks",
]

#: 视觉块的**包围 token** ✓（`minimax.py::VISION_START/END` ✓ id 由发布的词表固定 ✓）
VISION_START = 151652
VISION_END = 151653

#: 视觉特征占位 token ✓（``<|image_pad|>`` = **151655** ✓
#: 见 `comfy/ldm/hidream_o1/model.py::IMAGE_TOKEN_ID` 与 `qwen3vl.py:194` ✓）
#: ⚠️ 一个视觉块展开成 **N** 个它 ✓（N = ``(h/merge)·(w/merge)`` ✓ 视觉塔 merger 之后的 token 数 ✓）；
#: 写少 ⇒ 特征塞不下 ✓ 写多 ⇒ 序列尾部整体后移 ✓ 两者都**不报错** ✗
IMAGE_PAD = 151655

#: 兜底填空 token ✓（``MiniMaxH3ClipModel`` 的 ``special_tokens={"pad": 151643}`` ✓）
PAD_TOKEN_ID = 151643

#: FL2VA / Ref2VA 的 tokenizer_config 追加的 token ✓（id 由发布的词表固定 ✓ —— **不猜** ✗）
EXTRA_TOKENS: dict[str, int] = {
    "<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
    "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
    "<|caption_start|>": 151674, "<|caption_end|>": 151675,
}

#: 一个时间块吃几帧 ✓（= 视觉塔的 ``temporal_patch_size`` ✓）
FRAMES_PER_BLOCK = 2

#: 参考视频**采样率** ✓（``comfy/text_encoders/minimax.py``：``timestamps = [i / 2.0 …]`` ✓）
VIDEO_FPS = 2.0

#: 视觉块位置 = **0** ✓（参考注释：「VIDEO(0)」✓）
TAG_VISION = 0
#: 文本位置 = **1** ✓
TAG_TEXT = 1

#: 标签段的**字面形** ✓（结尾那个空格是词表的一部分 ✓ 少了 ⇒ 与 prompt 粘连 ✓✗）
_LABEL = re.compile(r"^<(Picture|Audio|Video) (\d+)>: $")


def default_timestamps(count: int) -> list[float]:
    """缺省时间戳 ✓ = 按 2 fps 采样 ✓（``[i / 2.0 for i in range(n)]`` ✓ 逐字照参考 ✓）。"""
    return [index / VIDEO_FPS for index in range(count)]


def video_blocks(count: int, timestamps: Sequence[float] | None = None) -> list[tuple[int, int, str]]:
    """把 ``count`` 帧切成**每 2 帧一块** ✓ ⇒ ``[(start, stop, "<T.T seconds>"), …]`` ✓。

    ⚠️ ``count`` 必须是**偶数** ✗✗：参考对奇数帧做的是
    ``frames = cat([frames, frames[-1:]]); timestamps += [timestamps[-1]]`` ✓（**重复末帧**）——
    那一步要动像素 ✓ ⇒ 归 :mod:`.qwen3vl_pixels` 的预处理 ✓，本模块**拒绝**奇数帧 ✓
    （在这里悄悄补一帧 ⇒ 时间戳与像素不同源 ✓ 且不报错 ✗）。
    """
    if count <= 0:
        raise ValueError(f"视频块帧数必须 > 0（收到 {count} ✗）")
    if count % FRAMES_PER_BLOCK:
        raise ValueError(
            f"视频块帧数 {count} 是奇数 ✗ —— 参考做法是**重复末帧**补齐到 {FRAMES_PER_BLOCK} 的倍数 ✓，"
            f"那一步要动像素 ⇒ 必须先过像素预处理 ✓（在这里补 ⇒ 时间戳与像素不同源 ✗）")
    stamps = list(timestamps) if timestamps is not None else default_timestamps(count)
    if len(stamps) != count:
        raise ValueError(
            f"时间戳 {len(stamps)} 个 ✗ ≠ 帧数 {count} ✓（少给会让后面的块整体错位 ✓ 且不报错 ✗）")
    blocks: list[tuple[int, int, str]] = []
    for start in range(0, count, FRAMES_PER_BLOCK):
        stop = start + FRAMES_PER_BLOCK
        middle = (float(stamps[start]) + float(stamps[start + 1])) / 2.0
        blocks.append((start, stop, "<%.1f seconds>" % middle))
    return blocks


def presentation(prompt: str, *, refs: Sequence[Mapping[str, Any]] | None = None,
                 images: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    """算出**段序列** ✓（文本段与视觉块**交替** ✓ —— 谁都不能事后补 ✗）。

    两种口径互斥 ✓（同时给 ⇒ 当场报错 ✗）：

    * ``refs``（ref2va ✓）：按请求顺序，``type`` ∈ ``image`` / ``audio`` / ``video`` ✓；
      ``video`` 还要 ``frames``（帧数 ✓）与可选 ``timestamps`` ✓。
    * ``images``（fl2va ✓）：每张图一个 ``<Picture i>: `` + 视觉块 ✓。

    返回项：

    * ``{"kind": "text", "text": "…"}`` ✓（**必须单独分词** ✓）
    * ``{"kind": "vision", "frames": None, "videoBlock": False}`` ✓（单图）
    * ``{"kind": "vision", "frames": (start, stop), "videoBlock": True}`` ✓（视频的 2 帧块）

    ⚠️ ``frames`` 是**帧下标**✗ 不是像素 ✓ —— 像素在调用方手里 ✓（本模块不 import torch ✓）。
    """
    if refs is not None and images is not None:
        raise ValueError("``refs`` 与 ``images`` 是**两条不同口径** ✗（ref2va / fl2va）⇒ 只许给一个 ✓")
    pieces: list[dict[str, Any]] = []
    if refs is not None:
        counters = {"image": 0, "audio": 0, "video": 0}
        for position, ref in enumerate(refs):
            kind = str(ref.get("type") or "")
            if kind not in counters:
                raise ValueError(
                    f"第 {position} 个参考项的类型 {kind!r} 不认识 ✗ —— 只认 "
                    f"{sorted(counters)} ✓（``audio`` 也只占文本 ✓ 不进 Qwen 视觉塔 ✓）")
            counters[kind] += 1
            if kind == "image":
                pieces.append({"kind": "text", "text": "<Picture %d>: " % counters["image"]})
                pieces.append({"kind": "vision", "frames": None, "videoBlock": False})
            elif kind == "audio":
                # ⚠️ 音频**只留文本标签** ✓：「audio never enters Qwen」✓（没有视觉块 ✓）
                pieces.append({"kind": "text", "text": "<Audio %d>: " % counters["audio"]})
            else:
                count = int(ref.get("frames") or 0)
                stamps = ref.get("timestamps")
                blocks = video_blocks(count, stamps)
                pieces.append({"kind": "text", "text": "<Video %d>: " % counters["video"]})
                for start, stop, stamp in blocks:
                    pieces.append({"kind": "text", "text": stamp})
                    #: ⚠️ ``frames`` 是**本路视频内**的帧下标 ✓（每路都从 0 起 ✓）⇒ 多路视频时
                    #: 光看区间**分不出是哪一路** ✓✗（三路各 4 帧 ⇒ 全是 ``(0,2)/(2,4)`` ✓）⇒
                    #: 必须带 ``videoIndex`` ✓（1 基 ✓，:func:`~.h3_encode.encode_condition` 靠它挑
                    #: 该路的帧序列 ✓；靠"区间从 0 起就换路"去猜 ⇒ 第一路只有一块时必错 ✓✗）。
                    pieces.append({"kind": "vision", "frames": (start, stop), "videoBlock": True,
                                   "videoIndex": counters["video"]})
    elif images is not None:
        for index, _image in enumerate(images):
            pieces.append({"kind": "text", "text": "<Picture %d>: " % (index + 1)})
            pieces.append({"kind": "vision", "frames": None, "videoBlock": False})
    if prompt:
        pieces.append({"kind": "text", "text": prompt})
    return pieces


def segments(pieces: Iterable[Mapping[str, Any]]) -> list[str]:
    """按顺序取出**文本段** ✓ —— 每段**各自**分词 ✓（合起来分一次 ⇒ BPE 边界变了 ✗）。"""
    return [str(piece["text"]) for piece in pieces if piece.get("kind") == "text"]


def text_of(pieces: Iterable[Mapping[str, Any]]) -> str:
    """把段拼成一条字符串 ✓ —— **只给人看 / 只给对照用** ✓✗ 别拿去分词（见 :func:`segments` ✓）。"""
    return "".join(segments(pieces))


def token_tags(seq_len: int, spans: Iterable[tuple[int, int]]) -> list[int]:
    """``seq_len`` 个位置的标签 ✓：视觉块（含两侧包围 token）= 0 ✓、其余 = 1 ✓。

    ``spans`` 是**视觉特征**在序列里的 ``(index, size)`` ✓（``index`` = 第一个视觉特征 token 的下标 ✓，
    包围用的 ``<|vision_start|>`` 在它**前一位** ✓、``<|vision_end|>`` 在 ``index + size`` ✓）。
    这正是参考 ``embeds_info`` 的 ``index`` / ``size`` 语义 ✓。

    ⚠️ 越界 ⇒ **当场报错** ✗（静默截断会把尾部正文误标成视觉 ✓ 且不报错 ✗）。
    """
    tags = [TAG_TEXT] * int(seq_len)
    for index, size in spans:
        start = int(index)
        size = int(size)
        if size <= 0:
            raise ValueError(f"视觉块大小必须 > 0（收到 {size} ✗）")
        stop = start + size + 1
        if start < 0 or stop > len(tags) + 1:
            raise ValueError(
                f"视觉块 [{start}, {stop}) 越出序列长度 {len(tags)} ✗ —— "
                f"位置是**算出序列之后**才能定的 ✓（先算 tags 再拼视觉 ⇒ 全错 ✓✗）")
        for position in range(max(0, start - 1), min(stop, len(tags))):
            tags[position] = TAG_VISION
    return tags


def check_pieces(pieces: Sequence[Mapping[str, Any]], prompt: str = "") -> list[str]:
    """形状自检 ✓ ⇒ 问题列表（空 = 没问题 ✓）。**不判**语义 ✗ —— 只判「段序与计数」这类机械事实 ✓。"""
    problems: list[str] = []
    vision_count = 0
    seen: dict[str, int] = {}
    for position, piece in enumerate(pieces):
        kind = piece.get("kind")
        if kind == "text":
            text = str(piece.get("text") or "")
            if not text:
                problems.append(f"第 {position} 段是**空文本** ✗（空段也会多切一刀分词 ✓ 无意义 ✗）")
            label = _LABEL.match(text)
            if label:
                name, ordinal = label.group(1), int(label.group(2))
                seen[name] = seen.get(name, 0) + 1
                if ordinal != seen[name]:
                    problems.append(
                        f"第 {position} 段的 ``<{name} {ordinal}>`` 序号不对 ✗ —— "
                        f"按**类型独立**计数该是 {seen[name]} ✓（全局序号 ✗）")
            if "\n" in text and text != prompt:
                problems.append(f"第 {position} 段含换行 ✗（标签段不该有换行 ✓）")
        elif kind == "vision":
            vision_count += 1
            frames = piece.get("frames")
            if piece.get("videoBlock"):
                if not (isinstance(frames, tuple) and len(frames) == 2
                        and int(frames[1]) - int(frames[0]) == FRAMES_PER_BLOCK):
                    problems.append(
                        f"第 {position} 段是视频块 ⇒ ``frames`` 必须是长度 {FRAMES_PER_BLOCK} 的区间 ✓"
                        f"（收到 {frames!r} ✗）")
            elif frames is not None:
                problems.append(f"第 {position} 段是单图 ⇒ ``frames`` 必须是 None ✓（收到 {frames!r} ✗）")
        else:
            problems.append(f"第 {position} 段的 kind {kind!r} 不认识 ✗（只认 text / vision ✓）")
    return problems
