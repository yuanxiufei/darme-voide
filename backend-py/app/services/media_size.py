"""媒体尺寸换算：**档位 × 比例 → 像素**、比例反推、视频分辨率口径（``media-size`` 语义 ✓，自研重写 ✓）。

为什么单独一层：前端/客户端交上来的尺寸**不是像素**，而是 ``"2k"``、``"16:9"``、``"auto"``、
``"high"`` 这类**档位与比例**（画布、下拉框都是这么存的）。而 ComfyUI 的尺寸节点只吃
``"2048x1152"`` ⇒ 直接把 ``"2k"`` 交给 :func:`app.services.comfyui_bindings.parse_size`
会得到 ``None``（**静默变成"没指定尺寸"** ✗，图按默认尺寸出、不报错 ✗）。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``（MIT ✓）
的 ``infinite-canvas/web/src/lib/media-size.ts`` ⇒ 只搬**表格与换算规则**，代码自己写 ✗。

## 四处「能算出数、但数不对」的地方

1. ⭐ **``auto`` 有两种含义** ✓：``compute_media_size("auto", "16:9")`` 返回的是**比例串** ``"16:9"``
   （交给模型自己按比例定），而 ``compute_media_size("1k", "auto")`` 返回 ``"auto"``（连比例都不定）
   —— 两者都**不是像素** ✗，调用方必须能接住「拿到的仍不是像素」这件事 ✓。
2. ⭐ **``infer_media_ratio`` 是"就近取"，不是"精确匹配"** ✓：``2048x1152`` 要判成 ``16:9``
   （2048/1152 = 1.777… vs 1.7778 ✓），而 ``1360x2048`` 要判成 ``2:3``（0.664 vs 0.667 ✓）。
   用整数相等去判会一个都命中不了 ✗。
3. ⭐ **视频长边是"短边 = 分辨率"** ✓：``720`` + ``9:16`` ⇒ ``720x1280``（竖屏 720 是**宽**）；
   而 ``720`` + ``16:9`` ⇒ ``1280x720``（横屏 720 是**高**）✓ —— 一律把 720 当宽会出竖屏方图 ✗。
4. ⭐ **视频尺寸必须**偶数** ✓（``even_round``，H.264 的 yuv420 对奇数尺寸要么报错要么偷偷裁 1 px ✗）；
   而图片尺寸**不做**这一步 —— 图片的偶数化在 :mod:`comfyui_bindings` 的 ``snap16/snap32`` 里 ✓。

未搬：``formatElapsedDuration`` 已搬（纯格式化，审计页要用）；``image-thumbnail`` /
``image-utils`` 属前端渲染细节，本仓前端另有实现，不重复 ✗。

运行::

    from app.services.media_size import compute_media_size, resolve_size
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "IMAGE_SIZE_PRESETS",
    "MEDIA_RATIO_OPTIONS",
    "MEDIA_SCALE_OPTIONS",
    "VIDEO_SECONDS_MAX",
    "VIDEO_SECONDS_MIN",
    "clamp_video_seconds",
    "compute_media_size",
    "compute_video_size",
    "format_elapsed_duration",
    "infer_media_ratio",
    "infer_media_scale",
    "infer_video_ratio",
    "normalize_media_scale",
    "normalize_pixel_size",
    "parse_aspect_ratio",
    "parse_pixel_size",
    "parse_video_resolution",
    "read_media_dimensions",
    "read_video_dimensions",
    "resolve_size",
]

#: 档位（``auto`` 表示"让模型按比例自己定" ✓）
MEDIA_SCALE_OPTIONS = ("1k", "2k", "4k", "auto")

#: 图片可选比例（``auto`` 是占位，不含宽高 ✓）
MEDIA_RATIO_OPTIONS: tuple[dict[str, Any], ...] = (
    {"value": "1:1", "width": 1, "height": 1},
    {"value": "2:3", "width": 2, "height": 3},
    {"value": "3:2", "width": 3, "height": 2},
    {"value": "4:3", "width": 4, "height": 3},
    {"value": "3:4", "width": 3, "height": 4},
    {"value": "16:9", "width": 16, "height": 9},
    {"value": "9:16", "width": 9, "height": 16},
    {"value": "21:9", "width": 21, "height": 9},
    {"value": "9:21", "width": 9, "height": 21},
    {"value": "auto", "width": 0, "height": 0},
)

#: 档位 × 比例 → 像素 ✓（表就是口径，别现场算：21:9 的 2688x1152 算不出来，是**定的** ✓）
IMAGE_SIZE_PRESETS: dict[str, dict[str, str]] = {
    "1k": {
        "1:1": "1024x1024",
        "2:3": "1024x1536",
        "3:2": "1536x1024",
        "4:3": "1024x768",
        "3:4": "768x1024",
        "16:9": "1536x864",
        "9:16": "864x1536",
        "21:9": "2016x864",
        "9:21": "864x2016",
    },
    "2k": {
        "1:1": "2048x2048",
        "2:3": "1360x2048",
        "3:2": "2048x1360",
        "4:3": "2048x1536",
        "3:4": "1536x2048",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
        "21:9": "2688x1152",
        "9:21": "1152x2688",
    },
    "4k": {
        "1:1": "2880x2880",
        "2:3": "2336x3520",
        "3:2": "3520x2336",
        "4:3": "3312x2480",
        "3:4": "2480x3312",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "21:9": "3840x1648",
        "9:21": "1648x3840",
    },
}

#: 视频比例（比图片少 ``2:3``/``3:2`` ✓ —— 视频没有竖版 2:3 这个档）
VIDEO_RATIO_OPTIONS: tuple[dict[str, Any], ...] = (
    {"value": "1:1", "width": 1, "height": 1},
    {"value": "3:4", "width": 3, "height": 4},
    {"value": "4:3", "width": 4, "height": 3},
    {"value": "16:9", "width": 16, "height": 9},
    {"value": "9:16", "width": 9, "height": 16},
    {"value": "21:9", "width": 21, "height": 9},
    {"value": "auto", "width": 0, "height": 0},
)

VIDEO_SECONDS_MIN = 2
VIDEO_SECONDS_MAX = 30

_PIXEL_RE = re.compile(r"^(\d+)\s*[xX×]\s*(\d+)$")
_ASPECT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$")


def _js_round(value: float) -> int:
    """JS ``Math.round`` 口径（**.5 一律向上** ✓）—— Python 的 ``round`` 是银行家舍入 ✗。"""
    return math.floor(value + 0.5)


def _even_round(value: float) -> int:
    """视频尺寸偶数化 ✓（奇数尺寸在 H.264 上会被偷偷裁掉 1 px ✗）。"""
    return max(2, _js_round(value / 2) * 2)


def parse_pixel_size(value: Any) -> dict[str, int] | None:
    """``"1024x1536"`` → ``{"width":1024,"height":1536}`` ✓（认不出 ``None`` ✓）。"""
    match = _PIXEL_RE.match(str(value or "").strip())
    if not match:
        return None
    return {"width": int(match.group(1)), "height": int(match.group(2))}


def parse_aspect_ratio(value: Any) -> dict[str, float] | None:
    """``"16:9"`` → ``{"width":16.0,"height":9.0}`` ✓（``0`` 当分母 ⇒ ``None`` ✓）。"""
    match = _ASPECT_RE.match(str(value or "").strip())
    if not match:
        return None
    width, height = float(match.group(1)), float(match.group(2))
    if not width or not height:
        return None
    return {"width": width, "height": height}


def normalize_media_scale(value: Any) -> str:
    """档位归一 ✓：``"2048"``→``2k``、``"3840"``→``4k``、认不出 → ``1k``（**默认不是 auto** ✓）。"""
    scale = str(value or "").strip().lower()
    if scale in ("2k", "2048"):
        return "2k"
    if scale in ("4k", "3840"):
        return "4k"
    if scale == "auto":
        return "auto"
    if scale in MEDIA_SCALE_OPTIONS:
        return scale
    return "1k"


def infer_media_scale(size: Any, stored_scale: Any = None) -> str:
    """从像素反推档位 ✓：命中预设表就按表；否则按长边 **≥3072→4k / ≥1536→2k / 其余 1k** ✓。"""
    if stored_scale:
        return normalize_media_scale(stored_scale)
    text = str(size or "").strip()
    if not text or text == "auto":
        return "auto"
    if parse_aspect_ratio(text):
        return "auto"
    pixels = parse_pixel_size(text)
    if not pixels:
        return "auto"
    wanted = f"{pixels['width']}x{pixels['height']}"
    for scale, table in IMAGE_SIZE_PRESETS.items():
        if wanted in table.values():
            return scale
    long_side = max(pixels["width"], pixels["height"])
    if long_side >= 3072:
        return "4k"
    if long_side >= 1536:
        return "2k"
    return "1k"


def _nearest_ratio(target: float, options: tuple[dict[str, Any], ...], fallback: str) -> str:
    """就近取比例 ✓（**不是精确相等** ✗：2048/1152 与 16/9 差 0.0002，判等会一个都不中）。"""
    best = fallback
    for item in options:
        if item["value"] == "auto":
            continue
        current = item["width"] / item["height"]
        best_option = next((option for option in options if option["value"] == best), None)
        best_ratio = (best_option["width"] if best_option else 1) / max(1, (best_option["height"] if best_option else 1))
        if abs(current - target) < abs(best_ratio - target):
            best = item["value"]
    return best


def infer_media_ratio(size: Any, fallback: str = "1:1") -> str:
    """像素/比例 → 最接近的图片比例 ✓（认不出回 ``fallback`` ✓）。"""
    text = str(size or "").strip()
    if not text or text == "auto":
        return "auto"
    if any(item["value"] == text for item in MEDIA_RATIO_OPTIONS):
        return text
    pixels = parse_pixel_size(text) or parse_aspect_ratio(text)
    if not pixels:
        return fallback
    return _nearest_ratio(pixels["width"] / pixels["height"], MEDIA_RATIO_OPTIONS, fallback)


def infer_video_ratio(size: Any) -> str:
    """像素/比例 → 最接近的视频比例 ✓（认不出回 ``16:9`` ✓）。"""
    text = str(size or "").strip()
    if not text or text == "auto":
        return "auto"
    if any(item["value"] == text for item in VIDEO_RATIO_OPTIONS):
        return text
    pixels = parse_pixel_size(text) or parse_aspect_ratio(text)
    if not pixels:
        return "16:9"
    return _nearest_ratio(pixels["width"] / pixels["height"], VIDEO_RATIO_OPTIONS, "16:9")


def compute_media_size(scale: Any, ratio: Any) -> str | None:
    """档位 × 比例 → 像素 ✓。

    ⭐ 返回值**不一定是像素**：``("auto","16:9")`` → ``"16:9"`` ✓、``(任意,"auto")`` → ``"auto"`` ✓。
    调用方必须接着做"能不能解析成像素"的判断 ✗（直接 int() 会炸 ✗）。
    """
    text = str(ratio or "").strip()
    if not text or text == "auto":
        return "auto"
    normalized = normalize_media_scale(scale)
    if normalized == "auto":
        return text
    return IMAGE_SIZE_PRESETS.get(normalized, {}).get(text)


def normalize_pixel_size(value: Any, scale: Any = "1k") -> str | None:
    """把 ``"2k"``/``"16:9"``/``"1024x1024"`` 一律归到像素串 ✓；实在归不了 → ``None`` ✓。

    ⚠️ ``"2k"`` 单独给（不给比例）**归不出来** ⇒ ``None`` ✓ —— 与上游一致：
    档位必须配一个比例才有像素（默认 1:1 是**调用方**该补的默认，不是这里的 ✗）。
    """
    if not value:
        return None
    if parse_pixel_size(value):
        return str(value).strip()
    if parse_aspect_ratio(value):
        return compute_media_size(scale, value)
    return None


def read_media_dimensions(scale: Any, ratio: Any, size: Any = None) -> dict[str, int]:
    """取图片的像素：先看 ``size``，否则按 ``scale × ratio`` 算 ✓（都拿不到 ⇒ ``0x0`` ✓）。"""
    pixels = parse_pixel_size(size)
    if pixels:
        return pixels
    computed = compute_media_size("1k" if normalize_media_scale(scale) == "auto" else scale, "1:1" if str(ratio or "") in ("", "auto") else ratio)
    return parse_pixel_size(computed) or {"width": 0, "height": 0}


def parse_video_resolution(value: Any) -> str:
    """视频分辨率口径 ✓：``low``→480、``auto/high/medium``→720、``"1080p"``→``"1080"`` ✓。"""
    raw = str(value or "").strip().lower()
    if raw == "low":
        return "480"
    if raw in ("auto", "high", "medium"):
        return "720"
    number = raw[:-1] if raw.endswith("p") else raw
    return number if number.isdigit() and int(number) > 0 else "720"


def compute_video_size(resolution: Any, ratio: Any) -> str | None:
    """视频尺寸 ✓：**短边 = 分辨率**，另一边按比例算并**偶数化** ✓。

    ``("720","9:16")`` → ``"720x1280"``（竖屏 720 是宽 ✓）；``("720","16:9")`` → ``"1280x720"`` ✓。
    """
    text = str(ratio or "").strip()
    if not text or text == "auto":
        return "auto"
    parsed = parse_aspect_ratio(text)
    if not parsed:
        return "auto"
    short_side = max(1, int(parse_video_resolution(resolution) or 720))
    landscape = parsed["width"] >= parsed["height"]
    width = _even_round(short_side * parsed["width"] / parsed["height"]) if landscape else _even_round(short_side)
    height = _even_round(short_side) if landscape else _even_round(short_side * parsed["height"] / parsed["width"])
    return f"{width}x{height}"


def read_video_dimensions(size: Any, resolution: Any, ratio: Any) -> dict[str, int]:
    """取视频像素：先看 ``size``，否则按 ``resolution × ratio`` 算 ✓。"""
    pixels = parse_pixel_size(size)
    if pixels:
        return pixels
    computed = compute_video_size(resolution, "16:9" if str(ratio or "") in ("", "auto") else ratio)
    return parse_pixel_size(computed) or {"width": 0, "height": 0}


def clamp_video_seconds(value: Any) -> str:
    """秒数夹在 **2–30** ✓（空值默认 6 秒 ✓，返回**字符串** ✓ —— 输入框要的是字符串）。"""
    try:
        seconds = math.floor(float(value))
    except (TypeError, ValueError):
        seconds = 6
    return str(max(VIDEO_SECONDS_MIN, min(VIDEO_SECONDS_MAX, seconds)))


def format_elapsed_duration(ms: Any) -> str:
    """耗时人话 ✓：``1500`` → ``"1.5s"``、``65000`` → ``"1m 05s"`` ✓。"""
    try:
        total_seconds = max(0.0, float(ms)) / 1000
    except (TypeError, ValueError):
        total_seconds = 0.0
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes = int(total_seconds // 60)
    seconds = _js_round(total_seconds % 60)
    return f"{minutes}m {seconds:02d}s"


def resolve_size(value: Any, *, scale: Any = None, ratio: Any = None) -> str | None:
    """把客户端交来的尺寸**归到像素串** ✓（给 :func:`comfyui_bindings.parse_size` 用）。

    认得的写法：``"2048x1152"`` ✓、``"16:9"``（配 ``scale``，默认 1k ✓）、
    ``"2k"``（配 ``ratio``，默认 1:1 ✓）、``"2k"+"16:9"`` 由调用方拆开给 ✓。
    实在归不出 ⇒ ``None``（调用方据此判"没指定尺寸" ✓，**不许瞎猜一个** ✗）。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if parse_pixel_size(text):
        return text
    if parse_aspect_ratio(text):
        return compute_media_size(scale or "1k", text)
    lowered = text.lower()
    normalized = normalize_media_scale(text)
    if normalized == "auto":
        return compute_media_size("auto", ratio or "auto")
    # ⚠️ 只认**真档位**（含 "2048"/"3840" 两种写法）⇒ 乱字符串一律 None ✓
    # （`normalize_media_scale` 对认不出的值回 "1k"，直接拿它算会把 "abc" 静默变成 1024x1024 ✗）
    if lowered in MEDIA_SCALE_OPTIONS or lowered in ("2048", "3840"):
        return compute_media_size(normalized, ratio or "1:1")
    return None
