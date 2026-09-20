"""生成用的**几何数学**（零依赖，纯 Python ✓）—— 分辨率 / 帧数 / 潜空间形状。

**为什么这层重要**：模型的输入尺寸/帧数必须落在它的「合法网格」上，
差一点就是**形状断言失败或画面抖动** ✗，而这些换算**跟 GPU 无关** ⇒ 可以脱离权重与显卡先钉死 ✓。

## 事实来源（不猜 ✗）

* **帧数网格**：H3 核心节点的 schema（``io.Int.Input("length", min=5, step=17,
  tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid"`` ✓）⇒ ``24fps + 17k+5`` ✓。
* **分辨率步长**：同 schema 的 ``width/height`` 都是 ``step=32`` ✓。
* **分辨率↔像素预算的换算**：用参考项目 README 的**实测输出**反推并验证 ✓ ——
  ``0.65MP @16:9 ⇒ 1088×608``、``0.98MP @16:9 ⇒ 1344×768``（见 ``tests/engine_core_test.py`` ✓）。

## ⚠️ 明确**不猜**的部分

**潜空间的时间压缩比**（视频 VAE 把 N 帧压成几帧）**取决于具体权重配置** ✗，
本仓库的 ``configs/models.json`` 里没有这个字段 ⇒ 本模块**要求显式传入** ``temporal_compression`` ✓，
**不给默认值**（给个"看起来合理"的 4 或 8 只会在真机上错得莫名其妙 ✗）。
"""
from __future__ import annotations

import math

__all__ = [
    "AUDIO_LATENT_CHANNELS",
    "AUDIO_LATENT_HZ",
    "H3_FPS",
    "H3_FRAME_GRID",
    "H3_MIN_FRAMES",
    "audio_latent_frames",
    "ceil_to_multiple",
    "latent_frames",
    "megapixels_for_size",
    "parse_ratio",
    "size_for_megapixels",
    "snap_frames",
    "snap_to_multiple",
]

#: H3 的帧网格事实（来自核心节点 schema ✓）
H3_FPS = 24
H3_FRAME_GRID = 17
H3_MIN_FRAMES = 5
#: 音频潜空间的事实（来自 H3 模型的**文件头说明** ✓：「stereo audio (32ch, 40 Hz) latents」✓）
#: ⇒ 40 Hz 时间轴 ✓、**立体声 2 声道** ✓（通道数 32 是**特征维** ✓ 不是声道 ✓ —— 别混 ✗）
AUDIO_LATENT_HZ = 40
AUDIO_LATENT_CHANNELS = 2


def audio_latent_frames(seconds: float, *, mode: str, hz: int = AUDIO_LATENT_HZ) -> int:
    """秒数 → **音频潜帧数** ✓；⚠️ ``mode``（取整方式）**必须显式给** ✗。

    为什么**不给默认** ✓：音频潜帧数在参考实现里是**从调用方的张量读出来的**
    （``audio_t = audio_x.shape[-1]`` ✓）⇒ 「**怎么从秒数算出来**」这一步**我没核过** ✗。
    按本模块的纪律（见文件头 ✓：「给个看起来合理的数只会**在真机上错得莫名其妙**」✓）⇒
    **把没核过的地方逼成必填参数** ✓：``mode`` 取 ``"round" | "ceil" | "floor"`` ✓，
    调用方（或以后核到的事实）来定 ✓ —— 一旦核清，**只改这一处** ✓。

    ``hz`` 是**已核实的** ✓（40 Hz ✓）；``seconds`` 与视频那侧的
    :func:`snap_frames` **各走各的网格** ✓（不必整除 ✓）。
    """
    if mode not in ("round", "ceil", "floor"):
        raise ValueError(f"mode 必须是 round/ceil/floor 之一（收到 {mode!r} ✗）")
    value = max(0.0, float(seconds)) * int(hz)
    if mode == "ceil":
        return int(math.ceil(value))
    if mode == "floor":
        return int(math.floor(value))
    return int(round(value))
#: 分辨率步长（schema 的 ``step=32`` ✓）
RESOLUTION_MULTIPLE = 32


def snap_to_multiple(value: float, multiple: int = RESOLUTION_MULTIPLE) -> int:
    """四舍五入到 ``multiple`` 的整数倍（**下限 1 倍** ✓）。"""
    multiple = max(1, int(multiple))
    snapped = int(round(float(value) / multiple)) * multiple
    return max(multiple, snapped)


def ceil_to_multiple(value: float, multiple: int = RESOLUTION_MULTIPLE) -> int:
    """**向上**取到 ``multiple`` 的整数倍（像素预算用这个 ✓ 见 :func:`size_for_megapixels` ✓）。"""
    multiple = max(1, int(multiple))
    steps = -(-int(math.ceil(float(value))) // multiple)
    return max(multiple, steps * multiple)


def snap_frames(seconds: float, *, fps: int = H3_FPS, grid: int = H3_FRAME_GRID,
                min_frames: int = H3_MIN_FRAMES) -> int:
    """秒 → **帧数**，向上取到 ``grid*k + min_frames`` 网格 ✓（H3 的 17k+5 ✓）。

    校验（与参考模板默认值一致 ✓）：``5s @24fps`` ⇒ ``round(120)`` ⇒ 向上到 ``5+7×17=124`` ✓。
    """
    raw = max(int(min_frames), int(round(float(seconds or 0) * fps)))
    steps = -(-(raw - min_frames) // max(1, grid))  # 向上取整
    return min_frames + steps * max(1, grid)


def parse_ratio(ratio: str | float | tuple[float, float] | None) -> float:
    """画幅比 → **宽/高** 浮点（``"16:9"`` / ``1.777`` / ``(16, 9)`` 都收 ✓）。"""
    if ratio is None:
        return 16.0 / 9.0
    if isinstance(ratio, (int, float)):
        return float(ratio) if float(ratio) > 0 else 16.0 / 9.0
    if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
        width, height = float(ratio[0]), float(ratio[1])
        return width / height if height else 16.0 / 9.0
    text = str(ratio).strip().replace("：", ":")
    if ":" in text:
        width, _, height = text.partition(":")
        try:
            w, h = float(width), float(height)
            return w / h if h else 16.0 / 9.0
        except ValueError:
            return 16.0 / 9.0
    try:
        value = float(text)
        return value if value > 0 else 16.0 / 9.0
    except ValueError:
        return 16.0 / 9.0


def size_for_megapixels(megapixels: float, ratio: str | float | None = "16:9",
                        *, multiple: int = RESOLUTION_MULTIPLE) -> tuple[int, int]:
    """像素预算 + 画幅比 → ``(宽, 高)``（两边都对齐 ``multiple`` ✓）。

    算法（**由参考项目的实测输出反推并验证** ✓）—— 宽高**各自独立**从像素预算开方：

    1. ``宽 = ceil32(√(像素 × 宽高比))``
    2. ``高 = ceil32(√(像素 ÷ 宽高比))``

    ⚠️ 两点都是自检抓出来的（⑩⑪ ✓）：
    * 必须**向上**取整 —— 四舍五入会把 ``0.98MP`` 落成 ``1312×736`` ✗（参考实测是 ``1344×768`` ✓）；
    * 高**不能**用「像素 ÷ 宽」回算 —— 那会得到 ``736`` ✗（回算只保面积、不保画幅比 ✓）。
    校验两点：``0.65MP ⇒ 1088×608`` ✓、``0.98MP ⇒ 1344×768`` ✓（与参考 README 的实测输出一致 ✓）。
    """
    pixels = max(1.0, float(megapixels or 0) * 1_000_000.0)
    aspect = parse_ratio(ratio)
    width = ceil_to_multiple(math.sqrt(pixels * aspect), multiple)
    height = ceil_to_multiple(math.sqrt(pixels / aspect), multiple)
    return width, height


def megapixels_for_size(width: int, height: int) -> float:
    """``(宽, 高)`` → 百万像素（用于回显与成本估算 ✓）。"""
    return round(float(width) * float(height) / 1_000_000.0, 4)


def latent_frames(frames: int, *, temporal_compression: int) -> int:
    """帧数 → **潜空间帧数**；``temporal_compression`` **必须显式给** ✓（见模块头「不猜的部分」✓）。

    ``(frames - 1) // compression + 1`` 是视频 VAE 的通用对齐方式 ✓（首帧独立、后续压缩 ✓）。
    """
    compression = int(temporal_compression)
    if compression < 1:
        raise ValueError("temporal_compression 必须 >= 1 ✓")
    frames = max(1, int(frames))
    return (frames - 1) // compression + 1
