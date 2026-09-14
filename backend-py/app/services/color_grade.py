"""校色 —— 移植 ``backend/src/services/color-grade.ts``（**整域关闭**，含像素管线）。

原 TS 的 ``applyColorGrade`` 用 ``sharp``（libvips）做 8 步调整：

======  ==========================  ==========================================
步骤    操作                         性质 / 是否真生效
======  ==========================  ==========================================
1       ``recomb`` RGB 通道增益      逐通道线性 ✅ 生效
2       ``recomb`` 白平衡（色温/色调） 逐通道线性 ✅ 生效
3       ``modulate({brightness})``   **感知域（L 星）乘法** ✅ 生效
4       ``modulate({saturation})``   **跨通道**（LCh 彩度缩放）✅ 生效
5       ``linear(slope, 128*(1-slope))`` 逐通道线性，锁定中灰 128 ✅ 生效
6       ``gamma(2.2 - γ/100*1.2)``   ❌ **实测 no-op**（见下）
7       ``gamma(2.2 - s/100*0.7 + h/100*0.7)`` ❌ **实测 no-op**
8       ``recomb`` 肤色还原          逐通道线性 ✅ 生效
======  ==========================  ==========================================

⇒ 有效步骤全是「逐通道点式」+ 一个跨通道的饱和度 ⇒ 整条链 =
``RGB 乘法 LUT → eq=saturation → 对比度/肤色 LUT``，用 ffmpeg ``lutrgb`` 表达式实现，
**不需要 Pillow**（本仓有意保持 venv 依赖最小）。

**实测基准值**（``backend/probe-sharp.cjs`` 跑真实 sharp，用于校准与写测试）：

====================  ====================  =========================
参数                   输入                    输出
====================  ====================  =========================
校准 red +50           128,128,128           **192**,128,128
校准 red +50           200,100,50            255,100,50（钳位）
对比度 +50 ``linear``   128,128,128           128,128,128（锁中灰）
对比度 +50             200,200,200           **236**,236,236
对比度 +50             200,100,50            236,86,11（= v*1.5-64）
曝光 +50 ``modulate``   128,128,128           **199**（≈ L\* 乘 1.5，**不是** 192）
gamma 全部取值          任意                   **≈ 原值 ±1**（no-op）
====================  ====================  =========================

⚠️ 与 Node 的**已知差异（有意接受）**：

1. **不是逐字节一致**：编码器不同（libvips vs ffmpeg）⇒ 压缩细节必然不同；
2. ``saturation`` 是模型近似（sharp 在 LCh 缩彩度，ffmpeg ``eq`` 在 YUV 域）；
3. ``exposure`` 用 RGB 乘法近似 L\* 感知乘法 ⇒ 中灰处约差 7/255（128 时 192 vs 199，≈3%），
   高值处会**更早钳位**（但两者最终都到 255）；
4. LUT 在 8bit 量化两次（pre/post 各一次），与 sharp 的多轮 8bit 往返同量级。
5. 输出质量按扩展名定（jpg 走 ``-q:v 2``、png 无损、其余 ffmpeg 默认）；带 alpha 的 PNG 输入
   经 ``format=rgb24`` **丢掉 alpha**（与原 TS 的 ``removeAlpha()`` 同义）。

⚠️ ``n(v)`` 的判据是 JS ``typeof v === 'number' && Number.isFinite(v)``：
**字符串 ``"5"`` 会得到 0，而不是 5**（不做隐式转换）。照抄，别顺手 ``float()`` 一下。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Any

from .file_storage import get_absolute_path

EMPTY_COLOR_GRADE: dict[str, Any] = {
    "colorCalibration": {"red": 0, "green": 0, "blue": 0},
    "toneMapping": {"gamma": 0},
    "whiteBalance": {"temperature": 0, "tint": 0},
    "exposure": 0,
    "saturation": 0,
    "contrast": 0,
    "skinTone": 0,
    "shadowsHighlights": {"shadows": 0, "highlights": 0},
}


def _n(v: Any) -> float | int:
    """对齐 JS ``n(v)``：只有「有限数值」才采用，其余（含数字字符串、NaN、bool）一律 0。"""
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)) and math.isfinite(v):
        return v
    return 0


def _obj(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def normalize_color_grade(input_params: Any = None) -> dict[str, Any]:
    """规整校色参数：过滤非法值、补齐默认值。"""
    if not input_params or not isinstance(input_params, dict):
        return json.loads(json.dumps(EMPTY_COLOR_GRADE))  # 深拷贝

    cc = _obj(input_params.get("colorCalibration"))
    wb = _obj(input_params.get("whiteBalance"))
    sh = _obj(input_params.get("shadowsHighlights"))
    tm = _obj(input_params.get("toneMapping"))
    skin = _n(input_params.get("skinTone"))
    return {
        "colorCalibration": {"red": _n(cc.get("red")), "green": _n(cc.get("green")), "blue": _n(cc.get("blue"))},
        "toneMapping": {"gamma": _n(tm.get("gamma"))},
        "whiteBalance": {"temperature": _n(wb.get("temperature")), "tint": _n(wb.get("tint"))},
        "exposure": _n(input_params.get("exposure")),
        "saturation": _n(input_params.get("saturation")),
        "contrast": _n(input_params.get("contrast")),
        "skinTone": max(0, min(100, skin)),
        "shadowsHighlights": {"shadows": _n(sh.get("shadows")), "highlights": _n(sh.get("highlights"))},
    }


def has_color_grade(params: Any = None) -> bool:
    """是否有任何非中性的校色调整。"""
    p = normalize_color_grade(params)
    return bool(
        p["colorCalibration"]["red"] or p["colorCalibration"]["green"] or p["colorCalibration"]["blue"]
        or p["toneMapping"]["gamma"]
        or p["whiteBalance"]["temperature"] or p["whiteBalance"]["tint"]
        or p["exposure"] or p["saturation"] or p["contrast"] or p["skinTone"]
        or p["shadowsHighlights"]["shadows"] or p["shadowsHighlights"]["highlights"]
    )


def parse_color_grade(json_text: str | None) -> dict[str, Any] | None:
    """从 DB 字符串解析校色参数；无效或无调整时返回 None。"""
    if not json_text:
        return None
    try:
        normalized = normalize_color_grade(json.loads(json_text))
    except (ValueError, TypeError):
        return None
    return normalized if has_color_grade(normalized) else None


#: 通道顺序（lutrgb 的 c0/c1/c2 = r/g/b）
_CHANNELS = ("r", "g", "b")


def build_grade_plan(params: dict[str, Any]) -> tuple[dict[str, float], float, dict[str, float]]:
    """把有效步骤**合成**为 ``(前段逐通道乘子, 后段逐通道乘子, 对比度斜率)``。

    * 前段 = ①色彩校准 × ②白平衡 × ③曝光（都是「乘」⇒ 可交换合并）；
    * 后段 = ⑤对比度（线性，锁中灰 128）→ ⑧肤色（乘）。
    ④饱和度是**跨通道**的，不参与合成，单独用 ffmpeg ``eq`` 夹在两者之间。

    🔴 **⑥⑦两个 ``.gamma()`` 被有意跳过** —— 用真实 sharp 实测（``backend/probe-sharp.cjs``）：
    ``.gamma(1.6)`` / ``.gamma(2.8)`` / ``.gamma(1.5)`` / ``.gamma(2.9)`` 对 ``128,128,128``
    **全都输出 ``127,127,127``**、对 ``200,100,50`` 输出 ``199,9?,4?``（仅 ±1 取整差异）。
    原因：sharp/libvips 的 ``gamma`` 是**配合 resize 的编码 gamma**（pre-resize 降编码、
    post-resize 升编码），而本链**没有任何 resize** ⇒ 实际是 no-op。
    ⇒ 若照参数公式去 ``pow()``，反而会引入 Node **没有**的色调偏移（这是最容易照抄错的一处）。
    """
    cc = params["colorCalibration"]
    wb = params["whiteBalance"]
    temperature = wb["temperature"] / 100
    tint = wb["tint"] / 100
    brightness = 1 + params["exposure"] / 100
    pre = {
        "r": (1 + cc["red"] / 100) * (1 + temperature * 0.2 + tint * 0.1) * brightness,
        "g": (1 + cc["green"] / 100) * (1 - tint * 0.1) * brightness,
        "b": (1 + cc["blue"] / 100) * (1 - temperature * 0.2 + tint * 0.1) * brightness,
    }

    skin = params["skinTone"] / 100
    post = {"r": 1 + skin * 0.06, "g": 1 + skin * 0.02, "b": 1 - skin * 0.04}
    slope = 1 + params["contrast"] / 100
    return pre, post, slope


def _is_one(value: float) -> bool:
    return abs(value - 1) < 1e-9


def build_filter_chain(params: dict[str, Any]) -> str | None:
    """把参数编成 ffmpeg 滤镜链；**全中性时返回 ``None``**（连 ffmpeg 都不用起）。

    ⚠️ 表达式里的逗号必须转义成 ``\\,`` —— 滤镜图解析器按逗号切链，不转义会把
    ``clip(a\\,0\\,255)`` 截断（已实测：不转义直接报 filterchain 解析错）。
    """
    pre, post, slope = build_grade_plan(params)
    intercept = 128 * (1 - slope)

    parts = ["format=rgb24"]
    if not all(_is_one(pre[channel]) for channel in _CHANNELS):
        parts.append("lutrgb=" + ":".join(
            f"c{index}=clip(val*{pre[channel]:.6f}\\,0\\,255)"
            for index, channel in enumerate(_CHANNELS)))
    if params["saturation"]:
        parts.append(f"eq=saturation={1 + params['saturation'] / 100:.6f}")
    if (not _is_one(slope)) or not all(_is_one(post[channel]) for channel in _CHANNELS):
        parts.append("lutrgb=" + ":".join(
            f"c{index}=clip({post[channel]:.6f}*("
            f"clip({slope:.6f}*val+{intercept:.6f}\\,0\\,255))\\,0\\,255)"
            for index, channel in enumerate(_CHANNELS)))
    return None if len(parts) == 1 else ",".join(parts)


def _quality_args(extension: str) -> list[str]:
    """按扩展名给编码质量（png 无损、jpg/webp 高质量）。

    ⚠️ jpg 显式指定 ``-pix_fmt yuv420p``：libvips 出 JPEG 默认 **4:2:0**，而 ffmpeg 面对
    纯色图会自动挑 4:4:4 ⇒ 不指定的话同一张图在两侧的色度采样不同（虽不影响观感，
    但会让「同一参数在两边出不同文件」多一条无谓的口子）。
    """
    lowered = extension.lower()
    if lowered in (".jpg", ".jpeg"):
        return ["-q:v", "2", "-pix_fmt", "yuv420p"]
    if lowered == ".webp":
        return ["-q:v", "90"]
    return []


async def apply_color_grade_to_file(local_path: str, color_grade_json: str | None = None) -> str:
    """对**已落盘**的图片原地应用校色，返回路径（无调整时原样返回）。

    与 Node 的失败路径一致：ffmpeg 出错时**抛异常**（调用方记 ``color-grade-failed`` 并保留原图），
    失败时清掉临时文件，绝不留下半成品。
    """
    params = parse_color_grade(color_grade_json)
    if not params:
        return local_path

    absolute = get_absolute_path(local_path)
    chain = build_filter_chain(params)
    if chain is None:
        return local_path

    extension = os.path.splitext(absolute)[1]
    # 临时文件**必须保留扩展名**：ffmpeg 靠它推封装格式；也保证与源文件同目录（os.replace 原子）
    temp = f"{absolute}.grade{extension}"
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", "-i", absolute, "-vf", chain, "-frames:v", "1",
        *_quality_args(extension), "-y", temp,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await process.communicate()
    if process.returncode != 0 or not os.path.exists(temp):
        if os.path.exists(temp):
            os.remove(temp)
        raise RuntimeError(
            "校色失败（ffmpeg 非零退出）："
            + (stderr.decode("utf-8", errors="replace").strip()[:300] or "无错误输出")
        )
    os.replace(temp, absolute)
    return local_path
