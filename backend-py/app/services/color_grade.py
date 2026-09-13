"""校色参数规整 —— 移植 ``backend/src/services/color-grade.ts`` 的**纯函数部分**。

只搬不依赖 sharp 的部分（``normalizeColorGrade`` / ``hasColorGrade`` / ``parseColorGrade``）；
``applyColorGrade``（真正对像素做校色）属图像管线，待图像域迁移时再做 —— 那时用 Pillow
或直接调外部 ffmpeg/Python 侧实现。

⚠️ ``n(v)`` 的判据是 JS ``typeof v === 'number' && Number.isFinite(v)``：
**字符串 ``"5"`` 会得到 0，而不是 5**（不做隐式转换）。照抄，别顺手 ``float()`` 一下。
"""

from __future__ import annotations

import json
import math
from typing import Any

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


class ColorGradeUnavailable(RuntimeError):
    """像素级校色尚未移植（原实现走 ``sharp``，Python 侧需要 Pillow）。"""


async def apply_color_grade_to_file(local_path: str, color_grade_json: str | None = None) -> str:
    """对**已落盘**的图片原地应用校色，返回路径。

    ⚠️ **像素管线未迁**（``applyColorGrade`` 走 sharp 的 RGB 增益 / gamma / 白平衡 /
    曝光 / 饱和度 / 对比度 / 肤色 / 阴影高光，共 142 行）。Pillow 与 sharp 的重采样与
    JPEG 编码**不会逐字节一致**，而校色结果是**会被持久化并被用户看到的资产** ⇒
    不适合在没有对照验证的情况下悄悄换实现，故留到独立一批（连同参考图压缩一起做）。

    **当前行为刻意与 Node 的「校色失败」完全一致**：

    * 参数为空/无效 → 原样返回（与 Node 相同，这条路径是正常的）；
    * 有参数 → 抛 :class:`ColorGradeUnavailable`，由调用方
      （``handle_image_complete``）的 ``try/except`` 记为 ``color-grade-failed`` 并
      **保留未校色的图** —— Node 侧校色抛错时也是这个结果。

    也就是说：升级前后**都不会写出「看起来不对」的资产**，只是暂时享受不到校色。
    """
    params = parse_color_grade(color_grade_json)
    if not params:
        return local_path
    raise ColorGradeUnavailable(
        "像素级校色尚未移植（需要 Pillow）；已跳过校色并保留原图。"
    )
