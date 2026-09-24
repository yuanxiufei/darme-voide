"""配音**情绪/语速契约**（2026-09-24 补 ✓，口径来自逆向 IndexTTS-2.5 的节点面 ✓）。

## 为什么要有这层
本仓角色的配音字段本来只有**一个单值** ``voice_emotion``（默认 ``"happy"`` ✓，见
``app/core/models.py`` ✓）✗。而配音引擎那边给的是**两种入口** ✓✗：

* **预设模式** ✓：给一个名字（如 ``happy`` ✓）；
* **8 维情绪向量** ✓：``[happy, angry, sad, fear, disgust, melancholy, surprise, calm]`` ✓
  —— ⚠️ **维度顺序本身是接口** ✗✗（写错序 = 情绪整体错位 ✓ 且不报错 ✓✗）。

⇒ 这一层只做三件事：**维度定序** ✓、**值域校验** ✓、**单值 ⇒ 向量的确定转换** ✓。

## 不猜（边界 ✓）
* ⚠️ **不内置「好取值」** ✗：语速上下限由调用方给 ✓（见 :func:`validate_speed` ✓ ——
  与 :mod:`app.services.engine.conditioning` 的 ``keep/fade`` 同一口径 ✓：好取值取决于引擎与权重 ✓）；
* ⚠️ **不造上游字段名** ✗：本模块只产出**规范化后的值** ✓（向量/名字/数值 ✓），
  请求体的键名以各引擎自己的契约为准 ✓（别在这里编 ✗）；
* ⚠️ **不自动归一化** ✗：给 ``1.5`` 就是**越界报错** ✓✗，不是悄悄缩放到 1.0 ✓
  （静默改用户意图 = 出片不对还查不出来 ✓✗）；
* ⚠️ 未知情绪名 ⇒ **拒** ✗ 不猜最近邻 ✓✗（猜错等于换了个情绪，且用户看不出来 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = ["EMOTION_ORDER", "EMOTION_PRESETS", "EmotionVector", "VoiceContractError",
           "emotion_payload", "parse_emotion", "validate_speed"]

#: 8 维情绪的**固定顺序** ✓✗ —— 逐字照抄引擎口径 ✓：写错顺序 = 情绪整体错位 ✓ **且不报错** ✗✗。
EMOTION_ORDER: tuple[str, ...] = ("happy", "angry", "sad", "fear", "disgust", "melancholy",
                                  "surprise", "calm")

#: 可用作**预设模式**的名字 ✓（同一批名字 ✓ —— 两种入口共用一套词表 ✓）。
EMOTION_PRESETS: tuple[str, ...] = EMOTION_ORDER

#: 兼容别名 ✓：口语/中文写法 → 协议名 ✓（⚠️ 只做**同义**映射 ✗，不做"猜最近邻" ✗）。
_ALIASES: dict[str, str] = {
    "happy": "happy", "开心": "happy", "高兴": "happy", "快乐": "happy",
    "angry": "angry", "愤怒": "angry", "生气": "angry",
    "sad": "sad", "悲伤": "sad", "难过": "sad",
    "fear": "fear", "恐惧": "fear", "害怕": "fear",
    "disgust": "disgust", "厌恶": "disgust",
    "melancholy": "melancholy", "忧郁": "melancholy", "低落": "melancholy",
    "surprise": "surprise", "惊讶": "surprise",
    "calm": "calm", "平静": "calm", "neutral": "calm",
}


class VoiceContractError(ValueError):
    """情绪/语速**给不出来或不合法** ✓ ⇒ 宁可当场拒 ✗（静默替默认值 = 出片情绪错还查不出 ✓✗）。"""


def _normalize_name(raw: Any) -> str:
    return str(raw or "").strip().lower().replace(" ", "")


@dataclass(frozen=True)
class EmotionVector:
    """**已校验**的 8 维情绪 ✓（顺序恒为 :data:`EMOTION_ORDER` ✓）。"""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(EMOTION_ORDER):
            raise VoiceContractError(
                f"情绪向量必须是 {len(EMOTION_ORDER)} 维" + f"（收到 {len(self.values)} 维 ✗）"
                f" —— 顺序固定为 {' / '.join(EMOTION_ORDER)} ✓✗（**少一维等于整体错位** ✗）")
        for name, value in zip(EMOTION_ORDER, self.values):
            if not (0.0 <= float(value) <= 1.0):
                raise VoiceContractError(
                    f"情绪 {name} = {value} 越界 ✗ ⇒ 每维必须在 [0, 1] ✓"
                    f"（**不自动归一化** ✗ —— 悄悄缩放到 1.0 等于改了你的意图 ✓✗）")

    def get(self, name: str) -> float:
        key = _normalize_name(name)
        if key not in EMOTION_ORDER:
            raise VoiceContractError(
                f"不知道情绪 {name!r} ✗ ⇒ 只能在 {' / '.join(EMOTION_ORDER)} 里选 ✓")
        return float(self.values[EMOTION_ORDER.index(key)])

    @property
    def dominant(self) -> str | None:
        """主导情绪名 ✓（**全 0 向量 ⇒ None** ✓ —— 表示"不带情绪"✓，不是 happy ✗）。"""
        best = max(range(len(EMOTION_ORDER)), key=lambda index: self.values[index])
        return None if float(self.values[best]) <= 0.0 else EMOTION_ORDER[best]

    def to_list(self) -> list[float]:
        return [float(value) for value in self.values]

    def to_dict(self) -> dict[str, Any]:
        return {"order": list(EMOTION_ORDER), "values": self.to_list(),
                "dominant": self.dominant}


def parse_emotion(value: Any) -> EmotionVector | str:
    """把**本仓的单值** ``voice_emotion`` ✓ 或一串数值 ✓ 转成契约形态 ✓。

    * 名字（``"happy"`` / ``"开心"`` ✓）⇒ 返回**名字** ✓（预设模式 ✓ —— 引擎自己会展开 ✓）；
    * 8 个数的序列 ⇒ 返回 :class:`EmotionVector` ✓；
    * 单个数 ✗ / 未知名字 ⇒ **拒** ✗（不猜最近邻 ✓✗）；
    * ⚠️ ``""`` / ``None`` ⇒ **拒** ✗ 不默认 ``happy`` ✓✗（默认值是产品侧的事 ✓，不是契约的 ✓）。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise VoiceContractError(
            "情绪没给 ✓ ⇒ 这一层**不替你默认** ✗（本仓模型字段的默认值 ``happy`` 属产品侧 ✓）；"
            "要么给名字、要么给 8 维向量 ✓")
    if isinstance(value, str):
        key = _normalize_name(value)
        if key in _ALIASES:
            return _ALIASES[key]
        raise VoiceContractError(
            f"不认识的情绪 {value!r} ✗ ⇒ 只能在 {' / '.join(EMOTION_PRESETS)} 里选 ✓"
            f"（**不做最近邻猜测** ✗ —— 猜错等于换了个情绪且你看不出来 ✓✗）")
    if isinstance(value, (bytes, bytearray)):
        raise VoiceContractError("情绪不能用二进制给 ✗（要么名字、要么 8 维数值 ✓）")
    if isinstance(value, Mapping):
        missing = [name for name in EMOTION_ORDER if name not in value]
        unknown = [key for key in value if str(key).lower() not in EMOTION_ORDER]
        if missing or unknown:
            raise VoiceContractError(
                f"情绪字典不完整 ✗：缺 {missing or '无'} ✓、多 {unknown or '无'} ✗"
                f" ⇒ 必须**恰好**给齐 8 维 {' / '.join(EMOTION_ORDER)} ✓✗")
        return EmotionVector(tuple(float(value[name]) for name in EMOTION_ORDER))
    if isinstance(value, Sequence):
        values = [float(item) for item in value]
        if len(values) == 1:
            raise VoiceContractError(
                f"收到单个数值 {values[0]} ✗ ⇒ 单值**不是**向量 ✓（想用预设请给名字 · "
                f"想用向量请给齐 {len(EMOTION_ORDER)} 维 ✓）")
        return EmotionVector(tuple(values))
    raise VoiceContractError(
        f"情绪形态不认识：{type(value).__name__} ✗ ⇒ 名字 / 8 维序列 / 8 键字典 三选一 ✓")


def validate_speed(speed: Any, *, low: float, high: float) -> float:
    """语速校验 ✓ —— ⚠️ **区间必须由调用方给** ✗（好取值取决于引擎 ✓ 本层不内置 ✗）。"""
    if not (float(low) < float(high)):
        raise VoiceContractError(f"语速区间不合法：low={low} ≥ high={high} ✗")
    try:
        value = float(speed)
    except (TypeError, ValueError) as error:
        raise VoiceContractError(f"语速必须是数值（收到 {speed!r} ✗）") from error
    if value != value or value in (float("inf"), float("-inf")):
        raise VoiceContractError(f"语速必须是有限数（收到 {speed!r} ✗）")
    if not (float(low) <= value <= float(high)):
        raise VoiceContractError(
            f"语速 {value} 不在 [{low}, {high}] 内 ✗ ⇒ **不悄悄钳位** ✓✗（钳位=改了你的设置还看不出来 ✓✗）")
    return value


def emotion_payload(value: Any) -> dict[str, Any]:
    """规范化结果 ✓ 给调用方按各自引擎的键名去拼请求体 ✓（⚠️ 本层**不造字段名** ✗）。"""
    parsed = parse_emotion(value)
    if isinstance(parsed, EmotionVector):
        return {"mode": "vector", "vector": parsed.to_list(), "dominant": parsed.dominant}
    return {"mode": "preset", "preset": parsed}
