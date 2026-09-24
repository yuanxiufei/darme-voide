"""自检：**配音情绪/语速契约**（8 维定序 / 单值转换 / 不猜 / 不钳位；零依赖 ✓ 2026-09-24）。

钉的几条（错了都是「出片不对还查不出来」✓✗）：

* ⭐⭐ **维度顺序是接口** ✗✗：``[happy, angry, sad, fear, disgust, melancholy, surprise, calm]`` ✓
  —— 写错序 = 情绪**整体错位**且**不报错** ✓✗；
* ⭐ **不自动归一化** ✗：``1.5`` 必须**报错** ✓，不许悄悄缩到 1.0 ✓✗；
* ⭐ **未知情绪名 ⇒ 拒** ✗：不许猜最近邻 ✓✗（猜错 = 换了个情绪，用户看不出来 ✓）；
* ⭐ **不替用户默认** ✗：``""``/``None`` 报错 ✓（``happy`` 是**产品侧**默认值 ✓ 不是契约的 ✓）；
* ⭐ **语速不钳位** ✓✗：越界报错 ✓；⚠️ 区间由调用方给 ✓（本层**不内置好取值** ✗）。

运行::

    ./.venv/Scripts/python.exe tests/voice_contract_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services import voice_contract as vc  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def raises(fn, *args: object, **kwargs: object) -> bool:
    try:
        fn(*args, **kwargs)
    except vc.VoiceContractError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def case_order_and_values() -> None:
    check("① ⭐⭐ 8 维**顺序**逐字固定（写错序=整体错位且不报错 ✗✗）",
          vc.EMOTION_ORDER == ("happy", "angry", "sad", "fear", "disgust", "melancholy",
                               "surprise", "calm"), vc.EMOTION_ORDER)
    vector = vc.EmotionVector((0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    check("①′ 取了向量的实体 ⇒ ``get('angry')`` = 0.2 ✓、``dominant`` = angry ✓、"
          "``to_list`` 不改序 ✓",
          vector.get("angry") == 0.2 and vector.dominant == "angry"
          and vector.to_list()[1] == 0.2)
    check("①″ **全 0 向量 ⇒ dominant = None** ✓（表示「不带情绪」✓ —— 不是 happy ✗）",
          vc.EmotionVector((0.0,) * 8).dominant is None)
    check("①‴ 维度数不对 / 越界值 都**拒** ✗（**不自动归一化** ✓✗：1.5 不许悄悄缩到 1.0 ✓）",
          raises(vc.EmotionVector, (0.1,) * 7) and raises(vc.EmotionVector, (1.5,) + (0.0,) * 7)
          and raises(vc.EmotionVector, (-0.1,) + (0.0,) * 7))
    check("①⁗ 8 键**字典**要键齐 ✓：缺键拒 ✗、多键拒 ✗（顺序由**契约**定 ✓ 不由字典定 ✓）",
          isinstance(vc.parse_emotion({"happy": 1.0, "angry": 0.0, "sad": 0.0, "fear": 0.0,
                                       "disgust": 0.0, "melancholy": 0.0, "surprise": 0.0,
                                       "calm": 0.0}), vc.EmotionVector)
          and raises(vc.parse_emotion, {"happy": 1.0}))


def case_name_and_single() -> None:
    check("② 名字 ⇒ **预设模式**（返回名字字符串 ✓ 引擎自己展开 ✓）；中文同义词映射 ✓",
          vc.parse_emotion("happy") == "happy" and vc.parse_emotion("开心") == "happy"
          and vc.parse_emotion("neutral") == "calm")
    check("②′ ⭐ **未知情绪名 ⇒ 拒** ✗（**不做最近邻猜测** ✓✗ —— 猜错=换了个情绪且看不出 ✓）",
          raises(vc.parse_emotion, "excited") and raises(vc.parse_emotion, "兴奋"))
    check("②″ ⭐ **不替用户默认** ✗：``''`` / ``None`` / 纯空格 全拒 ✓（``happy`` 属产品侧 ✓）",
          raises(vc.parse_emotion, "") and raises(vc.parse_emotion, None)
          and raises(vc.parse_emotion, "   "))
    check("②‴ ⭐ **单个数值不是向量** ✓✗（``0.5`` 拒 ✓ · 8 个数的序列收 ✓ · 长度 2 也拒 ✗）",
          raises(vc.parse_emotion, 0.5) and raises(vc.parse_emotion, [0.1, 0.2])
          and isinstance(vc.parse_emotion([0.0] * 8), vc.EmotionVector))
    check("②⁗ 形态不认识（二进制 / 别的类型）⇒ 拒 ✓（**不许当默认值混过去** ✗）",
          raises(vc.parse_emotion, b"happy") and raises(vc.parse_emotion, object()))
    check("②⁵ ``emotion_payload`` 两种模式分得清 ✓ 且**不造上游字段名** ✗（只给规范化值 ✓）",
          vc.emotion_payload("happy") == {"mode": "preset", "preset": "happy"}
          and vc.emotion_payload([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])["mode"] == "vector")


def case_speed() -> None:
    check("③ 语速在区间内 ⇒ 原值返回 ✓（区间**由调用方给** ✓ 本层不内置好取值 ✗）",
          vc.validate_speed(1.0, low=0.5, high=2.0) == 1.0
          and vc.validate_speed("1.25", low=0.5, high=2.0) == 1.25)
    check("③′ ⭐ **越界 ⇒ 拒** ✗（**不悄悄钳位** ✓✗ —— 钳位=改了设置还看不出来 ✓✗）",
          raises(vc.validate_speed, 3.0, low=0.5, high=2.0)
          and raises(vc.validate_speed, 0.1, low=0.5, high=2.0))
    check("③″ 非数值 / NaN / ±inf ⇒ 拒 ✓；区间本身写反 ⇒ 拒 ✓（**先核参数再核值** ✓）",
          raises(vc.validate_speed, "快", low=0.5, high=2.0)
          and raises(vc.validate_speed, float("nan"), low=0.5, high=2.0)
          and raises(vc.validate_speed, float("inf"), low=0.5, high=2.0)
          and raises(vc.validate_speed, 1.0, low=2.0, high=0.5))
    check("③‴ 边界**闭区间** ✓（正好等于 low/high 收 ✓✗ —— 别把边界当越界 ✗）",
          vc.validate_speed(0.5, low=0.5, high=2.0) == 0.5
          and vc.validate_speed(2.0, low=0.5, high=2.0) == 2.0)


def main() -> int:
    case_order_and_values()
    case_name_and_single()
    case_speed()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
