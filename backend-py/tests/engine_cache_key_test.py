"""S7 自检：**参考素材指纹**（「哪一段可以跳过」的判据 ✓ 零依赖 ✓ 鸭子类型 ✓ 2026-09-24）。

钉的几条（错的指纹 = **白算一次** 或 **用错产物** ✓✗ 两个方向都贵 ✓）：

* ⭐ **顺序无关** ✓✗：字典 key 顺序变了 ⇒ 指纹**不许变**（否则「同样素材」被当成变了 ⇒ 白重算 ✓✗）；
* ⭐ **两种形态都要认** ✗✗：参考音频是 ``{waveform, sample_rate}`` ✓、其它是裸张量 ✓ ——
  上游就为此修过一版（``'dict' object has no attribute 'shape'`` ✓✗，与「同一份配置两种形态」同族 ✓）；
* ⭐ **采样率参与指纹** ✓（换了采样率的同名音频 ⇒ 就是不同素材 ✓✗）；
* ⭐ **出错 ⇒ 退化成「不相等」** ✓✗：没给 ``sampler`` / 取不出内容 ⇒ 带 ``h?`` 标记 ✓
  （**重算** ✓ 安全）—— 而不是**退化成通过** ✗（那会拿旧产物当新的 ✓✗）；
* ⭐ **空 vs 有** 要分得开 ✓；**形状变了**要分得开 ✓；**内容变了**要分得开 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_cache_key_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import cache_key as ck  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


class FakeTensor:
    """假张量 ✓（只满足最小协议：``shape`` ✓ + 可被 sampler 取值 ✓）。"""

    def __init__(self, shape: Any, payload: str = "data") -> None:
        self.shape = tuple(shape)
        self.payload = payload


def sampler(value: Any) -> bytes:
    """默认采样器 ✓：把内容变成字节 ✓（内容变了 ⇒ 指纹就该变 ✓）。"""
    return f"{getattr(value, 'payload', '')}".encode("utf-8")


def case_empty_and_shape() -> None:
    """① 空 / 有 要分得开 ✓；形状变了也要分得开 ✓。"""
    none = ck.refs_fingerprint({})
    check("① 没有任何参考 ⇒ 指纹固定 ✓ 且长度为上游口径 **32** ✓（⚠️ 与缓存守卫的 16 位无关 ✗）",
          isinstance(none, str) and len(none) == ck.FINGERPRINT_CHARS, (none, len(none)))
    check("①′ ⭐ **空 vs 有** 必须不同 ✓✗（混了就等于「加了素材却判定没变」✓✗）",
          none != ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}},
                                      sampler=sampler))
    one = ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}}, sampler=sampler)
    check("①″ **形状变了** ⇒ 指纹变 ✓；**内容变了** ⇒ 也变 ✓（sampler 生效 ✓ 不是摆设 ✗）",
          one != ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 16, 16))}},
                                     sampler=sampler)
          and one != ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8), "别的")}},
                                         sampler=sampler))


def case_order_and_forms() -> None:
    """② ⭐ 顺序无关 ✓✗ + 两种形态都认 ✓✗。"""
    first = ck.refs_fingerprint({"ref_images": {"a": FakeTensor((1, 3, 8, 8), "A"),
                                               "b": FakeTensor((1, 3, 8, 8), "B")}}, sampler=sampler)
    swapped = ck.refs_fingerprint({"ref_images": {"b": FakeTensor((1, 3, 8, 8), "B"),
                                                 "a": FakeTensor((1, 3, 8, 8), "A")}},
                                  sampler=sampler)
    check("② ⭐⭐ **字典顺序无关** ✓✗：key 顺序换了 ⇒ 指纹**不变**（按 key 排序 ✓）"
          "—— 否则「素材没动」会被判成变了 ⇒ 白重算一次 ✓✗",
          first == swapped, (first, swapped))
    audio = ck.refs_fingerprint({"ref_audios": {"0": {"waveform": FakeTensor((1, 2, 16000), "W"),
                                                     "sample_rate": 16000}}}, sampler=sampler)
    check("②′ ⭐⭐ **两种形态都要认** ✗✗：音频是 ``{waveform, sample_rate}`` ✓、其它是裸张量 ✓"
          "（上游就为此修过一版 ✓ —— 与「同一份配置两种形态」同族 ✓）",
          audio != ck.refs_fingerprint({"ref_audios": {"0": FakeTensor((1, 2, 16000), "W")}},
                                       sampler=sampler))
    check("②″ ⭐ **采样率参与指纹** ✓：波形一样但采样率不同 ⇒ 就是**不同素材** ✓✗",
          audio != ck.refs_fingerprint({"ref_audios": {"0": {
              "waveform": FakeTensor((1, 2, 16000), "W"), "sample_rate": 22050}}}, sampler=sampler))
    check("②‴ ``None`` 条目 / 非张量条目 也各有稳定形态 ✓（``None`` vs 别的 ≠ 相同 ✓）",
          ck.refs_fingerprint({"ref_images": {"0": None}}, sampler=sampler)
          != ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1,))}}, sampler=sampler)
          and ck.refs_fingerprint({"ref_images": {"0": "路径字符串"}}, sampler=sampler)
          != ck.refs_fingerprint({"ref_images": {"0": None}}, sampler=sampler))


def case_degrade() -> None:
    """③ ⭐ **出错/没采样器 ⇒ 退化成「不相等」** ✗（重算 ✓ 安全 ✓），不是退化成通过 ✗。"""
    without = ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}})
    with_sampler = ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}},
                                       sampler=sampler)
    check("③ 不给 ``sampler`` ⇒ 仍然给**稳定指纹** ✓（同一输入两次一致 ✓）且与给了的不同 ✓"
          "（⚠️ 内容不参与 ✓ —— 那些条目按上游口径退化成 ``h?`` ✓）",
          without == ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}})
          and without != with_sampler, (without, with_sampler))

    def boom(_value: Any) -> Any:
        raise RuntimeError("取不出内容 ✓")

    broken = ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}}, sampler=boom)
    check("③′ ⭐⭐ sampler **抛异常** ⇒ **不崩** ✓ 且指纹仍稳定 ✓（∅ 崩溃让人无从排查 ✓✗）"
          "、且与正常取内容的**不同** ✓⇒ 该段会被**重算** ✓✗",
          broken == ck.refs_fingerprint({"ref_images": {"0": FakeTensor((1, 3, 8, 8))}}, sampler=boom)
          and broken != with_sampler, broken)
    check("③″ ``refs`` 给成非映射（如列表 ✓）⇒ 当空处理 ✓（**别去猜它是什么** ✗）",
          ck.refs_fingerprint([1, 2, 3]) == ck.refs_fingerprint({}))
    check("③‴ 四类字段**顺序固定** ✓（``REF_FIELDS`` ✓ —— 换个字段名集合 ⇒ 换指纹 ✓）",
          ck.REF_FIELDS == ("ref_images", "ref_audios", "ref_videos", "ref_video_audios")
          and ck.FINGERPRINT_CHARS == 32 and ck.SAMPLE_POINTS == 512)


def main() -> int:
    case_empty_and_shape()
    case_order_and_forms()
    case_degrade()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
