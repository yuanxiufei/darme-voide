"""S7 自检：**缓存/权重的完整性守卫**（纯判定 ✓ 零依赖 ✓ 2026-09-24）。

钉的都是「**坏缓存会被当成好缓存反复加载**」那条路 ✗✗（上游为此专门记过一条真实故障 ✓）：

* ⭐ **头必须自洽** ✓：截断 / 头长荒谬 / JSON 坏 / 不是对象 / **偏移与 dtype·形状不符** /
  越界 / ⭐ **数据段没恰好吃满**（尾部多余或缺口 ✓）⇒ **一律拒** ✗✗；
* ⭐ **指纹必须有头哈希** ✗✗：只用 ``size+mtime`` 时，「重新下载一个**保留时间戳**的同名文件」
  会被判成「没变」⇒ 加载旧缓存 ✓✗（上游 v3.5.5 的教训 ✓）；
* ⭐ **源数也要对上** ✓：sidecar 里多出/少了源 ⇒ 失效 ✓；
* ⭐ **校验器自己出错 ⇒ 也是拒** ✗（**不抛** ✓ —— 崩了等于没校验 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_cache_guard_test.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import cache_guard as guard  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _st(header: Any, *, data: int = 0, drop_head: int = 0) -> bytes:
    """造一段 safetensors 字节 ✓（``drop_head`` 用来模拟**头被截断** ✗）。"""
    raw = json.dumps(header).encode("utf-8")
    if drop_head:
        raw = raw[:-drop_head]
    return struct.pack("<Q", len(raw)) + raw + b"\0" * data


GOOD_HEADER = {"w": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}


def case_sane() -> None:
    """① 头自洽 ✓（正向 + 七类坏法 ✓）。"""
    check("① 正常产物 ⇒ **True** ✓（头长合法 ✓、偏移恰好吃满 ✓）",
          guard.is_sane_head(_st(GOOD_HEADER, data=8)) is True)
    check("①′ 带 ``__metadata__`` ⇒ 仍然 **True** ✓（那个键**不是**张量 ✓ 要跳过 ✓✗）",
          guard.is_sane_head(_st({**GOOD_HEADER, "__metadata__": {"format": "x"}}, data=8)) is True)
    check("①″ 字节太短（连 8 字节头长都不够 ✓）/ 头长 = 0 / 头长 ≥ 64MB ⇒ 全拒 ✗",
          guard.is_sane_head(b"\x00" * 4) is False
          and guard.is_sane_head(struct.pack("<Q", 0) + b"{}") is False
          and guard.is_sane_head(struct.pack("<Q", guard.HEAD_LIMIT + 1) + b"x" * 16) is False)
    check("①‴ ⭐ **头被截断**（读不到声明的 ``n`` 字节 ✓）⇒ 拒 ✗✗（这正是「合并中断」的形态 ✓）",
          guard.is_sane_head(_st(GOOD_HEADER, data=8, drop_head=4)) is False)
    check("①⁴ 头 JSON 坏 / 头不是对象（是数组 ✓）⇒ 拒 ✗（**不抛** ✓ —— 校验器自己炸了等于没校验 ✓✗）",
          guard.is_sane_head(struct.pack("<Q", 5) + b"{bad}" + b"\0" * 8) is False
          and guard.is_sane_head(struct.pack("<Q", 2) + b"[]" + b"\0" * 8) is False)
    # ⭐ ``total_size``（2026-09-24 补 ✓）：只读「8 字节 + 头」也能判「数据区恰好吃满」✓✗ ——
    #   ⚠️ 真权重 19.5 GiB ⇒ 为了判这一条把整个文件读进内存是荒唐的 ✓。判据**一字未改** ✓：
    #   给了总长就该与整读**同结论** ✓（否则两条路会给出不同答案 ✓✗）。
    full = _st(GOOD_HEADER, data=8)
    head_only = _st(GOOD_HEADER)                                 # 只读「头长 + 头」✓（没有数据区 ✓）
    check("①⁵ ⭐ 只读头 + 给 ``total_size`` ⇒ 与整读**同结论** ✓✗（同一判据，不是宽松版 ✓）",
          guard.is_sane_head(head_only, total_size=len(full)) is True
          and guard.is_sane_head(head_only, total_size=len(full) - 4) is False)
    short = _st(GOOD_HEADER, data=8, drop_head=4)
    check("①⁶ 头被截断时**给不给总长都拒** ✓（`total_size` 不是绕过口子 ✗✗）",
          guard.is_sane_head(short, total_size=len(full)) is False)
    check("①⁵ ``dtype`` 不认识 ⇒ 拒 ✗（**别默认 1 或 0** ✗✗）；``shape`` 不是列表 ⇒ 也拒 ✗",
          guard.is_sane_head(_st({"w": {"dtype": "F999", "shape": [1], "data_offsets": [0, 1]}},
                                 data=1)) is False
          and guard.is_sane_head(_st({"w": {"dtype": "F16", "shape": 1, "data_offsets": [0, 2]}},
                                     data=2)) is False)


def case_offsets() -> None:
    """② ⭐ 偏移自洽与「恰好吃满」✓（这两条是上游最核心的判据 ✓）。"""
    check("② ⭐ ``end - start`` 与 ``dtype_size × ∏shape`` **不符** ⇒ 拒 ✗✗（「偏移错」就靠这条 ✓）",
          guard.is_sane_head(_st({"w": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 6]}},
                                 data=8)) is False)
    check("②′ ``end > data_size``（越界 ✓）⇒ 拒 ✗",
          guard.is_sane_head(_st({"w": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
                                 data=4)) is False)
    check("②″ ⭐⭐ **尾部多余**（数据段没吃满 ✓）⇒ 拒 ✗✗；**缺口**（多个张量之间有洞 ✓）⇒ 也拒 ✗",
          guard.is_sane_head(_st(GOOD_HEADER, data=16)) is False
          and guard.is_sane_head(_st({"a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                                      "b": {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]}},
                                     data=6)) is False)
    check("②‴ 两个张量**首尾相接**铺满 ⇒ True ✓（正向对照：②″ 不是「多张量必假」的套套逻辑 ✓）",
          guard.is_sane_head(_st({"a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                                  "b": {"dtype": "F16", "shape": [2], "data_offsets": [2, 6]}},
                                 data=6)) is True)
    check("②⁴ 没有任何张量（``max_end == 0`` ✓）⇒ 拒 ✗（空产物不该当有效缓存 ✓✗）",
          guard.is_sane_head(_st({}, data=8)) is False
          or guard.is_sane_head(_st({"__metadata__": {}}, data=8)) is False)


def case_fingerprint() -> None:
    """③ ⭐ 指纹**必须有头哈希** ✗✗（只 size+mtime 会漏掉「保留时间戳的重新下载」✓✗）。"""
    with tempfile.TemporaryDirectory(prefix="cg_") as tmp:
        target = Path(tmp) / "weights.safetensors"
        target.write_bytes(_st(GOOD_HEADER, data=8))
        first = guard.file_fingerprint(target)
        check("③ 指纹 = ``(大小, mtime, 头哈希)`` ✓（三元组 ✓ 不是二元组 ✗）",
              isinstance(first, tuple) and len(first) == 3 and isinstance(first[2], str)
              and len(first[2]) == 16, first)
        before = target.stat()
        target.write_bytes(_st({**GOOD_HEADER, "extra": {"dtype": "F16", "shape": [1],
                                                         "data_offsets": [8, 10]}}, data=10))
        os.utime(target, (before.st_atime, before.st_mtime))     # ⭐ 故意**保留时间戳** ✓✗
        second = guard.file_fingerprint(target)
        check("③′ ⭐⭐ **保留 mtime** 的「重新下载」⇒ 头哈希仍然变了 ✓✗"
              "（只比 size+mtime 的实现**会漏掉这一整类** ✓✗ —— 上游 v3.5.5 的教训 ✓）",
              second is not None and second[2] != first[2], (first[2], second and second[2]))
        check("③″ 读不到的文件 ⇒ ``None`` ✓（**不抛** ✓）",
              guard.file_fingerprint(Path(tmp) / "没有这个文件") is None)
        check("③‴ sidecar 路径 = ``<缓存>.meta.json`` ✓（上游口径 ✓）",
              guard.cache_sidecar_path("a/b/c.safetensors").as_posix() == "a/b/c.safetensors.meta.json",
              guard.cache_sidecar_path("c.safetensors").as_posix())


def case_cache_valid() -> None:
    """④ 缓存有效性 ✓：三元组 + **源数**都要对上 ✓。"""
    blob = _st(GOOD_HEADER, data=8)
    good = {"base.safetensors": {"size": 10, "mtime": 1.5, "hash": "abc"},
            "overlay.safetensors": {"size": 20, "mtime": 2.5, "hash": "def"}}
    sources = {"base.safetensors": (10, 1.5, "abc"), "overlay.safetensors": (20, 2.5, "def")}
    check("④ 头健全 + 两个源三元组全对 + 源数相等 ⇒ **True** ✓",
          guard.is_cache_valid(cache_bytes=blob, sidecar=good, sources=sources) is True)
    check("④′ 任一源的三元组变了（大小/mtime/哈希任一 ✓）⇒ 失效 ✓（宁可重合并 ✓）",
          guard.is_cache_valid(cache_bytes=blob, sidecar=good,
                               sources={**sources, "base.safetensors": (10, 1.5, "CHANGED")}) is False
          and guard.is_cache_valid(cache_bytes=blob, sidecar=good,
                                   sources={**sources, "base.safetensors": (11, 1.5, "abc")}) is False)
    check("④″ ⭐ **源数不等** ⇒ 失效 ✓✗（sidecar 里多出源 / 少了源，都说明合并输入集合变了 ✓）",
          guard.is_cache_valid(cache_bytes=blob,
                               sidecar={**good, "第三份.safetensors": {"size": 1, "mtime": 1,
                                                                      "hash": "x"}},
                               sources=sources) is False
          and guard.is_cache_valid(cache_bytes=blob, sidecar={"base.safetensors": good["base.safetensors"]},
                                   sources=sources) is False)
    check("④‴ 源文件读不到（指纹 ``None`` ✓）⇒ 失效 ✓；sidecar 不是对象/为空 ⇒ 也失效 ✓",
          guard.is_cache_valid(cache_bytes=blob, sidecar=good,
                               sources={**sources, "overlay.safetensors": None}) is False
          and guard.is_cache_valid(cache_bytes=blob, sidecar=[], sources=sources) is False
          and guard.is_cache_valid(cache_bytes=blob, sidecar={}, sources={}) is False)
    check("④⁴ ⭐ 产物本身**不健全** ⇒ 就算指纹全对也**失效** ✗✗"
          "（「指纹对但产物坏」正是那条反复崩的形态 ✓）",
          guard.is_cache_valid(cache_bytes=_st(GOOD_HEADER, data=4), sidecar=good,
                               sources=sources) is False)


def main() -> int:
    case_sane()
    case_offsets()
    case_fingerprint()
    case_cache_valid()
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
