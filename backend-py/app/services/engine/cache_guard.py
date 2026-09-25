"""**缓存/权重文件的完整性守卫**（纯判定 ✓ 零依赖 ✓ 2026-09-24 补 ✓，口径来自逆向 ✓）。

## 它解决什么
混合加载会把「基底 + 覆盖层」**流式合并**成一个新的 safetensors 落盘当缓存 ✓。
坑在 ✗：**坏缓存**（合并中断 / 并发写坏 / 磁盘损坏 ✓）配上「源文件没变」的指纹 ✗ ⇒ 会被当成
有效缓存反复加载、反复崩 ✓✗（上游专门记了一条：买家反复遇到 ``invalid start byte 0xfe`` ✓）。
⇒ 上两道守卫 ✓：

1. ⭐ **产物自证** ✓：safetensors 的**头**必须自洽 ✓（见 :func:`is_sane_head` ✓）——
   截断 / 偏移与 dtype·形状不符 / 尾部多余或缺口 ⇒ **一律拒** ✗✗；
2. ⭐ **源指纹** ✓：`(大小, mtime, **头 64KB 哈希**)` 三元组 ✓ **且源数也要对上** ✓✗ ——
   ⚠️ **只有 size+mtime 是不够的** ✗✗：**保留时间戳的重新下载**会被误判成「没变」✓✗
   （上游 v3.5.5 的教训 ✓，故补头哈希 ✓）。

## 两条口径取舍（值得照做 ✓）
* 校验器**自己出错**（读不了 / 解析炸 ✓）⇒ 也是 **拒** ✗（**不抛** ✓ —— 校验器崩了等于没校验 ✓✗）；
* 写 sidecar **失败不阻塞** ✓：没有指纹 = 缓存失效 ⇒ **下次重新合并** ✓✗
  （退化成**慢**，不是退化成**错** ✓ —— 这条本仓同样适用 ✓）。

## 不猜（边界 ✓）
* 本模块**不判张量内容对不对** ✗（那要真权重比对 ✓）；它只保证「文件**结构上**是个完整的
  safetensors ✓，且**来源没变** ✓」；
* 头长度上限固定 **64 MB** ✓（上游口径 ✓ —— 真头部远小于此 ✓✗）；
* dtype 表**可被调用方扩充** ✓（本仓只列常见的 ✓；不认识 ⇒ **拒** ✗ 不放过 ✓✗）。
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

__all__ = ["DTYPE_SIZE", "FINGERPRINT_SAMPLE", "HEAD_LIMIT", "cache_sidecar_path",
           "file_fingerprint", "is_cache_valid", "is_sane_head"]

#: safetensors 头部的长度上限 ✓（上游口径 ✓）。
HEAD_LIMIT = 64 * 1024 * 1024
#: 指纹取**头多少字节**做哈希 ✓（上游口径 ✓：64 KiB ✓ —— 足够区分不同权重版本 ✓）。
FINGERPRINT_SAMPLE = 65536
#: 常见的 safetensors dtype → 每元素字节数 ✓（⚠️ 不认识 ⇒ **拒** ✗，别默认 1 或 0 ✗✗）。
DTYPE_SIZE: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def _elements(shape: Any) -> int | None:
    """形状 → 元素数 ✓（形状不合法 ⇒ ``None`` ✓）。"""
    if not isinstance(shape, list):
        return None
    total = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            return None
        total *= dim
    return total


def is_sane_head(data: bytes, *, dtype_size: Mapping[str, int] | None = None,
                 total_size: int | None = None) -> bool:
    """这段字节是不是**完整的 safetensors** ✓（8 条判据合起来 ⇒ 截断/偏移错/尾部不齐都拦下 ✓✗）。

    ⚠️ 任何解析异常都返回 ``False`` ✓（**不抛** ✗ —— 校验器自己炸了等于没校验 ✓✗）。

    ``total_size``：**整个文件**的字节数 ✓（不给就用 ``len(data)`` ✓）。⭐ 为什么要这个口子 ✗：
    真权重动辄 19.5 GiB ✓ ⇒ 为了判「数据区恰好吃满」而把**整个文件读进内存**是荒唐的 ✓✗；
    现在只读「8 字节 + 头」✓、把文件长度单独告诉它 ✓（判据一字未改 ✓✗）。
    """
    table = dict(DTYPE_SIZE)
    if dtype_size:
        table.update({str(key): int(value) for key, value in dtype_size.items()})
    try:
        raw = bytes(data)
        if len(raw) < 8:
            return False
        count = struct.unpack("<Q", raw[:8])[0]
        if not 0 < count < HEAD_LIMIT:
            return False
        if len(raw) < 8 + count:
            return False                       # 头被截断 ✓✗
        header = json.loads(raw[8:8 + count].decode("utf-8", "strict"))
        if not isinstance(header, dict):
            return False
        whole = int(total_size) if total_size is not None else len(raw)
        data_size = whole - 8 - count
        max_end = 0
        spans: list[tuple[int, int]] = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, dict):
                return False
            offsets = tensor.get("data_offsets")
            dtype = tensor.get("dtype")
            elements = _elements(tensor.get("shape"))
            if (not isinstance(offsets, list) or len(offsets) != 2 or elements is None
                    or dtype not in table):
                return False
            start, end = offsets[0], offsets[1]
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, int) or not isinstance(end, int)):
                return False
            expected = table[str(dtype)] * elements
            if start < 0 or end < start or end - start != expected or end > data_size:
                return False               # ⭐ 偏移必须与 dtype·形状**自洽** ✓✗
            spans.append((start, end))
            max_end = max(max_end, end)
        if not (max_end == data_size and max_end > 0):
            return False                            # ⭐ 数据段必须**恰好吃满** ✓✗
        # ⚠️⚠️ 本仓这里**比上游严一格** ✗✗：上游只看「``max_end == data_size``」✓ ⇒
        #    两个张量区间**中间有洞**（如 [0,2] 与 [4,6] ✓）它照样放行 ✓✗ —— 而洞意味着
        #    偏移写错或写入有缺口 ✓✗（正是「坏缓存」的形态 ✓）。⇒ 追加一条：
        #    **区间必须从 0 起首尾相接、不重叠** ✓。
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                return False
            cursor = end
        return True
    except Exception:  # noqa: BLE001 —— 见模块头「校验器自己出错 ⇒ 也是拒」✗
        return False


def file_fingerprint(path: str | Path) -> tuple[int, float, str] | None:
    """文件指纹 ✓ = ``(大小, mtime, 头 64KB sha256 前 16 位)`` ✓；读不了 ⇒ ``None`` ✓。

    ⚠️ ⭐ **头哈希是必须的** ✗✗：只用 size+mtime 时，「重新下载了一个**保留时间戳**的同名文件」
    会被判成「没变」⇒ 加载旧缓存 ✓✗（上游 v3.5.5 的教训 ✓）。
    """
    try:
        target = Path(path)
        size = target.stat().st_size
        mtime = target.stat().st_mtime
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            digest.update(handle.read(min(FINGERPRINT_SAMPLE, size)))
        return int(size), float(mtime), digest.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def cache_sidecar_path(path: str | Path) -> Path:
    """缓存的 sidecar 路径 ✓（``<文件>.meta.json`` ✓ —— 上游口径 ✓）。"""
    return Path(f"{path}.meta.json")


def is_cache_valid(*, cache_bytes: bytes, sidecar: Any,
                   sources: Mapping[str, tuple[int, float, str] | None],
                   total_size: int | None = None) -> bool:
    """缓存有效吗 ✓ = **产物自证** ✓ + sidecar 里每个源的**三元组**都对 ✓ **且源数相等** ✓✗。

    ⚠️ 源数不等 = 失效 ✓（sidecar 里多出/少了源，说明合并时的输入集合变了 ✓✗）；
    ⚠️ 源文件读不到（指纹为 ``None`` ✓）⇒ 失效 ✓（宁可重合并 ✓ 不拿可疑缓存去跑 ✓）；
    ``total_size`` 透传给 :func:`is_sane_head` ✓（见那里的说明：大文件只读头 ✓）。
    """
    if not is_sane_head(cache_bytes, total_size=total_size):
        return False
    if not isinstance(sidecar, Mapping) or not sidecar:
        return False
    if len(sidecar) != len(sources):
        return False
    for name, recorded in sidecar.items():
        if not isinstance(recorded, Mapping):
            return False
        want = sources.get(str(name))
        if want is None:
            return False
        got = (int(recorded.get("size") or 0), float(recorded.get("mtime") or 0),
               str(recorded.get("hash") or ""))
        if got != (int(want[0]), float(want[1]), str(want[2])):
            return False
    return True
