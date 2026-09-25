"""**参考素材指纹**（决定「哪一段可以跳过不重算」✓ 纯逻辑 ✓ 零依赖 ✓ 2026-09-24 补 ✓，口径来自逆向 ✓）。

## 它解决什么
长视频是**逐段生成**的 ✓ ⇒ 必须能回答「这一段和上一次比，**参考素材变了没有**」✓✗。
变了的重算 ✓，没变的直接复用上次产物 ✓（省的是**分钟级**的时间 ✓✗）。

## 口径（逐条对应上游 ✓）
对四类参考（``ref_images`` / ``ref_audios`` / ``ref_videos`` / ``ref_video_audios`` ✓）各取一段指纹 ✓：

* 该类**为空** ⇒ 放 ``"0"`` 占位 ✓（⇒「没有素材」与「有素材」区分得开 ✓✗）；
* 有条目 ⇒ 先放**条数** ✓，再按 **key 排序**逐条 ✓（⇒ **字典顺序变了不算变** ✓✗）；
* 条目是 ``None`` ⇒ ``"k:None"`` ✓；
* ⭐ 条目是 **dict**（音频参考就是 ``{waveform, sample_rate}`` ✓）⇒ 用 ``waveform`` 的形状 +
  **``sample_rate``** + 内容哈希 ✓✗ —— ⚠️ 上游为此修过一版：**dict 与张量必须分开处理** ✗
  （否则 ``'dict' object has no attribute 'shape'`` ✓✗，与「同一份配置两种形态」同族 ✓）；
* 条目**不是**张量 ⇒ ``"k:other"`` ✓（**不静默当成张量** ✗）；
* 张量 ⇒ **形状 + 采样哈希** ✓ —— ⚠️ **不是全量哈希** ✗：至多取 **512 个采样点** ✓
  （快指纹 ✓，**理论上会撞** ✓✗ ⇒ 别把它当内容校验用 ✗）；
* 出错 ⇒ ``"?"`` / ``"h?"`` ✓（**退化成「不相等」⇒ 重算** ✓✗ —— **不是**退化成通过 ✓✓）。

最后 ``sha256("|".join(parts))[:32]`` ✓（⚠️ 长度 **32** 是上游口径 ✓；与本仓缓存守卫的 16 位
**是两回事** ✗ —— 用途不同 ✓ 别顺手统一 ✗）。

## 不猜（边界 ✓）
* ⚠️ **本模块不读张量内容** ✗：内容采样靠调用方给的 ``sampler`` ✓（本仓做 duck-typing ✓ ⇒
  自检不需要 torch ✓）；**没给 sampler ⇒ 内容不参与指纹** ✗，那些条目会带 ``h?`` 标记 ✓（见上 ✓）；
* ⚠️ 别拿它当**完整性校验** ✗（采样 512 点会撞 ✓）—— 那是 :mod:`app.services.engine.cache_guard` 的活 ✓。
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

__all__ = ["FINGERPRINT_CHARS", "SAMPLE_POINTS", "Sampler", "refs_fingerprint"]

#: 内容哈希**至多取多少个采样点** ✓（上游口径 ✓ —— 快指纹 ✓ 理论上会撞 ✓✗）。
SAMPLE_POINTS = 512
#: 最终指纹截断长度 ✓（上游口径 **32** ✓；⚠️ 与缓存守卫的 16 位无关 ✗）。
FINGERPRINT_CHARS = 32

#: 采样器 ✓：把一个「张量」变成可哈希的字节/浮点序列 ✓（本模块不 import torch ✓）。
Sampler = Callable[[Any], Any]

#: 四类参考的字段名 ✓（顺序固定 ✓ —— 顺序本身也进指纹 ✓✗）。
REF_FIELDS: tuple[str, ...] = ("ref_images", "ref_audios", "ref_videos", "ref_video_audios")


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _is_tensorish(value: Any) -> bool:
    return _shape_of(value) is not None


def _sample_digest(value: Any, sampler: Sampler) -> str:
    """取**内容采样哈希** ✓；取不到 ⇒ ``"h?"`` ✓（⇒ 指纹「不相等」⇒ **重算** ✓✗ 不崩 ✓）。"""
    digest = hashlib.sha256()
    try:
        data = sampler(value)
        if data is None:
            return "h?"
        if isinstance(data, (bytes, bytearray)):
            digest.update(bytes(data))
        else:
            flat = list(data)
            count = len(flat)
            step = max(1, count // SAMPLE_POINTS)
            digest.update(b"|".join(repr(float(item)).encode("ascii") for item in flat[::step]))
        return digest.hexdigest()[:20]
    except Exception:  # noqa: BLE001 —— 见模块头：出错 ⇒ 退化成「不相等」✗ 不崩 ✓✗
        return "h?"


def _entry_parts(key: str, value: Any, sampler: Sampler | None) -> list[str]:
    if value is None:
        return [f"{key}:None"]
    if isinstance(value, Mapping):
        waveform = value.get("waveform")
        if _is_tensorish(waveform):
            parts = [f"{key}:w{_shape_of(waveform)}", f"{key}:sr{value.get('sample_rate')}"]
            parts.append(f"{key}:h" + (_sample_digest(waveform, sampler) if sampler else "?"))
            return parts
        return [f"{key}:dict"]
    if isinstance(value, (str, bytes, bool, int, float)):
        # ⭐⭐ **本仓扩展** ✗✗：**标量/字符串要按值进指纹** ✓ —— 上游只看张量 ✓（它的参考素材是内存张量），
        #    而本仓**请求层**的参考是 **URL / 路径字符串** ✓ ⇒ 若按"非张量 ⇒ ``other``"处理，
        #    **改了 URL 指纹却不变** ✓✗（实测：``{"0": "a.png"}`` 与 ``{"0": "b.png"}`` **同指纹** ✓✗）
        #    ⇒ 会**误判成「素材没变」而跳过重生成** ✓✗✗（拿上一段的成片去交差，且看不出来 ✓）。
        #    ⚠️ 张量那条路**一字未改** ✗（判据仍在 ``_sample_digest`` ✓）。
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        return [f"{key}:v{len(raw)}:{hashlib.sha256(raw).hexdigest()[:16]}"]
    if not _is_tensorish(value):
        return [f"{key}:other"]
    parts = [f"{key}:{_shape_of(value)}"]
    parts.append(_sample_digest(value, sampler) if sampler else "h?")
    return parts


def refs_fingerprint(refs: Mapping[str, Any] | None = None, *, sampler: Sampler | None = None,
                     fields: Sequence[str] = REF_FIELDS) -> str:
    """四类参考素材 → **一个指纹** ✓（同素材 ⇒ 同指纹 ✓；变了 ⇒ 换指纹 ⇒ 该段重算 ✓✗）。

    ``refs`` 形如 ``{"ref_images": {"0": tensor, ...}, "ref_audios": {...}, ...}`` ✓
    （缺的键按**空**处理 ✓ ⇒ 放 ``"0"`` ✓）。⚠️ ``sampler`` 不给 ⇒ 内容不参与指纹 ✓（带 ``h?`` 标记 ✓）。
    """
    source = refs if isinstance(refs, Mapping) else {}
    parts: list[str] = []
    for field in fields:
        items = source.get(field)
        if not items:
            parts.append("0")
            continue
        if not isinstance(items, Mapping):
            parts.append("0")            # ⚠️ 形态不对 ⇒ 当空 ✓（别去猜它是什么 ✗）
            continue
        keys = sorted(str(key) for key in items)
        parts.append(str(len(keys)))
        for key in keys:
            parts.extend(_entry_parts(key, items.get(key), sampler))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]
