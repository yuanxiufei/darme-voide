"""混合加载**加载层**：``fl2va`` 基底 + ``ref2va`` 的 adaLN 覆盖层 → **一个能直接装的权重文件** ✓
（2026-09-24 补 ✓ —— 计划层 :mod:`hybrid_merge` 此前**没有任何调用方** ✗✗，判据立了却接不出去 ✓）。

## 三层分工（别混 ✗）
* :mod:`hybrid_merge`：**取哪些键 / 为什么** ✓（纯判定 ✓ 不碰字节 ✗）；
* **本模块**：**真搬字节** ✓ + **原子落盘** ✓ + **缓存复用判定** ✓ + **磁盘余量** ✓；
* :mod:`cache_guard`：**产物自证 + 源指纹** ✓（坏缓存 ⇒ 一律拒 ✗✗）。
* ⚠️ :mod:`cache_key` **不是**这条链的指纹 ✗✗ —— 那个是**参考素材**指纹（决定「哪段生成可以跳过」✓）；
  权重缓存用的是 :func:`cache_guard.file_fingerprint` 的 ``(大小, mtime, 头 64KB 哈希)`` ✓。

## 四条判据（都是「错了会静默」的形状 ✓✗）
1. ⭐⭐ **合并不改键集** ✗✗：产物键集 == **基底**键集（覆盖层只提供 adaLN 的**值** ✓）——
   落盘前**核一遍** ✓（少一个键 = 装出来少一层 ✓✗：装得上、跑得动、画面不对 ✓）；
2. ⭐⭐ **必须继承基底的内嵌元数据** ✗✗（``__metadata__`` ✓）—— ``TorchBackend.load_weights``
   **就是靠它读结构的** ✓；丢了它 ⇒ 合并出来的文件装不上（报「读不出结构」✗，而真因是合并时丢了它 ✓✗）；
3. ⭐ **复用要过两道** ✓：产物**结构自证** ✓ + sidecar 里每个源的 ``(大小, mtime, 头哈希)`` 都对 ✓
   **且源数相等** ✓✗（sidecar 缺了 / 源多了一个 ⇒ 失效 ✓ —— 宁可重合并 ✓✗）；
   ⚠️ 本层**不验张量内容** ✗（头自洽 ≠ 数据对 ✓，见 ``cache_guard`` 的边界 ✓）；
4. ⭐ **写 sidecar 失败不阻塞** ✓（退化成「下次重新合并」= 慢 ✓，**不是**退化成错 ✓✗），
   但报告里**必须看得见** ✗；磁盘余量**不足 ⇒ 拒** ✗ 且**一个字节都不写** ✗（先算尺寸再动手 ✓）。

## 与 torch 的关系
本模块**零依赖** ✓（不 import torch ✗）：产物就是一个正常 safetensors ✓ ⇒ 装的时候把路径
指过去即可（``load_module_weights(model, <合并产物>, device=...)`` ✓）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import cache_guard as cg
from . import hybrid_merge as hm
from . import safetensors as st

__all__ = ["HybridLoadError", "HybridLoadReport", "hybrid_cache_path", "plan_hybrid_load",
           "run_hybrid_load", "sha256_of_tensor"]

#: 搬运字节时的分块大小 ✓（别一次性读进内存 ✗ —— 真权重 19.5 GiB ✓）。
CHUNK_BYTES = 8 * 1024 * 1024
#: 写盘前要求的**余量系数** ✓（留 5% 给 sidecar / 文件系统元数据 ✓）。
DISK_HEADROOM = 1.05


class HybridLoadError(RuntimeError):
    """合并不可做 ✓（坏源 / 余量不足 / 键集对不上 ✓）—— 一律**当场拒** ✗ 不留半成品 ✓✗。"""


def _shapes(info: st.SafetensorsInfo) -> dict[str, tuple[int, ...]]:
    return {name: tuple(int(dim) for dim in entry.shape) for name, entry in info.tensors.items()}


def _read_sources(base: str | Path, overlay: str | Path) -> tuple[st.SafetensorsInfo, st.SafetensorsInfo]:
    """读两边头部 ✓（⚠️ **只读头** ✗ —— 合并前不需要把 19.5 GiB 读进来 ✓）。

    ⚠️ 坏源（不是 safetensors / 被截断 ✓）⇒ **当场拒** ✗ 并转成 :class:`HybridLoadError`
    （``st`` 自己抛的是 ``SafetensorsError`` ✓ —— 别让它漏成别的异常类型 ✓✗）。
    """
    infos: list[st.SafetensorsInfo] = []
    for label, path in (("基底", base), ("覆盖层", overlay)):
        try:
            info = st.inspect(path)
        except Exception as err:  # noqa: BLE001 —— 读不了/不是 safetensors ⇒ 拒 ✓ 不猜 ✓
            raise HybridLoadError(
                f"{label}读不了 ✗（{Path(path).name} ✓）：{type(err).__name__}: {err}"
                f" ⇒ **不拿坏文件去合并** ✓✗") from err
        if not info.ok:
            raise HybridLoadError(
                f"{label}的 safetensors 头有问题 ✗，**不拿坏文件去合并** ✗：{info.problems[:3]} ✓"
                f"（{Path(path).name} ✓）")
        infos.append(info)
    return infos[0], infos[1]


def hybrid_cache_path(base: str | Path, overlay: str | Path, *, directory: str | Path | None = None) -> Path:
    """默认缓存路径 ✓：``<目录>/<基底名>+<覆盖层名>.safetensors``（目录默认与基底同处 ✓）。

    ⚠️ 文件名带**两边**的名字 ✗：只按基底命名会让「换了覆盖层」时误用旧缓存 ✓✗
    （源指纹确实也会拦 ✓✗，但让文件名就说清是更省事的第二道 ✓）。
    """
    base_path, overlay_path = Path(base), Path(overlay)
    target_dir = Path(directory) if directory else base_path.parent
    return target_dir / f"{base_path.stem}+{overlay_path.stem}.safetensors"


@dataclass(frozen=True)
class HybridLoadReport:
    """一次合并/复用的结论 ✓（``reused=True`` 时 ``reason`` 说明**凭什么**复用 ✓）。"""

    cache_path: str
    reused: bool
    reason: str
    take: tuple[str, ...] = ()
    skip: tuple[tuple[str, str], ...] = ()
    key_count: int = 0
    bytes: int = 0
    bytes_from_overlay: int = 0
    sidecar_written: bool = False
    sidecar_error: str | None = None
    disk_free_bytes: int | None = None
    elapsed_ms: float = 0.0
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {"cachePath": self.cache_path, "reused": self.reused, "reason": self.reason,
                "take": list(self.take), "takeCount": len(self.take),
                "skip": [{"name": name, "reason": reason} for name, reason in self.skip],
                "skipCount": len(self.skip), "keyCount": self.key_count, "bytes": self.bytes,
                "bytesFromOverlay": self.bytes_from_overlay,
                "sidecarWritten": self.sidecar_written, "sidecarError": self.sidecar_error,
                "diskFreeBytes": self.disk_free_bytes, "elapsedMs": round(self.elapsed_ms, 3),
                "notes": list(self.notes)}


def plan_hybrid_load(base: str | Path, overlay: str | Path, *,
                     matcher: Callable[[str], bool] | None = None,
                     include_final_adaln: bool = False) -> dict[str, Any]:
    """**只做计划** ✓（不写盘 ✗）：取哪些键 / 跳哪些 + **产物会有多大** ✓。

    ⚠️ 三种「拒」都在这里就报 ✗（同族性 ✓ / 形状一致 ✓ / 覆盖层是坏文件 ✓）——
    别等到搬运了一半才发现 ✓✗。
    """
    base_info, overlay_info = _read_sources(base, overlay)
    plan = hm.plan_hybrid_merge(_shapes(base_info), _shapes(overlay_info), matcher=matcher,
                               include_final_adaln=include_final_adaln)
    overlay_bytes = sum(int(overlay_info.tensors[name].nbytes) for name in plan.take)
    return {
        **plan.to_dict(),
        "basePath": str(base), "overlayPath": str(overlay),
        "baseBytes": int(base_info.total_tensor_bytes),
        "mergedBytes": int(base_info.total_tensor_bytes),   # ⭐ 键集不变 ⇒ 数据区大小 == 基底 ✓✗
        "bytesFromOverlay": overlay_bytes,
        "inheritMetadataKeys": sorted(base_info.metadata),
        "suggestedCachePath": str(hybrid_cache_path(base, overlay)),
    }


def _head_bytes(path: Path) -> bytes:
    """只读 **8 字节头长 + 头** ✓（判产物自证用 ✓；真文件可能 19.5 GiB ⇒ 不能整读 ✗）。"""
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return struct.pack("<Q", length) + handle.read(length)


def _cache_usable(cache: Path, sources: Mapping[str, Any]) -> tuple[bool, str]:
    """产物**结构自证** ✓ + sidecar **源指纹** ✓ ⇒ 能不能直接复用 ✓。"""
    head = _head_bytes(cache)
    total = cache.stat().st_size
    if not cg.is_sane_head(head, total_size=total):
        return False, ("产物头部**不自洽** ✗（截断 / 偏移与 dtype·形状不符 / 数据区有洞 ✓）"
                       "⇒ 重合并 ✓✗（坏缓存会被当成有效缓存反复加载 ✓）")
    sidecar_path = cg.cache_sidecar_path(cache)
    sidecar: Any = None
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— sidecar 坏了 ⇒ 就是「没有指纹」✓ 不是崩 ✓
            sidecar = None
    if not cg.is_cache_valid(cache_bytes=head, total_size=total, sidecar=sidecar, sources=sources):
        return False, ("sidecar 对不上 ✗（源变了 / **源数不等** / sidecar 缺失或损坏 ✓）"
                       "⇒ 重合并 ✓✗ —— 宁可慢一次，不拿可疑缓存去跑 ✓")
    return True, "产物结构自证 ✓ + sidecar 源指纹一致 ✓ ⇒ 直接复用 ✓"


def _copy_range(handle: Any, start: int, length: int, out: Any, *, label: str) -> None:
    """从 ``handle`` 的 ``start`` 起搬 ``length`` 字节 ✓（分块 ✓ 不整读 ✗）。"""
    handle.seek(start)
    remaining = int(length)
    while remaining > 0:
        chunk = handle.read(min(CHUNK_BYTES, remaining))
        if not chunk:
            raise HybridLoadError(
                f"读 {label} 时提前到文件末尾 ✗ ⇒ 源文件在合并途中被改/被截断 ✓✗"
                f"（半成品**已删** ✓ 不留坏缓存 ✓）")
        out.write(chunk)
        remaining -= len(chunk)


def _write_merged(base_info: st.SafetensorsInfo, base_path: Path, overlay_info: st.SafetensorsInfo,
                  overlay_path: Path, plan: hm.MergePlan, out_path: Path, *,
                  metadata: Mapping[str, str] | None) -> tuple[int, int]:
    """按计划搬字节 ⇒ 原子落盘 ✓（``.tmp`` → ``os.replace`` ✓）⇒ ``(总字节, 来自覆盖层的字节)`` ✓。"""
    take = set(plan.take)
    # ⭐ 顺序**照基底** ✓（产物与基底逐键同序 ⇒ 便于比对 ✓），键集 = 基底键集 ✓✗
    order = [(name, overlay_info if name in take else base_info) for name in base_info.tensors]
    merged_keys = {name for name, _ in order}
    if merged_keys != set(base_info.tensors):
        raise HybridLoadError(
            f"合并后键集与基底不一致 ✗（多 {sorted(merged_keys - set(base_info.tensors))[:3]} ✓ / "
            f"少 {sorted(set(base_info.tensors) - merged_keys)[:3]} ✓）⇒ 产物会**少层或串键** ✗✗")
    table: dict[str, Any] = {}
    offset = 0
    for name, source in order:
        entry = source.tensors[name]
        table[name] = {"dtype": entry.dtype, "shape": list(entry.shape),
                       "data_offsets": [offset, offset + int(entry.nbytes)]}
        offset += int(entry.nbytes)
    # ⭐⭐ 基地的 ``__metadata__`` **必须继承** ✗✗（`load_weights` 靠它读结构 ✓），本层再补自己的键 ✓
    merged_metadata = {str(k): str(v) for k, v in base_info.metadata.items()}
    for key, value in (metadata or {}).items():
        merged_metadata[str(key)] = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, separators=(",", ":"))
    header = json.dumps({**table, "__metadata__": merged_metadata},
                        ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    overlay_bytes = 0
    try:
        with tmp.open("wb") as out, base_path.open("rb") as base_file, \
                overlay_path.open("rb") as overlay_file:
            out.write(struct.pack("<Q", len(header)))
            out.write(header)
            for name, source in order:
                entry = source.tensors[name]
                handle = overlay_file if name in take else base_file
                info = overlay_info if name in take else base_info
                # 数据区从「8 + 头长」之后开始 ✓（偏移是**相对数据区**的 ✓）
                _copy_range(handle, 8 + int(info.header_bytes) + int(entry.begin), int(entry.nbytes),
                            out, label=f"{Path(info.path).name}::{name}")
                if name in take:
                    overlay_bytes += int(entry.nbytes)
        os.replace(tmp, out_path)                 # ⭐ 原子 ✓：别人看不到半成品 ✓✗
    except Exception:
        try:
            tmp.unlink()                          # ⭐ 失败 ⇒ **删掉半成品** ✓✗
        except OSError:
            pass
        raise
    return offset, overlay_bytes


def run_hybrid_load(base: str | Path, overlay: str | Path, cache_path: str | Path | None = None, *,
                    force: bool = False, matcher: Callable[[str], bool] | None = None,
                    include_final_adaln: bool = False,
                    metadata: Mapping[str, Any] | None = None) -> HybridLoadReport:
    """合并（或复用缓存 ✓）⇒ :class:`HybridLoadReport` ✓。

    * ``cache_path`` 省略 ⇒ :func:`hybrid_cache_path` ✓；
    * ``force=True`` ⇒ **即使缓存有效也重合并** ✓（门禁统一支持 ``force`` ✓）；
    * ⭐ 余量不足 ⇒ **拒** ✗ 且**一个字节都不写** ✗（先算尺寸再动手 ✓）；
    * ⭐ sidecar 写失败 ⇒ **不阻塞** ✓（退化成「下次重合并」= 慢 ✓）但报告里带着 ``sidecarError`` ✓✗。
    """
    started = time.perf_counter()
    base_info, overlay_info = _read_sources(base, overlay)
    plan = hm.plan_hybrid_merge(_shapes(base_info), _shapes(overlay_info), matcher=matcher,
                               include_final_adaln=include_final_adaln)
    target = Path(cache_path) if cache_path else hybrid_cache_path(base, overlay)
    sources = {"base": cg.file_fingerprint(base), "overlay": cg.file_fingerprint(overlay)}
    if sources["base"] is None or sources["overlay"] is None:
        raise HybridLoadError("源权重读不到指纹 ✗（文件被删 / 权限不足 ✓）⇒ 不合并 ✓✗")

    if target.exists() and not force:
        usable, reason = _cache_usable(target, sources)
        if usable:
            return HybridLoadReport(
                cache_path=str(target), reused=True, reason=reason, take=plan.take, skip=plan.skip,
                key_count=len(base_info.tensors), bytes=int(target.stat().st_size),
                bytes_from_overlay=sum(int(overlay_info.tensors[n].nbytes) for n in plan.take),
                disk_free_bytes=_free_bytes(target),
                elapsed_ms=(time.perf_counter() - started) * 1000.0)
    elif target.exists():
        reason = "**force** ⇒ 即使缓存有效也重合并 ✓"
    else:
        reason = "没有产物 ⇒ 合并一次 ✓"

    needed = int(base_info.total_tensor_bytes) + 8 + 1024 * 1024     # 头留 1 MiB 余量 ✓
    free = _free_bytes(target)
    if free is not None and free < int(needed * DISK_HEADROOM):
        raise HybridLoadError(
            f"磁盘余量不足 ✗：需要 ≈{_gib(needed)}（含 {int((DISK_HEADROOM - 1) * 100)}% 余量 ✓）、"
            f"可用 {_gib(free)} ✓ ⇒ **一个字节都没写** ✗（先清盘再试 ✓）")

    merged_metadata = {
        "h3_hybrid": json.dumps({
            "base": Path(base).name, "overlay": Path(overlay).name,
            "takeCount": len(plan.take), "skipCount": len(plan.skip),
            "includeFinalAdaLn": bool(include_final_adaln),
            "baseFingerprint": list(sources["base"]), "overlayFingerprint": list(sources["overlay"]),
        }, ensure_ascii=False, separators=(",", ":")),
    }
    _data_bytes, overlay_bytes = _write_merged(base_info, Path(base), overlay_info, Path(overlay),
                                               plan, target,
                                               metadata={**merged_metadata, **(metadata or {})})

    sidecar_written, sidecar_error = _write_sidecar(target, sources)
    notes = ["产物**继承**了基底的内嵌元数据 ✓✗（丢了它 ⇒ 装的时候读不出结构 ✓）",
             "⚠️ 本层**不验张量内容** ✗：头自洽 ≠ 数值对 ✓（数值比对要真权重 ✓）"]
    if plan.skip:
        notes.append(f"跳过 {len(plan.skip)} 个非 adaLN 键 ✓（覆盖层只负责 adaLN ✓，理由逐条在 skip 里 ✓）")
    return HybridLoadReport(
        cache_path=str(target), reused=False, reason=reason, take=plan.take, skip=plan.skip,
        key_count=len(base_info.tensors), bytes=int(target.stat().st_size),
        bytes_from_overlay=overlay_bytes, sidecar_written=sidecar_written,
        sidecar_error=sidecar_error, disk_free_bytes=free,
        elapsed_ms=(time.perf_counter() - started) * 1000.0, notes=tuple(notes))


def _write_sidecar(cache: Path, sources: Mapping[str, Any]) -> tuple[bool, str | None]:
    """写源指纹 sidecar ✓ ⇒ ``(写没写成, 错因 ✓)`` —— ⚠️ **失败不阻塞** ✓✗（没有指纹 = 下次重合并 ✓）。"""
    try:
        payload = {name: {"size": int(value[0]), "mtime": float(value[1]), "hash": str(value[2])}
                   for name, value in sources.items() if value is not None}
        cg.cache_sidecar_path(cache).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8")
        return True, None
    except Exception as err:  # noqa: BLE001
        return False, f"{type(err).__name__}: {err}（⚠️ 没写成 ⇒ 下次会**重新合并** ✓，不是错 ✓✗）"


def _free_bytes(path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(path.parent if path.parent.exists() else Path.cwd()).free)
    except OSError:
        return None


def _gib(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"


def sha256_of_tensor(path: str | Path, name: str) -> str:
    """**自检用**的逐张量哈希 ✓（本层自己不验数值 ✓；比对是调用方的事 ✓）。"""
    info = st.inspect(path)
    entry = info.tensors.get(str(name))
    if entry is None:
        raise HybridLoadError(f"没有张量 {name!r} ✓")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        _copy_range(handle, 8 + int(info.header_bytes) + int(entry.begin), int(entry.nbytes),
                    _DigestSink(digest), label=f"{Path(path).name}::{name}")
    return digest.hexdigest()


class _DigestSink:
    """给 :func:`_copy_range` 用的「写进哈希」的假文件 ✓（不落盘 ✗）。"""

    def __init__(self, digest: Any) -> None:
        self._digest = digest

    def write(self, chunk: bytes) -> int:
        self._digest.update(chunk)
        return len(chunk)
