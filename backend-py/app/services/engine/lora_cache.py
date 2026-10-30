"""LoRA **内容身份 + 引用计数 + 释放事实** ✓（E2.5 ✓）。

为什么单独立这一层 ✗（不放进 ``lora.py`` ✓）：
``lora.py`` 是**纯算法**（解析 / 折算 / 挂载 ✓，无状态 ✓），而"哪一份张量还活着、
几个人在用、什么时候该丢"是**进程状态** ✓✗ —— 两者混一起，算法自检就得先造一份缓存 ✓；
更要紧的是：``lora.py`` 的 `read_tensors` 每次调用都**整份重读** ✓ ⇒
同一份 LoRA 换个路径／文件名再挂一次，就会**再读一份、再占一份** ✗（显存与 RSS 双翻倍 ✓✗，
而且没有任何地方能看出来 ✗）。这一层以**内容 sha256** 为身份 ✓ ⇒ 同内容只留**一份张量** ✓。

⚠️ 模块级**零依赖** ✓（只 stdlib ✓）：无 torch 也能 import 与自检 ✓ ——
与 ``lora.py`` 同一条纪律 ✓（引擎不许因为"没装 torch"就连账本都读不出来 ✗）。

⚠️ 口径（别读多 ✗）：这里只管**LoRA 张量**这份内存 ✓；torch allocator 的缓存块不在这里 ✓
（那是 `runtime._reclaim_cuda_cache` 的活 ✓），主干的装配复用门也不在这里 ✓（`status().cache` ✓）。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "LoraCacheError",
    "LoraEntry",
    "LoraLease",
    "LoraCache",
    "shared",
    "digest_file",
    "CONTENT_ALGO",
]

#: 内容身份的算法 ✓ —— 换它等于**换身份** ✗✗（历史摘要会全部对不上 ⇒ 需要迁移或清空 ✓）
CONTENT_ALGO = "sha256"

#: 分块读的块大小 ✓（1 MiB ✓：LoRA 一般 100～500 MB ✓，太小则 syscall 多、太大则吃 RSS 峰值 ✓✗）
_CHUNK = 1 << 20

#: `facts()["recent"]` 留几条 ✓（与 `status().cache.recent` 同口径 ✓：够看出"刚才发生了什么"✓，
#: 留全量 ✗ 长跑进程会涨 ✓✗）
_EVENT_KEEP = 8


class LoraCacheError(RuntimeError):
    """缓存层的**具名**错误 ✓ —— 文件没了 / 读不动 / 引用计数对不上 ✓。"""


def digest_file(path: str | Path) -> str:
    """整个文件的**内容摘要**（`sha256` 十六进制 ✓；分块读 ✓ 不整份进内存 ✓）。

    ⚠️ 为什么不用 ``facade._sha256_of_file`` ✗：那是 facade 的私有函数 ✓，而 facade **import**
    runtime ✓ ⇒ 这里反向 import 就是循环 ✓✗（模块依赖方向：facade → runtime → lora_cache ✓，
    只能往下 ✓）。两处都是"分块读整个文件算 sha256"的十行 ✓，重复的代价 < 循环依赖的代价 ✓。
    """
    target = Path(path)
    digest = hashlib.new(CONTENT_ALGO)
    try:
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as err:
        raise LoraCacheError(f"读不了 LoRA 文件 ✗：{target}（{type(err).__name__}: {err}）") from err
    return digest.hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class LoraEntry:
    """一份**已读进内存**的 LoRA ✓（同一内容的所有路径共享这一份 ✓）。"""

    digest: str
    path: str
    fileBytes: int
    tensors: Mapping[str, Any]
    tensorBytes: int | None = None
    refCount: int = 0
    firstSeen: int = 0
    #: ⚠️ 同一内容挂过的**其它路径** ✓ —— 这就是「同内容只占一份」的**凭据** ✓✗
    #: （没有它只能看到 refCount=2，看不出"这两个引用指向同一个文件还是两个不同文件" ✓）
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest[:12],
            "path": self.path,
            "aliases": list(self.aliases),
            "fileBytes": self.fileBytes,
            "tensorBytes": self.tensorBytes,
            "refCount": self.refCount,
            "firstSeen": self.firstSeen,
        }


@dataclass(frozen=True)
class LoraLease:
    """一次`acquire`的凭据 ✓ —— **拿到就必须还** ✗✗（不还 = 这份张量永远不释放 ✓✗）。"""

    digest: str
    path: str
    tensors: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"digest": self.digest[:12], "path": self.path}


class LoraCache:
    """进程级 LoRA 缓存 ✓（**一个进程一份** ✓，见 :func:`shared` ✓）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, LoraEntry] = {}
        #: ``(路径, 字节数, mtime_ns)`` ⇒ 摘要 ✓ —— 只是**省一次 sha256** 的快路 ✓，
        #: ⚠️ 它**不是**身份 ✗：文件被原地换掉且大小与 mtime **都不变** ⇒ 这里认不出来 ✓✗
        #: （`digest_file` 仍是唯一真源 ✓；要绝对正确就别走快路 ✓）
        self._byStat: dict[tuple[str, int, int], str] = {}
        self._hits = 0
        self._misses = 0
        self._deduped = 0
        self._freed = 0
        self._freedBytes = 0
        self._events: list[dict[str, Any]] = []
        self._since = _now_ms()

    # ── 记账 ✓ ───────────────────────────────────────────────────────────
    def _note(self, kind: str, *, digest: str, path: str, reason: str) -> None:
        self._events.append({"at": _now_ms(), "kind": kind, "digest": digest[:12],
                             "path": path, "reason": reason})
        del self._events[:-_EVENT_KEEP]

    @staticmethod
    def _stat_key(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path), int(stat.st_size), int(stat.st_mtime_ns))

    @staticmethod
    def _tensor_bytes(tensors: Mapping[str, Any]) -> int | None:
        """张量真占多少字节 ✓ —— 问不出来就 ``None`` ✗（**不编数字** ✗✗）。

        为什么不用文件大小顶替 ✗：``.safetensors`` 带 JSON 头与对齐填充 ✓，
        文件字节恒大于张量字节 ✓✗ —— 拿文件大小当"占了多少内存"会系统性高估 ✓。
        """
        total = 0
        try:
            for tensor in tensors.values():
                total += int(tensor.numel()) * int(tensor.element_size())
        except Exception:  # noqa: BLE001 —— 假张量 / 没 torch ⇒ 如实"算不出" ✓
            return None
        return total

    def digest_of(self, path: str | Path) -> str:
        """路径 ⇒ 内容摘要 ✓（走 ``(路径,大小,mtime)`` 快路 ✓；文件不在 ⇒ :class:`LoraCacheError` ✓）。"""
        target = Path(path)
        if not target.is_file():
            raise LoraCacheError(f"LoRA 文件不在 ✗：{target}")
        try:
            key = self._stat_key(target)
        except OSError as err:
            raise LoraCacheError(f"读不了 LoRA 文件属性 ✗：{target}（{err}）") from err
        with self._lock:
            known = self._byStat.get(key)
        if known is not None:
            return known
        digest = digest_file(target)
        with self._lock:
            self._byStat[key] = digest
        return digest

    # ── 借 / 还 ✓ ────────────────────────────────────────────────────────
    def acquire(self, path: str | Path, *,
                reader: Callable[[str], Mapping[str, Any]] | None = None,
                reason: str = "") -> LoraLease:
        """借一份 LoRA 张量 ✓（**引用计数 +1** ✓）—— 同内容**不再读第二遍** ✓。

        ``reader`` 是**唯一**读盘口 ✓（真机传 ``lora.read_tensors`` ✓，自检传假张量表 ✓）；
        省略 ⇒ 现场 import ``lora.read_tensors``（会拖 torch ✓，只在真机用 ✓）。
        """
        target = Path(path)
        digest = self.digest_of(target)
        load = reader
        if load is None:
            from . import lora as lora_mod  # noqa: PLC0415 —— 模块级零依赖 ✓，要用才拖 torch ✓

            load = lora_mod.read_tensors
        with self._lock:
            entry = self._entries.get(digest)
            if entry is not None:
                entry.refCount += 1
                if str(target) != entry.path and str(target) not in entry.aliases:
                    entry.aliases.append(str(target))
                    self._deduped += 1
                    self._note("dedup", digest=digest, path=str(target),
                               reason=("同内容换个路径再挂 ✓ ⇒ **复用已有的那份张量** ✓"
                                       "（没有第二次读盘、没有第二份内存 ✓）"))
                else:
                    self._hits += 1
                    self._note("hit", digest=digest, path=str(target),
                               reason="同内容同路径再借 ✓ ⇒ 只加引用计数 ✓（不重读 ✓）")
                return LoraLease(digest=digest, path=str(target), tensors=entry.tensors)
        # ⚠️ 读盘放在锁**外** ✗✗：读一个 500 MB 的文件要几百毫秒 ✓，持锁读会把别的请求全堵住 ✓✗
        tensors = load(str(target))
        size = self._file_bytes(target)
        tensor_bytes = self._tensor_bytes(tensors)
        with self._lock:
            # 双检 ✓：等我们读盘的时候，另一个线程可能已经把同一份读进来了 ✓ ⇒ 丢掉自己这份 ✓
            entry = self._entries.get(digest)
            if entry is not None:
                entry.refCount += 1
                if str(target) not in entry.aliases and str(target) != entry.path:
                    entry.aliases.append(str(target))
                    self._deduped += 1
                self._note("dedup", digest=digest, path=str(target),
                           reason="读盘期间发现已有同内容 ✓ ⇒ 丢掉自己这份、复用它的 ✓（并发去重 ✓）")
                return LoraLease(digest=digest, path=str(target), tensors=entry.tensors)
            entry = LoraEntry(digest=digest, path=str(target), fileBytes=size,
                              tensors=tensors, tensorBytes=tensor_bytes,
                              refCount=1, firstSeen=_now_ms())
            self._entries[digest] = entry
            self._misses += 1
            self._note("miss", digest=digest, path=str(target),
                       reason=reason or "还没有这份内容 ⇒ 读盘一次 ✓（之后同内容都复用它 ✓）")
            return LoraLease(digest=digest, path=str(target), tensors=entry.tensors)

    @staticmethod
    def _file_bytes(path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    def release(self, lease: LoraLease | str, *, reason: str = "") -> dict[str, Any]:
        """还一份 ✓（引用计数 -1 ✓）—— **减到 0 才真丢** ✓ ⇒ 返回这次到底丢了什么 ✓。

        ⚠️ 引用计数**不许减到负数** ✗✗（那说明还多了一次 ⇒ 当场报错 ✓ 而不是静默把别人的还掉 ✓）。
        """
        digest = lease.digest if isinstance(lease, LoraLease) else str(lease)
        with self._lock:
            entry = self._entries.get(digest)
            if entry is None:
                raise LoraCacheError(
                    f"还一份不存在的 LoRA ✗：{digest[:12]} ✓（要么摘要不对、要么已经丢过了 ✓）")
            if entry.refCount <= 0:
                raise LoraCacheError(
                    f"LoRA 引用计数已经是 0 ✗：{digest[:12]} ✓ ⇒ 还多了 ✓（说明 acquire 与 "
                    f"release 不成对 ✓✗）")
            entry.refCount -= 1
            if entry.refCount > 0:
                return {"digest": digest[:12], "released": False, "refCount": entry.refCount,
                        "tensorBytes": 0,
                        "note": f"还有 {entry.refCount} 个引用 ✓ ⇒ 张量**留着** ✓"}
            del self._entries[digest]
            self._byStat = {key: value for key, value in self._byStat.items() if value != digest}
            self._freed += 1
            self._freedBytes += int(entry.tensorBytes or 0)
            self._note("release", digest=digest, path=entry.path,
                       reason=reason or "最后一个引用也还了 ✓ ⇒ 张量真丢 ✓")
            return {"digest": digest[:12], "released": True, "refCount": 0,
                    "fileBytes": entry.fileBytes, "tensorBytes": entry.tensorBytes,
                    "note": "引用计数到 0 ⇒ 这份张量已丢 ✓"}

    def release_all(self, *, reason: str = "") -> dict[str, Any]:
        """把所有还活着的都丢掉 ✓（**卸载模型时用** ✓：模型没了 ⇒ LoRA 张量没人用了 ✓）。

        ``outstanding`` = 丢的时候**引用计数还不是 0** 的份数 ✓ —— 不为 0 说明有人**没还** ✓✗，
        这时张量仍被那边引用着（``del`` 后由引用计数回收 ✓），
        所以 ``released`` 只代表"本账本不再持有" ✓，**不代表**内存当场就少了 ✓✗。
        """
        with self._lock:
            outstanding = 0
            freed_bytes = 0
            file_bytes = 0
            entries = list(self._entries.values())
            for entry in entries:
                if entry.refCount > 0:
                    outstanding += 1
                freed_bytes += int(entry.tensorBytes or 0)
                file_bytes += entry.fileBytes
                self._note("release", digest=entry.digest, path=entry.path,
                           reason=reason or "模型卸载 ⇒ 缓存整份清掉 ✓")
            self._entries.clear()
            self._byStat.clear()
            if entries:
                self._freed += len(entries)
                self._freedBytes += freed_bytes
            return {"entries": len(entries), "outstanding": outstanding,
                    "fileBytes": file_bytes, "tensorBytes": freed_bytes,
                    "note": ("模型卸载 ⇒ 本账本不再持有这些张量 ✓"
                             + (f"；⚠️ {outstanding} 份仍有未归还的引用 ✗ ⇒ 内存要等那边放手 ✓✗"
                                if outstanding else ""))}

    # ── 自述 ✓ ───────────────────────────────────────────────────────────
    def facts(self) -> dict[str, Any]:
        """给 `status().lora` 的那份事实 ✓（**取同一瞬间** ✓ ⇒ 持锁快照 ✓）。"""
        with self._lock:
            entries = [entry.to_dict() for entry in self._entries.values()]
            tensor_total = sum(int(entry.tensorBytes or 0) for entry in self._entries.values() if entry.tensorBytes)
            unknown = sum(1 for entry in self._entries.values() if entry.tensorBytes is None)
            return {
                "entries": entries,
                "entryCount": len(entries),
                "refTotal": sum(entry.refCount for entry in self._entries.values()),
                "tensorBytes": tensor_total,
                #: ⚠️ 有算不出的份时**必须说出来** ✗✗：否则 `tensorBytes` 会被读成"总共就这么多" ✓✗
                "tensorBytesUnknown": unknown,
                "hits": self._hits,
                "misses": self._misses,
                "deduped": self._deduped,
                "freed": self._freed,
                "freedBytes": self._freedBytes,
                "since": self._since,
                "recent": [dict(event) for event in self._events],
                "scope": ("本进程内 LoRA 张量的**内容级复用** ✓（同 sha256 ⇒ 只留一份 ✓）；"
                          "不含主干装配复用门（那是 `cache` ✓）、不含 torch allocator 缓存块 ✓"),
                "note": ("⚠️ 进程重启即归零 ✗；`tensorBytes` 是**张量实占**（按 numel×itemsize ✓），"
                         "不等于文件字节 ✓；`tensorBytesUnknown > 0` 时总数**偏小** ✓✗"),
            }

    def reset(self) -> None:
        """清空（**只给自检与"换模型"用** ✓）—— 与 :meth:`release_all` 不同：这里连计数一起归零 ✓。"""
        with self._lock:
            self._entries.clear()
            self._byStat.clear()
            self._hits = self._misses = self._deduped = 0
            self._freed = 0
            self._freedBytes = 0
            self._events.clear()
            self._since = _now_ms()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_SHARED = LoraCache()


def shared() -> LoraCache:
    """进程级单例 ✓ —— 为什么要单例 ✗：`torch_backend` 每次换装配都会**新建**一个后端实例 ✓，
    缓存若挂在实例上 ⇒ 换一次装就丢一次 ✓✗（那正是"同一 LoRA 加载两次占两份"的另一半原因 ✓）。"""
    return _SHARED
