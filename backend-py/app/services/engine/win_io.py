"""Windows **无缓冲重叠读** ✓（移植自 ``reference/expert-stream/engine/expert-stream.c`` ✓）。

## 为什么照搬它 ✗

本仓装一件真权重是 13～19 GiB ✓，老路是 ``open()/mmap`` 读 ✓ ⇒ **两次拷贝** ✗：
盘 → 系统页缓存（小文件策略下还会做预读/顺序读提示 ✓✗）→ 用户缓冲 → torch 张量 ✓。
账很好算：19.53 GiB 走页缓存时**等于同时占 19.53 GiB 内存** ✓✗，而我们要装的东西本来就大，
⇒ 装第二件时"内存明明还有 90 GB 却换页" ✓✗。

``expert-stream`` 那份 C 做的事只有三件，三件都是我们缺的 ✓：

1. ``FILE_FLAG_NO_BUFFERING`` ✓ —— 绕开页缓存，读多少占多少 ✗；
2. ``FILE_FLAG_OVERLAPPED`` ＋ 一次提交多笔 ``ReadFile`` ✓ —— NVMe 的队列深度才用得上 ✓
   （同步读永远并发 1 ✓✗）；
3. 一个 ``VirtualAlloc`` 大区按扇区切槽 ✓ —— 无缓冲读要求**偏移、长度、目的地址三者都按扇区对齐** ✓，
   而 safetensors 的张量偏移是**任意字节** ✓✗ ⇒ 槽里留 ``slop``，读完再按真偏移切片 ✓（原实现同款）。

## 与原件不同的三处（都是本仓口径逼出来的 ✓）

* **不加 manifest / 专家表** ✗：这里是通用读取层 ✓，偏移由调用方给（``safetensors.py`` 的 header
  已经算好了 ✓）；
* **失败不静默** ✓✗：原件读失败就把该专家标 -1 让上层回落 mmap ✓；本仓落在「权重读短了」这条上
  后果是**装出半件模型还装得上** ✓✗ ⇒ 这里一律抛 ``OSError`` ✓（宁可不装 ✗）；
* **越界一律拒** ✓：请求尾巴超出文件长度 ⇒ ``ValueError`` ✓（读短了比不读更坏 ✓✗）。

## 平台

非 Windows ⇒ ``mode="plain"`` ✓ 并且 ``describe()["error"]`` 写明原因 ✓；
读法退化成普通 ``seek+read`` ✓（**结果逐字节相同** ✓、性能不同 ✓✗）——
所以本模块的判据在哪个平台都能跑 ✓，只是 Windows 上才真走无缓冲路径 ✓。
``force_plain=True`` 可强制走退化路径 ✓（A/B 测速用 ✓）。

## 本机实测（2026-11-01，RTX A5000 / 96 GB / D 盘 NVMe ✓）

拿盘上最大的真权重 `qwen_image_2512_bf16.safetensors`（38.05 GiB / 1933 张量 ✓）当靶子 ✓：

| 读法 | 乱序 200 段（9.57 GiB） | 顺序 2 GiB |
|---|---|---|
| 页缓存（`force_plain`） | 3.43 s / **2.79 GiB/s** | 0.89 s / 2.26 GiB/s |
| 无缓冲重叠（本模块） | 2.78 s / **3.44 GiB/s** | 0.91 s / 2.21 GiB/s |

两轮**逐字节相同** ✓。结论要如实说 ✓✗：**顺序读没有收益** ✗（顺读本来就吃满了盘 ✓），
收益只在**乱序多段**上（＋23% ✓）—— 也就是装载真权重那种访问模式 ✓；另一份收益是**内存账** ✓
（页缓存不再兜着 19 GiB ✓），这一项没法用秒表量 ✓，只能用「装第二件时内存没被第一件拖着」的事实看 ✓。

## 移植来源

* 上游：``expert-stream``（MIT，Copyright (c) 2026 Hemant Kumar），commit ``89275f1c50d2``；
* 原件：``engine/expert-stream.c`` 的 ``es_open`` / ``es_request`` / ``es_submit`` / ``es_get``；
* 本文件只取其中的 Win32 部分 ✓，不取 ggml 侧任何符号 ✓。
"""
from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = ["Span", "UnbufferedReader", "SECTOR_BYTES", "windows_available", "selftest"]

#: 扇区字节数 ✓（NVMe 物理扇区 4096 ✓；对 512 逻辑扇区也是整数倍 ✓ ⇒ 两种盘都合法 ✓）。
SECTOR_BYTES = 4096
#: 一个槽默认多大 ✓（8 MiB：够一次装一层注意力的权重 ✓，又不至于让 4 槽占光内存 ✓）。
DEFAULT_SLOT_BYTES = 8 << 20
#: 并发槽数 ✓（= NVMe 队列深度上限 ✓；再多只是内存换不来带宽 ✓）。
DEFAULT_SLOTS = 4

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
ERROR_IO_PENDING = 997
ERROR_HANDLE_EOF = 38
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
MEM_RELEASE = 0x00008000
PAGE_READWRITE = 0x04

_IS_WINDOWS = os.name == "nt"


class _OVERLAPPED(ctypes.Structure):
    """Win32 ``OVERLAPPED`` ✓（x64 上是 32 字节：4×指针/整数 ＋ 事件句柄 ✓）。"""

    _fields_ = [
        ("Internal", ctypes.c_void_p),
        ("InternalHigh", ctypes.c_void_p),
        ("Offset", ctypes.c_uint32),
        ("OffsetHigh", ctypes.c_uint32),
        ("hEvent", ctypes.c_void_p),
    ]


def _kernel32() -> Any:
    """拿 ``kernel32`` ✓（只在 Windows 上真调 ✓）。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                ctypes.c_void_p]
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                             ctypes.c_void_p, ctypes.POINTER(_OVERLAPPED)]
    k32.ReadFile.restype = ctypes.c_int
    k32.GetOverlappedResult.argtypes = [ctypes.c_void_p, ctypes.POINTER(_OVERLAPPED),
                                        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    k32.GetOverlappedResult.restype = ctypes.c_int
    k32.GetFileSizeEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    k32.GetFileSizeEx.restype = ctypes.c_int
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                 ctypes.c_uint32]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    k32.VirtualFree.restype = ctypes.c_int
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.CloseHandle.restype = ctypes.c_int
    k32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    k32.CreateEventW.restype = ctypes.c_void_p
    k32.ResetEvent.argtypes = [ctypes.c_void_p]
    k32.ResetEvent.restype = ctypes.c_int
    return k32


_K32: Any = _kernel32() if _IS_WINDOWS else None


def windows_available() -> bool:
    """本机能不能走无缓冲路径 ✓（不试读、只看平台 ✓ ⇒ 毫秒级 ✓）。"""
    return _IS_WINDOWS and _K32 is not None


def _align_down(value: int, unit: int) -> int:
    return (value // unit) * unit


def _align_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


@dataclass(frozen=True)
class Span:
    """一段要读的字节区间 ✓（``offset`` 是文件内绝对偏移 ✓）。"""

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length < 0:
            raise ValueError(f"Span 不许负值（收到 offset={self.offset} length={self.length} ✗）")


def _spans_of(spans: Iterable[Any]) -> list[Span]:
    """``(offset, length)`` 或 :class:`Span` ⇒ 一律 :class:`Span` ✓（两种写法都收 ✓）。"""
    out: list[Span] = []
    for span in spans:
        if isinstance(span, Span):
            out.append(span)
        else:
            offset, length = span
            out.append(Span(int(offset), int(length)))
    return out


class UnbufferedReader:
    """一个文件的无缓冲重叠读入口 ✓（``with`` 用法 ✓、``close`` 必调 ✓）。

    ⚠️ **不是线程安全**的 ✗：槽表是实例状态 ✓ ⇒ 一个实例归一个线程（本仓的装载本来就单线程 ✓）。
    跨线程要并发就各开各的实例 ✓（句柄便宜 ✓）。
    """

    def __init__(self, path: str | os.PathLike[str], *, slots: int = DEFAULT_SLOTS,
                 slot_bytes: int = DEFAULT_SLOT_BYTES, force_plain: bool = False) -> None:
        self.path = str(path)
        self._slots = max(1, int(slots))
        self._slot_bytes = _align_up(max(SECTOR_BYTES * 2, int(slot_bytes)), SECTOR_BYTES)
        #: 单个请求的上限 ✓：留一个扇区给 ``slop`` ✓（起点不对齐时最多偏 4095 字节 ✓）。
        self._max_request = self._slot_bytes - SECTOR_BYTES
        self._force_plain = bool(force_plain)
        self._handle: int | None = None
        self._arena: int | None = None
        self._size = 0
        self._mode = "closed"
        self._error = ""
        self._ovs = [_OVERLAPPED() for _ in range(self._slots)]
        self._events: list[int] = []
        #: 槽池 ✓：`_free` 是**可领的槽号**，`_lock` 保证领号原子 ✓（理由见 `_read_unbuffered` ✓）。
        self._lock = threading.Condition()
        self._free = list(range(self._slots))
        self.stats: dict[str, int] = {"reads": 0, "bytes": 0, "batches": 0, "fallbacks": 0}
        self._open()

    # ---- 生命周期 -------------------------------------------------------
    def _open(self) -> None:
        try:
            self._size = os.path.getsize(self.path)
        except OSError as err:
            self._degrade(f"取文件大小失败：{err}")
            return
        if self._force_plain:
            self._degrade("调用方要求 force_plain（A/B 测速用）")
            return
        if not windows_available():
            self._degrade("非 Windows 平台：无缓冲重叠读不可用")
            return
        handle = _K32.CreateFileW(
            self.path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, None)
        if handle == INVALID_HANDLE_VALUE or not handle:
            self._degrade(f"CreateFileW 失败（WinError={ctypes.get_last_error()}）")
            return
        size = ctypes.c_int64(0)
        if not _K32.GetFileSizeEx(ctypes.c_void_p(handle), ctypes.byref(size)):
            _K32.CloseHandle(ctypes.c_void_p(handle))
            self._degrade(f"GetFileSizeEx 失败（WinError={ctypes.get_last_error()}）")
            return
        arena = _K32.VirtualAlloc(None, self._slots * self._slot_bytes,
                                  MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not arena:
            _K32.CloseHandle(ctypes.c_void_p(handle))
            self._degrade("VirtualAlloc 失败（槽区分配不出来）")
            return
        #: ⚠️⚠️ **每槽一个手动复位事件** ✗✗（原件 ``es_open`` 同款 ✓）。不建事件的话，
        #: 同一句柄上多笔在飞时**文件句柄**被任何一笔完成就置信号 ✓ ⇒ ``GetOverlappedResult``
        #: 会对**还没完成**的那笔返回 TRUE ＋ 0 字节 ✓✗（实测症状：单笔读对、三笔一批必报
        #: 「读短了：要 8192 得 0」✗）—— 这不是优化，是**正确性** ✓✗。
        events: list[int] = []
        for index in range(self._slots):
            event = _K32.CreateEventW(None, 1, 0, None)  # 手动复位 TRUE ✓
            if not event:
                for made in events:
                    _K32.CloseHandle(ctypes.c_void_p(made))
                _K32.CloseHandle(ctypes.c_void_p(handle))
                _K32.VirtualFree(ctypes.c_void_p(arena), 0, MEM_RELEASE)
                self._degrade(f"CreateEventW 失败（WinError={ctypes.get_last_error()}）")
                return
            events.append(int(event))
            self._ovs[index].hEvent = ctypes.c_void_p(int(event))
        self._events = events
        self._handle = int(handle)
        self._arena = int(arena)
        self._size = int(size.value)
        self._mode = "unbuffered"

    def _degrade(self, reason: str) -> None:
        """降级到普通读 ✓（**不是**失败 ✗）—— 原因必须写进 ``describe()`` ✓，否则读得慢还查不出为什么 ✓✗。"""
        self._mode = "plain"
        self._error = reason

    def close(self) -> None:
        if self._handle is not None:
            _K32.CloseHandle(ctypes.c_void_p(self._handle))
            self._handle = None
        for event in self._events:
            _K32.CloseHandle(ctypes.c_void_p(event))
        self._events = []
        if self._arena is not None:
            _K32.VirtualFree(ctypes.c_void_p(self._arena), 0, MEM_RELEASE)
            self._arena = None
        #: ⚠️ 关了一律是「closed」✓（降级原因留在 ``_error`` 里 ✓）—— 若这里保留 "plain"，
        #: 关掉之后还能继续读 ✓✗：调用方就再也分不清「我关了没」 ✓✗。
        self._mode = "closed"

    def __enter__(self) -> "UnbufferedReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 读 -------------------------------------------------------------
    def read(self, offset: int, length: int) -> bytes:
        """读一段 ✓（批量请用 :meth:`read_many` ✓ —— 逐段调等于并发 1 ✓✗）。"""
        return self.read_many([Span(int(offset), int(length))])[0]

    def read_many(self, spans: Sequence[Any]) -> list[bytes]:
        """按**给定顺序**返回每段字节 ✓（允许乱序偏移 ✓；批量提交 ✓）。"""
        items = _spans_of(spans)
        if self._mode == "closed":
            raise RuntimeError("UnbufferedReader 已关闭（读过一次就关了？✗）")
        self._check_bounds(items)
        if not items:
            return []
        self.stats["batches"] += 1
        if self._mode != "unbuffered":
            self.stats["fallbacks"] += 1
            return self._read_plain(items)

        # ⚠️ 一段超过槽容量 ⇒ 先切块 ✓：单笔无缓冲读的缓冲必须落在**一个**槽里 ✓✗。
        #   切块后的顺序与回填顺序**必须同源** ✓（下面两趟都按 items 走，不各自算一遍 ✓✗）。
        pending: list[tuple[int, int]] = []
        for span in items:
            cursor, remaining = span.offset, span.length
            while remaining > 0:
                take = min(remaining, self._max_request)
                pending.append((cursor, take))
                cursor += take
                remaining -= take
        out_chunks = self._read_unbuffered(pending)
        pos = 0
        final: list[bytes] = []
        for span in items:
            parts = []
            remaining = span.length
            while remaining > 0:
                take = min(remaining, self._max_request)
                parts.append(out_chunks[pos])
                pos += 1
                remaining -= take
            final.append(b"".join(parts))
        return final

    def _check_bounds(self, items: Sequence[Span]) -> None:
        for span in items:
            if span.length and span.offset + span.length > self._size:
                raise ValueError(
                    f"越界读：offset={span.offset} length={span.length} 超出文件 "
                    f"{self._size} 字节 ✓✗（读短的后果是「装出半件模型还装得上」✗）")

    def _read_plain(self, items: Sequence[Span]) -> list[bytes]:
        out: list[bytes] = []
        with open(self.path, "rb", buffering=0) as fh:
            for span in items:
                fh.seek(span.offset)
                data = fh.read(span.length)
                if len(data) != span.length:
                    raise OSError(f"普通读也读短了：要 {span.length} 得 {len(data)} ✗")
                out.append(data)
        self.stats["reads"] += len(items)
        self.stats["bytes"] += sum(span.length for span in items)
        return out

    def _read_unbuffered(self, jobs: Sequence[tuple[int, int]]) -> list[bytes]:
        """乱序批量提交 ✓（一次最多 ``slots`` 笔在飞 ✓ —— 这是 NVMe 才吃得饱的原因 ✓）。

        ⚠️ **槽必须真占住**（2026-11-01 修的真缺陷 ✓✗）：早先按「批内下标 0..n-1」用槽 ⇒
        两个线程同时读时都拿到下标 0 ⇒ **同一个 OVERLAPPED ＋ 同一个事件 ＋ 同一块槽区** ✓✗，
        症状不是报错而是**短读/读到 0 字节**（`engine_readahead_test` 连跑三次才露头 ✓）。
        单线程用例永远测不出来 ✗ —— 这就是「只有并发才现形」的那类 ✗✗。现在改成从池里
        **原子领号**、用完归还 ✓（并发读不再串行 ✓，也不会互相踩 ✓）。
        """
        out: list[bytes] = []
        for start in range(0, len(jobs), self._slots):
            batch = jobs[start:start + self._slots]
            slot_ids = self._acquire_slots(len(batch))
            try:
                out.extend(self._run_batch(batch, slot_ids))
            finally:
                self._release_slots(slot_ids)
        return out

    def _acquire_slots(self, count: int) -> list[int]:
        """领 ``count`` 个**互不相同**的槽号 ✓（不够就等别人还 ✓✗：宁可等，也不许两家共用一个槽 ✗✗）。"""
        with self._lock:
            while len(self._free) < count:  # 池子总量 ≥1，单笔一定能等到 ✓
                self._lock.wait()
            ids = self._free[:count]
            del self._free[:count]
            return ids

    def _release_slots(self, slot_ids: Sequence[int]) -> None:
        with self._lock:
            self._free.extend(slot_ids)
            self._lock.notify_all()

    def _run_batch(self, batch: Sequence[tuple[int, int]],
                   slot_ids: Sequence[int]) -> list[bytes]:
        issued: list[tuple[Any, int, int]] = []  # (OVERLAPPED, lead, want)
        for index, (offset, length) in enumerate(batch):
            slot = slot_ids[index]
            aligned_start = _align_down(offset, SECTOR_BYTES)
            lead = offset - aligned_start
            want = length
            aligned_len = _align_up(lead + want, SECTOR_BYTES)
            if aligned_len > self._slot_bytes:
                raise AssertionError(f"内部错误：对齐后 {aligned_len} 超过槽 {self._slot_bytes} ✗")
            address = self._arena + slot * self._slot_bytes  # type: ignore[operator]
            ov = self._ovs[slot]
            ctypes.memset(ctypes.byref(ov), 0, ctypes.sizeof(ov))
            event = self._events[slot]
            _K32.ResetEvent(ctypes.c_void_p(event))  # ⚠️ 复位这槽的事件 ✓（不清＝上一笔的信号骗过这一笔 ✗✗）
            ov.hEvent = ctypes.c_void_p(event)
            ov.Offset = aligned_start & 0xFFFFFFFF
            ov.OffsetHigh = (aligned_start >> 32) & 0xFFFFFFFF
            ok = _K32.ReadFile(ctypes.c_void_p(self._handle), ctypes.c_void_p(address),
                               aligned_len, None, ctypes.byref(ov))
            if not ok and ctypes.get_last_error() != ERROR_IO_PENDING:
                err = ctypes.get_last_error()
                if err == ERROR_HANDLE_EOF:
                    raise OSError(f"读到文件尾：offset={offset} length={length} ✗")
                raise OSError(f"ReadFile 提交失败（WinError={err}）offset={offset} ✗")
            issued.append((ov, lead, want, slot))
        results: list[bytes] = []
        for index, (ov, lead, want, slot) in enumerate(issued):
            got = ctypes.c_uint32(0)
            if not _K32.GetOverlappedResult(ctypes.c_void_p(self._handle), ctypes.byref(ov),
                                            ctypes.byref(got), 1):
                err = ctypes.get_last_error()
                if err == ERROR_HANDLE_EOF:
                    raise OSError(f"读到文件尾（完成时）：第 {index} 笔 ✗")
                raise OSError(f"GetOverlappedResult 失败（WinError={err}）第 {index} 笔 ✗")
            if got.value < lead + want:
                raise OSError(f"无缓冲读短了：要 {lead + want} 得 {got.value} ✗")
            address = self._arena + slot * self._slot_bytes  # type: ignore[operator]
            results.append(ctypes.string_at(address + lead, want))
            self.stats["reads"] += 1
            self.stats["bytes"] += want
        return results

    # ---- 可见面 ---------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """这次到底走了哪条路 ✓（``error`` 非空＝降级原因 ✓，不是"坏了" ✓）。"""
        return {
            "path": self.path,
            "mode": self._mode,
            "windows": _IS_WINDOWS,
            "error": self._error,
            "fileSize": self._size,
            "slots": self._slots,
            "slotBytes": self._slot_bytes,
            "sectorBytes": SECTOR_BYTES,
            "stats": dict(self.stats),
        }


def selftest() -> int:
    """自检 ✓（写一个临时文件读回来 ✓，不依赖盘上任何权重 ✓）。"""
    import tempfile

    payload = bytes(range(256)) * 4097 + b"tail"
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "win_io_selftest.bin")
        with open(path, "wb") as fh:
            fh.write(payload)
        for force_plain in (True, False):
            with UnbufferedReader(path, slots=3, slot_bytes=SECTOR_BYTES * 3,
                                  force_plain=force_plain) as reader:
                for offset, length in ((0, 1), (1, 4096), (4095, 3), (0, len(payload))):
                    got = reader.read(offset, length)
                    if got != payload[offset:offset + length]:
                        failures.append(f"{reader.describe()['mode']} offset={offset} 不符")
                many = reader.read_many([(7, 100), (0, 3), (4000, 5000)])
                for (offset, length), got in zip(((7, 100), (0, 3), (4000, 5000)), many):
                    if got != payload[offset:offset + length]:
                        failures.append(f"批量 offset={offset} 不符")
    for line in failures:
        print("FAIL  " + line)
    print(f"SUMMARY: {'OK' if not failures else 'FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
