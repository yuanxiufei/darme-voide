"""``safetensors`` **读取器**（纯 Python，零依赖 ✓）—— 引擎的权重体检层。

## 为什么自己实现（而不是装 ``safetensors`` 库）

``safetensors`` 的**容器格式极简**，读它只需要 ``struct`` + ``json`` + ``seek`` ✓：

```
[0:8]         u64 小端 = 头部长度 N
[8:8+N]       头部 JSON（可带尾部空格做 8 字节对齐 ✓）
[8+N:]        张量数据区；头部里每个张量的 data_offsets 是**相对数据区起点**的 [start, end) ✓
```

自己实现带来三个**真需求**（都不是"为了不装库" ✗）：

1. **不下载也能验证**：模型动辄 **19.53 GiB** ✓，跑之前先确认「文件齐不齐、有没有被截断」✗
   —— 截断的 safetensors 在加载时才会炸（而且常常炸在 GPU 上下文里 ✗）。
2. **只读头部**：几十 GB 的文件，**头几 KB 就能拿到全部张量的名字/形状/精度** ✓
   ⇒ 不做任何大内存分配就能回答「这个权重是不是我要的那个」✓。
3. **按需切片**：真要某一层时只 ``seek`` 读那一段 ✓（避免整文件进内存 ✗）。

⚠️ 本模块**不解析 GGUF** ✗（``configs/models.json`` 里有 GGUF 变体 ✓）—— GGUF 是另一套容器格式，
**读取器在** :mod:`app.services.engine.gguf` ✓（2026-09-20 起有 ✓）；本模块遇到 ``.gguf``
会**明确报错**而不是猜 ✓。

⚠️ 本模块**不做反量化/不提权重到张量** ✗（那需要 torch/numpy ✓）—— 它只给「清单 + 形状 + 字节」✓。
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "DTYPE_ITEMSIZE",
    "SafetensorsError",
    "SafetensorsInfo",
    "TensorEntry",
    "TensorReader",
    "inspect",
    "read_header",
    "read_tensor_bytes",
]

#: dtype → 每元素字节数（safetensors 规范里出现的全部取值 ✓；未知取值会被明确报错 ✓）
DTYPE_ITEMSIZE: dict[str, int] = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

#: 头部长度上限（防呆：坏文件里读出来的 N 可能是天文数字 ✗）
MAX_HEADER_BYTES = 100 * 1024 * 1024


class SafetensorsError(RuntimeError):
    """文件不是合法 safetensors（或与头部自述不一致 ✓）。"""


@dataclass
class TensorEntry:
    """头部里的一个张量（**只有元信息，没有数据** ✓）。"""

    name: str
    dtype: str
    shape: list[int]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def element_count(self) -> int:
        total = 1
        for value in self.shape:
            total *= int(value)
        return total


@dataclass
class SafetensorsInfo:
    """一次体检的结论 ✓（``ok=False`` 时看 ``problems`` ✓）。"""

    path: str
    file_bytes: int = 0
    header_bytes: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    dtype_counts: dict[str, int] = field(default_factory=dict)
    total_tensor_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    def biggest(self, count: int = 8) -> list[TensorEntry]:
        """最大的几个张量（看结构时最有用 ✓）。"""
        return sorted(self.tensors.values(), key=lambda item: item.nbytes, reverse=True)[:count]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "fileBytes": self.file_bytes,
            "headerBytes": self.header_bytes,
            "tensorCount": self.tensor_count,
            "totalTensorBytes": self.total_tensor_bytes,
            "dtypeCounts": self.dtype_counts,
            "metadata": self.metadata,
            "problems": self.problems,
            "biggest": [{"name": t.name, "dtype": t.dtype, "shape": t.shape, "bytes": t.nbytes}
                        for t in self.biggest()],
        }


def _require_safetensors(path: Path) -> None:
    if path.suffix.lower() == ".gguf":
        raise SafetensorsError(
            f"{path.name} 是 GGUF ✗ —— 本读取器只认 safetensors（GGUF 是另一套容器格式 ✓）")
    if not path.exists():
        raise SafetensorsError(f"文件不存在：{path}")
    if not path.is_file():
        raise SafetensorsError(f"不是普通文件：{path}")


def _read_header(path: str | Path) -> tuple[dict[str, str], dict[str, Any], int]:
    """读头部并**连长度一起返回** ✓：``(metadata, raw_header, header_length)``。

    ⚠️ 只读一次是有原因的：初版在 :func:`inspect` 里又自己 ``seek(8) + read(8)`` 想再拿一次长度 ✗
    —— 但长度在文件**开头**（offset 0 ✓），``seek(8)`` 之后读到的是**数据区开头** ✗
    ⇒ 长度成了垃圾值、``data_bytes`` 算成 0 ⇒ **合法文件被判成"截断"** ✗（自检 ① 当场抓到 ✓）。
    现在头部只解析一次，长度与内容**同源** ✓，这类错误没有复发余地 ✓。
    """
    target = Path(path)
    _require_safetensors(target)
    file_bytes = target.stat().st_size
    if file_bytes < 8:
        raise SafetensorsError(f"文件太小（{file_bytes} 字节）⇒ 不是 safetensors ✗")
    with target.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > MAX_HEADER_BYTES:
            raise SafetensorsError(
                f"头部长度异常：{header_length} 字节（上限 {MAX_HEADER_BYTES}）⇒ 文件损坏或不是 "
                f"safetensors ✗")
        if 8 + header_length > file_bytes:
            raise SafetensorsError(
                f"头部长度 {header_length} 超出文件大小 {file_bytes - 8} ⇒ **文件被截断** ✗")
        payload = handle.read(header_length)
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise SafetensorsError(f"头部不是合法 JSON：{err}") from err
    if not isinstance(header, dict):
        raise SafetensorsError("头部 JSON 不是对象 ✗")
    metadata = header.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {str(k): str(v) for k, v in metadata.items()}, header, header_length


def read_header(path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    """只读头部 ✓ ⇒ 返回 ``(metadata, raw_header)``；不做任何大内存分配 ✓。"""
    metadata, header, _length = _read_header(path)
    return metadata, header


def _parse_tensor(name: str, raw: Any, data_bytes: int) -> TensorEntry:
    if not isinstance(raw, dict):
        raise SafetensorsError(f"张量 {name!r} 的头部不是对象 ✗")
    dtype = str(raw.get("dtype") or "")
    if dtype not in DTYPE_ITEMSIZE:
        raise SafetensorsError(f"张量 {name!r} 的 dtype {dtype!r} 不认识 ✗")
    shape = raw.get("shape") or []
    if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise SafetensorsError(f"张量 {name!r} 的 shape 非法：{shape!r} ✗")
    offsets = raw.get("data_offsets") or []
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise SafetensorsError(f"张量 {name!r} 的 data_offsets 非法：{offsets!r} ✗")
    begin, end = int(offsets[0]), int(offsets[1])
    if begin < 0 or end < begin:
        raise SafetensorsError(f"张量 {name!r} 的区间非法：[{begin}, {end}) ✗")
    if end > data_bytes:
        raise SafetensorsError(
            f"张量 {name!r} 的区间 [{begin}, {end}) 超出数据区（{data_bytes} 字节）⇒ **文件被截断** ✗")
    entry = TensorEntry(name=name, dtype=dtype, shape=[int(d) for d in shape], begin=begin, end=end)
    expected = entry.element_count * DTYPE_ITEMSIZE[dtype]
    if expected != entry.nbytes:
        raise SafetensorsError(
            f"张量 {name!r} 自相矛盾：shape×dtype = {expected} 字节，但区间给出 {entry.nbytes} 字节 ✗")
    return entry


def inspect(path: str | Path) -> SafetensorsInfo:
    """完整体检：头部 + 每个张量的区间/形状自洽性 + 是否被截断 ✓。

    **不做大数据读取** ✓（只 seek 头部的 ``8 + N`` 字节 ✓）⇒ 对 19 GiB 的文件也是毫秒级 ✓。
    """
    target = Path(path)
    info = SafetensorsInfo(path=str(target))
    try:
        # ⚠️ 先过一遍前置检查：初版直接 `target.stat()` ⇒ 缺文件时冒的是 Windows 原生
        #    `[WinError 2] 系统找不到指定的文件` ✗（英文、且没人知道这是哪个组件 ✗）；
        #    自检 ㉙ 当场抓到 ⇒ 现在「文件不存在：<路径>」这句会先说出来 ✓。
        _require_safetensors(target)
        info.file_bytes = target.stat().st_size
        info.metadata, header, header_length = _read_header(target)
    except (SafetensorsError, OSError) as err:
        info.problems.append(str(err))
        return info

    info.header_bytes = header_length
    data_bytes = max(0, info.file_bytes - 8 - header_length)

    for name, raw in header.items():
        try:
            entry = _parse_tensor(str(name), raw, data_bytes)
        except SafetensorsError as err:
            info.problems.append(str(err))
            continue
        info.tensors[entry.name] = entry
        info.dtype_counts[entry.dtype] = info.dtype_counts.get(entry.dtype, 0) + 1
        info.total_tensor_bytes += entry.nbytes

    # 数据区里如果有「谁都没声明」的尾部空间 ⇒ 很可能是写入中断（值得报出来 ✓）
    declared_end = max((entry.end for entry in info.tensors.values()), default=0)
    if not info.problems and declared_end < data_bytes:
        info.problems.append(
            f"数据区尾部有 {data_bytes - declared_end} 字节无人声明 ⇒ 可能写入中断 ✗")
    if not info.tensors:
        info.problems.append("头部里没有任何张量 ✗")
    return info


def read_tensor_bytes(path: str | Path, name: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """按需只读一个张量的**原始字节** ✓（超过 ``max_bytes`` 会拒绝 ✗ 防手滑把 19 GiB 拉进内存 ✓）。"""
    target = Path(path)
    _require_safetensors(target)
    with target.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8)
        header = json.loads(handle.read(header_length).decode("utf-8"))
        raw = header.get(name)
        if not isinstance(raw, dict):
            raise SafetensorsError(f"没有张量 {name!r}（可用：{len(header)} 个 ✓）")
        begin, end = (int(x) for x in raw["data_offsets"])
        if end - begin > max_bytes:
            raise SafetensorsError(
                f"张量 {name!r} 有 {end - begin} 字节，超过 max_bytes={max_bytes} ✗"
                f"（真要整块请显式调大 ✓）")
        handle.seek(8 + header_length + begin)
        return handle.read(end - begin)


class TensorReader:
    """**一次**解析头部、按名 seek 取字节 ✓ —— 流式装载的唯一 IO 入口 ✓。

    为什么不用 :func:`read_tensor_bytes` 循环：它每次调用都重开文件 + 重读头部 ✗
    —— 真实检查点上万个张量 ⇒ 上万次头部解析 ✓✗（头部本身就有几 MB ✓）。
    本类把「头部解析**一次**」与「按偏移取**一个**张量」分开 ✓，
    且**不持有任何张量数据** ✗：读出来的字节由调用方立刻消费后丢弃 ✓。

    ⚠️ 它**不做 mmap** ✗（2026-10-29 定）：`seek + read` 与 mmap 在「按层顺序读」这个
    访问模式下**效果等价** ✓（都是一次顺序 IO ✓），而 mmap 的页由 OS 工作集管理器回收，
    Windows 上**时机不可控** ✗ —— 页被挤掉时延迟会在最不该出现的时刻冒出来 ✓✗。
    显式 seek 的失败点可枚举（读短了就是截断 ✓），对「资源受限环境」更好归因 ✓。

    ⚠️ ``unbuffered``（2026-11-01 移植 ``win_io`` 后接上 ✓）：
    * ``None``（默认）＝**Windows 上用无缓冲重叠读、其他平台用 seek+read** ✓；
    * ``True``/``False`` 可强制指定 ✓（A/B 与排障的口 ✓）。
    两条路**返回的字节必须逐字节相同** ✓✗（判据在 ``tests/engine_win_io_test.py`` ✓）。
    ⚠️ 无缓冲读**不走页缓存** ⇒ 同一份权重装第二遍不会再从内存里捡 ✓，
    这既是它的好处（内存账不翻倍 ✓）也是它的代价（重读要吃满一次盘 ✓）✓✗。
    """

    #: 槽区上限 ✓：够装下最大张量就够 ✓（不够 ``win_io`` 自己会切块 ✓）。
    _SLOT_CAP_BYTES = 256 * 1024 * 1024
    #: 同时在飞的笔数 ✓（对应 NVMe 队列深度 ✓；4 起步不抢采样线程 ✗）。
    _SLOTS = 4

    def __init__(self, path: str | Path, *, unbuffered: bool | None = None) -> None:
        self._path = Path(path)
        self._metadata, header, header_length = _read_header(self._path)
        self._header_length = header_length
        self._data_start = 8 + header_length
        data_bytes = max(0, self._path.stat().st_size - self._data_start)
        self.entries: dict[str, TensorEntry] = {}
        for name, raw in header.items():
            self.entries[str(name)] = _parse_tensor(str(name), raw, data_bytes)
        self._handle: Any = None
        self._raw: Any = None
        self._raw_error = ""
        self._closed = False
        self._unbuffered = unbuffered
        self._open_raw()

    # ---------------------------------------------------------------- 无缓冲层

    def _open_raw(self) -> None:
        """建无缓冲读口 ✓（**失败一律退回 seek+read** ✓✗：读盘方式不许让装载失败 ✗）。"""
        if self._unbuffered is None:
            from . import win_io

            wanted = win_io.windows_available()
        else:
            wanted = bool(self._unbuffered)
        if not wanted:
            return
        from . import win_io

        biggest = max((entry.nbytes for entry in self.entries.values()), default=0)
        slot = min(self._SLOT_CAP_BYTES, max(win_io.SECTOR_BYTES * 8, biggest + win_io.SECTOR_BYTES))
        try:
            raw = win_io.UnbufferedReader(self._path, slots=self._SLOTS, slot_bytes=slot,
                                          force_plain=False)
        except OSError as err:  # pragma: no cover - 真机上少见
            self._raw_error = f"无缓冲读口建不起来，退回 seek+read：{err}"
            return
        if raw.describe()["mode"] != "unbuffered":
            self._raw_error = raw.describe().get("error") or "无缓冲路径不可用，退回 seek+read"
            raw.close()
            return
        self._raw = raw

    @property
    def ioMode(self) -> str:
        """实际在用哪条读路 ✓（``unbuffered`` / ``plain`` / ``closed`` ✓）—— 要能被报出来 ✓✗。"""
        if self._closed:
            return "closed"
        return "unbuffered" if self._raw is not None else "plain"

    @property
    def ioNote(self) -> str:
        """降级原因 ✓（空＝没降级 ✓✗：「读得慢还查不出为什么」的坑就在这句 ✓）。"""
        return self._raw_error

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    @property
    def header_bytes(self) -> int:
        """头部（含 8 字节长度前导）占多少字节 ✓ —— 「只读头部就判完」这句话的**成本证据** ✓。"""
        return self._data_start

    def meta(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """``{名字: (dtype 字符串, 形状)}`` ✓ —— **只有元信息** ✓，正是体检/预检要的形态 ✓。"""
        return {name: (entry.dtype, tuple(entry.shape)) for name, entry in self.entries.items()}

    def read(self, name: str) -> bytes:
        """读**一个**张量的原始字节 ✓（不解释、不转张量、不缓存 ✓）。"""
        entry = self._entry(name)
        if self._raw is not None:
            return self._raw.read(self._data_start + entry.begin, entry.nbytes)
        handle = self._ensure_open()
        handle.seek(self._data_start + entry.begin)
        payload = handle.read(entry.nbytes)
        if len(payload) != entry.nbytes:
            raise SafetensorsError(
                f"张量 {name!r} 只读到 {len(payload)}/{entry.nbytes} 字节 ⇒ **文件被截断** ✗")
        return payload

    def read_many(self, names: Sequence[str]) -> list[bytes]:
        """一次读**多段** ✓（顺序与入参一致 ✓）—— 只是把多笔提交给读口，不改语义 ✓✗。"""
        entries = [self._entry(name) for name in names]
        if self._raw is None:
            return [self.read(name) for name in names]
        return list(self._raw.read_many(
            [(self._data_start + entry.begin, entry.nbytes) for entry in entries]))

    def _entry(self, name: str) -> TensorEntry:
        entry = self.entries.get(name)
        if entry is None:
            raise SafetensorsError(f"没有张量 {name!r}（可用：{len(self.entries)} 个 ✓）")
        return entry

    def _ensure_open(self) -> Any:
        if self._closed:
            raise SafetensorsError("读口已关闭 ⇒ 不能再读 ✓✗（静默重开会把「关干净了没」这件事弄丢 ✗）")
        if self._handle is None:
            self._handle = self._path.open("rb")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._raw is not None:
            self._raw.close()
            self._raw = None
        self._closed = True

    def __enter__(self) -> "TensorReader":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.close()
        return False
