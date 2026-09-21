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
from typing import Any

__all__ = [
    "DTYPE_ITEMSIZE",
    "SafetensorsError",
    "SafetensorsInfo",
    "TensorEntry",
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
