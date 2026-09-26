"""``GGUF`` **读取器**（纯 Python，零依赖 ✓）—— 引擎权重体检层的第二容器。

## 为什么自己实现（与 :mod:`.safetensors` 同一条理由 ✓）

GGUF 容器**只是**「类型化的 KV 头 + 张量表 + 对齐数据区」✓ —— 读它只需要
``struct`` + ``seek`` ✓。而 ``configs/models.json`` 里的路线 B（H3 GGUF Q4_K_M ✓）与
Wan 2.6 GGUF 都是这一格式 ✗（此前 ``loader``/``inventory`` 对 ``.gguf`` 只能报
「在/不在 + 大小」✗ ``quantScheme='gguf-unknown'`` ✗ —— 本模块把这层补齐 ✓）。

## 头部格式（事实来源：``ollama/fs/gguf/gguf.go`` ✓ 2026-09-20 读全 ✓）

::

    [0:4]   magic：``GGUF``（小端 ✓）或 ``FUGG``（大端 ✓）
    [4:8]   u32 版本（≥1 ✓；字符串/张量表的长度一律 u64 ✓ —— 与 ollama 读取器一致 ✓）
    [8:16]  u64 张量个数
    [16:24] u64 KV 个数
    [24:]   KV × N：key（u64 长度 + 字节 ✓）+ 值类型 u32 + 值（见 ``_KV_SCALARS`` ✓）
            ⚠️ 版本 1 的字符串**带 null 终止**（长度含终止符 ✓ ⇒ 读侧要剥掉并校验 ✓）
    [..]    张量信息 × N：name（u64 长度 ✓）+ n_dims u32 + dims × n_dims（u64 ✓）
            + 类型 u32 + 数据区相对偏移 u64
    [..]    数据区起点 = 张量表结束处**向上对齐**（``general.alignment``，缺省 32 ✓）

## 字节数怎么核（事实来源：``tensor.go`` 的 typeSize/blockSize 公式 ✓）

每个张量 ``nbytes = (元素总数 ÷ 块大小) × 每块字节数`` ✓，且**第 0 维必须整除块大小**
✓（不整除 ⇒ 装不上 ✗）。公式算出的每块字节数全部**预计算成常量表** ``TENSOR_TYPES``
（Q4_K = 2+2+12+256/2 = **144** ✓、Q6_K = **210** ✓ …），与 ollama 逐一对齐 ✓。

## 量化方案名（事实来源：``file_type.go`` ✓）

GGUF 元数据 ``general.file_type``（0=F32、15=**Q4_K_M** ✓ …）是**打包命名**的权威 ✓
（张量表里只有 Q4_K/Q6_K 这类**块类型**，"M/S/L" 混合方案名只在 file_type 里 ✓）。

## ⚠️ 本模块**不做反量化/不提权重** ✗（那需要 llama.cpp/ComfyUI-GGUF ✓）
—— 与 :mod:`.safetensors` 同界：只给「清单 + 形状 + 字节 + 方案名」✓。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

__all__ = [
    "FILE_TYPES",
    "TENSOR_TYPES",
    "GgufError",
    "GgufInfo",
    "GgufTensor",
    "inspect",
    "quant_scheme_of",
    "tensor_nbytes",
]

#: magic（小端文件 ✓；``FUGG`` 是大端镜像 ✓ —— 两种都认，与 ollama 一致 ✓）
MAGIC_LITTLE = b"GGUF"
MAGIC_BIG = b"FUGG"

#: KV 标量值类型 → ``(struct 格式, 字节数)``（GGUF 规范 0..12 ✓，无 3 字节整型 ✓；
#: 格式串**不带字节序前缀** ✗ —— 字节序统一由 ``_Cursor.prefix`` 拼 ✓）
_KV_SCALARS: dict[int, tuple[str, int]] = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("B", 1),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
_KV_STRING = 8
_KV_ARRAY = 9

#: 张量类型 id → ``(名字, 块大小, 每块字节数)`` —— 事实来源：ollama ``tensor.go`` ✓
#: （未列出的 id：Q4_2/Q4_3、Q4_0_4_x、TQ1_0/TQ2_0、IQ4_NL_4_x —— ollama 同样**没有**
#: 块布局 ⇒ 遇到只能报「未知块布局」，不猜 ✗）
TENSOR_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18),
}

#: ``general.file_type`` → 方案名（事实来源：ollama ``file_type.go`` ✓ —— 15 = Q4_K_M ✓）
FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_F16", 7: "Q8_0",
    8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}

#: 防呆上限（坏文件读出来的个数可能是天文数字 ✗）—— 与 ollama 的常量对齐 ✓
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_ARRAY_ITEMS = 64 * 1024 * 1024
MAX_KV_COUNT = 1_000_000
MAX_TENSOR_COUNT = 1_000_000
MAX_TENSOR_DIMS = 4

#: 张量名 → 「量化配套」张量（GGUF 量化块的 scales/zeros 就存在块里 ✓，
#: 但社区导出偶有独立 scale 表 ⇒ 这类后缀视为**配套**，不算缺 ✗）
DEFAULT_ALIGNMENT = 32


class GgufError(RuntimeError):
    """文件不是合法 GGUF（或与头部自述不一致 ✓）。"""


@dataclass
class GgufTensor:
    """张量表里的一行（**只有元信息，没有数据** ✓）。"""

    name: str
    type_id: int
    type_name: str
    shape: list[int]
    offset: int            # 相对数据区起点的偏移 ✓
    nbytes: int            # 按块布局算出的字节数 ✓（未知块布局 ⇒ ``-1`` ✗）

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


@dataclass
class GgufInfo:
    """一次体检的结论 ✓（``ok=False`` 时看 ``problems`` ✓）。"""

    path: str
    file_bytes: int = 0
    byte_order: str = "little"
    version: int = 0
    alignment: int = DEFAULT_ALIGNMENT
    data_start: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, GgufTensor] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    dtype_counts: dict[str, int] = field(default_factory=dict)
    quant_scheme: str = "unknown"

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    def biggest(self, count: int = 8) -> list[GgufTensor]:
        """最大的几个张量（看结构时最有用 ✓）。"""
        return sorted(self.tensors.values(), key=lambda item: item.nbytes, reverse=True)[:count]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "fileBytes": self.file_bytes,
            "byteOrder": self.byte_order,
            "version": self.version,
            "alignment": self.alignment,
            "dataStart": self.data_start,
            "tensorCount": self.tensor_count,
            "dtypeCounts": self.dtype_counts,
            "quantScheme": self.quant_scheme,
            "metadata": self.metadata,
            "problems": self.problems,
            "biggest": [{"name": t.name, "type": t.type_name, "shape": t.shape,
                         "bytes": t.nbytes} for t in self.biggest()],
        }


# ── 底层读原语（带位置游标 ✓；长度一律 u64 —— 与 ollama 读取器一致 ✓）─────────

class _Cursor:
    """顺序读游标 ✓（记录位置 ⇒ 「数据区起点」不再自己数 ✗）。"""

    def __init__(self, handle: BinaryIO, *, big: bool, version: int) -> None:
        self._handle = handle
        self.prefix = ">" if big else "<"
        self.version = int(version)
        self.offset = 0

    def _unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        blob = self._handle.read(size)
        if len(blob) != size:
            raise GgufError(f"读到 {self.offset} 就没了（文件被截断 ✗）")
        self.offset += size
        return struct.unpack(self.prefix + fmt, blob)

    def u32(self) -> int:
        return int(self._unpack("I")[0])

    def u64(self) -> int:
        return int(self._unpack("Q")[0])

    def string(self) -> str:
        length = self.u64()
        if length > MAX_STRING_BYTES:
            raise GgufError(f"字符串长度 {length} 超上限 {MAX_STRING_BYTES} ⇒ 文件损坏 ✗")
        blob = self._handle.read(length)
        if len(blob) != length:
            raise GgufError(f"字符串声明 {length} 字节但读不到 ⇒ 文件被截断 ✗")
        self.offset += length
        if self.version == 1:
            # ⚠️ v1：长度**含 null 终止符** ⇒ 末字节必须是 0 ✓（ollama 同款校验 ✓）
            if not blob or blob[-1] != 0:
                raise GgufError("版本 1 的字符串缺 null 终止符 ✗")
            blob = blob[:-1]
        return blob.decode("utf-8", errors="replace")

    def raw(self, size: int) -> bytes:
        blob = self._handle.read(size)
        if len(blob) != size:
            raise GgufError(f"要跳 {size} 字节但读不到 ⇒ 文件被截断 ✗")
        self.offset += size
        return blob


def _read_kv_value(cursor: _Cursor, value_type: int) -> Any:
    """读一个 KV 值 ✓ —— 数组**截断保存**（元数据只做摘要 ✓ 不背 64 MiB 进内存 ✗）。"""
    if value_type == _KV_STRING:
        return cursor.string()
    if value_type == _KV_ARRAY:
        element_type = cursor.u32()
        count = cursor.u64()
        if count > MAX_ARRAY_ITEMS:
            raise GgufError(f"数组元素数 {count} 超上限 {MAX_ARRAY_ITEMS} ⇒ 文件损坏 ✗")
        if element_type == _KV_STRING:
            keep = min(count, 16)
            items = [cursor.string() for _ in range(keep)]
            for _ in range(count - keep):
                cursor.string()   # 大数组也要**读完**（位置才对 ✓），只是不存 ✗
            return {"__array__": "string", "len": count, "head": items}
        if element_type not in _KV_SCALARS:
            raise GgufError(f"数组的元素类型 {element_type} 不认识 ✗")
        fmt, size = _KV_SCALARS[element_type]
        total_bytes = int(count) * size
        # 大数组：读掉但只解析头部一小段 ✓（tokenizer 词表动辄几十万元素 ✓）
        head_bytes = cursor.raw(min(total_bytes, 64 * size))
        cursor.raw(max(0, total_bytes - len(head_bytes)))
        head = [struct.unpack(cursor.prefix + fmt, head_bytes[i * size:(i + 1) * size])[0]
                for i in range(len(head_bytes) // size)]
        return {"__array__": fmt, "len": count, "head": head[:16]}
    if value_type in _KV_SCALARS:
        fmt, _size = _KV_SCALARS[value_type]
        value = cursor._unpack(fmt)[0]
        return bool(value) if value_type == 7 else value
    raise GgufError(f"KV 值类型 {value_type} 不认识 ✗")


def tensor_nbytes(type_id: int, shape: list[int]) -> int:
    """按块布局算字节数 ✓（块不整除 ⇒ ``-1`` ✗；未知类型 ⇒ ``-1`` ✗）。"""
    layout = TENSOR_TYPES.get(int(type_id))
    if layout is None:
        return -1
    _name, block_size, block_bytes = layout
    numel = 1
    for dim in shape:
        numel *= int(dim)
    row = int(shape[0]) if shape else 1
    if row % block_size:
        return -1
    return (numel // block_size) * block_bytes


def quant_scheme_of(metadata: dict[str, Any], dtype_counts: dict[str, int]) -> str:
    """方案名 ✓：``general.file_type`` 是权威 ✓；缺了再从张量类型分布推 ✓（推不出说 ``none`` ✓）。"""
    file_type = metadata.get("general.file_type")
    if isinstance(file_type, int) and file_type in FILE_TYPES:
        return FILE_TYPES[file_type]
    quants = {name: count for name, count in dtype_counts.items()
              if name not in ("F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64")}
    if quants:
        return max(quants, key=lambda name: quants[name])
    if dtype_counts:
        return "none"
    return "unknown"


# ── 主入口 ─────────────────────────────────────────────────────────────────

def inspect(path: str | Path) -> GgufInfo:
    """完整体检 ✓：头部 + 张量表 + 「每个张量声明的大小是否真的在文件里」✓。

    **不做大数据读取** ✓（头部 + 张量表通常在**头几十 KB** ✓）⇒ 对 10 GiB 的
    GGUF 也是毫秒级 ✓ —— 与 :func:`app.services.engine.safetensors.inspect` 同界 ✓。
    """
    target = Path(path)
    info = GgufInfo(path=str(target))
    try:
        if not target.exists():
            raise GgufError(f"文件不存在：{target}")
        if not target.is_file():
            raise GgufError(f"不是普通文件：{target}")
        info.file_bytes = target.stat().st_size
        if info.file_bytes < 24:
            raise GgufError(f"文件太小（{info.file_bytes} 字节）⇒ 不是 GGUF ✗")
        with target.open("rb") as handle:
            _inspect_into(info, handle)
    except (GgufError, OSError, struct.error) as err:
        info.problems.append(str(err))
    return info


def _inspect_into(info: GgufInfo, handle: BinaryIO) -> None:
    magic = handle.read(4)
    if magic == MAGIC_LITTLE:
        info.byte_order = "little"
    elif magic == MAGIC_BIG:
        info.byte_order = "big"
    else:
        raise GgufError(f"magic {magic!r} 不是 GGUF/FUGG ⇒ 不是 GGUF 文件 ✗")

    cursor = _Cursor(handle, big=(info.byte_order == "big"), version=0)
    cursor.offset = 4  # ⚠️ magic 已直接读过 4 字节 ⇒ 游标补上 ✓（否则 data_start 偏 4 ✗）
    info.version = cursor.u32()
    if info.version < 1:
        raise GgufError(f"版本号 {info.version} < 1 ⇒ 文件损坏 ✗")
    cursor.version = info.version
    tensor_count = cursor.u64()
    kv_count = cursor.u64()
    if tensor_count > MAX_TENSOR_COUNT:
        raise GgufError(f"张量个数 {tensor_count} 超上限 ⇒ 文件损坏 ✗")
    if kv_count > MAX_KV_COUNT:
        raise GgufError(f"KV 个数 {kv_count} 超上限 ⇒ 文件损坏 ✗")

    # ── KV 段 ✓（元数据摘要：标量/短字符串全存 ✓，数组只存头部 ✓）
    for _ in range(kv_count):
        key = cursor.string()
        value_type = cursor.u32()
        info.metadata[key] = _read_kv_value(cursor, value_type)

    # ── 张量表 ✓
    for _ in range(tensor_count):
        name = cursor.string()
        n_dims = cursor.u32()
        if n_dims > MAX_TENSOR_DIMS:
            raise GgufError(f"张量 {name!r} 维数 {n_dims} > {MAX_TENSOR_DIMS} ⇒ 文件损坏 ✗")
        dims = [cursor.u64() for _ in range(n_dims)]
        type_id = cursor.u32()
        offset = cursor.u64()
        layout = TENSOR_TYPES.get(type_id)
        type_name = layout[0] if layout else f"UNKNOWN({type_id})"
        nbytes = tensor_nbytes(type_id, dims)
        info.tensors[name] = GgufTensor(name=name, type_id=type_id, type_name=type_name,
                                        shape=[int(dim) for dim in dims],
                                        offset=offset, nbytes=nbytes)
        info.dtype_counts[type_name] = info.dtype_counts.get(type_name, 0) + 1
        if layout is None:
            info.problems.append(f"张量 {name!r} 的类型 {type_id}（{type_name}）没有块布局"
                                 f"⇒ 算不出字节数、核不了截断 ✗")
        elif nbytes < 0:
            info.problems.append(f"张量 {name!r} 的第 0 维 {dims[0] if dims else 1}"
                                 f"不整除块大小 {layout[1]} ⇒ 装载器会拒载 ✗")

    # ── 数据区起点 = 张量表结束处向上对齐 ✓（general.alignment，缺省 32 ✓）
    alignment_value = info.metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    if not isinstance(alignment_value, int) or alignment_value <= 0:
        alignment_value = DEFAULT_ALIGNMENT
    info.alignment = alignment_value
    info.data_start = cursor.offset + (-cursor.offset) % alignment_value

    # ── 截断 / 越界 / 重叠 ✓（这是「跑之前发现」的核心 ✓）
    spans: list[tuple[int, int, str]] = []
    for tensor in info.tensors.values():
        if tensor.nbytes < 0:
            continue
        begin = info.data_start + tensor.offset
        end = begin + tensor.nbytes
        if end > info.file_bytes:
            info.problems.append(
                f"张量 {tensor.name!r} 需要 [{begin}, {end}) 但文件只有 {info.file_bytes} 字节"
                f"⇒ **文件被截断** ✗")
        spans.append((begin, end, tensor.name))
    spans.sort()
    for (_b1, e1, n1), (b2, _e2, n2) in zip(spans, spans[1:]):
        if b2 < e1:
            info.problems.append(f"张量 {n1!r} 与 {n2!r} 的数据区**重叠** ⇒ 文件被拼错 ✗")

    if not info.tensors:
        info.problems.append("张量表是空的 ✗")
    info.quant_scheme = quant_scheme_of(info.metadata, info.dtype_counts)
