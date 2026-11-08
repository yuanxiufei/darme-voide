"""**物化缓存**（G3 ✓ 2026-11-07）：把「反量化/升精度算出来的字节」按源指纹落到盘上 ✓。

## 它解决什么

`gguf_to_llm` 装一份 Q4_K_M 的 4B 文本塔时，**每次装载**都要把 398 个张量从 4-bit 反量化成
fp32 再 cast 成 bf16 ✓ —— 实测 `Q4_K` 向量化后 **100.6 M 元素/秒**（`gguf_dequant` 头部记的
真机数 ✓），整份约 **31 秒**，而这段 CPU 时间**每次进程启动都要再付一遍** ✗✗。
物化缓存把「反量化 ＋ cast」的**结果字节**写成一份正常的 safetensors ✓ ⇒ 第二次起只是**读盘**
（8 GiB 级顺序读 ≈ 数秒 ✓）⇒ 省的是 CPU 与**峰值内存**（不再需要 fp32 中间量 ✓）✓。

## 三条纪律（都是「错了会静默」的形状 ✓✗）

1. ⭐⭐ **只加速装载、不改数值** ✗✗：物化的 dtype **就是装载目标的 dtype** ✓（`bf16` 就存 bf16 ✓），
   而且写盘的字节取自**真装进模型的那张 torch 张量的原始字节** ✓
   （`gguf_to_llm._tensor_bytes` 的 ``.view(torch.int16).numpy().tobytes()`` ✓）⇒
   「逐位相同」是**构造**出来的 ✓，不是靠两条公式对拍出来的 ✓✗。
   ⚠️ 自己写「fp32 取高 16 位」**不行** ✗：那是**向零舍入** ✓✗（torch 的 ``.to(bfloat16)`` 是 RNE ✓）
   ⇒ 差 1 ulp ⇒ 命中的模型与不命中的模型**不是同一个模型** ✓✗ —— 而判据正是
   「命中装出来的权重与不命中**逐位相等**」✓，1 ulp 当场红 ✓。
2. ⭐⭐ **键 = 源内容指纹 ＋ 目标 dtype ＋ 转换器 ＋ 格式版本** ✗✗：四者任一变了就**换一个文件名** ✓
   （不是「命中后再比一遍」✗ —— 文件名不同 ⇒ 根本不会撞上 ✓）。指纹复用
   :func:`cache_guard.file_fingerprint` 的 ``(大小, mtime, 头 64KB 哈希)`` ✓，但**只取大小与头哈希**
   ✓✗：⚠️ 只有 size+mtime 时，「保留时间戳的重新下载」会被误判成「没变」✓✗（上游 v3.5.5 的教训 ✓）；
   ⚠️ 而把 mtime 塞进键也不行 ✗✗：「``touch`` 一下换一个文件名 ⇒ 多留一份 8 GiB」✓✗。
   `converter` 是**转换逻辑的版本号**（公式改了就换 ✓），格式版本一起进哈希 ✓。
3. ⭐ **写失败一律退化成「慢」** ✓✗：余量不足 / 超配额 / 没权限 / 写了一半 ⇒ **拒写并回退重算** ✓
   （`MaterializeSkip` ✓ 不抛给装载层 ✗），**一个字节都不留给下次** ✗（``.tmp`` 必删 ✓、
   ``os.replace`` 保证别人看不到半成品 ✓）。sidecar 写失败**不阻塞** ✓（没有指纹 = 下次重算 ✓）。

## 与邻居的分工（别混 ✗）

* :mod:`cache_guard`：**判**「产物自证 + 源指纹」✓（本模块只调它 ✓，不另写一套 ✓✗）；
* :mod:`hybrid_load`：**搬字节**（基底 + 覆盖层 → 合并产物 ✓）—— 它搬的是**原字节**，不做 dtype 转换 ✗；
* **本模块**：**算出来的字节**落盘 ✓（键、原子写、余量、配额、命中账 ✓），**纯 stdlib** ✓
  （不 import torch ✗ / 不 import numpy ✗ —— 数值转换在调用方 ✓，本模块只认 bytes ✓）。

## 不做的事（边界）

* 不做 LRU 淘汰 ✗：超配额只**拒新写** ✓，不替用户删旧产物 ✓（删权重类文件得是人决定的 ✓）；
* 不校验张量**数值** ✗（和 `cache_guard` 同一条边界 ✓：头自洽 ≠ 数值对 ✓ —— 数值由
  「命中/不命中逐位相等」的判据在真装载上保证 ✓）；
* 不缓存「源文件本身就是目标 dtype」的情形 ✗（那等于把权重原样抄一份 ✓✗）：调用方
  （:func:`gguf_to_llm._plan_specs`）先看**整份文件有没有量化张量** ✓ —— 一个都没有（全 fp32/fp16）
  ⇒ 整份不物化 ✓ 并带一条 ``notes`` 说清 ✓✗（**不许**只物化一部分 ✗：半份产物意味着装载循环要
  维护两套账 ✓✗）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import cache_guard as cg
from . import safetensors as st

__all__ = ["DISK_HEADROOM", "MATERIALIZE_SCHEMA", "MaterializeError", "MaterializeSkip",
           "Materializer", "Report", "Spec", "cache_report", "default_directory", "enabled",
           "free_bytes", "gib", "limit_gib", "reset_stats", "stats", "store_usage"]

#: 产物格式版本 ✓（布局/元数据键改了就要 +1 ✓ —— 它进文件名哈希 ⇒ 旧产物自动不复用 ✓）。
MATERIALIZE_SCHEMA = 1
#: 写盘前的余量系数 ✓（留 5% 给头 / 文件系统元数据 ✓，与 :data:`hybrid_load.DISK_HEADROOM` 同口径 ✓）。
DISK_HEADROOM = 1.05
#: 产物后缀 ✓（**必须**是 safetensors ✓：读取侧直接用 :class:`safetensors.TensorReader` ✓）。
SUFFIX = ".safetensors"
#: 元数据里那条自证键 ✓。
META_KEY = "darme_materialize"
#: 环境变量：产物目录 ✓ / 配额 ✓ / 总开关 ✓。
ENV_DIR = "DARME_MATERIALIZE_DIR"
ENV_LIMIT = "DARME_MATERIALIZE_LIMIT_GIB"
ENV_SWITCH = "DARME_MATERIALIZE"


class MaterializeError(RuntimeError):
    """**用错了** ✓（计划与实写不一致 / 字节数不符 ✓）—— 这是编程错误 ⇒ **当场抛** ✗。"""


class MaterializeSkip(RuntimeError):
    """**这次不写** ✓（余量不足 / 超配额 / 没权限 ✓）—— 环境原因 ⇒ 退化成重算 ✓ **不抛给装载层** ✗。"""


@dataclass(frozen=True)
class Spec:
    """计划里的一个张量 ✓：名字（**源文件里的名字** ✓）、形状、目标 dtype、字节数 ✓。"""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    @staticmethod
    def of(name: str, shape: Iterable[int], dtype: str) -> "Spec":
        dims = tuple(int(dim) for dim in shape)
        size = cg.DTYPE_SIZE.get(dtype)
        if size is None:
            raise MaterializeError(
                f"目标 dtype {dtype!r} 不在 :data:`cache_guard.DTYPE_SIZE` 里 ✗ ⇒ 产物头无法自证 ✗"
                f"（可用：{sorted(cg.DTYPE_SIZE)} ✓）")
        count = 1
        for dim in dims:
            count *= dim
        return Spec(name=str(name), shape=dims, dtype=str(dtype), nbytes=count * size)


@dataclass(frozen=True)
class Report:
    """一次「命中 / 物化 / 放弃」的结论 ✓（``hit=True`` 时 ``reason`` 说明**凭什么**命中 ✓）。"""

    hit: bool
    path: str
    reason: str
    gib: float = 0.0
    written: bool = False
    tensor_count: int = 0
    sidecar_written: bool = False
    sidecar_error: str | None = None
    elapsed_ms: float = 0.0
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {"hit": self.hit, "path": self.path, "reason": self.reason,
                "gib": round(float(self.gib), 4), "written": self.written,
                "tensorCount": int(self.tensor_count),
                "sidecarWritten": self.sidecar_written, "sidecarError": self.sidecar_error,
                "elapsedMs": round(self.elapsed_ms, 3), "notes": list(self.notes)}


def gib(value: float) -> float:
    """字节 → GiB ✓（**一处**换算 ✓：两个地方各写一遍 ``/1024**3`` ⇒ 小数位迟早对不上 ✓✗）。"""
    return float(value) / 1024.0 ** 3


def enabled() -> bool:
    """总开关 ✓（``DARME_MATERIALIZE=0/off/false/no`` ⇒ 关 ✓；**默认开** ✓）。"""
    raw = (os.environ.get(ENV_SWITCH) or "").strip().lower()
    return raw not in {"0", "off", "false", "no", "disable", "disabled"}


def limit_gib() -> float | None:
    """配额 ✓（``None`` = 不限 ✓；读不懂的写法 ⇒ 当**不限** ✓ 而不是当 0 ✗ —— 0 会静默关掉整个功能 ✗）。"""
    raw = (os.environ.get(ENV_LIMIT) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def default_directory() -> Path:
    """产物目录 ✓：环境变量 → ``get_data_root()/materialized`` → ``~/.darme/materialized`` ✓。

    ⚠️ **不放在权重目录旁边** ✗✗：ComfyUI 会递归扫 ``models/**`` ✓，一份 8 GiB 的
    ``*.safetensors`` 掉进去会被当成模型列出来 ✓✗（而且用户会真去选它 ✓）。
    """
    raw = (os.environ.get(ENV_DIR) or "").strip()
    if raw:
        return Path(raw).expanduser()
    try:                                  # 引擎允许依赖 app.core.* ✓（见 engine_independence_test ✓）
        from ...core import config as core_config  # noqa: PLC0415

        getter = getattr(core_config, "get_data_root", None)
        if callable(getter):
            return Path(getter()) / "materialized"
    except Exception:  # noqa: BLE001 —— 脱离项目单独跑引擎时没有 core ✓ 不是错 ✓
        pass
    return Path.home() / ".darme" / "materialized"


def free_bytes(path: str | Path) -> int | None:
    """所在盘的可用字节 ✓（目录还不存在就往上找第一个存在的祖先 ✓）；读不到 ⇒ ``None`` ✓。"""
    target = Path(path)
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return None


#: 进程内的**命中账** ✓（§4.4 ✓ 2026-11-07 加 ✓）—— 模块级计数器 ✓（一次装载一份报告 ✗ 看不出
#: "这一程一共省了几次" ✓；`/engine/runtime` 的 `diskCache` 读的就是这里 ✓）。
#:
#: 五个数各指一件事 ✓✗（合成一个"命中率"会立刻读错 ✓）：
#: ``hits`` 直接复用了几次 ✓ / ``misses`` 判了不能用（没有或过期 ✓）/ ``written`` 真写成了几次 ✓ /
#: ``refused`` **一个字节都没写**就放弃（余量/配额/没权限 ✓）/ ``failed`` 写到一半失败 ✓。
_STATS: dict[str, Any] = {"hits": 0, "misses": 0, "written": 0, "refused": 0, "failed": 0,
                          "hitBytes": 0, "writeBytes": 0, "lastReason": ""}


def _bump(name: str, amount: int = 1, *, reason: str | None = None) -> None:
    """记一笔 ✓（**不许**在这里抛 ✗：记账失败把装载打挂是本末倒置 ✓✗）。"""
    try:
        _STATS[name] = int(_STATS.get(name) or 0) + int(amount)
        if reason is not None:
            _STATS["lastReason"] = str(reason)
    except Exception:  # noqa: BLE001
        pass


def reset_stats() -> None:
    """清零 ✓（测试与"这一程重新算"用 ✓）。"""
    for key in list(_STATS):
        _STATS[key] = "" if key == "lastReason" else 0


def stats() -> dict[str, Any]:
    """进程内的命中账 ✓ ⇒ ``{hits, misses, written, refused, failed, hitGiB, writeGiB, lastReason}``。"""
    return {
        "hits": int(_STATS["hits"]), "misses": int(_STATS["misses"]),
        "written": int(_STATS["written"]), "refused": int(_STATS["refused"]),
        "failed": int(_STATS["failed"]),
        #: 省下来的量 ✓（``hits`` 那次产物多大 ✓）与真写下去的量 ✓ —— **两个数**，别合成一个 ✗
        #: ⚠️ 精度 6 位（≈1 KB ✓）不是 3 位 ✗：3 位粒度是 1 MB，几 MB 的缓存会一律显示 ``0.0``
        #: ✓✗（"有产物但读成 0" 比不显示更糟 ✓）。
        "hitGiB": round(gib(int(_STATS["hitBytes"])), 6),
        "writeGiB": round(gib(int(_STATS["writeBytes"])), 6),
        "lastReason": str(_STATS["lastReason"]),
        "note": ("`hits` = 直接复用的次数 ✓、`misses` = 判了不能用 ✓、`refused` = 余量/配额不足"
                 "**一个字节都没写** ✓、`failed` = 写到一半失败 ✓（后两个都只是「慢」✓ 不是错 ✓）"),
    }


def store_usage(directory: str | Path | None = None) -> dict[str, Any]:
    """盘上**占了多少** ✓ ⇒ ``{dir, enabled, files, sidecars, diskGiB, limitGiB, freeGiB}``。

    ⚠️ 与 :func:`stats` 是**两件事** ✓✗：那个说"复用了几次"（次数 ✓），这个说"现在压着多少字节"
    （体积 ✓）—— 合成一格会得出"复用越多占越多" ✓✗（恰好相反 ✓）。
    ⚠️ 只数**产物**（``*.safetensors`` ✓）与 sidecar 的**个数** ✓，sidecar 字节并进 ``diskGiB`` 里
    **不算**（它们才几 KB ✓ 分栏列出只是为了让"目录里有一堆 json"不显得像 bug ✓）。
    """
    root = Path(directory) if directory is not None else default_directory()
    files = 0
    sidecars = 0
    total = 0
    try:
        for entry in root.glob("*" + SUFFIX):
            try:
                total += int(entry.stat().st_size)
            except OSError:
                continue
            files += 1
        sidecars = sum(1 for _ in root.glob("*.meta.json"))
    except OSError:
        pass
    free = free_bytes(root)
    return {
        "dir": str(root), "enabled": enabled(), "files": files, "sidecars": sidecars,
        "diskGiB": round(gib(total), 6),
        #: ``None`` = **不限** ✓（不是"限了 0"✗ —— 0 会读成"一个字节都存不下"✓✗）
        "limitGiB": limit_gib(),
        "freeGiB": None if free is None else round(gib(free), 2),
    }


def cache_report(directory: str | Path | None = None) -> dict[str, Any]:
    """对外**一格**回答三件事 ✓（§4.4 / §5.2 的 `diskCache` ✓）：盘上多大 ✓、限多少 ✓、复用了几次 ✓。"""
    return {**store_usage(directory), **stats()}


def _head_bytes(path: Path, *, limit: int = cg.HEAD_LIMIT) -> bytes:
    """只读 **8 字节头长 + 头** ✓（产物可能 8 GiB ✗ 不能整读 ✓）。

    ⚠️ 头长是**产物自述**的 ✓ ⇒ 坏产物会自述 4 GiB 头 ✓✗ ⇒ 读之前先夹一道
    :data:`cache_guard.HEAD_LIMIT` ✓：超了就只回 8 字节，让 :func:`cache_guard.is_sane_head`
    按「头被截断」拒掉 ✓（**不**照它说的字节数去 malloc ✗）。
    """
    import struct  # noqa: PLC0415 —— 只在读头时用 ✓

    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) != 8:
            return head
        length = struct.unpack("<Q", head)[0]
        if not 0 < length <= int(limit):
            return head
        return head + handle.read(length)


class Materializer:
    """一个「源文件 × 目标 dtype × 转换器」的物化缓存把手 ✓（构造即定键 ✓，命中判定不碰大文件 ✓）。"""

    def __init__(self, source: str | Path, *, converter: str, dtype: str,
                 directory: str | Path | None = None, quota_gib: float | None = None) -> None:
        self.source = Path(source)
        self.converter = str(converter)
        self.dtype = str(dtype)
        self.directory = Path(directory) if directory is not None else default_directory()
        self.quota_gib = quota_gib

    # ---- 键与路径 ----

    def key(self) -> str:
        """键 ✓ = ``sha256(schema, converter, dtype, 源大小, 源头 64KB 哈希)`` 的前 12 位 ✓。

        ⚠️ **刻意不含 mtime** ✗✗：含了就会「``touch`` 一下 ⇒ 换一个文件名 ⇒ 多留一份 8 GiB 产物」✓✗
        （谁复制/解压权重都会动 mtime ✓ ⇒ 越用越涨 ✓）。mtime 仍在 **sidecar** 里 ✓
        （:func:`cache_guard.is_cache_valid` 会比三元组 ✓）⇒ 「只动了时间戳」的源**判失效 ⇒ 重算并覆盖
        同一个文件** ✓✗ —— 既不误用 ✓，也不增生 ✓。
        ⚠️ 源指纹**读不到** ⇒ :class:`MaterializeSkip` ✗（宁可每次重算 ✓，也不要拿一个
        「不知道对不对」的键去复用别人的产物 ✓✗）。
        """
        fingerprint = cg.file_fingerprint(self.source)
        if fingerprint is None:
            raise MaterializeSkip(f"源指纹读不到 ✗（{self.source.name} ✓ 被删 / 没权限 ✓）⇒ 不物化 ✓")
        payload = json.dumps({
            "schema": MATERIALIZE_SCHEMA, "converter": self.converter,
            "dtype": self.dtype, "size": int(fingerprint[0]), "head": str(fingerprint[2]),
        }, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def path(self) -> Path:
        """产物路径 ✓：``<目录>/<源文件名>.<converter>.<dtype>.<键>.safetensors`` ✓。

        名字里带**源文件名**只为肉眼可读 ✓（键才是判据 ✓）；带 ``converter``/``dtype`` 是为了
        「换目标精度」时两个产物能并存 ✓✗（否则得互相覆盖、来回重算 ✓）。
        """
        return self.directory / f"{self.source.name}.{self.converter}.{self.dtype}.{self.key()}{SUFFIX}"

    # ---- 命中判定 ----

    def valid(self) -> tuple[bool, str]:
        """产物能不能直接用 ✓ = **文件在** ＋ **头自证** ＋ **sidecar 源指纹** ＋ **元数据自述一致** ✓。

        ⚠️ 四条**缺一不可** ✗✗：少最后一条时，「同名的旧产物（换过 converter 但键算法没变 ✓）」
        会被当成有效 ✓✗ —— 那是「装载出来数值不对、但一路绿灯」的形状 ✓✗。

        结论顺手记进 :func:`stats` ✓（一次判定一笔 ✓ —— 装载侧每条路只调一次 ✓）。
        """
        hit, reason, size = self._valid()
        if hit:
            _bump("hits", 1, reason=reason)
            _bump("hitBytes", size)
        else:
            _bump("misses", 1, reason=reason)
        return hit, reason

    def _valid(self) -> tuple[bool, str, int]:
        """:meth:`valid` 的实现 ✓ ⇒ ``(能不能用, 凭什么, 产物字节数 ✓)``（第三个只给记账用 ✓）。"""
        target = self.path()
        if not target.exists():
            return False, "没有产物 ⇒ 物化一次 ✓", 0
        try:
            head = _head_bytes(target)
            total = target.stat().st_size
        except OSError as err:
            return False, f"产物读不了 ✗（{type(err).__name__}: {err}）⇒ 重算 ✓", 0
        if not cg.is_sane_head(head, total_size=total):
            return False, ("产物头**不自洽** ✗（截断 / 偏移与 dtype·形状不符 / 数据区有洞 ✓）"
                           "⇒ 重算 ✓✗（坏产物会被反复加载 ✓）"), 0
        try:
            import struct  # noqa: PLC0415

            count = struct.unpack("<Q", head[:8])[0]
            header = json.loads(head[8:8 + count].decode("utf-8"))
            digest = header.get("__metadata__", {}).get(META_KEY)
            meta = json.loads(digest) if isinstance(digest, str) else None
        except Exception:  # noqa: BLE001 —— 头都解析不了的产物 ⇒ 就是「不能用」✓ 不是崩 ✓
            return False, "产物元数据读不懂 ✗ ⇒ 重算 ✓", 0
        if not isinstance(meta, Mapping):
            return False, f"产物里没有 ``{META_KEY}`` 自述 ✗（不是本模块写的文件 ✓）⇒ 重算 ✓", 0
        if (int(meta.get("schema") or 0) != MATERIALIZE_SCHEMA
                or str(meta.get("converter") or "") != self.converter
                or str(meta.get("dtype") or "") != self.dtype):
            return False, ("产物自述与本次要的不一致 ✗（schema/converter/dtype ✓）"
                           f"⇒ 重算 ✓（产物说 {meta.get('converter')}/{meta.get('dtype')} ✓）"), 0
        sidecar = self._read_sidecar(target)
        if not cg.is_cache_valid(cache_bytes=head, total_size=total, sidecar=sidecar,
                                 sources={"source": cg.file_fingerprint(self.source)}):
            return False, ("sidecar 对不上 ✗（源变了 / sidecar 缺失或损坏 ✓）"
                           "⇒ 重算 ✓✗ —— 宁可慢一次，不拿可疑产物去跑 ✓"), 0
        return (True, "产物结构自证 ✓ + 元数据自述一致 ✓ + sidecar 源指纹一致 ✓ ⇒ 直接复用 ✓",
                int(total))

    def _read_sidecar(self, target: Path) -> Any:
        sidecar_path = cg.cache_sidecar_path(target)
        if not sidecar_path.exists():
            return None
        try:
            return json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— sidecar 坏了 = 没有指纹 ✓ 不是崩 ✓
            return None

    def reader(self) -> st.TensorReader:
        """读产物 ✓（``TensorReader``：只读头 ＋ 按名取字节 ✓，不整份进内存 ✓）。"""
        return st.TensorReader(self.path())

    # ---- 写 ----

    def open_writer(self, specs: Sequence[Spec]) -> "Writer":
        """开一个写手 ✓（余量/配额**先判** ✓ —— 一个字节都不写就拒 ✓✗）。"""
        return Writer(self, list(specs))

    def _write_sidecar(self, target: Path) -> tuple[bool, str | None]:
        """写源指纹 sidecar ✓ ⇒ ``(写没写成, 错因 ✓)`` —— **失败不阻塞** ✓（下次重算 ✓）。"""
        try:
            fingerprint = cg.file_fingerprint(self.source)
            if fingerprint is None:
                return False, "源指纹读不到 ✗（⇒ 下次会重算 ✓）"
            cg.cache_sidecar_path(target).write_text(
                json.dumps({"source": {"size": int(fingerprint[0]), "mtime": float(fingerprint[1]),
                                       "hash": str(fingerprint[2])}},
                           ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                encoding="utf-8")
            return True, None
        except Exception as err:  # noqa: BLE001
            return False, f"{type(err).__name__}: {err}（⚠️ 没写成 ⇒ 下次会**重算** ✓，不是错 ✓✗）"


class Writer:
    """按计划写产物 ✓：先算头与尺寸 ⇒ 判余量/配额 ⇒ 流式写数据 ⇒ 原子替换 ⇒ sidecar ✓。

    ⚠️ 与 :class:`hybrid_load` 的同一条纪律：**键集不许变** ✗✗ —— 计划里有几个张量就得写几个 ✓，
    少一个 ⇒ :class:`MaterializeError` 且**不留产物** ✓（加载侧会「缺 key 具名拒绝」✓，但那是
    装载时的运气 ✓✗；让残缺产物根本不落盘更省事 ✓）。
    """

    def __init__(self, owner: Materializer, specs: list[Spec]) -> None:
        self.owner = owner
        self.specs = specs
        self.target = owner.path()
        self.tmp: Path | None = None
        self._handle: Any = None
        self._table: dict[str, Any] = {}
        self._offsets: dict[str, int] = {}
        self._written: set[str] = set()
        self._order = [spec.name for spec in specs]
        self._started = time.perf_counter()
        self.bytes_written = 0
        self.report: Report | None = None
        self._prepare()

    # ---- 准备：头 + 余量 + 配额 ----

    def _refuse(self, message: str) -> None:
        """**一个字节都没写**就放弃 ✓ ⇒ 记账 ＋ 抛 :class:`MaterializeSkip` ✓（调用方退化成重算 ✓）。

        ⚠️ 记账必须在抛之前 ✓✗：「放弃了但没留下任何痕迹」正是这一格要防的 ✓——
        用户看到「装得慢」却不知道是磁盘只剩 2 GiB ✓。
        """
        _bump("refused", 1, reason=message)
        raise MaterializeSkip(message)

    def _header(self) -> bytes:
        offset = 0
        for spec in self.specs:
            if spec.name in self._table:
                raise MaterializeError(f"计划里 {spec.name!r} 出现了两次 ✗")
            self._offsets[spec.name] = offset
            self._table[spec.name] = {"dtype": spec.dtype, "shape": list(spec.shape),
                                      "data_offsets": [offset, offset + int(spec.nbytes)]}
            offset += int(spec.nbytes)
        meta = {META_KEY: json.dumps({
            "schema": MATERIALIZE_SCHEMA, "converter": self.owner.converter,
            "dtype": self.owner.dtype, "source": self.owner.source.name,
            "tensorCount": len(self.specs), "dataBytes": offset,
        }, ensure_ascii=False, separators=(",", ":"), sort_keys=True)}
        header = json.dumps({**self._table, "__metadata__": meta}, ensure_ascii=False,
                            separators=(",", ":"), sort_keys=True).encode("utf-8")
        return header

    def _prepare(self) -> None:
        header = self._header()
        self.header = header
        self.data_bytes = sum(int(spec.nbytes) for spec in self.specs)
        self.total_bytes = 8 + len(header) + self.data_bytes
        quota = self.owner.quota_gib
        if quota is not None and gib(self.total_bytes) * DISK_HEADROOM > float(quota):
            self._refuse(
                f"超配额 ✗：产物约 {gib(self.total_bytes):.2f} GiB（含余量 ✓）、"
                f"配额 {float(quota):.2f} GiB ✓ ⇒ **不写** ✓ 本次改为重算（慢但不错 ✓）"
                f"（配额由 {ENV_LIMIT} 控制 ✓）")
        free = free_bytes(self.target)
        if free is not None and free < int(self.total_bytes * DISK_HEADROOM):
            self._refuse(
                f"磁盘余量不足 ✗：需要 ≈{gib(int(self.total_bytes * DISK_HEADROOM)):.2f} GiB、"
                f"可用 {gib(free):.2f} GiB ✓ ⇒ **一个字节都没写** ✓（本次重算 ✓）")
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.tmp = self.target.with_name(f".{self.target.name}.{os.getpid()}.tmp")
            self._handle = self.tmp.open("wb")
        except OSError as err:
            self.abort()
            self._refuse(f"产物文件打不开 ✗（{type(err).__name__}: {err}）⇒ 本次重算 ✓")
        self._handle.write(b"\x00" * 8)          # 8 字节头长先占位 ✓（头长写完才知道真值 ✓）
        self._handle.write(self.header)

    # ---- 写数据 ----

    def add(self, name: str, raw: bytes) -> None:
        """写一个张量的**目标 dtype 字节** ✓（顺序必须与计划一致 ✓，字节数必须与计划相同 ✓）。"""
        if not self._order or name != self._order[0]:
            want = self._order[0] if self._order else None
            raise MaterializeError(
                f"写入顺序与计划不一致 ✗：想写 {name!r} ✓、计划是 {want!r} ✓"
                f"（顺序错 ⇒ 偏移全错 ⇒ 产物「形状对、数值错」✗✗）")
        spec = self.specs[len(self._written)]
        if len(raw) != int(spec.nbytes):
            raise MaterializeError(
                f"{name!r} 字节数不符 ✗：实写 {len(raw)} ✓、计划 {spec.nbytes} ✓"
                f"（dtype·形状对不上 ⇒ 头会自证失败 ✗）")
        assert self._handle is not None
        self._handle.write(raw)
        self._written.add(name)
        self._order.pop(0)
        self.bytes_written += len(raw)

    # ---- 收尾 ----

    def close(self) -> Report:
        """收尾 ✓：键集核对 ⇒ 头长回填 ⇒ ``os.replace`` ⇒ sidecar ✓（环境失败 ⇒ 退化成「没写」✓）。

        结论同时留在 :attr:`report` 上 ✓：``with`` 用法里调用方**拿不到返回值** ✗
        （``__exit__`` 里 close 的返回值没人接 ✓✗）⇒ 回执得有个落脚点 ✓。
        """
        self.report = self._finish()
        return self.report

    def _finish(self) -> Report:
        try:
            if self._order:
                raise MaterializeError(
                    f"计划里还有 {len(self._order)} 个张量没写 ✗（{self._order[:3]} ✓）"
                    f"⇒ **不留残缺产物** ✗✗（少张量的产物装得上、跑得动、画面不对 ✗）")
            assert self._handle is not None and self.tmp is not None
            self._handle.seek(0)
            self._handle.write(len(self.header).to_bytes(8, "little"))
            self._handle.flush()
            try:
                os.fsync(self._handle.fileno())
            except OSError:
                pass                              # fsync 失败不算错 ✓（原子替换仍由 os.replace 保证 ✓）
            self._handle.close()
            self._handle = None
            os.replace(self.tmp, self.target)      # ⭐ 原子 ✓：别人看不到半成品 ✓✗
            self.tmp = None
        except MaterializeError as err:
            self.abort()
            _bump("failed", 1, reason=f"计划与实写不符 ✗（{err}）")
            raise
        except OSError as err:
            self.abort()
            _bump("failed", 1, reason=f"产物写不进去 ✗（{type(err).__name__}: {err}）")
            return Report(hit=False, path=str(self.target), reason=(
                f"产物写不进去 ✗（{type(err).__name__}: {err}）⇒ 本次重算 ✓ "
                f"（退化成「慢」而不是「错」✓✗）"), tensor_count=len(self.specs),
                elapsed_ms=(time.perf_counter() - self._started) * 1000.0)
        sidecar_written, sidecar_error = self.owner._write_sidecar(self.target)  # noqa: SLF001
        _bump("written", 1, reason="没有可用产物 ⇒ 物化一次 ✓")
        _bump("writeBytes", int(self.total_bytes))
        return Report(hit=False, path=str(self.target), reason="没有可用产物 ⇒ 物化一次 ✓",
                      gib=gib(self.bytes_written), written=True, tensor_count=len(self.specs),
                      sidecar_written=sidecar_written, sidecar_error=sidecar_error,
                      elapsed_ms=(time.perf_counter() - self._started) * 1000.0,
                      notes=("产物是**目标 dtype** 的 safetensors ✓（下次装载直接读盘 ✓）",
                             "⚠️ 不验张量数值 ✗：头自洽 ≠ 数值对 ✓（数值由「命中/不命中逐位相等」保证 ✓）"))

    def abort(self) -> None:
        """放弃 ✓：**删掉半成品** ✗（绝不留一个「看起来能命中」的残缺文件 ✓✗）。"""
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        tmp, self.tmp = self.tmp, None
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is not None:                 # 装载过程中出错 ⇒ 产物也没意义 ✓
            self.abort()
            return False
        self.close()
        return False
