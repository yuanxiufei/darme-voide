"""**权重装载**（torch 侧 ✓）—— 把 safetensors 里的张量装进 ``nn.Module``，并**逐条报出差异** ✓。

## 为什么不直接用 ``load_state_dict`` 了事（这是本模块存在的全部理由 ✓）

``module.load_state_dict(state, strict=False)`` 有**两个**危险行为 ✗：

1. **键对不上时静默放过** ✗ —— 名字差一个字母 ✓ 就是"少装了一层却照常跑" ✓，
   产物**看起来有画面** ✗ 但其实是残缺模型 ✓（本仓最怕这种"像是成功"的失败 ✗）；
2. 它会**改名/改语义地兼容**（``_orig_mod.`` / ``module.`` 前缀 ✓ 等），
   让人误以为"装上了" ✓ —— 我们宁可**先报清楚**再决定 ✓。

⇒ 本模块先**自己算差集** ✓（missing / unexpected / 形状不符 ✓），再调用 ``load_state_dict`` ✓，
最后把结果**如实**回给调用方 ✓：``complete=False`` 就代表"**不是**完整的模型" ✗。

## 与其它层的关系

* :mod:`app.services.engine.safetensors` —— 纯 Python **读头部**（体检用 ✓ 不需要 torch ✓）；
* :mod:`app.services.engine.loader` —— 该不该装 / 装得下吗（加载计划 ✓）；
* :mod:`app.services.engine.quant` —— 低精度权重怎么**还原**（fp8/int8 → 可算精度 ✓ 布局按形状判 ✓
  判不出来**拒绝** ✗）；
* :mod:`app.services.engine.quant_nvfp4` —— **nvfp4**（U8 打包 4 bit + 块 scale ✓，形状**自己证明
  不了**布局 ✗ ⇒ 判据是等式 ✓ 走另一条路 ✓，见下）；
* 本模块 —— **真的把张量交给模块** ✓。

⚠️ **nvfp4 是第三条路**（2026-10-30 ✓）：它的 ``weight`` 在文件里是 ``U8 [N, K/2]`` ✓（列数**正好**
是逻辑形状的一半 ✓），``weight_scale`` 是 swizzled 的块 scale ✓ ⇒ :func:`quant.layout_of` 那套
「形状能自证」的判据**判不出它** ✓✗（会报「对不上任何一种布局」✗）。
⇒ 本模块先 :func:`nvfp4_layers_of` **逐层认出**（走 :func:`quant.supported_format` ✓ 与体检同一入口 ✓），
再按**打包形状 = 逻辑形状的一半**放行 ✓，然后调 :func:`quant_nvfp4.dequantize` **在 float32 里**
还原 ✓、折 AWQ ✓、最后才降到参数 dtype ✓（**顺序不能反** ✗：先降精度再乘 9.7 倍的 AWQ 会把
舍入放大 ✓）。

⚠️ **装载顺序**（2026-09-22 ✓）：**先反量化，再归一 dtype** ✗ —— 目标主权重就是 **fp8**（19.53 GiB ✓），
而 fp8 张量直接 `to(bf16)` 是**把尺度丢掉** ✗ ⇒ `load_state_dict` 照单全收、**不报错** ✗✗，
症状要到出片才发现「画面是噪声」✓✗。⇒ 现在低精度权重走 `quant.dequantize_state` ✓
（布局 / 来源 scale 都随报告给出来 ✓），**判不出来就中止装载** ✓（不按猜的布局算 ✗）。
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from . import prefetch as pf_mod
from . import quant as quant_mod
from . import quant_nvfp4 as nvfp4_mod
from . import safetensors as st_mod

__all__ = ["WeightLoadReport", "RestoreError", "diff_state_dict", "load_module_weights",
           "make_reloader", "nvfp4_layers_of", "save_module_weights"]

#: safetensors 头部的 dtype 字符串 → torch 的属性名 ✓（**装载**要知道怎么把字节变成张量 ✓）
#: ⚠️ 这张表必须覆盖 :data:`safetensors.DTYPE_ITEMSIZE` 的全部取值 ✗ ——
#: 少一个就会在「读到了字节但造不出张量」处抛 ✓✗（而不是给出一句可行动的话 ✓）。
_ST_DTYPE: dict[str, str] = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "I16": "int16", "U16": "uint16", "F16": "float16", "BF16": "bfloat16",
    "I32": "int32", "U32": "uint32", "F32": "float32",
    "I64": "int64", "U64": "uint64", "F64": "float64",
}


@dataclass
class WeightLoadReport:
    """装载结果 ✓（``complete`` 看 **missing 与 shapeMismatch 都为空** ✓ —— unexpected 不影响完整性 ✓）。"""

    path: str = ""
    loadedKeys: int = 0
    sourceKeys: int = 0
    targetKeys: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shapeMismatch: list[dict[str, Any]] = field(default_factory=list)
    renamed: int = 0
    dtypeCasts: dict[str, int] = field(default_factory=dict)
    #: 被**当元数据吃掉**的 `*.comfy_quant` 个数 ✓（2026-10-29 ✓）—— 它们不是权重 ✗，
    #: 但一个模板里就有 400+ 个 ✓ ⇒ 落进 `unexpected` 会把真正的「多出来的键」淹掉 ✓✗
    consumedQuantMeta: int = 0
    #: **原样常驻**成 int8 的权重个数 ✓（不反量化 ✓ —— 见 `weights` 模块 docstring 的显存一段）
    #: ⚠️ 含 4bit 打包层 ✓（它们的参数也是 ``int8`` ✓、形状就是文件里的打包形状 ✓）——
    #: 于是 :func:`_int8_residency_problems` 的「常驻层数 ≥ 层数」那一条**同时**管住两者 ✓。
    int8Resident: int = 0
    #: 其中**4bit 打包**的层数 ✓（:meth:`Int8Linear.forward` 会多一步 :func:`quant.unpack_int4` ✓）。
    #: 单列出来的理由：4bit 与 per-row int8 **头部逐字同形** ✓✗ ⇒ 只报一个合计数时，
    #: 「说好 224 层 4bit、实际全按 int8 乘了」在报告里看不出来 ✓✗（2026-10-30 的假绿 ✓）。
    int4Resident: int = 0
    #: 被**当配套张量吃掉**的 nvfp4 伙伴个数 ✓（`weight_scale` / `weight_scale_2` / `pre_quant_scale` ✓）——
    #: 它们和 `comfy_quant` 一样**不是权重** ✗，但一个 32B 文本编码器有 700+ 个 ✓ ⇒ 落进
    #: `unexpected` 会把「真·多出来的键」淹掉 ✓✗
    consumedNvfp4: int = 0
    #: 被吃掉的 **fp8_scaled 配套**个数 ✓（`.scale_weight` ✓ / `.scale_input` ✓ 激活侧 /
    #: `scaled_fp8` ✓ 整份标记）—— 同上：一个 14B 文档就有 400+ 个 ✓，不单列就会把
    #: `unexpected` 淹掉 ✓✗；但**必须计数** ✗（静默吞掉就等于「装了什么、丢了什么」说不清 ✓）
    consumedScales: int = 0
    #: 反量化明细 ✓（动过才有内容 ✓）：每个低精度权重的**布局 / 来源 scale / 原 dtype** ✓
    #: ⚠️ 它必须**随报告给出来** ✗ —— 「装上了」与「按对的布局还原了」是两件事 ✓（同 `configSources` 的纪律 ✓）。
    dequant: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: ⭐ 被**内存这一级**卸过、因而**故意没装**的键数 ✓（2026-11-02 ✓）。
    #: ⚠️ 它**不是** missing ✗✗：missing 是"模型要、文件里没有"（真缺陷 ✓）；
    #: 这里是"文件里有、我们决定先放盘上"（策略 ✓）—— 混成一栏会让真缺键被埋掉 ✓✗。
    skipped: int = 0
    #: ⭐ 逐张量**还原器** ✓（2026-11-02 ✓）：**不进** :meth:`to_dict` ✗ ——
    #: 它握着文件 reader 与模块参数的引用 ✓，序列化既没意义也装不下 ✓。
    #: 内存这一级（`residency` 的 ``host_cache``）靠 ``report.restorer.group_values`` 重读 ✓；
    #: ``None`` ⇒ 这次装载**没走到**逐张量那一步（钥匙都对不上 / 预检就中止了 ✓），
    #: 别把它当成"能重读"✓✗（那时卸层就是一个填不回去的洞 ✗）。
    restorer: Any = None

    @property
    def complete(self) -> bool:
        """**完整**装载 ⇔ 没有缺键、没有形状不符 ✓（多出来的键只报告不阻断 ✓）。"""
        return not self.error and not self.missing and not self.shapeMismatch

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "complete": self.complete, "loadedKeys": self.loadedKeys,
            "sourceKeys": self.sourceKeys, "targetKeys": self.targetKeys,
            "missing": self.missing[:12], "missingCount": len(self.missing),
            "unexpected": self.unexpected[:12], "unexpectedCount": len(self.unexpected),
            "shapeMismatch": self.shapeMismatch[:6], "shapeMismatchCount": len(self.shapeMismatch),
            "renamed": self.renamed, "dtypeCasts": self.dtypeCasts,
            "consumedQuantMeta": self.consumedQuantMeta, "int8Resident": self.int8Resident,
            "int4Resident": self.int4Resident,
            "consumedNvfp4": self.consumedNvfp4, "consumedScales": self.consumedScales,
            "dequant": dict(self.dequant), "error": self.error,
        }


#: 读前置的**件数** ✓（2＝当前这件在反量化/写卡时，下一件已经在盘上读 ✓）。
_READ_AHEAD = 2
#: 读前置的**单件字节上限** ✓✗：20 GiB 的模型里也有 780 MiB 的 embedding ✓，
#: 提前读它等于峰值多吃 780 MiB（而它只换来几毫秒 ✗）⇒ 超过这条线的一律就地读 ✓。
_READ_AHEAD_MAX_BYTES = 256 * 1024 * 1024
#: 上一次装载的读前置统计 ✓（``engine_readahead_test`` 靠它证明**真的用上了** ✗✗ 而不是「接了个空壳」）。
_LAST_PREFETCH: dict[str, Any] = {}


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


#: `*.comfy_quant` 的读取上限 ✓（实测 55~72 字节 ✓ —— 给 4 KB 余量 ✓；超过就当「不是小 json」跳过 ✗）
_QUANT_META_MAX = 4096


def convrot_of(reader: Any) -> tuple[int | dict[str, int] | None, str | None]:
    """从**已打开的** reader 判「哪些层带 convrot、group 多大」✓（不读大权重 ✓）。

    ⚠️ 判据必须与体检**同源** ✗（:func:`quant.convrot_group_of` 只此一处 ✓），而且必须**逐层** ✗✗
    —— 2026-10-30 实测两处栽在「整文件级」这个假设上：
    ① H3 **全量**版同一文件里 group 有 **64 与 256 两种**（`blocks.0.adaln_proj` vs `attn.qkv_proj` ✓）
       ⇒ 旧实现当场把 31.7 GiB 的正品拒了 ✓；
    ② Krea2 int4 那族**一个 `*.comfy_quant` 都没有**、逐层说明全在 metadata
       `_quantization_metadata` 里 ✓ ⇒ 旧实现返回 ``None`` ⇒ 装载时**少一步反旋转** ⇒
       能装、能跑、画面全是噪声 ✓✗（这正是本仓最怕的「像是成功」的失败 ✓）。
    ⇒ 这里读**全部** `*.comfy_quant`（实测 350 层 × ~60 B ≈ 21 KB ✓）**并且**读文件级 metadata ✓。
    """
    return quant_mod.convrot_group_of(quant_specs_of_reader(reader))


def quant_specs_of_reader(reader: Any) -> dict[str, dict[str, Any]]:
    """``{层名: 说明}`` ✓ —— 读**全部** `*.comfy_quant` + 文件级 `_quantization_metadata` ✓。

    ⚠️ 抽成独立函数是为了**一份证词被两处读** ✗✗（2026-10-30）：convrot 与「4bit 打包」
    （`:func:`quant.int4_packed_of` ✓）必须看**同一批**说明 ✓ —— 各读各的就会分叉成
    「convrot 按 256 判 ✓、int4 没认出来 ⇒ 按 int8 乘」✓✗（那正是本轮修掉的假绿 ✓）。
    """
    specs: dict[str, dict[str, Any]] = {}
    suffix = quant_mod.QUANT_META_SUFFIX
    for name in reader.entries:
        if not name.endswith(suffix):
            continue
        entry = reader.entries[name]
        if int(getattr(entry, "nbytes", 0) or 0) > _QUANT_META_MAX:
            continue                       # 不是小 json ⇒ 不猜 ✓
        try:
            raw = reader.read(name)
        except (st_mod.SafetensorsError, OSError):
            continue                       # 补证据失败不该把装载搞崩 ✓
        spec = quant_mod.parse_format_spec(raw)
        if spec:
            specs[name[: -len(suffix)]] = spec
    # 文件级说明（StarNodes 那一族 ✓）：张量里的逐层说明优先 ✓，这边只补空位 ✓
    for stem, spec in quant_mod.specs_from_metadata(getattr(reader, "metadata", None)).items():
        specs.setdefault(stem, spec)
    return specs


#: nvfp4 层除 ``weight`` 之外必须一起读走的三类伙伴 ✓（它们**不是模型参数** ✗ —— 模型里只有 weight ✓）
_NVFP4_COMPANIONS: tuple[str, ...] = (
    nvfp4_mod.COMFY_SCALE_SUFFIX, nvfp4_mod.COMFY_SCALE_2_SUFFIX, nvfp4_mod.PRE_QUANT_SCALE_SUFFIX)


def nvfp4_layers_of(reader: Any) -> dict[str, dict[str, Any]]:
    """列出该文件里**自称 nvfp4** 的层 ✓ ⇒ ``{weight 原始键: {"stem": 原始模块名}}`` ✓（没有 ⇒ ``{}``）。

    ⚠️ 判族必须走 :func:`quant.supported_format` ✗（**不许自己认字符串** ✓）—— 与体检那条
    （``loader.dequant_plan_of`` → ``quant_nvfp4.plan``）**同一个入口** ✓，否则「体检说 nvfp4 ✓、
    装载按普通 U8 装 ✓✗」就会分叉：形状对不上时当场报错还算好 ✓，要是撞上一个 ``(N, K/2)``
    恰好等于别的目标形状，就是**静默装错** ✓✗。

    ⚠️ 这里读**全部** ``*.comfy_quant`` ✗（族级判断看几条就够 ✓，但**逐层路由**必须每一层都知道 ✓
    —— 漏一层就是那一层不还原 ✓✗）；实测 350 层 × ~60 字节 ≈ 21 KB ✓，代价可忽略 ✓。
    （:func:`convrot_of` 2026-10-30 起也读**全部** ✓ —— 它要给逐层 convrot 表 ✓，只读前 8 个
    会把逐层分组判成「单组」✓✗。）
    """
    out: dict[str, dict[str, Any]] = {}
    for name in reader.entries:
        if not name.endswith(quant_mod.QUANT_META_SUFFIX):
            continue
        entry = reader.entries[name]
        if int(getattr(entry, "nbytes", 0) or 0) > _QUANT_META_MAX:
            continue                       # 不是小 json ⇒ 不猜 ✓
        try:
            raw = reader.read(name)
        except (st_mod.SafetensorsError, OSError):
            continue                       # 补证据失败不该把装载搞崩 ✓
        hit = quant_mod.supported_format(quant_mod.parse_format_spec(raw))
        if hit is None or hit[0] != nvfp4_mod.FORMAT:
            continue
        stem = name[: -len(quant_mod.QUANT_META_SUFFIX)]
        out[f"{stem}.weight"] = {"stem": stem}
    return out


def _torch_dtype(name: str) -> Any:
    """头部 dtype 字符串 → torch dtype ✓（不认识 ⇒ 说清是哪种写法 ✗ 不猜 ✓）。"""
    attr = _ST_DTYPE.get(name)
    if attr is None:
        raise ValueError(f"不认识的 dtype {name!r} ✗（safetensors 规范之外的值 ⇒ 不装 ✓）")
    value = getattr(_torch(), attr, None)
    if value is None:
        raise ValueError(f"本机 torch 没有 {attr} ✗ ⇒ 装不了 {name} 这类权重 ✓")
    return value


def _tensor_from_bytes(raw: bytes, dtype: Any, shape: tuple[int, ...]) -> Any:
    """字节 → 张量 ✓（``frombuffer`` **零拷贝** ✓，不额外占一份权重大小的内存 ✓）。

    ⚠️ 返回的张量是**只读**的（共享 ``raw`` 的缓冲 ✓）⇒ 调用方**只能读它** ✓，
    真写进模型是由 ``param.copy_(tensor)`` 完成的 ✓（copy_ 读源、写目标 ✓）。
    """
    torch = _torch()
    shape = tuple(int(dim) for dim in shape) if shape else ()
    if 0 in shape:
        #: ⚠️ **0 元素张量**（哨兵 ✓，如 Edit 2511 的 ``__index_timestep_zero__`` 在 int8_convrot
        #: 那份里是 ``shape=[0]`` ✓、TE 的 ``scaled_fp8`` 同物种 ✓）：``frombuffer(b"")`` 当场抛
        #: ``both buffer length (0) and count (-1) must not be 0`` ✓✗（2026-11-01 真权重首跑踩到 ✓，
        #: 报错落在 ``torch.frombuffer`` 上、读起来像「文件坏了」✓，其实**字节本来就该是 0 个** ✓）。
        #: 它的意义只在「键存在」✓ ⇒ 给个同形状的空张量 ✓（``copy_`` 空张量也无事 ✓）。
        return torch.empty(shape, dtype=dtype)
    tensor = torch.frombuffer(raw, dtype=dtype)
    return tensor.reshape(shape if shape else ())


def _read_named(reader: Any, name: str) -> Any:
    """按**文件里的原名**读一个小张量 ✓（nvfp4 的 scale 伙伴走这条 ✓；不需要逻辑改名 ✓）。"""
    entry = reader.entries[name]
    return _tensor_from_bytes(reader.read(name), _torch_dtype(entry.dtype),
                              tuple(int(dim) for dim in entry.shape))


def diff_state_dict(module: Any, state: dict[str, Any]) -> WeightLoadReport:
    """只算差集 ✓（不装载 ✓）—— 便于**先看**再决定 ✓。"""
    report = WeightLoadReport(targetKeys=len(module.state_dict()), sourceKeys=len(state))
    own = module.state_dict()
    for key, tensor in state.items():
        if key not in own:
            report.unexpected.append(key)
            continue
        if tuple(own[key].shape) != tuple(tensor.shape):
            report.shapeMismatch.append({
                "key": key, "expected": list(own[key].shape), "got": list(tensor.shape)})
            continue
        report.loadedKeys += 1
    report.missing = [key for key in own if key not in state]
    return report


def _compile_key_map(key_map: dict[str, str] | None) -> list[tuple[Any, str]]:
    """把 ``{正则: 替换}`` 预编译一次 ✓（顺序有意义：**首个命中**说了算 ✓）。"""
    return [(re.compile(pattern), target) for pattern, target in (key_map or {}).items()]


def _map_name(key: str, compiled: list[tuple[Any, str]]) -> str:
    """单个键的改名 ✓ —— 流式与整份两条路径**共用这一处** ✗（各写一份必然漂 ✓✗）。"""
    for pattern, target in compiled:
        if pattern.search(key):
            return pattern.sub(target, key)
    return key


def _apply_key_map(state: dict[str, Any], key_map: dict[str, str]) -> tuple[dict[str, Any], int]:
    """按**显式给出的**正则映射改名 ✓（多条规则按顺序 ✓；没匹配上的原样保留 ✓）。

    ⚠️ 不做"猜前缀"式的自动改名 ✗ —— 那正是上面说的第 2 个危险行为 ✓。
    """
    compiled = _compile_key_map(key_map)
    renamed = 0
    out: dict[str, Any] = {}
    for key, tensor in state.items():
        new_key = _map_name(key, compiled)
        if new_key != key:
            renamed += 1
        out[new_key] = tensor
    return out, renamed


def load_module_weights(module: Any, path: str | Path, *, dtype: Any = None,
                        device: str | None = None,
                        key_map: dict[str, str] | None = None,
                        read_ahead: int = _READ_AHEAD) -> WeightLoadReport:
    """把 ``path`` 的权重**逐张量**装进 ``module`` ✓，并**如实**报告差异 ✓（不静默 ✗）。

    ## 为什么是流式（2026-10-29 起 ✓）

    旧实现用 ``load_file`` 把**整份**检查点读进内存 ✗ —— 主权重 19.53 GiB 的 fp8 还要再经
    一次**整份** ``dequantize_state`` 变成 bf16 ✓✗ ⇒ 峰值接近 **2 倍模型大小** ✓✗：
    「小内存跑大模型」在装载这一步就已经死了 ✗（还没开始采样就 OOM ✓）。

    现在改成三段，**每一段的失败都发生在「写第一个字节之前」** ✓：

    1. **只读头部**算差集（缺键 / 多键 / 形状不符 ✓）—— 形状不符时**一个字节都不读** ✓；
    2. **预检**（:func:`quant.precheck_dequant` ✓）：只读 scale 这类**小**张量 ✓，
       把「布局判不出来 / 缺 scale / scale 里有 0」一次收齐 ✓ ⇒ 不合格就中止 ✓ 模型**零污染** ✓；
    3. **逐张量**：读 → 反量化 →（可选）cast → ``param.copy_`` ✓ ⇒ 峰值 ≈ **单个张量** ✓。

    ⚠️ ``device`` 参数**不再参与读取** ✗：字节总是先进 CPU（:func:`_tensor_from_bytes` ✓），
    再由 ``copy_`` 落到 **param 自己所在的 device** ✓。

    现有 4 个调用点两种写法都有 ✓：3 处**先** ``.to(device)`` 再装载 ✓；
    放大器那处（``torch_backend`` 约 1378 行 ✓）是**装载后**才 ``net.to(device)`` ✓。
    两种都等价 ✓ —— 最终放置一律由 ``.to`` 决定 ✓，没有一处依赖 ``device=`` 把权重送卡上 ✓。
    旧实现反而在那 1 处**白占了一份显存** ✗：整份权重先读上卡 ✓、再从卡 copy 回 CPU 的 param ✓✗、
    最后 ``.to`` 又搬一次 ✓（流式实现只在最后一次 ``.to`` 上卡 ✓）。
    """
    target = Path(path)
    own = module.state_dict()          # ⚠️ 是 param 的**视图**（共享 storage ✓）⇒ 不复制数据 ✓
    report = WeightLoadReport(path=str(target), targetKeys=len(own))
    if not target.exists():
        report.error = f"权重文件不存在：{target} ✗（先按 `loader.plan_stage` 的报告去装 ✓）"
        return report

    try:
        reader = st_mod.TensorReader(target)
    except (st_mod.SafetensorsError, OSError, ValueError) as err:
        report.error = f"读取权重失败：{type(err).__name__}: {err}"
        return report

    #: ⭐ 内存这一级卸过的组 ✓（2026-11-02 ✓）：由 `residency._spill` 写在**模块上** ✓✗ ——
    #: 本文件有 13 处装载调用点 ✓，逐个加参数必漏 ✓（漏一处就是「前向到那层报形状错」✓✗），
    #: 所以走模块上的约定而不是参数 ✓。
    #: 被跳过的键**不读、不写** ✓：参数已被换成 0 元素占位 ✓，``copy_`` 进去会当场形状不符 ✗✗，
    #: 它们留给前向那一刻的 ``reload``（:func:`make_reloader` ✓）补 ✓。
    skip_prefixes = tuple(
        f"{name}." for name in getattr(module, "_residency_spilled", ()) or ()
        if name and name != "‹root›")

    with reader, _prefetch_scope(read_ahead) as prefetcher:
        return _load_streaming(reader, own, report, dtype=dtype, key_map=key_map,
                               read_ahead=read_ahead, prefetcher=prefetcher,
                               skip_prefixes=skip_prefixes, module=module)


def _read_ahead_bytes(entries: dict[str, Any], orig: str) -> int:
    """这件**有多少字节** ✓（读前置要按它筛 ✓✗：省一条 780 MiB 的 embedding 比多读十层值钱 ✓）。"""
    entry = entries.get(orig)
    return int(getattr(entry, "nbytes", 0) or 0)


def _submit_ahead(prefetcher: Any, reader: Any, plan: list[tuple[str, str]],
                  entries: dict[str, Any], start: int, count: int) -> None:
    """把后面 ``count`` 件**交给后台读** ✓（不阻塞 ✓、不抛 ✓、超上限的跳过 ✓）。

    ⚠️ 只在**本次装载调用**里有意义 ✗：跨阶段预取（趁 encode 时读 sample 要的 DiT ✓）听着更划算，
    但换装的第一步是 `release_part` ⇒ 预取好的字节会在「退干净」之后**指向已经不要的东西** ✓✗，
    且峰值要多吃一整件（11~19 GiB ✗）⇒ 那不属于「预取」✓，属于「两套同时在内存里」✗✗。
    """
    for orig, _logical in plan[start:start + count]:
        if _read_ahead_bytes(entries, orig) > _READ_AHEAD_MAX_BYTES:
            continue                      # 大件就地读 ✓（预取它等于把峰值翻倍 ✗）
        prefetcher.submit(orig, partial(reader.read, orig))


@contextmanager
def _prefetch_scope(read_ahead: int) -> Iterator[Any]:
    """开一个**只活一次装载**的预取器 ✓（``read_ahead<=0`` ⇒ 给 ``None`` ✓ 全走就地读 ✓）。

    ⚠️ 退出时 `close(wait=True)` **必须等干净** ✓✗：不等的话，本函数已经返回、装配线接着去
    `release_part` ⇒ 后台线程还捏着旧 reader 的字节在写 ✓，而调用方以为「退干净了」✓✗
    —— 这正是「跨阶段预取」被否掉的那个账 ✓，在这里用作用域把它焊死 ✓。
    """
    if int(read_ahead) <= 0:
        yield None
        return
    prefetcher = pf_mod.Prefetcher(name="load", max_inflight=max(1, int(read_ahead)))
    try:
        yield prefetcher
    finally:
        global _LAST_PREFETCH
        try:
            _LAST_PREFETCH = {**prefetcher.snapshot(), "readAhead": int(read_ahead)}
        except Exception:  # noqa: BLE001 - 观测面不许反过来搞坏装载 ✓✗
            _LAST_PREFETCH = {"readAhead": int(read_ahead)}
        prefetcher.close(wait=True)


def last_prefetch() -> dict[str, Any]:
    """上一次装载的读前置统计 ✓（``hits`` 为 0 ⇒ 一次都没用上 ✓：那这个优化等于没接 ✗✗）。"""
    return dict(_LAST_PREFETCH)


class RestoreError(RuntimeError):
    """单个张量**读不回来 / 还原不了** ✓（带键名与原因 ✓ —— 内存这一级拿它直接报给人看 ✓）。"""


class _Restorer:
    """「读一个键 ⇒ 拿到**最终值**」✓ —— 装配与**重读**共用这一段代码 ✓✗。

    ⚠️ 存在的唯一理由就是**不许两份还原逻辑** ✗✗：内存这一级（`residency` 的 ``host_cache``）
    卸层之后要重读 ✓，若另写一份"简化版"（只认 bf16 ✓、忘了 convrot / nvfp4 / int4 ✓ ✓），
    症状是**数值偏了但不报错** ✓✗。本仓已经用 `_int8_residency_problems` 那一套证明过：
    这类错误只能靠"同一处代码"防 ✓。

    ⚠️ 它**不写模块** ✗：只返回张量 ✓。写进去的动作留给调用方 ✓ ——
    装配时写 ``own[logical]`` ✓、重读时由 `residency._refill_group` 写回参数 ✓，
    两边都是"覆盖式 copy" ✓，可单元测试直接对拍 ✓。
    """

    def __init__(self, *, reader: Any, own: dict[str, Any], meta: dict[str, Any],
                entries: dict[str, Any], resident: set[str], nv_layers: dict[str, dict[str, Any]],
                int4: set[str], convrot: Any, converted: dict[str, dict[str, Any]],
                scales: dict[str, Any], dtype: Any,
                orig_of: dict[str, str] | None = None, path: str | Path | None = None) -> None:
        self.reader = reader
        #: ⭐ **重读专用**的第二把把手 ✓✗（2026-11-02 ✓）：装载那把活在
        #: ``load_module_weights`` 的 ``with reader:`` 里 ✓ ⇒ 出函数就关了 ✓，
        #: 而重读发生在**前向那一刻** ✓（可能几十分钟以后 ✓）—— 拿装载那把去读
        #: 会得到「读取已关闭」✓，症状是"卸了层、于是前向炸"✓✗。
        #: 懒开：不卸层就一个句柄都不多 ✓（不占 fd ✓）。
        self.path: Path | None = None if path is None else Path(path)
        self._reader: Any = None
        self.own = own
        self.meta = meta
        self.entries = entries
        self.resident = resident
        self.nv_layers = nv_layers
        self.int4 = int4
        self.convrot = convrot
        self.converted = converted
        self.scales = scales
        self.dtype = dtype
        #: 逻辑键 ⇒ 文件里的原名 ✓（重读要按原名去盘上取 ✓；别名改写会把这一步做错 ✗）
        self.orig_of: dict[str, str] = dict(orig_of or {})

    def value_of(self, orig: str, logical: str, prefetcher: Any = None,
                 reader: Any = None) -> tuple[Any, str | None]:
        """``(最终值, 被 cast 的原 dtype 名)`` ✓ —— 原 dtype 名仅在**真 cast 过**时非 ``None`` ✓。

        ``reader`` ✓：用哪把把手读 ✓（装配时是本次装载那把 ✓、重读时由
        :meth:`restore_reader` 现开的那把 ✓）—— 两把**读的是同一份字节** ✓✗，
        所以还原逻辑仍然只有一份 ✓。
        """
        torch = _torch()
        stream = self.reader if reader is None else reader
        try:
            raw = _read_raw(prefetcher, stream, orig)
        except (st_mod.SafetensorsError, OSError, pf_mod.PrefetchError) as err:
            raise RestoreError(
                f"读取 `{logical}` 失败：{type(err).__name__}: {err}") from err
        tensor = _tensor_from_bytes(raw, *self.meta[logical])
        del raw                      # 张量仍持着这块缓冲 ✓ ⇒ 这里删掉的只是本地名 ✓
        if logical in self.resident:
            # ⚠️ **原样写入** ✓：既**不**反量化 ✗（反量化就是把 17.94 GiB 变成 35.9 GiB ✗），
            #    也**不** cast ✗（`copy_` 会把 bf16 cast 进 int8 参数 ⇒ 静默截断成 0 ✓✗）。
            if tensor.dtype != torch.int8:
                raise RestoreError(
                    f"`{logical}` 的参数是 int8 常驻 ✓，但文件里是 {tensor.dtype} ✗"
                    f" ⇒ 拒绝写入（cast 进 int8 会被静默截断 ✓✗）")
            return tensor, None
        if logical in self.nv_layers:
            # ⚠️ nvfp4：**必须**按打包形状还原 ✗ —— 文件里的 `weight` 是 U8，两个 4 bit 挤一个字节 ✓，
            #    直接 `copy_` 进 bf16 参数只会得到「65536 分之 1 的值」✓✗（形状都对、数值全错 ✓✗）。
            #    算在 **float32** 里再 cast ✓：`fp4 × (块 scale × 二级 scale) × AWQ` 三步都不该在
            #    bf16 上连乘（尾数只有 8 位 ✓ 连乘误差会攒 ✓）。
            stem = self.nv_layers[logical]["stem"]
            #: ⚠️ 目标形状从**层体检的回执**取（``rows × inFeatures`` ✓），**不读** ``own[...].shape``
            #: ✗✗：内存这一级卸过的组，参数已经被换成 **0 元素占位** ✓ ⇒ 那里读到的是 ``(0,)`` ✓
            #: ⇒ 反量化按错形状拼 ⇒ 报一个「形状对不上」的错 ✓，而真正的原因在两级之外 ✓。
            plan_item = self.nv_layers[logical]
            target_shape = (int(plan_item["rows"]), int(plan_item["inFeatures"]))
            try:
                block_scale = _read_named(stream, stem + nvfp4_mod.COMFY_SCALE_SUFFIX)
                per_tensor = _read_named(stream, stem + nvfp4_mod.COMFY_SCALE_2_SUFFIX)
                pre_quant = (None if stem + nvfp4_mod.PRE_QUANT_SCALE_SUFFIX not in self.entries
                             else _read_named(stream, stem + nvfp4_mod.PRE_QUANT_SCALE_SUFFIX))
            except (st_mod.SafetensorsError, OSError) as err:
                raise RestoreError(
                    f"读取 `{logical}` 的 nvfp4 配套张量失败：{type(err).__name__}: {err}") from err
            try:
                value = nvfp4_mod.dequantize(tensor, block_scale, per_tensor,
                                             target_shape, dtype=torch.float32)
                value = nvfp4_mod.fold_pre_quant_scale(value, pre_quant)
            except ValueError as err:
                # 预检已用同一套等式验过 ✓ ⇒ 走到这里说明**文件在两次读之间变了**（或被别的
                # 进程改过 ✓）—— 一律中止并说清已写多少 ✗（继续写就是一锅乱 ✓✗）
                raise RestoreError(f"`{logical}` 的 nvfp4 还原被拒 ✗（{err}）") from err
            target = self.dtype if self.dtype is not None else self.own[logical].dtype
            if value.dtype != target:
                value = value.to(target)
            return value, None
        if logical in self.converted:
            # ⚠️ **反量化必须在 cast 之前** ✗（2026-09-22 ✓）：fp8 直接 `to(bf16)` 是把尺度丢掉 ✗，
            #   `copy_` 照单全收 ⇒ 不报错 ✗✗，症状要到出片才发现「画面是噪声」✓✗
            scale_name = self.converted[logical]["scale"]
            #: ⚠️ `int4` 必须**逐层传下去** ✗：预检认出了打包 ✓、逐层却没解包 ✓✗
            #: ⇒ 又是一次「报告说解了、实际没解」✓✗（`report.dequant["int4PackCount"]` 会露馅 ✓）
            single = quant_mod.dequantize_state(
                {logical: tensor, scale_name: self.scales[scale_name]},
                convrot=self.convrot, int4=self.int4)
            if not single.ok:        # pragma: no cover - 上面预检应当已拦下 ✓
                raise RestoreError(
                    f"`{logical}` 反量化失败 ✗（预检没拦住 ✓）：" + "；".join(single.problems[:2]))
            tensor = single.tensors[logical]
        if self.dtype is not None and tensor.dtype != self.dtype:
            return tensor.to(self.dtype), str(tensor.dtype).replace("torch.", "")
        return tensor, None

    def __call__(self, group: str) -> dict[str, Any]:
        """可调用 ✓（``slot(group)`` 直接落到 :meth:`group_values` ✓ —— 绑定与关闭都靠同一个对象 ✓）。"""
        return self.group_values(group)

    def group_values(self, group: str) -> dict[str, Any]:
        """把**一个驻留组**的全部键重新还原 ✓ ⇒ ``{相对名: 张量}`` ✓。

        ⚠️ 相对名而不是全名 ✗：`residency._refill_group` 用的是
        ``named_parameters(recurse=True)`` 的名字 ✓（那正是相对名 ✓）—— 两边都用全名就等于
        把"前缀剥到哪一层"变成第二个真源 ✓✗（`KREA2_PREFIXES` 那次已经栽过 ✓）。

        ⚠️ **顺序读、不预取** ✗：这是"前向到这一层"的**当拍** ✓，预取要提前一个层才叫预取 ✓
        （那属于 `residency` 的下一轮预取规则 ✓，见 `prefetch` 模块头"跨阶段预取被否掉" ✓）。
        """
        prefix = "" if group in ("‹root›", "") else group + "."
        stream = self.restore_reader()
        values: dict[str, Any] = {}
        for logical in self.own:
            if prefix and not logical.startswith(prefix):
                continue
            rel = logical[len(prefix):]
            if not rel:
                continue
            orig = self.orig_of.get(logical)
            if orig is None:         # pragma: no cover - 能装进来就一定有出处 ✓
                raise RestoreError(f"`{logical}` 没有登记出处 ⇒ 无法重读 ✗")
            value, _cast = self.value_of(orig, logical, reader=stream)
            values[rel] = value
        if not values:
            raise RestoreError(
                f"组 `{group}` 一个键都没匹配上 ✗（组名与模块参数名对不上 ⇒ 重读会静默返回空 ✗）")
        return values

    def restore_reader(self) -> Any:
        """重读用的把手 ✓（第一次用到时**现开** ✓，之后复用 ✓；文件没了 ⇒ :class:`RestoreError` ✓）。

        ⚠️ 为什么是**第二把**而不是"接着用装载那把" ✗✗：``load_module_weights`` 用
        ``with reader:`` 管装载 ✓ ⇒ 出函数时 reader 已经关了 ✓，而重读发生在**前向那一刻** ✓。
        症状是"卸层之后一开始采样就报「读取已关闭」"✓ —— 一个看起来像 IO 故障、
        实际是**句柄寿命**的错 ✓✗。

        ⚠️ 也**不是**每次重读都现开 ✗：一层的参数要读好几个键 ✓、一层还可能每步都卸 ✓
        ⇒ 每次现开等于每层重解析一遍头部（真机头部有几 MB ✓）。开一次留到
        :meth:`close` ✓（由 `residency.Handle.detach` 收 ✓）。
        """
        if self._reader is not None:
            return self._reader
        if self.path is None or not self.path.exists():
            raise RestoreError(
                f"权重文件不在盘上了：{self.path} ✗ ⇒ 卸掉的层读不回来"
                f"（内存这一级要靠它 ✓）—— 别把盘上的原件挪走 / 删掉 ✓")
        try:
            self._reader = st_mod.TensorReader(self.path)
        except (st_mod.SafetensorsError, OSError, ValueError) as err:
            raise RestoreError(
                f"重读口打不开：{type(err).__name__}: {err}") from err
        return self._reader

    def close(self) -> None:  # pragma: no cover - 只在退件路径上
        """把重读口还回去 ✓（**没开过就什么都不做** ✓ —— 不卸层的那些件一个 fd 都不占 ✓）。"""
        reader, self._reader = self._reader, None
        if reader is None:
            return
        try:
            reader.close()
        except Exception:  # noqa: BLE001 - 收尾不许反过来搞坏退件 ✓✗
            pass


def _read_raw(prefetcher: Any, reader: Any, orig: str) -> bytes:
    """取一件的字节 ✓（预取里有就等它 ✓、没有就地读 ✓ —— **两条路字节相同** ✓✗）。"""
    if prefetcher is None:
        return reader.read(orig)
    try:
        return prefetcher.take(orig)
    except KeyError:
        #: 没提交过（大件被筛掉 / 槽满被拒 ✓）⇒ 就地读 ✓：预取是**加速**，不是**通路** ✗✗
        return reader.read(orig)


def _load_streaming(reader: Any, own: dict[str, Any], report: WeightLoadReport, *,
                    dtype: Any = None, key_map: dict[str, str] | None = None,
                    read_ahead: int = _READ_AHEAD, prefetcher: Any = None,
                    skip_prefixes: tuple[str, ...] = (),
                    module: Any = None) -> WeightLoadReport:
    """流式装载的**主体** ✓（差集 → 预检 → 逐张量 ✓；分三段是为了「失败时不写字节」✓）。

    ``skip_prefixes`` ✓：属这些组前缀的键**故意不装** ✓（内存这一级已卸 ✓，留给前向重读 ✓）。
    ⚠️ 跳过**发生在预检之后** ✗✗：预检要看到完整形状表 ✓，先跳会让 nvfp4 的
    "列数恰好一半"判不出来 ✓✗（跳过 ≠ 缺键，混起来会把一次正常装载报成半装 ✗）。
    """
    entries = reader.entries
    report.sourceKeys = len(entries)

    # ① 逻辑名：显式改名只改**名字空间** ✓ —— 读字节仍按原名 ✓（文件里没有新名字 ✗）
    compiled = _compile_key_map(key_map)
    orig_of: dict[str, str] = {}
    for orig in entries:
        logical = _map_name(orig, compiled)
        if logical != orig:
            report.renamed += 1
        orig_of[logical] = orig
    #: 模型里**真有的**模块名 ✓（后面判「这个配套键是给谁的」要用 ✓ —— 提前算，别在循环里重复算 ✓）
    own_stems = {key.rsplit(".", 1)[0] for key in own}
    #: fp8_scaled 的**配套**键 ✓（`.scale_weight` ✓ 权重的 scale / `.scale_input` ✓ 激活侧 /
    #: `scaled_fp8` ✓ 整份标记）—— 它们**不是模型参数** ✗，但一个 14B 文件就有 400+ 个 ✓
    #: ⇒ 落进 `unexpected` 会把「真·多出来的键」淹掉 ✓✗（同 `*.comfy_quant` / nvfp4 伙伴的纪律 ✓）。
    #: 判据只认**这一层真会被写**的那些 ✓（`stem` 在模型里 ＋ 文件里那层确实是低精度 ✓）——
    #: 不满足就照旧报 `unexpected` ✓（宁可让人看一眼，也不悄悄吞掉 ✓）。
    aux_scales: set[str] = set()
    for logical, orig in orig_of.items():
        if not logical.endswith((".weight", ".weight_orig")):
            continue
        stem = (logical[: -len(".weight")] if logical.endswith(".weight")
                else logical[: -len(".weight_orig")])
        if stem not in own_stems or not quant_mod.is_low_precision(entries[orig].dtype):
            continue
        #: ⚠️ 配套名要在**逻辑名空间**里找 ✗（`key_map` 改过名后，源名与目标名对不上 ✓✗）
        for suffix in quant_mod.SCALE_SUFFIXES + quant_mod.INPUT_SCALE_SUFFIXES:
            name = f"{stem}{suffix}"
            if name in orig_of:
                aux_scales.add(name)
    if quant_mod.SCALED_FP8_MARKER in orig_of:
        aux_scales.add(quant_mod.SCALED_FP8_MARKER)

    # ①.5 nvfp4 的层：**必须逐层知道** ✓（见 :func:`nvfp4_layers_of` 的说明 ✓）——
    #     它们按「打包形状 = 逻辑形状的一半」匹配 ✓✗，不是形状不符 ✗
    nv_orig = nvfp4_layers_of(reader)
    nv_stems = {info["stem"] for info in nv_orig.values()}
    nv_logical: dict[str, str] = {}      # 逻辑 weight 键 → 原始模块名 ✓
    for orig_w in nv_orig:
        nv_logical[_map_name(orig_w, compiled)] = nv_orig[orig_w]["stem"]

    #: ⭐ 内存这一级卸过的键 ✓✗（2026-11-02 ✓）：**必须在形状比对之前认出来** ✗✗ ——
    #: 那些参数已经被换成 **0 元素占位** ✓ ⇒ ``own[logical].shape`` 是 ``(0,)`` ✓ ⇒
    #: 照常比形状会把**每一层卸过的**都判成 ``shapeMismatch`` ✓、``report.error`` 置位 ✓、
    #: 整次装载中止 ✓✗ —— 症状是「内存装不下 ⇒ 卸了几层 ⇒ 于是整个模型装不进来」✓，
    #: 而真正的病根（顺序反了）在报告里一个字都看不到 ✓✗。
    #: 跳过仍发生在**预检之后** ✓（见 :func:`_load_streaming` 抬头 ✓），这里只是让形状比对
    #: 别拿被破坏的东西当基准 ✓（与 :func:`residency._refill_group` 用快照核形状是同一条纪律 ✓）。
    skipped_keys = {logical for logical in orig_of if skip_prefixes
                    and logical.startswith(skip_prefixes)}

    # ② 差集：**只看头部** ✓ —— 形状不符时一个字节都不读 ✓（旧实现已经把 20 GiB 拉进内存了 ✗）
    meta: dict[str, tuple[Any, tuple[int, ...]]] = {}
    plan: list[tuple[str, str]] = []
    packed: set[str] = set()             # nvfp4 打包权重（逻辑键 ✓）：形状为逻辑的一半 ✓
    for logical, orig in orig_of.items():
        entry = entries[orig]
        try:
            torch_dtype = _torch_dtype(entry.dtype)
        except ValueError as err:
            report.error = f"权重里有装不了的 dtype ✗：{err}"
            return report
        shape = tuple(int(dim) for dim in entry.shape)
        meta[logical] = (torch_dtype, shape)
        if logical not in own:
            if logical.endswith(quant_mod.QUANT_META_SUFFIX):
                # ⚠️ `*.comfy_quant` 是**元数据**不是权重 ✓（内容是 55~72 字节的 json ✓）：它的
                #    信息已由 :func:`convrot_of` 读走 ✓；当成「多出来的键」报 ⇒ 400+ 条噪声把
                #    真正多出来的键淹掉 ✓✗（本仓纪律：报告要能一眼看出**有没有真问题** ✓）。
                report.consumedQuantMeta += 1
                continue
            if orig.rsplit(".", 1)[0] in nv_stems and orig.endswith(_NVFP4_COMPANIONS):
                # ⚠️ nvfp4 的 `weight_scale` / `weight_scale_2` / `pre_quant_scale` 同理 ✓：
                #    模型里**没有**这些键 ✗，但它们**必须**被读走（不读就还原不出来 ✓）⇒
                #    记成「吃掉的配套张量」✓，不当作「多出来的键」✗
                report.consumedNvfp4 += 1
                continue
            if logical in aux_scales:
                #: fp8_scaled 的配套键 ✓ —— **必须计数** ✗：静默吞掉就说不清「装了什么、丢了什么」✓
                report.consumedScales += 1
                continue
            report.unexpected.append(logical)
            continue
        if logical in skipped_keys:
            #: 卸过的键：**不比形状** ✓（占位是 ``(0,)`` ✓），也不先剔出 plan ✗
            #: —— 留着让预检看得见完整形状表 ✓，到下面 :data:`skip_prefixes` 那一步统一剔 ✓。
            plan.append((orig, logical))
            continue
        want = tuple(own[logical].shape)
        if want != shape:
            # ⚠️ nvfp4 的 `weight` 在文件里是 **U8 打包**（两个 4 bit 挤一个字节 ✓）⇒ 列数**正好是一半** ✓。
            #    这是唯一允许「形状不同」的情形 ✓✗：判据要求 **列数恰好一半、行数相同、dtype 是 U8** ✓
            #    —— 三条缺一条就按形状不符拒绝 ✗（放宽成「差不多」会把真错装进去 ✓✗）。
            if (logical in nv_logical and entry.dtype == "U8" and len(shape) == 2 and len(want) == 2
                    and want == (shape[0], shape[1] * 2)):
                packed.add(logical)
                plan.append((orig, logical))
                continue
            report.shapeMismatch.append({
                "key": logical, "expected": list(want), "got": list(shape)})
            continue
        plan.append((orig, logical))
    if skip_prefixes:
        kept = [(orig, logical) for orig, logical in plan
                if not logical.startswith(skip_prefixes)]
        report.skipped = len(plan) - len(kept)
        plan = kept
    report.missing = [key for key in own if key not in orig_of]
    report.loadedKeys = len(plan)
    if report.shapeMismatch:
        # ⚠️ 形状不符 ⇒ **不装** ✓（装了也没用 ✗ 而且会得到一个「结构不匹配」的模型 ✓✗）
        report.error = (f"{len(report.shapeMismatch)} 个张量形状不符 ✗ ⇒ 模型结构与权重不匹配，"
                        f"已**中止装载**（不是「部分装上」✗）：{report.shapeMismatch[:2]}")
        return report

    # ②.5 convrot：低精度 int8 还可能是**分组 Hadamard 旋转**过的 ✓（写在 `*.comfy_quant` 里 ✓）
    #      ⚠️ 必判 ✗：漏这一步 = 少做一次反旋转 ⇒ 数值全错、形状全对 ✓✗（不报错、出片全是噪声 ✗）
    layer_specs = quant_specs_of_reader(reader)
    convrot, convrot_gap = quant_mod.convrot_group_of(layer_specs)
    if convrot_gap:
        report.error = f"convrot 说明自相矛盾 ✗ ⇒ 已**中止装载**（不按猜的 group 算 ✗）：{convrot_gap}"
        return report
    #: ⚠️ 4bit **打包**（`w4a4`/`int4` ✓）：不在这里认出来 ⇒ 预检与逐层反量化都会把它当
    #: per-row **int8** 乘 ✓✗ —— 形状全对、数值全错、不报错 ✓✗（2026-10-30 实测：
    #: `krea2_v1.5_anime_int4_w4a4` 一度被判「可自动还原」✓）。同一批说明证词 ✓，不许各读各的 ✓。
    int4 = quant_mod.int4_stems_of(layer_specs)

    # ③ 预检：**只读 scale**（几十字节~几 KB ✓）⇒ 不合格时模型**零污染** ✓
    #    ⚠️ 这一步不能省 ✗：逐张量装载一旦在第 30 层才发现「scale 有 0」✗，
    #    前 29 层已经写进去了 ✓✗ ⇒ 半装状态最难查 ✓（旧实现是「整份读完才写」所以没这问题 ✓）。
    scales: dict[str, Any] = {}
    for logical, orig in orig_of.items():
        if not logical.endswith(quant_mod.SCALE_SUFFIXES):
            continue
        try:
            scales[logical] = _tensor_from_bytes(reader.read(orig), *meta[logical])
        except (st_mod.SafetensorsError, OSError):
            continue     # 读不到 ⇒ 预检会报「没有读到值」✓（不在这里猜 ✓）
    torch = _torch()
    #: ⚠️ **int8 常驻**（2026-10-29 ✓）：模板里**参数本身就是 int8** 的那些键 ✓ ⇒ **不反量化** ✗。
    #:    这是「19.53 GiB 权重装进 22.49 GiB 显存」的唯一路径 ✓✗：整份反量化成 bf16 = 35.9 GiB ✗
    #:    ⇒ 连装都装不进去 ✓（`h3_form.Int8Linear` 负责前向时逐层反量化 ✓）。
    #:    ⚠️ 判据只看**参数 dtype** ✓（不再多一个调用方开关 ✓ —— 少一个开关就少一种配错 ✓）。
    resident = {logical for logical in own if own[logical].dtype == torch.int8}
    #: ⚠️ 预检只验**这次真要写的**那些键 ✓✗（``meta`` 是**整份文件**的键 ✓ —— 里面有本模块根本
    #:    不装的层 ✓）。两个理由：
    #:    ① nvfp4 的打包权重**不能进** `precheck_dequant` ✗✗：那套判据是「形状自己就能证明布局」✓，
    #:       而打包 U8 在它眼里是「低精度权重 + 一个对不上任何布局的 scale」✗ ⇒ 必报「判不出布局」✓✗
    #:       （2026-10-30 实测：真 32B 文本编码器就是这样被拒的 ✓）。nvfp4 的判据在
    #:       :func:`quant_nvfp4.layer_plan` ✓（**等式**，与体检同一处 ✓）下面单独跑 ✓；
    #:    ② 更根本的：**别人层的问题不该拦这次装载** ✗ —— 从一个 350 层的文件里只装 1 层时，
    #:       另外 349 层若有一层 scale 怪，报错会指向一个**根本不会被写**的键 ✓✗（读的人无从下手 ✓）。
    #: ⚠️ 滤的**不是**键本身而是**模块** ✓✗：`quant.precheck_dequant` 要靠 `meta` 的键去找
    #: `weight_scale` / `.qweight` 这些**同模块**的伙伴 ✓ —— 只留 `own` 里的键会把伙伴也滤掉 ⇒
    #: 报成「找不到配套 scale」✓✗（本次实测：`engine_weights_streaming_test` ⑥/⑧ 当场变红 ✓）。
    pre_meta = {key: value for key, value in meta.items()
                if key not in packed and key.rsplit(".", 1)[0] in own_stems}
    pre = quant_mod.precheck_dequant(pre_meta, scales, convrot=convrot, int4=int4)
    report.dequant = pre.to_dict()
    if convrot is not None:
        #: 报告里必须**看得出**多了一步反旋转 ✓（否则「为什么比普通 int8 慢」无从解释 ✓）
        report.dequant["convrot"] = convrot
    if int4:
        #: 同理：**多一步 4bit 解包**也得写在报告里 ✓（`int4PackCount` 是**真的解了几层** ✓，
        #: 这里是「文件自称几层是打包的」✓ —— 两个数不相等就说明有层没解 ✓✗）
        report.dequant["int4Packed"] = sorted(int4)[:6]
        report.dequant["int4PackedCount"] = len(int4)
    #: nvfp4：**写第一个字节之前**把每一层都验一遍 ✓ ⇒ 不合格时模型**零污染** ✓
    #: （同 ③ 的理由 ✓：逐层装载到第 300 层才发现「块 scale 少一截」✗，前 299 层已经写进去了 ✓✗）
    nv_layers: dict[str, dict[str, Any]] = {}
    nv_problems: list[str] = []
    for logical in sorted(packed):
        stem = nv_logical[logical]
        scale_entry = entries.get(stem + nvfp4_mod.COMFY_SCALE_SUFFIX)
        scale_2_entry = entries.get(stem + nvfp4_mod.COMFY_SCALE_2_SUFFIX)
        pre_entry = entries.get(stem + nvfp4_mod.PRE_QUANT_SCALE_SUFFIX)
        if scale_entry is None or scale_2_entry is None:
            missing = "weight_scale" if scale_entry is None else "weight_scale_2"
            nv_problems.append(f"`{stem}` 是 nvfp4 打包权重（U8 ✓）但**缺** `{missing}` ✗")
            continue
        if logical in resident:
            #: 目标参数本身就是 int8 ✓ ⇒ 打包 U8 一 `copy_` 就是**静默截断** ✓✗（`int8 常驻` 那条路
            #: 只服务「文件里就是 int8 权重」的 convrot 主 DiT ✓，不是 nvfp4 ✓）
            nv_problems.append(f"`{stem}` 的目标参数是 int8 常驻 ✗，而文件里是 nvfp4 打包权重 ✓ "
                               f"⇒ 不能原样写入（会被静默截断 ✗）")
            continue
        item, issues = nvfp4_mod.layer_plan(
            stem,
            weight_shape=meta[logical][1], weight_dtype="U8",
            scale_shape=tuple(int(dim) for dim in scale_entry.shape), scale_dtype=scale_entry.dtype,
            scale_2_shape=tuple(int(dim) for dim in scale_2_entry.shape),
            scale_2_dtype=scale_2_entry.dtype,
            pre_quant_shape=None if pre_entry is None else tuple(int(dim) for dim in pre_entry.shape))
        nv_problems.extend(issues)
        if item is not None:
            item["stem"] = stem
            item["logical"] = logical
            nv_layers[logical] = item
    if resident:
        # ⚠️ 这些键**没有**反量化 ✓ ⇒ 不能出现在「反量化明细」里 ✗（否则同一份报告里
        #    「反量化了 200 个」与「200 个原样常驻」并存 ⇒ 自相矛盾 ✓✗）
        converted = {k: v for k, v in (report.dequant.get("converted") or {}).items()
                     if k not in resident}
        report.dequant["converted"] = converted
        report.dequant["count"] = len(converted)
        report.dequant["resident"] = len(resident)
    if nv_layers:
        #: 报告必须说得出「这 N 层是按 nvfp4 的**布局等式**还原的」✓（不是「看着像 U8 就照抄」✗）
        converted = dict(report.dequant.get("converted") or {})
        for logical, item in nv_layers.items():
            converted[logical] = {"layout": nvfp4_mod.FORMAT, "from": "torch.uint8",
                                  "scale": f"{item['stem']}{nvfp4_mod.COMFY_SCALE_SUFFIX}",
                                  "scale2": f"{item['stem']}{nvfp4_mod.COMFY_SCALE_2_SUFFIX}",
                                  "awq": bool(item["awq"]), "inverse": "False"}
        report.dequant["converted"] = converted
        report.dequant["count"] = len(converted)
        report.dequant["nvfp4"] = len(nv_layers)
    if not pre.ok:
        report.error = ("低精度权重**无法反量化** ✗ ⇒ 已**中止装载**（不按猜的布局算 ✗，"
                        "也不静默 cast ✗）：" + "；".join(pre.problems[:3]))
        return report
    if nv_problems:
        report.error = ("nvfp4 层**无法还原** ✗ ⇒ 已**中止装载**（写第一个字节之前 ✓，"
                        "不按猜的布局算 ✗）：" + "；".join(nv_problems[:3]))
        return report

    # ④ 逐张量：读 → 还原 → cast → 写入 ✓ ⇒ 峰值 ≈ **单个张量**（不是整份 ✗）
    #: ⭐ 读前置（2026-11-01 ✓，`prefetch` 的唯一接线点 ✓）：**当前这件在反量化/写卡时，
    #: 后面几件已经在盘上读** ✓ —— 省的是「等盘」那一段 ✓，不改变峰值（在飞的就 ``read_ahead`` 件、
    #: 且都 ≤ ``_READ_AHEAD_MAX_BYTES`` ✓）。
    #: ⚠️ 作用域由 :func:`_prefetch_scope` 管 ✓✗：出函数前 `close(wait=True)` 走完 ⇒ 不存在
    #: 「换装退干净了、预取的字节还挂着」那种账 ✗✗（跨阶段预取被明确否掉 ✓，见 :func:`_submit_ahead` ✓）。
    casts: dict[str, int] = {}
    written = 0
    #: ⭐ 单一真源 ✓：装配这一段与「内存这一级卸层之后的重读」走**同一个** :class:`_Restorer` ✓✗
    restorer = _Restorer(reader=reader, own=own, meta=meta, entries=entries,
                         resident=resident, nv_layers=nv_layers, int4=int4, convrot=convrot,
                         converted=pre.converted, scales=scales, dtype=dtype, orig_of=orig_of,
                         path=report.path)
    with torch.no_grad():
        for index, (orig, logical) in enumerate(plan):
            if prefetcher is not None:
                _submit_ahead(prefetcher, reader, plan, entries, index + 1, read_ahead)
            try:
                tensor, cast_from = restorer.value_of(orig, logical, prefetcher)
            except RestoreError as err:
                report.error = (f"{err}"
                                f" ⇒ 已写入 {written}/{len(plan)} 个键（模块处于**半装**状态 ✗）")
                return report
            own[logical].copy_(tensor)
            written += 1
            if logical in resident:
                #: ⭐ 4bit **打包**常驻层（``Int8Linear(packed4=True)`` ✓，2026-11-02 ✓）：
                #: 参数 dtype 也是 ``int8`` ✓、形状也是**文件里那个打包形状** ✓ ⇒ 与 int8 常驻
                #: 走**同一段原样写入** ✓（零特殊分支 ✓）；这里只为把两种账分开数 ✓：
                #: ``int8Resident`` 记「原样常驻的总层数」✓（:func:`_int8_residency_problems`
                #: 拿它对层数 ✓），``int4Resident`` 是其中「4bit 打包」的子计数 ✓。
                report.int8Resident += 1
                if orig.endswith(".weight") and orig[: -len(".weight")] in int4:
                    report.int4Resident += 1
            elif cast_from:
                casts[cast_from] = casts.get(cast_from, 0) + 1
            del tensor
    report.dtypeCasts = casts
    #: ⭐ 把还原器交出去 ✓（2026-11-02 ✓）：内存这一级（`residency` 的 ``host_cache``）
    #: 卸掉某层之后要靠它**重读** ✓ —— 走的是与上面**同一段**代码 ✓✗，
    #: 另写一份"只认 bf16 的简化版"就是给自己埋一个「数值偏了但不报错」✓✗
    report.restorer = restorer
    #: ⭐ 自动接上延迟绑定的重读入口 ✓✗（2026-11-02 ✓）：`residency` 在 ``place()`` 时
    #: （比装载**早** ✓）只能给一个空壳 :class:`residency.ReloadSlot` ✓，这里装完就绑 ✓。
    #: 走模块属性而不是函数参数，是为了让**13 处装载调用点一行都不用改** ✓✗。
    _bind_reload_slot(module, restorer)
    #: ⚠️ 4bit 常驻层的「账要对得上」✗✗（2026-11-02 ✓）：这些层**故意不解包** ✓
    #: ⇒ ``int4PackCount``（真解了几层）会比 ``int4PackedCount``（文件自称几层）**小** ✓。
    #: 不写明 ⇒ 下一次读报告的人会当成「有层漏解」✓✗，然后去修一个不存在的 bug ✓。
    if report.int4Resident:
        report.dequant["int4Resident"] = report.int4Resident
        report.dequant["int4ResidentNote"] = (
            f"{report.int4Resident} 层走 4bit **常驻壳** ✓（打包字节原样留在卡上 ✓，"
            f"解包挪到前向 ✓）⇒ 不算进 int4PackCount ✓")
    return report


def _bind_reload_slot(module: Any, restorer: Any) -> None:
    """装载成功后，把还原器接到**模块上那个空壳** ✓（见 :class:`residency.ReloadSlot` ✓）。

    三件都不做就静默返回 ✓：模块上没挂槽（没启用内存这一级 ✓）、还原器没造出来（整份路径 ✓）、
    槽不是槽（别的东西恰好用了同名属性 ✓）—— 都不该让一次**成功的装载**变成失败 ✗。
    """
    if restorer is None:
        return
    slot = getattr(module, "_residency_reload", None)
    binder = getattr(slot, "bind", None)
    if binder is None:
        return
    #: ⚠️ 接的是**还原器本身** ✓✗（不是 ``restorer.group_values`` ✓）：退件时
    #: `residency.Handle.detach` 要通过槽调 ``close()`` 把那把懒开的读口还回去 ✓，
    #: 而**绑定方法**上 `getattr(fn, "close")` 查的是底层函数的属性 ✓ ⇒ 查不到 ✓
    #: ⇒ 静默不关 ✓（症状＝Windows 上临时文件删不掉 / 长期跑掉 fd ✓）。
    binder(restorer)


def make_reloader(report: WeightLoadReport) -> Any:
    """从装载报告取出**重读入口** ✓ ⇒ ``Callable[[组名], {相对名: 张量}]`` ✓ 或 ``None`` ✓。

    直接喂给 ``residency.place(reload=…)`` / ``residency.attach(reload=…)`` ✓：

    ```python
    report = weights.load_module_weights(module, path)
    handle = residency.place(module, main_device="cuda", mode="split",
                             host_cache=host_cache.HostCache(quota),
                             reload=weights.make_reloader(report))
    ```

    ⚠️ ``None`` 必须**如实**返回 ``None`` ✗✗（不能返回一个空壳 lambda ✓）：
    ``residency`` 会因为「给了 host_cache 却没给 reload」当场拒 ✓；
    而空壳 lambda 会让它以为能重读 ✓、真卸了层之后再抛一个看不懂的 KeyError ✓✗。
    """
    restorer = getattr(report, "restorer", None)
    if restorer is None:
        return None
    #: 返回还原器**本身** ✓（它可调用 ⇒ 形状仍是"组名 → {相对名: 张量}" ✓，
    #: 而且它带 ``close()`` ✓ —— 裸的绑定方法不行，见 :func:`_bind_reload_slot` ✓）。
    return restorer


def save_module_weights(module: Any, path: str | Path, *,
                        metadata: dict[str, str] | None = None) -> str:
    """把模块权重写成 safetensors ✓（自检里造「真权重」用 ✓；也可用于导出 ✓）。"""
    from safetensors.torch import save_file  # noqa: PLC0415

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {key: tensor.detach().to("cpu").contiguous()
             for key, tensor in module.state_dict().items()}
    save_file(state, str(target), metadata=dict(metadata or {}))
    return str(target)
