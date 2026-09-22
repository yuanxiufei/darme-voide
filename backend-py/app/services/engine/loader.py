"""引擎的**加载计划**（零依赖 ✓）—— 从「文件体检」推进到「怎么把权重喂进去」。

## 与其它两层的关系（分工不重叠 ✓）

* :mod:`app.services.engine.safetensors` —— 读**一个文件**的头部（结构/截断 ✓）；
* :mod:`app.services.engine.inventory` —— 回答「**装齐没有 / 要下多少**」（就绪报告 ✓）；
* 本模块 —— 回答「**装齐之后怎么跑**」：每个文件是**什么角色**、**什么精度**、
  **要不要反量化**、以及最关键的 —— **显存怎么排班**（谁先进、谁先退 ✓）。

## ⚠️ 明确**不猜**的部分（这是本模块的纪律 ✗）

真实权重还没下载 ⇒ 本模块**不假设**张量名的具体前缀（写死 ``blocks.N.attn.wq`` 这种"看起来对"的
期望，只会在真机上错得莫名其妙 ✗）。它只做两件**可验证**的事：

1. **如实归纳文件里实际有什么**：把张量名按点号前缀分组（``blocks.12.attn`` ✓）并统计 ✓；
2. **校验自洽性** —— 这些检查不需要知道架构也能做，而且**恰好能抓住"没下完/下错"** ✗：
   * 数字段（``blocks.7.`` 里的 7 ✓）**从 0 开始、连续、不重** ⇒ 有洞说明文件被截断/拼接错 ✗；
   * 量化权重必须**配套**（``.weight`` 是 fp8/int8 ⇒ 应有对应的 ``.scale`` ✓）⇒ 缺 scale
     到推理时才炸 ✗（而且往往炸在显存里 ✗）；
   * 声明的字节数与文件大小一致（由 ``safetensors`` 层保证 ✓）。

⚠️ GGUF 此前只报「在/不在 + 大小」✗（读取器未实现 ✗）—— **已补齐** ✓（2026-09-20 ✓）：
现在 ``.gguf`` 走 :mod:`.gguf` 真读取（张量表 / 截断 / ``general.file_type`` 方案名 ✓），
与 safetensors 同一套「结构坏 ⇒ 阻断」口径 ✓。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import inventory as inv
from . import gguf as gguf_mod
from . import quant as quant_mod
from . import safetensors as st

__all__ = [
    "LoadPlan",
    "classify_dtype",
    "plan_component",
    "plan_stage",
    "prefix_groups",
    "residency_plan",
]

#: dtype → 类别（决定"能不能直接算"✓）
_DTYPE_CLASS: dict[str, str] = {
    "F64": "float", "F32": "float", "F16": "float", "BF16": "float",
    "F8_E4M3": "fp8", "F8_E5M2": "fp8",
    "I8": "int8", "U8": "int8",
    "I16": "int16", "U16": "int16", "I32": "int32", "U32": "int32",
    "I64": "int64", "U64": "int64", "BOOL": "bool",
}

#: 量化的"配套张量"命名约定（各家都用这套后缀 ✓）：权重是低精度 ⇒ 必须有 scale ✓
_SCALE_SUFFIXES = (".scale", ".scales", ".weight_scale", ".weight_scale_inv", ".zeros", ".qzeros")

#: 假定磁盘吞吐（GiB/s ✓）—— **只是估算** ✗，真机以实测为准 ✓
ASSUMED_READ_GIBPS = 1.2


def classify_dtype(dtype: str) -> str:
    """``F8_E4M3`` → ``fp8`` ✓（未知 dtype 返回 ``unknown`` ✓ 不猜 ✓）。"""
    return _DTYPE_CLASS.get(str(dtype).upper(), "unknown")


def prefix_groups(names: list[str], *, depth: int = 2) -> dict[str, int]:
    """按点号前缀分组计数 ✓：``blocks.12.attn.wq.weight`` → ``blocks.12.attn`` ✓。

    数字段会**保留**（``blocks.12`` 而不是 ``blocks.N`` ✓）⇒ 才能看出层数与是否有洞 ✓。
    """
    groups: dict[str, int] = {}
    for name in names:
        parts = str(name).split(".")
        key = ".".join(parts[:depth]) if len(parts) > depth else str(name)
        groups[key] = groups.get(key, 0) + 1
    return groups


def _numeric_slots(groups: dict[str, int], head: str) -> list[int]:
    """取出 ``head.<N>.`` 里的全部 N ✓（用于查「从 0 连续」✓）。"""
    pattern = re.compile(rf"^{re.escape(head)}\.(\d+)(?:\.|$)")
    found: list[int] = []
    for key in groups:
        match = pattern.match(key)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


@dataclass
class LoadPlan:
    """一个组件的**加载计划** ✓。"""

    key: str = ""
    name: str = ""
    role: str = ""
    path: str | None = None
    present: bool = False
    bytes: int = 0
    verified: bool = False
    tensorCount: int = 0
    dtypeClasses: dict[str, int] = field(default_factory=dict)
    computeDtype: str = ""
    quantScheme: str = "none"
    #: ⭐ 低精度权重的**自动还原计划** ✓（2026-09-22 ✓ `quant.plan_dequant` ✓）：
    #: 布局计数 / 判不出来的 / 缺 scale 的 / 分组量化的 ✓ + `supported` 一句话结论 ✓。
    #: ⚠️ **只靠形状就能算** ✗ ⇒ 体检阶段（没有 torch ✓）也能给出「这份 fp8 到手后能不能自动还原」✓。
    dequantPlan: dict[str, Any] = field(default_factory=dict)
    blockHead: str | None = None
    blockCount: int = 0
    loadMode: str = "full"
    readSecondsEstimate: float = 0.0
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def gib(self) -> float:
        return round(self.bytes / 1024 ** 3, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "role": self.role, "path": self.path,
            "present": self.present, "verified": self.verified, "gib": self.gib,
            "tensorCount": self.tensorCount, "dtypeClasses": self.dtypeClasses,
            "computeDtype": self.computeDtype, "quantScheme": self.quantScheme,
            "dequantPlan": dict(self.dequantPlan),
            "blockHead": self.blockHead, "blockCount": self.blockCount,
            "loadMode": self.loadMode, "readSecondsEstimate": self.readSecondsEstimate,
            "warnings": self.warnings, "problems": self.problems,
        }


def plan_component(entry: dict[str, Any], *, root: Path | None = None,
                   capacity_gib: float = inv.DEFAULT_CAPACITY_GIB) -> LoadPlan:
    """清单条目 → 加载计划 ✓（缺文件也是**结论** ✓ 不抛错 ✓）。"""
    plan = LoadPlan(key=str(entry.get("key") or ""), name=str(entry.get("name") or ""),
                    role=str(entry.get("kind") or ""))
    status = inv.component_status(entry, root)
    plan.path = status.get("path")
    plan.present = bool(status.get("present"))
    plan.bytes = int(status.get("bytes") or 0)
    # ⚠️ 「大小与清单不符」按 :mod:`.inventory` 的口径只是**可疑、不阻断** ✓（社区量化版/重新导出
    #    会让体积合法地变 ✓）⇒ 这里**降级成 warning** ✗，别混进 `problems` ✓。
    #    初版直接把 status 的全部 problems 抄进 problems ⇒ 于是 `plan_stage.ready` 因为
    #    「体积不符」变成 False ✗ —— 与 `inventory.readiness` 的判定**互相矛盾** ✗
    #    （自检 ㉓ 当场抓到 ✓）。**同一件事在两处必须有同一个口径** ✓。
    for problem in status.get("problems") or []:
        if "大小与清单不符" in problem:
            plan.warnings.append(problem.replace("⚠️", "（仅可疑，不阻断 ✓）"))
        else:
            plan.problems.append(problem)
    path = Path(plan.path) if plan.path else None

    if not plan.present or path is None:
        plan.loadMode = "missing"
        return plan

    if path.suffix.lower() == ".gguf":
        # ⚠️ 2026-09-20 起有**真读取器** ✓（此前只报 ``gguf-unknown`` ✗）：
        #    量化方案名来自 ``general.file_type`` ✓（Q4_K_M 这类打包名只在元数据里 ✓），
        #    截断 / 块不整除 / 重叠都进 ``problems`` ⇒ 与 safetensors 同一「阻断」口径 ✓。
        ginfo = gguf_mod.inspect(path)
        plan.verified = ginfo.ok
        plan.tensorCount = ginfo.tensor_count
        plan.dtypeClasses = dict(ginfo.dtype_counts)
        plan.quantScheme = ginfo.quant_scheme
        plan.computeDtype = (f"{ginfo.quant_scheme}（反量化由 ComfyUI-GGUF/llama.cpp 运行时完成 ✓）"
                             if ginfo.quant_scheme not in ("none", "unknown") else ginfo.quant_scheme)
        # ⚠️ GGUF 的量化是**块量化** ✓（Q4_K 一类 ✓ group size 在文件里 ✓）—— 本仓**不自研它的反量化** ✗
        #    （那要 llama.cpp/ComfyUI-GGUF ✓ 见 `gguf` 模块的边界 ✓）；这里只如实报「不归我管」✓，
        #    ⇒ `supported=False` 表示「**别指望 `quant.dequantize_state` 处理它**」✓（不是"坏了" ✗）。
        plan.dequantPlan = {"weights": plan.tensorCount if plan.quantScheme not in ("none", "unknown")
                            else 0, "paired": 0, "layouts": {}, "unresolved": [], "unresolvedCount": 0,
                            "unpaired": [], "unpairedCount": 0, "grouped": [], "groupedCount": 0,
                            "supported": False, "note": "GGUF：反量化由运行时（llama.cpp/ComfyUI-GGUF ✓）做 ✓"}
        plan.problems.extend(ginfo.problems)
        head, count, gaps = _block_gap_of(list(ginfo.tensors))
        plan.blockHead, plan.blockCount = head, count
        if gaps:
            plan.problems.append(gaps)
        plan.loadMode = _mode_for(plan.bytes, capacity_gib)
        plan.readSecondsEstimate = round(plan.gib / ASSUMED_READ_GIBPS, 1)
        if plan.quantScheme not in ("none", "unknown") and plan.loadMode != "full":
            plan.warnings.append("量化权重 + 显存吃紧 ⇒ 请**逐层反量化**（别整份反量化后再放 ✗）")
        return plan

    info = st.inspect(path)
    plan.verified = info.ok
    plan.tensorCount = info.tensor_count
    plan.dtypeClasses = {classify_dtype(dtype): count for dtype, count in info.dtype_counts.items()}
    plan.computeDtype, plan.quantScheme = _quant_of(info)
    head, count, gaps = _blocks_of(info)
    plan.blockHead, plan.blockCount = head, count
    if gaps:
        plan.problems.append(gaps)
    missing_scales = _missing_scales(info)
    if missing_scales:
        plan.problems.append(f"{len(missing_scales)} 个低精度权重缺少配套 scale（例如 "
                             f"{missing_scales[:2]}）⇒ 反量化时会炸 ✗")
    # ⭐ 反量化**能不能自动做**：由 `quant.plan_dequant` 按**形状**判 ✓（与装载时**同一套判据** ✗）
    plan.dequantPlan = quant_mod.plan_dequant(
        {name: (tensor.dtype, tensor.shape) for name, tensor in info.tensors.items()})
    if plan.dequantPlan["unresolvedCount"]:
        # ⚠️ 缺 scale 由上面那条报 ✓；这里报的是**布局判不出来**（块量化一类 ✓ 组大小在 json 里 ✗）
        plan.problems.append(
            f"{plan.dequantPlan['unresolvedCount']} 个低精度权重的 scale 形状**对不上任何布局** ✗"
            f"（例如 {plan.dequantPlan['unresolved']} ✓）⇒ 大概是**块量化** ✓，自动反量化会**中止装载** ✓"
            f"（不按猜的布局算 ✗）⇒ 需要随附 json 里的 group size ✓ 本仓不猜 ✗")
    if plan.dequantPlan["groupedCount"]:
        plan.problems.append(
            f"{plan.dequantPlan['groupedCount']} 个低精度权重带 `zeros`/`g_idx` ✗ ⇒ 分组量化 ✓"
            f"（例如 {plan.dequantPlan['grouped']} ✓）⇒ 组大小同样在 json 里 ✓ 本仓不猜 ✗")
    if plan.dequantPlan["supported"]:
        plan.warnings.append(
            f"低精度权重可**自动反量化** ✓（布局：{plan.dequantPlan['layouts']} ✓"
            f" —— 装的时候先还原再归一 dtype ✓）")
    plan.loadMode = _mode_for(plan.bytes, capacity_gib)
    plan.readSecondsEstimate = round(plan.gib / ASSUMED_READ_GIBPS, 1)
    if plan.quantScheme != "none" and plan.loadMode != "full":
        plan.warnings.append("量化权重 + 显存吃紧 ⇒ 请**逐层反量化**（别整份反量化后再放 ✗）")
    return plan


def _quant_of(info: st.SafetensorsInfo) -> tuple[str, str]:
    """从 dtype 分布 + 名字约定推断**计算精度**与**量化方案** ✓（推断不出就说 ``none``/``unknown`` ✓）。"""
    classes = {classify_dtype(dtype) for dtype in info.dtype_counts}
    names = list(info.tensors)
    has_scale = any(name.endswith(_SCALE_SUFFIXES) for name in names)
    if "fp8" in classes:
        return "fp8 (需反量化到 bf16/fp16 ✓)" if not has_scale else "fp8 + scales", "fp8"
    if "int8" in classes and has_scale:
        return "int8 (需反量化 ✓)", "int8"
    if "float" in classes:
        # 纯浮点：优先 bf16（新卡原生 ✓），否则看最大的那类
        floats = {dtype for dtype in info.dtype_counts if classify_dtype(dtype) == "float"}
        if "BF16" in floats:
            return "bf16", "none"
        if "F16" in floats:
            return "fp16", "none"
        return "fp32 (吃显存 ⚠️)", "none"
    return "unknown", "unknown"


def _blocks_of(info: st.SafetensorsInfo) -> tuple[str | None, int, str]:
    """找**层块**（``blocks.N.`` ✓）：返回 ``(头名, 层数, 问题描述)`` ✓。"""
    return _block_gap_of(list(info.tensors))


def _block_gap_of(names: list[str]) -> tuple[str | None, int, str]:
    """按张量名找**层块** ✓（safetensors 与 GGUF 两条路共用同一判据 ✓）。

    ⚠️ 这是**不依赖架构知识**的自洽性检查 ✓：数字段必须**从 0 开始且连续** ✓ ——
    有洞/重复几乎只有两种原因：**文件被截断** ✗ 或**权重被拼错** ✗，两者都该在加载前发现 ✓。
    """
    groups = prefix_groups(names, depth=2)
    # 取「``<头>.<数字>``」这一族里的**最大一支**（例如 ``blocks.N`` 有 30 个 ⇒ 就是它 ✓）
    # ⚠️ 不写死 `blocks` ✗ —— 不同实现叫 `blocks` / `layers` / `transformer_blocks` ✓ 都得认 ✓
    tally: dict[str, list[int]] = {}
    for key, _count in groups.items():
        head, _, tail = key.rpartition(".")
        if head and tail.isdigit():
            tally.setdefault(head, []).append(int(tail))
    if not tally:
        return None, 0, ""
    head = max(tally, key=lambda name: len(tally[name]))
    slots = sorted(tally[head])
    count = len(slots)
    problems: list[str] = []
    expected = list(range(count))
    if slots != expected:
        duplicates = sorted({value for value in slots if slots.count(value) > 1})
        missing = sorted(set(expected) - set(slots))
        detail = []
        if missing:
            detail.append(f"缺 {missing[:6]}")
        if duplicates:
            detail.append(f"重 {duplicates[:6]}")
        problems = [f"层块 {head}.N 的数字段**不连续**（{'；'.join(detail)}）⇒ "
                    f"典型是**文件被截断或拼错** ✗"]
    return head, count, problems[0] if problems else ""


def _missing_scales(info: st.SafetensorsInfo) -> list[str]:
    """低精度权重缺配套 scale ⇒ 反量化时才炸 ✗（提前抓 ✓）。

    判据用**逐张量的 dtype**（safetensors 层已经解析出来了 ✓）而不是猜名字 ✗：
    ``.weight`` 是 fp8/int8 且同 stem 下找不到任何 scale 后缀 ⇒ 才算缺 ✓。
    整份都是 bf16 的权重**不要求** scale（它本来就不需要 ✓）⇒ 不误报 ✓。
    """
    present = set(info.tensors)
    missing: list[str] = []
    for name, tensor in info.tensors.items():
        if not name.endswith((".weight", ".weight_orig")):
            continue
        if classify_dtype(tensor.dtype) not in ("fp8", "int8", "int16"):
            continue
        stem = name[: -len(".weight")] if name.endswith(".weight") else name[: -len(".weight_orig")]
        if not any(f"{stem}{suffix}" in present for suffix in _SCALE_SUFFIXES):
            missing.append(stem)
    return sorted(set(missing))


def _mode_for(size_bytes: int, capacity_gib: float) -> str:
    """显存策略（按**文件体积**判 ✓）：``full`` 整份驻留 / ``sequential`` 逐层进出 / ``streaming`` 必须流式。

    ⚠️ 初版乘了个"量化系数 1.35" ✗ ⇒ 把 **19.53 GiB 的 fp8 主权重**算成 26 GiB 判成「超尺寸」✗
    —— 这是错的 ✓：fp8 权重可以**逐层反量化**，不需要"整份反量化后再放" ✓。
    体积本来就是最靠谱的下界 ⇒ 直接用它判 ✓，反量化的开销改为**警告**（见调用处 ✓）。
    """
    gib = size_bytes / 1024 ** 3
    if gib <= capacity_gib * 0.75:
        return "full"
    if gib <= capacity_gib:
        return "sequential"
    return "streaming"


def residency_plan(components: list[LoadPlan], *, capacity_gib: float = inv.DEFAULT_CAPACITY_GIB,
                   order: tuple[str, ...] = ("text_encoders", "diffusion_models", "vae")) -> dict[str, Any]:
    """**显存排班** ✓ —— 谁必须先加载、谁可以在下一步前**释放** ✓。

    为什么这是重点：H3 主权重 **19.53 GiB** ✓ 而卡是 **24 GB** ✓ ⇒
    「全都在显存里」是**不成立**的 ✗，「文本编码器用完就退、再上 DiT、最后 VAE」才成立 ✓。
    这个顺序**不依赖任何架构知识** ✓（三个角色的生命周期本来就重叠不了 ✓）⇒ 零依赖也能算 ✓。
    """
    alive: list[str] = []
    steps: list[dict[str, Any]] = []
    peak = 0.0
    available = {component.role: component for component in components if component.present}
    sequence = [role for role in order if role in available] + \
               [role for role in available if role not in order]

    for role in sequence:
        component = available[role]
        # 每个角色用完即退（文本编码器在采样前就退 ✓；DiT 在解码前退 ✓）⇒ 峰值 = 单个最大 + 常驻
        alive.append(role)
        resident_gib = round(sum(available[item].gib for item in alive), 2)
        peak = max(peak, resident_gib)
        steps.append({
            "role": role, "key": component.key, "gib": component.gib,
            "loadMode": component.loadMode,
            "residentGiB": resident_gib, "fitsAtThisStep": resident_gib <= capacity_gib,
            "release": "本步完成后可释放 ✓（生命周期不重叠 ✓）",
        })
        alive.remove(role)   # 用完即退 ✓ ⇒ 峰值不会累加 ✗

    missing = [component.key for component in components if not component.present]
    biggest = max((component.gib for component in components), default=0.0)
    return {
        "capacityGiB": capacity_gib,
        "steps": steps,
        "peakResidentGiB": round(peak, 2),
        # ⚠️ 组件缺失时必须 `fits=False` ✗ —— 初版只比了体积 ⇒ **必需组件全缺**时反而回
        #    `fits: true`（因为"缺"的组件体积算 0 ✓）⇒ 前端会读成「能跑」✗，是**最坏的一种误导** ✗。
        "fits": bool(not missing) and peak <= capacity_gib and biggest <= capacity_gib,
        "missing": missing,
        "strategy": "sequential-residency" if peak > capacity_gib * 0.75 else "simple",
        "note": "顺序按「角色生命周期不重叠」推得（不依赖架构细节 ✓）；"
                "真正峰值仍取决于 offload/注意力/分块实现 ⇒ 以实测为准 ✓",
    }


def plan_stage(stage: str = "h3", *, root: Path | None = None,
               capacity_gib: float = inv.DEFAULT_CAPACITY_GIB) -> dict[str, Any]:
    """某阶段的**完整加载计划** ✓（= 逐组件计划 + 显存排班 ✓）。"""
    catalog = inv.load_catalog()
    rule = inv.STAGE_FILTERS.get(stage)
    entries = [item for item in catalog["models"] if not rule or item.get("category") == rule["category"]]
    components = [plan_component(entry, root=root, capacity_gib=capacity_gib) for entry in entries]
    required = [component for component, entry in zip(components, entries) if entry.get("required")]
    return {
        "stage": stage,
        "modelsDir": str(root or inv.models_dir() or ""),
        "components": [component.to_dict() for component in components],
        "residency": residency_plan(required, capacity_gib=capacity_gib),
        "ready": all(component.present and not component.problems for component in required),
    }
