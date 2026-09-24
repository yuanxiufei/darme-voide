r"""**反量化**（fp8 / int8 权重 → 可计算的高精度张量 ✓）—— 纯函数 ✓ 只依赖 torch ✓。

## 为什么必须有它 ✗

目标主权重是 **19.53 GiB 的 fp8** ✓（50 层 × [5376/14336] ✓ —— bf16 会是 ~39 GiB ✗）。
而 `load_module_weights` 现在的做法是 `tensor.to(dtype)` ✓✗ —— 对 fp8 张量来说这是**把尺度丢掉** ✗：
fp8 存的是「缩放过的整数/浮点」✓，**不乘 scale 直接 cast** ⇒ 得到一个**看着正常、数值全错**的模型 ✓✗
（而且**不会报错** ✗ —— `load_state_dict` 会照单全收 ✓，症状要到出片才发现「画面是噪声」✓✗）。

⚠️ **本模块的纪律：判不出来就拒绝** ✗（宁可报错 ✓ 也不按猜的布局反量化 ✓）：

* 布局**按形状判** ✓ 而不是按文件名猜 ✗（`scale` 是标量 ⇒ per-tensor ✓、`(out,)` ⇒ per-row ✓、
  `(in,)` ⇒ per-col ✓、与权重同形 ⇒ 逐元素 ✓）；
* 形状**对不上任何一种** ⇒ 拒绝 ✓（典型是**块量化**（block-wise ✓）：那要配套 json 里的 group size ✓
  本仓不猜 ✗）；
* `zeros`/`qzeros`（int4 一类 ✗）⇒ 拒绝 ✓（组大小同样在 json 里 ✓）；
* 低精度权重**没有**配套 scale ⇒ 拒绝 ✓（`loader` 只报「缺 scale」✓，这里**不装** ✗）。

## 两条 scale 方向约定 ✓

* `*.scale` / `*.scales` / `*.weight_scale` ⇒ ``w = q * s`` ✓；
* `*.weight_scale_inv` ⇒ ``w = q / s`` ✓（DeepSeek 系的写法 ✓ —— 名字里的 `inv` 就是「乘倒数」✓）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["DequantResult", "LOW_PRECISION_DTYPES", "SCALE_SUFFIXES", "UNSUPPORTED_FAMILIES",
           "dequantize_state", "is_low_precision", "layout_of", "plan_dequant", "quant_report",
           "unsupported_family"]

#: 需要反量化才能计算的 dtype ✓（**只有这些** ✗ —— bf16/fp16/fp32 本来就是可算精度 ✓）
#: ⚠️ 含**两套写法** ✓：torch 的 `float8_e4m3fn` ✓ 与 safetensors 头部的 `F8_E4M3` ✓（见 `is_low_precision` ✓）。
LOW_PRECISION_DTYPES: tuple[str, ...] = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
                                         "float8_e5m2fnuz", "int8", "uint8")

#: 归一化后的低精度名字集合 ✓（`is_low_precision` 用它 ✓）
_LOW_PRECISION_NAMES = frozenset(LOW_PRECISION_DTYPES) | {
    "f8_e4m3", "f8_e5m2", "f8_e4m3fn", "f8_e5m2fnuz", "f8_e4m3fnuz",
    "i8", "u8",
}

#: scale 的候选后缀 ✓（**顺序有意义** ✗：先找带 `_inv` 的 ✓ —— 否则 `weight_scale_inv` 会被
#: `weight_scale` **前缀命中** ✗ ⇒ 把「除」当成「乘」 ✓✗，那是最坏的一种错 ✓）
SCALE_SUFFIXES: tuple[str, ...] = (
    ".weight_scale_inv", ".weight_scale", ".scales", ".scale", ".scale_inv",
)

#: 明确**不做**的配套张量 ✗（要 group size ✓ 在随附 json 里 ✓ 本仓不猜 ✓）
_GROUP_SUFFIXES: tuple[str, ...] = (".zeros", ".qzeros", ".g_idx", ".qweight")


def is_low_precision(dtype: Any) -> bool:
    """这个 dtype 是不是「必须先反量化」✓（未知 dtype ⇒ ``False`` ✓ 不猜 ✓）。

    ⚠️ **两套写法都要认** ✗（2026-09-22 踩过 ✓）：**装载**路径拿的是 torch 的 ``float8_e4m3fn`` ✓，
    而**体检**路径拿的是 safetensors **头部里的字符串** ``F8_E4M3`` ✓ —— 只认前面那套 ⇒
    体检永远报「没有低精度权重」✓✗（而 `loader._DTYPE_CLASS` 用的是后面那套 ✓ 两处口径不同 ✗）。
    """
    return _normalize_dtype(dtype) in _LOW_PRECISION_NAMES


def _normalize_dtype(dtype: Any) -> str:
    """统一成小写、去掉 ``torch.`` 前缀 ✓（两套写法的**公共分母** ✓）。"""
    return str(dtype).replace("torch.", "").strip().lower()


@dataclass
class DequantResult:
    """一次反量化的结论 ✓（``ok=False`` ⇒ 调用方**不许继续装** ✗）。"""

    tensors: dict[str, Any] = field(default_factory=dict)
    #: 每个被反量化的权重 ⇒ (布局, 来源 scale 名 ✓)
    converted: dict[str, dict[str, str]] = field(default_factory=dict)
    #: 被吃掉的配套张量（它们**不进模型** ✓ —— 模型里没有这些键 ✓）
    companions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "count": len(self.converted), "converted": dict(self.converted),
                "companions": list(self.companions), "problems": list(self.problems)}


def layout_of(weight_shape: Any, scale_shape: Any) -> str | None:
    """按**形状**判 scale 的布局 ✓ ⇒ ``per-tensor`` / ``per-row`` / ``per-col`` / ``elementwise`` ✓。

    ⚠️ **纯形状** ✓ 不碰张量 ✗ ⇒ **体检阶段**（只有 safetensors 头部 ✓ 没有 torch 也能跑 ✓）与
    **装载阶段**用的是**同一句话** ✓ —— 判定分两处写就一定会漂 ✓✗（本仓踩过：`Split` 的拒绝理由 ✓）。

    ``None`` ⇒ **判不出来** ✓（块量化一类 ✓ 调用方**必须拒绝** ✗ 不猜 ✓）。
    """
    flat = tuple(int(dim) for dim in scale_shape)
    if flat in ((), (1,)):
        return "per-tensor"
    dims = tuple(int(dim) for dim in weight_shape)
    if len(dims) < 2:
        # 一维权重（norm / bias 一类 ✓）：只有**标量** scale 才说得清 ✓（向量 ⇒ 方向不明 ✗）
        return None
    rows, cols = dims[0], dims[1]
    if flat == (rows,):
        return "per-row"
    if flat == (cols,):
        return "per-col"
    if flat == (rows, cols):
        return "elementwise"
    return None


def plan_dequant(tensors: Mapping[str, Any]) -> dict[str, Any]:
    """**体检阶段**就能说的结论 ✓（零依赖 ✓ 不需要 torch ✓）：这份检查点的低精度权重**能不能自动还原** ✓。

    入参 ``tensors``：``{名字: (dtype 字符串, 形状)}`` ✓ —— ⚠️ **两项都要** ✗：
    只看形状说不出哪个张量是低精度 ✓✗，只看 dtype 说不出 scale 的布局 ✓✗。

    返回：``weights`` 低精度权重数 ✓ / ``paired`` 找到配套 scale 的 ✓ / ``layouts`` 各布局计数 ✓ /
    ``unresolved`` **判不出布局**的 ✓ / ``grouped`` 分组量化（带 `zeros`/`g_idx` ✓）的 ✓ /
    ``unpaired`` 缺 scale 的 ✓。

    ⚠️ 与 :func:`dequantize_state` **同一套判据** ✗（同一个 `layout_of` ✓ 同一张后缀表 ✓）——
    不然会出现「体检说能装、装载时被拒」✓✗（那种矛盾最费时间 ✓）。
    """
    names = list(tensors)
    count = 0
    paired = 0
    layouts: dict[str, int] = {}
    unresolved: list[str] = []
    unpaired: list[str] = []
    grouped: list[str] = []
    for name in sorted(name for name in names if name.endswith((".weight", ".weight_orig"))):
        dtype, shape = tensors[name]
        if not is_low_precision(dtype):
            continue
        count += 1
        stem = name[: -len(".weight")] if name.endswith(".weight") else name[: -len(".weight_orig")]
        if any(other.startswith(stem) and other.endswith(_GROUP_SUFFIXES) for other in names):
            grouped.append(stem)
            continue
        found = _scale_for(stem, tensors)
        if found is None:
            unpaired.append(stem)
            continue
        paired += 1
        layout = layout_of(shape, tensors[found[0]][1])
        if layout is None:
            unresolved.append(stem)
        else:
            layouts[layout] = layouts.get(layout, 0) + 1
    return {"weights": count, "paired": paired, "layouts": layouts,
            "unresolved": unresolved[:6], "unresolvedCount": len(unresolved),
            "unpaired": unpaired[:6], "unpairedCount": len(unpaired),
            "grouped": grouped[:6], "groupedCount": len(grouped),
            "supported": count > 0 and not (unresolved or unpaired or grouped)}


def _compose(weight: Any, scale: Any, *, inverse: bool) -> Any:
    """``w = q * s``（或 ``q / s`` ✓）—— 布局已由 :func:`layout_of` 定 ✓。"""
    torch = _torch()
    value = weight.to(torch.float32)
    factor = scale.to(torch.float32)
    rows = int(weight.shape[0])
    flat = tuple(int(dim) for dim in scale.shape)
    if flat == (rows,):
        factor = factor.reshape(rows, 1)
    elif len(tuple(int(d) for d in weight.shape)) >= 2 and flat == (int(weight.shape[1]),):
        factor = factor.reshape(1, -1)
    return value / factor if inverse else value * factor


def _scale_for(stem: str, present: Mapping[str, Any]) -> tuple[str, bool] | None:
    """给 ``stem`` 找配套 scale ✓ ⇒ ``(名字, 是否取倒数 ✓)``；找不到 ⇒ ``None`` ✓。

    ⚠️ 顺序**必须**是「长后缀优先」✗（见 `SCALE_SUFFIXES` 的注释 ✓）。
    """
    for suffix in SCALE_SUFFIXES:
        name = f"{stem}{suffix}"
        if name in present:
            return name, suffix.endswith("_inv")
    return None


def _apply_scale(weight: Any, scale: Any, *, inverse: bool, name: str,
                 problems: list[str]) -> tuple[Any, str] | None:
    """反量化一个权重 ✓ ⇒ ``(张量, 布局)``；**判不出来 ⇒ 记问题 + ``None``** ✓。

    ⚠️ 布局判定走 :func:`layout_of` **同一处** ✗（体检阶段也用它 ✓ —— 两处各写一份必然会漂 ✓✗）。
    在 **float32** 里算 ✓（scale 往往是小数量级 ✓，低精度里算会掉精度 ✓）。
    """
    torch = _torch()
    layout = layout_of(weight.shape, scale.shape)
    if layout is None:
        problems.append(
            f"`{name}` 的 scale 形状 {tuple(int(d) for d in scale.shape)} 与权重 "
            f"{tuple(int(d) for d in weight.shape)} **对不上任何一种布局** ✗"
            f"（per-tensor / per-row / per-col / 逐元素 ✓）⇒ 大概是**块量化** ✓"
            f"（group size 在随附 json 里 ✓ 本仓**不猜** ✗）⇒ 拒绝反量化 ✓")
        return None
    if inverse and bool((scale.to(torch.float32) == 0).any()):
        # ⚠️ 除以 0 ⇒ inf ✗ ⇒ 提前判（0 是**坏 scale** ✓，不是「合法的 1」✗）
        problems.append(f"`{name}` 的 scale 里有 0 ✗ ⇒ 反量化会出 inf ✓（不装 ✗）")
        return None
    return _compose(weight, scale, inverse=inverse), layout


def dequantize_state(state: Mapping[str, Any], *, target_dtype: Any = None) -> DequantResult:
    """把一份 ``state_dict`` 里的低精度权重反量化 ✓（**纯函数** ✓ 不改入参 ✓）。

    ⚠️ 只在**真需要**时才动 ✗：整份 bf16 的权重**原样返回** ✓（多一步 cast 只会白花时间 ✓）。
    """
    result = DequantResult()
    converted = dict(state)
    for key, tensor in list(state.items()):
        if not key.endswith((".weight", ".weight_orig")) or not is_low_precision(
                getattr(tensor, "dtype", None)):
            continue
        stem = key[: -len(".weight")] if key.endswith(".weight") else key[: -len(".weight_orig")]
        group = [name for name in state if name.startswith(stem) and name.endswith(_GROUP_SUFFIXES)]
        if group:
            result.problems.append(
                f"`{stem}` 带 {group[:2]} ✗ ⇒ 这是**分组量化**（int4 一类 ✓）⇒ 组大小在随附 json 里 ✓ "
                f"本仓**不猜** ✗（拒绝反量化 ✓ 不装 ✓）")
            continue
        found = _scale_for(stem, state)
        if found is None:
            result.problems.append(
                f"`{key}` 是低精度（{getattr(tensor, 'dtype', '?')} ✓）但**找不到配套 scale** ✗"
                f"（找过 {[f'{stem}{suffix}' for suffix in SCALE_SUFFIXES][:3]} … ✓）⇒ 不装 ✓")
            continue
        scale_name, inverse = found
        scaled = _apply_scale(tensor, state[scale_name], inverse=inverse, name=scale_name,
                              problems=result.problems)
        if scaled is None:
            continue
        value, layout = scaled
        if target_dtype is not None:
            value = value.to(target_dtype)
        converted[key] = value
        result.converted[key] = {"layout": layout, "scale": scale_name,
                                 "inverse": str(inverse), "from": str(getattr(tensor, "dtype", "?"))}
        result.companions.append(scale_name)
    # 配套张量**不进模型** ✓（模型里没有 `.scale` 这类键 ✓ ⇒ 留着只会变成 unexpected ✓）
    for name in set(result.companions):
        converted.pop(name, None)
    result.tensors = converted
    return result


def quant_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """**只看不装** ✓：这份 state 里有没有低精度权重 / 有没有配套 ✓（体检报告用 ✓）。"""
    low = sorted(key for key, tensor in state.items()
                 if key.endswith((".weight", ".weight_orig"))
                 and is_low_precision(getattr(tensor, "dtype", None)))
    stems = [key[: -len(".weight")] if key.endswith(".weight") else key[: -len(".weight_orig")]
             for key in low]
    paired = [stem for stem in stems if _scale_for(stem, state) is not None]
    return {"lowPrecision": len(low), "withScale": len(paired),
            "unpaired": [stem for stem in stems if _scale_for(stem, state) is None][:6],
            "dtypes": sorted({str(getattr(state[key], "dtype", "?")).replace("torch.", "")
                              for key in low})}


def _torch() -> Any:
    import torch  # noqa: PLC0415 —— 只有真要算时才 import ✓

    return torch


# ── 未实现的布局族：**认得出来就点名拒绝** ✗（别只说「判不出布局」✗）────────────────────────
# ⚠️ 2026-09-24 补 ✓（口径来自逆向 ✓）：H3 生态里确实存在 ``*_int8_convrot.safetensors`` 这一族
#    （卷积旋转量化 ✓ —— 模型文件名实锤 ✓）。本仓 :func:`layout_of` 只认「**形状能自证**」的几种布局 ✓
#    ⇒ 撞上它只会得到一句「判不出布局」✗ —— 那句话**指不到真因** ✗，还会让人以为「文件坏了 / 多试几次」✗。
#    ⇒ 补一层**具名识别** ✓：认得出来就说清「是什么 + 为什么不做」✓。
#: 已知但**未实现**的布局族 ✓（token → 为什么不做 ✓）。
#: ⚠️ 判据**只有名字** ✗：这些族在形状上自证不了自己 ✓（与 :func:`layout_of` 能自证的那几种不同 ✓）
#: ⇒ 调用方要**先**问本函数 ✓，再把 :func:`layout_of` 的失败当兜底 ✓。
UNSUPPORTED_FAMILIES: dict[str, str] = {
    "convrot": "卷积旋转量化（convrot ✗）—— 反量化必须**先按旋转参数把权重转回来** ✓，"
               "而权重文件里**没有**这个参数 ⇒ 本仓**不猜** ✗（猜错等于把权重解成噪声 ✓✗）",
}


def unsupported_family(*, source: str = "", tensor_names: Any = ()) -> str | None:
    """名字里带**已知但未实现**的布局族 ⇒ 返回可行动的拒绝理由 ✓；否则 ``None`` ✓。

    ``source`` 一般是权重**文件名** ✓；``tensor_names`` 是张量名（可迭代 ✓）。只做**子串匹配** ✓
    （大小写不敏感 ✓）—— ⚠️ 这不是「布局判定」✗，是「**这一族我们不支持**」的具名提示 ✓。
    """
    haystack = " ".join([str(source or "")] + [str(name) for name in tensor_names]).lower()
    for token, why in UNSUPPORTED_FAMILIES.items():
        if token in haystack:
            return (f"这个权重属于**未实现的布局族**「{token}」✗：{why}。"
                    f"⚠️ 这**不是**「判不出布局」✗ —— 不是文件坏了 ✓，是本仓**没实现**这一族 ✓；"
                    f"要么换成非 {token} 的量化产物 ✓，要么先把该族需要的参数（如旋转参数 ✓）拿到再实现 ✓。")
    return None
