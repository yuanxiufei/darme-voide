"""**超清模式（潜空间放大 + 二次去噪）的规划层** ✓（2026-09-24 补 ✓，口径来自逆向 ✓ 零依赖 ✓）。

## 它是什么（口径 ✓）
「超清模式」= 出片之后用**潜空间放大器**把 latent 放大 **2×** ✓，再走一遍**带掩码的轻量去噪** ✓
⇒ 输出分辨率翻倍（如 736×416 → 1472×832 ✓），代价是耗时 ≈**6×** ✓；
**权重缺失时自动回退普通模式** ✓（不阻断出片 ✓）。放大器权重要带**内嵌契约** ✓
（``format`` / ``strict_latent_only`` / ``base_config`` / ``config`` / ``step`` ✓ —— 见
:func:`read_upscaler_contract` ✓）。

## 三条已被踩过的规则 ✓（写进来，别重新踩 ✗）
1. **倍率只能按 2 的整数倍走** ✓：放大器做的是**空间 2×** ✓ ⇒ 想要「1.5×」就得**先做合法的 AI 2×**
   ✓、再**高质量缩放**到 1.5× ✓ —— 直接把 1.5 提交给放大器会被拒 ✓✗；
2. **4× 必须小分块 + 分阶段** ✓：大块解码会出 **NaN/Inf** ✓✗；
3. **权重缺失 ⇒ 回退普通模式** ✓，而且要**给理由** ✗ —— 静默降级会让用户以为「超清开了」（其实没开 ✓✗）。

## 不猜（本模块的边界 ✓）
* 放大器的**网络结构不在本模块** ✗（那是张量层 ✓，只有真权重才验得动 ✓）；
* **安全块大小不猜** ✗：要 4× 就**必须显式给** ``max_tile`` ✓，否则报错 ✓；
* 分块**是否需要重叠**未核 ✗ ⇒ 本模块只算块与覆盖 ✓，缝合策略留给张量层 ✓ 并标「未核」✗；
* 契约里的 ``base_config`` / ``config`` 只**原样带回** ✓，不解释字段语义 ✗（那是各自的模型配置 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import checkpoint_meta as meta_mod

__all__ = ["BASE_CONFIG_FIELDS", "CHECKPOINT_FORMAT", "CONFIG_FIELDS", "REQUIRED_CONTRACT_KEYS",
           "UpscalePlan", "check_upscaler_contract", "plan_upscale", "read_upscaler_contract",
           "tile_plan"]

#: 放大器检查点的内嵌契约**必须有**的键 ✓（口径来自逆向 ✓）。
REQUIRED_CONTRACT_KEYS: tuple[str, ...] = ("format", "base_config", "config")

#: ⭐ ``format`` 的**具体值** ✓（2026-09-24 核到 ✓ —— 上游实现里就写着这一串 ✓）。
#: ⚠️ 它同时说明了结构：``v3_factorized_attention`` ⇒ 放大器是 **V2 主干 + V3 因子化注意力** ✓。
CHECKPOINT_FORMAT = "minimax_h3_clean_latent_upscaler_v3_factorized_attention"

#: 两段配置的**字段集** ✓（⚠️ 判据是「**多一个少一个都拒**」✗ —— 上游就是这么严的 ✓）：
#: * ``base_config``（V2 主干）：``in_channels`` **必须 24** ✓、``temporal_kernel`` **必须奇数** ✓；
#: * ``config``（V3 因子化注意力）：``width`` 必须能被 ``heads`` 整除 ✓。
BASE_CONFIG_FIELDS: tuple[str, ...] = ("in_channels", "hidden_channels", "num_blocks",
                                       "refine_channels", "refine_blocks", "temporal_kernel")
CONFIG_FIELDS: tuple[str, ...] = ("width", "blocks", "heads", "window", "mlp_ratio")


def _check_config_block(raw: Any, fields_wanted: Sequence[str], *, label: str) -> dict[str, int]:
    """严格校验一段配置 ✓（字段集**多一个少一个都拒** ✗、值必须是**正整数** ✓ —— ``bool`` 不算 ✗）。"""
    if not isinstance(raw, Mapping):
        raise ValueError(f"放大器契约的 ``{label}`` 必须是对象 ✗（收到 {type(raw).__name__} ✓）")
    missing = sorted(set(fields_wanted) - set(raw))
    unknown = sorted(set(raw) - set(fields_wanted))
    if missing or unknown:
        raise ValueError(
            f"放大器契约的 ``{label}`` 字段对不上 ✗：缺 {missing} ✓ / 多 {unknown} ✓"
            f" —— ⚠️ 「多一个字段」也可能是**另一版检查点** ✓✗ ⇒ **宁可拒** ✗（拿新配置当旧配置读会静默错 ✓✗）")
    parsed: dict[str, int] = {}
    for name in fields_wanted:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"放大器契约的 ``{label}.{name}`` 必须是**正整数** ✗，实得 {value!r} ✓"
                f"（``bool`` / \"3\" / 0 / 负数都不收 ✓ —— 有值 ≠ 是对的类型 ✓✗）")
        parsed[name] = value
    return parsed


def check_upscaler_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """把已解出的契约**再往下核一层** ✓ ⇒ ``{"base_config": …, "config": …, "step": …}`` ✓。

    ⚠️ 这一层是**结构**校验（字段集 / 类型 / 值域 ✓），**不是**「能不能跑」✗（那要真权重 + 真张量 ✓）。
    """
    base = _check_config_block(contract.get("base_config"), BASE_CONFIG_FIELDS, label="base_config")
    if base["in_channels"] != 24:
        raise ValueError(
            f"放大器契约的 ``base_config.in_channels`` 必须是 **24** ✗，实得 {base['in_channels']} ✓"
            f" —— 这就是 H3 视频潜变量的通道数 ✓（与 `h3_form.H3_TRUNK_DEFAULTS['latents_dim']` 同一个事实 ✓）")
    if base["temporal_kernel"] % 2 != 1:
        raise ValueError(
            f"放大器契约的 ``base_config.temporal_kernel`` 必须是**奇数** ✗，实得 "
            f"{base['temporal_kernel']} ✓（时间维卷积要能对齐中心帧 ✓）")
    config = _check_config_block(contract.get("config"), CONFIG_FIELDS, label="config")
    if config["width"] % config["heads"]:
        raise ValueError(
            f"放大器契约的 ``config.width`` 必须能被 ``heads`` 整除 ✗：{config['width']} / "
            f"{config['heads']} ✓（注意力头要等分宽度 ✓）")
    step = contract.get("step")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError(f"放大器契约的 ``step`` 必须是非负整数 ✗，实得 {step!r} ✓")
    return {"base_config": base, "config": config, "step": step}


def read_upscaler_contract(metadata: Mapping[str, Any] | None,
                           *, expect_format: str | None = CHECKPOINT_FORMAT) -> dict[str, Any] | None:
    """读放大器检查点的内嵌契约 ✓ ⇒ 契约 dict ✓；**没有契约** ⇒ ``None`` ✓（**不是通过** ✗）。

    ⚠️ ``strict_latent_only`` **必须真是 True** ✓（有键 ≠ 为真 ✓✗ —— 松读会把它当已声明 ✓✗）；
    ⚠️ ``format`` 默认按 :data:`CHECKPOINT_FORMAT` 核 ✓（2026-09-24 核到真值 ✓ ——
    要读**别的**格式时显式传 ``expect_format=None`` ✓，但那样就**只剩**「有这个键」的松校验了 ✗）。
    """
    return meta_mod.read_contract(metadata, expect_format=expect_format,
                                  required=REQUIRED_CONTRACT_KEYS,
                                  truthy=("strict_latent_only",))


def tile_plan(width: int, height: int, *, max_tile: int, align: int = 32) -> tuple[tuple[int, int, int, int], ...]:
    """把画面切成**不超过 ``max_tile`` 的块** ✓ ⇒ ``((x, y, w, h), ...)`` ✓（左上原点 ✓）。

    * ``align``：块宽高**只取它的整数倍** ✓（默认 32 ✓ = 本仓分辨率步长 ✓，见
      :data:`app.services.engine.geometry.RESOLUTION_MULTIPLE` ✓）⇒ 块大小 = ``max_tile`` 向下取整到 align ✓；
    * ⚠️ **不做重叠** ✗（是否需要重叠**未核** ✗ —— 见模块头）；⚠️ 画面边长不是 align 的倍数时
      **最后一块**会是余数 ✓（真链路里画面本来就该对齐 ✓✗），并因此**报一条 note** ✓；
    * 覆盖是**精确划分** ✓（各块宽之和 == ``width`` ✓、高之和 == ``height`` ✓）—— 少了会缺边 ✗、多了会重复 ✗。
    """
    tile = (int(max_tile) // int(align)) * int(align)
    if tile <= 0:
        raise ValueError(f"max_tile={max_tile!r} 太小 ✗（要对齐到 {align} 的整数倍 ⇒ 至少要 {align} ✓）")
    for name, value in (("width", width), ("height", height)):
        if int(value) <= 0:
            raise ValueError(f"{name}={value!r} 必须为正 ✗")

    def axis(total: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        while cursor < total:
            span = min(tile, total - cursor)
            spans.append((cursor, span))
            cursor += span
        return spans

    return tuple((x, y, w, h) for y, h in axis(int(height)) for x, w in axis(int(width)))


@dataclass(frozen=True)
class UpscalePlan:
    """超清计划 ✓（``mode="normal"`` = **不启用**放大器 ✓，此时 ``fallback_reason`` 必须给 ✗✗）。"""

    mode: str
    scale: float
    steps: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    tiles: tuple[tuple[int, int, int, int], ...] = ()
    fallback_reason: str | None = None

    @property
    def uses_upscaler(self) -> bool:
        return self.mode != "normal"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "scale": self.scale, "usesUpscaler": self.uses_upscaler,
                "steps": list(self.steps), "notes": list(self.notes),
                "tiles": [list(tile) for tile in self.tiles],
                "fallbackReason": self.fallback_reason}


def plan_upscale(*, target_scale: float, contract: Mapping[str, Any] | None,
                 frame: Sequence[int] | None = None, max_tile: int | None = None,
                 align: int = 32) -> UpscalePlan:
    """决定「**能不能／怎么做**超清」✓ ⇒ :class:`UpscalePlan` ✓。

    * ``contract`` 是 :func:`read_upscaler_contract` 的结果 ✓；``None`` ⇒ **回退普通模式** ✓ 且给理由 ✗；
    * ``target_scale``：``2`` ⇒ 直接 AI 2× ✓；``1 < 目标 < 2``（如 1.5 ✓）⇒ **先 AI 2× 再缩放** ✓；
      ``≥4`` ⇒ **小分块**（必须给 ``max_tile`` ✓ 不给就报错 ✗）；``≤1`` ⇒ 不需要放大器 ✓。
    """
    scale = float(target_scale)
    if contract is None:
        return UpscalePlan(mode="normal", scale=1.0, fallback_reason=(
            "放大器**权重缺失 / 内嵌契约读不出来** ✗ ⇒ 自动回退普通模式 ✓（**不阻断出片** ✓；"
            "⚠️ 但必须让你看见 ✗ —— 静默降级会让人以为超清开着 ✓✗）"))
    if scale <= 1.0:
        return UpscalePlan(mode="normal", scale=1.0, fallback_reason=(
            f"目标倍率 {scale:g} ≤ 1 ⇒ **不需要**放大器 ✓（不是失败 ✗）"))
    if scale < 2.0:
        return UpscalePlan(
            mode="ai-2x-resize", scale=scale,
            steps=("AI 2×（合法的放大器倍率 ✓）", f"高质量缩放 2× → {scale:g}× ✓"),
            notes=(f"⚠️ 放大器只会做**空间 2×** ✗ ⇒ 直接把 {scale:g}× 提交给它**会被拒** ✓✗；"
                   f"所以是「先 2× 再缩」✓（口径来自逆向 ✓）",))
    if scale > 2.0:
        if max_tile is None:
            raise ValueError(
                "倍率 >2（如 4× ✓）**必须显式给 max_tile** ✗ ⇒ 本仓**不猜**安全块大小 ✓"
                "（大块解码会出 NaN/Inf ✓✗ —— 见模块头第 ② 条 ✓）")
        if not frame or len(frame) != 2:
            raise ValueError("倍率 >2 还要给 frame=(宽, 高) ✓ 才能算分块 ✓（否则不知道切几块 ✗）")
        tiles = tile_plan(int(frame[0]), int(frame[1]), max_tile=int(max_tile), align=align)
        notes = [f"⚠️ 大块解码会产生 **NaN/Inf** ✗ ⇒ 走**小分块 + 分阶段** ✓"
                 f"（本次 {len(tiles)} 块 ✓，块上限 {max_tile} ✓）",
                 "⚠️ 分块**是否需要重叠未核** ✗ ⇒ 缝合策略留给张量层 ✓"]
        if int(frame[0]) % int(align) or int(frame[1]) % int(align):
            notes.append(f"⚠️ 画面边长不是 {align} 的整数倍 ✗ ⇒ 最后一块是余数 ✓"
                         f"（真链路里画面本来就该对齐 ✓✗）")
        return UpscalePlan(mode="tiled-ai-2x", scale=scale, tiles=tiles, notes=tuple(notes),
                           steps=("AI 2×（分块 ✓）", f"带掩码的二次去噪（先 {scale:g}× 的目标域 ✓）"))
    return UpscalePlan(mode="ai-2x", scale=2.0,
                       steps=("AI 2×（潜空间放大器 ✓）", "带掩码的二次去噪 ✓"),
                       notes=("输出分辨率翻倍 ✓、耗时 ≈**6×** ✓（口径来自逆向 ✓）",))


def covered_area(tiles: Sequence[Sequence[int]]) -> int:
    """分块覆盖面积 ✓（**自检用**：必须等于画面面积 ✓ —— 少了缺边 ✗、多了重复 ✗）。"""
    return int(sum(int(tile[2]) * int(tile[3]) for tile in tiles))


def is_partition(tiles: Sequence[Sequence[int]], width: int, height: int) -> bool:
    """分块是不是画面的**精确划分** ✓。

    ⚠️ 判据是**逐行首尾相接** ✓，不是「面积对得上」✗✗ —— 面积相等但**错位/重叠**的组合太多了 ✓✗
    （本模块第一版就是拿面积糊的 ✗：漏了错位这一整类错 ✓）。
    规则：行按 ``y`` 排好后必须从 0 开始、每行高一致 ✓、``y+h`` 接着下一行、最后正好到 ``height`` ✓；
    每行内各块按 ``x`` 排好后同理正好拼满 ``width`` ✓。
    """
    tiles = [tuple(int(value) for value in tile[:4]) for tile in tiles]
    if not tiles:
        return False
    width, height = int(width), int(height)
    rows: dict[int, list[tuple[int, int]]] = {}
    row_heights: dict[int, set[int]] = {}
    for x, y, w, h in tiles:
        if w <= 0 or h <= 0:
            return False
        rows.setdefault(y, []).append((x, w))
        row_heights.setdefault(y, set()).add(h)
    cursor = 0
    for y in sorted(rows):
        if y != cursor or len(row_heights[y]) != 1:
            return False
        inner = 0
        for x, w in sorted(rows[y]):
            if x != inner:
                return False
            inner += w
        if inner != width:
            return False
        cursor = y + row_heights[y].pop()
    return cursor == height
