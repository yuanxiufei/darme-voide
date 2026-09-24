"""S7 自检：**超清模式的规划层**（契约校验 / 倍率规则 / 小分块 ✓ 零依赖 ✓ 2026-09-24）。

钉的三条都是**已经踩过的** ✓（逆向里写着的 ✓）：

* ⭐ **权重缺失 / 契约读不出来 ⇒ 回退普通模式** ✓ 但**必须给理由** ✗（静默降级会让用户以为
  超清开着 ✓✗）；且 `strict_latent_only` **必须真是 True** ✓（有键 ≠ 为真 ✓✗）；
* ⭐ **1.5× 不直接提交** ✗：先做**合法的 AI 2×** ✓ 再高质量缩放 ✓（直接给 1.5 会被拒 ✓✗）；
* ⭐ **4× 必须小分块** ✓ 且**不给块上限就报错** ✗（不猜安全块大小 ✓）；分块必须是画面的
  **精确划分** ✓ —— 判据是**逐行首尾相接** ✓，**不是**「面积对得上」✗✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_upscale_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import upscale as up  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def _meta(contract: Any, *, as_json: bool = True) -> dict[str, Any]:
    return {"metadata": json.dumps(contract) if as_json else contract}


#: ⚠️ 夹具必须用**真实形状** ✗（字段集是**严的** ✓：多一个少一个都会拒 ✓ ——
#: 本套第一版用的是「拍脑袋字段」✗ ⇒ 加了严格校验后当场全红 ✓）。
GOOD = {"format": up.CHECKPOINT_FORMAT, "strict_latent_only": True,
        "base_config": {"in_channels": 24, "hidden_channels": 128, "num_blocks": 6,
                        "refine_channels": 64, "refine_blocks": 2, "temporal_kernel": 3},
        "config": {"width": 256, "blocks": 4, "heads": 8, "window": 8, "mlp_ratio": 2},
        "step": 12000}


def case_contract() -> None:
    """① 契约校验 ✓（读不出来就回退 ⇒ **理由必须能说清** ✗）。"""
    contract = up.read_upscaler_contract(_meta(GOOD))
    check("① 契约齐全 ⇒ 原样带回 ✓（含 ``base_config``/``config`` ✓ —— 语义不解释 ✗）",
          contract == GOOD, contract)
    check("①′ 没有 ``metadata`` ⇒ ``None`` ✓（⚠️ **不是通过** ✗ —— 调用方据此回退 ✓）",
          up.read_upscaler_contract({}) is None and up.read_upscaler_contract(None) is None)
    check("①″ ``format`` 给错 ⇒ 报错（**两个值都印出来** ✓）",
          _raises(lambda: up.read_upscaler_contract(_meta(GOOD), expect_format="h3-v2"),
                  "h3-v2") is not None)
    off = _raises(lambda: up.read_upscaler_contract(
        _meta({**GOOD, "strict_latent_only": False})), "有键 ≠ 为真")
    check("①‴ ⭐ ``strict_latent_only`` 为 ``False`` ⇒ 报错 ✓（有键 ≠ 为真 ✓✗）；"
          "缺 ``base_config`` ⇒ 也报错 ✓",
          off is not None
          and _raises(lambda: up.read_upscaler_contract(
              _meta({key: value for key, value in GOOD.items() if key != "base_config"})),
              "base_config") is not None, off)


def case_scale_rules() -> None:
    """② 倍率规则 ✓：2× 直通 ✓ / 1.5× **先 2× 再缩** ✓ / >2× **必须给块上限** ✗ / ≤1× 不需要 ✓。"""
    two = up.plan_upscale(target_scale=2, contract=GOOD)
    check("② 2× ⇒ ``ai-2x`` ✓（潜空间放大器 + 带掩码二次去噪 ✓）且说明里**点名 6× 耗时** ✓",
          two.mode == "ai-2x" and two.uses_upscaler
          and any("6×" in note for note in two.notes), two.to_dict())
    one_five = up.plan_upscale(target_scale=1.5, contract=GOOD)
    check("②′ ⭐ 1.5× ⇒ ``ai-2x-resize`` ✓ 且说明里点出「直接把 1.5 提交**会被拒**」✗"
          "（口径来自逆向 ✓）",
          one_five.mode == "ai-2x-resize" and one_five.scale == 1.5
          and any("会被拒" in note for note in one_five.notes)
          and len(one_five.steps) == 2, one_five.to_dict())
    no_tile = _raises(lambda: up.plan_upscale(target_scale=4, contract=GOOD), "不猜")
    check("②″ ⭐ 4× **不给 ``max_tile``** ⇒ 报错且写明**不猜**安全块大小 ✓"
          "（大块解码会出 NaN/Inf ✓✗）",
          no_tile is not None, no_tile)
    four = up.plan_upscale(target_scale=4, contract=GOOD, frame=(1920, 1080), max_tile=512)
    check("②‴ 4× 且给了块上限 ⇒ ``tiled-ai-2x`` ✓、说明里点出 NaN/Inf 与「重叠**未核**」✗",
          four.mode == "tiled-ai-2x" and four.tiles
          and any("NaN" in note for note in four.notes)
          and any("未核" in note for note in four.notes), four.to_dict())
    check("②⁴ 目标 ≤1（如 1.0 ✓）⇒ ``normal`` ✓ 且理由是「**不需要**」而不是失败 ✓",
          up.plan_upscale(target_scale=1.0, contract=GOOD).mode == "normal"
          and "不需要" in (up.plan_upscale(target_scale=1.0, contract=GOOD).fallback_reason or ""))
    none = up.plan_upscale(target_scale=2, contract=None)
    check("②⁵ ⭐⭐ 权重缺失 / 契约读不出来 ⇒ ``normal`` ✓ 且 ``fallback_reason`` **必须非空** ✓✗"
          "（静默降级 = 用户以为超清开着 ✓✗）",
          none.mode == "normal" and not none.uses_upscaler and bool(none.fallback_reason)
          and "不阻断出片" in none.fallback_reason, none.to_dict())


def case_tiling() -> None:
    """③ 分块 ✓：**精确划分** ✓、不超块上限 ✓、对齐 ✓。"""
    # ⚠️ 用**真实的放大后尺寸** ✓（736×416 的 2× ✓）—— 顺带记住：1080 **不是** 32 的倍数 ✗
    #    （1080/32 = 33.75 ✓），拿它当「本来就对齐」是错的 ✓✗（本套真撞过 ✓）。
    tiles = up.tile_plan(1472, 832, max_tile=512, align=32)
    check("③ 1472×832（真放大后尺寸 ✓）/ 块上限 512 / 对齐 32 ⇒ **精确划分** ✓（逐行首尾相接 ✓）",
          up.is_partition(tiles, 1472, 832), tiles)
    check("③′ 每块都 ≤ 块上限 ✓、宽高都是 32 的整数倍 ✓（画面本身对齐时最后一块也对齐 ✓）",
          all(w <= 512 and h <= 512 and w % 32 == 0 and h % 32 == 0 for _x, _y, w, h in tiles),
          tiles)
    check("③″ 块数 = 「够盖住」的最少块数 ✓（1472→3 列、832→2 行 ⇒ 6 块 ✓）",
          len(tiles) == 6, len(tiles))
    odd = up.tile_plan(1000, 1000, max_tile=512, align=32)
    check("③‴ 画面边长**不是**对齐倍数（1000 ✓）⇒ 仍能**精确划分** ✓（最后一块是余数 ✓ 并说明 ✓）",
          up.is_partition(odd, 1000, 1000)
          and any("余数" in note for note in up.plan_upscale(
              target_scale=4, contract=GOOD, frame=(1000, 1000), max_tile=512).notes), odd)
    bad = [(0, 0, 1000, 500), (0, 400, 1000, 500)]
    check("③⁴ 反例（**套套逻辑守卫** ✓）：面积**相等**但**重叠 / 没盖到底**的分块 ⇒ "
          "``is_partition`` 必须为假 ✗ —— 这正是「拿面积糊过去」会漏掉的一整类错 ✓✗",
          up.covered_area(bad) == 1000 * 1000 and up.is_partition(bad, 1000, 1000) is False,
          (up.covered_area(bad), up.is_partition(bad, 1000, 1000)))
    check("③⁵ 块上限太小（小于对齐步长 ✓）⇒ 报错 ✓（不静默给个 0 宽的块 ✗）",
          _raises(lambda: up.tile_plan(1920, 1080, max_tile=16, align=32), "太小") is not None
          and _raises(lambda: up.tile_plan(0, 1080, max_tile=512), "必须为正") is not None)


def case_deep_contract() -> None:
    """④ **契约再往下核一层** ✓（字段集 / 类型 / 值域 ✓ —— 格式串 2026-09-24 已核到真值 ✓）。"""
    check("④ ``format`` 的真值已钉住 ✓（``…_v3_factorized_attention`` ✓ ⇒ 结构 = V2 主干 + V3 因子化注意力 ✓）"
          "且默认按它核 ✓（错 format ⇒ 报错 ✓）",
          up.CHECKPOINT_FORMAT.endswith("v3_factorized_attention")
          and _raises(lambda: up.read_upscaler_contract(_meta({**GOOD, "format": "别的"})), "别的")
          is not None)
    parsed = up.check_upscaler_contract(GOOD)
    check("④′ 正常契约 ⇒ 两段都解析出来 ✓（``base_config`` / ``config`` / ``step`` 原样带回 ✓）",
          parsed["base_config"]["in_channels"] == 24 and parsed["config"]["heads"] == 8
          and parsed["step"] == 12000, parsed)
    check("④″ ⭐ ``in_channels`` 不是 24 ⇒ 报错并**点名 24** ✓（这是 H3 视频潜变量的通道数 ✓，"
          "与 `h3_form` 的 `latents_dim` 同一个事实 ✓）",
          _raises(lambda: up.check_upscaler_contract(
              {**GOOD, "base_config": {**GOOD["base_config"], "in_channels": 48}}), "24") is not None)
    check("④‴ ``temporal_kernel`` 偶数 / ``width % heads != 0`` ⇒ 各自报错 ✓",
          _raises(lambda: up.check_upscaler_contract(
              {**GOOD, "base_config": {**GOOD["base_config"], "temporal_kernel": 4}}), "奇数") is not None
          and _raises(lambda: up.check_upscaler_contract(
              {**GOOD, "config": {**GOOD["config"], "heads": 7}}), "整除") is not None)
    msg = _raises(lambda: up.check_upscaler_contract(
        {**GOOD, "base_config": {**GOOD["base_config"], "extra": 1, "num_blocks": None}}), "多")
    check("④⁴ ⭐ 字段**多一个或少一个都拒** ✗ 且**两边都点名** ✓（⚠️「多一个字段」可能是**另一版检查点** ✓✗"
          " ⇒ 拿新配置当旧配置读会静默错 ✓）",
          msg is not None and "缺" in msg, msg)
    check("④⁵ 值必须是**正整数** ✓：``bool`` / ``0`` / 字符串数字 ⇒ 都拒 ✗（有值 ≠ 是对的类型 ✓✗）",
          all(_raises(call) is not None for call in (
              lambda: up.check_upscaler_contract(
                  {**GOOD, "config": {**GOOD["config"], "blocks": True}}),
              lambda: up.check_upscaler_contract(
                  {**GOOD, "config": {**GOOD["config"], "blocks": 0}}),
              lambda: up.check_upscaler_contract(
                  {**GOOD, "config": {**GOOD["config"], "blocks": "4"}}))))


def main() -> int:
    case_contract()
    case_deep_contract()
    case_scale_rules()
    case_tiling()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
