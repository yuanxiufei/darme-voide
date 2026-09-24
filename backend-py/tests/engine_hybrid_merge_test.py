"""S7 自检：**混合加载计划层**（fl2va 基底 + ref2va 的 adaLN 覆盖 ✓ 零依赖 ✓ 2026-09-24）。

钉的是「**四类情况必须分得清**」✓（混在一起就一定会静默出错 ✗✗）：

* 能盖 ⇒ 进 ``take`` ✓；
* **不是 adaLN** ⇒ **跳过并给理由** ✓（静默忽略 = 用户以为覆盖生效了 ✓✗）；
* **基底里没有这个键** ⇒ **拒** ✗✗（说明两边不是同族 ✓ —— 混模型/混版本就靠这条拦 ✓）；
* **形状不一致** ⇒ **拒** ✗（硬塞只会换来形状错 ✓✗）；
* ⚠️ 匹配器**可传** ✓（默认只到「名字里有 adaln」这一层 ✗ —— 本套用**自定义匹配器**证明它真生效 ✓，
  否则那个参数就是死的 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_hybrid_merge_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import hybrid_merge as hm  # noqa: E402

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


BASE = {
    "blocks.0.adaln_proj.linear.weight": (2688, 5376),
    "blocks.0.mlp.fc2.weight": (5376, 14336),
    "final_layer.adaln.weight": (2688, 5376),
    "video_patch_proj.weight": (5376, 96),
}
OVERLAY = {
    "blocks.0.adaln_proj.linear.weight": (2688, 5376),
    "final_layer.adaln.weight": (2688, 5376),
}


def case_happy() -> None:
    """① 正常：覆盖层**只带 adaLN** ✓ ⇒ 全进 ``take`` ✓、``skip`` 空 ✓。"""
    # ⚠️ 夹具里**故意**放了 ``final_layer`` 那条 ✓ ⇒ 默认它**就该被跳过** ✗
    #    （本套第一版却断言「``skip`` 为空」✗ —— 又是拿「我以为的行为」当判据 ✓✗）。
    plan = hm.plan_hybrid_merge(BASE, OVERLAY)
    check("① 两份只差 adaLN ⇒ ``take`` 恰好是要盖的那条（按名排序 ✓）；⚠️ ``final_layer`` 那条"
          "默认**不进** ✓（口径：参考实现那个开关默认不带 final ✓）且**给了理由** ✓",
          plan.take == ("blocks.0.adaln_proj.linear.weight",)
          and [name for name, _reason in plan.skip] == ["final_layer.adaln.weight"]
          and "final" in plan.skip[0][1], plan.to_dict())
    enabled = hm.plan_hybrid_merge(BASE, OVERLAY, include_final_adaln=True)
    check("①″ 开了 ``include_final_adaln`` ⇒ 那条进 ``take`` ✓（开关真的有用 ✓ 不是摆设 ✗）",
          "final_layer.adaln.weight" in enabled.take, enabled.to_dict())


def case_skips() -> None:
    """② 跳过必须**带理由** ✓（静默忽略 = 用户以为覆盖生效了 ✓✗）。"""
    plan = hm.plan_hybrid_merge(BASE, {**OVERLAY, "blocks.0.mlp.fc2.weight": (5376, 14336)})
    check("② 覆盖层带了**非 adaLN** 的键 ⇒ 跳过 ✓ 且理由点名「不是 adaLN」✓"
          "（不是静默 ✗）、``take`` 里**没有**它 ✓",
          any(name == "blocks.0.mlp.fc2.weight" and "不是 adaLN" in reason
              for name, reason in plan.skip)
          and "blocks.0.mlp.fc2.weight" not in plan.take, plan.to_dict())
    empty = hm.plan_hybrid_merge(BASE, {})
    check("②′ 空覆盖层 ⇒ ``take`` 空 ✓ 且**不报错** ✓（「没得合并」是正常情况 ✓）",
          empty.take == () and not empty.skip and empty.to_dict()["overlayCount"] == 0)


def case_refusals() -> None:
    """③ 两种**必须拒**的情况 ✓✗：键不在基底里 / 形状不一致。"""
    stranger = _raises(lambda: hm.plan_hybrid_merge(
        BASE, {"blocks.0.adaln_proj.linear.weight": (2688, 5376),
               "blocks.99.adaln.extra.weight": (10, 10)}), "不是同族")
    check("③ ⭐ 覆盖层有个基底**没有**的 adaLN 键 ⇒ **拒** ✗✗ 且理由说清「不是同族权重」✓"
          "（混模型/混版本就靠这条拦 ✓）",
          stranger is not None and "fl2va" in stranger, stranger)
    mismatch = _raises(lambda: hm.plan_hybrid_merge(
        BASE, {"blocks.0.adaln_proj.linear.weight": (2688, 2688)}), "形状对不上")
    check("③′ ⭐ 形状不一致 ⇒ **拒** ✗ 且**两个形状都印出来** ✓（只印一个没法判断谁错 ✓）",
          mismatch is not None and "(2688, 5376)" in mismatch and "(2688, 2688)" in mismatch,
          mismatch)


def case_matcher() -> None:
    """④ ⚠️ 匹配器**可传**且真生效 ✓（默认只到「名字里有 adaln」这一层 ✗ ⇒ 得能换 ✓）。"""
    custom = hm.plan_hybrid_merge(
        BASE, {"blocks.0.mlp.fc2.weight": (5376, 14336)},
        matcher=lambda name: name.endswith("mlp.fc2.weight"))
    check("④ 自定义匹配器 ⇒ 被它认成 adaLN 的键进 ``take`` ✓（证明这个参数**不是摆设** ✗）",
          custom.take == ("blocks.0.mlp.fc2.weight",) and not custom.skip, custom.to_dict())
    check("④′ 默认匹配器**大小写不敏感** ✓（``ADALN`` 也认 ✓ —— 权重命名风格不统一是常态 ✓）",
          hm.matches_adaln("blocks.0.ADALN_proj.weight") is True
          and hm.matches_adaln("blocks.0.mlp.weight") is False)


def case_apply() -> None:
    """⑤ 合并**映射** ✓：只覆盖 ``take`` 的键 ✓、**不改原映射** ✓、计划与数据对不上要报 ✗。"""
    plan = hm.plan_hybrid_merge(BASE, OVERLAY, include_final_adaln=True)
    sentinel = "OVERLAY"
    merged = hm.apply_merge(BASE, {name: sentinel for name in OVERLAY}, plan)
    check("⑤ 只覆盖 ``take`` 里的键 ✓、基底其余键**原样** ✓、值搬的是覆盖层那份 ✓",
          merged["blocks.0.adaln_proj.linear.weight"] == sentinel
          and merged["final_layer.adaln.weight"] == sentinel
          and merged["blocks.0.mlp.fc2.weight"] == BASE["blocks.0.mlp.fc2.weight"], merged)
    check("⑤′ ⭐ **纯函数** ✓：原映射**一个键都没动** ✗（改到原字典会污染调用方 ✓✗）",
          BASE["blocks.0.adaln_proj.linear.weight"] == (2688, 5376)
          and set(merged) == set(BASE))
    stale = hm.MergePlan(take=("blocks.0.no_such.weight",), skip=())
    check("⑤″ 拿**过期/对不上**的 MergePlan 去合并 ⇒ 报错 ✓（别把错计划静默跑完 ✗）",
          _raises(lambda: hm.apply_merge(BASE, OVERLAY, stale), "对不上") is not None)


def main() -> int:
    case_happy()
    case_skips()
    case_refusals()
    case_matcher()
    case_apply()
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
