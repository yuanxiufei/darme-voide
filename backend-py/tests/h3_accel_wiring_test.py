"""S7 自检：**加速链接到真 ComfyUI 注册表**（接缝 ✓ 零依赖、零真实网络 ✓ 2026-09-24）。

⚠️ 用**假客户端** ✓（只实现 ``object_info_all()`` 这一个方法 ✓）—— 真 HTTP 那条路由
``h3_comfyui_test.py`` 覆盖 ✓，本套只管**接缝判据** ✓（本仓规矩：测试不许依赖真机装了什么 ✗）。

钉的是：

* ⭐ **ComfyUI 连不上 ⇒ 判「没查」（``unchecked``）** ✗ —— **不是**判错 ✓✗（连不上 ≠ 节点没装 ✓）；
  且此时**不接线** ✓（输出槽位要从注册表数出来 ✓ 不猜 ✗）；
* ⭐ 判 ``error`` 的项 ⇒ ``blocked=True`` 且 ``build=None`` ✗（**不静默少一环** ✓），理由随报告给出 ✓；
* ``warn`` 不拦 ✓（能提交 ✓ —— 与 ``error`` 必须分得开 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/h3_accel_wiring_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.local_services.h3 import accel as accel_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


NODE_TYPES: dict[str, Any] = {
    "LoraLoader": {"name": "LoraLoader",
                   "input": {"required": {"model": ["MODEL"], "clip": ["CLIP"],
                                          "lora_name": ["STRING", {"default": ""}],
                                          "strength_model": ["FLOAT", {"default": 1.0}],
                                          "strength_clip": ["FLOAT", {"default": 1.0}]}},
                   "output": ["MODEL", "CLIP"]},
    "TESpeedMiniMaxH3": {"name": "TESpeedMiniMaxH3",
                         "input": {"required": {"model": ["MODEL"]}}, "output": ["MODEL"]},
    "PathchSageAttentionKJ": {"name": "PathchSageAttentionKJ",
                              "input": {"required": {"model": ["MODEL"]}}, "output": ["MODEL"]},
    "SpectrumApplyMiniMaxH3": {"name": "SpectrumApplyMiniMaxH3",
                               "input": {"required": {"model": ["MODEL"]}},
                               "output": ["MODEL"]},
}

#: 默认链那四项（口径：参考实现的内置默认链 ✓ 见 `engine/accel_chain.DEFAULT_CHAIN` ✓）。
CHAIN = [
    {"id": "lora", "class_type": "LoraLoader", "kind": "lora",
     "inputs": {"lora_name": "x.safetensors"}},
    {"id": "tespeed", "class_type": "TESpeedMiniMaxH3", "kind": "model_only"},
    {"id": "sage", "class_type": "PathchSageAttentionKJ", "kind": "model_only"},
    {"id": "spectrum", "class_type": "SpectrumApplyMiniMaxH3", "kind": "model_only"},
]


class FakeClient:
    """假客户端 ✓：只喂 ``object_info_all`` ✓（真 HTTP 由 `h3_comfyui_test.py` 管 ✓）。"""

    def __init__(self, table: Any = None, *, boom: str = "") -> None:
        self._table = NODE_TYPES if table is None else table
        self._boom = boom

    def object_info_all(self) -> Any:
        if self._boom:
            raise ConnectionError(self._boom)
        return self._table


def case_happy() -> None:
    """① 注册表齐全 ⇒ 全 ``ok`` ✓、``blocked=False`` ✓、**接线**给出来了 ✓。"""
    report = accel_mod.check_chain(FakeClient(), CHAIN, model=["4", 0], clip=["5", 0])
    check("① 全 ``ok`` ✓、``blocked=False`` ✓、``unreachable=None`` ✓",
          report["counts"]["ok"] == 4 and not report["blocked"]
          and report["unreachable"] is None, report["counts"])
    build = report["build"] or {}
    check("①′ ⭐ 顺手**接线**了 ✓：四环全进图 ✓、最终 ``model`` 指针指到最后一环 ✓、"
          "LoRA 那环同时接管 ``clip`` ✓",
          len(build.get("nodes", {})) == 4 and build.get("model") == ["accel_spectrum", 0]
          and build.get("clip") == ["accel_lora", 1], build.get("model"))

    warn = accel_mod.check_chain(
        FakeClient(), [{**CHAIN[0], "inputs": {"lora_name": "x", "wrong_extra": 1}}],
        model=["4", 0], clip=["5", 0])
    check("①″ ``warn``（参数名不存在 ✓）**不拦** ✓：``blocked=False`` 且**照样接线** ✓"
          "（与 ``error`` 分得开 ✗）",
          warn["counts"]["warn"] == 1 and not warn["blocked"] and warn["build"] is not None,
          warn["counts"])


def case_error() -> None:
    """② 真错 ⇒ ``blocked=True`` ✓ 且 **build=None** ✗（不静默少一环 ✓）。"""
    table = {key: value for key, value in NODE_TYPES.items() if key != "TESpeedMiniMaxH3"}
    report = accel_mod.check_chain(FakeClient(table), CHAIN, model=["4", 0], clip=["5", 0])
    reason = (report["statuses"][1]["issues"] or [{}])[0].get("message", "")
    check("② 有一环没注册 ⇒ ``error`` ✓、``blocked=True`` ✓、**``build=None``** ✗"
          "（宁可整条不提交，也不静默少一环 ✓）",
          report["counts"]["error"] == 1 and report["blocked"] and report["build"] is None,
          report["counts"])
    check("②′ 理由**可行动** ✓（点名「完全重启 ComfyUI」✓ 并说清刷新网页不算 ✗）",
          "完全重启" in reason and "刷新" in reason, reason)


def case_unreachable() -> None:
    """③ ⭐ ComfyUI 连不上 ⇒ **「没查」**✗ 而不是判错 ✓；且**不接线** ✓（槽位不猜 ✗）。"""
    report = accel_mod.check_chain(FakeClient(boom="connection refused"), CHAIN,
                                  model=["4", 0], clip=["5", 0])
    check("③ ⭐⭐ 连不上 ⇒ 四环全 ``unchecked`` ✓、``counts['error'] == 0`` ✓✗"
          "（**连不上 ≠ 节点没装** ✓）、且 ``unreachable`` 带原因 ✓",
          report["counts"]["unchecked"] == 4 and report["counts"]["error"] == 0
          and "connection refused" in (report["unreachable"] or ""), report["unreachable"])
    check("③′ ``unchecked`` **不拦提交** ✓ 但**也不接线** ✗（``build=None`` ✓ —— "
          "输出槽位要从注册表数出来 ✓ 不猜 ✗）",
          report["blocked"] is False and report["build"] is None)
    empty = accel_mod.check_chain(FakeClient({}), CHAIN, model=["4", 0], clip=["5", 0])
    check("③″ 注册表**空**（``{}`` ✓）也当「没查」✓（空表说明根本没查到 ✓ 别逐项报「没装」✗）",
          empty["counts"]["unchecked"] == 4 and empty["build"] is None
          and "空" in (empty["unreachable"] or ""), empty["unreachable"])


def main() -> int:
    case_happy()
    case_error()
    case_unreachable()
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
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
