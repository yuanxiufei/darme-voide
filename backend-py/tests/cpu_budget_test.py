"""S7 自检：**CPU 线程预算**（`app/core/cpu_budget.py` ✓ 纯 stdlib ✓ 不依赖 GPU ✓）。

为什么单开一套
  ⚠️ 这是 2026-09-26 用户点名的病 ✓：「这几次都是 CPU 占比高、GPU 几乎没有用到，造成 CPU 负载过高」✓✗
  —— 病根之一就是**全仓没有任何线程钳制** ✗（实测 grep：``set_num_threads`` / ``OMP_NUM_THREADS``
  全仓 **0 命中** ✓）⇒ torch/numpy 默认并行度 = 逻辑核数 ⇒ 一次张量运算就能把整机拉满 ✗✗。
  故本套钉的不是"某台机器上跑出几线程" ✗（那会**随机器翻脸** ✓✗），而是**口径本身** ✓：

* ⭐ **显式优先** ✓：入参 > ``VOIDE_CPU_THREADS`` ✓，两者都不给才走自动档 ✓；
* ⭐⭐ **越界/非整数 ⇒ 报错** ✗：不静默夹取 ✗（"我设了 64"悄悄变成"其实 32" 与静默兜底同类 ✗）；
* ⭐ **自动档 = ``clamp(核数//4, 2, 8)``** ✓（留裕量 ✓），且**只有这一处**算 ✓；
* ⭐ **不许动 ``os.environ``** ✗：给子进程的用法是**返回 env 片段** ✓（调用方自己合并 ✓）；
* ⭐ **接线是真的** ✓（静态断言 ✓）：两处装载（`torch_backend` / `sdxl_backend` ✓）与自检驱动
  （`tests/run_all.py` ✓）真的挂在这个口径上 ✓ —— 后人把这几行删掉 ⇒ **本套立刻红** ✗
  （否则默认并行度会悄悄变回逻辑核数 ✗、CPU 又被打满 ✓✗，而没人在意 ✗✗）；
* ⭐ ``apply_torch`` 报「改前 / 改后」两个**数字** ✓（可核对的**事实** ✓，不是"我调过了"的自述 ✗）；
  没装 torch ⇒ 如实报 ``applied=False`` ✓ **不抛** ✗（torch 是懒依赖 ✓）。

⚠️ 全省的桩都是 ``cores=`` / ``env=`` / 假 torch 模块 ⇒ **结论与这台机器装了什么无关** ✓。

运行::

    backend-py> python tests/cpu_budget_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

from app.core import cpu_budget as cb  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(call: Any, *needles: str) -> str | None:
    """调 ``call`` 必须**报错** ✓ 且消息里有全部 ``needles`` ✓（少一个就返回 None ✗）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if all(n in text for n in needles) else None
    return None


class _FakeTorch:
    """假 torch ✓ —— 只记「被怎么调了」✓（不 import 真 torch ⇒ 本套秒跑 ✓）。"""

    def __init__(self, *, threads: int = 32, interop_error: Exception | None = None,
                 has_interop: bool = True) -> None:
        self._threads = threads
        self._interop_error = interop_error
        self.calls: list[tuple[str, int]] = []
        if not has_interop:
            # 钉成**不可调用** ✓ ⇒ 模块里 ``callable(...)`` 为假 ⇒ 走「老 torch 没有这个 API」那条 ✓
            self.set_num_interop_threads = None  # type: ignore[assignment]

    def get_num_threads(self) -> int:  # pragma: no cover - 假件
        return self._threads

    def set_num_threads(self, value: int) -> None:  # pragma: no cover - 假件
        self.calls.append(("set_num_threads", int(value)))
        self._threads = int(value)

    def set_num_interop_threads(self, value: int) -> None:  # pragma: no cover - 假件
        self.calls.append(("set_num_interop_threads", int(value)))
        if self._interop_error is not None:
            raise self._interop_error


def case_auto() -> None:
    """① 自动档 = ``clamp(核数//4, 2, 8)`` ✓（不随这台机器变 ✓）。"""
    check("① 32 逻辑核 ⇒ 8 ✓（= 25%，两侧都留裕量 ✓）",
          cb.resolve(cores=32) == 8, cb.resolve(cores=32))
    check("①′ 16 核 ⇒ 4 ✓、8 核 ⇒ 2 ✓（下界顶住 ✓）",
          cb.resolve(cores=16) == 4 and cb.resolve(cores=8) == 2,
          (cb.resolve(cores=16), cb.resolve(cores=8)))
    check("①″ 4 核 ⇒ **2** ✓ 而不是 1 ✗（4//4 = 1 会被下界抬到 2 ✓：给 1 会让 SIMD 退化成单线程 ✗）",
          cb.resolve(cores=4) == 2, cb.resolve(cores=4))
    check("①‴ 单核机也能用 ✓：1 核 ⇒ 1 ✓（不越过核数 ✓）",
          cb.resolve(cores=1) == 1, cb.resolve(cores=1))
    check("①⁗ 大机器也**封顶 8** ✓（64 核 ⇒ 8 ✓ —— 本仓重活在 GPU / 自检里 ✓，再多只会抢喂数时机 ✗）",
          cb.resolve(cores=64) == 8, cb.resolve(cores=64))
    check("①⁵ 自动档来源**可报** ✓（``source`` 逐字 = 规则本身 ✓）",
          cb.describe(cores=32)["source"] == cb.AUTO_RULE_LABEL, cb.describe(cores=32)["source"])


def case_explicit_wins() -> None:
    """② 显式优先 ✓（入参 > 环境变量 > 自动 ✓）。"""
    facts = cb.describe(16, cores=32, env={cb.ENV_OVERRIDE: "4"})
    check("② 入参 16 压过 env 的 4 ✓、来源 = ``explicit`` ✓",
          facts["threads"] == 16 and facts["source"] == "explicit", facts)
    facts = cb.describe(None, cores=32, env={cb.ENV_OVERRIDE: "12"})
    check("②′ 没给入参 ⇒ 读 env ✓（12 ✓、来源点名那个环境变量 ✓）",
          facts["threads"] == 12 and facts["source"] == f"env:{cb.ENV_OVERRIDE}", facts)
    facts = cb.describe(None, cores=32, env={})
    check("②″ env 是**空串** ⇒ 当没设 ✓ 走自动档 ✓（空串不该被解析成 0 ✗）",
          facts["threads"] == 8 and facts["source"] == cb.AUTO_RULE_LABEL, facts)
    check("②‴ 上限就是**核数本身** ✓：显式 = 核数 ⇒ 放行 ✓（这是「显式自己负责」的边界 ✓）",
          cb.resolve(32, cores=32) == 32, cb.resolve(32, cores=32))


def case_refusals() -> None:
    """③ ⭐⭐ 越界/非整数 ⇒ **报错** ✗（不静默夹取 ✗ —— 这是本套最要紧的一条 ✓）。"""
    check("③ 非整数 ⇒ 报错 ✓ 且说清「要么删掉走自动档 ✓」",
          _raises(lambda: cb.describe(None, cores=32, env={cb.ENV_OVERRIDE: "abc"}),
                  cb.ENV_OVERRIDE, "不是整数", "自动档") is not None)
    check("③′ 0 / 负数 ⇒ 报错 ✓ 且说清「想让它自己决定就**别设** ✗，不是设 0 ✗」",
          _raises(lambda: cb.describe(None, cores=32, env={cb.ENV_OVERRIDE: "0"}),
                  "< 1", "不是设 0") is not None
          and _raises(lambda: cb.describe(-2, cores=32), "< 1") is not None)
    check("③″ 超过逻辑核数 ⇒ 报错 ✓（点名「必然打满整机 ✗」⇒ 与「留裕量」相悖 ✓）",
          _raises(lambda: cb.describe(64, cores=32), "64", "逻辑核数 32", "打满") is not None)
    check("③‴ ⚠️ **不静默夹取** ✓✗：越界请求**抛**而不是悄悄给 8 / 给 32 ✓",
          _raises(lambda: cb.resolve(99, cores=32), "99") is not None)
    check("③⁗ 逻辑核数本身不合理 ⇒ 也报错 ✓（不拿 0 去除 ✗）",
          _raises(lambda: cb.describe(cores=0), "不合理") is not None)


def case_env_plumbing() -> None:
    """④ 环境变量**只通过返回片段**给出去 ✓（不许偷改 ``os.environ`` ✗）。"""
    import os

    before = {key: os.environ.get(key) for key in cb.THREAD_ENV_KEYS}
    segment = cb.env_for_child(cores=32)
    check("④ ``env_for_child`` 五个并行度键**齐** ✓（OMP / MKL / OpenBLAS / NumExpr / VECLIB ✓）",
          set(segment) == set(cb.THREAD_ENV_KEYS) and len(cb.THREAD_ENV_KEYS) == 5, sorted(segment))
    check("④′ 值 = 同一个预算 ✓（8 ✓ —— 各家 BLAS 各读各的名字 ✗，只设一个会出现「钳住一个、另一个满核」✓✗）",
          set(segment.values()) == {"8"}, segment)
    check("④″ ⚠️ **没动** ``os.environ`` ✓✗（给子进程的东西不该顺手改自己 ✓）",
          {key: os.environ.get(key) for key in cb.THREAD_ENV_KEYS} == before)
    sink: dict[str, str] = {}
    facts = cb.apply_env(cores=32, target=sink)
    check("④‴ ``apply_env`` 就地写目标 ✓ 且**报出写了什么** ✓（``envWritten`` ✓）",
          sink == segment and facts["envWritten"] == segment, (sink, facts.get("envWritten")))
    check("④⁗ 事实表**键齐** ✓（前端/日志要照它读 ⇒ 少键就是又一次口径分裂 ✗）",
          set(cb.describe(cores=32)) >= {"logicalCores", "threads", "source", "autoRule",
                                        "explicitRange", "envKeys", "note"}, sorted(cb.describe(cores=32)))


def case_apply_torch() -> None:
    """⑤ ``apply_torch`` ✓：报「改前 / 改后」+ 钉死 inter-op = 1 ✓。"""
    fake = _FakeTorch(threads=32)
    facts = cb.apply_torch(fake, cores=32)
    check("⑤ 按预算钳住 ✓：``previous=32`` ⇒ ``threadsNow=8`` ✓（两个数字都是**事实** ✓ 不是自述 ✗）",
          facts["applied"] and facts["previous"] == 32 and facts["threadsNow"] == 8, facts)
    check("⑤′ inter-op 钉 **1** ✓（本仓推理是单流 ✓ ⇒ >1 只会抢核 ✗）",
          facts.get("interopThreads") == 1
          and ("set_num_interop_threads", 1) in fake.calls, fake.calls)
    check("⑤″ 显式给的值照样穿过 ✓（``apply_torch(fake, 12)`` ⇒ 12 ✓）",
          cb.apply_torch(_FakeTorch(), 12)["threadsNow"] == 12)
    late = _FakeTorch(interop_error=RuntimeError("must be called before any parallel work"))
    facts = cb.apply_torch(late, cores=32)
    check("⑤‴ ⚠️ inter-op 改不动 ⇒ **如实报** ✓✗（``interopThreads=None`` + 原因 ✓；**不吞、不假装成功** ✗）",
          facts["applied"] and facts.get("interopThreads") is None
          and "只许在并行区开始前改" in str(facts.get("interopReason")), facts.get("interopReason"))
    check("⑤⁗ 没有 ``set_num_interop_threads`` 的老 torch ⇒ 也如实报 ✓（不崩 ✗）",
          cb.apply_torch(_FakeTorch(has_interop=False), cores=32).get("interopThreads") is None)


def case_no_torch() -> None:
    """⑥ 没装 torch ⇒ 如实报 ✓ **不抛** ✗（torch 是懒依赖 ✓）。"""
    saved = sys.modules.get("torch", ...)
    sys.modules["torch"] = None  # type: ignore[assignment]  # ⚠️ 标准桩：import 它会抛 ImportError ✓
    try:
        facts = cb.apply_torch(cores=32)
    finally:
        if saved is ...:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved  # type: ignore[assignment]
    check("⑥ 没 torch ⇒ ``applied=False`` ✓ + 原因 ✓（**不抛** ✗ —— 缺懒依赖不该把调用方炸掉 ✓）",
          facts["applied"] is False and "没有 torch" in str(facts.get("reason")), facts.get("reason"))
    check("⑥′ 预算数字**照样给** ✓（钳不了也要报事实 ✓）", facts["threads"] == 8, facts)


def case_invariant_on_this_machine() -> None:
    """⑦ 不打桩也能守住的不变量 ✓（不随这台机器翻脸 ✓）。"""
    facts = cb.describe()
    cores = int(facts["logicalCores"])
    check("⑦ 本机事实表**自洽** ✓：``1 <= threads <= 核数`` ✓、且**没打满** ✓（threads < 核数 ✓）",
          1 <= int(facts["threads"]) <= cores and int(facts["threads"]) < cores, facts)
    check("⑦′ 自动档**永远小于核数** ✓（留裕量是本模块的存在理由 ✓ —— 若哪天等于核数，就是有人把口径改坏了 ✗）",
          cb.resolve(cores=cores) < cores, (cores, cb.resolve(cores=cores)))


def case_wiring_is_real() -> None:
    """⑧ ⭐ 静态守卫 ✓：**两处装载**与**自检驱动**真的接在同一个口径上 ✓（删掉即红 ✗）。

    ⚠️ 为什么用静态断言而不是跑一遍装载 ✓：真装载要么需要 39 GiB 真权重 ✓、要么要 GPU ✗
    ⇒ 那会让本套**不再秒跑** ✗、还会随机器翻脸 ✗✗。本仓已有同类先例（`local_models_test.py`
    就是静态扫 `app/` ✓）。它拦的是**最要命的那种退化**：后人重构时把这两行顺手删了 ✗，
    于是默认并行度又变回逻辑核数 ✗、CPU 又被打满 ✓✗，而**没有任何自检会红** ✗✗。
    """
    engine = BACKEND_PY / "app" / "services" / "engine"
    cases = {
        "视频侧装载 torch_backend.load_weights ✓": (engine / "torch_backend.py",
                                                     "cpu_budget.apply_torch("),
        "图片侧装载 sdxl_backend.load_weights ✓": (engine / "sdxl_backend.py",
                                                   "cpu_budget.apply_torch("),
        "自检驱动 run_all（子进程 env ✓）": (BACKEND_PY / "tests" / "run_all.py",
                                             "cpu_budget.env_for_child("),
        "体检报告转述预算 ✓": (BACKEND_PY / "app" / "services" / "engine_readiness.py",
                               '"cpuThreads"'),
    }
    for name, (path, needle) in cases.items():
        text = path.read_text(encoding="utf-8")
        check(f"⑧ {name}", needle in text and "cpu_budget" in text, (str(path), needle))
    check("⑧′ 引擎与自检**共用同一个模块** ✓（不许在两处各写一套预算 ✗✗ —— 那正是"
          "「同一件事两个说法」的病根 ✓）",
          (engine / "torch_backend.py").read_text(encoding="utf-8").count("cpu_budget.apply_torch(") == 1
          and (engine / "sdxl_backend.py").read_text(encoding="utf-8").count("cpu_budget.apply_torch(") == 1)


def main() -> int:
    case_auto()
    case_explicit_wins()
    case_refusals()
    case_env_plumbing()
    case_apply_torch()
    case_no_torch()
    case_invariant_on_this_machine()
    case_wiring_is_real()
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
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条
    #    `print` 就 `UnicodeEncodeError` 崩 ✗（守卫口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
