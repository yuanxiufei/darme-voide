"""CPU 线程预算 —— 「两侧都留裕量」的**唯一**一处口径 ✓。

为什么要有这个模块
  本仓口径（用户原话，2026-09-26 ✓）：

    「跑推理/训练等重负载任务时，CPU 与 GPU 要**相互配合、分担计算**，让两者都参与、谁也别闲着；
      前提是**两侧都留在合理范围内** —— 不允许任何一侧被打满过载 ✗。」

  ⚠️ 而 torch / numpy / OpenBLAS 的**默认**并行度就是**逻辑核数** ✗（实测：本机 32 逻辑核
  ⇒ 一次张量运算即可把整机拉到 100% ✓✗）—— 这正是上面那条的反面 ✓。

  ⚠️ 实测反面案例（用户点名不满 ✓）：探针脚本把 SDXL 硬编码成 ``device="cpu"`` + fp32 ✓✗
  ⇒ 32 逻辑核全满、整机负载 55%+、烧掉两千多秒 CPU 时间，而 A5000 只有 **9%** 在摸鱼 ✗✗
  —— 既没分工 ✓✗、也没控负载 ✗。

  ⇒ 于是把「用几个 CPU 线程」收成**一处** ✓：自检驱动（`tests/run_all.py` ✓）与引擎装配
  （`services/engine/` ✓）**共用** ✓ —— ⚠️ 两处各写一套就是又一次「同一件事两个说法」✗✗。

口径（分档只有一档自动 ✓，其余全**显式** ✓）
  - **显式优先** ✓：入参 ``explicit`` > 环境变量 :data:`ENV_OVERRIDE`（``VOIDE_CPU_THREADS`` ✓）。
    显式值只要在 ``[1, 逻辑核数]`` 内就**照用** ✓；越界/不是整数 ⇒ **当场报错** ✗✗ ——
    ⚠️ **不静默夹取** ✗（"我设了 64"变成"其实 32"与静默兜底同一类病 ✓✗）。
  - **否则自动** ✓：``clamp(逻辑核数 // 4, 2, 8)`` ✓ —— 本机 32 逻辑核 ⇒ **8** ✓（占 25% ✓）。
    两条边界的道理：**下限 2** ✓（给 1 会让 SIMD 内核退化成单线程，反而更慢 ✓✗）；
    **上限 8** ✓（本仓重活在 GPU 或自检里 ✓，再多的 CPU 线程只会去抢喂数时机、把机器拉满 ✗）。
  - ⚠️ 自动档**刻意按「留裕量」写死成温和值** ✓：想全速（例如确认机器上没有 GPU 的纯 CPU 机器 ✓）
    就**显式**设 :data:`ENV_OVERRIDE` ✓ ⇒ 那是调用方对自己机器负责的**显式**选择 ✓，
    不是本模块替它猜 ✗。

⚠️ 生效时机（最容易白设 ✓✗）
  - 环境变量只有在该进程 **import torch / numpy 之前**写进去才管用 ✓ ⇒ 给**子进程**用的一定要在
    ``subprocess`` **起进程前**塞进 ``env=`` ✓（见 :func:`env_for_child` ✓）。
  - 进程已经 import 过了 ⇒ 只能走 :func:`apply_torch`（``torch.set_num_threads`` ✓）✓
    —— 它**照样生效** ✓（torch 的线程池是显式 API ✓，不依赖环境变量 ✓）。

用法::

    from app.core import cpu_budget

    cpu_budget.describe()                      # 事实表：几线程 / 为什么 / 范围
    cpu_budget.env_for_child()                 # 给 subprocess 的 env 片段（import 前生效）
    cpu_budget.apply_env()                     # 就地写本进程 os.environ（import 前用）
    cpu_budget.apply_torch()                   # 已经 import 过 torch ⇒ 用显式 API 钳 ✓
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "AUTO_DIVISOR",
    "AUTO_MAX",
    "AUTO_MIN",
    "AUTO_RULE_LABEL",
    "ENV_OVERRIDE",
    "THREAD_ENV_KEYS",
    "CpuBudgetError",
    "apply_env",
    "apply_torch",
    "describe",
    "env_for_child",
    "logical_cores",
    "resolve",
]

#: 显式覆盖用的环境变量 ✓（**是显式** ✓ —— 设了就照用 ✓，越界则报错 ✗）
ENV_OVERRIDE = "VOIDE_CPU_THREADS"

#: 自动档：``clamp(逻辑核数 // AUTO_DIVISOR, AUTO_MIN, AUTO_MAX)`` ✓
AUTO_DIVISOR = 4
AUTO_MIN = 2
AUTO_MAX = 8

#: 要一起写的并行度环境变量 ✓ —— 各家 BLAS / OpenMP 读的名字不同 ✗，
#: 只设一个会出现「OMP 钳住了、OpenBLAS 还在满核」✓✗（实测最快的自查办法就是逐个设同值 ✓）。
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: 记号：自动档来源（报事实用 ✓ —— **公开** ✓：自检要逐字比它 ✓，不许两处各写一份串 ✗）
AUTO_RULE_LABEL = "auto:clamp(cores//%d, %d, %d)" % (AUTO_DIVISOR, AUTO_MIN, AUTO_MAX)
_SRC_EXPLICIT = "explicit"
_SRC_ENV = "env:" + ENV_OVERRIDE


class CpuBudgetError(ValueError):
    """线程预算**不可用** ✓ —— 消息必须说清「该怎么给」✓（**不静默回落** ✗）。"""


def logical_cores() -> int:
    """本机逻辑核数 ✓（``os.cpu_count()`` ✓；它在任何平台都答得出，拿不到才退 ``1`` ✓）。"""
    return int(os.cpu_count() or 1)


def _parse(raw: str) -> int:
    """把**显式**字符串解成线程数 ✓；解不出 ⇒ 报错 ✗（不猜 ✗）。"""
    text = str(raw).strip()
    try:
        value = int(text, 10)
    except ValueError:
        raise CpuBudgetError(
            f"{ENV_OVERRIDE}={text!r} 不是整数 ✗ ⇒ 要么删掉它走自动档 ✓、要么给个正整数 ✓"
            f"（例如 {ENV_OVERRIDE}=8 ✓）")  # noqa: TRY003
    return value


def describe(explicit: int | None = None, *, cores: int | None = None,
             env: Any = None) -> dict[str, Any]:
    """事实表 ✓ —— 「用几个线程、凭什么是这几个」**只有这一处算** ✓。

    ``cores`` / ``env`` 只为自检打桩 ✓（生产调用不传 ✓）。
    """
    total = int(cores) if cores is not None else logical_cores()
    if total < 1:
        raise CpuBudgetError(f"逻辑核数 {total} 不合理 ✗（至少是 1 ✓）")

    source = AUTO_RULE_LABEL
    chosen: int | None = None
    if explicit is not None:
        chosen = int(explicit)
        source = _SRC_EXPLICIT
    else:
        lookup = os.environ if env is None else env
        raw = lookup.get(ENV_OVERRIDE)
        if raw is not None and str(raw).strip():
            chosen = _parse(raw)
            source = _SRC_ENV

    if chosen is None:
        threads = min(max(total // AUTO_DIVISOR, AUTO_MIN), AUTO_MAX)
        threads = min(threads, total)  # 单核机器也得能用 ✓
    else:
        if chosen < 1:
            raise CpuBudgetError(
                f"线程数 {chosen} < 1 ✗（至少要 1 ✓；想让 torch 自己决定请**别设** ✓，"
                f"不是设 0 ✗）")
        if chosen > total:
            raise CpuBudgetError(
                f"线程数 {chosen} > 逻辑核数 {total} ✗ ⇒ 给这么多只会让调度器互相抢 ✓✗、"
                f"并且**必然打满整机** ✗（与「两侧都留裕量」相悖 ✓）；上限就是 {total} ✓")
        threads = chosen

    return {
        "logicalCores": total,
        "threads": threads,
        "source": source,
        "autoRule": AUTO_RULE_LABEL,
        "explicitRange": [1, total],
        "envKeys": list(THREAD_ENV_KEYS),
        "note": "CPU 只做数据加载/预处理/解码/IO/编排 ✓，重张量运算留给 GPU ✓",
    }


def resolve(explicit: int | None = None, *, cores: int | None = None,
            env: Any = None) -> int:
    """只要那个数字 ✓（口径仍在 :func:`describe` ✓）。"""
    return int(describe(explicit, cores=cores, env=env)["threads"])


def env_for_child(explicit: int | None = None, *, cores: int | None = None,
                  env: Any = None) -> dict[str, str]:
    """给 ``subprocess`` 的 **env 片段** ✓（**不改** ``os.environ`` ✗ ⇒ 调用方自己合并 ✓）。

    ⚠️ 必须用在 ``subprocess.run(..., env=...)`` 里 ✓ —— 那样子进程在 import torch/numpy
    **之前**就看见它 ✓，本模块的自动档与显式档才真的生效 ✓。
    """
    threads = resolve(explicit, cores=cores, env=env)
    return {key: str(threads) for key in THREAD_ENV_KEYS}


def apply_env(explicit: int | None = None, *, cores: int | None = None,
              env: Any = None, target: Any = None) -> dict[str, Any]:
    """就地写并行度环境变量 ✓（``target`` 省略 ⇒ ``os.environ`` ✓）。

    ⚠️ 只对「**还没 import** torch/numpy 的进程」有意义 ✓✗（已经 import 过的请用
    :func:`apply_torch` ✓）。
    """
    facts = describe(explicit, cores=cores, env=env)
    sink = os.environ if target is None else target
    written = {key: str(facts["threads"]) for key in THREAD_ENV_KEYS}
    sink.update(written)
    return {**facts, "envWritten": written}


def apply_torch(torch_module: Any = None, explicit: int | None = None, *,
                cores: int | None = None, env: Any = None) -> dict[str, Any]:
    """已经 import 过 torch ⇒ 用**显式 API** 钳住它 ✓（``set_num_threads`` ✓）。

    * ``torch_module`` 省略 ⇒ 自己 ``import torch`` ✓；**没装** torch ⇒ 如实报
      ``applied=False`` ✓（**不抛** ✗ —— torch 是懒依赖 ✓，缺它时这里不该把调用方炸掉 ✓）。
    * ``set_interop_threads`` 钉 **1** ✓：本仓推理是**单流** ✓，inter-op > 1 只会抢核 ✗；
      ⚠️ torch 只允许在**并行区开始前**调它 ✓，晚了会 ``RuntimeError`` ⇒ **如实报** ✓（不吞 ✗）。
    * 返回「改前 / 改后」两个数字 ✓（``previous`` ✓）—— 这是可核对的**事实** ✓，
      不是"我调过了"的自述 ✓。
    """
    facts = describe(explicit, cores=cores, env=env)
    threads = int(facts["threads"])

    module = torch_module
    if module is None:
        try:
            import torch as module  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError as error:
            return {**facts, "applied": False, "reason": f"没有 torch ⇒ 无需钳 ✓（{error} ✓）"}

    previous: int | None = None
    getter = getattr(module, "get_num_threads", None)
    if callable(getter):
        try:
            previous = int(getter())
        except Exception:  # noqa: BLE001 - 自述不许把调用方打挂 ✗
            previous = None

    module.set_num_threads(threads)

    interop: dict[str, Any] = {}
    setter = getattr(module, "set_num_interop_threads", None)
    if callable(setter):
        try:
            setter(1)
            interop = {"interopThreads": 1}
        except Exception as error:  # noqa: BLE001 - torch 只许在并行区前改 ⇒ 如实报 ✓
            interop = {"interopThreads": None,
                       "interopReason": f"torch 不肯现在改 inter-op ✗（它只许在并行区开始前改 ✓）：{error} ✗"}
    else:  # pragma: no cover - 老版本 torch
        interop = {"interopThreads": None, "interopReason": "这个 torch 没有 set_num_interop_threads ✗"}

    return {**facts, "applied": True, "previous": previous, "threadsNow": int(module.get_num_threads())
            if callable(getattr(module, "get_num_threads", None)) else threads, **interop}
