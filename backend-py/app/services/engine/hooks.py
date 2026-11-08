"""引擎 → 服务层的**单向接线口**（引擎侧只认这张表，不 import 服务层 ✓）。

为什么要有它：`engine/*.py` 的仓内依赖**只许** `app.core.*` 与引擎自己（硬契约，判定在
`tests/engine_independence_test.py` ②）。而引擎有四处确实要读业务数据 —— 跑次索引
（`outcome_index`）、知识台账（`knowledge_ledger`）、质检复核（`qc_reviews`）—— 那三个模块
都在服务层。原先的做法是**在函数里 late import**，形式上躲开了"顶层依赖"，但**方向仍然是
引擎 → 服务层** ⇒ 引擎被单独拉起（CLI / 纯引擎进程）时就隐式拖上 sqlite 与业务表，而且
`tuning.suggest()` / `check_evidence()` 在那种环境下会以"ImportError 或空数据"的样子失败 ——
读起来像 bug，其实是没有数据源（★ 不报错地给 0 条建议 = 假绿 ✓✗）。

现在反过来：**服务层在启动时把源登记进来**（`app/main.py` 一处），引擎只查这张表：

* 登记了 ⇒ 走真数据（Web 通路的行为一字不变 ✓）；
* 没登记 ⇒ 各调用点**如实说"没接入"**（`suggest()` 不给候选并写明原因；`check_evidence()`
  返回 `ok=False` 且带 `unavailable=True`，**不是**静默放行 ✓✗）。

⚠️ 这里登记的是**模块对象**（不是数据快照）✓：那三个模块自己读环境变量定位 DB、自带缓存，
登记模块即"以后一切查询都走它"；换库仍然是改环境变量的事 ✓。

⚠️ **名字写错要在当场炸** ✓✗（`provide` / `get` / `provided` 三处都拦）：只有登记口拦的话，
读的一侧把名字拼错会静默回 ``None`` ⇒ 引擎如实说"没接入" ✓ ——「我拼错了」被伪装成
「服务层没登记」，排查方向直接错 ✗。同理 ``provide(run_index=None)`` 也拒 ✓：那等价于没登记，
却会让 ``provided()`` 说"登记了" ⇒ 两个口自相矛盾 ✓✗。

⚠️ **写测试的人也得登记**（2026-11-07 实测踩到 ✓✗）：注入式接线的代价就是"谁用谁登记" ——
单跑 `qc_reviews_test` / `qc_thresholds_test` / `knowledge_loop_test` / `outcome_wiring_test` 时
**没有** `app/main.py` 那一步 ⇒ 四套会集体红，而且症状**像业务 bug**（`check_evidence()` 回
`unavailable=True` 并"按核不过处理"、`tuning.set_field()` 全部拒写、`suggest()` 一条候选都不给）。
固定写法（那四套现在都这么写 ✓）::

    from app.services.engine import hooks
    hooks.provide(ledger=knowledge_ledger, run_index=outcome_index, qc_reviews=qc_reviews)

登记是**进程级全局**的（测试逐个进程跑，互不干扰 ✓）；`clear()` 只给 `engine_hooks_test` 用 ✓。
"""

from __future__ import annotations

from typing import Any

__all__ = ["SOURCES", "clear", "get", "provide", "provided", "status"]

#: 引擎认的三个源的**固定名字**（写错名字 = 引擎读不到，登记处会当场报错 ✓✗）
SOURCES: tuple[str, ...] = ("run_index", "ledger", "qc_reviews")

#: 已登记的实现（名字 → 模块/替身）。进程级、可重复登记（后登记的覆盖前面的）。
_SOURCES: dict[str, Any] = {}


def _known(name: str) -> str:
    """名字必须在 :data:`SOURCES` 里 ✓。

    ⚠️ 三个入口（``provide`` / ``get`` / ``provided``）**都**拦 ✓✗：只有登记口拦的话，
    读的一侧把名字拼错（``hooks.get("runindex")``）会静默回 ``None`` ⇒ 引擎如实说
    「跑次索引没接入」✓ —— 于是**拼错名字**被伪装成**服务层没登记**，查起来朝错的方向走 ✓✗。
    """
    if name not in SOURCES:
        raise ValueError(f"未知的引擎源 {[name]}；只认 {list(SOURCES)}")
    return name


def provide(**sources: Any) -> dict[str, str]:
    """登记源。名字必须在 :data:`SOURCES` 里，否则当场 ``ValueError``（不静默丢 ✓）。"""
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        raise ValueError(f"未知的引擎源 {unknown}；只认 {list(SOURCES)}")
    #: ⚠️ ``provide(run_index=None)`` 要拒 ✓✗：它等价于"没登记"，却会让 ``provided()`` 说
    #: 「登记了」⇒ 两个口自相矛盾，读起来像"登记过但查不到"，是最难查的那种形态 ✓。
    #: 要清空就用 ``clear()``（那是测试口）✓。
    empty = sorted(name for name, impl in sources.items() if impl is None)
    if empty:
        raise ValueError(f"源 {empty} 传的是 None —— 等价于没登记，请在登记处给真模块"
                         f"（要清空用 clear()）")
    for name, impl in sources.items():
        _SOURCES[name] = impl
    return {name: type(impl).__name__ for name, impl in sources.items()}


def get(name: str) -> Any:
    """取源；**没登记就返回 ``None``**（调用点必须自己对 ``None`` 表态 ✓✗）。

    名字不在 :data:`SOURCES` 里 ⇒ 当场 ``ValueError``（那不是"没登记"，是拼错了 ✓✗）。
    """
    return _SOURCES.get(_known(name))


def provided(name: str) -> bool:
    """这个源登记了没有（名字拼错同样是 ``ValueError``，不是 ``False`` ✓✗）。"""
    return _known(name) in _SOURCES


def _label(impl: Any) -> str:
    """登记物的可读名字 ✓✗。

    ⚠️ 不能只写 ``type(impl).__name__`` ✓✗：登记的都是**模块**，那样三个源会一律显示
    「module」（2026-11-08 实测 `doctor` 打出 `module@run_index` —— 等于没写 ✓）。
    模块取 ``__name__`` 的末段、其他对象退回类名 ✓。
    """
    named = getattr(impl, "__name__", None)
    return str(named).rsplit(".", 1)[-1] if named else type(impl).__name__


def status() -> dict[str, Any]:
    """**当前进程**的接线状态（给 `doctor` / 诊断看 ✓）。

    为什么要有它：这套接线的失效方式是**静默降级** ✓✗ —— 少登记一个源，用到它的口只是少说话
    （`suggest()` 不给候选、`tuning.set_field()` 全拒、`check_evidence()` 回 `unavailable`），
    主链路照跑、退出码照 0 ✓。于是"引擎能力不全"与"引擎本来就这样"在外部**分不开** ✓✗。
    一个能一眼看清"哪几个源没登记"的出口，比十条降级文案都有用 ✓。

    ⚠️ 它**只许**放在运行时面（`doctor` 的 CLI 输出、诊断口）✓✗，**不许**塞进
    :func:`facade.capabilities` —— 那是**静态**能力表（`tests/engine_env_docs_test` 拿它跟
    `docs/env/*.json` 逐字段对，同一台机器两次跑必须逐字相同 ✓）。接线状态是**进程态**
    （有没有 `app/main.py` 那一步、测试里 clear 过没有），塞进去会让静态表随进程漂移 ✓✗。
    """
    missing = [name for name in SOURCES if name not in _SOURCES]
    return {
        "sources": list(SOURCES),
        "provided": {name: name in _SOURCES for name in SOURCES},
        "registered": {name: _label(_SOURCES[name])
                       for name in SOURCES if name in _SOURCES},
        "missing": missing,
        "allProvided": not missing,
    }


def clear() -> None:
    """清空（测试用）。"""
    _SOURCES.clear()
