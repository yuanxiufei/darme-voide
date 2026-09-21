r"""**对 `transformers` 做运行时改造（tuning）** —— 不 fork 源码 ✓ 打**受控补丁** ✓ 一处集中 ✓ 可撤 ✓ 可验 ✓。

## 为什么是「运行时补丁」而不是 fork / vendored

用户 2026-09-20：「**你也可以改造 transformers 并实现优化升级这个 transformers**」✓。
落地前的取舍（都记着 ✗，免得下次再纠结一遍 ✓）：

* **vendored（把源码搬进本仓）** ✗：`transformers` 是**数千文件 / 数十 MB** ✓ ⇒ 仓库体积与升级维护
  成本都不划算 ✓（而且它的许可是 Apache-2.0 ✓ 允许 ✓，问题不是许可 ✓ 是**不必要** ✓）；
* **fork** ✗：同上 ✓ 且会长出"我们的版本落后 upstream 多少"的新账 ✓；
* ⭐ **运行时补丁** ✓（本模块）：**只改行为开关** ✓ —— 离线 ✓ 缓存目录 ✓ 降噪 ✓ 计数 ✓ ——
  改动**集中一处** ✓、**幂等** ✓、**可撤销** ✓、**可自证** ✓。
  ⚠️ 边界要写清 ✗：本模块**不碰**它的数学/算法/权重加载语义 ✓（那不属于 tuning ✓）。

## 改造清单（每条：为什么 / 收益 ✓ 与**能触发**的验证点 ✓）

1. **离线兜底** ✓：把 `PreTrainedTokenizerBase.from_pretrained` / `PreTrainedModel.from_pretrained`
   包装成"离线策略下**强制** `local_files_only=True`" ✓ —— 本仓实测**直连 HF 全超时** ✗，
   而漏网的一次外呼 = 生产机上卡住 ✓✗（只靠 `local_files_only` 参数**总会漏** ✗：库内部
   还有别的入口 ✓）⇒ 兜在**类方法**这一层 ✓（全部 tokenizer/model 都走它 ✓）；
2. **缓存目录统一** ✓：把 HF 缓存钉到**本仓数据根** `<dataRoot>/hf-cache` ✓
   （不写用户主目录 ✓ ⇒ 整体清理 / 迁移数据根时不会漏东西 ✓）；
3. **降噪** ✓：`transformers.logging` 降到 ERROR ✓ + `TOKENIZERS_PARALLELISM=false` ✓
   （fork 警告）✓ + `HF_HUB_DISABLE_TELEMETRY=1` ✓（遥测）；
4. **可观测** ✓：包装里计数 ✓ —— 「打了补丁」必须能**自证** ✗（否则只是一句声明 ✓✗）；
5. **可撤销** ✓：:func:`unpatch` 还原原函数 ✓（改错了能退 ✓ 也便于 A/B 对比 ✓）。

⚠️ **幂等是硬要求** ✗：重复 :func:`apply_tuning` **只包一层** ✓（否则计数与调用栈会叠加 ✓✗，
表现为"越跑越慢"这种最难查的账 ✓）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["TUNING_ITEMS", "apply_tuning", "default_cache_dir", "tuning_status", "unpatch"]

#: 改造清单 ✓（**响亮**：报告里逐条对着它 ✓，少一条就看得见 ✓）
TUNING_ITEMS: tuple[str, ...] = (
    "离线兜底：`PreTrainedTokenizerBase` / `PreTrainedModel` / **Auto 工厂**的 `from_pretrained` "
    "一律强制 `local_files_only=True`（⚠️ **Auto 工厂必须单独包** ✗ —— 实测它在**解析出具体类之前**"
    "就外呼 ✓✗，只包基类会漏 ✓）",
    "缓存目录统一：HF 缓存 = `<数据根>/hf-cache` ✓（不写用户主目录 ✓）",
    "降噪：`transformers.logging` → ERROR ✓ + `TOKENIZERS_PARALLELISM=false` ✓ + 关遥测 ✓",
    "可观测：包装内计数（调用 / 强制离线 / 注入缓存目录 ✓）——「打了补丁」要能自证 ✓",
    "可撤销：`unpatch()` 还原原函数 ✓（改错能退 ✓ 也能 A/B 对比 ✓）",
)

#: 包装过的类方法 ✓（原函数存这儿 ⇒ `unpatch` 还原 ✓）
_ORIGINALS: dict[str, Any] = {}
_STATS: dict[str, Any] = {}
_APPLIED = False


def default_cache_dir() -> str | None:
    """HF 缓存目录缺省值 ✓：**本仓数据根**下 ``hf-cache`` ✓（取不到 ⇒ ``None`` 交给 HF 默认 ✓ 不抛 ✗）。

    ⚠️ 权威 API 是 ``core.config.get_data_root()`` ✓（**不是**模块属性 ``DATA_ROOT`` ✗ ——
    按属性取会**静默拿到 None** ✓✗，缓存目录悄悄回到用户主目录 ✓）。
    """
    try:
        from ...core import config as core_config  # noqa: PLC0415 —— 同包内引，避免顶层循环 ✓

        getter = getattr(core_config, "get_data_root", None)
        root = getter() if callable(getter) else getattr(core_config, "DATA_ROOT", None)
        if root:
            return str(Path(root) / "hf-cache")
    except Exception:  # noqa: BLE001 —— 取不到就不指定 ✓（不影响功能 ✓）
        return None
    return None


def _import_transformers() -> Any | None:
    """懒导入 `transformers` ✓（没装 ⇒ ``None`` ✓ **不抛** ✗ —— 它是**可选**依赖 ✓）。"""
    try:
        import transformers  # noqa: PLC0415

        return transformers
    except ImportError:
        return None


def _empty_stats() -> dict[str, Any]:
    return {"fromPretrainedCalls": 0, "offlineForced": 0, "cacheDirInjected": 0,
            "lastKwargs": None}


def apply_tuning(*, offline: bool = True, cache_dir: str | None = None, quiet: bool = True,
                 parallel: bool = False) -> dict[str, Any]:
    """给 `transformers` 打上改造补丁 ✓（**幂等** ✓；没装 ⇒ ``applied=False`` ✓ 不抛 ✗）。

    ``offline``：强制离线 ✓（默认开 ✓）；``cache_dir``：缓存目录 ✓（缺省 = :func:`default_cache_dir` ✓）；
    ``quiet``：日志降到 ERROR ✓；``parallel``：`TOKENIZERS_PARALLELISM` ✓（默认关 ✓ 免得 fork 警告刷屏 ✓）。
    """
    resolved_cache = cache_dir or default_cache_dir()
    # ⚠️ 环境变量**先落** ✓（即使库没装也把策略声明清楚 ✓ —— 别让"将来某处引入 HF"再静默外呼 ✗）
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if parallel else "false")
    if resolved_cache:
        os.environ.setdefault("HF_HOME", resolved_cache)

    library = _import_transformers()
    if library is None:
        return {"applied": False, "offline": offline, "cacheDir": resolved_cache,
                "items": list(TUNING_ITEMS), "stats": dict(_STATS) or _empty_stats(),
                "reason": "本机没装 `transformers` ✓（它是**可选**依赖 ✓）⇒ 只声明策略 ✓ 不打补丁 ✓"}

    global _APPLIED
    if quiet:
        try:
            library.logging.set_verbosity_error()
        except Exception:  # noqa: BLE001 —— 降噪失败不算事 ✓ 不为此报错 ✗
            pass

    if not _APPLIED:
        _STATS.clear()
        _STATS.update(_empty_stats())
        for kind, target in _targets().items():
            if target is not None:
                _wrap_from_pretrained(kind, target, resolved_cache, offline)
        _APPLIED = True
    return {"applied": True, "offline": offline, "cacheDir": resolved_cache,
            "items": list(TUNING_ITEMS), "stats": dict(_STATS), "reason": None}


def _targets() -> dict[str, Any]:
    """要包住的对象 ✓（⚠️ **Auto 工厂必须单列** ✗ —— 见 :data:`TUNING_ITEMS` 第 1 条 ✓）。"""
    targets: dict[str, Any] = {}
    try:
        from transformers.tokenization_utils_base import (  # noqa: PLC0415
            PreTrainedTokenizerBase,
        )

        targets["tokenizer"] = PreTrainedTokenizerBase
    except ImportError:  # pragma: no cover - 版本差异时逐个降级 ✓ 不整体失败 ✗
        pass
    try:
        import transformers  # noqa: PLC0415

        targets["model"] = transformers.PreTrainedModel
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer  # noqa: PLC0415

        targets["auto-tokenizer"] = AutoTokenizer
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    try:
        from transformers.models.auto.modeling_auto import AutoModel  # noqa: PLC0415

        targets["auto-model"] = AutoModel
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    return targets


def _wrap_from_pretrained(kind: str, target: Any, cache_dir: str | None, offline: bool) -> None:
    """包住 ``target.from_pretrained`` ✓（tokenizer / model / Auto 工厂**同一套逻辑** ✓）。"""
    bound = target.__dict__["from_pretrained"]
    original = getattr(bound, "__func__", bound)      # classmethod ⇒ 取底层函数 ✓ 直接带 cls 调 ✓
    _ORIGINALS[kind] = (target, bound)

    def wrapper(cls: Any, *args: Any, **kwargs: Any) -> Any:
        # ① 缓存目录 ✓（调用方显式给了就尊重它 ✓）
        if cache_dir and not kwargs.get("cache_dir"):
            kwargs["cache_dir"] = cache_dir
            _STATS["cacheDirInjected"] += 1
        # ② 离线兜底 ✓：**一律**填 True ✓ —— 包括调用方显式给 `False` 的情形 ✓。
        #    为什么压过显式的 False ✗：这是「策略」不是「偏好」 ✓ —— 离线策略下任何一次外呼
        #    都是**生产机卡住**的原因 ✓✗；而且发生了几次**计数里看得见** ✓（不静默 ✓）。
        if offline:
            if kwargs.get("local_files_only") is not True:
                _STATS["offlineForced"] += 1
            kwargs["local_files_only"] = True
        _STATS["fromPretrainedCalls"] += 1
        _STATS["lastKind"] = kind
        _STATS["lastKwargs"] = {"local_files_only": kwargs.get("local_files_only"),
                               "cache_dir": kwargs.get("cache_dir")}
        return original(cls, *args, **kwargs)

    target.from_pretrained = classmethod(wrapper)


def unpatch() -> dict[str, Any]:
    """还原补丁 ✓（幂等 ✓ 没打过也安全 ✓）。⚠️ 环境变量**不还原** ✗（它们本就该是进程级策略 ✓）。"""
    global _APPLIED
    restored: list[str] = []
    for kind, entry in list(_ORIGINALS.items()):
        if not isinstance(entry, tuple):       # pragma: no cover - 旧结构兜底 ✓
            continue
        target, bound = entry
        target.from_pretrained = bound
        restored.append(kind)
    _ORIGINALS.clear()
    _APPLIED = False
    return {"restored": restored, "stats": dict(_STATS) or _empty_stats()}


def tuning_status() -> dict[str, Any]:
    """现状 ✓：打了没 / 打了哪几条 / 计数 / 环境变量 ✓（**都从实底读** ✓ 不猜 ✗）。"""
    return {
        "transformersInstalled": _import_transformers() is not None,
        "applied": _APPLIED,
        "items": list(TUNING_ITEMS),
        "stats": dict(_STATS) or _empty_stats(),
        "env": {key: os.environ.get(key) for key in
                ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                 "TOKENIZERS_PARALLELISM", "HF_HOME")},
        "defaultCacheDir": default_cache_dir(),
    }
