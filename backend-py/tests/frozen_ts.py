"""TS 源码快照的**读取 / 物化**入口（数据在生成物 ``frozen_ts_source.py`` 里）。

为什么不再直接存 ``.ts`` 文件树（2026-09-15 起）
  ① 快照的唯一用途是「给守卫读」，而守卫是用 ``Path`` 读的 ⇒ **物化成临时目录**即可
     让 15 处调用点**零改动**沿用；
  ② 存成 Python 模块后，仓库里**不再有 TS 文件**（删 ``backend/`` 后整棵树都是 Python）；
  ③ ``git diff`` 能直接看到改动了哪一条文本，不必在 74 个小文件之间翻。

用法（守卫侧）::

    from frozen_ts import snapshot_root
    _SRC_ROOT = snapshot_root()          # 真源码缺失 / PARITY_USE_FROZEN=1 时用

人工查看原文::

    python backend-py/tests/freeze_ts_snapshot.py --dump tmp/frozen_ts

⚠️ **同一进程内只物化一次**（``_CACHE``）：守卫会多次取 ``snapshot_root()``，
   反复写盘既慢又会留下多份临时目录。
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "frozen_ts_source.py"

_CACHE: dict[str, str] | None = None
_ROOT: Path | None = None


def load() -> dict[str, str]:
    """读取快照数据（``相对路径 -> 逐字文本``）。"""
    global _CACHE
    if _CACHE is None:
        if not DATA_FILE.is_file():
            raise FileNotFoundError(
                f"缺少 TS 快照数据 {DATA_FILE.name} —— 请先在 backend/ 还在时跑 "
                f"`python tests/freeze_ts_snapshot.py`"
            )
        spec = importlib.util.spec_from_file_location("frozen_ts_source", DATA_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        _CACHE = dict(module.FILES)
    return _CACHE


def materialize(target: Path) -> Path:
    """把快照**逐字**写成 ``.ts`` 文件树到 ``target``（供守卫读取 / 人工查看）。"""
    for relative, text in load().items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    return target


def snapshot_root() -> Path:
    """物化到临时目录并返回其路径（同进程复用）。"""
    global _ROOT
    if _ROOT is None:
        _ROOT = materialize(Path(tempfile.mkdtemp(prefix="frozen_ts_")))
    return _ROOT
