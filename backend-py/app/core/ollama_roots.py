"""Ollama 模型库**根候选** —— 只读环境变量与盘，不碰业务数据 ✓。

## 为什么它住 ``app/core``（2026-11-08 搬 ✓）

``services/ollama_store.py`` 是「把 Ollama 的 manifests / blobs 读成模型清单」那一层（业务面 ✓：
``read_manifest`` / ``verify_blob`` / ``list_models`` ✓）。但**根候选**这段（``OLLAMA_MODELS`` ✓、
桌面端 ``db.sqlite`` 里声明的库根 ✓、新版与经典落点 ✓）与业务无关 —— 它读的只有环境变量与一个
**只读**打开的 sqlite ✓，答的是「盘上的 Ollama 库可能在哪」✗。

引擎侧要它：``model_ecosystems`` 的静态表里，Ollama 那一行的 ``candidates`` 就取自这里 ✓；而
``model_ecosystems`` 又要进 ``app/core``（``inventory.candidate_roots`` 要生态落点 ✓，引擎不许
import 服务层 ✓）。留一份在服务层的后果是 core 得反过来 import services ✗（``layering_test`` 判红）。

⇒ 实现搬到这里，``services/ollama_store.py`` **再导出同一批对象** ✓
（``models_root_candidates is app.core.ollama_roots.models_root_candidates`` ✓），不是副本 ✗。

⚠️ 只搬「候选」那半 ✗：``models_root()``（挑一个真能读的 ✓）与 ``is_available()``（看 ``manifests/``）
留服务层 ✓ —— 它们回答的是「用它哪个、能不能读」，属 store 自己的口径 ✓，引擎不需要 ✗。

## 候选顺序（**保序** ✓，语义与真机一致）

``OLLAMA_MODELS`` ✓ → 桌面端声明的库根 ✓ → ``%LOCALAPPDATA%\\Ollama\\models`` ✓ → ``~/.ollama/models`` ✓

⭐ 为什么非得读桌面端那个 db：桌面端把库装到自定义目录时，``OLLAMA_MODELS`` 只注入给
``ollama serve`` 子进程 ✗ ⇒ 后端自己看不见 ✗（真机复现过 ✓）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "DESKTOP_DB_RELATIVE",
    "DESKTOP_MODELS_COLUMN",
    "declared_models_root",
    "desktop_settings_databases",
    "models_root_candidates",
]

#: Ollama **桌面端**配置库的位置（相对 Windows 的 ``LOCALAPPDATA`` / ``APPDATA`` ✓）
DESKTOP_DB_RELATIVE = os.path.join("Ollama", "db.sqlite")

#: 桌面端 ``settings`` 表里存「模型库根」的列名 ✓（真机实测 ✓ 不是猜的 ✓）
DESKTOP_MODELS_COLUMN = "models"

#: ⚠️ 请求链路上会被反复调用 ⇒ 不能每次都开一次 sqlite ✗
#: 缓存形如 ``db 路径 -> (mtime, 大小, 声明值)`` ✓：文件一变就重读 ✓
_DECLARED_CACHE: dict[str, tuple[float, int, str | None]] = {}


def _env(key: str) -> str:
    """环境变量（trim + 剥引号 ✓）。口径同 ``services.ollama_native.envconfig.Var`` ✓。

    ⚠️ 剥引号是上游的真实行为（``strings.Trim(s, "\\"'")``）：Windows 用户写
    ``set OLLAMA_MODELS="<盘符>:\\models"`` 时会带进引号 ✗，不剥就会拼出
    ``"<盘符>:\\models"/blobs`` 这种不存在的路径 ✓✗。
    """
    return os.environ.get(key, "").strip().strip("\"'")


def desktop_settings_databases() -> list[str]:
    """Ollama 桌面端配置库的候选位置 ✓（Windows 真机实测在 ``%LOCALAPPDATA%\\Ollama\\db.sqlite`` ✓）。"""
    found: list[str] = []
    for variable in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(variable)
        if not base or not base.strip():
            continue
        path = os.path.join(os.path.abspath(base.strip()), DESKTOP_DB_RELATIVE)
        if path not in found:
            found.append(path)
    return found


def _valid_declared_root(value: Any) -> str | None:
    """桌面端声明值 → 可用的**绝对路径** ✓（不合格 ⇒ ``None`` ✓：绝不能把垃圾塞进候选 ✗）。"""
    if not isinstance(value, str):
        return None
    text = value.strip().strip('"').strip("'")
    if not text or len(text) > 4096:
        return None
    # JSON / 多行 / 数组 ⇒ 那不是「一个路径」✗（版本升级换过存储格式时别硬读 ✓）
    if any(mark in text for mark in ("\n", "\r", "\t", "{", "}", "[", "]")):
        return None
    if not os.path.isabs(text):
        return None
    return os.path.abspath(text)


def _read_declared_root(database: str) -> str | None:
    """真的去读一次桌面端配置库 ✓（**只读** ✓ 短超时 ✓ 桌面端正用着也不能把后端卡住 ✗）。"""
    try:
        import sqlite3
    except ImportError:   # pragma: no cover - 标准库缺失才可能
        return None

    connection = None
    try:
        # ⚠️ 必须 ``mode=ro``：桌面端此刻也在写这个库 ⇒ 我们绝不能建日志/改文件 ✗
        connection = sqlite3.connect(f"{Path(database).as_uri()}?mode=ro", uri=True, timeout=0.5)
        columns = {row[1] for row in connection.execute("pragma table_info(settings)")}
        if DESKTOP_MODELS_COLUMN not in columns:
            return None
        row = connection.execute(
            f"select {DESKTOP_MODELS_COLUMN} from settings limit 1").fetchone()
    except sqlite3.Error:
        return None      # 不是 sqlite / 被独占 / 版本不认识 ⇒ 当它没说 ✓ 不抛错 ✓
    finally:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:   # pragma: no cover - 关不上也不该影响调用方 ✓
                pass
    return _valid_declared_root(row[0]) if row else None


def declared_models_root() -> str | None:
    """桌面端**声明**的模型库根 ✓（读不到 ⇒ ``None`` ✓ 不抛错 ✓）。

    ⭐ 为什么非得读它：桌面端把库装到自定义目录时，那个 ``OLLAMA_MODELS``
    只注入给 ``ollama serve`` 子进程 ✗ ⇒ 后端自己看不见 ✗（真机复现过 ✓）。
    """
    for database in desktop_settings_databases():
        try:
            stat = os.stat(database)
        except OSError:
            continue      # 没装桌面端 / 没这个库 ⇒ 静默跳过 ✓
        cached = _DECLARED_CACHE.get(database)
        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            if cached[2]:
                return cached[2]
            continue
        value = _read_declared_root(database)
        _DECLARED_CACHE[database] = (stat.st_mtime, stat.st_size, value)
        if value:
            return value
    return None


def models_root_candidates() -> list[str]:
    """候选模型库根目录（**保序** ✓：环境变量 > 桌面端声明 > Windows 新版落点 > 经典落点 ✓）。"""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if value and str(value).strip():
            cleaned = os.path.abspath(str(value).strip())
            # ⚠️ 按 ``normcase`` 去重：Windows 盘符**大小写不敏感** ✗（同一目录的两种写法会撞 ⇒ 不去重就重复统计 ✗）
            key = os.path.normcase(cleaned)
            if key not in seen:
                seen.add(key)
                candidates.append(cleaned)

    add(_env("OLLAMA_MODELS"))
    # ⭐ 桌面端自己选的库根（放在默认落点**之前** ✓：那是用户真实生效的设置 ✓）
    add(declared_models_root())
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        # Ollama 0.3.x 起在 Windows 上改落这里 ✓（老装机仍是 ``~/.ollama/models`` ✓）
        add(os.path.join(local_app_data, "Ollama", "models"))
    home = os.path.expanduser("~")
    if home and home != "~":
        add(os.path.join(home, ".ollama", "models"))
    return candidates
