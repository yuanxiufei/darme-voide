"""全局配置加载 —— 与 ``backend/src/config.ts`` 逐项对齐。

优先级：环境变量 > ``configs/config.yaml`` > 硬编码默认值

数据根目录（SQLite 数据库 + 全部生成文件）：
``.data-root`` 标记文件 > ``DATA_ROOT`` 环境变量 > ``config.yaml database.path`` 所在目录 > ``./data``
数据库固定为 ``<dataRoot>/drama.db``，生成文件固定为 ``<dataRoot>/static``。

与 Node 版的唯一差异：默认端口 ``5790``（Node 后端仍占用 5789），可用 ``PORT`` 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - 缺依赖时降级为「全部走默认值」
    yaml = None  # type: ignore[assignment]

# backend-py/app/core/config.py -> core -> app -> backend-py -> 仓库根
# ⚠️ 2026-09-15 本文件从 `app/config.py` 移进 `app/core/` ⇒ **深度多了一层**，两个 parents[N]
#    都跟着 +1（当时漏改 ⇒ 冒烟立刻红：「PROJECT_ROOT == repo root」✗）。移动路径类模块时，
#    必须先数清 `parents[N]`。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 后端包根（``backend-py/``）—— **唯一权威**。凡「属于后端、又不该放仓库根」的路径（技能库、
#: 本地服务根…）都从这里派生；⚠️ 与 ``scripts/`` 侧的同名常量靠注释同步（scripts 不 import ``app.*``）。
BACKEND_PY_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = Path(os.environ["CONFIG_PATH"]) if os.environ.get("CONFIG_PATH") else (
    PROJECT_ROOT / "configs" / "config.yaml"
)


def skills_dir() -> Path:
    """技能库目录（``backend-py/app/skills``）—— **唯一权威**；``SKILLS_DIR`` 环境变量可覆盖（测试隔离用）。

    ⚠️ 2026-09-15 收口：此前这个路径在**三处**各写一遍（``services/skills.py`` 的常量、
    ``services/agents/skills.py`` 的 ``skills_dir()``、``scripts/check_skill_refs.py`` 的常量），
    并靠注释互相提醒「必须一致」✗ —— 那是「搬库漏改一处**不会报错**、只是静默读不到技能」的坑。
    现在前两处都从这里取；``scripts/`` 那份**刻意保留**独立实现（scripts 不 import ``app.*``，
    见 ``scripts/README.md`` 的依赖约定），由守卫自检对齐。

    与 ``PROJECT_ROOT`` 的区别：技能库在 **`app/` 内**（``backend-py/app/skills/``），故这里从
    **本文件位置**推导（``app/core/config.py`` 上跳两级 = ``backend-py``），与 ``process.cwd()`` 解耦。
    """
    override = os.environ.get("SKILLS_DIR")
    if override:
        return Path(override)
    return BACKEND_PY_ROOT / "app" / "skills"


def _load_raw() -> dict[str, Any]:
    """解析 config.yaml；缺失/解析失败时降级为空对象（全部走默认值，不阻塞启动）。"""
    if yaml is None or not CONFIG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - 与 Node 版一致：任何异常都不阻断启动
        print(f"[config] 读取 {CONFIG_PATH} 失败，回退默认配置: {exc}")
        return {}


_RAW = _load_raw()
_SERVER: dict[str, Any] = _RAW.get("server") or {}
_DATABASE: dict[str, Any] = _RAW.get("database") or {}
_EVALUATION: dict[str, Any] = _RAW.get("evaluation") or {}
_AUTO_OPTIMIZE: dict[str, Any] = _EVALUATION.get("auto_optimize") or {}


def _resolve_path(value: Any) -> Path | None:
    """config.yaml 中的相对路径统一相对项目根解析。"""
    if not isinstance(value, str) or not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


DATA_ROOT_MARKER = PROJECT_ROOT / ".data-root"


def _read_data_root_marker() -> str:
    try:
        v = DATA_ROOT_MARKER.read_text(encoding="utf-8").strip()
        if v and Path(v).exists():
            return v
    except OSError:
        pass
    return ""


_default_data_dir = PROJECT_ROOT / "data"
_yaml_db_path = _resolve_path(_DATABASE.get("path"))

# 是否通过「标记文件 / DATA_ROOT 环境变量」显式指定过数据根目录。
# 显式指定后 DB_PATH / STORAGE_PATH 不再覆盖，确保目录切换真正生效。
_marker_root = _read_data_root_marker()

if _marker_root:
    _data_root = Path(_marker_root)
    _data_root_explicit = True
elif os.environ.get("DATA_ROOT"):
    _data_root = (PROJECT_ROOT / os.environ["DATA_ROOT"]).resolve()
    _data_root_explicit = True
else:
    _data_root = _yaml_db_path.parent if _yaml_db_path else _default_data_dir
    _data_root_explicit = False


def get_data_root() -> str:
    """当前数据根目录（含数据库与全部生成文件）。"""
    return str(_data_root)


def get_db_path() -> str:
    """当前 SQLite 数据库文件绝对路径。"""
    if not _data_root_explicit and os.environ.get("DB_PATH"):
        return os.environ["DB_PATH"]
    return str(_data_root / "drama.db")


def get_storage_root() -> str:
    """当前生成文件存储根目录绝对路径（<dataRoot>/static）。"""
    if not _data_root_explicit and os.environ.get("STORAGE_PATH"):
        return os.environ["STORAGE_PATH"]
    return str(_data_root / "static")


def set_data_root(new_root: str) -> None:
    """运行时切换数据根目录（写标记文件，重启后依然生效）。"""
    global _data_root, _data_root_explicit
    abs_path = Path(new_root).resolve()
    _data_root = abs_path
    _data_root_explicit = True
    abs_path.mkdir(parents=True, exist_ok=True)
    DATA_ROOT_MARKER.write_text(str(abs_path), encoding="utf-8")


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _cors_origins() -> list[str]:
    from_env = _env_list("CORS_ORIGINS")
    if from_env:
        return from_env
    from_yaml = _SERVER.get("cors_origins")
    if isinstance(from_yaml, list) and from_yaml:
        return [str(x) for x in from_yaml]
    # 与 Node 版默认值一致
    return ["http://localhost:3013", "http://localhost:5789", "http://localhost:5790"]


server = {
    # ⚠️ 刻意**不读** config.yaml 的 server.port：它属于 Node 后端（当前是 5789），
    # 绞杀者阶段两个后端要并存，读同一个端口号必然抢占。
    # 端口优先级：PY_PORT > PORT > 5790。完全迁移后想单端口（5789 同时服务 API 与前端）时，
    # 启动时显式给 PY_PORT=5789 即可。
    "port": int(os.environ.get("PY_PORT") or os.environ.get("PORT") or 5790),
    "host": str(os.environ.get("HOST") or _SERVER.get("host") or "0.0.0.0"),
    "cors_origins": _cors_origins(),
}

evaluation = {
    "auto_optimize": {
        "enabled": os.environ.get("AUTO_OPTIMIZE_ENABLED") == "true"
        or _AUTO_OPTIMIZE.get("enabled") is True,
        "hour": int(os.environ.get("AUTO_OPTIMIZE_HOUR") or _AUTO_OPTIMIZE.get("hour") or 3),
        "minute": int(os.environ.get("AUTO_OPTIMIZE_MINUTE") or _AUTO_OPTIMIZE.get("minute") or 0),
        "iterations": int(
            os.environ.get("AUTO_OPTIMIZE_ITERATIONS") or _AUTO_OPTIMIZE.get("iterations") or 3
        ),
        "run_on_startup": os.environ.get("AUTO_OPTIMIZE_RUN_ON_STARTUP") == "true"
        or _AUTO_OPTIMIZE.get("run_on_startup") is True,
    }
}

# ===== 绞杀者（strangler）遗留开关 =====
# ⚠️ 2026-09-15：Node 后端（`backend/`）**已删除** ⇒ 这两个开关**没有反代对象**了。
# 保留只为「接缝」语义：未实现的路径仍然回 501 并说明原因（比静默 404 好排查）。
# 换句话说：**现在不要设 PROXY_TO_NODE=1** —— 那只会连不上并返 502。
PROXY_TO_NODE = os.environ.get("PROXY_TO_NODE", "0") == "1"
NODE_BACKEND_URL = (os.environ.get("NODE_BACKEND_URL") or "http://127.0.0.1:5789").rstrip("/")

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
