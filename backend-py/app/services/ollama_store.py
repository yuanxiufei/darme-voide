"""本机 Ollama 模型库的**离线**读取 ✓（不依赖 ``ollama serve`` ✓）—— 2026-09-25 接。

## 它解决什么真问题

`/api/tags` 是**唯一**能列出 Ollama 模型的官方接口 ✗ —— 但它要求 ``ollama serve`` 在跑 ✗。
于是「电脑里明明装着模型，服务没起就一个都看不见」✗✗（前端只显示「未运行」+ 空列表 ✗），
与「**只要后端跑起来就能扫本机模型**」冲突 ✗。本模块把 Ollama 的模型库当**文件系统**读 ✓：

* 根目录：``OLLAMA_MODELS`` > **桌面端声明的库根** > ``%LOCALAPPDATA%\\Ollama\\models``
  > ``~/.ollama/models`` ✓；
* 清单：``manifests/<registry>/[<namespace>/]<model>/<tag>`` ✓（JSON ✓ 一个 tag 一个文件 ✓）；
* 内容：``blobs/sha256-<hex>`` ✓（**内容寻址** ⇒ 多个 tag 共享同一个 blob ⇒ 求和前先按 digest 去重 ✓）。

## ⚠️ 两个踩过的真坑（2026-09-25 真机复现 ✓）

1. **桌面端把库装到自定义目录时，后端进程看不见那个环境变量** ✗✗：
   ``OLLAMA_MODELS`` 只被 Ollama 桌面端**注入给 ``ollama serve`` 子进程** ✓，
   ``HKCU``/``HKLM`` 环境里**没有** ✗ ⇒ 后端只剩两个默认落点可选 ✗。
   真机实测（2026-09-25 ✓）：库被装到**非默认盘的自建目录**（该机 14 个模型 ✓），
   而候选里最靠前的**存在**目录是空壳 ``~/.ollama/models`` ✗ ⇒ ``list_models()`` 返回 ``[]`` ✗✗。
   ⚠️ 具体盘符/目录**不写进代码** ✗（用户 2026-09-26 口径 ✓：扫描路径一律不写死 ✗、
   机器布局不进注释 ✗）⇒ 这里只留「确实发生过这件事」+ 机制本身 ✓。
   正解：桌面端把该路径**持久化在自己的配置库里**（``settings.models`` ✓）⇒ 读它 ✓（只读、仍不依赖服务 ✓）。
2. **「目录存在」不等于「库可读」** ✗：只有 ``manifests/`` 在，才真的列得出模型 ✓。
   按「存在」挑根 ⇒ ``is_available()`` 报 True 而列表是空 ✗✗（最坏的一种：前端显示「0 个模型」，
   连「库找错地方」都看不出来 ✗）⇒ 挑根条件改成**先挑真能读的** ✓。

## 边界（本仓铁律：**没查 ≠ 通过** ✓）

* **只读** ✓：不写、不删、不改。删除仍走服务端 ``/api/delete`` ✓ —— 那是模型库一致性的边界 ✓；
* ``size`` 优先取**盘上 blob 的真实字节** ✓（清单自述的 ``size`` 可能对不上真实磁盘 ⇒ 那正是要暴露的事 ✓），
  blob 缺失才回落到自述值 ✓；
* ``missingBlobs`` 明确数出缺失的 blob ✓（**含 config 层** ✓ —— config 没了同样加载不了 ✓✗，
  另给 ``missingConfig`` 布尔 ✓，但 ``size`` 口径仍**不含** config ✓ 以对齐 ``/api/tags`` ✓）；
* ``digest`` = **清单文件自身的 sha256** ✓（与 ``/api/tags`` 同口径 ✓）；**blob 的内容 sha256** 要
  单独查 ✓（:func:`verify_blob` ✓）—— 两者是两件事 ✗✗。

## 清单 ↔ blob 的 sha256 核对（2026-10-29 补）

内容寻址仓的**唯一**自证方式：blob 文件名里的摘要 = 该文件内容的 sha256 ✓。清单里的
``digest``/``size`` 只是**自述** ✗ —— 下载中断、磁盘错误、手工拷贝都会留下「名字对、内容错」的 blob ✗，
而 :func:`list_models` 只 stat 大小 ⇒ **看不出来** ✗（大小相同的坏文件照样通过 ✗）。
故补 :func:`verify_blob`（真算 sha256 ✓）/ :func:`verify_model` / :func:`verify_store`：
大权重可以按 ``max_blob_bytes`` 跳过 ✓，但**跳过要说出来**（``status='skipped'`` ✓ 不算通过 ✗）。

真机实测（2026-10-29，4 个模型 / 16 个 blob）：12 个小 blob（17 B…11 KB）与 1 个 274 MB 权重
**sha256 全部等于文件名摘要** ✓（274 MB 用时 153 ms ≈ 1.8 GB/s 热缓存 ✓）；
其余 4 个大权重（0.9/5.2/9.3/16.5 GB）按预算跳过 ✓ —— 这不是「验证通过」✗。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .ollama import format_bytes
# ⚠️ 只借它剥引号：Windows 用户常写 `set OLLAMA_MODELS="<盘符>:\models"`，
# 引号会一路拼进 `"<盘符>:\models"/manifests` ⇒ 目录判不存在 ⇒ 模型全丢。
# 上游 `envconfig.Var` 做的就是 `strings.Trim(s, "\"'")`，这里复用同一份口径，
# 不在本模块再写第二套（那正是「同一个变量两处猜、结果分叉」的来源）。
from .ollama_native.envconfig import Var as _env_var

__all__ = [
    "DIGEST_RE",
    "blob_path",
    "declared_models_root",
    "desktop_settings_databases",
    "find_manifest",
    "is_available",
    "list_models",
    "manifest_files",
    "models_root",
    "models_root_candidates",
    "read_manifest",
    "verify_blob",
    "verify_model",
    "verify_store",
]

#: 模型库内的两个固定子目录（Ollama 自己的约定 ✓）
MANIFEST_DIR = "manifests"
BLOB_DIR = "blobs"

#: 默认注册表主机名与「无命名空间」的占位命名空间 ✓
DEFAULT_REGISTRY = "registry.ollama.ai"
LIBRARY_NAMESPACE = "library"

#: Ollama **桌面端**配置库的位置（相对 Windows 的 ``LOCALAPPDATA`` / ``APPDATA`` ✓）
DESKTOP_DB_RELATIVE = os.path.join("Ollama", "db.sqlite")

#: 桌面端 ``settings`` 表里存「模型库根」的列名 ✓（真机实测 ✓ 不是猜的 ✓）
DESKTOP_MODELS_COLUMN = "models"

#: ⚠️ 请求链路上 :func:`models_root` 会被反复调用 ⇒ 不能每次都开一次 sqlite ✗
#: 缓存形如 ``db 路径 -> (mtime, 大小, 声明值)`` ✓：文件一变就重读 ✓
_DECLARED_CACHE: dict[str, tuple[float, int, str | None]] = {}

#: 合法摘要形状 ✓：``sha256`` + ``:``/``-``（**可省** ✓）+ **64 位 hex** ✓（大小写不敏感 ✓）。
#: ⚠️ 摘要会被**拼成文件名** ✓（清单来自注册表 / 别人的模型 ✓）⇒ 形状必须卡死 ✗：
#: 不卡就能把 ``../`` 一类的段写进路径（能不能真跳出 ``blobs/`` 取决于中间目录在不在 ✓ 别赌 ✗），
#: 更常见的坏处是**非法摘要被当成「缺点文件」** ✗ —— 坏清单与缺文件是两回事 ✗✗。
#: 前缀可省是**宽容**而非松懈 ✓：只放行 64 位 hex ⇒ 裸 hex 拼不出 ``..`` ✓（真机清单都带前缀 ✓）。
DIGEST_RE = re.compile(r"^(?:sha256[:-])?([0-9a-f]{64})$", re.IGNORECASE)


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

    add(_env_var("OLLAMA_MODELS"))
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


def models_root() -> str:
    """实际使用的模型库根 ✓：**第一个真能读的候选** ✓ > 第一个存在的候选 ✓ > 经典落点 ✓。

    ⚠️ 都不存在时返回候选而不是空串 —— 调用方拿它拼「装在哪才找得到」的提示 ✓
    （本仓口径：说得出**怎么让它可查** ✓，而不是干巴巴一句「没找到」✗）。

    ⚠️ **不能只挑「存在的」**：空壳目录（例如只有 ``~/.ollama/models`` 这个空目录 ✗）
    会让 :func:`is_available` 报 True、而 :func:`list_models` 返回空 ✗✗
    ⇒ 前端只会说「0 个模型」，看不出是**根挑错了** ✗（真机踩过 ✓）。
    """
    candidates = models_root_candidates()
    for candidate in candidates:
        if is_available(candidate):
            return candidate
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[-1] if candidates else ""


def is_available(root: str | None = None) -> bool:
    """模型库是否**真的可读** ✓（看 ``manifests/`` 在不在 ✓）。"""
    target = root if root is not None else models_root()
    return bool(target) and os.path.isdir(os.path.join(target, MANIFEST_DIR))


def digest_hex(digest: Any) -> str | None:
    """摘要 → **规范化的 64 位小写 hex** ✓（形状不对 ⇒ ``None`` ✓）。

    ⚠️ 不合法就当 ``None``：摘要要拼文件名 ✗ ⇒ 放着不管就是路径穿越（见 :data:`DIGEST_RE` ✓）。
    """
    match = DIGEST_RE.match(str(digest or "").strip())
    return match.group(1).lower() if match else None


def blob_path(digest: str, root: str | None = None) -> str | None:
    """按摘要定位盘上的 blob ✓（``sha256:abc…`` ⇒ ``blobs/sha256-abc…`` ✓；不存在/形状不对 ⇒ ``None`` ✓）。"""
    target = root if root is not None else models_root()
    hexdigest = digest_hex(digest)
    if not target or not hexdigest:
        return None
    # ⚠️ 落盘命名是**短横**，不是 API 里的冒号 ✓；且只用**校验过的 hex** 拼名 ✓
    path = os.path.join(target, BLOB_DIR, f"sha256-{hexdigest}")
    return path if os.path.isfile(path) else None


def _iter_manifest_files(manifests_dir: str) -> Iterator[str]:
    """遍历 ``manifests/`` 下的所有清单文件 ✓（跳过隐藏文件名 ⇒ 不把 ``.DS_Store`` 当清单 ✓）。"""
    for current, dirnames, filenames in os.walk(manifests_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            yield os.path.join(current, filename)


def _model_name(parts: list[str]) -> str | None:
    """``manifests`` 下的相对路径 → 模型名 ✓。

    * ``registry.ollama.ai/library/qwen3/8b`` ⇒ ``qwen3:8b`` ✓（``library`` 是默认命名空间 ⇒ 不显示 ✓）；
    * ``registry.ollama.ai/myorg/mymodel/v2`` ⇒ ``myorg/mymodel:v2`` ✓。

    ⚠️ 首段含 ``.`` / ``:`` / 是 ``localhost`` ⇒ 那是**注册表主机名**，不是命名空间 ✗（别把它读进名字里 ✗）。
    """
    if len(parts) < 3:
        return None
    head, tag = parts[0], parts[-1]
    middle = parts[1:-1]
    if not middle or not tag:
        return None
    if head == "localhost" or re.search(r"[.:]", head):
        namespace, model = middle[0], middle[-1]
    else:
        namespace, model = head, middle[-1]
    if not model:
        return None
    return f"{model}:{tag}" if namespace == LIBRARY_NAMESPACE else f"{namespace}/{model}:{tag}"


def _file_size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """文件内容的 sha256 ✓（清单小文件与大权重共用 ✓；读不了 ⇒ ``""`` ✓ 不抛错 ✓）。

    ⚠️ 块大小不是小事：64 KiB 块在 16 GB 权重上比 8 MiB 块慢一截 ✗（真机实测 274 MB：
    8 MiB 块 153 ms ✓）。默认 1 MiB 是「小清单不吃亏、大文件不太慢」的折中 ✓。
    """
    digest = hashlib.sha256()
    buffer = bytearray(chunk_size)
    view = memoryview(buffer)
    try:
        with open(path, "rb") as handle:
            while True:
                read = handle.readinto(buffer)
                if not read:
                    break
                digest.update(view[:read])
    except OSError:
        return ""
    return digest.hexdigest()


def _iso_utc(timestamp: float) -> str:
    """``mtime`` → ``2026-09-25T12:00:00.000Z`` ✓（与 ``/api/tags`` 的毫秒 + ``Z`` 同形 ✓）。"""
    moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _layer_digests(manifest: dict[str, Any]) -> list[tuple[str, int]]:
    """清单里的权重层 ``(digest, 自述 size)`` ✓ —— **不含 config 层** ✓（与 ``/api/tags`` 的 size 同口径 ✓）。"""
    found: list[tuple[str, int]] = []
    for layer in manifest.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        digest = str(layer.get("digest") or "").strip()
        if not digest:
            continue
        try:
            declared = int(layer.get("size") or 0)
        except (TypeError, ValueError):
            declared = 0
        found.append((digest, max(0, declared)))
    return found


def _config_layer(manifest: dict[str, Any]) -> tuple[str, int]:
    """清单的 config 层 ``(digest, 自述 size)`` ✓（没有/不是对象 ⇒ ``("", 0)`` ✓）。

    ⚠️ config 不是「可选的装饰」✗：它装着模型的参数与模板元数据 ✓ —— 没了照样加载不了 ✗✗，
    但 :func:`list_models` 的 ``size`` 口径**不含**它 ✓（要与 ``/api/tags`` 对齐 ✓）。
    """
    config = manifest.get("config")
    if not isinstance(config, dict):
        return "", 0
    digest = str(config.get("digest") or "").strip()
    try:
        declared = int(config.get("size") or 0)
    except (TypeError, ValueError):
        declared = 0
    return digest, max(0, declared)


def manifest_files(root: str | None = None) -> list[tuple[str, str]]:
    """``[(模型名, 清单路径)]`` ✓ —— **唯一**的「名字 → 文件」映射 ✓（按名反查不许自己拼路径 ✗）。

    ⚠️ 为什么不拼路径：仓库布局有 ``<host>/<ns>/<model>/<tag>`` 与「无 host」两种 ✓，
    再叠上 Windows 盘符大小写不敏感 ✗ —— 拼出来的路径「看着对、打开是另一个文件」不会报错 ✗✗。
    用 :func:`_model_name` 从**真实文件**推名字再比，名字一定反查得回去 ✓。
    """
    target = root if root is not None else models_root()
    manifests_dir = os.path.join(target, MANIFEST_DIR) if target else ""
    if not manifests_dir or not os.path.isdir(manifests_dir):
        return []
    found: list[tuple[str, str]] = []
    for path in _iter_manifest_files(manifests_dir):
        relative = os.path.relpath(path, manifests_dir).replace("\\", "/")
        name = _model_name(relative.split("/"))
        if name:
            found.append((name, path))
    found.sort()
    return found


def _name_key(name: str) -> str:
    """名字归一 ✓（``registry.ollama.ai/library/qwen3:8b`` ⇒ ``qwen3:8b`` ✓）。

    只剥**注册表主机**与默认命名空间 ``library`` ✓ —— 其它命名空间是名字的一部分 ✓（不能剥 ✗）。
    """
    parts = [part for part in str(name or "").strip().replace("\\", "/").split("/") if part]
    if len(parts) >= 3 and (parts[0].lower() == "localhost" or "." in parts[0] or ":" in parts[0]):
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == LIBRARY_NAMESPACE:
        parts = parts[1:]
    return "/".join(parts)


def find_manifest(name: str, root: str | None = None) -> str | None:
    """模型名 → 清单文件路径 ✓（找不到 ⇒ ``None`` ✓）。

    同一个名字在多个注册表下都存在时取**路径排序第一个** ✓（确定性 ✓，别随 ``os.walk`` 顺序漂 ✗）。
    """
    key = _name_key(name)
    if not key:
        return None
    for found_name, path in manifest_files(root):
        if _name_key(found_name) == key:
            return path
    return None


def _blob_entry(digest: str, declared: int, role: str, root: str | None) -> dict[str, Any]:
    """清单里的一个 blob 在盘上的实况 ✓（存在性 / 盘上字节 / 摘要形状 ✓）。"""
    canonical = digest_hex(digest)
    path = blob_path(digest, root)
    disk = _file_size(path) if path else 0
    return {
        "role": role,
        "digest": f"sha256:{canonical}" if canonical else str(digest),
        "declaredSize": declared,
        # 盘上优先 ✓（自述值可能是下载前写下的 ✓）
        "size": disk if path else declared,
        "path": path,
        "missing": path is None and canonical is not None,
        "invalidDigest": canonical is None,
        "sizeMismatch": bool(path and declared and disk != declared),
    }


def _role(media_type: Any) -> str:
    """``application/vnd.ollama.image.params`` ⇒ ``params`` ✓（认不出 ⇒ ``unknown`` ✓ 不猜 ✗）。"""
    text = str(media_type or "").strip()
    return text.rsplit(".", 1)[-1] if text else "unknown"


def read_manifest(name: str, root: str | None = None) -> dict[str, Any] | None:
    """单个模型的**完整清单视图** ✓（找不到 / 不是 JSON ⇒ ``None`` ✓ 不抛错 ✓）。

    返回：``name`` / ``path`` / ``sha256``（**清单文件自身**的 sha256 ✓ = ``/api/tags`` 的 digest 口径 ✓）/
    ``config`` / ``layers``（每层带 ``role`` ✓、盘上实况 ✓）/ ``totalSize``（**不含 config** ✓）/
    ``missing``（含 config ✓）/ ``missingConfig`` / ``sizeMismatch`` / ``invalidDigest``。
    """
    path = find_manifest(name, root)
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None

    config_digest, config_declared = _config_layer(manifest)
    config_entry = (_blob_entry(config_digest, config_declared, _role(
        (manifest.get("config") or {}).get("mediaType")), root) if config_digest else None)

    layers: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for item in manifest.get("layers") or []:
        if not isinstance(item, dict):
            continue
        digest = str(item.get("digest") or "").strip()
        if not digest:
            continue
        try:
            declared = max(0, int(item.get("size") or 0))
        except (TypeError, ValueError):
            declared = 0
        entry = _blob_entry(digest, declared, _role(item.get("mediaType")), root)
        layers.append(entry)
        key = entry["digest"]
        if key in seen:                 # 内容寻址 ⇒ 同一 blob 被多层引用时体积只算一次 ✓
            continue
        seen.add(key)
        total += entry["size"]

    missing = [entry["digest"] for entry in layers if entry["missing"]]
    invalid = [entry["digest"] for entry in layers if entry["invalidDigest"]]
    if config_entry is not None:
        if config_entry["missing"]:
            missing.append(config_entry["digest"])
        if config_entry["invalidDigest"]:
            invalid.append(config_entry["digest"])

    return {
        "name": _name_key(name),
        "path": os.path.abspath(path),
        "sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "schemaVersion": int(manifest.get("schemaVersion") or 2),
        "mediaType": str(manifest.get("mediaType") or ""),
        "config": config_entry,
        "layers": layers,
        "totalSize": total,
        "missing": missing,
        "missingConfig": bool(config_entry and config_entry["missing"]),
        "sizeMismatch": [entry["digest"] for entry in layers if entry["sizeMismatch"]],
        "invalidDigest": invalid,
    }


def verify_blob(digest: str, root: str | None = None,
                chunk_size: int = 1 << 20) -> dict[str, Any]:
    """**真算**一个 blob 的内容 sha256 并比对文件名摘要 ✓（内容寻址的唯一自证 ✓）。

    ``status`` 四态 ✓：``ok`` / ``mismatch``（内容被改/截断 ✗）/ ``missing`` / ``invalid-digest``
    （清单里的摘要形状不合法 ✗ —— ⚠️ 不能归进 ``missing`` ✗，那会把坏清单说成「缺点文件」✗）。
    """
    canonical = digest_hex(digest)
    path = blob_path(digest, root)
    report: dict[str, Any] = {
        "digest": f"sha256:{canonical}" if canonical else str(digest or ""),
        "path": path,
        "size": _file_size(path) if path else 0,
        "actual": "",
        "status": "invalid-digest",
        "ok": False,
    }
    if canonical is None:
        return report
    if path is None:
        report["status"] = "missing"
        return report
    started = time.monotonic()
    actual = _sha256_file(path, chunk_size)
    report["actual"] = f"sha256:{actual}" if actual else ""
    report["seconds"] = round(time.monotonic() - started, 3)
    report["ok"] = actual == canonical
    report["status"] = "ok" if report["ok"] else "mismatch"
    return report


def verify_model(name: str, root: str | None = None,
                 max_blob_bytes: int | None = 256 << 20) -> dict[str, Any]:
    """按清单逐个 blob 校验 ✓（``max_blob_bytes`` 以上的 blob ⇒ ``skipped`` ✓ **不算通过** ✗）。

    ⚠️ 大权重默认跳过：16 GB 权重一遍 sha256 是**真成本** ✓，而「悄悄跳过之后报 ok」是最坏的假绿 ✗✗
    ⇒ 结果里既有 ``complete``（有没有跳过 ✓）也有 ``skipped`` 明细 ✓。
    """
    detail = read_manifest(name, root)
    if detail is None:
        return {"name": _name_key(name), "found": False, "ok": False, "status": "no-manifest",
                "checked": [], "failed": [], "skipped": [], "missing": []}

    entries = ([detail["config"]] if detail["config"] else []) + detail["layers"]
    checked: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for entry in entries:
        over_budget = (max_blob_bytes is not None
                       and not entry["missing"] and not entry["invalidDigest"]
                       and entry["size"] > max_blob_bytes)
        if over_budget:
            report = {"digest": entry["digest"], "path": entry["path"], "size": entry["size"],
                      "actual": "", "status": "skipped", "ok": False,
                      "skippedBecause": f"{entry['size']} B > max_blob_bytes={max_blob_bytes}"}
        else:
            # 缺失 / 摘要不合法也走这里 ✓ —— :func:`verify_blob` 自己分会是 ``missing`` 还是
            # ``invalid-digest`` ✓（两者不能混 ✗：坏清单不是「少个文件」✗）
            report = verify_blob(entry["digest"], root)
        report = {**report, "role": entry["role"]}
        checked.append(report)
        if report["status"] in ("mismatch", "invalid-digest"):
            failed.append(report)
        elif report["status"] in ("missing",):
            missing.append(report)
        elif report["status"] == "skipped":
            skipped.append(report)

    return {
        "name": detail["name"],
        "found": True,
        "path": detail["path"],
        "manifestSha256": detail["sha256"],
        "ok": not failed and not missing,
        "complete": not skipped,
        "status": ("failed" if (failed or missing) else ("partial" if skipped else "ok")),
        "checked": checked,
        "failed": [item["digest"] for item in failed],
        "skipped": [item["digest"] for item in skipped],
        "missing": [item["digest"] for item in missing],
    }


def verify_store(root: str | None = None,
                 max_blob_bytes: int | None = 256 << 20) -> dict[str, Any]:
    """整库校验 ✓（每个清单一份报告 + 汇总 ✓）。只读 ✓ 不改任何 blob ✓。"""
    reports = [verify_model(name, root, max_blob_bytes) for name, _ in manifest_files(root)]
    return {
        "root": (root if root is not None else models_root()),
        "models": reports,
        "count": len(reports),
        "ok": all(item["ok"] for item in reports) and bool(reports),
        "complete": all(item["complete"] for item in reports),
        "failed": [item["name"] for item in reports if not item["ok"]],
        "skippedBlobs": sum(len(item["skipped"]) for item in reports),
    }


def list_models(root: str | None = None) -> list[dict[str, Any]]:
    """列出**盘上**的 Ollama 模型 ✓（零外部依赖 ✓ 只读 ✓）。读不了 ⇒ 空列表 ✓ 不抛错 ✓。

    返回项的键与 ``/api/tags`` 对齐（``name`` / ``size`` / ``size_label`` / ``modified_at`` /
    ``digest``）✓，另加 ``source='disk'``、``path``、``missingBlobs``（**含 config 层** ✓）、
    ``missingConfig`` 供调用方区分来源与告警 ✓。

    ⚠️ ``missingBlobs`` 只说明「清单点名的文件在不在」✗ —— **内容对不对**要另走
    :func:`verify_blob` ✓（大小相同的坏文件在这里看不见 ✗✗）。
    """
    target = root if root is not None else models_root()
    manifests_dir = os.path.join(target, MANIFEST_DIR) if target else ""
    if not manifests_dir or not os.path.isdir(manifests_dir):
        return []

    models: list[dict[str, Any]] = []
    for path in _iter_manifest_files(manifests_dir):
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
            manifest = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError):
            continue   # 坏清单/读不了 ⇒ 跳过这一条 ✓（不能因为一个坏文件就整片列不出来 ✗）
        if not isinstance(manifest, dict):
            continue

        relative = os.path.relpath(path, manifests_dir).replace("\\", "/")
        name = _model_name(relative.split("/"))
        if not name:
            continue

        # ⚠️ 按 digest **去重求和**：内容寻址 ⇒ 多个层（甚至跨模型）可共享同一 blob ✓
        seen: set[str] = set()
        total = 0
        missing = 0
        for digest, declared in _layer_digests(manifest):
            if digest in seen:
                continue
            seen.add(digest)
            found = blob_path(digest, target)
            if found is None:
                missing += 1
                total += declared       # 盘上没有 ⇒ 只能用自述值（并计入 missingBlobs ✓）
                continue
            actual = _file_size(found)
            total += actual if actual > 0 else declared

        # config 层：**不计进 size**（与 /api/tags 同口径 ✓）但**缺了要报** ✓✗
        config_digest, _config_declared = _config_layer(manifest)
        missing_config = bool(config_digest) and blob_path(config_digest, target) is None

        try:
            modified = _iso_utc(os.stat(path).st_mtime)
        except OSError:
            modified = ""

        manifest_sha = _sha256_file(path)
        models.append({
            "name": name,
            "size": total,
            "size_label": format_bytes(total),
            "modified_at": modified,
            "digest": f"sha256:{manifest_sha}" if manifest_sha else "",
            "source": "disk",
            "path": os.path.abspath(path),
            "blobs": len(seen),
            "missingBlobs": missing + (1 if missing_config else 0),
            "missingConfig": missing_config,
        })

    models.sort(key=lambda item: item["name"])
    return models
