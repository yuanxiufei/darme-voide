"""本机 Ollama 模型库的**离线**读取 ✓（不依赖 ``ollama serve`` ✓）—— 2026-09-25 接。

## 它解决什么真问题

`/api/tags` 是**唯一**能列出 Ollama 模型的官方接口 ✗ —— 但它要求 ``ollama serve`` 在跑 ✗。
于是「电脑里明明装着模型，服务没起就一个都看不见」✗✗（前端只显示「未运行」+ 空列表 ✗），
与「**只要后端跑起来就能扫本机模型**」冲突 ✗。本模块把 Ollama 的模型库当**文件系统**读 ✓：

* 根目录：``OLLAMA_MODELS`` > ``%LOCALAPPDATA%\\Ollama\\models`` > ``~/.ollama/models`` ✓；
* 清单：``manifests/<registry>/[<namespace>/]<model>/<tag>`` ✓（JSON ✓ 一个 tag 一个文件 ✓）；
* 内容：``blobs/sha256-<hex>`` ✓（**内容寻址** ⇒ 多个 tag 共享同一个 blob ⇒ 求和前先按 digest 去重 ✓）。

## 边界（本仓铁律：**没查 ≠ 通过** ✓）

* **只读** ✓：不写、不删、不改。删除仍走服务端 ``/api/delete`` ✓ —— 那是模型库一致性的边界 ✓；
* ``size`` 优先取**盘上 blob 的真实字节** ✓（清单自述的 ``size`` 可能对不上真实磁盘 ⇒ 那正是要暴露的事 ✓），
  blob 缺失才回落到自述值 ✓；
* ``missingBlobs`` 明确数出缺失的 blob ✓ —— ⚠️ 「清单在、blob 没了」**不是**「模型可用」✗✗，
  调用方要把它当**警告**呈现（真的能加载吗？只有服务端说了算 ✓）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from .ollama import format_bytes

__all__ = [
    "blob_path",
    "is_available",
    "list_models",
    "models_root",
    "models_root_candidates",
]

#: 模型库内的两个固定子目录（Ollama 自己的约定 ✓）
MANIFEST_DIR = "manifests"
BLOB_DIR = "blobs"

#: 默认注册表主机名与「无命名空间」的占位命名空间 ✓
DEFAULT_REGISTRY = "registry.ollama.ai"
LIBRARY_NAMESPACE = "library"


def models_root_candidates() -> list[str]:
    """候选模型库根目录（**保序** ✓：环境变量 > Windows 新版落点 > 经典落点 ✓）。"""
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if value and value.strip():
            cleaned = os.path.abspath(value.strip())
            if cleaned not in candidates:
                candidates.append(cleaned)

    add(os.environ.get("OLLAMA_MODELS"))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        # Ollama 0.3.x 起在 Windows 上改落这里 ✓（老装机仍是 ``~/.ollama/models`` ✓）
        add(os.path.join(local_app_data, "Ollama", "models"))
    home = os.path.expanduser("~")
    if home and home != "~":
        add(os.path.join(home, ".ollama", "models"))
    return candidates


def models_root() -> str:
    """实际使用的模型库根 ✓：**第一个存在的候选** ✓；都不存在 ⇒ 返回经典落点 ✓。

    ⚠️ 都不存在时返回候选而不是空串 —— 调用方拿它拼「装在哪才找得到」的提示 ✓
    （本仓口径：说得出**怎么让它可查** ✓，而不是干巴巴一句「没找到」✗）。
    """
    candidates = models_root_candidates()
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[-1] if candidates else ""


def is_available(root: str | None = None) -> bool:
    """模型库是否**真的可读** ✓（看 ``manifests/`` 在不在 ✓）。"""
    target = root if root is not None else models_root()
    return bool(target) and os.path.isdir(os.path.join(target, MANIFEST_DIR))


def blob_path(digest: str, root: str | None = None) -> str | None:
    """按摘要定位盘上的 blob ✓（``sha256:abc`` ⇒ ``blobs/sha256-abc`` ✓；不存在 ⇒ ``None`` ✓）。"""
    target = root if root is not None else models_root()
    text = str(digest or "").strip()
    if not target or not text:
        return None
    name = text.replace(":", "-")   # ⚠️ 落盘命名是**短横**，不是 API 里的冒号 ✓
    path = os.path.join(target, BLOB_DIR, name)
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


def _sha256_file(path: str) -> str:
    """清单自带的摘要 = **清单文件的 sha256** ✓（与 ``/api/tags`` 的 ``digest`` 同口径 ✓）。"""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
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


def list_models(root: str | None = None) -> list[dict[str, Any]]:
    """列出**盘上**的 Ollama 模型 ✓（零外部依赖 ✓ 只读 ✓）。读不了 ⇒ 空列表 ✓ 不抛错 ✓。

    返回项的键与 ``/api/tags`` 对齐（``name`` / ``size`` / ``size_label`` / ``modified_at`` /
    ``digest``）✓，另加 ``source='disk'``、``path``、``missingBlobs`` 供调用方区分来源与告警 ✓。
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

        try:
            modified = _iso_utc(os.stat(path).st_mtime)
        except OSError:
            modified = ""

        digest_hex = _sha256_file(path)
        models.append({
            "name": name,
            "size": total,
            "size_label": format_bytes(total),
            "modified_at": modified,
            "digest": f"sha256:{digest_hex}" if digest_hex else "",
            "source": "disk",
            "path": os.path.abspath(path),
            "blobs": len(seen),
            "missingBlobs": missing,
        })

    models.sort(key=lambda item: item["name"])
    return models
