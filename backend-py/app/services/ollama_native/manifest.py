"""本地模型仓库（manifests + blobs）的**写入侧** —— 移植 ``reference/ollama/manifest``。

ollama 的本地仓库是「内容寻址」的：真正的大文件放 ``blobs/sha256-<hex>``，
``manifests/<host>/<namespace>/<model>/<tag>`` 只是一张 JSON 清单，
列出这个模型由哪几个 blob 拼成。

    模型目录/
      blobs/sha256-abc…            ← 权重、模板、参数、license 都是 blob
      manifests/registry.ollama.ai/library/qwen3/14b   ← 清单 JSON

⚠️ 与 ``app/services/ollama_store.py`` 的分工：那边是**只读**的（列模型、找 blob、
读 Desktop 的 db 找根目录），这里负责**写**（落 blob、写/删清单、回收孤立 blob）。
读侧不要在这里重写一遍，写侧也不要绕开 ``ollama_store.models_root()`` 自己猜路径 ——
两边必须是同一个根，否则「列表里看得到、删除时找不到」。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable

from .names import Name


def models_root() -> str:
    """**延迟导入** ``ollama_store.models_root``（读侧的唯一根目录真源）。

    ⚠️ 不能在模块顶部 ``from ..ollama_store import models_root``：``ollama_store`` 为了剥
    ``OLLAMA_MODELS`` 的引号会 import ``ollama_native.envconfig``，而导入子模块必先执行
    ``ollama_native/__init__.py``（它导入本模块）⇒ **循环导入**，ImportError 落在
    ``models_root`` 这个名字还没绑定的瞬间。函数内导入把环断开，且不影响「同一个根」。
    """
    from ..ollama_store import models_root as _root

    return _root()

# ── layer 类型（与上游字面值逐字一致）────────────────────────────────────
MEDIA_TYPE_MODEL = "application/vnd.ollama.image.model"
MEDIA_TYPE_ADAPTER = "application/vnd.ollama.image.adapter"
MEDIA_TYPE_TEMPLATE = "application/vnd.ollama.image.template"
MEDIA_TYPE_SYSTEM = "application/vnd.ollama.image.system"
MEDIA_TYPE_PARAMS = "application/vnd.ollama.image.params"
MEDIA_TYPE_LICENSE = "application/vnd.ollama.image.license"
MEDIA_TYPE_MESSAGES = "application/vnd.ollama.image.messages"
MEDIA_TYPE_PROJECTOR = "application/vnd.ollama.image.projector"
MEDIA_TYPE_JSON = "application/vnd.ollama.image.json"
MEDIA_TYPE_TENSOR = "application/vnd.ollama.image.tensor"
MEDIA_TYPE_DRAFT = "application/vnd.ollama.image.draft"
MEDIA_TYPE_CONFIG = "application/vnd.ollama.image.config"
MEDIA_TYPE_EMBED = "application/vnd.ollama.image.embed"
MEDIA_TYPE_PROMPT = "application/vnd.ollama.image.prompt"

#: 清单文件自身的 mediaType（docker distribution v2 格式）
MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
#: config blob 的 mediaType
CONFIG_MEDIA_TYPE = "application/vnd.docker.container.image.v1+json"

_DIGEST_RE = re.compile(r"^sha256[:-][0-9a-fA-F]{64}$")

#: Modelfile 指令名 → 该指令落成哪种 layer（``create`` 组装清单时用）
PARAM_LAYER_TYPE = MEDIA_TYPE_PARAMS


class DigestError(ValueError):
    """digest 格式不合法（必须是 ``sha256:<64 位 hex>`` 或 ``sha256-<64 位 hex>``）。"""


class UnqualifiedNameError(ValueError):
    """模型名不完整，无法定位清单文件。"""


def validate_digest(digest: str) -> None:
    """:raises DigestError: 格式不对"""
    if not _DIGEST_RE.match(digest or ""):
        raise DigestError(f"invalid digest format: {digest!r}")


def manifests_dir(create: bool = True) -> str:
    """``<models>/manifests``；``create=True`` 时确保存在。"""
    path = os.path.join(models_root(), "manifests")
    if create:
        os.makedirs(path, mode=0o755, exist_ok=True)
    return path


def blobs_dir(create: bool = True) -> str:
    path = os.path.join(models_root(), "blobs")
    if create:
        os.makedirs(path, mode=0o755, exist_ok=True)
    return path


def blobs_path(digest: str, create: bool = True) -> str:
    """digest → blob 文件路径。**清单里写 ``:``，文件名用 ``-``**（上游就这规矩）。"""
    if digest:
        validate_digest(digest)
    filename = digest.replace(":", "-")
    path = os.path.join(blobs_dir(create), filename)
    if create:
        os.makedirs(os.path.dirname(path) if filename else path, mode=0o755, exist_ok=True)
    return path


def manifest_path(name: Name) -> str:
    """:raises UnqualifiedNameError: 名字四段不全"""
    if not name.is_valid():
        raise UnqualifiedNameError(f"unqualified name: {name}")
    return os.path.join(manifests_dir(), *name.filepath().split("/"))


def prune_directory(path: str) -> None:
    """自底向上删掉空目录（删完模型后清理 ``manifests/<host>/<ns>/<model>/``）。"""
    if not os.path.exists(path) or os.path.islink(path) or not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        prune_directory(os.path.join(path, entry))
    if not os.listdir(path):
        os.rmdir(path)


@dataclass
class Layer:
    """清单里的一条 layer。``from_`` 序列化成 ``from``（``from`` 是 Python 关键字）。"""

    media_type: str = ""
    digest: str = ""
    size: int = 0
    from_: str = ""
    name: str = ""
    status: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
        }
        if self.from_:
            data["from"] = self.from_
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer":
        return cls(
            media_type=str(data.get("mediaType") or ""),
            digest=str(data.get("digest") or ""),
            size=int(data.get("size") or 0),
            from_=str(data.get("from") or ""),
            name=str(data.get("name") or ""),
        )

    def read_bytes(self) -> bytes:
        with open(blobs_path(self.digest, create=False), "rb") as handle:
            return handle.read()

    def read_text(self) -> str:
        return self.read_bytes().decode("utf-8", errors="replace")


def new_layer(data: bytes, media_type: str) -> Layer:
    """把一段内容写成 blob（内容已存在则复用），返回对应 :class:`Layer`。"""
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    path = blobs_path(digest)
    status = "using existing layer"
    if not os.path.exists(path):
        status = "creating new layer"
        # 先写临时文件再改名：避免中途失败留下半截 blob 被当成完好的
        fd, temp_path = tempfile.mkstemp(prefix="sha256-", dir=blobs_dir())
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temp_path, path)
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    _touch(path)
    return Layer(
        media_type=media_type,
        digest=digest,
        size=len(data),
        status=f"{status} {digest}",
    )


def new_layer_from_file(path: str, media_type: str, chunk_size: int = 1 << 20) -> Layer:
    """把磁盘文件写成 blob（大模型权重走这条，不整块读进内存）。"""
    digest_hash = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest_hash.update(chunk)
            size += len(chunk)

    digest = f"sha256:{digest_hash.hexdigest()}"
    target = blobs_path(digest)
    status = "using existing layer"
    if not os.path.exists(target):
        status = "creating new layer"
        fd, temp_path = tempfile.mkstemp(prefix="sha256-", dir=blobs_dir())
        try:
            with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
                while chunk := src.read(chunk_size):
                    out.write(chunk)
            os.replace(temp_path, target)
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    _touch(target)
    return Layer(
        media_type=media_type,
        digest=digest,
        size=size,
        status=f"{status} {digest}",
    )


def new_layer_from_layer(digest: str, media_type: str, from_: str = "") -> Layer:
    """引用已有 blob 生成 layer（复制模型时用，不重写文件）。"""
    if not digest:
        raise DigestError("creating new layer from layer with empty digest")
    path = blobs_path(digest, create=False)
    if not os.path.exists(path):
        raise FileNotFoundError(f"blob 不存在: {digest}")
    _touch(path)
    return Layer(
        media_type=media_type,
        digest=digest,
        size=os.path.getsize(path),
        from_=from_,
        status=f"using existing layer {digest}",
    )


def _touch(path: str) -> None:
    """更新访问/修改时间 —— ollama 靠 mtime 做 LRU 淘汰，读一次就要"摸"一次。"""
    try:
        os.utime(path, None)
    except OSError:
        pass


@dataclass
class Manifest:
    schema_version: int = 2
    media_type: str = MANIFEST_MEDIA_TYPE
    config: Layer = field(default_factory=Layer)
    layers: list[Layer] = field(default_factory=list)

    #: 运行时信息（不参与序列化）
    filepath: str = ""
    digest: str = ""

    def size(self) -> int:
        return sum(layer.size for layer in [*self.layers, self.config])

    def config_layer(self, config_path: str) -> Layer | None:
        for layer in self.layers:
            if layer.media_type == MEDIA_TYPE_JSON and layer.name == config_path:
                return layer
        return None

    def read_config(self, config_path: str) -> bytes:
        layer = self.config_layer(config_path)
        if layer is None:
            raise KeyError(f'config "{config_path}" not found in manifest')
        return layer.read_bytes()

    def read_config_json(self, config_path: str) -> Any:
        return json.loads(self.read_config(config_path).decode("utf-8"))

    def tensor_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.media_type == MEDIA_TYPE_TENSOR]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "mediaType": self.media_type,
            "config": self.config.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        return cls(
            schema_version=int(data.get("schemaVersion") or 2),
            media_type=str(data.get("mediaType") or MANIFEST_MEDIA_TYPE),
            config=Layer.from_dict(data.get("config") or {}),
            layers=[Layer.from_dict(item or {}) for item in (data.get("layers") or [])],
        )

    # ── 删除 ────────────────────────────────────────────────────────────
    def remove(self) -> None:
        """删掉这份清单文件，并回收空目录。"""
        os.unlink(self.filepath)
        prune_directory(manifests_dir())

    def remove_layers(self) -> list[str]:
        """删掉**没有其它清单引用**的 blob，返回被删的 digest 列表。"""
        in_use: set[str] = set()
        for _, other in all_manifests(continue_on_error=True).items():
            for layer in [*other.layers, other.config]:
                if layer.digest:
                    in_use.add(layer.digest)

        removed: list[str] = []
        for layer in [*self.layers, self.config]:
            if not layer.digest or layer.digest in in_use:
                continue
            path = blobs_path(layer.digest, create=False)
            if os.path.exists(path):
                os.unlink(path)
            removed.append(layer.digest)
        return removed


def _manifest_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_manifest(name: Name, config: Layer, layers: Iterable[Layer]) -> str:
    """写清单文件，返回其路径（父目录自动创建）。"""
    path = manifest_path(name)
    os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
    manifest = Manifest(config=config, layers=list(layers))
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, separators=(",", ":"))
    fd, temp_path = tempfile.mkstemp(prefix="manifest-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return path


def read_manifest(name: Name) -> Manifest:
    """读清单（带上文件路径与内容摘要）。"""
    path = manifest_path(name)
    with open(path, "rb") as handle:
        raw = handle.read()
    manifest = Manifest.from_dict(json.loads(raw.decode("utf-8")))
    manifest.filepath = path
    manifest.digest = _manifest_digest(raw)
    return manifest


def all_manifests(continue_on_error: bool = True) -> dict[Name, Manifest]:
    """扫描全部清单（``continue_on_error=False`` 时遇到坏文件直接抛）。"""
    from .names import parse_name_from_filepath

    root = manifests_dir()
    found: dict[Name, Manifest] = {}
    if not os.path.isdir(root):
        return found

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            relative = os.path.relpath(full, root)
            name = parse_name_from_filepath(relative)
            if not name.is_valid():
                if not continue_on_error:
                    raise UnqualifiedNameError(f"bad manifest name: {relative}")
                continue
            try:
                found[name] = read_manifest(name)
            except (OSError, ValueError):
                if not continue_on_error:
                    raise
    return found


__all__ = [
    "CONFIG_MEDIA_TYPE",
    "DigestError",
    "Layer",
    "MANIFEST_MEDIA_TYPE",
    "Manifest",
    "MEDIA_TYPE_ADAPTER",
    "MEDIA_TYPE_CONFIG",
    "MEDIA_TYPE_DRAFT",
    "MEDIA_TYPE_EMBED",
    "MEDIA_TYPE_JSON",
    "MEDIA_TYPE_LICENSE",
    "MEDIA_TYPE_MESSAGES",
    "MEDIA_TYPE_MODEL",
    "MEDIA_TYPE_PARAMS",
    "MEDIA_TYPE_PROMPT",
    "MEDIA_TYPE_PROJECTOR",
    "MEDIA_TYPE_SYSTEM",
    "MEDIA_TYPE_TEMPLATE",
    "MEDIA_TYPE_TENSOR",
    "UnqualifiedNameError",
    "all_manifests",
    "blobs_dir",
    "blobs_path",
    "manifest_path",
    "manifests_dir",
    "new_layer",
    "new_layer_from_file",
    "new_layer_from_layer",
    "prune_directory",
    "read_manifest",
    "validate_digest",
    "write_manifest",
]
