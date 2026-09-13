"""文件存储工具 —— 移植 ``utils/storage.ts``。

已迁：落盘上传文件、取安全绝对路径、base64 存取、**下载远程文件**、**压缩参考图**。
``getAbsolutePath`` 的路径穿越防护是安全关键，必须逐行对齐。

⚠️ ``readImageAsCompressedDataUrl`` 的**真实压缩未迁**（依赖 ``sharp``）——
原实现是「按长边 ≤768 等比缩放（不放大）→ 有 alpha 就 flatten 白底 → JPEG q68 mozjpeg」。
当前**退化为原图 data URL**并告警：内容与画质不变，只是发给厂商的载荷更大。
装上 Pillow 后在此补齐即可（参数已在函数 docstring 里写全）。
"""

from __future__ import annotations

import base64
import os
import re
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..config import get_storage_root
from .task_logger import log_task_warn

#: mimeType ↔ 扩展名映射（与 TS 的两张表一字不差）
_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def get_absolute_path(relative_path: str) -> str:
    """相对路径 → 绝对路径（**防路径穿越**）。

    ⚠️ 一处容易看漏的原逻辑：若路径以 ``static/`` 开头，基准要**上跳到 storageRoot 的父目录**
    —— 因为 ``static/xxx`` 这类返回值是「相对数据根」而非「相对 storageRoot」的。
    路径穿越检查最终仍以**真实的 storageRoot** 为界，所以上跳本身不构成逃逸。
    """
    storage_root = Path(get_storage_root()).resolve()
    if relative_path.startswith("static/") or relative_path.startswith("static\\"):
        base = storage_root.parent
    else:
        base = storage_root

    resolved = (base / relative_path).resolve()
    if not (
        str(resolved).startswith(str(storage_root) + os.sep) or str(resolved) == str(storage_root)
    ):
        raise ValueError(f"Path traversal blocked: {relative_path}")
    return str(resolved)


def save_uploaded_file(data: bytes, sub_dir: str, original_name: str) -> str:
    """保存上传的文件，返回**相对数据根**的路径（``static/<subDir>/<uuid><ext>``）。"""
    directory = Path(get_storage_root()) / sub_dir
    directory.mkdir(parents=True, exist_ok=True)

    ext = os.path.splitext(original_name)[1] or ".bin"
    filename = f"{uuid.uuid4()}{ext}"
    (directory / filename).write_bytes(data)
    return f"static/{sub_dir}/{filename}"


def save_base64_image(base64_data: str, mime_type: str, sub_dir: str) -> str:
    """保存 Base64 图片（Gemini 等只返回 base64 的厂商用）。"""
    directory = Path(get_storage_root()) / sub_dir
    directory.mkdir(parents=True, exist_ok=True)

    ext = _MIME_TO_EXT.get(mime_type, ".png")
    filename = f"{uuid.uuid4()}{ext}"
    (directory / filename).write_bytes(base64.b64decode(base64_data))
    return f"static/{sub_dir}/{filename}"


def read_image_as_data_url(relative_path: str) -> str:
    """读本地图片为 data URL（不缩放；缩放版未迁，见模块头）。"""
    file_path = Path(get_absolute_path(relative_path))
    buffer = file_path.read_bytes()
    mime_type = _EXT_TO_MIME.get(file_path.suffix.lower(), "image/png")
    return f"data:{mime_type};base64,{base64.b64encode(buffer).decode('ascii')}"


def parse_data_url(data_url: str) -> dict[str, str] | None:
    """``data:image/png;base64,xxxx`` → ``{mimeType, data}``；不匹配返回 None。"""
    match = re.match(r"^data:([^;]+);base64,(.+)$", str(data_url or ""), re.S)
    if not match:
        return None
    return {"mimeType": match.group(1), "data": match.group(2)}


#: 下载体积上限 50MB（与原 TS 一致，防 OOM）
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def _ext_from_url(url: str) -> str:
    """镜像 TS 的 ``getExtFromUrl``：取 URL **路径**的扩展名，无/过长则 ``.bin``。

    原实现用 ``new URL(url)``（相对地址会抛错 → ``.bin``）+ ``path.extname``。
    ⚠️ ``path.extname('/a/b.')`` 是 ``'.'``（长度 1，会被接受）—— Python 的
    ``os.path.splitext`` 同样返回 ``'.'``，行为一致，别"顺手修好"。
    """
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url or ""):
        return ".bin"
    parsed = urllib.parse.urlsplit(url)
    ext = os.path.splitext(parsed.path)[1]
    if ext and len(ext) <= 5:
        return ext
    return ".bin"


async def download_file(url: str, sub_dir: str) -> str:
    """下载远程文件到 ``<storageRoot>/<subDir>/``，返回 **相对数据根** 的路径。

    与原 TS 一致：120s 超时、**50MB 上限**（先读全量再判长度）、失败抛
    ``Download failed: <status>``。
    """
    directory = Path(get_storage_root()) / sub_dir
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4()}{_ext_from_url(url)}"
    file_path = directory / filename

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.get(url)
    if not response.is_success:
        raise ValueError(f"Download failed: {response.status_code}")

    content = response.content
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"Download exceeds size limit: {len(content)} > {MAX_DOWNLOAD_BYTES}")

    file_path.write_bytes(content)
    return f"static/{sub_dir}/{filename}"


async def read_image_as_compressed_data_url(
    relative_path: str, options: dict[str, Any] | None = None
) -> str:
    """读本地图片为**压缩后**的 data URL（用作生图参考图）。

    ⚠️ **压缩本身未迁**：原实现是 sharp 的
    ``rotate().resize({width:768, height:768, fit:'inside', withoutEnlargement:true})``
    →（有 alpha 则 ``flatten({background:'#ffffff'})``）→ ``jpeg({quality:68, mozjpeg:true})``。
    Python 侧需要 Pillow。当前**退化为原图 data URL**：内容与画质不变，只是载荷更大
    （原本压到 768px / q68 是为了控制体积与厂商限制）。装上 Pillow 后替换本函数体即可。

    ``options`` 接受 ``maxWidth`` / ``maxHeight`` / ``quality``（默认 768/768/68）。
    """
    options = options or {}
    log_task_warn(
        "Storage",
        "reference-not-compressed",
        {
            "path": relative_path,
            "maxWidth": options.get("maxWidth", 768),
            "maxHeight": options.get("maxHeight", 768),
            "quality": options.get("quality", 68),
        },
    )
    return read_image_as_data_url(relative_path)
