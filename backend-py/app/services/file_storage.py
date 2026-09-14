"""文件存储工具 —— 移植 ``utils/storage.ts``。

已迁：落盘上传文件、取安全绝对路径、base64 存取、**下载远程文件**、**压缩参考图**。
``getAbsolutePath`` 的路径穿越防护是安全关键，必须逐行对齐。
"""

from __future__ import annotations

import asyncio
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


#: 含 alpha 的常见像素格式（决定是否需要 flatten 白底）
_ALPHA_PIX_FMTS = ("rgba", "argb", "abgr", "bgra", "yuva", "ya8", "pa")


def quality_to_qscale(quality: Any) -> int:
    """``quality``(0~100，越大越好) → ffmpeg mjpeg 的 ``-q:v``(2~31，**越小越好**)。

    ⚠️ 两侧**编码器不同**（ffmpeg mjpeg vs sharp 的 mozjpeg）⇒ **字节不可能对齐**：
    实测 1024×768 源图，sharp ``q68 mozjpeg`` = **17399 B**，而 ffmpeg 即便 ``-q:v 12``
    仍有 25713 B（同视觉质量下 ffmpeg 体积约大 1.4~1.7×，mozjpeg 本就是为压缩率优化的编码器）。
    故这里采用 **qscale ↔ quality 的线性近似**（``q68 → -q:v 10``，业界常用口径），
    优先保**视觉质量一致**；若调用方更在意载荷体积，传更小的 ``quality`` 即可。
    """
    try:
        value = float(quality)
    except (TypeError, ValueError):
        value = 68.0
    return max(2, min(31, int(31 - value * 0.31 + 0.5)))  # +0.5：JS Math.round 语义


async def _probe_image_pix_fmt(absolute: str) -> str:
    """ffprobe 取像素格式（失败抛 RuntimeError —— 与原 TS 的 sharp 抛错同形）。"""
    process = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=pix_fmt", "-of", "csv=p=0", absolute,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError("读取图片失败（ffprobe 非零退出）："
                           + (stderr.decode("utf-8", errors="replace").strip()[:200] or "无输出"))
    return stdout.decode("utf-8", errors="replace").strip()


def _build_scale_chain(pix_fmt: str, max_width: Any, max_height: Any) -> str:
    """拼滤镜链：等比缩到「不放大」的框内 →（有 alpha）白底 flatten → 偶数尺寸。

    * ``fit:'inside' + withoutEnlargement`` 的 ffmpeg 等价物 =
      ``scale=min(W,iw):min(H,ih):force_original_aspect_ratio=decrease``
      （``min`` 把目标框夹到输入尺寸 ⇒ 小图**不会被放大** ✓）；
    * ``<= 0`` 视为「该维不约束」（sharp 对 ``width:0`` 也是这样）；
    * mjpeg 的 4:2:0 要求**偶数**尺寸 ⇒ 末尾 ``trunc(iw/2)*2``（与 sharp 最多差 1px，已记录）。
    """
    width_expr = f"min({int(max_width)}\\,iw)" if isinstance(max_width, (int, float)) \
        and max_width > 0 else "iw"
    height_expr = f"min({int(max_height)}\\,ih)" if isinstance(max_height, (int, float)) \
        and max_height > 0 else "ih"

    chain = [f"scale={width_expr}:{height_expr}:"
             f"force_original_aspect_ratio=decrease:flags=lanczos"]
    has_alpha = any(token in pix_fmt for token in _ALPHA_PIX_FMTS)
    if has_alpha:
        # flatten 到白底：r' = r*a/255 + 255*(1-a/255)（非预乘 RGBA ⇒ 直接线性合成）
        # ⚠️ geq 的 alpha 取用函数名是 **`alpha(x,y)`**，不是 `a(x,y)`（实测 `a(...)` 会报
        #    `Unknown function`；输出平面**选项**才叫 `a`）
        alpha = "alpha(X\\,Y)"
        white = f"255*(1-{alpha}/255)"
        chain.append("format=rgba")
        chain.append("geq=" + ":".join(
            f"{channel}='{channel}(X\\,Y)*{alpha}/255+{white}'"
            for channel in ("r", "g", "b")) + ":a='255'")
    chain.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    return ",".join(chain)


async def read_image_as_compressed_data_url(
    relative_path: str, options: dict[str, Any] | None = None
) -> str:
    """读本地图片为**压缩后**的 data URL（用作生图参考图）。

    对齐原实现的语义：等比缩到长边 ≤ ``maxWidth``×``maxHeight``（**不放大**）→
    有 alpha 则 **flatten 白底** → **JPEG** 编码 → ``data:image/jpeg;base64,…``。

    ``options``：``maxWidth`` / ``maxHeight`` / ``quality``（默认 768/768/68；
    **nullish 语义** —— 显式传 0 会被保留，此时该维不约束）。
    """
    options = options or {}
    max_width = 768 if options.get("maxWidth") is None else options["maxWidth"]
    max_height = 768 if options.get("maxHeight") is None else options["maxHeight"]
    quality = 68 if options.get("quality") is None else options["quality"]

    absolute = get_absolute_path(relative_path)
    pix_fmt = await _probe_image_pix_fmt(absolute)

    temp = f"{absolute}.ref.jpg"
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", "-i", absolute,
        "-vf", _build_scale_chain(pix_fmt, max_width, max_height),
        "-frames:v", "1", "-q:v", str(quality_to_qscale(quality)), "-y", temp,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _stdout, stderr = await process.communicate()
    if process.returncode != 0 or not os.path.exists(temp):
        if os.path.exists(temp):
            os.remove(temp)
        raise RuntimeError("压缩参考图失败（ffmpeg 非零退出）："
                           + (stderr.decode("utf-8", errors="replace").strip()[:200] or "无输出"))

    try:
        encoded = base64.b64encode(Path(temp).read_bytes()).decode("ascii")
    finally:
        os.remove(temp)
    return f"data:image/jpeg;base64,{encoded}"
