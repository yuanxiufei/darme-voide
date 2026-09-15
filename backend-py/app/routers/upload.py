"""``/api/v1/upload`` —— 与 ``backend/src/routes/upload.ts`` 对齐（3 端点，全部迁移）。

图片 / 音频 / 视频素材上传：校验 MIME 类型与体积 → 落盘到数据根的 ``static/<dir>/`` → 返回相对路径。

⚠️ 三个照抄点：

1. **校验顺序是类型 → 体积**（原 TS 如此）。先报类型错，再看大小。
2. **类型判据是客户端给的 Content-Type**（浏览器的 ``file.type``），不是嗅探文件头 —— 
   Python 侧对应 ``UploadFile.content_type``。行为一致（都可能被伪造），但**不要"顺手加固"**：
   改成嗅探会让原本能上传的文件被拒，属于契约变更。
3. **错误文案逐字对齐**，尤其体积那条：``File too large: 1.5MB. Max: 20MB``
   （体积保留 1 位小数，上限是整数 —— 用 Python 直接 f-string 会打出 ``Max: 20.0MB``）。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..core.response import bad_request, success
from ..services.file_storage import save_uploaded_file

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"]

MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_AUDIO_TYPES = [
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac",
    "audio/flac", "audio/ogg", "audio/webm",
]

MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_VIDEO_TYPES = [
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-m4v", "video/x-msvideo", "video/mpeg",
]


def _mb_label(size: int) -> str:
    """``${maxSize / 1024 / 1024}`` —— JS 里整数会输出 ``20`` 而不是 ``20.0``。"""
    value = size / 1024 / 1024
    return str(int(value)) if value == int(value) else str(value)


async def _handle_upload(
    request: Request, allowed_types: list[str], max_size: int, dir_name: str
):
    form = await request.form()
    file = form.get("file")

    # 对齐 `!file || !(file instanceof File)`：缺字段、或字段是纯文本值 → 同一个错误
    if file is None or isinstance(file, str) or not hasattr(file, "read"):
        return bad_request("file is required")

    file_type = getattr(file, "content_type", None) or ""
    if file_type not in allowed_types:
        return bad_request(
            f"Unsupported file type: {file_type}. Allowed: {', '.join(allowed_types)}"
        )

    data = await file.read()
    size = len(data)
    if size > max_size:
        return bad_request(
            f"File too large: {size / 1024 / 1024:.1f}MB. Max: {_mb_label(max_size)}MB"
        )

    saved_path = save_uploaded_file(data, dir_name, getattr(file, "filename", "") or "")
    return success({"url": f"/{saved_path}", "path": saved_path})


@router.post("/image")
async def upload_image(request: Request):
    try:
        return await _handle_upload(request, ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE, "uploads")
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("/audio")
async def upload_audio(request: Request):
    """配乐 / 背景音乐素材。"""
    try:
        return await _handle_upload(request, ALLOWED_AUDIO_TYPES, MAX_AUDIO_SIZE, "audio")
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("/video")
async def upload_video(request: Request):
    """自定义首尾帧视频素材。"""
    try:
        return await _handle_upload(request, ALLOWED_VIDEO_TYPES, MAX_VIDEO_SIZE, "uploads")
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))
