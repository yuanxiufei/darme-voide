"""视频处理算子路由（``/api/v1/videos/*`` 的**加工**侧）。

与 ``routers/videos.py`` 的分工：
* ``videos.py`` 管**生成**（提交/查询/重生成/删除，走 provider）；
* 本文件管**已存在视频的加工**（目前是插帧 ✓）—— 不碰生成记录，只吃 URL、吐新 URL。

⚠️ 与 ``videos.py`` **同前缀**是故意的 ✓：前端调 ``/api/v1/videos/interpolate`` 一眼知道是同域能力 ✓
（FastAPI 允许多个 router 共前缀 ✓，路径不冲突 ✓）。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services.video_interpolate import interpolate_video

router = APIRouter(prefix="/api/v1/videos", tags=["video-ops"])


@router.post("/interpolate")
async def interpolate(request: Request) -> Any:
    """插帧：``{videoUrl, multiplier=2|3}`` → ``{url, sourceFps, targetFps, elapsedMs}`` ✓。

    **同步等结果**（不是入队）：一次插帧是单条 ffmpeg 命令，前端要的就是"这条视频的新版本"；
    入队反而要再轮询一次、多一张表 ✗。耗时由前端按 ``elapsedMs`` 提示 ✓。
    """
    try:
        body = await read_json(request)
    except Exception as exc:  # noqa: BLE001 —— 请求体不是 JSON
        return bad_request(str(exc))

    video_url = str(body.get("videoUrl") or body.get("video_url") or "").strip()
    if not video_url:
        return bad_request("videoUrl 必填")
    try:
        # 倍率校验在服务里做（一处真源 ✓）—— 这里再校验一遍只会让两处文案慢慢漂开 ✗
        return success(await interpolate_video(video_url, body.get("multiplier", 2)))
    except ValueError as exc:
        return bad_request(str(exc))
    except FileNotFoundError as exc:
        return bad_request(str(exc))
    except Exception as exc:  # noqa: BLE001 —— ffmpeg 失败/超时都当 400 回（消息已带原因）
        return bad_request(str(exc))
