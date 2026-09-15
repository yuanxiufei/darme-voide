"""webhooks 域 —— 与 ``backend/src/routes/webhooks.ts``（123 行）对齐。

**1 个端点**：``POST /api/v1/webhooks/vidu`` —— Vidu 在任务完成后回调用它推进结果。

⚠️ 六处保真点：

* ``WEBHOOK_SECRET`` 在 **模块加载时**读一次（原实现是模块级 ``const``），
  配了就必须带 ``x-webhook-secret`` 头，否则 400 ``Unauthorized``；
* **没带 task_id** → 400 ``Missing task_id``；**task_id 查不到记录** → **200**
  ``{"message":"Task not found"}``（故意返回成功，避免厂商重复回调）；
* 成功回调要：下载视频 → **ffprobe 探测时长**（探不到就不写 duration 字段）→
  更新 ``video_generations``（completed）→ 更新 ``storyboards.video_url/duration`` → 触发镜头 QC；
* 下载失败 → 记录 ``failed`` + ``Webhook download failed: …``，返回 400；
* ``state==='failed'`` → 用归因层格式化错误（中文）并记录，返回 **200**
  ``{"message":"Error recorded"}``；
* 其它状态（``processing`` 等）**什么都不做**，返回 200 ``{"message":"Status noted"}``。
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..core.db import get_tx
from ..core.models import storyboards, video_generations
from ..core.request_utils import read_json
from ..core.response import bad_request, now, success
from ..services.file_storage import download_file
from ..services.task_logger import (
    log_task_error,
    log_task_progress,
    log_task_success,
    log_task_warn,
)
from ..services.vendor_errors import format_vendor_task_error
from ..services.video_probe import probe_video_duration
from ..services.qc_scoring import run_qc_after_video_complete

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])

#: ⚠️ 模块加载时读一次（原 TS 是模块级 const，运行期改环境变量**不生效**）
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET") or ""


# ⚠️ 2026-09-15 校正：此处原先是一个**空实现**（``return None`` + 「逻辑未移植 ≈510 行」的说明），
#    但 ``services/qc_scoring.run_qc_after_video_complete``（规则打分 + 技术维度）**早已迁完**，
#    Node 的 ``webhooks.ts`` 也正是直接调 ``runQcAfterVideoComplete`` —— 于是 Python 的 vidu 回调
#    此前**从不打分**，而 Node 会 ⇒ 真行为缺口。现在直接复用服务层实现（薄包装会多一份抄本）。
#    调用点已改为显式传 ``conn``（写回要用同一个连接）。


@router.post("/vidu")
async def vidu_webhook(request: Request, conn: Connection = Depends(get_tx)):
    """Vidu 回调：``{task_id, state, video_url, error}``。"""
    # 配了共享密钥才校验（未配 => 任何人可回调，与原实现一致）
    if WEBHOOK_SECRET:
        signature = request.headers.get("x-webhook-secret")
        if signature != WEBHOOK_SECRET:
            log_task_warn("Webhook", "vidu-auth-failed", {
                "received": f"{signature[:8]}..." if signature else signature,
            })
            return bad_request("Unauthorized")

    body: dict[str, Any] = await read_json(request)
    task_id = body.get("task_id")
    state = body.get("state")
    video_url = body.get("video_url")
    error = body.get("error")

    log_task_progress("Webhook", "vidu-callback", {
        "taskId": task_id, "state": state, "hasVideoUrl": bool(video_url), "error": error,
    })

    if not task_id:
        log_task_warn("Webhook", "vidu-callback-missing-task-id", {"state": state})
        return bad_request("Missing task_id")

    row = conn.execute(
        select(video_generations).where(video_generations.c.task_id == task_id).limit(1)
    ).first()

    if row is None:
        # 可能任务还没写入（极少见），返回成功避免重复回调
        log_task_warn("Webhook", "vidu-task-not-found", {"taskId": task_id})
        return success({"message": "Task not found"})

    if state == "success" and video_url:
        try:
            local_path = await download_file(video_url, "videos")

            # Vidu 回调不返回时长，用 ffprobe 探测本地文件实际时长（供字幕/合成使用）
            probed = await probe_video_duration(local_path)
            duration = probed if probed > 0 else None

            conn.execute(
                update(video_generations).where(video_generations.c.id == row.id).values(
                    video_url=video_url, local_path=local_path, status="completed",
                    completed_at=now(), updated_at=now(),
                )
            )

            if row.storyboard_id:
                values: dict[str, Any] = {"video_url": local_path, "updated_at": now()}
                # `duration` 是 undefined 时 JSON.stringify 会丢键 ⇒ 这里也不写该列
                if duration is not None:
                    values["duration"] = duration
                conn.execute(
                    update(storyboards).where(storyboards.c.id == row.storyboard_id).values(**values)
                )
                # 触发镜头级 QC 打分（fire-and-forget；规则分 + 技术维度，见 qc_scoring）
                run_qc_after_video_complete(conn, row.storyboard_id, row.id)

            log_task_success("Webhook", "vidu-video-updated", {
                "taskId": task_id, "generationId": row.id,
                "storyboardId": row.storyboard_id, "localPath": local_path, "duration": duration,
            })
            return success({"message": "Video updated successfully"})
        except Exception as err:  # noqa: BLE001
            log_task_error("Webhook", "vidu-download-failed", {
                "taskId": task_id, "generationId": row.id, "error": str(err),
            })
            conn.execute(
                update(video_generations).where(video_generations.c.id == row.id).values(
                    status="failed", error_msg=f"Webhook download failed: {err}"
                )
            )
            return bad_request(str(err))

    if state == "failed":
        message = format_vendor_task_error(error, "video")
        log_task_error("Webhook", "vidu-generation-failed", {
            "taskId": task_id, "generationId": row.id, "error": message,
        })
        conn.execute(
            update(video_generations).where(video_generations.c.id == row.id).values(
                status="failed", error_msg=message
            )
        )
        return success({"message": "Error recorded"})

    # 其他状态（processing 等），不处理
    log_task_progress("Webhook", "vidu-status-noted", {
        "taskId": task_id, "generationId": row.id, "state": state,
    })
    return success({"message": "Status noted"})
