"""merge 域 —— 与 ``backend/src/routes/merge.ts``（43 行）对齐。

**2 个端点全部迁移**：

* ``POST /episodes/{id}/merge`` 触发整集拼接（**立即返回** ``{merge_id, status:'processing'}``，
  真正的拼接在后台跑）；
* ``GET /episodes/{id}/merge`` 查最近一次拼接记录（**没有记录时返回 ``success(null)``**，
  不是 404）。

⚠️ 与 compose 区分：compose 是「单镜合成」（视频+配音+字幕 → 片段），
merge 是「整集拼接」（多个片段 → 成片）。路径同样挂在独立前缀 ``/api/v1/merge`` 下
（Node 是 ``api.route('/merge', merge)``）。

⚠️ ``GET`` 返回的是 **snake_case**（TS 用 ``toSnakeCase``，Python 侧行本身就是蛇形键）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import episodes, video_merges
from ..response import bad_request, not_found, parse_param_id, success
from ..services.ffmpeg_merge import merge_episode_videos
from ..services.task_logger import log_task_error, log_task_start, log_task_success

router = APIRouter(prefix="/api/v1/merge", tags=["merge"])


@router.post("/episodes/{episode_id}/merge")
async def merge_episode(episode_id: str, conn: Connection = Depends(get_tx)):
    """串接整集所有已合成镜头（火忘型：立即返回 merge_id）。"""
    eid = parse_param_id(episode_id)
    if eid is None:
        return not_found("Invalid episode id")
    try:
        episode = conn.execute(
            select(episodes.c.drama_id).where(episodes.c.id == eid)
        ).first()
        if episode is None:
            return bad_request("Episode not found")

        log_task_start("MergeAPI", "episode-merge", {"episodeId": eid, "dramaId": episode[0]})
        # ⚠️ merge_episode_videos 自己管连接：它写 video_merges 记录后就把拼接丢到后台，
        #    借来的连接会随响应关闭（后台任务还得自己开事务）。
        merge_id = merge_episode_videos(eid, episode[0])
        log_task_success("MergeAPI", "episode-merge", {"episodeId": eid, "mergeId": merge_id})
        return success({"merge_id": merge_id, "status": "processing"})
    except Exception as err:  # noqa: BLE001
        log_task_error("MergeAPI", "episode-merge", {"episodeId": episode_id, "error": str(err)})
        return bad_request(str(err))


@router.get("/episodes/{episode_id}/merge")
def get_merge(episode_id: str, conn: Connection = Depends(get_conn)):
    """查最近一次拼接记录（无记录 -> ``success(null)``）。"""
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")
        row = conn.execute(
            select(video_merges)
            .where(video_merges.c.episode_id == eid)
            .order_by(video_merges.c.id.desc())
            .limit(1)
        ).first()
        if row is None:
            return success(None)
        return success({key: row._mapping[key] for key in row._mapping.keys()})
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(err)})
