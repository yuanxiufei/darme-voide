"""``/api/v1/storage`` —— 数据存储信息。

**只迁移了 ``GET /info``**（只读）。``POST /change``（切换数据根目录）**刻意不注册**：

* 它写项目级 ``.data-root`` 标记文件，而 **Node 启动时也读这个文件** ⇒ 从 Python 切一次
  目录会连带把 Node 的数据根一起挪走，是跨进程副作用；
* 还要关库重开，叠加「两进程并发 + 标记文件共享」，在绞杀期收益极低风险不小。

⇒ 等 Node 完全下线（单后端）再迁。详见 ``services/data_storage.py`` 模块头。
"""

from __future__ import annotations

from fastapi import APIRouter

from ..response import success
from ..services.data_storage import get_storage_info

router = APIRouter(prefix="/api/v1/storage", tags=["storage"])


@router.get("/info")
def storage_info():
    try:
        return success(get_storage_info())
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})
