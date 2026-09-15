"""``/api/v1/storage`` —— 数据存储信息与数据根目录切换（**整域 2/2**）。

* ``GET  /info``   —— 当前数据存储信息（只读）
* ``POST /change`` —— 切换数据根目录（默认迁移旧数据，旧目录保留）

⚠️ ``POST /change`` 会写项目级 ``.data-root`` 标记文件，而 **Node 后端启动时也读它** ⇒
绞杀期从 Python 切一次目录会连带挪动 Node 的数据根（跨进程副作用）。它早先被列为
「刻意未迁移」，现已迁移：**Node 下线后单后端必须自己提供它**，否则前端设置页在新架构下
会直接 501（它正是「删 ``backend/``」的最后一块阻塞）。风险与取舍详见
``services/data_storage.py`` 模块头。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services.data_storage import change_data_root, get_storage_info

router = APIRouter(prefix="/api/v1/storage", tags=["storage"])


@router.get("/info")
def storage_info():
    try:
        return success(get_storage_info())
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


@router.post("/change")
async def storage_change(request: Request):
    try:
        body = await read_json(request)
        new_path = body.get("path")
        # 路由层先拦空路径；文案与 `change_data_root` 的兜底文案**不同**（逐字对齐 TS）
        if not new_path or not str(new_path).strip():
            return bad_request("请填写目标目录路径")
        migrate = body.get("migrate", True)
        # `migrate !== false` → 只认字面量 false 才跳过迁移
        return success(change_data_root(str(new_path), migrate is not False))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))
