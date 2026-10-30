"""**运行时体检**端点（把「服务活着没有」摆到开跑之前 ✓ 2026-10-30）。

与另外两个 ``ready`` **各查一件事**，别混 ✓：

* ``POST /api/v1/production/preflight`` —— 查**内容/结构**（提示词落位、连续性、资产门 ✓）
* ``GET  /api/v1/production/engine-readiness`` —— 查**权重就绪**（盘上有没有、够不够显存 ✓）
* ``GET  /api/v1/production/runtime-doctor`` —— 查**服务连通**（本机四类进程起没起 ✓）← 本模块

判据只有一份：:mod:`app.services.runtime_doctor` 的 ``run_doctor`` ✓，
本端点**不新增任何判定** ✗，只是把它连 JSON 化 ✓。
"""
from __future__ import annotations

from fastapi import APIRouter

from ..core.response import success
from ..services import runtime_doctor

router = APIRouter(prefix="/api/v1/production", tags=["production"])


@router.get("/runtime-doctor")
def runtime_doctor_endpoint(probe: bool = True) -> object:
    """四类 AI 服务体检 ⇒ ``{ready, services[], blockers[], warnings[], nextActions[]}``。

    ``probe=false`` 只查配置（不探活、毫秒级 ✓）；默认探活（≤3 秒 ✓）。
    有阻断时 ``ready=false``，但 HTTP 仍 200 —— **这是报告不是错误** ✓
    （前端要拿 ``nextActions`` 显示修复指引 ✓）。
    """
    return success(runtime_doctor.run_doctor(probe=probe).to_dict())
