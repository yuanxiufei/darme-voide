"""**资产清单 + 验收门** 端点 ✓（`asset-manifest.json` 形状 ✓ + `blocked_by_missing_asset` ✓）。

* ``POST /api/v1/assets/manifest/validate`` —— 校验清单 ✓：必填字段 / id 唯一 / 依赖悬空 /
  **依赖成环** ✓ / **逐镜被谁卡住** ✓ / **生成顺序** ✓ / ⭐ ``canGenerateVideo`` ✓；
* ``POST /api/v1/assets/gate/check`` —— **执行器调用前**的最后一道 ✓：给一组镜头 ⇒
  回 ``{镜头: [未验收资产]}`` ✓（空 ⇔ 放行 ✓）；
* ``POST /api/v1/assets/acceptance/validate`` —— 校验**验收报告** ✓：判「这一件产物凭什么
  算过」✓（``plan`` 先声明 + ``cases`` 挂证据 + 覆盖率门禁 ✓，见
  ``app/services/acceptance_report.py`` ✓）。

⚠️ 三个端点都**不碰 DB、不调模型** ✓（纯校验 ✓ 毫秒级 ✓）——
它的价值就在「在**花钱之前**说不行」✓（参考项目原话：*不要从被拒或缺失的锚定资产提交付费视频任务* ✓）。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services import acceptance_report as ar
from ..services import asset_manifest as am

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])


@router.post("/manifest/validate")
async def validate(request: Request) -> Any:
    """校验资产清单 ✓ —— ``manifest`` 为清单本体 ✓（形状见 ``asset-to-video-pipeline.md`` ✓）。

    返回 ``canGenerateVideo`` ✓ 与 ``gateReasons`` ✓（门为什么锁着 ✓ 逐条可执行 ✓）、
    ``order``（**生成顺序** ✓ 拓扑排序 ✓）、``cycles``（**环上的资产** ✓ 点名 ✓）、
    ``blockedShots``（每镜被谁卡住 ✓）。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    manifest = body.get("manifest", body)
    return success(am.validate_manifest(manifest).to_dict())


@router.post("/gate/check")
async def gate_check(request: Request) -> Any:
    """**调用前**的门禁 ✓ —— ``{\"manifest\": …, \"shots\": [\"kf003\", …]}`` ⇒ ``{镜头: [资产]}``。

    ⚠️ 与 ``first_needed_by`` 视角互补 ✓：这里只看**这一批**要跑的镜头 ✓，
    是执行器**真正**该调的那一个 ✓（结果是空对象 ⇔ 可以跑 ✓）。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    manifest = body.get("manifest")
    shots = body.get("shots")
    if not isinstance(shots, list) or not shots:
        return bad_request("缺少 shots（要跑哪几个镜头）")
    blocked = am.gate_for_shots(manifest, [str(shot) for shot in shots])
    return success({"blocked": blocked, "allowed": not blocked})


@router.post("/acceptance/validate")
async def acceptance_validate(request: Request) -> Any:
    """校验**验收报告** ✓ —— ``{"report": {...}, "manifest": {...}}`` ✓。

    * ``report.plan`` 为空而给了 ``manifest`` ⇒ 用清单的 ``acceptance_criteria`` 现生成计划 ✓
      （回执 ``planSource="manifest"`` ✓ 不假装是调用方写的 ✓）；
    * 覆盖率门禁会**改判** ✓：声明要 ``audio`` 只交 ``text`` ⇒ 该条 fail ✓（``enforcedStatus`` ✓）；
    * ``verdict`` **现算** ✓：手写 ``summary.verdict`` 与现算不一致 ⇒ problem ✓（绿灯写不死 ✓）。
    """
    body = await read_json(request)
    report = body.get("report")
    if not isinstance(report, dict):
        return bad_request("缺少 report（验收报告本体：plan / cases / summary）")
    manifest = body.get("manifest")
    return success(ar.check_report(report, manifest=manifest).to_dict())
