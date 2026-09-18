"""**连续性表体检**端点 ✓（契约 §6-§11 ✓）。

``POST /api/v1/continuity/check`` —— 传一份**连续性计划** ✓（``{locations, characters, props,
clues, shots}`` ✓），回逐条问题 ✓ + **道具时间线** ✓ + **线索揭示顺序** ✓ + 「可删或可并」的动作 ✓。

⚠️ 与已有的 ``consistency_qc`` **互补而不重叠** ✓：

* ``consistency_qc`` 是**事后**的（对**已生成**的相邻画面算 dHash ✓「拍完才发现不对」✗）；
* 本端点是**事前**的（还没生成就说「这一镜接不上上一镜」✓ ⇒ **省钱** ✓）。

⚠️ **纯计算、不碰 DB、不调模型** ✓（毫秒级 ✓）⇒ 可在分镜编辑时**实时**调 ✓。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services import continuity

router = APIRouter(prefix="/api/v1/continuity", tags=["continuity"])


@router.post("/check")
async def check(request: Request) -> Any:
    """体检连续性 ✓ —— ``{\"plan\": {...}}``（也可直接把计划放在顶层 ✓）。

    返回 ``ok`` ✓（``False`` ⇔ 有阻断级问题 ✓）、``violations`` ✓、``warnings`` ✓、
    ``propTimeline`` ✓、``revealOrder`` ✓、``deletable`` ✓、``counts`` ✓。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    plan = body.get("plan", body)
    if not isinstance(plan, dict) or not plan.get("shots"):
        return bad_request("缺少 plan.shots（要体检哪几个镜头）")
    return success(continuity.check_continuity(plan).to_dict())


@router.get("/rules")
def rules() -> Any:
    """**把判据本身暴露出来** ✓ —— 前端可以直接在界面上写「为什么这镜被拦」✓。

    这也是回答「你们凭什么说它不连续」的唯一诚实方式 ✓：
    规则都在 :mod:`app.services.continuity` 里 ✓，且**逐条照抄**参考契约 §6-§11 ✓。
    """
    return success({
        "tableFields": {name: list(fields) for name, fields in continuity.TABLE_FIELDS.items()},
        "transitionMotives": dict(continuity.TRANSITION_MOTIVES),
        "actionEffects": dict(continuity.ACTION_EFFECTS),
        "notes": [
            "§8 道具像时间线：状态变了就必须有镜头交代谁弄的 ✓",
            "§9 线索不许提前暴露 ✓",
            "§10 五个作用一个都给不出的动作 ⇒ 标为可删或可并 ✓（不阻断 ✓）",
            "§11 每次切都要给动机，不许只写「切到下一镜」✓",
        ],
    })
