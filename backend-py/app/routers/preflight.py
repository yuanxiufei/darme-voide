"""**开跑前体检**端点 ✓（把四块能力串成**一次调用** ✓）。

``POST /api/v1/production/preflight`` —— 传 ``{promptText, maps, polish, plan, manifest}`` ✓
（全是**各自模块原本的形状** ✓，一个都不新增 ✗），回：

* ``ready`` ✓ —— **唯一**一个开跑判据 ✓；
* ``blockers`` / ``warnings`` ✓ —— 逐条、带 ``stage`` ✓（前端能直接分栏显示 ✓）；
* ``nextActions`` ✓ —— **可执行的修复顺序** ✓（先零成本的文字/结构 ✓ 最后才谈钱 ✓）；
* ``sections`` ✓ —— 四个环节各自的**完整原报告** ✓（想深挖时不必再调一次 ✓）。

⚠️ 本端点**不产生新判据** ✗ —— 它就是 :mod:`app.services.production_preflight` 这段黏合 ✓，
而那段只**调用** :mod:`.shot_placeholders` / :mod:`.prompt_polish` / :mod:`.continuity` /
:mod:`.asset_manifest` ✓。**判据只有一份** ✓，改一处就够 ✓。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services import continuity_store, preflight_source, production_preflight

router = APIRouter(prefix="/api/v1/production", tags=["production"])


#: 体检能认出并处理的段 ✓（**给一个都没有** ⇒ 400 ✓ 见下 ✓）
_SECTIONS: tuple[str, ...] = ("promptText", "polish", "plan", "manifest")


@router.post("/preflight")
async def preflight(request: Request, conn: Connection = Depends(get_conn)) -> Any:
    """开跑前体检 ✓ —— 两种用法 ✓。

    1. **``{"episodeId": 3}``** ✓（推荐 ✓）：后端**自己取数组装** ✓
       （分镜/场景/角色/道具 + ``continuity_states`` ✓ → 适配器 → 判定 ✓）⇒
       前端**一个调用**就能拿到结论 ✓（不必知道 plan/manifest 长什么样 ✓）；
    2. 显式给 ``plan`` / ``manifest`` / ``polish`` / ``promptText`` ✓：给**外部/调试**用 ✓
       （取自 ``_SECTIONS`` ✓）。

    ⚠️ 这里**不做** ``isinstance(body, dict)`` 检查 ✗ —— ``core.request_utils.read_json``
    已经把非法 JSON / 非对象**归一成 ``{}``** 了 ✓（那是有意为之 ✓：对齐 TS，
    让**业务校验**去给出 400 ✓）。所以判据只能写在业务层 ✓：
    **既没给 ``episodeId``、也一段都没给 ⇒ 400** ✓（那是客户端错误 ✓）；
    而服务层 :func:`production_preflight.run_preflight` 仍保持宽容 ✓
    （空输入会**如实说「结果不代表任何」**✓ —— 那是给内部调用方用的 ✓）。
    """
    body = await read_json(request)
    raw_episode = body.get("episodeId")
    if raw_episode not in (None, ""):
        try:
            episode_id = int(raw_episode)
        except (TypeError, ValueError):
            return bad_request(f"episodeId 必须是整数：{raw_episode!r} ✗")
        # ⚠️ 取数失败/缺表 ⇒ **降级 + 如实报** ✓（绝不让体检端点 500 ✗ —— 见 preflight_source ✓）
        return success(preflight_source.run_episode_preflight(conn, episode_id))
    if not any(body.get(key) for key in _SECTIONS):
        return bad_request(f"至少要给一段可体检的内容：{list(_SECTIONS)} ✓，"
                           f"或者给 episodeId 让后端自己组装 ✓"
                           f"（都不给的话，体检结果不代表任何 ✗）")
    return success(production_preflight.run_preflight(body).to_dict())


@router.put("/continuity-states")
async def put_continuity_states(request: Request, conn: Connection = Depends(get_tx)) -> Any:
    """**写入连续性状态** ✓（`continuity_states` ✓ —— 体检链的"喂数据"那一端 ✓）。

    ``{"episodeId": 3, "shots": [{"storyboardId": 11, "states": [
        {"state_type": "prop", "entity_key": "9", "state_value": "hand"},
        {"state_type": "transition", "entity_key": "11", "state_value": "prop"},
        {"state_type": "action", "entity_key": "7", "state_value": "捡起刀",
         "constraints": "[\"plot\"]", "meta": "{\"prop\": \"9\"}"}]}]}``

    ⚠️ 语义（见 :mod:`app.services.continuity_store` ✓）：
    **同一镜是"先删后写"（幂等 ✓）**、**没提到的镜一行不碰** ✓、
    **`state_type` 必须来自词汇表** ✓（不在词汇里 ⇒ **拒收并回显词汇表** ✓ —— 静默忽略会让你
    以为填了却没生效 ✓）、**镜必须属于该集** ✓（否则一条都不写 ✓）。
    """
    body = await read_json(request)
    raw_episode = body.get("episodeId")
    try:
        episode_id = int(raw_episode)
    except (TypeError, ValueError):
        return bad_request(f"episodeId 必须是整数：{raw_episode!r} ✗")
    shots = body.get("shots")
    if not isinstance(shots, list) or not shots:
        return bad_request("缺少 shots（要写哪几个镜的状态）✓ —— "
                           "**只动显式给出的镜** ✓，不做「清空整集」这种事 ✗")
    result = continuity_store.write_episode_states(conn, episode_id=episode_id, shots=shots)
    # ⚠️ 有问题时**仍回 200** ✓（带 problems ✓）—— 因为可能是"部分镜写成了 ✓ 部分没写成 ✓"，
    #    用 4xx 会把"已经落库的部分"一并说成失败 ✗（更难排查 ✓）。
    return success(result)


@router.get("/continuity-states/vocabulary")
def continuity_vocabulary() -> Any:
    """回显 `state_type` **词汇表** ✓ —— 写入方（前端/Agent）照它填 ✓，别自己造词 ✗。"""
    return success({"stateTypes": continuity_store.describe_vocabulary(),
                    "note": "state_type 不在表里 ⇒ **拒收**（不会静默忽略 ✓）"})


@router.get("/preflight/schema")
def preflight_schema() -> Any:
    """体检**接受哪些段**、**按什么顺序**跑 ✓ —— 前端照这个拼请求即可 ✓。"""
    return success({
        "order": list(production_preflight.ORDER),
        "sections": {
            "promptText": "要落位的提示词原文（配 maps ✓）",
            "maps": {"locations": {}, "roles": {}, "props": {}, "clues": {}},
            "polish": "质感层输入（themeTags/characterScene/lens/palette/texture/"
                      "shotType/movement/screenDirection/slice/sounds/imperfections/"
                      "handheld/innerMonologue/restrainedEnding ✓）",
            "plan": {"locations": [], "characters": [], "props": [], "clues": [], "shots": []},
            "manifest": {"episode": 1, "assets": []},
        },
        "notes": [
            "任何一段缺省 ⇒ 跳过该段并在 checked 里如实标出 ✓（不假装检查过 ✗）",
            "质感层的问题只进 warnings ✓（约束不够硬 ⇒ 画得差点，不是接不上 ✓）",
            "连续性/占位符/验收门的问题**阻断** ✓",
            "nextActions 按「先零成本、后花钱」排 ✓",
        ],
    })
