"""**按 episodeId 取数并组装**体检输入（DB 层 ✓ 与纯逻辑分开 ✓ 2026-09-18）。

## 为什么要单独一个模块

:mod:`.production_preflight` 是**纯函数** ✓（给定 payload 出报告 ✓ 不碰 DB ✓）；
本模块负责**取数** ✓（查表 → 适配器 → 纯函数 ✓）。
这样：**判据与组装各在一处** ✓，且取数错了只影响"喂进去什么" ✓ 不会污染判定 ✓。

## 取数时又发现两个**现成的家**（延续第 89 步的教训 ✓ 先找再盖 ✓）

* **验收状态**：`storyboards.asset_status`（默认 ``missing`` ✓）**早就有** ✓ ——
  不需要新造 ✓；但它**是逐镜的** ✓ ⇒ 这里按"**最早需要该资产的那一镜**"取值 ✓
  并在 ``notes`` 里**明说这个取法** ✓（不偷偷换算 ✗）；
* **逐镜角色（含服装）**：`storyboard_characters` ✓（§7 角色连续性要的正是这个 ✓）。

## 仍然**没有家**的（如实报 ✓ 不糊弄 ✗）

* **资产依赖关系**：manifest 的 `dependencies` 在 schema 里**没有列** ✗ ⇒ 一律给空数组 ✓
  **并报出来** ✓ —— 空数组会被 :func:`.asset_manifest.generation_order` 当成"无依赖"✓，
  于是**顺序信息其实缺失** ✓（覆盖率思路同上 ✓：缺要说出来 ✓）。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.models import (
    asset_versions,
    characters as characters_tbl,
    continuity_states,
    episode_characters,
    episode_props,
    prop_templates,
    scenes as scenes_tbl,
    storyboard_characters,
    storyboard_props,
    storyboards as storyboards_tbl,
)
from . import production_preflight, storyboard_continuity

__all__ = ["ASSET_STATUSES", "MANIFEST_GAPS", "build_manifest", "load_rows",
           "run_episode_preflight"]

#: ``storyboards.asset_status`` 的**词汇表** ✓（原表没规定取值 ✓ ⇒ 这里定 ✓ 并标注是否算"已验收"✓）
ASSET_STATUSES: dict[str, bool] = {
    "missing": False,        # 还没有这个资产（默认值 ✓）
    "pending": False,        # 生成了但没人看过
    "generated": False,      # ⚠️ 与验收门同一条语义：**生成 ≠ 验收** ✗
    "needs_review": False,   # 等人工看
    "approved": True,        # ✅ 只有它算通过
    "rejected": False,       # 被否 ⇒ 要重做
}

#: manifest 里**当前 schema 无处承载**的字段 ✓（目前就一条 ✓ —— 如实报 ✓）
MANIFEST_GAPS: tuple[dict[str, str], ...] = (
    {"field": "dependencies", "problem":
     "资产依赖关系**没有列** ✗ ⇒ 一律给空数组 ⇒ 生成顺序会退化成「无依赖」✓（这是信息缺失 ✓ 不是真的无依赖 ✗）"},
    {"field": "acceptance_criteria", "problem":
     "逐资产的验收标准**没有列** ✗ ⇒ 由前端/提示词侧提供 ✓（本模块不编 ✗）"},
)


def _rows(conn: Connection, statement: Any) -> list[Any]:
    return [dict(row._mapping) for row in conn.execute(statement)]


def load_rows(conn: Connection, episode_id: int) -> dict[str, list[dict[str, Any]]]:
    """取这一集用到的各表行 ✓（逐表 try ✓ —— 某表缺列/缺表时**退回空**并留痕 ✓）。

    ⚠️ 退回空**不是**"没有问题" ✗ —— 它是"没读到" ✓；调用方据 ``notes`` 区分 ✓。
    """
    out: dict[str, list[dict[str, Any]]] = {}
    plan: list[tuple[str, Any]] = [
        ("storyboards", select(storyboards_tbl).where(storyboards_tbl.c.episode_id == episode_id)),
        ("storyboard_characters", select(storyboard_characters)),
        ("storyboard_props", select(storyboard_props)),
        ("continuity_states", select(continuity_states).where(
            continuity_states.c.episode_id == episode_id)),
        ("asset_versions", select(asset_versions)),
    ]
    for name, statement in plan:
        try:
            out[name] = _rows(conn, statement)
        except Exception:  # noqa: BLE001 —— 取数失败只降级 ✓（不让体检端点 500 ✗）
            out[name] = []
            out.setdefault("_errors", []).append(name)

    # 场景 / 角色 / 道具：由本集的分镜**反查** ✓（比按 drama 全量取更准 ✓）
    scene_ids = {row.get("scene_id") for row in out.get("storyboards") or [] if row.get("scene_id")}
    shot_ids = {row.get("id") for row in out.get("storyboards") or [] if row.get("id")}
    for name, statement, key in (
        ("scenes", select(scenes_tbl).where(scenes_tbl.c.id.in_(scene_ids or {0})), "id"),
        ("characters", select(characters_tbl).where(characters_tbl.c.id.in_(
            {row.get("character_id") for row in out.get("storyboard_characters") or []
             if row.get("character_id")} or {0})), "id"),
    ):
        try:
            out[name] = _rows(conn, statement)
        except Exception:  # noqa: BLE001
            out[name] = []
            out.setdefault("_errors", []).append(name)

    prop_ids = {row.get("prop_id") for row in out.get("storyboard_props") or []
                if row.get("prop_id")}
    try:
        out["props"] = _rows(conn, select(prop_templates).where(
            prop_templates.c.id.in_(prop_ids or {0})))
    except Exception:  # noqa: BLE001
        out["props"] = []
        out.setdefault("_errors", []).append("props")

    out.setdefault("_episode_props", [])
    out.setdefault("_shot_ids", sorted(item for item in shot_ids if item))
    return out


def build_manifest(rows: dict[str, list[dict[str, Any]]], episode_id: int) -> tuple[dict, list[str]]:
    """逐镜的 ``asset_status`` → **逐资产**的验收状态 ✓ ⇒ ``(manifest, notes)``。

    ⚠️ ``asset_status`` 是**逐镜**列 ✓ ⇒ 取「**最早需要该资产的那一镜**」的状态 ✓
    （并在 notes 里说明这个取法 ✓ —— 不做隐性换算 ✗）。
    """
    notes: list[str] = []
    shots = rows.get("storyboards") or []
    order = {row.get("id"): index for index, row in enumerate(shots)}

    # 资产 → (类型, 最早需要的镜头, 状态) ✓
    assets: dict[str, dict[str, Any]] = {}

    def touch(asset_id: str, kind: str, shot_id: Any, status: Any) -> None:
        if not asset_id:
            return
        current = assets.get(asset_id)
        rank = order.get(shot_id, len(order))
        if current is None or rank < current["rank"]:
            assets[asset_id] = {"key": asset_id, "type": kind,
                               "first_needed_by": f"sb{shot_id}" if shot_id else "",
                               "rank": rank, "raw": str(status or "missing")}

    for row in shots:
        if row.get("scene_id"):
            touch(f"scene_{row['scene_id']}", "scene", row.get("id"), row.get("asset_status"))
    for row in rows.get("storyboard_characters") or []:
        status = next((item.get("asset_status") for item in shots
                       if item.get("id") == row.get("storyboard_id")), None)
        touch(f"char_{row.get('character_id')}", "character", row.get("storyboard_id"), status)
    for row in rows.get("storyboard_props") or []:
        status = next((item.get("asset_status") for item in shots
                       if item.get("id") == row.get("storyboard_id")), None)
        touch(f"prop_{row.get('prop_id')}", "prop", row.get("storyboard_id"), status)

    known = {row.get("id"): row for row in rows.get("storyboards") or []}
    manifest_assets: list[dict[str, Any]] = []
    unknown_status: set[str] = set()
    for key, item in sorted(assets.items(), key=lambda pair: pair[1]["rank"]):
        raw = item["raw"].strip().lower()
        if raw not in ASSET_STATUSES:
            unknown_status.add(raw)
            status = "missing"          # ⚠️ 认不出 ⇒ 按**最保守**处理 ✓（绝不默认通过 ✗）
        else:
            status = "approved" if ASSET_STATUSES[raw] else raw
        manifest_assets.append({
            "asset_id": key, "type": item["type"], "name": key,
            "first_needed_by": item["first_needed_by"], "dependencies": [],
            "target_path": "", "status": status,
        })

    if unknown_status:
        notes.append(f"⚠️ `asset_status` 里有 {len(unknown_status)} 种取值不在词汇表 "
                     f"{sorted(unknown_status)} ✗ ⇒ 已按**最保守**（missing ✓）处理 ✓ —— "
                     f"默认放行是更坏的选择 ✗")
    notes.append(f"manifest 由**逐镜** `asset_status` 推出 ✓（取该资产**最早需要它的那一镜** ✓）"
                 f"—— 共 {len(manifest_assets)} 个资产 ✓")
    notes.append(f"⚠️ `dependencies` 一律给空 ✓：schema **没有**依赖列 ✗ ⇒ "
                 f"生成顺序会退化成「无依赖」✓（是**信息缺失** ✓ 不是真的无依赖 ✗）")
    approved = sum(1 for item in manifest_assets if item["status"] == "approved")
    notes.append(f"当前**已验收**资产 {approved}/{len(manifest_assets)} ✓"
                 + ("（一个都没有 ⇒ 验收门会拦住 ✓ —— 这是**数据问题** ✓ 不是代码问题 ✓）"
                    if manifest_assets and not approved else ""))
    return {"episode": episode_id, "assets": manifest_assets}, notes


def run_episode_preflight(conn: Connection, episode_id: int) -> dict[str, Any]:
    """⭐ 一集 → **完整体检报告** ✓（取数 → 适配 → 判定 ✓ 全程纯函数负责判 ✓）。"""
    rows = load_rows(conn, episode_id)
    plan, plan_notes = storyboard_continuity.build_plan_from_rows(
        rows.get("storyboards") or [], scenes=rows.get("scenes") or [],
        characters=rows.get("characters") or [], props=rows.get("props") or [],
        continuity_states=rows.get("continuity_states") or [],
        storyboard_props=rows.get("storyboard_props") or [])
    manifest, manifest_notes = build_manifest(rows, episode_id)

    report = production_preflight.run_preflight({"plan": plan, "manifest": manifest}).to_dict()

    # ⚠️⚠️ **没有分镜 ⇒ 必须 `ready=False`** ✗ —— 纯函数那边会"跳过没给的内容" ✓，
    #    于是**不存在的集**会看起来「可开跑」✗✗（没有任何证据却给了绿灯 ✓）。
    #    这是最坏的一类误导 ✓ ⇒ 在这里补一条**阻断** ✓（"无从体检" ≠ "通过" ✓）。
    if not (rows.get("storyboards") or []):
        report["blockers"].append({
            "stage": "source",
            "message": "这一集**没有分镜行** ⇒ 无从体检 ✓（这不等于通过 ✗ —— "
                       "先有分镜，体检才有对象 ✓）"})
        report["ready"] = False
        report["nextActions"] = ["先给这一集生成分镜 ✓（之后体检才有东西可查 ✓）"] \
            + [item for item in report["nextActions"] if "可以开跑" not in item]

    report["source"] = {
        "episodeId": episode_id,
        "counts": {name: len(value) for name, value in rows.items() if isinstance(value, list)},
        "errors": rows.get("_errors") or [],
        "notes": plan_notes + manifest_notes + [
            f"manifest 缺口：{item['field']} —— {item['problem']}" for item in MANIFEST_GAPS],
    }
    return report
