"""S7 自检：**按 episodeId 取数并组装**体检输入（2026-09-18）。

这一步是把"事前省钱链"接到**真实数据**上 ✓：`{episodeId}` → 取数 → 适配器 → 判定 ✓。
本套钉四条**容易糊过去**的语义 ✓：

1. ⭐⭐ **取数失败/没有分镜 ⇒ `ready=False`** ✗✓ —— 纯函数那边"没给的内容就跳过"✓，
   于是**不存在的集**会看起来「可开跑」✗（没有任何证据却给绿灯 ✓，最坏的一类误导 ✓）；
2. ⭐ **`generated` 不算验收通过** ✗（与验收门同一条语义 ✓ —— 同一个仓库里
   两处对同一个词给相反口径就完了 ✓）；
3. ⭐⭐ **认不出的 `asset_status` ⇒ 按最保守处理**（当 missing ✓）并**报出来** ✓
   —— 默认放行是更坏的选择 ✗；
4. ⭐ **`dependencies` 一律空要明说** ✓ —— 空数组会被当成"无依赖"✓，
   而真相是"schema 没这一列"✓（信息缺失 ✓ 不是真无依赖 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/preflight_source_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.preflight_source import (  # noqa: E402
    ASSET_STATUSES, MANIFEST_GAPS, build_manifest, run_episode_preflight)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


ROWS: dict = {
    "storyboards": [
        {"id": 11, "scene_id": 5, "asset_status": "approved"},
        {"id": 12, "scene_id": 5, "asset_status": "approved"},
    ],
    "storyboard_characters": [{"storyboard_id": 11, "character_id": 7, "costume": "湿外套"},
                              {"storyboard_id": 12, "character_id": 7, "costume": "湿外套"}],
    "storyboard_props": [{"storyboard_id": 12, "prop_id": 9}],
}


# ══════════════════════════════════════════════════════════════════════════
# ① 逐镜 asset_status → 逐资产状态
# ══════════════════════════════════════════════════════════════════════════
def case_manifest() -> None:
    manifest, notes = build_manifest(ROWS, 3)
    assets = {item["asset_id"]: item for item in manifest["assets"]}
    check("① 从镜头/角色/道具推出资产 ✓（类型对 ✓）",
          assets.get("scene_5", {}).get("type") == "scene"
          and assets.get("char_7", {}).get("type") == "character"
          and assets.get("prop_9", {}).get("type") == "prop", list(assets))
    check("② ⭐ `first_needed_by` 取**最早需要它的那一镜** ✓"
          "（`asset_status` 是逐镜列 ✓ ⇒ 取法必须写明 ✓ 不做隐性换算 ✗）",
          assets["char_7"]["first_needed_by"] == "sb11"
          and assets["prop_9"]["first_needed_by"] == "sb12",
          {key: item["first_needed_by"] for key, item in assets.items()})
    check("③ `approved` ⇒ 状态就是 approved ✓", assets["char_7"]["status"] == "approved", "")
    check("④ ⭐⭐ **`generated` 不算验收通过** ✗（与验收门同一条语义 ✓ —— "
          "同一个仓库对同一个词不能有两种口径 ✗）",
          build_manifest({"storyboards": [{"id": 11, "scene_id": 5,
                                           "asset_status": "generated"}]}, 3)[0]["assets"][0][
              "status"] == "generated", "")

    weird, weird_notes = build_manifest({"storyboards": [
        {"id": 11, "scene_id": 5, "asset_status": "看过还行"}]}, 3)
    check("⑤ ⭐⭐ 认不出的取值 ⇒ **按最保守处理**（当 missing ✓）**并报出来** ✓✓"
          "（默认放行是更坏的选择 ✗）",
          weird["assets"][0]["status"] == "missing"
          and any("不在词汇表" in item for item in weird_notes), weird_notes)

    check("⑥ 词汇表：**只有 `approved` 算通过** ✓",
          sum(1 for passed in ASSET_STATUSES.values() if passed) == 1
          and ASSET_STATUSES.get("approved") is True, ASSET_STATUSES)

    _manifest, notes = build_manifest(ROWS, 3)
    check("⑦ ⭐⭐ **`dependencies` 一律空要明说** ✓✓（空数组会被当成「无依赖」✓，"
          "而真相是「schema 没这一列」✗ ⇒ 必须报 ✓）",
          all(item["dependencies"] == [] for item in _manifest["assets"])
          and any("dependencies" in item and "信息缺失" in item for item in notes), notes)
    check("⑧ 仍无承载的 manifest 字段逐条列出 ✓（依赖 / 验收标准 ✓）",
          {item["field"] for item in MANIFEST_GAPS} == {"dependencies", "acceptance_criteria"},
          MANIFEST_GAPS)
    # ⚠️ 我的夹具全是 approved ⇒ 这条得**换个夹具** ✓（初版用同一份 ⇒ 期望落空 ✓，
    #    是实现对、检查写错 ✓）
    _m2, pending_notes = build_manifest({"storyboards": [
        {"id": 11, "scene_id": 5, "asset_status": "pending"}]}, 3)
    check("⑨ 全未验收时会**明说这是数据问题** ✓（不是代码问题 ✓ —— 免得去瞎改代码 ✗）",
          any("数据问题" in item for item in pending_notes), pending_notes)


# ══════════════════════════════════════════════════════════════════════════
# ② ⭐⭐ 没有分镜 ⇒ 不能给绿灯（真 DB、不存在的集）
# ══════════════════════════════════════════════════════════════════════════
def case_empty_episode() -> None:
    from app.core.db import engine

    with engine.connect() as conn:
        report = run_episode_preflight(conn, 999_999)

    check("⑩ ⭐⭐ **不存在的集 ⇒ `ready=False`** ✓✓（纯函数会「跳过没给的内容」✓，"
          "于是会看起来可开跑 ✗ —— 没有任何证据却给绿灯 ✓ 是最坏的一类误导 ✓）",
          report.get("ready") is False, report.get("ready"))
    check("⑪ 阻断项**说清为什么** ✓（没有分镜行 ⇒ 无从体检 ✓ 且明说「不等于通过」✓）",
          any(item.get("stage") == "source" and "不等于通过" in str(item.get("message"))
              for item in report.get("blockers") or []), report.get("blockers"))
    check("⑫ 行动项换成**可执行的第一步** ✓（先出分镜 ✓），并**撤掉**「可以开跑」✓",
          any("生成分镜" in item for item in report.get("nextActions") or [])
          and not any("可以开跑" in item for item in report.get("nextActions") or []),
          report.get("nextActions"))

    with engine.connect() as conn:
        report2 = run_episode_preflight(conn, 999_999)
    source = report2.get("source") or {}
    check("⑬ 报告里带**取数痕迹** ✓（各表读了几行 ✓ / 读不到的表 ✓ —— "
          "便于定位是「没数据」还是「没读到」✓）",
          isinstance(source.get("counts"), dict) and "episodeId" in source
          and "notes" in source, list(source))


# ══════════════════════════════════════════════════════════════════════════
# ③ 路由：`{episodeId}` 是**一个调用**就够
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post("/api/v1/production/preflight", json={"episodeId": 999_999})
    data = response.json().get("data") or {}
    check("⑭ POST `{episodeId}` ⇒ 200 ✓ 且**后端自己组装**（响应里带 `source` ✓）",
          response.status_code == 200 and data.get("source", {}).get("episodeId") == 999_999,
          (response.status_code, list(data)))
    check("⑮ 同上：不存在的集 ⇒ `ready=False` 经 API 也成立 ✓（不因为「没数据」而放行 ✗）",
          data.get("ready") is False, data.get("ready"))

    bad = client.post("/api/v1/production/preflight", json={"episodeId": "第三集"})
    check("⑯ `episodeId` 非整数 ⇒ **400** ✓（不 500 ✗、也不静默当成 0 ✗）",
          bad.status_code == 400, bad.status_code)

    check("⑰ 既没给 episodeId 也没给任何段 ⇒ 400 ✓（且提示里有 episodeId 这条出路 ✓）",
          client.post("/api/v1/production/preflight", json={}).status_code == 400
          and "episodeId" in client.post("/api/v1/production/preflight",
                                         json={}).json().get("message", ""), "")


def main() -> int:
    case_manifest()
    case_empty_episode()
    case_api()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
