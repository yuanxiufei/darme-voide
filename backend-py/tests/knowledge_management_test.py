"""知识库管理面守卫（2026-11-07 立）—— 编辑/删除/版本/标签/分类/审计/HTTP 契约。

为什么要有这一套
================
台账从「只能 upsert + 检索」升级成「可管理」之后，新长出来的失效方式**全都不报错**：
编辑被做成 upsert ⇒ support 被当编辑次数用（改三遍错别字看着像三份独立证据）；
乐观锁没接上 ⇒ 后保存的正文静默覆盖前一个人的整段；``detail`` 忘了过滤软删 ⇒
``GET /entries/{id}`` 成了绕过软删的后门；硬删没有前置闸 ⇒ 打错一个参数就把
被引用几十次的成法连行用历史一起蒸发；标签用 ``LIKE %节奏%`` ⇒ 点「节奏」冒出
「节奏慢」；``search.total`` 回本页条数 ⇒ 分页永远只有一页。

八组判据
========
① 派生字段；② 编辑（support 不动/version+1/快照/FTS/乐观锁）；③ 版本与回滚；
④ 软删与恢复；⑤ 硬删两道路闸；⑥ 标签词边界与分类；⑦ 审计；⑧ HTTP 错误码 + 文档双向对账。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/knowledge_management_test.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend-py"))
os.environ.setdefault("PROXY_TO_NODE", "0")

#: 夹具库与真实库**必须分开**：这套含硬删，跑在真数据根上就是在删创作者的知识。
TMP = Path(tempfile.mkdtemp(prefix="kb-manage-"))
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome-index.sqlite").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "knowledge-ledger.sqlite").as_posix()

from app.services import knowledge_ledger, outcome_index  # noqa: E402

#: 证据核验会拼一次数据根路径，换成夹具目录（防真数据根上的同名文件让假证据蒙过去）
knowledge_ledger.get_data_root = lambda: TMP.as_posix()

DOC = REPO / "docs" / "knowledge-base-contract.md"
BASE = "/api/v1/knowledge-loop"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:
    #: ★证据 id 必须与 ``record_run`` 写进去的 ``run_id`` **逐字相同** ——
    #: ``check_evidence`` 拿原串查索引，不接受 ``run:`` 之类的自造前缀。
    outcome_index.record_run({"run_id": "trace:kb/a", "kind": "image", "status": "ok"})
    outcome_index.record_run({"run_id": "trace:kb/b", "kind": "agent", "status": "failed"})
    ev, ev2 = ["trace:kb/a"], ["trace:kb/b"]
    L = knowledge_ledger

    # ================================================================ ① 创建与派生
    made = L.upsert(kind="lesson", scope="EP7", title="过曝要先降曝光再压高光",
                    body="把曝光降 0.3 再压高光，比直接拉曲线稳。", evidence=ev,
                    tags="节奏 ，过曝、节奏; 光影", confidence=0.4,
                    operator="tester", role="editor")
    eid = made["id"]
    row = L.detail(eid)
    check("① 创建 created=True、support=1、version=1",
          made["ok"] and made["created"] and made["support"] == 1 and row["version"] == 1, made)
    check("① ★标签写入口归一（全角逗号/顿号/分号都当分隔、去重保序）",
          row["tags"] == "节奏,过曝,光影", row["tags"])
    check("① 派生 tagList/evidence/deleted 由服务端算",
          row["tagList"] == ["节奏", "过曝", "光影"] and row["evidence"] == ["trace:kb/a"]
          and row["deleted"] is False and row["deletedNs"] == 0, row["tagList"])
    parts = L.normalize_tags(["长" * (L.MAX_TAG_LEN + 5), *[f"t{i}" for i in range(L.MAX_TAGS + 4)]]).split(",")
    check("① 标签超长/超量只截断不报错（单标签 24、每条 12 个）",
          len(parts[0]) == L.MAX_TAG_LEN and len(parts) == L.MAX_TAGS, (len(parts[0]), len(parts)))
    check("① ★读入口不归一（只有写入口归一）：手工塞的原样拆",
          L.tag_list(" a , b ") == ["a", "b"])

    # ================================================================ ② 编辑
    edited = L.update_entry(eid, body="把曝光降 0.5 再压高光更稳。", reason="补充数值",
                            operator="alice", role="editor")
    check("② 编辑 version=2、changed 只含实际改的字段",
          edited["ok"] and edited["version"] == 2 and edited["changed"] == ["body"], edited.get("changed"))
    fresh = L.detail(eid)
    check("② ★编辑不动 support（改错 ≠ 又一份证据）",
          fresh["support"] == 1 and "0.5" in fresh["body"], fresh["support"])
    conflict = L.update_entry(eid, body="基于旧版本的覆盖", expected_version=1)
    check("② ★乐观锁：expectedVersion 落后 ⇒ conflict + 带回当前版本，且正文未被覆盖",
          conflict["ok"] is False and conflict.get("conflict") is True
          and conflict.get("version") == 2 and conflict.get("code") == "conflict"
          and "基于旧版本" not in L.detail(eid)["body"], conflict)
    bump = L.update_entry(eid, title="过曝要先降曝光再压高光（改）", expected_version=2)
    check("② expectedVersion 对上就放行、version=3", bump["ok"] and bump["version"] == 3)
    for label, got, code in (
        ("空标题", L.update_entry(eid, title="   "), "invalid"),
        ("空正文", L.update_entry(eid, body=""), "invalid"),
        ("没给字段", L.update_entry(eid), "invalid"),
        ("假证据", L.update_entry(eid, evidence=["run:made-up-999"]), "evidence"),
        ("不存在的条目", L.update_entry("lesson-nope", body="x"), "not_found"),
    ):
        check(f"② 拒改：{label} ⇒ ok=False + code={code}（不静默成功）",
              got["ok"] is False and got.get("code") == code, got)
    check("② ★编辑后 FTS 同步：旧正文搜不到",
          not L.search("直接拉曲线")["items"])
    check("② ★编辑后 FTS 同步：新正文搜得到", bool(L.search("再压高光更稳")["items"]))

    # ================================================================ ③ 版本
    vers = L.versions(eid)
    check("③ 历史只到 current-1（当前值留在 entries 里）",
          vers["current"] == 3 and [v["version"] for v in vers["items"]] == [2, 1],
          [v["version"] for v in vers["items"]])
    check("③ ★版本列表不含正文（只给 bodyLen）",
          all("body" not in v and v["bodyLen"] > 0 for v in vers["items"]))
    #: 注意：``items[0]`` 是 v2 的快照（那次没带 operator/reason），带理由的是 v1 那条快照 ——
    #: 快照存的是「被替换掉的旧值」，谁替换的记在那条快照上，别按版本号倒着猜。
    check("③ 快照记了 editor/reason（谁、在什么理由下改的）",
          any(v["editor"] == "alice" and "补充数值" in v["reason"] for v in vers["items"]),
          [(v["version"], v["editor"], v["reason"]) for v in vers["items"]])
    v1, v3 = L.version_detail(eid, 1), L.version_detail(eid, 3)
    check("③ 历史版本可读正文、isCurrent=False；当前版 isCurrent=True 且与详情一致",
          v1["isCurrent"] is False and "0.3" in v1["body"] and v3["isCurrent"] is True
          and v3["body"] == L.detail(eid)["body"])
    #: ★同端点两个来源必须同形状：当前版来自 ``entries``（``_row_to_entry``），历史版来自
    #: ``entry_versions``。早先前者回 snake 列名、后者回原始行 ⇒ 前端只有一条路能跑通，
    #: 另一条静默拿到 undefined（不报错、只是空白），所以这里按**键集**判。
    keys = {"id", "version", "kind", "scope", "title", "body", "bodyLen", "tags",
            "evidence", "confidence", "status", "editor", "reason", "createdNs", "isCurrent"}
    check("③ ★版本详情的当前版与历史版逐键同形状（两套来源收成一套）",
          set(v1) == keys and set(v3) == keys,
          [sorted(set(v1) ^ keys), sorted(set(v3) ^ keys)])
    check("③ ★两版的 tags/evidence 都是数组（不是逗号串），createdNs 都是纳秒整数",
          isinstance(v1["tags"], list) and isinstance(v3["tags"], list)
          and isinstance(v1["evidence"], list) and isinstance(v3["evidence"], list)
          and "created_ns" not in v1 and "created_ns" not in v3)
    check("③ 不存在的版本号 ⇒ None（接口 404，不是空对象）", L.version_detail(eid, 99) is None)
    rolled = L.revert(eid, 1, reason="回滚", operator="alice", role="editor")
    check("③ ★回滚是一次普通编辑：version=4、正文回到 v1",
          rolled["ok"] and rolled["version"] == 4 and "0.3" in L.detail(eid)["body"])
    check("③ ★回滚不删后续版本（v2/v3 还在 ⇒ 还能再滚回去）",
          [v["version"] for v in L.versions(eid)["items"]] == [3, 2, 1],
          [v["version"] for v in L.versions(eid)["items"]])
    check("③ 回滚到当前版本 ⇒ changed=False（幂等）", L.revert(eid, 4)["changed"] is False)
    check("③ 回滚不存在的版本 ⇒ code=not_found", L.revert(eid, 42).get("code") == "not_found")

    # ================================================================ ④ 软删/恢复
    gone = L.soft_delete(eid, reason="写错了", operator="alice", role="editor")
    check("④ 软删 changed=True、hard=False", gone["ok"] and gone["changed"] and gone["hard"] is False)
    check("④ ★软删后：列表看不见、检索看不见、详情默认 None（详情口不是后门）",
          not any(e["id"] == eid for e in L.list_entries(limit=100)["items"])
          and not any(e["id"] == eid for e in L.search("过曝", limit=50)["items"])
          and L.detail(eid) is None)
    recycle = L.detail(eid, include_deleted=True)
    check("④ 回收站显式口径看得见且带 deleted/deletedBy",
          recycle["deleted"] is True and recycle["deletedBy"] == "alice")
    check("④ 回收站列表（include_deleted）能列出来",
          any(e["id"] == eid for e in L.list_entries(include_deleted=True, limit=100)["items"]))
    check("④ 重复软删幂等不报错", L.soft_delete(eid)["changed"] is False)
    check("④ 已删条目不能编辑/迁移状态 ⇒ code=deleted",
          L.update_entry(eid, body="改已删的").get("code") == "deleted"
          and L.set_status(eid, "accepted").get("code") == "deleted")
    #: ★id 由 ``(kind, scope, **创建时**标题)`` 决定，改名不改 id —— 所以这里必须用**原标题**
    #: 才能命中同一条；用改名后的标题会算出另一个 id、正常新建一条（这也是已知边界，
    #: 写在契约文档「边界条件」里：改名后同一主题再蒸馏会变成两条，需人工合并）。
    revived = L.upsert(kind="lesson", scope="EP7", title="过曝要先降曝光再压高光",
                       body="蒸馏又提炼了一遍", evidence=ev)
    check("④ ★upsert 不静默复活已删条目（恢复是人的动作，带审计）",
          revived["ok"] is False and revived.get("code") == "deleted", revived)
    check("④ 恢复成功且恢复后可见、重复恢复幂等",
          L.restore(eid, reason="误删", operator="bob")["changed"] is True
          and L.detail(eid) is not None and L.restore(eid)["changed"] is False)
    L.record_usage(eid, "trace:agent/x", "ok")
    L.soft_delete(eid, operator="bob")

    # ================================================================ ⑤ 硬删
    keep = L.upsert(kind="lesson", scope="EP7", title="先软删才允许硬删",
                    body="这道闸是防手滑的。", evidence=ev)
    early = L.purge(keep["id"])
    check("⑤ ★未软删就硬删 ⇒ code=not_deleted（第二道闸），且什么都没删",
          early["ok"] is False and early.get("code") == "not_deleted"
          and L.detail(keep["id"]) is not None, early)
    audit_before = L.audit_log(entry=eid)["total"]
    purged = L.purge(eid, operator="bob", role="owner")
    check("⑤ 已软删 ⇒ 硬删成功、逐表计数如实回",
          purged["ok"] and purged["counts"]["entry_versions"] >= 1, purged.get("counts"))
    check("⑤ 硬删后：详情 None、检索不到、版本读不到",
          L.detail(eid, include_deleted=True) is None
          and not L.search("压高光", limit=50)["items"] and L.versions(eid)["current"] == 0)
    audit_after = L.audit_log(entry=eid)
    check("⑤ ★审计不随硬删删除（账不是内容）：purge 后审计更多且最新是 purge",
          audit_after["total"] > audit_before and audit_after["items"][0]["action"] == "purge",
          audit_after["items"][:1])
    check("⑤ purge 的审计 detail 记了删前各表条数（可回溯删了什么）",
          "entry_versions=" in audit_after["items"][0]["detail"])

    # ================================================================ ⑥ 标签/分类
    a = L.upsert(kind="recipe", scope="global", title="节奏快的剪辑要短镜",
                 body="每镜不超过三秒。", evidence=ev, tags="节奏,剪辑")["id"]
    b = L.upsert(kind="recipe", scope="global", title="节奏慢的剪辑要长镜",
                 body="让情绪有呼吸。", evidence=ev2, tags="节奏慢,剪辑")["id"]
    c = L.upsert(kind="fact", scope="global", title="相机宽容度常识",
                 body="宽容度高的机器更耐调。", tags="节奏,宽容度")["id"]
    hit = L.list_entries(tags="节奏", limit=50)
    check("⑥ ★标签走词边界：只回「节奏」，不回「节奏慢」",
          {e["id"] for e in hit["items"]} == {a, c}, [e["title"] for e in hit["items"]])
    check("⑥ ★多标签是 AND（同时具备）",
          [e["id"] for e in L.list_entries(tags="节奏,剪辑", limit=50)["items"]] == [a])
    facets = L.tag_facets()
    counts = {i["tag"]: i["count"] for i in facets["items"]}
    check("⑥ facet 计数与列表 total 同口径（点进去条数对得上）",
          counts.get("节奏") == L.list_entries(tags="节奏")["total"]
          and counts.get("剪辑") == L.list_entries(tags="剪辑")["total"], counts)
    check("⑥ facet 按条数倒序（顺序稳定）",
          [i["count"] for i in facets["items"]] == sorted((i["count"] for i in facets["items"]),
                                                          reverse=True))
    L.soft_delete(b, operator="carol")
    check("⑥ 软删条目不计入 facet；显式 includeDeleted 才计入",
          {i["tag"]: i["count"] for i in L.tag_facets()["items"]}.get("节奏慢") is None
          and {i["tag"]: i["count"] for i in L.tag_facets(
              include_deleted=True)["items"]}.get("节奏慢") == 1)
    tax = L.taxonomy()
    check("⑥ ★taxonomy.kinds 覆盖全部 kind 且每类带中文 label（前端不写死）",
          [k["kind"] for k in tax["kinds"]] == list(L.KINDS)
          and all(k["label"] and k["label"] != k["kind"] for k in tax["kinds"]))
    check("⑥ taxonomy 的 kind 计数与列表 total 对得上",
          {k["kind"]: k["count"] for k in tax["kinds"]}["recipe"]
          == L.list_entries(kind="recipe")["total"])
    check("⑥ taxonomy 带 scope/status 分布与标签总数，且软删口径一致",
          bool(tax["scopes"]) and len(tax["statuses"]) == len(L.STATUSES)
          and tax["tagTotal"] == L.tag_facets()["total"])

    # ================================================================ ⑦ 审计
    log = L.audit_log(limit=200)
    actions = {i["action"] for i in log["items"]}
    check("⑦ 审计覆盖全部写动作，且动作名都在闭集里",
          {"create", "update", "delete", "restore", "revert", "purge"} <= actions
          and actions <= set(L.ACTIONS), sorted(actions))
    check("⑦ 审计记了 operator 与 role",
          any(i["operator"] == "alice" and i["role"] == "editor" for i in log["items"]))
    page = L.audit_log(limit=2)
    check("⑦ ★total 是全量条数，不是本页条数（分页 UI 靠它）",
          page["total"] == log["total"] and len(page["items"]) == 2)
    check("⑦ 按 action / operator 过滤生效",
          all(i["action"] == "purge" for i in L.audit_log(action="purge")["items"])
          and all(i["operator"] == "bob" for i in L.audit_log(operator="bob")["items"]))

    http_checks(ev, keep["id"])
    return report()


def report() -> int:
    failed = [name for name, ok, _ in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"✗ {name}  ->  {detail!r}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def http_checks(ev: list[str], keep_id: str) -> None:
    """⑧ HTTP 契约：错误码各就各位 ＋ 契约文档与 openapi **双向**对账。"""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415
    from app.services import knowledge_ledger as L  # noqa: PLC0415

    client = TestClient(app)
    tid = L.upsert(kind="lesson", scope="EP9", title="接口层编辑对象",
                   body="第一版正文内容", evidence=ev, tags="接口,测试")["id"]

    ok = client.patch(f"{BASE}/entries/{tid}", json={
        "body": "第二版正文内容", "expectedVersion": 1, "operator": "dave", "role": "editor"})
    check("⑧ PATCH 成功 → 200 + 统一信封 + 新版本号",
          ok.status_code == 200 and ok.json()["code"] == 200
          and ok.json()["data"]["version"] == 2, ok.json())
    stale = client.patch(f"{BASE}/entries/{tid}", json={"body": "旧版覆盖", "expectedVersion": 1})
    check("⑧ ★版本冲突 → 409（不是 400、更不是 200）",
          stale.status_code == 409 and "版本冲突" in stale.json()["message"], stale.json())
    lie = client.patch(f"{BASE}/entries/{tid}", json={"evidence": ["run:made-up-123"]})
    check("⑧ ★证据核不过 → 422（格式都对、内容站不住）",
          lie.status_code == 422 and "证据核不过" in lie.json()["message"], lie.json())
    check("⑧ 未知角色 → 400；viewer 越权改 → 403",
          client.patch(f"{BASE}/entries/{tid}", json={"body": "x", "role": "god"}).status_code == 400
          and client.patch(f"{BASE}/entries/{tid}", json={"body": "x",
                                                          "role": "viewer"}).status_code == 403)
    check("⑧ ★硬删需要 owner ⇒ editor 403",
          client.delete(f"{BASE}/entries/{tid}",
                        params={"hard": "true", "role": "editor"}).status_code == 403)
    check("⑧ 编辑不存在条目 → 404；空 PATCH → 400",
          client.patch(f"{BASE}/entries/lesson-nope", json={"body": "x"}).status_code == 404
          and client.patch(f"{BASE}/entries/{tid}", json={}).status_code == 400)
    check("⑧ 非数字 expectedVersion → 400（不猜成 0）",
          client.patch(f"{BASE}/entries/{tid}",
                       json={"body": "x", "expectedVersion": "abc"}).status_code == 400)

    dele = client.delete(f"{BASE}/entries/{tid}", params={"reason": "接口删", "operator": "dave"})
    check("⑧ DELETE 默认软删 → 200 + hard=False",
          dele.status_code == 200 and dele.json()["data"]["hard"] is False, dele.json())
    check("⑧ 软删后 GET 详情 → 404；带 includeDeleted=true 才 200",
          client.get(f"{BASE}/entries/{tid}").status_code == 404
          and client.get(f"{BASE}/entries/{tid}",
                         params={"includeDeleted": "true"}).status_code == 200)
    check("⑧ 硬删未软删的条目 → 409（状态冲突，不是参数错）",
          client.delete(f"{BASE}/entries/{keep_id}",
                        params={"hard": "true", "role": "owner"}).status_code == 409)
    check("⑧ owner 硬删已软删条目 → 200",
          client.delete(f"{BASE}/entries/{tid}",
                        params={"hard": "true", "role": "owner"}).status_code == 200)

    vs = client.get(f"{BASE}/entries/{keep_id}/versions")
    check("⑧ GET versions → 200 + current/items；不存在的条目 → 404",
          vs.status_code == 200 and vs.json()["data"]["current"] >= 1
          and client.get(f"{BASE}/entries/lesson-nope/versions").status_code == 404, vs.json())
    check("⑧ GET 单个版本：存在的 200 + isCurrent，不存在的 404",
          client.get(f"{BASE}/entries/{keep_id}/versions/1").json()["data"]["isCurrent"] is True
          and client.get(f"{BASE}/entries/{keep_id}/versions/99").status_code == 404)
    check("⑧ POST revert 缺 version → 400",
          client.post(f"{BASE}/entries/{keep_id}/revert", json={}).status_code == 400)
    check("⑧ GET /tags/facets 与 /taxonomy 挂上并走统一信封",
          client.get(f"{BASE}/tags/facets").json()["code"] == 200
          and client.get(f"{BASE}/taxonomy").json()["data"]["available"] is True)
    check("⑧ GET /audit 未知 action → 400（不静默回全量）",
          client.get(f"{BASE}/audit", params={"action": "hack"}).status_code == 400
          and client.get(f"{BASE}/audit", params={"limit": 1}).json()["data"]["total"] > 1)
    L.upsert(kind="fact", scope="EP9", title="接口层标签透传", body="透传用",
             tags="接口,透传")
    check("⑧ list/search 的 tags 与 includeDeleted 透传生效（不是收下参数就丢掉）",
          client.get(f"{BASE}/entries", params={"tags": "接口"}).json()["data"]["total"] >= 1
          and client.get(f"{BASE}/search", params={"q": "透传", "tags": "接口"}
                         ).json()["data"]["total"] >= 1)
    check("⑧ POST /entries 假证据 → 422（与 PATCH 同口径）",
          client.post(f"{BASE}/entries", json={"kind": "lesson", "title": "接口层假证据",
                                               "body": "x",
                                               "evidence": ["trace:agent/none"]}).status_code == 422)

    paths = app.openapi()["paths"]
    managed = ["/entries/{entry_id}", "/entries/{entry_id}/status",
               "/entries/{entry_id}/restore", "/entries/{entry_id}/versions",
               "/entries/{entry_id}/versions/{version}", "/entries/{entry_id}/revert",
               "/tags/facets", "/taxonomy", "/audit"]
    check("⑧ 管理面端点全部挂在 /api/v1/knowledge-loop 下",
          not [p for p in managed if f"{BASE}{p}" not in paths],
          [p for p in managed if f"{BASE}{p}" not in paths])

    text = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    check("⑧ 契约文档 docs/knowledge-base-contract.md 存在", bool(text))
    named = {f"{m.group(1)} {m.group(2).split('?')[0].rstrip('.,;)`')}"
             for m in re.finditer(r"\b(GET|POST|PATCH|DELETE)\s+(/api/v1/knowledge-loop/\S+)", text)}
    check("⑧ ★文档点名的端点都在 openapi 里（文档不会凭空承诺）",
          not [n for n in named if n.split(" ")[1] not in paths],
          [n for n in named if n.split(" ")[1] not in paths])
    code_eps = {f"{verb.upper()} {path}" for path, ops in paths.items()
                if path.startswith(BASE) for verb in ops}
    check("⑧ ★openapi 里的端点文档都提了（新增端点必须同步契约文档）",
          not [e for e in code_eps if e not in named], sorted(e for e in code_eps if e not in named))
    check("⑧ 文档写明四种错误码与「未鉴权」这条事实",
          all(token in text for token in ("400", "403", "404", "409", "422"))
          and any(phrase in text for phrase in ("没有认证", "未做鉴权", "无认证")))


if __name__ == "__main__":
    sys.exit(main())
