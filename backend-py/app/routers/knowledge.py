"""``/api/v1/knowledge`` —— 剧本/编剧知识库的读口（7 端点）＋ 写口（6 端点，2026-11-06）。

本文件**不做任何检索逻辑**：分词、bm25/LIKE 分流、旧盘路径重映射、目录过滤、盘点全在
``app/services/script_library.py``（＋ ``script_catalog``）。Agent 工具
``app/agent/tools/library_tools.py`` 与命令行 ``app/scripts/corpus/library.py`` 调的是同一份
⇒ 三者不可能给出三种答案。写操作同理：定位/校验/落库全在 ``script_library`` 的
``write_chunk`` / ``bury_document`` / ``raise_document`` 与 ``library_ingest.ingest``。

读为什么只读、写为什么能写
==========================

三个索引库**都不是本仓的真源**，谁都不归这一层就地改：

* ``剧本知识库/`` —— 第三方只读资料，``curated-screenplay-methods.sqlite`` 是供应商**预建**的，
  本仓连打开都用 ``?mode=ro``；
* ``indexes/library-ocr.sqlite``（含 ``scanned-*.sqlite`` 分卷）—— ``ocr_index.py`` 烧出来的，``--redo`` 就覆盖；
* ``indexes/library-extra.sqlite`` —— ``lib_index.py`` 从盘上抽的，``--all`` 就覆盖。

所以「编辑/删除」不落在它们身上，而是落在**第四库** ``data/db/library-overlay.sqlite``
（``app/services/library_overlay.py``）：编辑写 ``edits``、删除写 ``tombstones``，读取时叠加/过滤。
好处是「改完立刻能搜到、能看到」而**原库重建也不会把改动冲掉**；代价是原库若重建把某一份
删了，那些编辑会**如实报 ``staleEdits``**（不假装还在）。

仍然**不提供**的两个动作：
* 「重建索引」—— 需要重新抽 5403 份 PDF（19 GB 含大量扫描件），是离线批处理，
  不该藏在一个 HTTP 端点后面（走 ``app/scripts/corpus/`` 下的脚本，可审、可回滚）；
* 「改写供应商/OCR 库里的正文」—— 那会让下一次重建静默抹掉改动，也会让
  ``onDisk=true`` 变成一句假话。

七个端点
========

* ``GET /report``    —— 四个来源的盘点（methods 覆盖率 / ocr 进度 / playbook 小节 / catalog 目录）
* ``GET /tree``      —— 目录树 ＋ 每个目录的正文覆盖（前端左侧浏览）
* ``GET /documents`` —— 某目录下的文档清单，分页（前端「点开目录看书」）
* ``GET /search``    —— 检索：``country``/``genre``/``category``/``docType``/``permission``/
                        ``festival`` 六个目录维度（只筛 methods）＋ ``skill``（只筛 playbook）
* ``GET /facets``    —— 六个维度 + ``playbookSkills`` 的可选值（别让模型/前端瞎猜词）
* ``GET /document``  —— 按结果里的相对路径翻原文（页号 + 块号 + 正文，分页）
* ``GET /scenes``    —— 该文档的**场次表**（A 侧派生索引，按需生成；没建过就如实说"没解析过"，
  而不是回一个空数组让调用方以为"这部只有 0 场"）

六个写端点（都落覆盖层，都不动那三个索引库）
==========================================

* ``GET    /document/state``  —— 这份能不能改/能不能删/能不能物理删（**界面先问再画按钮**）；
* ``POST   /document/chunk``  —— 改一块正文（JSON body；``chunkNo=-1`` 改标题）；
* ``DELETE /document/chunk``  —— 撤销某块的改动，回到原库正文；
* ``DELETE /document``        —— 删除：默认进**回收站**（可还原），``hard=true`` 只对本仓上传的件有效；
* ``POST   /document/restore``—— 从回收站还原；
* ``POST   /upload``          —— 上传语料（多文件，落 ``00-本仓补录/`` 并即刻入补录库）。

回执纪律（三条，任何一条破了就等于说谎）
========================================

1. ``available=false``（库没装）与「检索到 0 条」**必须能区分** ⇒ 原样透出 ``available``/``hint``；
2. ``onDisk=false``（索引里有、盘上没有）**必须原样透出**，不许前端吞成「找到一条参考」；
3. ``ranking`` 必须透出（``bm25`` / ``like`` / ``gram``）—— 三种排序**不可比**，
   混在一起展示就是错误结论。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Query, UploadFile

from ..core.response import bad_request, success
from ..services import library_ingest, scene_index, script_catalog, script_library

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

#: 六个别名到目录维度的映射（对外全用 camelCase，内部维度名在 ``script_catalog`` 里）
_FILTER_KEYS = ("country", "genre", "category", "docType", "permission", "festival")


@router.get("/report")
def get_report() -> Any:
    """盘点：库多大、缺几份、正文覆盖率多少、目录 CSV 在不在。"""
    return success(script_library.library_report())


@router.get("/tree")
def get_tree(
    depth: int = Query(2, ge=1, le=3,
                       description="目录层级：1=资料大类，2=题材/合集，3=最细一层目录（书/剧本所在的目录）"),
) -> Any:
    """目录树 ＋ 每个目录的正文覆盖（``documents/withText/ocr/pending``）—— 前端「浏览」态的数据源。"""
    return success(script_library.library_tree(depth=depth))


@router.get("/documents")
def get_documents(
    category: str | None = Query(None, description="目录前缀（取 /tree 的 groups[].path）"),
    q: str = Query("", description="按文件名包含匹配"),
    only: str = Query("all", pattern="^(all|withText|ocr|partial|pending|extra|edited|buried)$",
                      description="all / withText 有正文 / ocr 本仓已补整本 / "
                                  "partial 补了一半 / pending 待补（含 partial）/ extra 本仓补录 / "
                                  "edited 改过的 / buried 回收站（唯一能看到已删的档位）"),
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=200),
) -> Any:
    """某目录下的文档清单（分页）—— 点开一个目录看它有哪些书/剧本。"""
    return success(script_library.list_documents(
        category=category, q=q, only=only, offset=offset, limit=limit))


@router.get("/search")
def get_search(
    q: str = Query("", description="检索词；多个词用空格分隔"),
    source: str = Query("all", description="all / methods / playbook"),
    limit: int = Query(5, ge=1, le=30),
    path: str | None = Query(None, description="只在该路径片段下找（相对库根）"),
    country: str | None = Query(None, description="国家地区，可用 | 或 , 分隔多个"),
    genre: str | None = Query(None, description="二级分类（类型）"),
    category: str | None = Query(None, description="一级分类（目录）"),
    doc_type: str | None = Query(None, alias="docType", description="资料类型"),
    permission: str | None = Query(None, description="权限状态"),
    festival: str | None = Query(None, description="电影节（含关系，如「戛纳·主竞赛」）"),
    skill: str | None = Query(None, description="技能目录，如 film-acting（只作用于 playbook）"),
    include_partial: bool = Query(
        False, alias="includePartial",
        description="是否连「只 OCR 了前几页」的半成品一起返回（默认排除；返回时带 coverage）"),
) -> Any:
    """检索。目录维度只筛 ``methods``，``skill`` 只筛 ``playbook``；给错源会标 ``filtersApplied=false``。"""
    query = (q or "").strip()
    if not query:
        return bad_request("缺少检索词 q（例：q=三幕结构）")
    filters: dict[str, Any] = {
        "country": country,
        "genre": genre,
        "category": category,
        "docType": doc_type,
        "permission": permission,
        "festival": festival,
    }
    try:
        outcome = script_library.search(
            query,
            source=source or "all",
            limit=limit,
            path_contains=(path or None),
            filters={k: v for k, v in filters.items() if v},
            skill=(skill or None),
            include_partial=bool(include_partial),
        )
    except ValueError as exc:  # source 非法 —— 明确 400，别静默当成 all
        return bad_request(str(exc))
    return success(outcome)


@router.get("/facets")
def get_facets() -> Any:
    """目录六个维度（methods）＋ ``playbookSkills``（playbook）的可选值。前端下拉、Agent 自查都用它。"""
    report = script_catalog.report(script_library.LIBRARY_ROOT,
                                   marker=script_library.LOCAL_MARKER)
    if not report.get("available"):
        return success({"available": False, "hint": report.get("hint"),
                        "dimensions": {key: {} for key in _FILTER_KEYS},
                        "playbookSkills": script_library.playbook_skills()})
    return success({
        "available": True,
        "entries": report.get("entries"),
        "dimensions": script_catalog.facets(script_library.LIBRARY_ROOT,
                                            marker=script_library.LOCAL_MARKER),
        #: 技能名只有 11 个、且是会变的（有人加一个 film-xxx）⇒ 一并透出，别让调用方猜。
        "playbookSkills": script_library.playbook_skills(),
    })


@router.get("/document")
def get_document(
    path: str = Query(..., description="检索结果里的相对路径（也接受 G:\\… 原始路径）"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """翻原文：文档头 + 正文块（页号/块号齐全），分页。"""
    if not (path or "").strip():
        return bad_request("缺少 path（用 /search 结果里的 path 字段）")
    return success(script_library.open_document(path.strip(), offset=offset, limit=limit))


@router.get("/scenes")
def get_scenes(
    path: str = Query(..., description="文档的相对路径（同 /document 的 path）"),
    page: int = Query(1, ge=1, description="场次分页；一次给完整场次表通常是几十到几百行"),
    limit: int = Query(200, ge=1, le=1000),
) -> Any:
    """该文档的场次表（场号 / 内外景 / 地点 / 时段 / 出场角色 / 页区间 / 首句引用）。

    三种回执必须能区分（同本文件头部的回执纪律）：

    * ``available=false`` —— **场次索引还没建**（先跑 ``python -m app.scripts.corpus.scene_index``）；
    * ``available=true`` + ``scenes=[]`` + ``hint`` —— 索引在、但**这份没解析过**（按需生成，不是漏做）；
    * ``document.status="no-heading"`` —— 解析过了但**没认出任何场次边界**，整篇记一场。

    只读：不在这里触发解析（解析要读全本正文，属批处理，走 ``app/scripts/corpus/``）。
    """
    rel = script_catalog.normalize((path or "").strip(), script_library.LOCAL_MARKER)
    if not rel:
        return bad_request("路径解析不出相对路径（缺库目录锚点）。请传 /search 结果里的 path。")
    outcome = scene_index.read_scenes(rel)
    scenes = outcome.get("scenes") or []
    start = (page - 1) * limit
    return success({
        **outcome,
        "scenes": scenes[start:start + limit],
        "total": len(scenes),
        "page": page,
        "limit": limit,
    })


# ===========================================================================
# 写：状态 / 改 / 撤销 / 删 / 还原 / 上传
# ===========================================================================


@router.get("/document/state")
def get_document_state(
    path: str = Query(..., description="文档的相对路径（同 /document 的 path）"),
) -> Any:
    """这份能改吗、能删吗、能物理删吗 —— **界面先问再画按钮**，别让人点了才报错。"""
    if not (path or "").strip():
        return bad_request("缺少 path")
    return success(script_library.document_state(path.strip()))


@router.post("/document/chunk")
def post_document_chunk(payload: dict[str, Any]) -> Any:
    """改一块正文。落在**覆盖层**：检索与翻页立刻按改后的走，原库文件一个字不动。

    正文走 **JSON body** 而不是 query —— 一块正文可达 800 字以上，走 query 会被各级代理的
    URL 长度上限截断，症状是「保存成功但后面半段丢了」，最难查。
    """
    path = str(payload.get("path") or "").strip()
    if not path:
        return bad_request("缺少 path")
    if "chunkNo" not in payload:
        return bad_request("缺少 chunkNo（改标题传 -1）")
    try:
        chunk_no = int(payload["chunkNo"])
    except (TypeError, ValueError):
        return bad_request("chunkNo 必须是整数（改标题传 -1）")
    text = payload.get("text")
    if text is None:
        return bad_request("缺少 text（要清空请传空串）")
    outcome = script_library.write_chunk(path, chunk_no, str(text),
                                         note=payload.get("note"))
    if not outcome.get("ok"):
        return bad_request(outcome.get("reason") or "保存失败")
    return success(outcome)


@router.delete("/document/chunk")
def delete_document_chunk(
    path: str = Query(..., description="文档的相对路径"),
    chunk_no: int = Query(..., alias="chunkNo", description="要还原的块号（-1 = 标题）"),
) -> Any:
    """撤销某块的改动，回到原库那份正文。"""
    if not (path or "").strip():
        return bad_request("缺少 path")
    outcome = script_library.undo_chunk(path.strip(), chunk_no)
    if not outcome.get("ok"):
        return bad_request(outcome.get("reason") or "撤销失败")
    return success(outcome)


@router.delete("/document")
def delete_document(
    path: str = Query(..., description="文档的相对路径"),
    hard: bool = Query(False, description="true = 彻底删除（**只对本仓上传的件有效**）；"
                                         "默认进回收站，可还原"),
    reason: str | None = Query(None, description="删除理由（会记进回收站）"),
) -> Any:
    """删除一份。默认进回收站：清单/检索/目录树都不再出现，原文件仍在盘上、随时可还原。"""
    if not (path or "").strip():
        return bad_request("缺少 path")
    outcome = script_library.bury_document(path.strip(), hard=bool(hard), reason=reason)
    if not outcome.get("ok"):
        return bad_request(outcome.get("reason") or "删除失败")
    return success(outcome)


@router.post("/document/restore")
def post_document_restore(
    path: str = Query(..., description="回收站里那份的相对路径"),
) -> Any:
    """从回收站还原。"""
    if not (path or "").strip():
        return bad_request("缺少 path")
    outcome = script_library.raise_document(path.strip())
    if not outcome.get("ok"):
        return bad_request(outcome.get("reason") or "还原失败")
    return success(outcome)


@router.get("/overlay")
def get_overlay() -> Any:
    """覆盖层盘点：改了哪些、删了哪些、最近传了什么（前端「回收站」抽屉的唯一数据源）。

    顺带下发 ``upload`` 口径（可选扩展名/单份上限/一次几份）：前端拿它做**提交前**校验，
    与后端判据同一真源 —— 前端抄一份就会变成「界面说能传、后端说不行」。
    """
    return success({**script_library.overlay_report(), "upload": library_ingest.limits()})


@router.post("/upload")
async def post_upload(
    files: list[UploadFile] = File(..., description="语料文件，可多选；"
                                                   f"支持 {library_ingest.ALLOWED_LABEL}"),
) -> Any:
    """上传语料：落 ``00-本仓补录/`` → 抽正文 → 入补录库 → 立刻能搜到。

    逐个文件独立处理、独立回结果：一份太大或格式不对**不会**连累其它几份，
    也不会让整批报一个笼统的失败（前端按 ``status`` 分别显示「已入库 / 待处理 / 被拒」）。
    单份上限 ``library_ingest.MAX_FILE_SIZE``，一次最多 ``library_ingest.MAX_FILES`` 份。
    """
    if not files:
        return bad_request("没有收到文件（字段名应为 files）。")
    if len(files) > library_ingest.MAX_FILES:
        return bad_request(f"一次最多 {library_ingest.MAX_FILES} 份，"
                           f"这次给了 {len(files)} 份，请分批上传。")
    results: list[dict[str, Any]] = []
    for upload in files:
        name = upload.filename or "未命名"
        try:
            data = await upload.read()
        except Exception as exc:  # noqa: BLE001
            results.append({"ok": False, "status": "rejected", "name": name,
                            "reason": f"读不到上传内容：{exc}", "size": 0})
            continue
        results.append(library_ingest.ingest(data, name))
    ok_count = sum(1 for row in results if row.get("status") == "ingested")
    return success({
        "results": results,
        "total": len(results),
        "ingested": ok_count,
        "quarantined": sum(1 for row in results if row.get("status") == "quarantined"),
        "rejected": sum(1 for row in results if row.get("status") == "rejected"),
        "allowed": library_ingest.ALLOWED_LABEL,
        "maxSize": library_ingest.MAX_FILE_SIZE,
        "maxFiles": library_ingest.MAX_FILES,
        "hint": ("已入库的文件立刻能在清单里看到（目录：00-本仓补录）。"
                 "「待处理」表示文件在盘上但抽不出正文（扫描件/图片型）——那类需要走 OCR。"),
    })
