"""知识库目录层 + 只读路由自检（``script_catalog`` / ``/api/v1/knowledge``，2026-11-04）。

这一层最容易犯的错，**全是"不报错但结论错"**：

1. **编码**：供应商两份 CSV 编码不同（主目录 UTF-8 BOM、电影节标签 **GBK**）。用 utf-8 硬读
   GBK **不抛错**，只会得到乱码 ⇒ 后果是「按电影节筛永远 0 条」，看起来像"库里没这部"。
   本测试既正向验（读得出戛纳），也**反向验**（那串字节用 utf-8 解确实会抛错 ⇒ 夹具真的有区分度）。
2. **三态**：``没给条件`` / ``给了但目录不在`` / ``给了但没人满足`` 必须能分开。
   最坏的是把「目录 CSV 没装」退化成空集 —— 那等于告诉模型"库里没有"。
3. **过滤发生在取样之后**：目录字段不在 sqlite 里，只能在 bm25 窗口内筛 ⇒ 必须回报
   ``window``，不许让"命中 300 段只给 2 段"看起来像库里只有 2 段。
4. **引用可回溯**：``open_document`` 要能真的翻到页号/块号；索引里没有但目录里有的文件
   要说清「登记过但没抽出文字（多为扫描件）」，不能只说"没找到"。
5. **路由只读**：四条 GET；缺 ``q`` 是 400 而不是"空结果"。

夹具是真 CSV（含真 GBK 字节）+ 真 fts5 trigram 库 + 真文件，不 mock —— 出问题的全是
「三份数据源对不上」，mock 掉就什么都测不到。

运行::

    ./.venv/Scripts/python.exe tests/script_catalog_test.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.routers import knowledge as kn  # noqa: E402
from app.services import library_overlay as ov  # noqa: E402
from app.services import script_catalog as sc  # noqa: E402
from app.services import script_library as sl  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

_G = "G:\\电影剧本\\全球热门与获奖影视剧本知识库"

_MAIN_HEADER = ("知识库ID,文件名,一级分类,国家地区,二级分类,资料类型,语言判断,可检索范围,"
                "扩展名,大小MB,页数,索引状态,权限状态,本地相对路径,本地完整路径,备注")
_MAIN_ROWS = [
    ("1", "韩国犯罪.txt", "类型片", "韩国", "犯罪", "剧本", "韩语", "可检索", "txt", "1.0", "120",
     "已索引", "可引用", "05-类型片/韩国犯罪.txt", "", ""),
    ("2", "美国科幻.txt", "类型片", "美国", "科幻", "剧本", "英语", "可检索", "txt", "1.0", "110",
     "已索引", "可引用", "05-类型片/美国科幻.txt", "", ""),
    ("3", "故事.txt", "编剧方法", "美国", "理论", "方法书", "英语", "可检索", "txt", "1.0", "300",
     "已索引", "可引用", "08-编剧方法/故事.txt", "", ""),
]
_FESTIVAL_HEADER = "电影节,影片,电影节关系,本地完整路径,电影节官方证明页,公开剧本来源,索引状态"
_FESTIVAL_ROWS = [("戛纳", "韩国犯罪", "主竞赛", f"{_G}\\05-类型片\\韩国犯罪.txt", "", "", "已索引")]


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _build_fixture(root: Path) -> tuple[Path, Path]:
    kb = root / "剧本知识库" / "全球热门与获奖影视剧本知识库"
    index_dir = kb / "99-索引与检索规则"
    index_dir.mkdir(parents=True)
    (kb / "05-类型片").mkdir(parents=True)
    (kb / "08-编剧方法").mkdir(parents=True)
    for name in ("韩国犯罪.txt", "美国科幻.txt"):
        (kb / "05-类型片" / name).write_text("x", encoding="utf-8")
    (kb / "08-编剧方法" / "故事.txt").write_text("x", encoding="utf-8")

    main_csv = index_dir / sc.MAIN_CSV
    #: **UTF-8 带 BOM**（供应商原生就是这样，不带 BOM 的夹具测不出「BOM 会不会留在首列名上」）
    main_csv.write_text("\ufeff" + _MAIN_HEADER + "\n"
                        + "\n".join(",".join(row) for row in _MAIN_ROWS) + "\n",
                        encoding="utf-8")
    #: **GBK**（这一步是有意的：utf-8 硬读会乱码且不抛错）
    festival_csv = index_dir / sc.FESTIVAL_CSV
    festival_csv.write_bytes(
        (_FESTIVAL_HEADER + "\n" + "\n".join(",".join(row) for row in _FESTIVAL_ROWS) + "\n")
        .encode("gbk"))

    db = index_dir / "curated-screenplay-methods.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, name TEXT, "
                 "extension TEXT, size INTEGER, pages INTEGER, status TEXT)")
    conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(document_id, path, name, page, "
                 "chunk_no, text, tokenize=trigram)")
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", [
        (1, f"{_G}\\05-类型片\\韩国犯罪.txt", "韩国犯罪.txt", "txt", 100, 120, "indexed"),
        (2, f"{_G}\\05-类型片\\美国科幻.txt", "美国科幻.txt", "txt", 100, 110, "indexed"),
        (3, f"{_G}\\08-编剧方法\\故事.txt", "故事.txt", "txt", 100, 300, "indexed"),
    ])
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", [
        (1, f"{_G}\\05-类型片\\韩国犯罪.txt", "韩国犯罪.txt", 12, 0,
         "雨夜追逐之后，三幕结构在第二幕中点断裂。"),
        (2, f"{_G}\\05-类型片\\美国科幻.txt", "美国科幻.txt", 33, 0,
         "美国科幻也讲三幕结构，但节奏更快。"),
        (3, f"{_G}\\08-编剧方法\\故事.txt", "故事.txt", 7, 0,
         "激励事件把必备场景投射到观众想象之中。"),
    ])
    conn.commit()
    conn.close()

    playbook = root / "playbook"
    #: 夹具目录名必须落在真实 :data:`sl.PLAYBOOK_DIRS` 白名单里（2026-11-04 扁平化后包名直接在
    #: playbook 根下，`packs/` 壳已不存在）—— 造个白名单外的名字会让整棵夹具不被索引。
    acting = playbook / "film-expert-bridge"
    acting.mkdir(parents=True)
    (acting / "SKILL.md").write_text(
        "# 表演层\n\n## 微表情与反应\n\n三幕结构的每一幕，情绪强度都要有可见的肌肉变化。\n",
        encoding="utf-8")
    #: 第二个技能：用来验 skill 过滤真的在筛（只有一个技能时"筛"和"不筛"结果一样）
    cinedance = playbook / "film-cinedance"
    cinedance.mkdir(parents=True)
    (cinedance / "SKILL.md").write_text(
        "# 镜头调度\n\n## 轴线与走位\n\n三幕结构的转身必须压在同一侧轴线上，越轴会让观众瞬间失向。\n",
        encoding="utf-8")
    #: 超长小节（> PLAYBOOK_SECTION_MAX）：检索词只出现在中段 ⇒ 必须切窗并标 part/parts，
    #: 且 excerpt 要落在带词的那一窗（不切窗时 excerpt 会取全文开头，离词半页远）
    filler = "\n\n".join(f"第 {index} 段铺垫：走位与光比都要跟着情绪推。" for index in range(60))
    (acting / "LONG.md").write_text(
        f"# 长节\n\n## 走位与情绪\n\n{filler}\n\n中段的锚点：三幕结构的转折点在这里。\n\n"
        f"{filler}\n", encoding="utf-8")
    #: 白名单外的两份：根目录使用说明 + project-template 空壳 ⇒ **不许进索引**
    (playbook / "AGENTS.md").write_text(
        "# 使用规约\n\n## 怎么用这个技能包\n\n三幕结构不是这个包讲的东西，这里讲的是怎么调用脚本。\n",
        encoding="utf-8")
    template = playbook / "project-template"
    template.mkdir()
    (template / "README.md").write_text(
        "# 模板\n\n## 占位\n\n三幕结构 请替换成你的项目内容。\n", encoding="utf-8")
    return kb, index_dir


def main() -> int:  # noqa: C901
    root = Path(tempfile.mkdtemp(prefix="scriptcat_"))
    kb, index_dir = _build_fixture(root)
    db = index_dir / "curated-screenplay-methods.sqlite"
    original = (sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB,
                sl.EXTRA_DB, ov.OVERLAY_DB)
    sl.LIBRARY_ROOT, sl.METHODS_DB = kb, db
    sl.PLAYBOOK_ROOT = root / "playbook"
    #: OCR sidecar 必须一起隔离：它是开发机 `indexes/` 下的真实产物，不指走就会
    #: 让夹具里「扫描件无正文」那些断言读到真 OCR 库（实测踩过：window 与 available 都对不上）。
    sl.OCR_DB = root / "ocr" / "不存在.sqlite"
    #: 第五库（本仓补录）与第四库（覆盖层）同理，且**必须一起指走**：`_scan_documents`
    #: 会把 extra 合进同一张清单，开发机上真库 880 份会让「夹具 4 份」的断言全部对不上
    #: （实测 884 份、pending 2 条）。
    sl.EXTRA_DB = root / "extra" / "不存在.sqlite"
    ov.OVERLAY_DB = root / "overlay" / "不存在.sqlite"
    sl.reset_cache()

    # ================= 1. 路径归一化 =================
    check("归一化: G:\\ 旧盘路径 -> 相对库根的 posix 路径",
          sc.normalize(f"{_G}\\05-类型片\\韩国犯罪.txt") == "05-类型片/韩国犯罪.txt",
          sc.normalize(f"{_G}\\05-类型片\\韩国犯罪.txt"))
    check("归一化: 已经是相对路径的原样（换成 /）",
          sc.normalize("05-类型片\\韩国犯罪.txt") == "05-类型片/韩国犯罪.txt")
    check("归一化: 绝对盘符但没有库目录锚点 -> \"\"（**不猜**）",
          sc.normalize("D:\\别的地方\\韩国犯罪.txt") == "")

    # ================= 2. 两份 CSV（编码不同） =================
    table = sc.entries(kb)
    check("目录: 主 CSV（UTF-8 BOM）读出 3 条", len(table) == 3, len(table))
    hit = table.get("05-类型片/韩国犯罪.txt")
    check("目录: 字段落位（国家/类型/一级分类/权限）",
          hit is not None and (hit.country, hit.genre, hit.category, hit.permission)
          == ("韩国", "犯罪", "类型片", "可引用"), hit.as_dict() if hit else None)
    check("目录: 电影节 CSV（**GBK**）真的被解出来（戛纳·主竞赛）",
          hit is not None and hit.festivals == ("戛纳·主竞赛",),
          hit.festivals if hit else None)
    raw_gbk = (index_dir / sc.FESTIVAL_CSV).read_bytes()
    try:
        raw_gbk.decode("utf-8")
        utf8_ok = True
    except UnicodeDecodeError:
        utf8_ok = False
    check("目录: 反证 —— 那串字节用 utf-8 解确实会抛错（否则上一条测的是空气）", not utf8_ok)

    # ================= 3. 三态：没给 / 不在 / 没人满足 =================
    only_kr = sc.select(kb, country="韩国")
    check("过滤: country=韩国 -> 只允许那一部", only_kr == {"05-类型片/韩国犯罪.txt"}, only_kr)
    check("过滤: 多值用逗号分隔（韩国,美国 -> 三部，含 08-编剧方法/故事.txt）",
          sc.select(kb, country="韩国,美国") == {"05-类型片/韩国犯罪.txt",
                                                 "05-类型片/美国科幻.txt",
                                                 "08-编剧方法/故事.txt"})
    check("过滤: 条件太窄 -> 空集（真结论，要如实回报）",
          sc.select(kb, country="韩国", genre="恐怖") == set())
    check("过滤: 没给条件 -> None（**不是**空集）", sc.select(kb) is None)
    orphan = Path(tempfile.mkdtemp(prefix="scriptcat_nocat_"))
    check("过滤: 目录 CSV 不在 -> None（**不能退化成空集**）",
          sc.select(orphan, country="韩国") is None)

    facets = sc.facets(kb)
    check("分面: country 计数正确（韩国 1 / 美国 2）",
          facets["country"].get("韩国") == 1 and facets["country"].get("美国") == 2,
          facets["country"])
    check("分面: festival 只有一部带标签",
          facets["festival"] == {"戛纳·主竞赛": 1}, facets["festival"])

    report = sc.report(kb)
    check("盘点: 目录 3 条 + 两份文件都在",
          report["available"] and report["entries"] == 3
          and report["files"] == {"main": True, "festival": True}, report)

    # ================= 4. 检索层联动 =================
    filtered = sl.search_methods("三幕结构", limit=5, filters={"country": "韩国"})
    titles = [row["title"] for row in filtered["results"]]
    check("联动: 国别过滤真的筛掉了美国那部", titles == ["韩国犯罪.txt"], titles)
    check("联动: 回执写明过滤已生效 + 允许集合大小",
          filtered["filters"]["applied"] is True
          and filtered["filters"]["allowed"] == 1, filtered.get("filters"))
    check("联动: 带过滤时交代取样窗口（不许让「只给 1 条」看起来像库里就 1 条）",
          filtered.get("window", {}).get("size", 0) >= 200, filtered.get("window"))

    both = sl.search("三幕结构", source="all", limit=5, filters={"country": "韩国"})
    check("联动: all 模式逐源交代谁被筛过（playbook 无国别维度 -> False + 明说）",
          both["sources"]["methods"]["filtersApplied"] is True
          and both["sources"]["playbook"]["filtersApplied"] is False
          and "未参与过滤" in both["hint"], both["hint"])

    empty = sl.search_methods("三幕结构", limit=5, filters={"country": "火星"})
    check("联动: 没人满足 -> 0 条且 hint 说明是条件太窄（不是「没检索到」）",
          empty["results"] == [] and "筛选条件太窄" in (empty.get("hint") or ""),
          empty.get("hint"))

    sl.LIBRARY_ROOT = orphan  # 目录 CSV 不在，但索引还在
    degraded = sl.search_methods("三幕结构", limit=5, filters={"country": "韩国"})
    degraded_titles = sorted(row["title"] for row in degraded["results"])
    check("降级: 目录缺失 -> applied=False 且**结果未被静默过滤**",
          degraded["filters"]["applied"] is False
          and degraded_titles == sorted(["美国科幻.txt", "韩国犯罪.txt"]),
          (degraded.get("filters"), degraded_titles))
    check("降级: 明确说「这些条件没有生效」", "没有生效" in (degraded["filters"].get("hint") or ""),
          degraded["filters"].get("hint"))
    sl.LIBRARY_ROOT = kb

    # ================= 4b. 参考侧三处收紧（白名单 / skill / 长节切窗） =================
    pb = sl._playbook_report()
    check("参考: 只索引白名单来源目录（根目录规约与 project-template 不进）",
          pb["files"] == 3
          and all(path.split("/")[0] in sl.PLAYBOOK_DIRS
                  for path in {row["path"] for row in sl._playbook_sections()}),
          sorted({row["path"] for row in sl._playbook_sections()}))
    check("参考: 白名单外**照样数出来**（2 份没索引要说出来，不是静默缩小范围）",
          pb["unindexedFiles"] == 2, pb.get("unindexedFiles"))
    check("参考: 盘点里带技能清单（files/sections 分列；长节多窗 = 多小节）",
          pb["skills"]["film-expert-bridge"]["files"] == 2
          and pb["skills"]["film-expert-bridge"]["sections"] >= 4, pb.get("skills"))

    acting_hits = sl.search_playbook("三幕结构", limit=8, skill="film-expert-bridge")
    check("参考: skill 过滤只留该技能（项目模板/使用规约里的同词不再混进来）",
          acting_hits["results"]
          and {row["skill"] for row in acting_hits["results"]} == {"film-expert-bridge"}
          and acting_hits["filters"]["applied"] is True, acting_hits.get("filters"))
    check("参考: skill 名支持 packs/xxx/ 与旧 skills/xxx/ 两种写法（归一后同一条路）",
          sl.search_playbook("三幕结构", skill="packs/film-cinedance/")["matched"] > 0
          and sl.search_playbook("三幕结构", skill="skills/film-cinedance/")["matched"] > 0)
    missing_skill = sl.search_playbook("三幕结构", skill="film-nope")
    check("参考: 技能名不存在 -> applied=False 且列可选项（**不是**「没检索到」）",
          missing_skill["matched"] == 0
          and missing_skill["filters"]["applied"] is False
          and "film-expert-bridge" in (missing_skill["filters"].get("hint") or ""),
          missing_skill.get("filters"))

    long_hits = [row for row in sl.search_playbook("三幕结构", limit=10, skill="film-expert-bridge")["results"]
                 if row["path"].endswith("LONG.md")]
    check("参考: 长节切窗（>1500 字切成多窗，回执带 part/parts）",
          long_hits and long_hits[0].get("parts", 1) > 1 and long_hits[0]["part"] >= 1,
          [(row.get("part"), row.get("parts")) for row in long_hits])
    check("参考: excerpt 落在带词的那一窗（不切窗时会取全文开头、离词半页远）",
          long_hits and "【三幕结构】" in long_hits[0]["excerpt"], long_hits[0]["excerpt"][:80] if long_hits else "")

    skill_only = sl.search("三幕结构", source="all", limit=5, skill="film-cinedance")
    check("参考: skill 给到 all 模式 -> playbook 被筛、methods 如实标未参与过滤",
          skill_only["sources"]["playbook"]["filtersApplied"] is True
          and skill_only["sources"]["methods"]["filtersApplied"] is False
          and "只对 playbook 生效" in skill_only["hint"], skill_only["hint"])

    # ================= 4c. 从参考仓搬来的两件事：切窗句界/重叠 + 跨源 RRF =================
    long_para = "。".join(f"第{index}句讲走位与光比" for index in range(1, 121)) + "。"
    cuts = sl._split_oversized(long_para, 200)
    check("切窗: 无空行的长段落按句界展开（片尾落在句号上，不硬切半句）",
          all(len(piece) <= 200 for piece in cuts)
          and all(piece.endswith("。") for piece in cuts)
          and "".join(cuts) == long_para, [len(piece) for piece in cuts])
    check("切窗: 仅单句本身超宽才硬切（兜底）",
          [len(piece) for piece in sl._split_oversized("无标点" * 200, 200)] == [200, 200, 200])
    messy = "甲。\n\n乙。丙。" * 40
    check("切窗: 拼回去逐字无损（纯空白片挂到下一片，不丢分隔符）",
          "".join(sl._split_oversized(messy, 60)) == messy)
    windows = sl._windows("\n\n".join(f"第{index}段落：走位与光比跟着情绪推。" for index in range(120)))
    check("切窗: 相邻窗重叠（跨窗前后的半句不再只剩一半）",
          len(windows) > 1 and windows[1][:30] in windows[0],
          (len(windows), windows[1][:30]))
    check("切窗: 末窗不留碎块（<20% 窗宽并入前窗，而不是丢弃）",
          len(windows[-1]) >= sl.PLAYBOOK_WINDOW * 0.2, len(windows[-1]))

    fused = sl._fuse([
        {"source": "methods", "title": "m1"}, {"source": "methods", "title": "m2"},
        {"source": "playbook", "title": "p1"}, {"source": "playbook", "title": "p2"},
    ])
    check("融合: 两源第 N 名等权 -> 交替，不再「methods 全在前」",
          [row["title"] for row in fused] == ["m1", "p1", "m2", "p2"],
          [row["title"] for row in fused])
    check("融合: 分数按秩 1/(k+rank) 递减（与 txtai rrf 同口径）",
          [row["fusion"] for row in fused] == [0.5, 0.5, 0.333333, 0.333333],
          [row["fusion"] for row in fused])
    check("融合: 单源顺序不变（RRF 是单调变换，不该打乱源内顺序）",
          [row["title"] for row in sl._fuse([{"source": "playbook", "title": "a"},
                                             {"source": "playbook", "title": "b"}])] == ["a", "b"])

    all_hits = sl.search("三幕结构", source="all", limit=3)
    seq = [row["source"] for row in all_hits["results"]]
    check("融合: all 模式真的交错（不是两源各自一段拼接）",
          len(seq) >= 4 and seq[0] != seq[1] and seq[1] != seq[2], seq)
    check("融合: 回执交付融合口径（method/k/权重 + score 不是排序依据）",
          (all_hits.get("fusion") or {}).get("method") == "rrf"
          and "排序依据" in (all_hits.get("fusion") or {}).get("note", "")
          and "fusion" in all_hits["results"][0], all_hits.get("fusion"))

    # ================= 5. 引用可回溯 =================
    opened = sl.open_document("05-类型片/韩国犯罪.txt", limit=1)
    check("翻原文: 文档头 + 页号 + 块号都在",
          opened["document"]["title"] == "韩国犯罪.txt"
          and opened["chunks"][0]["page"] == 12
          and opened["chunks"][0]["chunkNo"] == 0, opened.get("document"))
    check("翻原文: 目录信息一并带回（国家/奖项）",
          (opened.get("catalog") or {}).get("country") == "韩国"
          and (opened.get("catalog") or {}).get("festivals") == ["戛纳·主竞赛"],
          opened.get("catalog"))
    (kb / "05-类型片" / "空登记.txt").write_text("x", encoding="utf-8")
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                 (9, f"{_G}\\05-类型片\\空登记.txt", "空登记.txt", "txt", 1, 88, "no_text"))
    conn.commit()
    conn.close()
    main_csv = index_dir / sc.MAIN_CSV
    main_csv.write_text(main_csv.read_text(encoding="utf-8")
                        + "9,空登记.txt,类型片,韩国,犯罪,剧本,韩语,可检索,txt,1.0,88,已索引,可引用,"
                          "05-类型片/空登记.txt,,扫描件\n", encoding="utf-8")
    scanned = sl.open_document("05-类型片/空登记.txt")
    check("翻原文: 登记了但没正文块 -> 说清「多为扫描件」，不是静默返回 0 块",
          scanned["total"] == 0 and scanned["chunks"] == []
          and "扫描件" in (scanned.get("hint") or ""), scanned.get("hint"))

    # ================= 6. 路由（只读 + 信封） =================
    mini = FastAPI()
    mini.include_router(kn.router)
    client = TestClient(mini)
    paths = {path for path in mini.openapi()["paths"]}
    read_paths = {"/api/v1/knowledge/report", "/api/v1/knowledge/tree",
                  "/api/v1/knowledge/documents", "/api/v1/knowledge/search",
                  "/api/v1/knowledge/facets", "/api/v1/knowledge/document",
                  "/api/v1/knowledge/scenes"}
    write_paths = {"/api/v1/knowledge/document/state", "/api/v1/knowledge/document/chunk",
                   "/api/v1/knowledge/document/restore", "/api/v1/knowledge/document",
                   "/api/v1/knowledge/overlay", "/api/v1/knowledge/upload"}
    check("路由: 七条只读端点 + 六个写端点，一条不多一条不少",
          paths == read_paths | write_paths, paths)
    check("路由: 只有写端点带写方法（纯读口仍必须 GET-only；/document 是读+删同一路径）",
          all(set(item.keys()) <= {"get", "parameters"}
              for path, item in mini.openapi()["paths"].items()
              if path in read_paths - {"/api/v1/knowledge/document"})
          and {"delete", "get"} == set(mini.openapi()["paths"]["/api/v1/knowledge/document"])
          and "post" in mini.openapi()["paths"]["/api/v1/knowledge/upload"],
          {path: sorted(item) for path, item in mini.openapi()["paths"].items()})
    check("路由: 正文改写只有 POST（JSON body）＋ DELETE（撤销），没有 PUT —— "
          "旧的 PUT/query 版会因 URL 上限把 800 字正文截成「保存成功但后半段丢了」",
          {"post", "delete"} == set(mini.openapi()["paths"]["/api/v1/knowledge/document/chunk"]),
          sorted(mini.openapi()["paths"]["/api/v1/knowledge/document/chunk"]))
    bad = client.get("/api/v1/knowledge/search")
    check("路由: 缺 q -> 400（不是空结果）",
          bad.status_code == 400 and bad.json()["code"] == 400, bad.json())
    ok_query = client.get("/api/v1/knowledge/search",
                          params={"q": "三幕结构", "country": "韩国"}).json()
    check("路由: 过滤生效 + 信封 code=200 + data 里带 filters",
          ok_query["code"] == 200
          and [row["title"] for row in ok_query["data"]["results"]
               if row["source"] == "methods"] == ["韩国犯罪.txt"]
          and ok_query["data"]["filters"]["applied"] is True,
          ok_query["data"]["sources"])
    skill_query = client.get("/api/v1/knowledge/search",
                             params={"q": "三幕结构", "skill": "film-cinedance"}).json()
    check("路由: skill 参数透到 playbook（返回行自带 skill 字段）",
          skill_query["code"] == 200
          and {row["skill"] for row in skill_query["data"]["results"]
               if row["source"] == "playbook"} == {"film-cinedance"},
          skill_query["data"]["sources"].get("playbook"))
    report_body = client.get("/api/v1/knowledge/report").json()["data"]
    check("路由: 盘点带正文覆盖率（登记 4 / 有正文 3 = 0.75）",
          report_body["methods"]["coverage"] == 0.75, report_body["methods"])
    facets_body = client.get("/api/v1/knowledge/facets").json()["data"]
    check("路由: 分面返回维度可选值 + playbook 技能清单",
          facets_body["available"] is True
          and "韩国" in facets_body["dimensions"]["country"]
          and set(facets_body.get("playbookSkills") or {}) == {"film-expert-bridge", "film-cinedance"},
          facets_body.get("playbookSkills"))
    doc_body = client.get("/api/v1/knowledge/document",
                          params={"path": "05-类型片/韩国犯罪.txt", "limit": 1}).json()["data"]
    check("路由: 翻原文可用（中文正文不乱码）",
          "三幕结构" in doc_body["chunks"][0]["text"], doc_body["chunks"][0]["text"][:20])

    # ================= 7. 浏览态两个端点（前端左侧树 + 中间清单）=================
    tree_body = client.get("/api/v1/knowledge/tree", params={"depth": 2}).json()["data"]
    check("浏览: /tree 的分组是目录（4 份全在 05-类型片 与 08-编剧方法 下，后者按数降序在前）",
          tree_body["available"] is True
          and [(group["path"], group["documents"]) for group in tree_body["groups"]]
          == [("05-类型片", 3), ("08-编剧方法", 1)]
          and tree_body["totals"]["documents"] == 4, tree_body["groups"])
    check("浏览: /tree 的 pending 是「两库都没正文」的真数（登记 4 / 有正文 3 ⇒ 待补 1）",
          (tree_body["totals"]["withText"], tree_body["totals"]["pending"]) == (3, 1),
          tree_body["totals"])
    docs_body = client.get("/api/v1/knowledge/documents",
                           params={"category": "05-类型片", "only": "withText"}).json()["data"]
    check("浏览: /documents 前缀匹配 + only 过滤，total 与实际条数一致",
          docs_body["total"] == 2 and len(docs_body["items"]) == 2
          and all(item["withText"] for item in docs_body["items"]), docs_body)
    check("浏览: depth / only 越界由 FastAPI 当场 422（不静默兜底）",
          client.get("/api/v1/knowledge/tree", params={"depth": 5}).status_code == 422
          and client.get("/api/v1/knowledge/documents",
                         params={"only": "nonsense"}).status_code == 422)

    (sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB,
     sl.EXTRA_DB, ov.OVERLAY_DB) = original
    sl.reset_cache()

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {json.dumps(detail, ensure_ascii=False, default=str)[:300]}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
