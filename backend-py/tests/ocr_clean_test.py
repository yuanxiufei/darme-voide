"""OCR 的**校对机制**：清洗 + 质量门禁 + 隔离 + 审计。

为什么单钉这一套：原有唯一的门槛是 ``MIN_PAGE_CHARS = 12``，引擎把整页认成一排
符号（``||||``/``────``）或把一行刷几十遍时字数轻松过 12，脏文本直接进 fts5，
检索命中一堆读不出意思的片段，而且**块里没有"质量"字段，事后看不出来**。

判据只看可确定的东西（不加载 rapidocr、不碰真语料）：

* 清洗只改**排版碎块**（孤字/纯标点行、跨页页眉页脚），两字以上的行一律不动 ——
  合并两字行会把「雨水」「他说」并成假句子，那是制造错误而不是修复；
* 门禁三档 ``ok|suspect|reject``，``reject`` 页**不写块**但必须留审计（含理由与预览）；
* 整份隔离要求 reject 占比 ≥ 50% 且参与判定的页 ≥ 3（单页坏是常态，成片坏才是没认对）；
* CJK 占比**不参与判定**（英文原版剧本 CJK 恒 0，拿它判会把整批英文剧本误杀）。

运行::

    ./.venv/Scripts/python.exe tests/ocr_clean_test.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

from app.scripts.corpus import ocr_index as oi  # noqa: E402
from app.scripts.corpus import text_clean as tc  # noqa: E402
from app.services import script_library as sl  # noqa: E402

CHECKS = 0
FAILED = 0


def check(label: str, ok: bool, detail=None) -> None:
    global CHECKS, FAILED
    CHECKS += 1
    if ok:
        print(f"  ✓ {label}")
    else:
        FAILED += 1
        print(f"  ✗ {label}")
        if detail is not None:
            print(f"      {detail}")


JUNK_PAGE = "|||||||| ||||||| ──────── ████████ ○○○○○○ <<<<<<>>>>>> ****************"

GOOD_ZH = ("第三场 内景 夜 旅店\n\n他推开门，屋里没有人。桌上的茶还热着，说明人刚走不久。\n"
           "“你们要的那批货，我带来了。”凌云把皮箱放下去，声音压得很低。\n")
GOOD_EN = ("INT. HOTEL ROOM - NIGHT\n\nA man pushes the door open. The room is empty, "
           "but the tea on the table is still warm.\n")


# ===========================================================================
# ① 清洗：碎行合并
# ===========================================================================


def case_merge() -> None:
    print("\n① 碎行合并（孤字/纯标点并回上一行，两字以上不动）")

    broken = "他推开门，屋里没有人\n布\n、\n桌上的茶还热着\n，\n“夜色很深”"
    merged = tc.clean_page(broken).split("\n")
    check("纯标点行并回上一行", merged[0].endswith("没有人布、"), merged[:2])
    check("第二个标点行同样并回", merged[1].endswith("还热着，"), merged)

    check("页首孤字向下并到下一行",
          tc.clean_page("布\n他说完就走了").split("\n")[0] == "布他说完就走了",
          tc.clean_page("布\n他说完就走了"))

    #: ★反例：两字行绝不能合并 —— 中文两字成词，合并会把两句并成一句
    two = "他说\n雨水\n忽然"
    check("两字行一律不动（合并会造出假句子）",
          tc.clean_page(two).split("\n") == ["他说", "雨水", "忽然"],
          tc.clean_page(two).split("\n"))

    check("控制字符与零宽字符被清掉",
          tc.clean_page("正文\x00内容\u200b继续").split("\n") == ["正文内容继续"],
          tc.clean_page("正文\x00内容\u200b继续"))
    check("连续空行压成一个空行",
          tc.clean_page("甲\n\n\n\n乙") == "甲\n\n乙",
          repr(tc.clean_page("甲\n\n\n\n乙")))


# ===========================================================================
# ② 页眉页脚剔除
# ===========================================================================


def case_running_heads() -> None:
    print("\n② 跨页页眉页脚剔除（只删短行、只删跨页重复的）")

    head, foot = "·232·", "实惠图书之家独家制作"
    pages = [f"{head}\n世界电影 1987 年第 3 期\n{GOOD_ZH}{foot}" for _ in range(4)]
    out = tc.clean_document(pages)
    check("页码行被删", all("·232·" not in page for page in out["pages"]), out["pages"][0][:30])
    check("刊名行被删", all("1987 年第 3 期" not in page for page in out["pages"]))
    check("水印商号被删", all("独家制作" not in page for page in out["pages"]))
    check("正文一字不少", all("凌云把皮箱放下去" in page for page in out["pages"]))
    check("删除行为有据可查（droppedHeads 非空）", bool(out["droppedHeads"]),
          out["droppedHeads"])

    #: ★反例：两页不足以判"重复"（50% 门槛 ＋ 3 页下限）
    few = tc.clean_document([f"{head}\n{GOOD_ZH}", f"{head}\n{GOOD_ZH}"])
    check("只有两页时不判页眉（样本不够）",
          all("·232·" in page for page in few["pages"]), few["pages"][0][:20])

    #: ★反例：长行即使跨页重复也不删（正文不该被当版头）
    long_line = "这是一段特意写得超过四十个字的长句子，用来验证长行永远不会被当成页眉剔除掉的保护规则。"
    kept = tc.clean_document([f"{long_line}\n{GOOD_ZH}" for _ in range(5)])
    check("超过 24 字的长行永不判页眉",
          all(long_line in page for page in kept["pages"]), kept["pages"][0][:20])

    #: ★反例（实测踩过）：短正文行在页尾、且跨页重复 ⇒ 不许当页脚删掉。
    #: 判据靠「带句读」，不是靠长度 —— 34 字的收尾句照样短。
    tail_line = "“你们要的那批货，我带来了。”凌云把皮箱放下去，声音压得很低。"
    prose = tc.clean_document([f"{GOOD_ZH}{tail_line}" for _ in range(5)])
    check("带句读的短正文行不当页脚删",
          all("凌云把皮箱放下去" in page for page in prose["pages"]),
          prose["pages"][0][-40:])

    #: 实测水印形态：45 字、带全角冒号、无句号、每页都出现 ⇒ 必须删（否则商号成高频词）
    watermark = "· ((&)实惠图书之家独家制作 联系方式 旺旺：实惠图书之家 联系QQ：957719200"
    dirty = tc.clean_document([f"{GOOD_ZH}{watermark}" for _ in range(5)])
    check("长水印行被删（超 24 字但跨页逐字重复）",
          all("957719200" not in page for page in dirty["pages"]),
          dirty["pages"][0][-50:])
    check("水印删除记进 droppedHeads",
          any("957719200" in line for line in dirty["droppedHeads"]), dirty["droppedHeads"])

    #: ★反例：长行要 80% 页重复才删。10 页里只 6 页有 ⇒ 达不到，保留（正文误删的防线）
    partial_long = [f"{GOOD_ZH}{watermark}" if index < 6 else GOOD_ZH for index in range(10)]
    kept_long = tc.clean_document(partial_long)
    check("长行只在 6/10 页重复时不删（长行门槛 80%）",
          sum("957719200" in page for page in kept_long["pages"]) == 6,
          kept_long["pages"][0][-40:])
    #: 短行同一比例（6/10 ≥ 50%）就该删
    short_mark = "· 内部资料 请勿外传"
    mixed = [f"{GOOD_ZH}{short_mark}" if index < 6 else GOOD_ZH for index in range(10)]
    kept_short = tc.clean_document(mixed)
    check("短版头在 6/10 页重复时被删（短行门槛 50%）",
          all("请勿外传" not in page for page in kept_short["pages"]),
          kept_short["pages"][0][-40:])


# ===========================================================================
# ③ 门禁三档
# ===========================================================================


def case_head_order() -> None:
    print("\n②b 顺序：先剔版头、后重排（实测踩过的坑）")

    #: 真实形态复现（《世界电影》1984-02 实测）：正文被字形簇切成 9 字/行 ⇒ 触发整页
    #: 重排；碎行合并把「)」粘在句号后 ⇒ 行尾不再是句号 ⇒ 重排继续吞下紧随的水印行
    #: ⇒ **粘连行含句号** ⇒ 被 `_PROSE_PUNCT`（正文保护）跳过 ⇒ 一页都数不到。
    watermark = "· ((&)实惠图书之家独家制作 联系方式 旺旺：实惠图书之家 联系QQ：957719200"
    body = "\n".join(["在诸如断代", "、国别", "、样式", "、风格", "、流派",
                      "等方方面面", "作出区分这项工作。", ")", watermark])
    pages = [body for _ in range(8)]
    out = tc.clean_document(pages)
    check("水印行被删（重排前它是独立行，键与长度都稳定）",
          all("957719200" not in page for page in out["pages"]), out["pages"][0][-60:])
    check("正文末句完整（没被水印污染）",
          all("作出区分这项工作。" in page for page in out["pages"]),
          out["pages"][0][-60:])
    check("紧贴水印的花纹行一并删除",
          all("((&)" not in page for page in out["pages"]), out["pages"][0][-60:])

    #: 反向验证：**先重排再剔** 必然一个都删不掉 —— 钉住三步顺序，防有人图省事合并回去
    wrong = [tc.clean_page(text) for text in pages]
    check("先重排后剔版头 ⇒ 一个也删不掉（这就是顺序不能换的证据）",
          tc.drop_running_heads(wrong)[1] == [],
          [line[-40:] for line in wrong[0].split("\n")][-2:])

    #: 装饰残留的边界（单个判据直测：放进页面会被它自己的版头统计干扰）
    check("花纹行认得出来（实测两种 OCR 误认形态）",
          tc._is_decoration("· ((&)") and tc._is_decoration("一((+/"),
          ["· ((&)", "一((+/"])
    check("含句号的短行不是花纹", not tc._is_decoration("一。"))
    check("页码/编号不是花纹（避免把编号当花纹删）", not tc._is_decoration("232"))
    check("稍长的正文行不是花纹", not tc._is_decoration("这行有八个字了"))

    #: ★实测形态（《世界电影》1984-05 第 5 页）：水印框被 OCR 认成两个句号 ⇒ 整行成了
    #: 「。。实惠图书之家…」⇒ 「含句读就是正文」的保护规则把十页里唯一这一页放行，
    #: 于是九页删干净、剩一页水印一直躺在 fts5 里。
    dotted = "。。" + watermark.lstrip("· ((&)")
    dotted_out = tc.clean_document([f"{GOOD_ZH}{dotted}" for _ in range(8)])
    check("行首被认成句号的水印也删得掉（先剥行首符号再判正文）",
          all("957719200" not in page for page in dotted_out["pages"]),
          dotted_out["pages"][0][-50:])
    #: 反过来：句读出现在句中/行尾的短行仍算正文，不许当版头删
    prose_out = tc.clean_document([f"{GOOD_ZH}。。他说。\n" for _ in range(8)])
    check("剥掉行首符号后仍有句读的短行照旧受保护",
          all("他说。" in page for page in prose_out["pages"]),
          prose_out["pages"][0][-40:])


def case_judge() -> None:
    print("\n③ 质量门禁（ok/suspect/reject，且不拿 CJK 占比判）")

    check("正常中文页 ok", tc.judge_page(GOOD_ZH)["verdict"] == "ok",
          tc.judge_page(GOOD_ZH))
    english = tc.judge_page(GOOD_EN)
    check("英文剧本 ok（CJK 占比 0 不参与判定）", english["verdict"] == "ok", english)
    check("英文页的 cjkRatio 仍如实报 0", english["metrics"]["cjkRatio"] == 0.0)

    junk = tc.judge_page(JUNK_PAGE * 2)
    check("纯符号页 reject", junk["verdict"] == "reject", junk)
    check("纯符号页给出理由", bool(junk["reasons"]), junk["reasons"])

    replacement = tc.judge_page("正常文字" + "\ufffd" * 40 + "继续")
    check("替换字符(U+FFFD)页 reject", replacement["verdict"] == "reject", replacement)
    check("理由里点名非法字符", any("非法字符" in reason for reason in replacement["reasons"]),
          replacement["reasons"])

    repeat = tc.judge_page("啊" * 300)
    check("整页同一字刷屏 reject", repeat["verdict"] == "reject", repeat)

    #: ★反例：中文短句重复是合法形态，不能因为"去重少"被 reject（会删内容）
    short_repeat = tc.judge_page("第一页正文。" * 40)
    check("短句重复不判 reject（只至多 suspect）",
          short_repeat["verdict"] != "reject", short_repeat)

    #: ★真跑露头（2026-11-04 批次）：英文页被**页页判存疑**。一个 1500 字的英文页去重字符
    #: 只有 30–60 个（26 个字母天然反复），去重比恒在 2–6% ⇒ 永远低于 12% 的门槛。
    #: 实证《Monsters, Inc.》改编本 170 页全进 page_audit，把真正要看的坏页埋掉。
    long_en = ("Sulley leaned close to his little friend and said goodbye to Boo. "
               "The kid just stared at Mike, then laughed at the joke again. ") * 12
    en_page = tc.judge_page(long_en)
    check("英文长页不因「去重比」判存疑（那条判据是中文口径）",
          en_page["verdict"] == "ok",
          (en_page["verdict"], en_page["reasons"], en_page["metrics"]["distinctRatio"]))
    check("英文长页确实落在旧判据的射程内（否则这条反证是空转）",
          en_page["metrics"]["cjkRatio"] < tc.DISTINCT_CJK_MIN
          and en_page["metrics"]["distinctRatio"] < tc.DISTINCT_SUSPECT, en_page["metrics"])
    #: 另一头：中文重复页不能被一起放宽（那是这条判据真正要抓的形态）
    zh_repeat = tc.judge_page("他把信放在桌上，然后转身走了出去。" * 30)
    check("中文重复页仍判存疑（放宽只针对非中文页）",
          zh_repeat["verdict"] == "suspect"
          and any("去重比" in reason for reason in zh_repeat["reasons"]), zh_repeat)

    check("空白页 ok 且不报理由", tc.judge_page("   ")["verdict"] == "ok")
    check("乱码判定不看 CJK 占比（英文页不会因 cjkRatio=0 被拦）",
          not any("CJK" in reason or "中文" in reason for reason in english["reasons"]))

    #: ★反例（真跑露头）：764 字正常评论页里混进一处表格线，被整页判废 ⇒ 764 字正文
    #: 一个字都没进 fts5。假阳性丢内容比留噪声严重得多，所以判据改成看**占比**。
    one_junk = tc.judge_page(GOOD_ZH + "\n" + "—" * 10 + "\n" + GOOD_ZH)
    check("长页里零星一处符号段不判废（假阳性会整页丢内容）",
          one_junk["verdict"] != "reject", one_junk)
    check("但那一处必须报出来（能人工看见）",
          any("符号连续段" in reason for reason in one_junk["reasons"]),
          one_junk["reasons"])
    check("量得出符号段有多长（正则不能带捕获组）",
          one_junk["metrics"]["maxRun"] >= 8, one_junk["metrics"])

    #: 另一头：整块被认成符号仍然要拦
    block = tc.judge_page("正常开头。" + "|" * 260 + "正常结尾。")
    check("整块被认成符号（连续段超长）仍 reject", block["verdict"] == "reject", block)
    dense = tc.judge_page("正常文字" + "|||" * 40 + "结尾")
    check("符号段占比高时 reject", dense["verdict"] == "reject", dense)


# ===========================================================================
# ④ 接进切块：reject 页不产块
# ===========================================================================


def case_page_chunks() -> None:
    print("\n④ 门禁接进切块（坏页不写块，但留审计）")

    pages = [(1, GOOD_ZH), (2, JUNK_PAGE * 3), (3, GOOD_ZH)]
    packed = oi._page_chunks(pages)
    bodies = [body for _page, _no, body in packed["chunks"]]
    check("坏页一个块都没写", all("|||" not in body for body in bodies), bodies[:2])
    check("好页的正文照常进块", any("凌云把皮箱放下去" in body for body in bodies))
    check("rejected 计数如实", packed["rejected"] == 1, packed["rejected"])
    check("坏页留下审计行", [row["page"] for row in packed["audit"] if row["verdict"] == "reject"]
          == [2], packed["audit"])
    check("审计里有理由与预览",
          all(row["reasons"] and row["preview"] for row in packed["audit"]),
          packed["audit"])

    #: 清洗必须在切块之前生效（否则块里仍带页眉）
    with_head = [(index + 1, f"·12{index}·\n{GOOD_ZH}") for index in range(4)]
    packed_head = oi._page_chunks(with_head)
    check("页眉不会进块（清洗在切块之前）",
          all("·12" not in body for _p, _n, body in packed_head["chunks"]),
          packed_head["chunks"][0][2][:40])

    #: suspect 页要保留内容（只标记不删）
    odd = "他说\n雨水\n忽然\n" * 20  #: 大量两字行 ⇒ 行均短，属"读序可疑"
    verdict = tc.judge_page(odd)
    check("行均过短只判 suspect 不是 reject", verdict["verdict"] == "suspect", verdict)
    packed_odd = oi._page_chunks([(1, GOOD_ZH), (2, odd)])
    check("suspect 页仍然入库（只标记不删）",
          any("忽然" in body for _p, _n, body in packed_odd["chunks"]))


# ===========================================================================
# ⑤ 整份隔离与落库
# ===========================================================================


def item_of(name: str) -> dict:
    return {"path": f"G:\\语料\\{name}", "name": name, "extension": ".pdf",
            "size": 100, "modifiedNs": 1, "signature": f"sig-{name}"}


def case_status_and_save() -> None:
    print("\n⑤ 整份隔离与落库（quarantine 不写块，但算已处理）")

    check("试跑未完 ⇒ partial",
          oi._status_of({"pages": 10, "scanned": 3, "judged": 3, "rejected": 3})
          == oi.STATUS_PARTIAL)
    check("reject 过半且页数够 ⇒ quarantine",
          oi._status_of({"pages": 4, "scanned": 4, "judged": 4, "rejected": 3})
          == oi.STATUS_QUARANTINE)
    check("reject 少量 ⇒ 正常 ocr",
          oi._status_of({"pages": 10, "scanned": 10, "judged": 10, "rejected": 2})
          == oi.STATUS_OCR)
    check("页数不足 3 页时不整份隔离（单页样本没有统计意义）",
          oi._status_of({"pages": 2, "scanned": 2, "judged": 2, "rejected": 2})
          == oi.STATUS_OCR)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB)
        sl.LIBRARY_ROOT = root
        sl.METHODS_DB = root / "curated.sqlite"
        sl.OCR_DB = root / "ocr" / "sidecar.sqlite"
        try:
            bad = item_of("全是符号.pdf")
            oi._save(bad, {"pages": 3, "scanned": 3, "blank": 0, "judged": 3,
                           "rejected": 3, "suspect": 0, "seconds": 1.0,
                           "chunks": [(1, 1, "本不该落库的文字")],
                           "audit": [{"page": 1, "verdict": "reject", "reasons": "纯符号",
                                      "chars": 40, "preview": "|||||"}]},
                     dpi=150, engine="t")
            side = sqlite3.connect(sl.OCR_DB)
            try:
                status, gated = side.execute(
                    "SELECT status, coalesce(gated, 0) FROM documents").fetchone()
                chunks = side.execute("SELECT count(*) FROM chunks").fetchone()[0]
                audit = side.execute("SELECT page, verdict, reasons FROM page_audit").fetchall()
            finally:
                side.close()
            check("隔离文档 status=quarantine", status == oi.STATUS_QUARANTINE, status)
            check("落库时标 gated=1（旧口径产物能被 --audit 认出来）", gated == 1, gated)
            check("隔离文档一个块都没写", chunks == 0, chunks)
            check("隔离文档有审计行（否则库外看不出这页为什么消失）",
                  audit and audit[0][1] == "reject", audit)

            good = item_of("正常.pdf")
            oi._save(good, {"pages": 3, "scanned": 3, "blank": 0, "judged": 3,
                            "rejected": 0, "suspect": 0, "seconds": 1.0,
                            "chunks": [(1, 1, "正常正文")],
                            "audit": [{"page": 2, "verdict": "suspect", "reasons": "存疑",
                                       "chars": 30, "preview": "..."}]},
                     dpi=150, engine="t")
            side = sqlite3.connect(sl.OCR_DB)
            try:
                written = side.execute("SELECT count(*) FROM chunks").fetchone()[0]
            finally:
                side.close()
            check("正常文档照常写块", written == 1, written)

            #: 重跑同一份必须清掉旧审计，否则审计行会累积成假象
            oi._save(bad, {"pages": 3, "scanned": 3, "blank": 0, "judged": 3,
                           "rejected": 3, "suspect": 0, "seconds": 1.0, "chunks": [],
                           "audit": [{"page": 1, "verdict": "reject", "reasons": "又跑一次",
                                      "chars": 40, "preview": "|||"}]},
                     dpi=150, engine="t")
            side = sqlite3.connect(sl.OCR_DB)
            try:
                rows = side.execute(
                    "SELECT reasons FROM page_audit WHERE name = ?", ("全是符号.pdf",)).fetchall()
            finally:
                side.close()
            check("重跑清掉旧审计（不累积）",
                  [row[0] for row in rows] == ["又跑一次"], rows)

            done = oi._done_signatures()
            check("quarantine 算已处理（不会每次批处理都重烧 CPU）",
                  done.get(bad["path"]) == bad["signature"], done)
            check("正常文档也在已处理里", done.get(good["path"]) == good["signature"])

            report = oi._audit_report(40, "reject")
            check("--audit 能报出隔离清单",
                  [row["name"] for row in report["quarantined"]] == ["全是符号.pdf"],
                  report["quarantined"])
            check("--audit 能报出隔离文档的坏页数",
                  report["quarantined"][0]["rejectedPages"] == 1, report["quarantined"])
            check("--audit 能报出坏页明细", len(report["pages"]) == 1, report["pages"])
        finally:
            sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB = saved


# ===========================================================================
# ⑥ 阈值自洽
# ===========================================================================


def case_thresholds() -> None:
    print("\n⑥ 阈值与判据自洽")

    check("存疑阈值比拒绝阈值更宽（顺序不许写反）",
          tc.VALID_SUSPECT > tc.VALID_REJECT, (tc.VALID_SUSPECT, tc.VALID_REJECT))
    check("去重判据有字数下限（防短页误杀）", tc.DISTINCT_MIN_CHARS >= 60)
    check("页眉上限小于常见正文行长度", tc.DEFAULT_RUNNING_HEAD_MAX <= 40)
    check("三档名字与落库状态对得上",
          oi.STATUS_OCR in tc.JUDGEMENTS or oi.STATUS_OCR == "ocr",
          (tc.JUDGEMENTS, oi.STATUS_OCR))
    #: 反向验证：把「有效字符」判据换成 \w 会让它恒为 1（初版踩过），这里钉住
    metrics = tc.page_metrics("◆◆◆◆❖❖❖❖✦✦✦✦ 正常文字")
    check("有效字符占比不是恒 1（装饰符号确实算无效）",
          metrics["validRatio"] < 1.0, metrics)


def case_scan_receipt() -> None:
    print("\n⑦ 扫描回执与隔离接通（真跑路径）")

    #: ``_status_of`` 读 ``outcome["judged"]``；扫描函数漏透传 ⇒ `int(None or 0) >= 3`
    #: 恒假 ⇒ **整份隔离永远不会触发**，坏书照旧写块入库。而单测直接喂 outcome 是
    #: 看不出来的，所以这里扫源码里的回执字典字面量，钉住键必须存在。
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(oi))
    receipts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            #: 只认「扫描回执」：``_stats()`` 也有 pages/chunks，但它的扫描计数叫
            #: ``scannedPages`` —— 用 scanned 区分，免得把统计回执混进来。
            if "pages" in keys and "chunks" in keys and "scanned" in keys:
                receipts.append(keys)
    #: 五条回执：PDF／docx／图片／**纯文本**（``_ocr_text``）／**并行分段的合并**
    #: （``_merge_parts``—— 大 PDF 会切成几段并行跑，合并回执漏 judged 的话，
    #: 分段跑的书「整份隔离」就静默失效，而单段跑的书仍正常，最难发现）。
    check("扫描回执 5 条（PDF／docx／图片／纯文本／分段合并）",
          len(receipts) == 5, len(receipts))
    for keys in receipts:
        check("回执带 judged", "judged" in keys, keys)
        check("回执带 droppedHeads（版头删了多少行要能核对）",
              "droppedHeads" in keys, keys)

    #: 隔离的算术：只有 judged 撑着，缺了就恒不触发
    whole = {"scanned": 9, "pages": 9, "rejected": 9, "judged": 9}
    check("隔离用的是 judged 而不是 scanned",
          oi._status_of(whole) == oi.STATUS_QUARANTINE)
    check("judged 缺失时不许判隔离（宁可入库也不误隔离）",
          oi._status_of({"scanned": 9, "pages": 9, "rejected": 9}) == oi.STATUS_OCR)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    case_merge()
    case_running_heads()
    case_head_order()
    case_judge()
    case_page_chunks()
    case_status_and_save()
    case_thresholds()
    case_scan_receipt()
    print(f"\nSUMMARY: {CHECKS - FAILED}/{CHECKS} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
