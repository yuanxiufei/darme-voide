"""场次结构三层守卫：解析器（纯函数）／库内派生索引（A 侧）／项目剧本事实（B 侧）。

钉住的三件事
============
1. **解析器不猜**：认不出边界就 ``strategy=none`` ＋ ``no_source=missing`` ＋ 置信度 0，
   而不是按空行切出一堆假场次（假结构比没结构更坏：它看起来很成功）。
2. **偏移对原文成立**：``char_start``/``char_end`` 切片必须正好是那一场的正文，不许含下一场
   标题、不许被 ``\\r\\n`` 带偏 —— 这是"按场取原文"的唯一依据。
3. **两侧归属不混**：A 侧派生可重建、B 侧项目事实带指纹。B 侧 ``stale`` 必须用**当前正文现算**
   的指纹判定，不能读 ``episodes.script_hash`` 那一列（列是缓存，直接改库时它是陈的 ⇒
   最该报 stale 的时候不报）。

运行::

    ./.venv/Scripts/python.exe tests/scene_parser_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

from sqlalchemy import create_engine, insert, select, update  # noqa: E402

from app.core.models import characters, episodes, metadata  # noqa: E402
from app.services import scene_index, script_catalog, script_library  # noqa: E402
from app.services.scene_parser import (  # noqa: E402
    SCENES_SCHEMA_VERSION,
    SCENE_STRATEGIES,
    extract_cast,
    parse_scenes,
)
from app.services.script_scenes import list_episode_scenes, parse_episode_scenes  # noqa: E402

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


MD = (
    "片名：试验\n"
    "\n"
    "## S01 | 内景 · 大厅 | 夜\n"
    "凌云：走。\n"
    "（若云 上）\n"
    "\n"
    "## S02 | 外景 · 街道 | 黄昏\n"
    "若云：好。\n"
)


# ===== 一、解析器 =====
def case_strategies() -> None:
    print("\n[解析器] 四种识别路线")
    parsed = parse_scenes(MD)
    check("md-heading：strategy/置信度", parsed["strategy"] == "md-heading"
          and parsed["confidence"] == 0.95, parsed["strategy"])
    check("md-heading：场号/内外景/地点/时段",
          [(item["scene_no"], item["int_ext"], item["location"], item["time"])
           for item in parsed["scenes"]]
          == [(1, "INT", "大厅", "夜"), (2, "EXT", "街道", "黄昏")],
          [(item["scene_no"], item["int_ext"], item["location"], item["time"])
           for item in parsed["scenes"]])
    check("schema 版本随输出带上", parsed["schema_version"] == SCENES_SCHEMA_VERSION)
    check("strategy 在册（含 inline）", all(
        parse_scenes(text)["strategy"] in SCENE_STRATEGIES
        for text in (MD, "场景1：内景 客厅 夜\n凌云：走。\n",
                     "INT. X - DAY\n凌云：走。\n", "", "随便写点东西。")
    ) and "inline" in SCENE_STRATEGIES)

    cn = parse_scenes("场景1：内景 客厅 夜\n凌云：走。\n\n场景2：外景 街道 黄昏\n若云：好。\n")
    check("cn-numbered（场景N：）0.9",
          cn["strategy"] == "cn-numbered" and cn["confidence"] == 0.9
          and [item["scene_no"] for item in cn["scenes"]] == [1, 2],
          (cn["strategy"], cn["confidence"]))

    weak = parse_scenes("1. 内景 客厅 夜\n凌云：走。\n\n2. 外景 街道 黄昏\n若云：好。\n")
    check("裸数字序号 ＋ 内外景/时段 ⇒ cn-numbered 0.85",
          weak["strategy"] == "cn-numbered" and weak["confidence"] == 0.85,
          (weak["strategy"], weak["confidence"]))

    slug = parse_scenes("INT. COFFEE SHOP - DAY\n\nBOB: hi\n\nEXT. STREET - NIGHT\n\nBOB: bye\n")
    check("slug（INT./EXT.）0.9 且时段取原文",
          slug["strategy"] == "slug" and slug["confidence"] == 0.9
          and [(item["int_ext"], item["time"]) for item in slug["scenes"]]
          == [("INT", "DAY"), ("EXT", "NIGHT")],
          (slug["strategy"], [(item["int_ext"], item["time"]) for item in slug["scenes"]]))


EP_SCENE = (
    "1-1 空地 日/外\n"
    "叶婉：你们放开他。\n"
    "\n"
    "1-2 叶婉家客厅 日/内\n"
    "叶婉：坐吧。\n"
    "\n"
    "2-1 叶婉家客厅 日/内\n"
    "肖庆元：签了。\n"
)

PLACE_TIME = (
    "纯黑环境 夜内\n"
    "旁白：那一夜。\n"
    "\n"
    "二层客厅 日内\n"
    "苏：你来了。\n"
    "\n"
    "苏写字楼下广场 日外\n"
    "众人散去。\n"
)


HASH_SCENE = (
    "S#1. 치킨 집 앞 (밤)\n"
    "써니\t왜 그래요?\n"
    "\n"
    "S#2. 치킨 집 (밤)\n"
    "도깨비\t뭐가.\n"
)


def case_short_drama_formats() -> None:
    """短剧剧本两种主用写法：集-场号 / 裸地点+时段（2026-11-06 补）。

    库内一批短剧剧本此前一律落到 ``none``、整篇被记成"一场"——那是解析器不认识格式，
    不是这些剧本没有场次。这里把两类写法连同**误判面**一起钉住。
    """
    print("\n[解析器] 短剧格式：集-场号 / 裸地点+时段")

    ep = parse_scenes(EP_SCENE)
    check("ep-scene：strategy/置信度 0.85",
          ep["strategy"] == "ep-scene" and ep["confidence"] == 0.85,
          (ep["strategy"], ep["confidence"]))
    check("ep-scene：场号只取场段（集号不进 scene_no）",
          [(item["scene_no"], item["location"], item["int_ext"], item["time"])
           for item in ep["scenes"]]
          == [(1, "空地", "EXT", "日"), (2, "叶婉家客厅", "INT", "日"), (1, "叶婉家客厅", "INT", "日")],
          [(item["scene_no"], item["location"]) for item in ep["scenes"]])
    check("ep-scene：heading 保留原文整串（集-场号可回查）",
          [item["heading"] for item in ep["scenes"]]
          == ["1-1 空地 日/外", "1-2 叶婉家客厅 日/内", "2-1 叶婉家客厅 日/内"],
          [item["heading"] for item in ep["scenes"]])
    check("ep-scene：集号每集重置须在 notes 里说清",
          any("每集重置" in note for note in ep["notes"]), ep["notes"])

    # ``场1-1 金銮殿 日 内``：带「场」字前缀的集-场号（188-爆款短剧剧本合集 整批如此）。
    # 曾经被 ``场`` 强场号抢走 ⇒ 剩下 "-1 金銮殿" 当地点、集号丢失、场次重复。
    hash_like = parse_scenes(
        "场1-1 金銮殿 日 内\n凌云：参见。\n"
        "场3-2 水池边 日/外\n若云：走。\n"
        "场12 客厅 日内\n两人坐下。\n"
    )
    check("场N-N：集-场号带「场」前缀也认，且场号只取场段",
          [(item["scene_no"], item["location"], item["int_ext"]) for item in hash_like["scenes"]]
          == [(1, "金銮殿", "INT"), (2, "水池边", "EXT"), (12, "客厅", "INT")],
          [(item["scene_no"], item["location"], item["int_ext"]) for item in hash_like["scenes"]])
    check("场N-N：``场12``（没有集段）走 cn-numbered 强场号",
          hash_like["scenes"][-1]["heading"] == "场12 客厅 日内",
          hash_like["scenes"][-1]["heading"])

    # 内外景还有第三种写法：挂在地点尾巴上的「内／外」（``民政局内 日``）
    tail_ie = parse_scenes(
        "1-1 民政局内 日\n凌云：领证。\n"
        "1-2 （回忆）客厅内 夜\n若云：嗯。\n"
        "1-3 郊外 日外\n空空如也。\n"
    )
    check("ep-scene：地点尾字的 内/外 也认内外景（地名不截字）",
          [(item["location"], item["int_ext"]) for item in tail_ie["scenes"]]
          == [("民政局", "INT"), ("（回忆）客厅", "INT"), ("郊外", "EXT")],
          [(item["location"], item["int_ext"]) for item in tail_ie["scenes"]])

    # 第三种主流写法：韩语／英语剧本的 ``S#1. 치킨 집 앞 (밤)``
    hs = parse_scenes(HASH_SCENE)
    check("hash-scene：strategy/置信度 0.85",
          hs["strategy"] == "hash-scene" and hs["confidence"] == 0.85,
          (hs["strategy"], hs["confidence"]))
    check("hash-scene：场号/地点抽对、heading 留原文",
          [(item["scene_no"], item["location"], item["heading"]) for item in hs["scenes"]]
          == [(1, "치킨 집 앞 (밤)", "S#1. 치킨 집 앞 (밤)"),
              (2, "치킨 집 (밤)", "S#2. 치킨 집 (밤)")],
          [(item["scene_no"], item["location"]) for item in hs["scenes"]])
    check("hash-scene：``#1`` 单独起头不认（markdown/脚注的误判面）",
          parse_scenes("#1. 小标题\n正文一行。\n")["strategy"] == "none",
          parse_scenes("#1. 小标题\n正文一行。\n")["strategy"])

    pt = parse_scenes(PLACE_TIME)
    check("place-time：strategy/置信度 0.6",
          pt["strategy"] == "place-time" and pt["confidence"] == 0.6,
          (pt["strategy"], pt["confidence"]))
    check("place-time：地点/内外景/时段抽对",
          [(item["location"], item["int_ext"], item["time"]) for item in pt["scenes"]]
          == [("纯黑环境", "INT", "夜"), ("二层客厅", "INT", "日"), ("苏写字楼下广场", "EXT", "日")],
          [(item["location"], item["int_ext"], item["time"]) for item in pt["scenes"]])
    check("place-time：无场号 ⇒ 补生产编号并声明是位置判断",
          all(item["no_source"] == "production-assigned" for item in pt["scenes"])
          and any("place-time" in note for note in pt["notes"]), pt["notes"])

    # 误判面①：单条 place-time 头不认（否则整篇被切出假结构）
    single = parse_scenes("他走进客厅 日内。\n凌云：走。\n")
    check("place-time：单条不认（≥3 个才采纳）", single["strategy"] == "none", single["strategy"])

    # 误判面②：接续/插镜头注记不是新场次
    deny = parse_scenes(
        "2-1 客厅 日/内\nA：一。\n接2-1 客厅 日/内\nB：二。\n插入镜头办公室 日/内\nC：三。\n"
        "2-2 院子 日/外\nD：四。\n"
    )
    check("接续注记（接N-N / 插入镜头）不新增场次",
          [item["scene_no"] for item in deny["scenes"]] == [1, 2],
          [item["heading"] for item in deny["scenes"]])

    # 误判面③：正文里的日期不是集场号
    noise = parse_scenes("2024-1-1 那天他走了。\n凌云：走。\n")
    check("4 位数字开头的日期行不算集场号", noise["strategy"] == "none", noise["strategy"])

    # 压平行：场次头后面直接粘正文（OCR 一页一行）
    flat = parse_scenes(
        "1-1b 日内康郡-乔府-乔圭寝屋【小乔看着乔圭身下的虎皮。小乔：父亲。\n"
        "1-1c日外焉州-康郡-马场（闪回新拍）【马队疾驰。\n"
    )
    check("压平行：heading 截到动作括号前，不带正文",
          flat["strategy"] == "ep-scene"
          and [item["heading"] for item in flat["scenes"]]
          == ["1-1b 日内康郡-乔府-乔圭寝屋", "1-1c日外焉州-康郡-马场（闪回新拍）"],
          (flat["strategy"], [item["heading"] for item in flat["scenes"]]))
    check("压平行：location 不带正文残句",
          all(item["location"] and "小乔" not in item["location"]
              for item in flat["scenes"]),
          [item["location"] for item in flat["scenes"]])


def case_offsets() -> None:
    print("\n[解析器] 偏移对原文成立")
    for label, text in (("LF", MD), ("CRLF", MD.replace("\n", "\r\n"))):
        parsed = parse_scenes(text)
        slices = [text[item["char_start"]:item["char_end"]] for item in parsed["scenes"]]
        check(f"{label}：每场切片不含下一场标题",
              all("## S0" not in piece for piece in slices), slices)
        check(f"{label}：切片内容就是该场正文（首场含若云上场提示）",
              "凌云：走。" in slices[0] and "若云：好。" in slices[1], slices)
        check(f"{label}：末场切到尾", slices[1].strip().endswith("若云：好。"), repr(slices[1]))
    check("空行不切场（不产假场次）",
          len(parse_scenes(MD.replace("凌云：走。", "凌云：走。\n\n\n\n"))["scenes"]) == 2)


def case_no_heading() -> None:
    print("\n[解析器] 认不出就如实标注")
    none = parse_scenes("实力划分。凌云是筑基期。")
    check("无标记 ⇒ 1 场 / strategy=none / 置信度 0",
          none["strategy"] == "none" and none["confidence"] == 0.0
          and len(none["scenes"]) == 1
          and none["scenes"][0]["no_source"] == "missing"
          and none["scenes"][0]["scene_no"] is None,
          (none["strategy"], none["confidence"], len(none["scenes"])))
    check("无标记 ⇒ notes 明说不要据此断言只有一场",
          any("不要" in note and "只有一场" in note for note in none["notes"]), none["notes"])
    check("空正文 ⇒ 0 场（不是 1 场空场）", parse_scenes("")["scenes"] == []
          and parse_scenes("  \n\n ")["scenes"] == [])
    check("语义字段一律留空（不拿首句糊弄）",
          all(item["core_event"] is None and item["scene_tail"] is None
              for item in parse_scenes(MD)["scenes"]))

    anon = parse_scenes("## 内景 · 客厅 | 夜\n凌云：走。\n\n## 外景 · 街道 | 日\n若云：好。\n")
    check("有内外景无场号 ⇒ slug，且按顺序补生产编号",
          anon["strategy"] == "slug"
          and [(item["scene_no"], item["no_source"]) for item in anon["scenes"]]
          == [(1, "production-assigned"), (2, "production-assigned")],
          (anon["strategy"], [(item["scene_no"], item["no_source"]) for item in anon["scenes"]]))


def case_preamble() -> None:
    print("\n[解析器] 首个场次头之前的内容")
    long_pre = ("人物表：凌云、若云、东离。\n" * 20) + "\n## S01 | 内景 · 大厅 | 夜\n凌云：走。\n"
    parsed = parse_scenes(long_pre)
    check("≥200 字 ⇒ 单列一场（extractor=preamble、scene_no=None）",
          len(parsed["scenes"]) == 2
          and parsed["scenes"][0]["extractor"] == "preamble"
          and parsed["scenes"][0]["scene_no"] is None
          and parsed["scenes"][0]["no_source"] == "missing",
          [(item["extractor"], item["scene_no"]) for item in parsed["scenes"]])
    check("preamble 不被假装成第 0 场（场号不从 0 起）",
          parsed["scenes"][1]["scene_no"] == 1, parsed["scenes"][1]["scene_no"])
    check("preamble 的 0.0 不拉低整篇置信度",
          parsed["confidence"] == 0.95, parsed["confidence"])
    check("preamble 的偏移从 0 起到首个场次头",
          parsed["scenes"][0]["char_start"] == 0
          and long_pre[parsed["scenes"][0]["char_end"]:].lstrip().startswith("## S01"),
          parsed["scenes"][0]["char_end"])

    short = "片名\n## S01 | 内景 · 大厅 | 夜\n凌云：走。\n"
    parsed_short = parse_scenes(short)
    check("<200 字 ⇒ 不单列，只在 notes 说明",
          len(parsed_short["scenes"]) == 1
          and any("未计入任何场次" in note for note in parsed_short["notes"]),
          (len(parsed_short["scenes"]), parsed_short["notes"]))


def case_inline() -> None:
    print("\n[解析器] 压平正文（OCR 常见）")
    flat = ("心迷宫。场景1：内景 丽琴家 日。一只手点燃观音像前的两支蜡烛。丽琴：你在吗？"
            "场景2：外景 山路 日。面包车行驶在山路。")
    parsed = parse_scenes(flat)
    check("按行内标记重切 ⇒ 2 场、strategy=inline",
          parsed["strategy"] == "inline" and len(parsed["scenes"]) == 2,
          (parsed["strategy"], len(parsed["scenes"])))
    check("inline 场次头字段仍正确",
          [(item["scene_no"], item["int_ext"], item["location"], item["time"])
           for item in parsed["scenes"]]
          == [(1, "INT", "丽琴家", "日"), (2, "EXT", "山路", "日")],
          [(item["scene_no"], item["int_ext"], item["location"], item["time"])
           for item in parsed["scenes"]])
    check("inline 的 cast 不因压平而全空（按句读当行看）",
          parsed["scenes"][0]["cast"] == ["丽琴"], parsed["scenes"][0]["cast"])
    check("inline 的 opening 只取第一句（不吞台词）",
          parsed["scenes"][0]["opening"] == "一只手点燃观音像前的两支蜡烛。",
          parsed["scenes"][0]["opening"])
    check("inline 的偏移仍对原文成立",
          flat[parsed["scenes"][0]["char_start"]:parsed["scenes"][0]["char_end"]]
          .startswith("一只手点燃") , parsed["scenes"][0]["char_start"])

    single = parse_scenes("场景1：内景 丽琴家 日。凌云：你在吗？这里还有一个场景标记在说话里。")
    check("只有一个标记的压平文本不启用 inline（不产多场）",
          len(single["scenes"]) == 1, len(single["scenes"]))


def case_cast() -> None:
    print("\n[解析器] 出场角色")
    names = extract_cast("凌云：走。\n云姐摇了摇头：不行。\n司马长风：好。\n旁白：从前。\n"
                         "若云（冷笑）：你说呢？\n（陈姨 上）\n凌云：再走。")
    check("只认语法标记、去重保序、排除旁白",
          names == ["凌云", "司马长风", "若云", "陈姨"], names)
    check(">4 字的动作行（云姐摇了摇头：）不被当名字", "云姐摇了摇头" not in names, names)
    check("known 只放宽长度上限（不用于过滤）",
          extract_cast("镇北侯夫人：好。", known=["镇北侯夫人"]) == ["镇北侯夫人"]
          and extract_cast("镇北侯夫人：好。") == [],
          (extract_cast("镇北侯夫人：好。", known=["镇北侯夫人"]),
           extract_cast("镇北侯夫人：好。")))
    check("known 之外的新名字照样保留（宁可多留）",
          "楚天阔" in extract_cast("凌云：走。\n楚天阔：嗯。", known=["凌云"]),
          extract_cast("凌云：走。\n楚天阔：嗯。", known=["凌云"]))


def case_fence_and_stops() -> None:
    print("\n[解析器] 围栏与非场次小标题")
    fenced = "```\n## S01 | 内景 · 大厅 | 夜\n```\n\n## S02 | 外景 · 街 | 夜\n凌云：走。\n"
    parsed = parse_scenes(fenced)
    check("围栏内的场次头不算场次（只算出 S02）",
          [(item["scene_no"], item["heading"]) for item in parsed["scenes"]]
          == [(2, "## S02 | 外景 · 街 | 夜")],
          [(item["scene_no"], item["heading"]) for item in parsed["scenes"]])
    check("未闭合围栏不切场（宁可少切）",
          parse_scenes("```\n## S01 | 内景 · 大厅 | 夜\n凌云：走。\n")["scenes"][0]["scene_no"] is None)

    flat = "```场景9：内景 示例 夜```。场景1：内景 丽琴家 日。丽琴：你在吗？场景2：外景 山路 日。开车。"
    check("压平文本里围栏内的标记同样不算",
          [item["scene_no"] for item in parse_scenes(flat)["scenes"]] == [1, 2],
          [item["scene_no"] for item in parse_scenes(flat)["scenes"]])

    appendix = ("## S01 | 内景 · 大厅 | 夜\n凌云：走。\n\n## 人物设定\n凌云：男，二十岁。\n")
    parsed_app = parse_scenes(appendix)
    check("非场次小标题本身不成为一场",
          len(parsed_app["scenes"]) == 1, len(parsed_app["scenes"]))
    check("附录不被算进最后一场（opening 不是小标题）",
          parsed_app["scenes"][0]["opening"] is None, parsed_app["scenes"][0]["opening"])
    check("stops 有说明", any("小标题" in note for note in parsed_app["notes"]),
          parsed_app["notes"])


def case_scene_match_adapter() -> None:
    print("\n[适配层] scene_match.script_scene_headings 与解析器同源")
    from app.services.scene_match import script_scene_headings  # noqa: PLC0415

    headings = script_scene_headings(
        "## S01 | 外景 · 民国上海街道 | 黄昏\n正文\n\n## 人物设定\n凌云：男。\n\n"
        "1. 内景 客栈大堂 夜\n凌云：走。\n"
    )
    check("竖线写法 ＋ 裸数字写法都能读出（旧实现只认前一种）",
          [(item["location"], item["time"]) for item in headings]
          == [("民国上海街道", "黄昏"), ("客栈大堂", "夜")], headings)
    check("raw 去掉 md 前缀（老调用方按字面比对）",
          headings[0]["raw"] == "S01 | 外景 · 民国上海街道 | 黄昏", headings[0]["raw"])
    check("非场次小标题（## 人物设定）不再变成假场景（旧实现会回灌一个假缺口）",
          all(item["location"] != "人物设定" for item in headings), headings)
    check("光杆标题（## S01）不产生条目", script_scene_headings("## S01\n凌云：走。\n") == [],
          script_scene_headings("## S01\n凌云：走。\n"))


# ===== 二、A 侧派生索引 =====
def build_methods_db(path: Path) -> None:
    """造一份最小的方法库（列与 ``scene_index.load_text`` 读的**逐列对齐**）。"""
    import sqlite3

    conn = sqlite3.connect(path.as_posix())
    conn.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, pages INTEGER, status TEXT);"
        "CREATE TABLE chunks (document_id INTEGER, page INTEGER, chunk_no INTEGER, text TEXT);"
    )
    conn.execute("INSERT INTO documents (id, path, pages, status) VALUES (1, ?, 2, 'curated')",
                 ["G:\\电影剧本\\EP001_终稿.pdf"])
    conn.execute("INSERT INTO documents (id, path, pages, status) VALUES (2, ?, 1, 'curated')",
                 ["G:\\电影剧本\\EP001X终稿.pdf"])
    #: ⚠️ 故意打乱插入顺序：``chunk_no`` 是**页内**编号，每页都从 1 开始。
    #: 按 chunk_no 排序会得到"所有页的第 1 块"排在一起 —— 正文全乱且不报错。
    rows = [
        (1, 2, 1, "## S02 | 外景 · 街 | 夜\n若云：好。\n"),
        (1, 1, 2, "凌云：走。\n"),
        (1, 1, 1, "## S01 | 内景 · 大厅 | 夜\n"),
        (2, 1, 1, "## S01 | 内景 · 别家 | 夜\n别人：走。\n"),
    ]
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()


def case_index() -> None:
    print("\n[A 侧] 派生索引（临时库夹具）")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        methods = root / "methods.sqlite"
        build_methods_db(methods)
        old_methods, old_ocr, old_scenes = (
            script_library.METHODS_DB, script_library.OCR_DB, scene_index.SCENES_DB)
        script_library.METHODS_DB = methods
        script_library.OCR_DB = root / "nope.sqlite"
        scene_index.SCENES_DB = root / "scenes.sqlite"
        try:
            rel = "EP001_终稿.pdf"
            loaded = scene_index.load_text(rel)
            check("按 page, chunk_no 排序（chunk_no 是页内编号）",
                  loaded["text"].startswith("## S01 | 内景 · 大厅 | 夜"),
                  repr(loaded["text"])[:60])
            check("LIKE 里下划线按字面量处理（不串到 EP001X终稿）",
                  "别家" not in loaded["text"] and "大厅" in loaded["text"],
                  repr(loaded["text"])[:80])
            check("跨页 chunk_no 重来不报假 gap", loaded["gaps"] == [], loaded["gaps"])

            built = scene_index.build_document(rel)
            check("落库：2 场、status=ok", built["scenes"] == 2 and built["status"] == "ok", built)
            again = scene_index.build_document(rel)
            check("正文 sha256 未变 ⇒ 跳过（可反复调用）", again["skipped"] is True, again)

            #: ★解析器一改就必须重算：只比 sha256 会把"解析器换了、正文没换"的旧派生行当最新，
            #: 于是全量重跑变成整批 skipped（看起来"跑完了"，其实一行没动）。
            real_sha = scene_index.PARSER_SHA
            scene_index.PARSER_SHA = "0" * 12
            try:
                stale = scene_index.build_document(rel)
            finally:
                scene_index.PARSER_SHA = real_sha
            check("解析器指纹变了 ⇒ 即使正文未变也重算（不是 skipped）",
                  stale.get("skipped") is False and stale["scenes"] == 2, stale)
            check("重算时把原因写进 notes",
                  any("解析器" in note for note in stale["notes"]), stale["notes"])
            #: 恢复真实指纹后再跑一次：上一步把"假指纹"写进库了 ⇒ 这一份现在算过期，
            #: 必须被认出来并重算（`--all` 断点续跑就是靠这个：没跑完的、过期的都会被捡回来）。
            check("库里留着旧指纹 ⇒ 盘点报过期 1 份",
                  scene_index.scene_report()["staleDocuments"] == 1,
                  scene_index.scene_report())
            healed = scene_index.build_document(rel)
            check("过期份被自动重算、重算后过期归零",
                  healed.get("skipped") is False
                  and scene_index.scene_report()["staleDocuments"] == 0,
                  healed)

            data = scene_index.read_scenes(rel)
            check("读回：场次按原文位置（顺带即场号升序）、页码区间对得上",
                  [(item["sceneNo"], item["pageFrom"]) for item in data["scenes"]]
                  == [(1, 1), (2, 2)],
                  [(item["sceneNo"], item["pageFrom"]) for item in data["scenes"]])
            check("读回：cast 已拆成列表", isinstance(data["scenes"][0]["cast"], list)
                  and data["scenes"][0]["cast"] == ["凌云"], data["scenes"][0]["cast"])

            empty = scene_index.build_document("没有这份.pdf")
            check("索引里查不到 ⇒ status=missing 且说明原因",
                  empty["status"] == "missing" and empty["notes"], empty)

            report = scene_index.scene_report()
            check("盘点：1 份 / 2 场", report["documents"] == 1 and report["scenes"] == 2, report)
        finally:
            script_library.METHODS_DB = old_methods
            script_library.OCR_DB = old_ocr
            scene_index.SCENES_DB = old_scenes


def case_ep_scene_order() -> None:
    """集-场号每集重置 ⇒ ``scene_no`` 不是全篇序。

    按 ``scene_no`` 排序会把「第 2 集第 1 场」插到「第 1 集第 2 场」前面 —— 混排不报错，
    用户只看到场次顺序莫名跳跃。排序必须按原文位置。
    """
    print("\n[A 侧] 集-场号重复时按原文位置读回（不按场号混排）")
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        methods = root / "methods.sqlite"
        conn = sqlite3.connect(methods.as_posix())
        conn.executescript(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, pages INTEGER, status TEXT);"
            "CREATE TABLE chunks (document_id INTEGER, page INTEGER, chunk_no INTEGER, text TEXT);"
        )
        conn.execute("INSERT INTO documents (id, path, pages, status) VALUES (1, ?, 1, 'curated')",
                     ["G:\\电影剧本\\两集短剧.pdf"])
        body = (
            "1-1 空地 日外\n凌云：走。\n"
            "1-2 客厅 夜内\n若云：坐。\n"
            "2-1 空地 日外\n凌云：又来。\n"
            "2-2 客厅 夜内\n若云：嗯。\n"
        )
        conn.execute("INSERT INTO chunks VALUES (1, 1, 1, ?)", [body])
        conn.commit()
        conn.close()

        old_methods, old_ocr, old_scenes = (
            script_library.METHODS_DB, script_library.OCR_DB, scene_index.SCENES_DB)
        script_library.METHODS_DB = methods
        script_library.OCR_DB = root / "nope.sqlite"
        scene_index.SCENES_DB = root / "scenes.sqlite"
        try:
            built = scene_index.build_document("两集短剧.pdf")
            check("集-场号：4 场都切出来、路线是 ep-scene",
                  built["scenes"] == 4 and built["strategy"] == "ep-scene", built)
            heads = [item["heading"] for item in scene_index.read_scenes("两集短剧.pdf")["scenes"]]
            check("读回顺序＝原文顺序（1-1, 1-2, 2-1, 2-2）",
                  heads == ["1-1 空地 日外", "1-2 客厅 夜内", "2-1 空地 日外", "2-2 客厅 夜内"],
                  heads)
        finally:
            script_library.METHODS_DB = old_methods
            script_library.OCR_DB = old_ocr
            scene_index.SCENES_DB = old_scenes


def case_gap_same_page() -> None:
    print("\n[A 侧] 同页缺块要报（跨页不报）")
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        methods = root / "methods.sqlite"
        conn = sqlite3.connect(methods.as_posix())
        conn.executescript(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, pages INTEGER, status TEXT);"
            "CREATE TABLE chunks (document_id INTEGER, page INTEGER, chunk_no INTEGER, text TEXT);"
        )
        conn.execute("INSERT INTO documents VALUES (1, 'a.pdf', 2, 'curated')")
        conn.executemany("INSERT INTO chunks VALUES (?,?,?,?)", [
            (1, 1, 1, "第一页第一块，这里凑够四十个字用来触发重叠检测的阈值判断逻辑。"),
            (1, 1, 3, "第一页第三块（缺了第二块），这里也凑够四十个字以上。"),
            (1, 2, 1, "第二页第一块（页内编号重来，不算缺块），凑够四十个字。"),
        ])
        conn.commit()
        conn.close()
        old_methods, old_ocr = script_library.METHODS_DB, script_library.OCR_DB
        script_library.METHODS_DB = methods
        script_library.OCR_DB = root / "nope.sqlite"
        try:
            loaded = scene_index.load_text("a.pdf")
            check("同页块号跳号 ⇒ 报一处缺口", loaded["gaps"] == [3], loaded["gaps"])
            check("跨页 chunk_no 从 3 回到 1 不算缺口", 1 not in loaded["gaps"], loaded["gaps"])
        finally:
            script_library.METHODS_DB = old_methods
            script_library.OCR_DB = old_ocr


def case_route_scenes() -> None:
    print("\n[路由] GET /knowledge/scenes 三种回执不许混")
    from fastapi import FastAPI  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.routers import knowledge as kn  # noqa: PLC0415

    mini = FastAPI()
    mini.include_router(kn.router)
    client = TestClient(mini)

    old_normalize, old_read = script_catalog.normalize, scene_index.read_scenes
    script_catalog.normalize = lambda path, marker=None: "x.pdf"  # type: ignore[assignment]
    try:
        scene_index.read_scenes = lambda rel: {  # type: ignore[assignment]
            "available": False, "hint": "场次索引还没建，先跑 python -m app.scripts.corpus.scene_index"}
        first = client.get("/api/v1/knowledge/scenes", params={"path": "x.pdf"}).json()
        check("库没建 ⇒ available=False（不许被吞成'这部 0 场'）",
              first["data"]["available"] is False and first["data"]["hint"]
              and first["data"]["total"] == 0, first["data"])

        scene_index.read_scenes = lambda rel: {  # type: ignore[assignment]
            "available": True, "hint": "这一份还没解析过场次（按需生成，不是漏做）",
            "document": None, "scenes": [{"sceneNo": index} for index in range(1, 6)]}
        second = client.get("/api/v1/knowledge/scenes",
                            params={"path": "x.pdf", "page": 2, "limit": 2}).json()
        check("分页：total 是全部场次数、本页按 offset 切",
              second["data"]["total"] == 5 and len(second["data"]["scenes"]) == 2
              and second["data"]["scenes"][0]["sceneNo"] == 3, second["data"])
        check("索引在但没解析过 ⇒ available=True ＋ scenes=[]（与上一种分得开）",
              second["data"].get("hint") and second["data"]["available"] is True,
              second["data"].get("hint"))

        bad = client.get("/api/v1/knowledge/scenes")
        doc_bad = client.get("/api/v1/knowledge/document")
        check("缺必填 path ⇒ 与 /document 同一口径（框架层 422，不返回空表冒充成功）",
              bad.status_code == 422 and bad.status_code == doc_bad.status_code,
              (bad.status_code, doc_bad.status_code))

        script_catalog.normalize = lambda path, marker=None: ""  # type: ignore[assignment]
        worse = client.get("/api/v1/knowledge/scenes", params={"path": "??"}).json()
        check("路径解析不出 ⇒ 400（不猜一份出来）", worse["code"] == 400, worse)
    finally:
        script_catalog.normalize, scene_index.read_scenes = old_normalize, old_read


# ===== 三、B 侧项目事实 =====
def with_temp_db(fn):
    """临时 darme 库（只建 schema，不碰真实 ``data/db/drama.db``）。"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = create_engine("sqlite:///" + (Path(tmp) / "darme.sqlite").as_posix())
        metadata.create_all(engine)
        try:
            fn(engine)
        finally:
            engine.dispose()


def _add_episode(conn, *, script: str | None, content: str | None = None) -> int:
    from app.core.response import now

    ts = now()
    return conn.execute(insert(episodes).values(
        drama_id=1, episode_number=1, title="EP001",
        script_content=script, content=content, created_at=ts, updated_at=ts,
    )).inserted_primary_key[0]


def case_project_scenes(engine) -> None:
    print("\n[B 侧] 项目剧本事实（临时 darme 库）")
    with engine.begin() as conn:
        episode_id = _add_episode(conn, script="实力划分。凌云是筑基期。")
        conn.execute(insert(characters).values(
            drama_id=1, name="司马长风", created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        ))

    with engine.begin() as conn:
        parsed = parse_episode_scenes(conn, episode_id)
    check("无标记正文 ⇒ 1 场 / none / 0.0（如实标注）",
          parsed["scenes"] == 1 and parsed["strategy"] == "none" and parsed["confidence"] == 0.0,
          parsed)
    check("sourceField 指明该按哪个字段切片", parsed["sourceField"] == "scriptContent", parsed)

    with engine.begin() as conn:
        again = parse_episode_scenes(conn, episode_id)
    check("指纹未变 ⇒ skipped（不重复解析）", again["skipped"] is True, again)

    with engine.connect() as conn:
        listed = list_episode_scenes(conn, episode_id)
    check("刚解析完 ⇒ stale=False", listed["stale"] is False and len(listed["scenes"]) == 1, listed)

    #: 直接改库（绕开 ``PUT /episodes``）：``episodes.script_hash`` 那一列会是陈的。
    with engine.begin() as conn:
        conn.execute(update(episodes).where(episodes.c.id == episode_id)
                     .values(script_content="## S01 | 内景 · 大厅 | 夜\n凌云：走。\n"))
        conn.execute(update(episodes).where(episodes.c.id == episode_id)
                     .values(script_hash="陈旧的缓存值"))
    with engine.connect() as conn:
        stale = list_episode_scenes(conn, episode_id)
    check("正文改了（script_hash 列是陈的）⇒ 仍报 stale=True",
          stale["stale"] is True, (stale["stale"], stale["parsedHash"], stale["scriptHash"]))
    check("stale 时 hint 明说别拿它做分镜",
          stale["hint"] and "过期" in stale["hint"], stale["hint"])

    with engine.begin() as conn:
        fixed = parse_episode_scenes(conn, episode_id, force=True)
    check("重解析后：md-heading 0.95、1 场",
          fixed["scenes"] == 1 and fixed["strategy"] == "md-heading"
          and fixed["confidence"] == 0.95, fixed)
    with engine.connect() as conn:
        fresh = list_episode_scenes(conn, episode_id)
    check("重解析后 stale=False", fresh["stale"] is False, fresh)

    #: 覆盖写：重解析不许把旧行留着（同一集两套场次比没有场次更坏）。
    with engine.begin() as conn:
        conn.execute(update(episodes).where(episodes.c.id == episode_id)
                     .values(script_content="## S01 | 内景 · 大厅 | 夜\n凌云：走。\n"
                                            "\n## S02 | 外景 · 街 | 夜\n若云：好。\n"))
        parse_episode_scenes(conn, episode_id)
    with engine.connect() as conn:
        grown = list_episode_scenes(conn, episode_id)
    check("覆盖写不残留旧场次", [item["sceneNo"] for item in grown["scenes"]] == [1, 2],
          [item["sceneNo"] for item in grown["scenes"]])

    #: preamble 场的 ``scene_no`` 是 NULL —— 按场号排会把它甩到最后，正文顺序就断了。
    with engine.begin() as conn:
        conn.execute(update(episodes).where(episodes.c.id == episode_id)
                     .values(script_content=("人物表：凌云、若云、东离。\n" * 20)
                             + "\n## S01 | 内景 · 大厅 | 夜\n凌云：走。\n"))
        parse_episode_scenes(conn, episode_id)
    with engine.connect() as conn:
        pre = list_episode_scenes(conn, episode_id)
    check("preamble 场排在首位（按插库顺序，不按场号）",
          len(pre["scenes"]) == 2 and pre["scenes"][0]["extractor"] == "preamble"
          and pre["scenes"][0]["sceneNo"] is None,
          [(item["sceneNo"], item["extractor"]) for item in pre["scenes"]])

    #: known_cast 来自项目角色表：5 字名在没有词典时会被长度卡掉。
    with engine.begin() as conn:
        conn.execute(update(episodes).where(episodes.c.id == episode_id)
                     .values(script_content="## S01 | 内景 · 大厅 | 夜\n司马长风：走。\n"))
        parse_episode_scenes(conn, episode_id)
    with engine.connect() as conn:
        cast_rows = list_episode_scenes(conn, episode_id)
    check("项目角色表放宽名字长度（司马长风 出现在 cast）",
          cast_rows["scenes"][0]["cast"] == ["司马长风"], cast_rows["scenes"][0]["cast"])

    with engine.begin() as conn:
        missing = parse_episode_scenes(conn, 99999)
        missing_list = list_episode_scenes(conn, 99999)
    check("不存在的剧集 ⇒ available=False（不臆造空场次）",
          missing["available"] is False and missing_list["available"] is False,
          (missing, missing_list))

    with engine.begin() as conn:
        blank = _add_episode(conn, script=None, content=None)
        blank_parsed = parse_episode_scenes(conn, blank)
    check("没有正文 ⇒ 0 场 + 说明（不是 1 场空场）",
          blank_parsed["scenes"] == 0 and "没有剧本正文" in (blank_parsed["hint"] or ""),
          blank_parsed)

    with engine.begin() as conn:
        legacy = _add_episode(conn, script=None, content="## S01 | 内景 · 大厅 | 夜\n凌云：走。\n")
        legacy_parsed = parse_episode_scenes(conn, legacy)
    check("script_content 为空 ⇒ 回退 content 且 sourceField=content",
          legacy_parsed["sourceField"] == "content" and legacy_parsed["scenes"] == 1,
          legacy_parsed)


def main() -> int:
    case_strategies()
    case_short_drama_formats()
    case_offsets()
    case_no_heading()
    case_preamble()
    case_inline()
    case_cast()
    case_fence_and_stops()
    case_scene_match_adapter()
    case_index()
    case_ep_scene_order()
    case_gap_same_page()
    case_route_scenes()
    with_temp_db(case_project_scenes)

    print(f"\nSUMMARY: {CHECKS - FAILED}/{CHECKS} passed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main())
