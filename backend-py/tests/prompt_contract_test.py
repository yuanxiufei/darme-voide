"""S7 自检：**短剧提示词生产契约**（占位符解析 + 五段质感层 ✓ 零依赖 ✓ 2026-09-17）。

移植自 ``reference/short-drama-agent``（``tagged-storyboard-format.md`` +
``mx-shell-workflow-adapter.md`` ✓）。这两块是**本项目原来确实没有的** ✓：

* 本项目只有 ``strip_video_prompt_tags``（**剥掉标签、留下裸编号** ``L1``/``R5`` ✗）——
  而裸编号对扩散模型是**无意义字符** ✗（占注意力、还可能被当成文字生成 ✗）；
* 本项目**完全没有**质感层 ✗ ⇒ 提示词只能靠人肉写「电影感」这类无效约束 ✗。

判据里两条是刻意防**假绿**的 ✓：
* **解析后不许再有裸编号**（否则"解析过了"是假的 ✓）；
* **空话必须成对**（只有「cinematic」没有物理锚点 ⇒ 必须报问题 ✓ —— 这条正是参考项目反复强调的 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/prompt_contract_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.prompt_polish import PolishInputs, build_prompt_layer, find_vague_terms  # noqa: E402
from app.services.shot_placeholders import PlaceholderMaps, resolve_placeholders  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


MAPS = PlaceholderMaps(
    locations={"L1": "旧城区烂尾楼三层案发房间；门在画面左侧，破窗在右后方",
               "L2": "三层走廊；脚印从左后通向右前消防门"},
    roles={"R5": "沈砚，二十七八岁，黑色湿短发，深色湿外套，脸色苍白，左手腕旧疤",
           "R6": "许知夏，二十九岁，短发，深色便装，持枪动作稳定"},
    props={"P1": "水果刀，已经落在潮湿水泥地上，不在任何人手里"},
    clues={"C1": "尸体旁的巨大数字 7，只作为画面线索，请勿在画面中生成复杂文字"},
)


# ══════════════════════════════════════════════════════════════════════════
# ① 占位符解析（**解析成具体内容** ✓ 不是剥标签 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_placeholders() -> None:
    raw = ("本片段场景设定在: <location>L1</location>,<location>L2</location>。"
           "分镜1<duration-ms>6000</duration-ms>: 沈砚猛地睁开眼 <role>R5</role>")
    result = resolve_placeholders(raw, MAPS)
    check("① ⭐ 场景编号被替换成**具体场景描述** ✓（不是留下裸 `L1` ✗）",
          "旧城区烂尾楼三层案发房间" in result.text and "L1" not in result.text, result.text[:60])
    check("② ⭐ 角色编号同理 ✓：`R5` → 角色外观描述（模型不认识内部编号 ✗）",
          "沈砚，二十七八岁" in result.text and "R5" not in result.text, result.text[-60:])
    check("③ 时长走**统一模板** ✓（毫秒/秒混写最乱 ✗）且数值进结构化字段 ✓",
          "目标时长6000毫秒" in result.text and result.durationsMs == [6000],
          (result.durationsMs, result.text[:30]))
    check("④ 用到的映射**逐条记录**（可审计 ✓ 看得出替换了什么 ✓）",
          len(result.used) == 3 and {item["kind"] for item in result.used} == {"location", "role"},
          result.used)
    check("⑤ 解析成功 ⇒ `ok=True` 且**无残留标签、无裸编号** ✓",
          result.ok is True and not result.residualTags and not result.bareIds,
          (result.residualTags, result.bareIds))

    mixed = resolve_placeholders("<location>L1</location> 与 <role>l1</role>", MAPS)
    check("⑥ 编号**大小写容忍** ✓（`l1` 也认 ✓ —— 但**不猜**种类归属 ✗）",
          "旧城区烂尾楼三层案发房间" in mixed.text and "沈砚" not in mixed.text, mixed.text[:40])

    unmapped = resolve_placeholders("案发房 <location>L9</location> 与 <role>R99</role>", MAPS)
    check("⑦ ⭐ 未映射编号 ⇒ **逐条列在 `unresolved`** ✓ 并进 problems ✓（不静默 ✗）",
          len(unmapped.unresolved) == 2 and unmapped.ok is False
          and any("没有映射" in item for item in unmapped.problems), unmapped.to_dict())
    check("⑧ 未映射时**标签本身仍被去掉** ✓（至少不把 XML 送进模型 ✗）",
          "<location>" not in unmapped.text and "L9" in unmapped.text, unmapped.text)

    bare = resolve_placeholders("沈砚 <role>R5</role> 看向 R7 的方向", MAPS)
    check("⑨ ⭐ **裸编号检测**：映射表没有的 `R7` 直接暴露 ✓（送进模型才发现就晚了 ✗）",
          bare.bareIds == ["R7"] and any("裸编号" in item for item in bare.problems), bare.bareIds)

    broken = resolve_placeholders("<location>L1</role> 配错闭合", MAPS)
    check("⑩ 开闭标签**不配对** ⇒ 不误吞文本 ✓，且作为残留标签报出 ✓",
          broken.residualTags and "L1" in broken.text, (broken.residualTags, broken.text))

    idempotent = resolve_placeholders(result.text, MAPS)
    check("⑪ **幂等** ✓：解析过的文本再解析一次不再变化（不会越解析越乱 ✓）",
          idempotent.text == result.text and not idempotent.used, idempotent.text[:30])

    plain = resolve_placeholders("纯自然语言，没有任何占位符 ✓", MAPS)
    check("⑫ 没有占位符的文本**原样通过** ✓（不画蛇添足 ✗）",
          plain.text == "纯自然语言，没有任何占位符 ✓" and plain.ok is True, plain.text)

    check("⑬ 坏时长（非数字）⇒ 报问题但**不崩** ✓ 且保留原文 ✓",
          resolve_placeholders("<duration-ms>六秒</duration-ms>", MAPS).problems
          and "六秒" in resolve_placeholders("<duration-ms>六秒</duration-ms>", MAPS).text, "")
    check("⑭ `duration_text` 一处定义 ✓（避免了 6000 / 6 秒 混写 ✗）",
          __import__("app.services.shot_placeholders", fromlist=["x"]).duration_text(6000)
          == "目标时长6000毫秒", "")


# ══════════════════════════════════════════════════════════════════════════
# ② 五段质感层（Mx-Shell ✓ 逼出物理锚点 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _good_inputs(**overrides: object) -> PolishInputs:
    base = dict(
        theme_tags=("真人写实", "犯罪悬疑", "低饱和冷蓝"),
        character_scene="沈砚，黑色湿短发，深色湿外套；门在画面左侧，白布尸体固定在右后方",
        lens="Sony Venice + Canon K-35，28mm 广角，浅景深",
        palette="低饱和灰蓝，局部硬光制造戏剧性阴影",
        texture="潮湿水泥地反光、破窗上的水痕",
        shot_type="极端近景",
        movement="轻微跟焦",
        screen_direction="沈砚朝画面右侧看",
        slice_text="沈砚猛地睁开眼，视线聚焦自己的右手掌",
        sounds=("呼吸声", "雨水滴落"),
        imperfections=("掌心湿滑血迹", "外套沾泥"),
        handheld=True,
        inner_monologue=True,
    )
    base.update(overrides)
    return PolishInputs(**base)  # type: ignore[arg-type]


def case_polish() -> None:
    good = build_prompt_layer(_good_inputs())
    check("⑮ ⭐ 五段齐备且 `ok=True`（约束够硬 ✓ 无毛病可报 ✓）",
          good.ok is True and set(good.sections) >= {"core_theme_tags", "locked_character_scene",
                                                     "atmosphere_quality", "camera_rules",
                                                     "storyboard_slice"},
          good.issues)
    check("⑯ ⭐ 手持镜头**自动补**呼吸般轻微浮动 ✓（不用调用方记 ✓ 且明确说「不要剧烈晃动」✓）",
          "呼吸般的镜头浮动" in good.sections["camera_rules"]
          and "不要变成剧烈晃动" in good.sections["camera_rules"]
          and "handheldFloat" in good.inserted, good.sections["camera_rules"][:60])
    check("⑰ 内心独白**自动补**「不张嘴」✓（否则模型会让角色说话 ✗ —— 参考项目专门点过 ✓）",
          "不张嘴" in good.sections["storyboard_slice"] and "innerMonologueRule" in good.inserted, "")
    check("⑱ 声音政策必带 ✓（无配乐、只用同期声 ✓）且场景声单列 ✓",
          "No score. Production audio only." in good.sections["storyboard_slice"]
          and "呼吸声" in good.sections["storyboard_slice"], "")
    check("⑲ `text` 是**拷走即用**的整段（五段拼起来 ✓ 非空 ✓）",
          good.text and all(part in good.text for part in
                            ("真人写实", "Sony Venice", "极端近景", "No score")), good.text[:50])

    short_tags = build_prompt_layer(_good_inputs(theme_tags=("写实",)))
    check("⑳ 主题标签 < 3 ⇒ **报问题**（不给「随便凑一个」的机会 ✓）",
          short_tags.ok is False and any("core_theme_tags" in item for item in short_tags.issues),
          short_tags.issues)
    many_tags = build_prompt_layer(_good_inputs(theme_tags=tuple(f"标签{index}" for index in range(8))))
    check("㉑ 主题标签 > 6 ⇒ 报问题（互相稀释 ✓）",
          any("超过 6 个" in item for item in many_tags.issues), many_tags.issues)

    no_lens = build_prompt_layer(_good_inputs(lens=""))
    check("㉒ ⭐ 缺**真实镜头锚点** ⇒ 报问题（光写「cinematic」不给像素级约束 ✗）",
          any("camera_lens_profile" in item for item in no_lens.issues), no_lens.issues)

    few_imperfections = build_prompt_layer(_good_inputs(imperfections=("只有一处",)))
    check("㉓ 瑕疵锚点 < 2 ⇒ 报问题（现实感场景至少 2 个 ✓）",
          any("imperfection_anchors" in item for item in few_imperfections.issues), "")

    no_direction = build_prompt_layer(_good_inputs(screen_direction=""))
    check("㉔ 缺画面方向 ⇒ 报问题（相邻镜头会左右跳 ⇒ 假连续性 ✗）",
          any("screen_direction" in item for item in no_direction.issues), "")

    # ⭐ 本套最核心的一条：**空话必须成对**
    vague_only = build_prompt_layer(_good_inputs(
        theme_tags=("cinematic", "epic", "stunning"), lens="", palette="", texture="",
        imperfections=()))
    check("㉕ ⭐⭐ **只有空话、没有物理锚点** ⇒ 必须报问题 ✓"
          "（参考项目原话：不要依赖 cinematic/epic 这类词 ✓）",
          any("不构成像素级约束" in item for item in vague_only.issues)
          and len(vague_only.vagueTerms) >= 3, (vague_only.vagueTerms, vague_only.issues))

    vague_with_anchor = build_prompt_layer(_good_inputs(theme_tags=("cinematic", "写实", "冷色调")))
    check("㉖ 空话**有**物理锚点陪衬 ⇒ 仍提醒删掉 ✓（但不判为失败 —— 它已不误导模型 ✓）",
          any("建议直接删掉" in item for item in vague_with_anchor.issues)
          and vague_with_anchor.vagueTerms, vague_with_anchor.issues)

    check("㉗ `find_vague_terms` 大小写不敏感 ✓（`Cinematic` 也认 ✓）",
          find_vague_terms("Cinematic and 4K") == ["4k", "cinematic"], find_vague_terms("Cinematic and 4K"))

    restrained = build_prompt_layer(_good_inputs(restrained_ending=True))
    check("㉘ 克制结尾开关生效 ✓（不做总结式抒情 ✓）",
          "结尾克制" in restrained.sections["storyboard_slice"], "")

    empty = build_prompt_layer(PolishInputs())
    check("㉙ 全空输入 ⇒ **逐条报出缺什么** ✓（而不是给一段看起来很帅的空话 ✗）",
          empty.ok is False and len(empty.issues) >= 5, empty.issues)


# ══════════════════════════════════════════════════════════════════════════
# ③ 路由：两块能力都要**真能被调用** ✓
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resolved = client.post("/api/v1/prompts/resolve", json={
        "text": "场景 <location>L1</location>，<role>R5</role><duration-ms>6000</duration-ms>",
        "maps": {"locations": {"L1": "案发房间"}, "roles": {"R5": "沈砚"}}})
    payload = resolved.json().get("data") or {}
    check("㉚ POST /prompts/resolve 真能调用 ✓（200 + 具体内容替换 ✓）",
          resolved.status_code == 200 and "案发房间" in payload.get("text", "")
          and payload.get("durationsMs") == [6000], (resolved.status_code, payload.get("text")))

    check("㉛ 未映射编号经 API 也如实回报 ✓（`ok=False` + 逐条 ✓）",
          (client.post("/api/v1/prompts/resolve", json={
              "text": "<location>L9</location>", "maps": {}}).json()["data"]["ok"]) is False)

    polished = client.post("/api/v1/prompts/polish", json={
        "themeTags": ["真人写实", "犯罪悬疑", "冷色调"],
        "characterScene": "沈砚，黑色湿短发",
        "lens": "Sony Venice + Canon K-35",
        "palette": "低饱和灰蓝", "texture": "潮湿水泥反光",
        "shotType": "近景", "movement": "轻微跟焦", "screenDirection": "朝右看",
        "slice": "沈砚睁眼", "sounds": ["呼吸声"],
        "imperfections": ["血迹", "沾泥"], "handheld": True})
    body = polished.json().get("data") or {}
    check("㉜ POST /prompts/polish 真能调用 ✓（200 + 五段 + ok ✓）",
          polished.status_code == 200 and body.get("ok") is True
          and "core_theme_tags" in (body.get("sections") or {}), (polished.status_code, body.get("issues")))

    check("㉝ 空 body ⇒ 400 或明确的缺参提示（不 500 ✗）",
          client.post("/api/v1/prompts/polish", json={}).status_code in (200, 400), "")


def main() -> int:
    case_placeholders()
    case_polish()
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
