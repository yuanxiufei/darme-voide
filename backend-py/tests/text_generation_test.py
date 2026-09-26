"""文本生成服务自检（``app/services/text_generation.py``）。

只测**纯逻辑**部分（提示词拼装、JSON 剥壳、本地规则拆分器、防幻觉校验、打标过滤）——
``generate_text`` 那条链要 DB 配置 + HTTP，由上层路由的冒烟用例覆盖。

重点是那个 **本地规则拆分器**：8 张词表 + 相邻子句合并，是「AI 失败也要有输出」的兜底，
而且它的输出会被写进角色卡（用户看得见）。

运行::

    ./.venv/Scripts/python.exe tests/text_generation_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="textgen_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import text_generation as tg  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    # ================= clean_json_string =================
    check(
        "json: 剥 ```json 围栏",
        tg.clean_json_string('```json\n{"a":1}\n```') == '{"a":1}',
        tg.clean_json_string('```json\n{"a":1}\n```'),
    )
    check("json: 剥无语言标记的围栏", tg.clean_json_string('```\n{"a":1}\n```') == '{"a":1}')
    check(
        "json: 前后有解释文字时截取首个 { 到末个 }",
        tg.clean_json_string('好的，结果如下：\n{"a":1}\n希望有帮助') == '{"a":1}',
        tg.clean_json_string('好的，结果如下：\n{"a":1}\n希望有帮助'),
    )
    check(
        "json: 嵌套对象时取到最外层的 }",
        tg.clean_json_string('x {"a":{"b":1}} y') == '{"a":{"b":1}}',
        tg.clean_json_string('x {"a":{"b":1}} y'),
    )
    check(
        "json: 没有花括号时原样（仅去空白）",
        tg.clean_json_string("  not json  ") == "not json",
    )

    # ================= parse_sub_shots 的三段错误文案 =================
    shots = tg.parse_sub_shots('{"subShots":[{"shotSize":"全景","cameraMovement":"静止","actionSummary":"内容","visualFocus":"焦点"}]}')
    check("subShots: 正常解析", len(shots) == 1 and shots[0]["shotSize"] == "全景", str(shots)[:80])
    for raw, expected in [
        ("这不是 JSON", "AI 返回的拆分结果不是有效 JSON"),
        ("{}", "AI 拆分结果为空"),
        ('{"subShots":[]}', "AI 拆分结果为空"),
        ('{"subShots":"x"}', "AI 拆分结果为空"),
        ('{"subShots":[{"cameraMovement":"静止"}]}', "子镜头缺少必要字段（shotSize / actionSummary）"),
    ]:
        try:
            tg.parse_sub_shots(raw)
            check(f"subShots: {expected}", False, "未抛错")
        except ValueError as err:
            check(f"subShots: {expected}", str(err) == expected, str(err))

    # ================= 时间轴 DSL 检测 =================
    for text, expected in [
        ("0-3秒 两人对视 <n> 3-6秒 转身走开", True),
        ("画面<n>接第二段", True),
        ("<n/>", True),
        ("<location>客栈</location>", True),
        ("<role>林昭</role>", True),
        ("<voice>低沉</voice>", True),
        ("一段普通的视频提示词", False),
        ("", False),
        ("0- 3 秒", True),
    ]:
        check(f"时间轴 DSL: {text[:22]!r} -> {expected}", tg.has_timeline_dsl(text) is expected, tg.has_timeline_dsl(text))

    # ================= 提示词拼装（纯函数） =================
    action_prompt = tg._build_action_suggestion_prompt({
        "title": "开场", "shotType": "近景", "angle": "俯拍", "atmosphere": "紧张",
        "imagePrompt": "雨夜街头", "action": "回头", "movement": "推镜",
    })
    check(
        "action: 字段顺序 + 结尾指令",
        action_prompt
        == "镜头标题：开场\n景别：近景\n拍摄角度：俯拍\n氛围：紧张\n画面内容：雨夜街头\n"
           "已有动作描述：回头\n已有运镜：推镜\n\n请为以上镜头生成运镜与动作建议。",
        action_prompt,
    )
    check(
        "action: 完全没有信息时用单句兜底",
        tg._build_action_suggestion_prompt({}) == "请为这个镜头生成运镜与动作建议。",
    )
    check(
        "action: imagePrompt 优先于 description",
        tg._build_action_suggestion_prompt({"imagePrompt": "A", "description": "B"}) == "画面内容：A\n\n请为以上镜头生成运镜与动作建议。",
        tg._build_action_suggestion_prompt({"imagePrompt": "A", "description": "B"}),
    )

    split_prompt = tg._build_split_shot_prompt({
        "sceneInfo": {"location": "客栈", "time": "夜晚", "atmosphere": "肃杀"},
        "title": "对峙", "shotType": "全景", "characterNames": ["林昭", "黑衣人"],
        "visualStyle": "水墨", "description": "两人对峙", "dialogue": "你来了",
    })
    check(
        "splitShot: 行序 + 空行 + JSON 模板",
        split_prompt.startswith(
            "场景地点：客栈\n场景时间：夜晚\n场景氛围：肃杀\n镜头标题：对峙\n原始景别：全景\n"
            "出场角色：林昭、黑衣人\n视觉风格：水墨\n原始动作/画面描述：两人对峙\n"
            "对白：你来了（请将对白放入最合适的子镜头，通常是角色说话的中景或近景）\n\n"
            "请将以上镜头拆分为 2-5 个子镜头，输出 JSON：\n"
        )
        and '"subShots":[{"shotSize":"全景"' in split_prompt,
        split_prompt[:120],
    )
    check(
        "splitShot: 无 visualStyle 时默认「电影写实风格」",
        "视觉风格：电影写实风格" in tg._build_split_shot_prompt({}),
    )
    check(
        "splitShot: 无 description 时回退 action",
        "原始动作/画面描述：打了过去" in tg._build_split_shot_prompt({"action": "打了过去"}),
    )

    # 续写：只取末尾 2400 字
    long_text = "前" * 3000 + "尾部"
    prompt = tg._build_continue_script_prompt(long_text)
    check(
        "continue: 只取末尾 2400 字（末尾的『尾部』保留，前面的被截掉）",
        prompt
        == "以下是已有内容：\n\n" + "前" * 2398 + "尾部\n\n请直接继续往下写，输出续写内容。",
        prompt[:20] + " ... " + prompt[-30:],
    )
    check(
        "continue: 短文本原样带入",
        tg._build_continue_script_prompt("短") == "以下是已有内容：\n\n短\n\n请直接继续往下写，输出续写内容。",
        tg._build_continue_script_prompt("短"),
    )

    optimize_prompt = tg._build_optimize_prompt({
        "sceneInfo": {"location": "客栈", "time": "夜", "atmosphere": "肃杀"},
        "characterNames": ["林昭"], "title": "对峙", "shotType": "近景", "movement": "推",
        "atmosphere": "紧张", "visualStyle": "水墨", "description": "对视", "currentPrompt": "  两人对视  ",
    })
    check(
        "optimize: 行序与 currentPrompt 的 trim",
        optimize_prompt.startswith(
            "场景地点：客栈\n场景时间：夜\n场景氛围：肃杀\n出场角色：林昭\n镜头标题：对峙\n"
            "景别：近景\n运镜：推\n氛围：紧张\n视觉风格：水墨\n画面/动作描述：对视\n\n用户当前提示词：\n两人对视\n\n"
            "请输出优化后的视频生成提示词。"
        ),
        optimize_prompt,
    )
    check(
        "optimize: 未填提示词时给明确暗示",
        "用户当前未填写提示词，请根据以上信息生成。" in tg._build_optimize_prompt({}),
    )
    check(
        "optimize: 空串/纯空白也算未填写",
        "用户当前未填写提示词" in tg._build_optimize_prompt({"currentPrompt": "   "}),
    )

    # 拆分视觉信息 / 音色打标的提示词
    check(
        "splitVisuals: few-shot 示例在前，原文在后",
        tg._build_split_visuals_prompt("某角色").startswith("示例 1：")
        and tg._build_split_visuals_prompt("某角色").endswith("现在拆分以下角色描述：\n某角色\n\n请严格按规则只输出 JSON。"),
    )
    check(
        "voiceTag: 有描述时拼上「描述=」，无描述时省略该段",
        tg._build_voice_tag_prompt([{"voiceId": "v1", "voiceName": "少年", "description": ["清亮", "少年感"]}])
        == "请为以下音色打角色类型标签：\n- voice_id=v1，名称=少年，描述=清亮、少年感\n\n标签只能从「旁白 / 主角 / 反派 / 配角」中选择。"
        and "- voice_id=v2，名称=老者\n" in tg._build_voice_tag_prompt([{"voiceId": "v2", "voiceName": "老者", "description": []}]),
    )

    # ================= clean_visual_fragment =================
    check(
        "clean: 前缀循环剥离（穿着一身 -> 去掉「穿着」再去掉「一身」）",
        tg.clean_visual_fragment("穿着一身玄色道袍") == "玄色道袍",
        tg.clean_visual_fragment("穿着一身玄色道袍"),
    )
    check(
        "clean: 前缀 + 后缀同时剥离",
        tg.clean_visual_fragment("长发以简单束带松松挽于脑后") == "简单束带",
        tg.clean_visual_fragment("长发以简单束带松松挽于脑后"),
    )
    check(
        "clean: 逗号分段逐段清洗后重新拼接（不破坏分隔）",
        tg.clean_visual_fragment("身着玄色道袍，腰系玉带，脚蹬云纹靴") == "玄色道袍，玉带，云纹靴",
        tg.clean_visual_fragment("身着玄色道袍，腰系玉带，脚蹬云纹靴"),
    )
    check(
        "clean: 去掉句读符号（。；：、换行）",
        tg.clean_visual_fragment("三尺青锋。") == "三尺青锋",
        tg.clean_visual_fragment("三尺青锋。"),
    )
    check(
        "clean: 全空片段 -> 空串（不残留分隔符）",
        tg.clean_visual_fragment("，,，") == "",
        repr(tg.clean_visual_fragment("，,，")),
    )

    # ================= split_visuals_by_rules（规则引擎） =================
    ruled = tg.split_visuals_by_rules(
        "少年身着朴素古意的青色长衫，长发以简单束带松松挽于脑后，腰间别着一柄三尺青锋长剑，手中把玩着一把白面折扇。"
    )
    check(
        "rules: 服装/武器/首饰各归各类（折扇属随身器物，不输出到任何字段）",
        ruled["weapons"] == "三尺青锋长剑" and ruled["accessories"] == "简单束带"
        and "长衫" in ruled["clothing"] and "折扇" not in str(ruled),
        str(ruled),
    )
    # ⚠️ 去重判据是「已提取的条目里是否含本次命中的关键词」，而**相邻句会被合并**——
    #    所以紧接着重复同一武器时，重复文本会随合并一起进来（原实现如此）。
    check(
        "rules: 同一武器紧邻重复 -> 会被合并进同一片段（去重判据挡不住）",
        tg.split_visuals_by_rules("腰间别着三尺青锋长剑，又见三尺青锋长剑")["weapons"]
        == "三尺青锋长剑，又见三尺青锋长剑",
        tg.split_visuals_by_rules("腰间别着三尺青锋长剑，又见三尺青锋长剑")["weapons"],
    )
    check(
        "rules: 中间隔了杂质句（不合并）时，重复的武器才会被去重丢掉",
        tg.split_visuals_by_rules("腰间别着三尺青锋长剑，他目光冷冽，又见三尺青锋长剑")["weapons"]
        == "三尺青锋长剑",
        tg.split_visuals_by_rules("腰间别着三尺青锋长剑，他目光冷冽，又见三尺青锋长剑")["weapons"],
    )
    check(
        "rules: 武器屏蔽词生效（「出剑」这类动作过程且命中词≤2字时跳过）",
        tg.split_visuals_by_rules("他出剑了")["weapons"] == "",
        repr(tg.split_visuals_by_rules("他出剑了")["weapons"]),
    )
    # `hit.length <= 2` 是豁免条件：命中 3 字词（狼牙棒）时即使同句有「挥剑」也不跳过
    check(
        "rules: 命中屏蔽词但关键词长于 2 字 -> 不跳过",
        "狼牙棒" in tg.split_visuals_by_rules("他挥剑击出狼牙棒")["weapons"],
        repr(tg.split_visuals_by_rules("他挥剑击出狼牙棒")["weapons"]),
    )
    check(
        "rules: 神态/环境杂质句不会被合并进来",
        "沉静" not in tg.split_visuals_by_rules("他沉静如枯木，身着玄色道袍")["clothing"],
        tg.split_visuals_by_rules("他沉静如枯木，身着玄色道袍")["clothing"],
    )
    check(
        "rules: 他类关键词阻止合并（服装后紧跟首饰不会被吞进服装）",
        "项链" not in tg.split_visuals_by_rules("身着玄色道袍，颈间坠着蓝宝石项链")["clothing"],
        tg.split_visuals_by_rules("身着玄色道袍，颈间坠着蓝宝石项链")["clothing"],
    )
    check(
        "rules: 首饰归 accessories 而不是 clothing",
        tg.split_visuals_by_rules("身着玄色道袍，颈间坠着蓝宝石项链")["accessories"] == "蓝宝石项链",
        tg.split_visuals_by_rules("身着玄色道袍，颈间坠着蓝宝石项链")["accessories"],
    )
    check(
        "rules: 描述性子句被合并保留（三尺青锋 + 剑鞘细节）",
        "剑鞘古朴无华" in tg.split_visuals_by_rules(
            "身旁横放一柄三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒"
        )["weapons"],
        tg.split_visuals_by_rules("身旁横放一柄三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒")["weapons"],
    )
    check(
        "rules: 三类都不命中 -> 三个空串（不编造）",
        tg.split_visuals_by_rules("他站在山顶眺望远方") == {"clothing": "", "weapons": "", "accessories": ""},
        tg.split_visuals_by_rules("他站在山顶眺望远方"),
    )
    check(
        "rules: 合并长度上限 34 —— 过长的相邻句不并入",
        "，" not in tg.split_visuals_by_rules("身着玄色道袍，" + "很" * 40)[  "clothing"],
        tg.split_visuals_by_rules("身着玄色道袍，" + "很" * 40)["clothing"],
    )

    # ================= appears_in_source（防幻觉） =================
    check(
        "防幻觉: 片段都在原文 -> True",
        tg.appears_in_source("朴素古意的青色长衫，简单束带", "少年身着朴素古意的青色长衫，长发以简单束带挽起"),
    )
    check(
        "防幻觉: 有一段不在原文 -> False（如「三尺青芒」幻觉）",
        not tg.appears_in_source("三尺青芒", "腰间别着一柄三尺青锋长剑"),
    )
    check(
        "防幻觉: 单字片段（长度<2）判不通过",
        not tg.appears_in_source("剑", "腰间别着一柄三尺青锋长剑"),
    )
    check("防幻觉: 空值 -> False", not tg.appears_in_source("", "任何原文"))
    check("防幻觉: 只有逗号 -> False", not tg.appears_in_source("，，", "任何原文"))

    # ================= filter_voice_role_tags =================
    check(
        "打标: 非法标签被丢弃、空数组的键不输出",
        tg.filter_voice_role_tags({"v1": ["主角", "路人"], "v2": [], "v3": ["配音"]}) == {"v1": ["主角"]},
        str(tg.filter_voice_role_tags({"v1": ["主角", "路人"], "v2": [], "v3": ["配音"]})),
    )
    check(
        "打标: 非数组值 -> 丢弃",
        tg.filter_voice_role_tags({"v1": "主角", "v2": 1, "v3": None}) == {},
    )
    check(
        "打标: 4 个合法标签就是 VOICE_ROLE_TAGS（与前端对齐）",
        tg.VOICE_ROLE_TAGS == ("旁白", "主角", "反派", "配角")
        and tg.filter_voice_role_tags({"v": list(tg.VOICE_ROLE_TAGS)}) == {"v": list(tg.VOICE_ROLE_TAGS)},
    )

    # ================= 提示词常量存在且非空 =================
    for name in (
        "ACTION_SYSTEM_PROMPT", "SPLIT_SYSTEM_PROMPT", "CONTINUE_RAW_SYSTEM_PROMPT",
        "CONTINUE_SCRIPT_SYSTEM_PROMPT", "OPTIMIZE_PROMPT_SYSTEM_PROMPT",
        "OPTIMIZE_TIMELINE_SYSTEM_PROMPT", "SPLIT_VISUALS_SYSTEM_PROMPT",
        "SPLIT_VISUALS_EXAMPLES", "VOICE_TAG_SYSTEM_PROMPT",
    ):
        value = getattr(tg, name)
        check(f"常量: {name} 非空且是 str", isinstance(value, str) and len(value) > 20, len(value))

    # 词表条数（**逐项比对交给 route_parity_test 的漂移守卫**，这里只做体量哨兵）
    check(
        "词表: 五张词表的条目数（改词表会改变拆分结果）",
        (
            len(tg.SPLIT_CLOTHING_KEYWORDS),
            len(tg.SPLIT_WEAPON_KEYWORDS),
            len(tg.SPLIT_ACCESSORY_KEYWORDS),
            len(tg.SPLIT_WEAPON_BANNED),
            len(tg.SPLIT_REDUNDANT_PREFIXES),
        )
        == (38, 32, 29, 7, 56),
        str((
            len(tg.SPLIT_CLOTHING_KEYWORDS), len(tg.SPLIT_WEAPON_KEYWORDS), len(tg.SPLIT_ACCESSORY_KEYWORDS),
            len(tg.SPLIT_WEAPON_BANNED), len(tg.SPLIT_REDUNDANT_PREFIXES),
        )),
    )

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
