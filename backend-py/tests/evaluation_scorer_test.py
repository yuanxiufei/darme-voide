"""S6 自检：评测闭环的**确定性部分**（``services/evaluation/`` —— types/catalog/scorer）。

打分器是评测闭环的地基：分数不可比，整条「提示词优化」链路就毫无意义。因此本测试盯三件事：

1. **位精对齐 JS**：``Math.round(n*10)/10`` 与 ``toFixed(0)`` 的后半舍入方式与 Python 内置
   ``round`` **不同**（JS 半数向 +∞ / 远离零，Python 是银行家舍入）—— 专门构造一个
   ``2.5`` 的用例把差异钉住；
2. **`isFilled` 的五分类**：``0`` 算填了、``False`` **也算填了**（TS 的 ``typeof false``
   落到最后的 ``return true``）、``[]``/``'  '``/``None`` 不算；
3. **真实基准文件**：4 个 case JSON 的 ``statement``/``rubric`` 键必须与本模块记录的契约
   一致（TS 侧靠类型系统，运行期没人校验）。

运行::

    ./.venv/Scripts/python.exe tests/evaluation_scorer_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="eval_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.evaluation import catalog as cat  # noqa: E402
from app.agent.evaluation import scorer as sc  # noqa: E402
from app.agent.evaluation import types as et  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


STORYBOARD_RUBRIC = {
    "minShots": 2, "maxShots": 3, "requiredFields": ["title", "action", "duration"],
    "videoPromptTags": ["<location>"], "durationRange": [5, 10], "titleLengthRange": [2, 8],
}


def _shot(**overrides) -> dict:
    shot = {"title": "开场", "action": "推门", "duration": 8, "video_prompt": "<location>客栈",
            "scene_id": 1, "character_ids": [1]}
    shot.update(overrides)
    return shot


def main() -> int:  # noqa: C901
    # ================= 契约与真实基准文件 =================
    check("契约: LITERAL_KINDS 与 AGENT_BY_KIND 的键一致（加了一种忘了另一种就红）",
          set(et.LITERAL_KINDS) == set(et.AGENT_BY_KIND) == set(et.STATEMENT_KEYS_BY_KIND)
          and set(et.AGENT_BY_KIND) == set(et.RUBRIC_KEYS_BY_KIND),
          (et.LITERAL_KINDS, list(et.AGENT_BY_KIND)))
    check("契约: 四种 kind 的映射值就是 Agent 类型",
          et.AGENT_BY_KIND == {"storyboard": "storyboard_breaker", "extractor": "extractor",
                               "script_rewriter": "script_rewriter",
                               "voice_assigner": "voice_assigner"})

    metas = cat.list_benchmark_cases()
    check("目录: 基准目录里 4 个 case 都能列出（只有 id/kind/agentType 三个键）",
          len(metas) == 4
          and all(set(m) == {"id", "kind", "agentType"} for m in metas),
          metas)
    check("目录: agentType 由 kind 映射得出（不是文件里的字段说了算）",
          all(m["agentType"] == et.AGENT_BY_KIND[m["kind"]] for m in metas), metas)

    shapes_ok = True
    detail = []
    for meta in metas:
        case = cat.load_case_by_id(meta["id"])
        assert case is not None
        kind = case["kind"]
        ok = (set(case) == set(et.CASE_KEYS)
              and set(case["statement"]) == set(et.STATEMENT_KEYS_BY_KIND[kind])
              and set(case["rubric"]) == set(et.RUBRIC_KEYS_BY_KIND[kind])
              and bool(case["id"]) and case["id"] == meta["id"])
        if not ok:
            shapes_ok = False
            detail.append((kind, sorted(case), sorted(case["statement"]), sorted(case["rubric"])))
    check("契约: 四个真实 case 的 statement/rubric 键与记录完全一致", shapes_ok, detail)
    check("目录: 未知 id -> None（不抛）", cat.load_case_by_id("没有这个 case") is None)

    # 目录定位与容错
    empty_dir = Path(tempfile.mkdtemp(prefix="ev_empty_"))
    os.environ["BENCHMARKS_DIR"] = str(empty_dir)
    check("目录: BENCHMARKS_DIR 覆盖生效 + 目录为空 -> 空列表",
          cat.benchmarks_dir() == empty_dir and cat.list_benchmark_cases() == [])
    (empty_dir / "bogus.json").write_text(json.dumps({"id": "x", "kind": "未知kind"}),
                                          encoding="utf-8")
    check("目录: **未知 kind 的 case 被跳过**（AGENT_BY_KIND 里没有）",
          cat.list_benchmark_cases() == [])
    (empty_dir / "bad.json").write_text("{不是 JSON", encoding="utf-8")
    raised = ""
    try:
        cat.list_benchmark_cases()
    except Exception as err:  # noqa: BLE001
        raised = str(err)
    check("目录: 坏 JSON **会抛**（与 TS 的 JSON.parse 一致，路由层没兜底 -> 500）",
          raised != "", raised)
    os.environ.pop("BENCHMARKS_DIR", None)

    # ================= 分镜评分 =================
    legal = {"characterIds": {1, 2}, "sceneIds": {1, 2}}
    perfect = sc.score_storyboards([_shot(), _shot(title="反应", character_ids=[2])],
                                   STORYBOARD_RUBRIC, legal)
    check("分镜: 满分输入 -> 100 分、6 个维度、满分权重 10/30/20/10/10/20",
          perfect["total"] == 100
          and [d["max"] for d in perfect["dimensions"]] == [10, 30, 20, 10, 10, 20]
          and [d["name"] for d in perfect["dimensions"]]
          == ["镜头数量合理", "核心字段完整率", "video_prompt 标记", "duration 范围",
              "title 长度", "角色/场景绑定合法"],
          (perfect["total"], [d["score"] for d in perfect["dimensions"]]))
    check("分镜: 回执 caseId 是**空串**、kind 固定 storyboard（caseId 由 evaluator 填）",
          perfect["caseId"] == "" and perfect["kind"] == "storyboard")

    empty = sc.score_storyboards([], STORYBOARD_RUBRIC, legal)
    check("分镜: 零分镜 -> 0 分；第 1 维 0、第 2 维 detail 是「无分镜」",
          empty["total"] == 0 and empty["dimensions"][0]["score"] == 0
          and empty["dimensions"][1]["detail"] == "无分镜",
          [d["detail"] for d in empty["dimensions"]][:2])
    check("分镜: 镜头数超区间 -> 第 1 维给 5 分（不是 0）",
          sc.score_storyboards([_shot()], STORYBOARD_RUBRIC,
                               legal)["dimensions"][0]["score"] == 5)
    in_range = sc.score_storyboards([_shot(), _shot()], STORYBOARD_RUBRIC,
                                    legal)["dimensions"][0]
    check("分镜: 镜头数在区间内 -> 10 分，detail 写明期望区间",
          in_range["score"] == 10 and "期望 2~3 之间" in in_range["detail"], in_range)

    missing = sc.score_storyboards([_shot(action=""), _shot(title="  ", action="跑")],
                                   STORYBOARD_RUBRIC, legal)
    detail2 = missing["dimensions"][1]
    check("分镜: 常缺字段会被点名（含 `平均覆盖率` 与 `常缺字段`）",
          "常缺字段" in detail2["detail"] and "平均覆盖率" in detail2["detail"]
          and detail2["score"] < 30,
          (detail2["score"], detail2["detail"]))

    # isFilled 的五分类（用 requiredFields 把语义逼出来）
    fill_rubric = {**STORYBOARD_RUBRIC, "requiredFields": ["a", "b", "c", "d", "e"]}
    fill = sc.score_storyboards(
        [{"a": 0, "b": False, "c": [], "d": "  ", "e": None, "title": "t", "duration": 8,
          "video_prompt": "<location>x"}],
        {**fill_rubric, "durationRange": [5, 10], "titleLengthRange": [1, 5]},
        {"characterIds": set(), "sceneIds": set()},
    )["dimensions"][1]["detail"]
    check("分镜: **isFilled 五分类** —— 0 与 False 算填了，[] / '  ' / None 不算（2/5 = 40%）",
          "平均覆盖率 40%" in fill, fill)

    tags = sc.score_storyboards(
        [_shot(video_prompt="<location>客栈"), _shot(video_prompt="没有标记")],
        {**STORYBOARD_RUBRIC, "minShots": 2, "maxShots": 2}, legal)["dimensions"][2]
    check("分镜: video_prompt 标记按**镜头平均**覆盖（一半 -> 10/20）",
          tags["score"] == 10 and "平均覆盖 50%" in tags["detail"], tags)

    duration = sc.score_storyboards(
        [_shot(duration=5), _shot(duration=10), _shot(duration=4), _shot(duration=None)],
        {**STORYBOARD_RUBRIC, "minShots": 4, "maxShots": 4}, legal)["dimensions"][3]
    check("分镜: duration **含上下界**、None 不算（2/4 -> 5/10）",
          duration["score"] == 5 and duration["detail"].startswith("2/4"), duration)

    title = sc.score_storyboards(
        [_shot(title="标题"), _shot(title=" 短 "), _shot(title="太长的标题超过八字")],
        {**STORYBOARD_RUBRIC, "minShots": 3, "maxShots": 3}, legal)["dimensions"][4]
    # ⚠️ 这条正是 trim 的证据：`" 短 "` trim 后只有 **1** 字（下限 2）⇒ 只有 1/3 合格；
    #    若**不** trim，它会算 3 字而过关，分数就变成 6.7 了。
    check("分镜: title 长度**先 trim 再比**（1/3 -> 3.3/10，不 trim 的话会是 2/3）",
          title["score"] == 3.3 and title["detail"].startswith("1/3"), title)

    binding = sc.score_storyboards(
        [_shot(scene_id=None, character_ids=[]),          # 都缺 -> 合法
         _shot(scene_id=99),                              # 场景捏造
         _shot(character_ids=[1, 99])],                   # 角色捏造
        {**STORYBOARD_RUBRIC, "minShots": 3, "maxShots": 3}, legal)["dimensions"][5]
    check("分镜: 绑定合法性 —— 空/None 算过，捏造 ID 不算（1/3 -> 6.7/20）",
          binding["score"] == 6.7 and binding["detail"].startswith("1/3"), binding)

    # 位精：`toFixed(0)` 半数远离零 vs Python 银行家舍入
    many_fields = [f"f{i}" for i in range(40)]
    tiny = sc.score_storyboards(
        [{**{f: "" for f in many_fields}, "f0": "x"}],
        {**STORYBOARD_RUBRIC, "minShots": 1, "maxShots": 1, "requiredFields": many_fields},
        legal)
    check("位精: `toFixed(0)` —— 2.5% 应是 **3%**（Python 银行家舍入会给 2%，故必须用 js_round）",
          "平均覆盖率 3%" in tiny["dimensions"][1]["detail"]
          and tiny["dimensions"][1]["score"] == 0.8,
          (tiny["dimensions"][1]["detail"], tiny["dimensions"][1]["score"]))

    # ================= 提取评分 =================
    rubric_ex = {"goldenCharacters": ["林昭", "阿晚"], "goldenScenes": [{"location": "客栈", "time": "夜"}],
                 "minAppearanceLength": 10, "minPromptLength": 20}
    chars = [{"name": "林昭", "appearance": "青衫少年，腰间挂一枚玉坠，眉目清冷"},
             {"name": " 阿晚 ", "appearance": "红衣女子，手持长鞭，眼神锐利"}]
    scenes = [{"location": "客栈", "prompt": "客栈内景，烛火摇曳，木桌斑驳"},
              {"location": "客栈", "prompt": "重复的地点不会抬高召回率"}]
    perfect_ex = sc.score_extraction(chars, scenes, rubric_ex)
    check("提取: 满分输入 -> 100 分、权重 35/25/15/10/15",
          perfect_ex["total"] == 100
          and [d["max"] for d in perfect_ex["dimensions"]] == [35, 25, 15, 10, 15],
          (perfect_ex["total"], [d["score"] for d in perfect_ex["dimensions"]]))
    check("提取: 名字 trim 后匹配（' 阿晚 ' 算命中）+ 地点去重",
          perfect_ex["dimensions"][0]["score"] == 35
          and perfect_ex["dimensions"][3]["score"] == 10,
          [d["detail"] for d in perfect_ex["dimensions"]])

    miss = sc.score_extraction([{"name": "林昭", "appearance": "很短"}],
                               [{"location": "客栈", "prompt": "x"}], rubric_ex)
    check("提取: 漏提/多提/短外貌的 detail 文案（漏提点名、无多提写「无」、外貌不足 0 分）",
          "漏提：阿晚" in miss["dimensions"][0]["detail"]
          and "多提：无" in miss["dimensions"][1]["detail"]
          and "0/1 个角色外貌描述 ≥ 10 字" in miss["dimensions"][4]["detail"]
          and miss["dimensions"][4]["score"] == 0,
          [d["detail"] for d in miss["dimensions"]])

    dup = sc.score_extraction([{"name": "林昭", "appearance": "一" * 12},
                               {"name": "林昭", "appearance": "一" * 12}],
                              [], {"goldenCharacters": ["林昭"], "goldenScenes": [],
                                   "minAppearanceLength": 10, "minPromptLength": 5})
    check("提取: 角色名去重（重复名不会拉低精确率：1/1 -> 25 分）",
          dup["dimensions"][1]["score"] == 25 and dup["dimensions"][1]["detail"].startswith("1/1"),
          dup["dimensions"][1])
    check("提取: 外貌分母是**角色条数**（重复名按 2 条算），空数组 -> 0 分不炸",
          dup["dimensions"][4]["detail"].startswith("2/2")
          and sc.score_extraction([], [], {"goldenCharacters": [], "goldenScenes": [],
                                           "minAppearanceLength": 1,
                                           "minPromptLength": 1})["total"] == 0)

    # ================= 剧本改写评分 =================
    rubric_sw = {"minScenes": 2, "forbiddenCameraWords": ["特写", "推镜"]}
    script = ("## S01 | 内景 · 客栈 | 夜\n\n"
              "林昭：（沉声）你终于来了。\n\n"
              "## S02 | 外景 · 长街 | 晨\n\n"
              "阿晚：走。\n")
    script_padded = script + "旁白：" + "剧" * 120
    perfect_sw = sc.score_script_rewrite(script_padded, rubric_sw)
    check("剧本: 满分输入 -> 100 分、权重 20/25/15/20/20",
          perfect_sw["total"] == 100
          and [d["max"] for d in perfect_sw["dimensions"]] == [20, 25, 15, 20, 20],
          (perfect_sw["total"], [d["score"] for d in perfect_sw["dimensions"]]))
    check("剧本: 长度三档（0 -> 0、<100 -> 10、>=100 -> 20）",
          sc.score_script_rewrite("", rubric_sw)["dimensions"][0]["score"] == 0
          and sc.score_script_rewrite("短剧本", rubric_sw)["dimensions"][0]["score"] == 10
          and sc.score_script_rewrite("字" * 100, rubric_sw)["dimensions"][0]["score"] == 20,
          [sc.score_script_rewrite(x, rubric_sw)["dimensions"][0]["score"]
           for x in ("", "短剧本", "字" * 100)])
    check("剧本: 场景头数量不足按比例计分（1/2 -> 12.5）",
          sc.score_script_rewrite("## S01 | 内景 · 客栈 | 夜", rubric_sw)["dimensions"][1]["score"]
          == 12.5,
          sc.score_script_rewrite("## S01 | 内景 · 客栈 | 夜", rubric_sw)["dimensions"][1])
    check("剧本: 场景头**缺第三段**不算（正则要求两个竖线）",
          sc.score_script_rewrite("## S01 | 内景 · 客栈", rubric_sw)["dimensions"][1]["score"] == 0)
    check("剧本: 编号不连续扣分（S01/S03 -> 1/2 连续）",
          sc.score_script_rewrite("## S01 | a | b\n## S03 | c | d", rubric_sw)["dimensions"][2]
          ["detail"].startswith("1/2"),
          sc.score_script_rewrite("## S01 | a | b\n## S03 | c | d",
                                  rubric_sw)["dimensions"][2]["detail"])
    check("剧本: 对白行识别（含「（动作）台词」；`#`/`|` 开头的行不算）",
          sc.score_script_rewrite("林昭：（笑）你来了。", rubric_sw)["dimensions"][3]["score"] == 20
          and sc.score_script_rewrite("# 标题 | 表格", rubric_sw)["dimensions"][3]["score"] == 0)
    forbidden = sc.score_script_rewrite("## S01 | a | b\n特写。", rubric_sw)["dimensions"][4]
    check("剧本: 出现违禁镜头语言 -> 第 5 维 0 分并点名",
          forbidden["score"] == 0 and "出现违禁词：特写" in forbidden["detail"], forbidden)

    # ================= 音色分配评分 =================
    rubric_vo = {"legalVoiceIds": ["v1", "v2"], "requireReason": True}
    assignments = [{"character_id": 1, "voice_id": "v1", "reason": "少年感"},
                   {"character_id": 2, "voice_id": "v2", "reason": "清冷"}]
    perfect_vo = sc.score_voice_assignment(assignments, rubric_vo, {"characterIds": {1, 2}})
    check("音色: 满分输入 -> 100 分、权重 40/30/15/15",
          perfect_vo["total"] == 100
          and [d["max"] for d in perfect_vo["dimensions"]] == [40, 30, 15, 15],
          (perfect_vo["total"], [d["score"] for d in perfect_vo["dimensions"]]))
    check("音色: requireReason=False -> 第 3 维直接满分并注明「跳过」",
          sc.score_voice_assignment([], {"legalVoiceIds": [], "requireReason": False},
                                    {"characterIds": set()})["dimensions"][2]
          == {"name": "分配理由说明", "score": 15, "max": 15, "detail": "不要求理由（跳过）"})
    check("音色: 非法音色与缺 voice_id 都算不合法（1/3 -> 10）",
          sc.score_voice_assignment(
              [{"character_id": 1, "voice_id": "v1", "reason": "r"},
               {"character_id": 2, "voice_id": "v9", "reason": "r"},
               {"character_id": 2, "voice_id": None, "reason": "r"}],
              rubric_vo, {"characterIds": {1, 2}})["dimensions"][1]["score"] == 10)
    clamped = sc.score_voice_assignment(
        [{"character_id": 9, "voice_id": "v1"}, {"character_id": 9, "voice_id": "v2"}],
        {"legalVoiceIds": ["v1", "v2"], "requireReason": False}, {"characterIds": set()})
    check("音色: 第 4 维比率为**负**时夹紧到 0（重复 + 全非法 -> 0 分，不是负分）",
          clamped["dimensions"][3]["score"] == 0
          and "重复 1 条" in clamped["dimensions"][3]["detail"],
          clamped["dimensions"][3])
    half = sc.score_voice_assignment(
        [{"character_id": 1, "voice_id": "v1"}, {"character_id": 1, "voice_id": "v2"}],
        {"legalVoiceIds": ["v1", "v2"], "requireReason": False}, {"characterIds": {1}})
    check("音色: 重复但合法 -> (2-1)/2 = 50% -> 7.5 分",
          half["dimensions"][3]["score"] == 7.5, half["dimensions"][3])
    check("音色: 四个评分器的 kind 各自正确、caseId 都是空串",
          [r["kind"] for r in (perfect, perfect_ex, perfect_sw, perfect_vo)]
          == ["storyboard", "extractor", "script_rewriter", "voice_assigner"]
          and all(r["caseId"] == "" for r in (perfect, perfect_ex, perfect_sw, perfect_vo)))

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail3 in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail3!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
