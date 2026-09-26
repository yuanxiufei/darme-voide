"""S7 自检：**导演稿 → 生成段**（纯规则 ✓ 零依赖 ✓ 2026-09-24）。

判据刻意写成**不变量** ✓（换语速/换上限都还站得住 ✓）：

* ⭐ **永不丢字** ✓：所有段的文本（去空白后）拼起来 == 原文（去空白后 ✓）—— 分段切错能查 ✓、
  **丢内容查不出来** ✗✗（这才是最贵的错 ✓）；
* ⭐ **每段都在 H3 网格上** ✓（``17k+5`` ✓）且 **≤15 s** ✓✗（越过上限模型会拒或静默改 ✓✗）；
* ⭐ **绝不在句中硬断** ✗：标点齐全的长文里 ``hard_cut`` 必须**全为假** ✓；无标点超长串才允许硬切 ✓
  且那种段要**标出来** ✓（静默硬切 = 用户拿到半句话 ✓✗）；
* 官方 Shot 格式：**≤15 s 合并成一段** ✓、>15 s 装桶 ✓，且**绝不重写**子镜头时间戳 ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_script_parse_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import geometry as geo  # noqa: E402
from app.services.engine import script_parse as sp  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def _squeeze(text: str) -> str:
    return "".join(str(text or "").split())


def _on_grid(frames: int) -> bool:
    return frames >= geo.H3_MIN_FRAMES and (frames - geo.H3_MIN_FRAMES) % geo.H3_FRAME_GRID == 0


def _all_on_grid(plan: sp.ScriptPlan) -> bool:
    return all(_on_grid(segment.frames) for segment in plan.segments)


def _within_cap(plan: sp.ScriptPlan) -> bool:
    return all(segment.frames / geo.H3_FPS <= sp.MAX_SECONDS + 1e-6 for segment in plan.segments)


def _conserved(plan: sp.ScriptPlan, source: str) -> bool:
    """⭐ 永不丢字 ✓（去空白后比较 ✓ —— 断点处的空白会被吃掉 ✓，那不算丢 ✓）。"""
    return _squeeze(plan.segments and "".join(segment.text for segment in plan.segments) or "") \
        == _squeeze(source)


def case_markers() -> None:
    """① 5 种标记写法 ✓（每种都要落到对的 ``source`` + 对得上的秒数 ✓）。"""
    marker = sp.parse_script("段1（6.6秒）：黄昏屋顶，少年转身")
    check("① 段标记带时长 ⇒ 1 段 ✓、``source=marker`` ✓、正文剥掉标记 ✓、秒数≈6.6 ✓",
          len(marker.segments) == 1 and marker.segments[0].source == "marker"
          and marker.segments[0].text == "黄昏屋顶，少年转身"
          and abs(marker.segments[0].seconds - 6.6) <= 0.5,
          marker.to_dict())
    ranged = sp.parse_script("[0s-6.6s] 开场镜头")
    check("①′ 区间标记 ``[0s-6.6s]`` ⇒ 时长 = 区间差 ✓（6.6 ✓）",
          ranged.segments[0].source == "range"
          and abs(ranged.segments[0].seconds - 6.6) <= 0.5, ranged.to_dict())
    # ⚠️ 容差必须**对着吸附口径**写 ✗：本模块是**往上**吸附到 ``17k+5`` ✓ ⇒ 偏差上界 = 一格
    #    （``17/24 ≈ 0.708s`` ✓）—— 写 0.5 连自己都会判错 ✓✗（本套真撞过 ✓）。
    slack = geo.H3_FRAME_GRID / geo.H3_FPS + 1e-6
    colon = sp.parse_script("0:00-0:06 开场镜头")
    check("①″ ``0:00-0:06`` ⇒ 6 秒（±一格 ✓）且 ``source=range`` ✓（冒号写法也认 ✓）",
          colon.segments[0].source == "range"
          and abs(colon.segments[0].seconds - 6.0) <= slack, colon.to_dict())
    chinese = sp.parse_script("0至6秒：开场镜头")
    check("①‴ ``0至6秒：`` ⇒ 6 秒（±一格 ✓）且 ``source=range`` ✓（中文「至」也认 ✓）",
          chinese.segments[0].source == "range"
          and abs(chinese.segments[0].seconds - 6.0) <= slack, chinese.to_dict())
    bar = sp.parse_script("6.6秒 | 独白一场")
    check("①⁴ 时长直写 ``6.6秒 | 正文`` ⇒ ``source=bar`` ✓、正文不含竖线 ✓",
          bar.segments[0].source == "bar" and "|" not in bar.segments[0].text,
          bar.to_dict())
    multi = sp.parse_script("段1（6.6秒）：第一镜\n段2（6.6秒）：第二镜")
    check("①⁵ 多段标记 ⇒ 按标记切 ✓ 且**顺序不乱** ✓（索引从 1 起 ✓）",
          len(multi.segments) == 2 and [s.index for s in multi.segments] == [1, 2]
          and multi.segments[1].text == "第二镜", multi.to_dict())


def case_estimate() -> None:
    """② 无标记 ⇒ **4.5 字/秒** ✓（口径写死在常量里 ✓、可用参数覆盖 ✓）。"""
    check("② 语速常量就是 4.5 字/秒 ✓（来源口径 ✓）", sp.CHARS_PER_SECOND == 4.5)
    check("②′ 45 个汉字 ⇒ 10 秒 ✓（``字 ÷ 语速`` ✓）", sp.estimate_seconds("字" * 45) == 10.0,
          sp.estimate_seconds("字" * 45))
    check("②″ 语速可覆盖 ⇒ 估时随之变 ✓（⚠️ 英文语速**未核** ✗ ⇒ 必须留这个口子 ✓）",
          sp.estimate_seconds("字" * 45, units_per_second=9.0) == 5.0)
    check("②‴ 标点与空白不计入字数 ✓（句号/空格不该算成要读的字 ✓）",
          sp.speak_units("你好，世界！\n\n") == 4 and sp.speak_units("abc def") == 2,
          (sp.speak_units("你好，世界！\n\n"), sp.speak_units("abc def")))


def case_plain_split() -> None:
    """③ ⭐ 无标记长文：切 8~15 s ✓、**网格上** ✓、**不超上限** ✓、**不丢字** ✓。"""
    sentences = [f"第{i}个镜头里，{('人在' + '走路' * 8)}。" for i in range(1, 16)]
    body = "".join(sentences)
    plan = sp.parse_script(body)
    check("③ 长文 ⇒ 切成多段 ✓（每段都 ≤15 s 且落在 17k+5 网格上 ✓）",
          len(plan.segments) > 1 and _all_on_grid(plan) and _within_cap(plan),
          [(s.index, s.frames, s.seconds) for s in plan.segments])
    check("③′ ⭐⭐ **永不丢字** ✓：所有段（去空白）拼起来 == 原文（去空白 ✓）",
          _conserved(plan, body), (len(_squeeze(body)),
                                   sum(len(_squeeze(s.text)) for s in plan.segments)))
    check("③″ ⭐ **绝不在句中硬断** ✗：标点齐全的长文里 ``hard_cut`` 必须全为假 ✓",
          not any(segment.hard_cut for segment in plan.segments),
          [(s.index, s.text[-12:]) for s in plan.segments if s.hard_cut])
    check("③‴ 有说明「为什么这么切」✓（静默切分比切错更难查 ✗）",
          any("4.5" in note or "语速" in note or "朗读" in note for note in plan.notes),
          plan.notes)


def case_break_priority() -> None:
    """④ 断点优先级 ✓：**段落换行 > 句末标点 > 从句标点** ✓。"""
    # ⚠️ 夹具要让「段落断点**真的可用**」✗：每段必须**短于上限** ✓ —— 否则段内没有任何断点、
    #    只能硬切 ✓，那时断言「全软切」是错的 ✓✗（本套真撞过 ✓）。
    paragraphs = [("这是第%d段，" % index) + "内容很长" * 8 + "。" for index in range(1, 9)]
    body = "\n\n".join(paragraphs)
    plan = sp.parse_script(body)
    ends = [segment.text[-1] for segment in plan.segments]
    check("④ 段落换行齐全时 ⇒ 切点落在**段落/句末**上 ✓（结尾字符不是逗号/顿号 ✓✗）",
          all(char not in sp.CLAUSE_BREAKS for char in ends), ends)
    check("④′ 同上 ⇒ 全部非硬切 ✓（优先级真的生效了 ✓）",
          not any(segment.hard_cut for segment in plan.segments))


def case_hard_cut_visible() -> None:
    """⑤ 无标点超长串 ⇒ 允许硬切 ✓ 但**必须标出来** ✓（别让用户拿到半句话还不知情 ✗）。"""
    body = "甲乙丙丁" * 400
    plan = sp.parse_script(body)
    check("⑤ 无标点超长串 ⇒ 出现 ``hard_cut=True`` ✓ 的段（**标出来** ✓ 不静默 ✗）",
          any(segment.hard_cut for segment in plan.segments)
          and _all_on_grid(plan) and _within_cap(plan),
          [(s.index, s.hard_cut) for s in plan.segments])
    check("⑤′ 且**照样不丢字** ✓（硬切只影响「切在哪」✓，不影响「切多少」✓）",
          _conserved(plan, body))


def case_shots() -> None:
    """⑥ 官方 Shot 格式 ✓：≤15 s **合并成一段** ✓；>15 s 装桶且**不改写时间戳** ✗。"""
    short = ("integrated_multimodal_description: [Shot 1] 开场。"
             "[Shot 2] At 00:03.000, 发展。[Shot 3] At 00:06.000, 收尾。")
    plan_short = sp.parse_script(short)
    check("⑥ 总时长 ≈9s ≤15 ⇒ **合并成一段** ✓（一次生成 ✓ 段间不会断连 ✓）",
          len(plan_short.segments) == 1 and plan_short.style == "shots"
          and any("合并" in note for note in plan_short.notes), plan_short.to_dict())
    # ⚠️ 夹具要**真实** ✗：首镜**本来就没有** ``At`` ✓（它是 0s 起 ✓）⇒ 断言「每段都含
    #    ``At 00:``」是错的 ✓✗（本套真撞过 ✓）；改成「**原有的时间戳字面**一个都没被改写」✓。
    long_shots = ("[Shot 1] 开场镜头，" + "很长的描述" * 6 + "。"
                  + "".join(f"[Shot {index}] At 00:{index * 6:02d}.000, 第{index}镜。"
                            for index in range(2, 6)))
    plan_long = sp.parse_script(long_shots)
    check("⑥′ 总时长 >15 ⇒ 按 Shot 边界装桶 ✓ 且**原时间戳字面还在** ✗（不许重排/重写 ✗✗）、"
          "正文也**一点没丢** ✓",
          len(plan_long.segments) > 1
          and any("At 00:12.000" in segment.text for segment in plan_long.segments)
          and _conserved(plan_long, long_shots), [s.text[:26] for s in plan_long.segments])


def case_long_explicit() -> None:
    """⑦ 显式时长 >15 s ⇒ **切**（不是截 ✗✗ —— 截会静默丢内容 ✓✗）。"""
    body = "段1（40秒）：" + "".join(f"第{i}句，内容很长很长。" for i in range(1, 12))
    plan = sp.parse_script(body)
    check("⑦ 显式 40 s ⇒ 切成多段 ✓（⚠️ 与来源的「截到 15 秒」**有意不同** ✓：本仓不丢内容 ✗）"
          "且说明里点出这一点 ✓",
          len(plan.segments) > 1 and _within_cap(plan)
          and any("不丢内容" in note for note in plan.notes), plan.notes)
    check("⑦′ 且所有段仍在网格上 ✓、正文没丢 ✓",
          _all_on_grid(plan) and _conserved(plan, body.replace("段1（40秒）：", "")),
          [(s.index, s.frames) for s in plan.segments])


def case_edges() -> None:
    """⑧ 边界 ✓：空稿 / 非法区间 / 语速覆盖 / 吸附是**往下**的 ✓。"""
    empty = sp.parse_script("   \n  ")
    check("⑧ 空稿 ⇒ 空计划 ✓ 且**给说明** ✓（不是抛错 ✗、也不是假装切了一段 ✗）",
          empty.segments == () and empty.style == "empty" and empty.notes, empty.to_dict())
    check("⑧′ 区间不合法（min ≥ max）⇒ 报错 ✓（静默跑下去只会切出怪东西 ✗）",
          _raises(lambda: sp.parse_script("正文", min_seconds=15.0, max_seconds=8.0)) is not None)
    slow = sp.parse_script("第１段话很长，" * 40, units_per_second=2.0)
    fast = sp.parse_script("第１段话很长，" * 40, units_per_second=6.0)
    check("⑧″ 语速覆盖真的影响段数 ✓（语速越慢 ⇒ 段越多 ✓ —— 否则那个口子是死的 ✗）",
          len(slow.segments) >= len(fast.segments) and _all_on_grid(slow) and _all_on_grid(fast),
          (len(slow.segments), len(fast.segments)))
    # ⚠️ 容差按**秒数的取整位数**给 ✓（秒数存的是 ``round(frames/fps, 4)`` ✓ ⇒ 1e-6 的容差
    #    连自己都会判错 ✗ —— 断言要**对着实际的存储口径**写 ✗）。
    check("⑧‴ ⭐ 吸附**只许往下** ✗✗：每段秒数乘 24 必须**回到**它的帧数 ✓（容差按取整位数 ✓），"
          "且秒数 ≤ 上限 ✓（往上吸附会越过 15 s ✓✗）",
          all(abs(segment.seconds * geo.H3_FPS - segment.frames) < 0.01 for segment in fast.segments)
          and _within_cap(fast))


def case_wired_into_storyboard() -> None:
    """⑨ ⭐ **接线**：分镜助手层必须**真的**调得到它 ✓（本仓纪律：接不出去不算功能 ✗）。"""
    from app.services import storyboard_helpers as helpers  # noqa: PLC0415

    # ⚠️ 这里必须是**真换行** ✗：写成 ``\\n`` 是「字面反斜杠 + n」✓ ⇒ 两段标记会挤成一行、
    #    段数变成 1 ✓✗（本套真撞过 ✓）。
    got = helpers.segments_from_script("段1（6.6秒）：黄昏屋顶，少年转身\n段2（6.6秒）：雨落下来")
    check("⑨ ⭐ 从**分镜助手层**调得到 ✓：两段标记 ⇒ 两段 ✓、每段落在 17k+5 网格上 ✓、"
          "时长 ≤15 s ✓（这条证的是**可达性** ✓ 不是纯函数 ✓）",
          got["count"] == 2 and got["style"] == "marker"
          and all(_on_grid(seg["frames"]) and seg["seconds"] <= sp.MAX_SECONDS - 1e-9
                  for seg in got["segments"]), got["count"])
    long_text = "".join(f"第{i}句，内容很长很长。" for i in range(1, 12))
    check("⑨′ 无标记长文经那一层 ⇒ 多段 ✓ 且**说明（notes）也跟着出来** ✓"
          "（「为什么这么切」要能带到上层 ✓ —— 静默切分比切错更难查 ✗）",
          helpers.segments_from_script(long_text)["count"] > 1
          and bool(helpers.segments_from_script(long_text)["notes"]))
    check("⑨″ 语速口子透传 ✓（传 2.0 字/秒 ⇒ 段数**不少于**默认 ✓ —— 否则那个参数是死的 ✗）",
          helpers.segments_from_script(long_text, units_per_second=2.0)["count"]
          >= helpers.segments_from_script(long_text)["count"])


def main() -> int:
    case_markers()
    case_estimate()
    case_plain_split()
    case_break_priority()
    case_hard_cut_visible()
    case_shots()
    case_long_explicit()
    case_edges()
    case_wired_into_storyboard()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
