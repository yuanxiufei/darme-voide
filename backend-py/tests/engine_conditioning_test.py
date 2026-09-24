"""S7 自检：**H3 参考素材的 prompt 契约**（声明 / 跳过判据 / 标签 / 任务选择 ✓ 零依赖 ✓ 2026-09-24）。

钉的是「**会让参考素材失效**」的判据 ✗✗：

* ⭐ **只写 `<Audio 1>` 绑定句不算声明** ✗ ⇒ 自动声明必须**照常追加**（否则模型自由发挥、
  成片音轨与配音相关性≈0 ✓✗ —— 参考实现 v1.14.1 的根因 ✓）；
* 协议键/协议词**逐字** ✓（写错等于没声明 ✗）；声明块**只追加**不改写 ✓、且**幂等** ✓；
* 标签编号错位要在**提交前**报 ✓（`<Picture 3>` 而只有 2 张 ✓）；「有素材没引用」**不报** ✗（旁白常见 ✓）；
* ⭐ 任务选择：**有参考素材 ⇒ 只能 Ref2VA** ✗（FL2VA 强行生成会丢素材/失败 ✓✗）；判不出形态 ⇒ 拒 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_conditioning_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import conditioning as cond  # noqa: E402

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


def _keys(block: str) -> list[str]:
    """声明块里出现的**协议键**（按行首取 ✓）。"""
    return [line.split(":", 1)[0] for line in block.splitlines() if ":" in line]


def case_protocol() -> None:
    """① 协议词/键**逐字** ✓（写错等于没声明 ✗）。"""
    check("① 参考关系就是那三个协议词 ✓ 且顺序固定 ✓",
          cond.AUDIO_RELATIONSHIPS == ("fully_copy", "partially_copy", "reference"),
          cond.AUDIO_RELATIONSHIPS)
    check("①′ 三行协议键逐字 ✓（模型按它解析 ✓；写错就等于没声明 ✗）",
          cond.DECLARATION_KEYS == ("subject_definitions", "retention_analysis",
                                    "detailed_description"), cond.DECLARATION_KEYS)
    check("①″ 参考视频上限 3 路 ✓（H3 原生口径 ✓）", cond.MAX_REFERENCE_VIDEOS == 3)


def case_audio_declaration() -> None:
    """② 三档关系各自产出**完整三行** ✓（缺一行模型可能就不认 ✓）。"""
    blocks = {}
    for relation in cond.AUDIO_RELATIONSHIPS:
        blocks[relation] = cond.audio_declaration([cond.AudioReference(1, relation, 2)])
        check(f"② 关系 {relation}：三行协议键**齐全** ✓ 且顺序固定 ✓",
              _keys(blocks[relation]) == list(cond.DECLARATION_KEYS), blocks[relation])
    check("②′ ``fully_copy`` ⇒ 说的是「整轨 1:1 复用」✓（配音当最终音轨 ✓）",
          "complete final audio track" in blocks["fully_copy"]
          and "<Audio 1>" in blocks["fully_copy"] and "(S2)" in blocks["fully_copy"],
          blocks["fully_copy"])
    check("②″ ``partially_copy`` ⇒ 说的是「只复用对话层 ✓ + 环境音/音效/配乐**新生成** ✓」",
          "dialogue layer" in blocks["partially_copy"] and "ambience" in blocks["partially_copy"],
          blocks["partially_copy"])
    check("②‴ ``reference`` ⇒ 说的是「只学音色与语气 ✓、**不复制原信号** ✓」",
          "voice-timbre" in blocks["reference"] and "without copying" in blocks["reference"],
          blocks["reference"])
    two = cond.audio_declaration([cond.AudioReference(1, "fully_copy", 1),
                                  cond.AudioReference(2, "reference", 2)])
    check("②⁴ 多路 ⇒ 每路各自那三行都在 ✓（拼在一起 ✓、不互相吞 ✓；行数 = 题注 + 三行 ✓）",
          "<Audio 1>" in two and "<Audio 2>" in two and "(S1)" in two and "(S2)" in two
          and len(two.splitlines()) == 4, two)
    check("②⁵ ``header=None`` ⇒ 不带题注 ✓（题注**未核出**是协议的一部分 ✗ ⇒ 要能去掉 ✓）",
          not cond.audio_declaration([cond.AudioReference(1, "fully_copy")], header=None)
          .startswith("["))
    check("②⁶ 关系词写错 ⇒ 报错并列出合法的三个 ✓（不许静默替换成默认关系 ✗）",
          _raises(lambda: cond.audio_declaration([cond.AudioReference(1, "full_copy")]), "partially_copy")
          is not None)
    check("②⁷ 同一路编号声明两次 / 编号为 0 / 空列表 ⇒ 各自报错 ✓",
          all(_raises(call) is not None for call in (
              lambda: cond.audio_declaration([cond.AudioReference(1, "fully_copy"),
                                              cond.AudioReference(1, "reference")]),
              lambda: cond.audio_declaration([cond.AudioReference(0, "fully_copy")]),
              lambda: cond.audio_declaration([]))))


def case_video_declaration() -> None:
    """③ 参考视频声明 ✓：`reference` 关系 + 路数上限 ✓。"""
    one = cond.video_declaration(1)
    check("③ 一路：协议键齐全 ✓ 且关系用 ``reference`` ✓（动作跟视频、外观跟参考图 ✓）",
          _keys(one) == list(cond.DECLARATION_KEYS) and "<Video 1>: reference" in one
          and "replaced by the subjects from the reference images" in one, one)
    check("③′ 三路：三个标签都在 ✓ 且限制句在 ✓（一镜到底、无转场 ✓）",
          all(tag in cond.video_declaration(3) for tag in ("<Video 1>", "<Video 2>", "<Video 3>"))
          and "No cuts" in cond.video_declaration(3), cond.video_declaration(3)[:120])
    check("③″ 0 路 / 4 路 ⇒ 报错 ✓（4 路那条要点名上限 3 ✓）",
          _raises(lambda: cond.video_declaration(0)) is not None
          and _raises(lambda: cond.video_declaration(4), "3 路") is not None)


def case_plan() -> None:
    """④ ⭐ **跳过判据** —— 这个坑的落点 ✓（只写标签**不算**声明 ✗✗）。"""
    refs = [cond.AudioReference(1, "fully_copy")]
    check("④ ⭐⭐ 提示词里只有 ``<Audio 1>`` 绑定句 ⇒ **照样要追加声明** ✓✗ —— "
          "照标签跳会让模型忽略参考音频 ✓✗",
          cond.declaration_plan("<Audio 1> 是 <Picture 1> 的音色参考", audio_refs=refs).audio_needed
          is True)
    mine = cond.declaration_plan("retention_analysis: <Audio 1>: fully_copy", audio_refs=refs)
    check("④′ 用户**自己写了** ``retention_analysis`` ⇒ 不重复追加 ✓ 且**必须给理由** ✓"
          "（跳过不是「无事发生」✗ —— 得能说清是谁写的 ✓）",
          mine.audio_needed is False and "保留声明" in (mine.audio_skipped or ""), mine.to_dict())
    check("④″ 视频那半同理 ✓：只写 ``<Video 1>`` 引用句 ⇒ 仍要追加 ✓✗；"
          "``retention_analysis`` **与** ``<Video`` 同时出现 ⇒ 才跳过 ✓",
          cond.declaration_plan("<Video 1> 跟着这个视频", video_count=1).video_needed is True
          and cond.declaration_plan("retention_analysis: <Video 1>: reference", 
                                    video_count=1).video_needed is False)
    check("④‴ 没有参考素材 ⇒ 两块都不追加 ✓（且理由是 ``None`` ✓ —— 不是「跳过」✗）",
          cond.declaration_plan("随便写").to_dict() == {"audioNeeded": False, "videoNeeded": False,
                                                        "audioSkipped": None, "videoSkipped": None})


def case_apply() -> None:
    """⑤ 追加是**只追加** ✓ 且**幂等** ✓（原样不改写 ✓）。"""
    text, plan = cond.apply_declarations("黄昏屋顶，少年转身", audio_refs=[cond.AudioReference(1, "fully_copy")],
                                         video_count=0)
    check("⑤ 原文**逐字保留** ✓、声明追加在后 ✓、计划里标了「追加了音频」✓",
          text.startswith("黄昏屋顶，少年转身") and "subject_definitions:" in text
          and plan.audio_needed is True, text[:60])
    again, plan2 = cond.apply_declarations(text, audio_refs=[cond.AudioReference(1, "fully_copy")])
    check("⑤′ ⭐ **幂等**：对已经带声明的提示词再调一次 ⇒ **一字不变** ✓（否则每轮都会堆一份 ✗）",
          again == text and plan2.audio_needed is False, (len(text), len(again)))
    custom, _ = cond.apply_declarations("x", audio_refs=[cond.AudioReference(1, "fully_copy")],
                                        audio_header="[my header]")
    check("⑤″ 题注可覆盖 ✓（题注**未核出**是协议 ✗ ⇒ 得能换 ✓）", "[my header]" in custom, custom[:40])


def case_markup() -> None:
    """⑥ 文本写法 → 标签 ✓（编号错位是最常见的错 ✗）。"""
    check("⑥ 四种写法都转 ✓（``@图1`` / ``【图2】`` / ``@image3`` / ``@音1`` ✓）",
          cond.h3_markup("@图1 与 【图2】 与 @image3 与 @音1")
          == "<Picture 1> 与 <Picture 2> 与 <Picture 3> 与 <Audio 1>",
          cond.h3_markup("@图1 与 【图2】 与 @image3 与 @音1"))
    check("⑥′ 已经是标签的原样留着 ✓（**幂等** ✓）",
          cond.h3_markup("<Picture 1> 和 <Audio 2>") == "<Picture 1> 和 <Audio 2>")
    check("⑥″ 编号 0 ⇒ 报错 ✓（``<Picture 0>`` 不是合法标签 ✓）",
          _raises(lambda: cond.h3_markup("@图0")) is not None)
    check("⑥‴ 已出现的标签能列出来 ✓（去重且升序 ✓）",
          cond.markup_references("<Picture 2> <Picture 1> <Picture 1> <Audio 1>")
          == {"Picture": [1, 2], "Audio": [1]})


def case_validate_markup() -> None:
    """⑦ **提交前**核编号 ✓（引用了但没素材 ⇒ 报 ✓；有素材没引用 ⇒ **不报** ✗）。"""
    msg = _raises(lambda: cond.validate_markup("看着 <Picture 3> 说话", pictures=2), "续接尾帧")
    check("⑦ 引用 ``<Picture 3>`` 而只给 2 个素材 ⇒ 报错 ✓ 且提示**编号会前移**那条坑 ✓",
          msg is not None and "最前" in msg, msg)
    check("⑦′ 给足了 ⇒ 不抛 ✓；素材多于引用 ⇒ 也**不抛** ✓（旁白/备用素材很常见 ✓）",
          cond.validate_markup("看着 <Picture 2> 说话", pictures=3) is None
          and cond.validate_markup("纯旁白", pictures=2, audios=1) is None)


def case_slots() -> None:
    """⑧ 编号对照表 ✓：**续接尾帧排最前** ✓ ⇒ 取消它会让后面全部**前移** ✗✗。"""
    check("⑧ 续接尾帧占 ``<Picture 1>`` ✓、上传图从 2 起 ✓",
          cond.picture_slots(tail=True, images=2) == [(1, "续接尾帧"), (2, "上传图 1"), (3, "上传图 2")],
          cond.picture_slots(tail=True, images=2))
    check("⑧′ ⭐ **换场景**（取消续接尾帧 ✓）⇒ 上传图变成从 **1** 起 ✓✗ "
          "—— 提示词里手写的 ``@图N`` 会因此指错 ✓（所以要点缩略图重插 ✓）",
          cond.picture_slots(tail=False, images=2) == [(1, "上传图 1"), (2, "上传图 2")],
          cond.picture_slots(tail=False, images=2))
    check("⑧″ 本段配音占 ``<Audio 1>`` 时，音色槽**自动从 2 起** ✓；不占时从 1 起 ✓",
          cond.audio_slots(segment_audio=True, voices=2) == [(1, "本段配音（参考驱动）"),
                                                             (2, "音色槽 A1"), (3, "音色槽 A2")]
          and cond.audio_slots(segment_audio=False, voices=1) == [(1, "音色槽 A1")],
          cond.audio_slots(segment_audio=True, voices=2))


def case_select_task() -> None:
    """⑨ ⭐ 走 Ref2VA 还是 FL2VA ✓ —— **有参考素材只能 Ref2VA** ✗✗（硬约束 ✓）。"""
    check("⑨ 主模型 Ref2VA + 有参考素材 ⇒ ``ref2va`` ✓",
          cond.select_h3_task(primary_model_kind="ref2va", has_references=True) == "ref2va")
    bad = _raises(lambda: cond.select_h3_task(primary_model_kind="fl2va", has_references=True),
                  "Ref2VA")
    check("⑨′ ⭐⭐ 主模型 FL2VA + 有参考素材 ⇒ **拒** ✗✗ 且理由说清「会丢素材或失败」✓ "
          "并给出可执行解法（下载目录 ✓）",
          bad is not None and "丢素材" in bad and "diffusion_models" in bad, bad)
    hard = _raises(lambda: cond.select_h3_task(primary_model_kind="ref2va", has_first_frame=True),
                   "FL2VA")
    check("⑨″ 要**硬首帧**却没有 FL2VA ⇒ 拒 ✓ 且给两条出路（接第二模型 / 改软参考 ✓）",
          hard is not None and "软参考" in hard, hard)
    check("⑨‴ 接了第二模型 ⇒ 硬首帧走 ``fl2va`` ✓；纯文本且偏好 FL2VA ⇒ ``fl2va`` ✓；"
          "其余 ⇒ ``ref2va`` ✓",
          cond.select_h3_task(primary_model_kind="ref2va", has_first_frame=True,
                              has_optional_fl2va=True) == "fl2va"
          and cond.select_h3_task(primary_model_kind="ref2va", prefer_fl2va=True,
                                  has_optional_fl2va=True) == "fl2va"
          and cond.select_h3_task(primary_model_kind="ref2va") == "ref2va")
    check("⑨⁴ 形态判不出来（如 ``sd15`` / 空）⇒ **拒** ✗ 不猜 ✓（猜错会装出「形状对、能力不对」的模型 ✗✗）",
          _raises(lambda: cond.select_h3_task(primary_model_kind="sd15"), "不猜") is not None
          and _raises(lambda: cond.select_h3_task(primary_model_kind=""), "不猜") is not None)


def main() -> int:
    case_protocol()
    case_audio_declaration()
    case_video_declaration()
    case_plan()
    case_apply()
    case_markup()
    case_validate_markup()
    case_slots()
    case_select_task()
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
    sys.exit(main())
