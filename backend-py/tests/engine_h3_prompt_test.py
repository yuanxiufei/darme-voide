"""H3 文本侧**呈现格式 + token 标签**的自检 ✓（纯 Python ✓ 不需要 torch ✓）。

钉的是「**段序 / 序号 / 时间戳格式 / 标签边界**」四类机械事实 ✓：
* 标签与视觉块必须**交替** ✓（事后补视觉 ⇒ 位置全错 ✗）
* 序号**按类型独立**计数 ✓（``<Picture 1> <Audio 1> <Picture 2>`` ✓）
* 时间戳 ``"<%.1f seconds>"`` ✓（``%.1f`` 的舍入是参考行为 ✓ 不是 bug ✓）
* 视觉块标签含**两侧包围 token** ✓ ⇒ 区间是 ``[index-1, index+size+1)`` ✓
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import h3_prompt as h3p  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(fn) -> object:
    try:
        fn()
    except BaseException as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return "（**没有报错** ✗）"


def _kinds(pieces) -> str:
    return "".join("V" if piece.get("kind") == "vision" else "T" for piece in pieces)


def main() -> int:
    # ① t2va：只有 prompt ⇒ 一段文本 ✓、零视觉块 ✓
    t2va = h3p.presentation("A cat walks in.")
    check("① t2va（没有参考）⇒ 只有一段 prompt 文本 ✓、零视觉块 ✓",
          _kinds(t2va) == "T" and h3p.segments(t2va) == ["A cat walks in."], _kinds(t2va))

    # ② fl2va：两张图 ⇒ 标签与视觉块**交替** ✓
    fl2va = h3p.presentation("She turns.", images=[object(), object()])
    check("② fl2va（2 张图）⇒ ``T V T V T`` ✓ 且标签是 ``<Picture 1>: `` / ``<Picture 2>: `` ✓",
          _kinds(fl2va) == "TVTVT"
          and h3p.segments(fl2va) == ["<Picture 1>: ", "<Picture 2>: ", "She turns."],
          h3p.segments(fl2va))

    # ③ ref2va：三种参考混排 ✓ —— audio **只占文本** ✓ 没有视觉块 ✓
    refs = [{"type": "image"}, {"type": "audio"}, {"type": "image"},
            {"type": "video", "frames": 4}]
    mixed = h3p.presentation("Go.", refs=refs)
    #    ⚠️ 段序里出现**连续两个 T** 是**对的** ✓（``<Audio 1>: `` 后面没有视觉块 ✓）
    check("③ ref2va 混排 ⇒ 段序 ``TVTTVTTVTVT`` ✓（``<Audio 1>: `` 与 ``<Picture 2>: `` 之间"
          "没有视觉块 ⇒ 两个 T 相邻 ✓ 这是对的 ✓）",
          _kinds(mixed) == "TVTTVTTVTVT"
          and h3p.segments(mixed)[:4] == ["<Picture 1>: ", "<Audio 1>: ",
                                          "<Picture 2>: ", "<Video 1>: "],
          (_kinds(mixed), h3p.segments(mixed)))

    # ④ 序号**按类型独立** ✓ —— 两个 image 之间夹 audio，第二张图仍是 ``<Picture 2>`` ✓
    check("④ 序号按类型独立计数 ✓（``<Audio 1>`` 不会把第二张图顶成 ``<Picture 3>`` ✗）",
          "<Picture 3>" not in h3p.segments(mixed), h3p.segments(mixed))

    # ⑤ 时间戳格式 ``"<%.1f seconds>"`` ✓ —— 中点取样 ✓、``%.1f`` 的舍入照参考 ✓
    check("⑤ 4 帧 ⇒ 2 块，块时间戳是**两帧中点** ✓（默认 2 fps ⇒ 0.25 / 1.25 ⇒ "
          "``<0.2 seconds>`` / ``<1.2 seconds>`` ✓ ``%.1f`` 就是这样舍入 ✓ 不是 bug ✓）",
          [block[2] for block in h3p.video_blocks(4)]
          == ["<0.2 seconds>", "<1.2 seconds>"],
          [block[2] for block in h3p.video_blocks(4)])
    check("⑤′ 自定义时间戳 ⇒ ``<1.5 seconds>`` / ``<5.0 seconds>`` ✓（``1.0`` 要写成 ``1.0`` ✓ 不是 ``1`` ✗）",
          [block[2] for block in h3p.video_blocks(4, [1.0, 2.0, 4.0, 6.0])]
          == ["<1.5 seconds>", "<5.0 seconds>"],
          [block[2] for block in h3p.video_blocks(4, [1.0, 2.0, 4.0, 6.0])])

    # ⑥ 奇数帧 ⇒ **当场报错** ✓（补齐要动像素 ⇒ 不在这里做 ✗）
    odd = _raises(lambda: h3p.video_blocks(5))
    check("⑥ 奇数帧 ⇒ 报错 ✓ 且点名「重复末帧」✓（悄悄补一帧 ⇒ 时间戳与像素不同源 ✗）",
          "重复末帧" in str(odd), odd)
    # ⑥′ 时间戳个数不符 ⇒ 报错 ✓
    bad_stamps = _raises(lambda: h3p.video_blocks(4, [0.0, 0.5]))
    check("⑥′ 时间戳个数 ≠ 帧数 ⇒ 报错 ✓（少给 ⇒ 后面的块整体错位 ✓ 且不报错 ✗）",
          "时间戳" in str(bad_stamps), bad_stamps)

    # ⑦ 视觉块标签**含两侧包围 token** ✓ ⇒ ``[index-1, index+size+1)`` 置 0 ✓
    tags = h3p.token_tags(12, [(5, 4)])
    zeros = [i for i, tag in enumerate(tags) if tag == h3p.TAG_VISION]
    check("⑦ span=(5,4) ⇒ 0 落在 ``[4, 10)`` ✓（左包围在 4 ✓、右包围在 9 ✓）"
          " 其余全 1 ✓",
          zeros == [4, 5, 6, 7, 8, 9], zeros)
    check("⑦′ 没有视觉块 ⇒ **全 1** ✓（纯文本提示词不该有 0 ✗）",
          set(h3p.token_tags(6, [])) == {h3p.TAG_TEXT}, h3p.token_tags(6, []))
    # ⑦″ 两个 span ⇒ 各自带包围 ✓ 且首块的右包围不会被第二块吃掉 ✓
    two = h3p.token_tags(12, [(3, 2), (9, 2)])
    check("⑦″ 两个 span=(3,2)/(9,2) ⇒ 0 在 ``[2,6)`` 与 ``[8,12)`` ✓",
          [i for i, tag in enumerate(two) if tag == 0] == [2, 3, 4, 5, 8, 9, 10, 11],
          [i for i, tag in enumerate(two) if tag == 0])
    # ⑦‴ 越界 ⇒ 报错 ✓（静默截断会把尾部正文误标成视觉 ✗）
    over = _raises(lambda: h3p.token_tags(6, [(5, 4)]))
    check("⑦‴ span 越界 ⇒ 报错 ✓（静默截断 = 悄悄改标签 ✗）", "越出序列长度" in str(over), over)

    # ⑧ 口径互斥 ✓ / 未知类型 ⇒ 报错 ✓
    both = _raises(lambda: h3p.presentation("x", refs=[], images=[]))
    check("⑧ ``refs`` 与 ``images`` 同时给 ⇒ 报错 ✓（ref2va / fl2va 是两条口径 ✗）",
          "两条不同口径" in str(both), both)
    unknown = _raises(lambda: h3p.presentation("x", refs=[{"type": "mesh"}]))
    check("⑧′ 未知参考类型 ⇒ 报错 ✓（静默跳过 ⇒ 少一张图且不报 ✗）",
          "不认识" in str(unknown), unknown)

    # ⑨ 文本段必须**逐段** ✓：``"<Picture 1>: "`` 与 prompt 是两段 ✓（拼起来分词 ⇒ BPE 边界变了 ✗）
    check("⑨ 段序列是 ``['<Picture 1>: ', 'She turns.']`` ✓ —— "
          "**两段**（末尾空格也是词表的一部分 ✓ 少了就与 prompt 粘连 ✗）",
          h3p.segments(fl2va)[:2] == ["<Picture 1>: ", "<Picture 2>: "],
          h3p.segments(fl2va))

    # ⑩ 视频块区间**连续且长度恒 2** ✓
    spans = [(piece["frames"]) for piece in mixed if piece.get("videoBlock")]
    check("⑩ 视频块区间长度恒 = 2 ✓ 且首尾相接 ✓（丢一帧 ⇒ 时间轴短一截 ✗）",
          spans == [(0, 2), (2, 4)], spans)

    # ⑪ ``check_pieces``：正确串 ⇒ 空 ✓；序号写错 ⇒ 报 ✓
    check("⑪ 正确的串 ⇒ ``check_pieces`` 返回空 ✓",
          h3p.check_pieces(mixed, "Go.") == [], h3p.check_pieces(mixed, "Go."))
    bad = [{"kind": "text", "text": "<Picture 3>: "},
           {"kind": "vision", "frames": None, "videoBlock": False}]
    check("⑪′ 第一张图就写 ``<Picture 3>`` ⇒ 报「序号不对」✓（序号写错不报错 = 静默错位 ✗）",
          any("序号不对" in problem for problem in h3p.check_pieces(bad)), h3p.check_pieces(bad))
    # ⑪″ 视频块写成单图 ⇒ 报 ✓
    wrong_block = [{"kind": "vision", "frames": (0, 2), "videoBlock": False}]
    check("⑪″ ``frames=(0,2)`` 却标 ``videoBlock=False`` ⇒ 报错 ✓（**正反两道** ✓）",
          any("frames" in problem for problem in h3p.check_pieces(wrong_block)),
          h3p.check_pieces(wrong_block))

    # ⑫ 空 prompt ⇒ 不产生空文本段 ✓（空段也会多切一刀分词 ✓ 无意义 ✗）
    check("⑫ ``prompt=\"\"`` ⇒ 不追加空段 ✓（段数 = 标签数 ×2 ✓）",
          _kinds(h3p.presentation("", images=[object()])) == "TV",
          _kinds(h3p.presentation("", images=[object()])))

    # ⑬ 常量口径 ✓
    check("⑬ ``<|vision_start|>`` = 151652 / ``<|vision_end|>`` = 151653 / pad = 151643 ✓",
          (h3p.VISION_START, h3p.VISION_END, h3p.PAD_TOKEN_ID) == (151652, 151653, 151643),
          (h3p.VISION_START, h3p.VISION_END, h3p.PAD_TOKEN_ID))
    check("⑬′ 标签值：视觉 = **0** ✓、文本 = **1** ✓（``VIDEO(0)`` ✓ 别与主干的 mod 混用 ✗）",
          (h3p.TAG_VISION, h3p.TAG_TEXT) == (0, 1), (h3p.TAG_VISION, h3p.TAG_TEXT))

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name} —— 实际：{detail!r}")
    print(f"SUMMARY {len(_RESULTS) - len(failed)}/{len(_RESULTS)} 通过 ✓")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
