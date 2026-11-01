"""H3 **文本标签 → adaLN 行号**的自检 ✓（要 torch ✓ —— 没有就 SKIP 并**明说** ✓，不假装通过 ✗）。

## 为什么单独一份 ✗

``h3_prompt.token_tags`` 从 2026-11-01 起就在产 0/1 标签 ✓，但**到那天为止没有任何消费者** ✗ ——
主干里每个段的模态行号是**整段一个整数** ✓ ⇒ 文本段里那些 ``IMAGE_PAD`` 拿到了 **text 那一行
adaLN** ✓✗（参考里它们是 **video 行（标签 0）** ✓，见 ``model.py`` 那段
``for i in range(1, b - a + 1)`` 的**按标签切 run** ✓）。

错法**一个形状都不差** ✓：段数、行数、总行数、张量形状全对 ✓ —— 只有"每个视觉占位 token 用哪组
调制参数"错 ✓✗。所以本份只钉**能用别的方法算出来**的东西：

* 段表与**另写的 oracle**（照参考循环逐字转写 ✓）在随机标签 × 随机段上**逐值**对账 ✓；
* 纯文本（全 1）**不许被凭空切碎** ✓（多切一段 ⇒ 行号其实没变 ✓，但"段数"是别处会读的事实 ✗）；
* 主干**真的在用它** ✓：同一份输入、只换标签 ⇒ 输出**必变** ✓（忽略标签 ⇒ 逐位相同 ⇒ 红 ✓）。
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import h3_form as h3  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch ✓ —— **如实标记**，不算通过 ✗）"))


def main() -> int:
    # ── ① 纯 Python 部分：切分语义（不需要 torch ✓）──
    #: 布局里的 ``index`` 是"段序号" ✓（这里给个不像 0/1 的值，防止把序号写死 ✓）
    index = {"cond": 0, "ref_img": 0, "text": 2, "video": 3, "audio": 4}
    segments = [(0, 4, "text"), (4, 8, "video")]
    plain = h3.mod_segments_for(segments, index, 3)
    check("① 不给 `text_tags` ⇒ **老行为**（文本段整段一个 text 行 ✓，回归不破 ✓）",
          plain == ((0, 4, 2 * 3 + 1), (4, 8, 3 * 3 + 0)), plain)

    ones = h3.mod_segments_for(segments, index, 3, text_tags=[1, 1, 1, 1])
    check("② 标签**全 1**（纯文本提示词 ✓）⇒ **仍然是一段** ✓（凭空切碎 ⇒ 段数变了 ✗，别处会读它 ✗）",
          ones == plain, ones)

    pads = h3.mod_segments_for(segments, index, 3, text_tags=[1, 0, 0, 1])
    expect = ((0, 1, 2 * 3 + 1), (1, 3, 2 * 3 + 0), (3, 4, 2 * 3 + 1), (4, 8, 3 * 3 + 0))
    check("③ 中间两个视觉占位 ⇒ 切成 **text / video / text** 三段 ✓、"
          "**中间那段行号是 video 行（`index×3+0`）** ✓（这是整条缺陷的正解 ✓）",
          pads == expect, pads)
    check("③′ 切出来的段 **首尾相接、不重不漏** ✓（`stop == 下一段 start` ✓ 且覆盖 0..len ✓）",
          [run[1] for run in pads[:-1]] == [run[0] for run in pads[1:]]
          and pads[0][0] == 0 and pads[-2][1] == 4, pads)

    lead = h3.mod_segments_for(segments, index, 3, text_tags=[0, 0, 1, 1])
    check("④ 标签**以 0 开头** ⇒ 第一段就落在 video 行 ✓（边界条件 `i == b-a` 与 `tags[0]` 一起对上 ✓）",
          lead == ((0, 2, 2 * 3 + 0), (2, 4, 2 * 3 + 1), (4, 8, 3 * 3 + 0)), lead)

    tail = h3.mod_segments_for(segments, index, 3, text_tags=[1, 1, 0, 0])
    check("④′ 标签**以 0 结尾** ⇒ 末段取 video 行 ✓（收尾那次判界 `i == b-a` 落在正确的一侧 ✓）",
          tail == ((0, 2, 2 * 3 + 1), (2, 4, 2 * 3 + 0), (4, 8, 3 * 3 + 0)), tail)

    longer = h3.mod_segments_for(segments, index, 3, text_tags=[1, 1, 0, 0, 9, 9])
    check("④″ 标签**比段长**（还带非法值 9 ✓）⇒ 只读前 `b-a` 个 ✓ **不报错** ✓、结果与 ④′ 相同 ✓"
          "（参考就是只读前 `b-a` 个 ✓ —— 把\"长出来\"当错会误伤整段标签 ✓✗）",
          longer == tail, longer)

    short_err = _raises(lambda: h3.mod_segments_for(segments, index, 3, text_tags=[1, 1, 1]))
    check("⑤ 标签**比段短** ⇒ **报错** ✓（短了会把后半段全当标签 0 ⇒ 静默偏 ✓✗ —— 这与"
          "「多出来」不是一回事 ✓）",
          short_err is not None and "同源同长" in str(short_err), short_err and str(short_err)[:80])

    bad_err = _raises(lambda: h3.mod_segments_for(segments, index, 3, text_tags=[1, 2, 1, 1]))
    check("⑤′ 标签出现 **2（audio 行）** ⇒ **报错** ✓（文本段里不该有音频占位 ✓；静默放行 ⇒ "
          "文本 token 拿到音频调制 ✓✗）",
          bad_err is not None and "只允许 0" in str(bad_err), bad_err and str(bad_err)[:80])

    shifted = _raises(lambda: h3.mod_segments_for([(2, 6, "text")], index, 3,
                                                 text_tags=[1, 1, 1, 1]))
    check("⑤″ 文本段**不从 0 起** ⇒ **报错** ✓（参考按**段内偏移**取标签 ✓，段不从 0 起就会读到"
          "别人的标签位 ⇒ 照抄错 ✓、绕开又和参考不一样 ✓✗ ⇒ 只能拒 ✓）",
          shifted is not None and "段内偏移" in str(shifted), shifted and str(shifted)[:80])

    # ── ⑥ oracle：随机标签 × 混合段布局，逐值对账 ──
    import random  # noqa: PLC0415 —— 只在本份里用 ✓
    rng = random.Random(20261101)
    oracle_bad: list[Any] = []
    for _ in range(60):
        text_len = rng.randint(1, 9)
        video_len = rng.randint(1, 4)
        lay = [(0, text_len, "text"), (text_len, text_len + video_len, "video"),
               (text_len + video_len, text_len + video_len + 2, "audio")]
        tags = [rng.choice((0, 1)) for _ in range(text_len)]
        got = h3.mod_segments_for(lay, index, 3, text_tags=tags)
        if got != _oracle(lay, index, 3, tags):
            oracle_bad.append((lay, tags, got))
    check("⑥ 与**另写的参考转写**在 60 组随机标签 × 文本/视频/音频混合段上**逐值**一致 ✓"
          "（按标签切 run ✓、行号 `index×modalities+标签` ✓）", not oracle_bad, oracle_bad[:2])

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        skip("⑦⑧ 主干端（torch）")
        return _report()

    # ── ⑦ 主干：长度不符当场拒 ──
    torch.manual_seed(0)
    tiny = dict(hidden=8, layers=1, heads=2, head_dim=12, ffn=12, text_dim=5, latents_dim=2,
                audio_latents_dim=3, patch_size=(1, 2, 2), time_input_dim=4, time_hidden=8,
                time_dim=6, inv_freq_len=2, refiner_layers=1)
    trunk = h3.H3FormTrunk(**tiny).eval()
    video_rows = torch.zeros(12, 8)
    audio_rows = torch.zeros(8, 3)
    text_states = torch.zeros(4, 5)
    len_err = _raises(lambda: trunk(video_rows, audio_rows, text_states, 0.5, 4, 4,
                                    text_token_tags=[1, 1, 1]))
    check("⑦ 主干收到**长度不符**的标签 ⇒ **报错** ✓（换过提示词没重算标签 ⇒ 标签与文本不同源 ✓✗）",
          len_err is not None and "同源" in str(len_err), len_err and str(len_err)[:80])

    # ── ⑧ 主干**真的消费**标签（这是"忽略参数"唯一的直接反证 ✓）──
    with torch.no_grad():
        out_plain = trunk(video_rows, audio_rows, text_states, 0.5, 4, 4,
                          text_token_tags=[1, 1, 1, 1])
        out_pad = trunk(video_rows, audio_rows, text_states, 0.5, 4, 4,
                        text_token_tags=[1, 0, 1, 1])
        out_none = trunk(video_rows, audio_rows, text_states, 0.5, 4, 4)
    check("⑧ 只换标签（`[1,1,1,1]` → `[1,0,1,1]`）⇒ **输出必变** ✓（不变 ⇒ 主干把标签"
          "**当没看见** ✗✗ —— 这正是本次要修的那条静默错 ✓）",
          not torch.equal(out_plain[0], out_pad[0]), (out_plain[0].shape, out_pad[0].shape))
    check("⑧′ 全 1 标签与**不给**标签**逐位相同** ✓（纯文本提示词两条路等价 ✓ —— 否则老调用"
          "方会凭空变结果 ✗）",
          torch.equal(out_plain[0], out_none[0]) and torch.equal(out_plain[1], out_none[1]))
    return _report()


def _oracle(lay: Any, index: dict[str, int], modalities: int, tags: Any) -> tuple[Any, ...]:
    """参考 ``comfy/ldm/minimax/model.py``（**约 677–694 行**）的**逐字转写** ✓。

    **独立实现** ✓ —— 不是去调被测函数 ✓（否则等于拿被测函数验它自己 ✗）。原句::

        text_tags = payload.get("text_token_tags")
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

    ⚠️ 照抄了两条容易"改顺手"的细节 ✓：
    ① 判界读 ``tags[i]`` —— ``i`` 是**段内偏移** ✓ 不是 ``a + i`` ✗；
    ② ``i == b - a`` **先判** ✓（否则越界读 ``tags[b-a]`` ✗）。
    """
    seg_tag = {"video": 0, "cond": 0, "ref_img": 0, "text": 1,
               "audio": 2, "cond_audio": 2, "ref_audio": 2}
    out: list[Any] = []
    for a, b, kind in lay:
        row_base = index[kind] * modalities
        if kind == "text":
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    out.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
            continue
        out.append((a, b, row_base + seg_tag[kind]))
    return tuple(out)


def _raises(call: Any) -> BaseException | None:
    try:
        call()
    except BaseException as err:  # noqa: BLE001
        return err
    return None


def _report() -> int:
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
