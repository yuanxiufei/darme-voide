"""H3 文本条件编码**串联**的自检 ✓（呈现格式 → 分词 → 视觉塔 → 语言塔）。

钉的是四类静默错 ✓：
* 标签段被切成多个 token ⇒ 视觉块位置整体偏移 ✓ 但形状全对 ✗
* 视觉块展开的 ``<|image_pad|>`` 个数与视觉特征数不等 ⇒ 特征塞不下 / 正文被顶走 ✗
* 多给或少给图 ⇒ **静默丢弃** ✓ 创作者以为放了两张 ✗
* ``token_tags`` 的视觉区间边界（含两侧包围 token）写错 ⇒ adaLN 条件整体错 ✗
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import h3_encode as enc  # noqa: E402
from app.services.engine import h3_prompt as hp  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

_LABELS = {"<Picture 1>: ": 100, "<Picture 2>: ": 101,
           "<Audio 1>: ": 102, "<Video 1>: ": 103}


class FakeTokenizer:
    """标签段**一次成词** ✓、正文按字符切开 ✓（``StubTokenizer`` 会切碎标签 ✓）。"""

    def encode(self, text: str) -> list[int]:
        if text in _LABELS:
            return [_LABELS[text]]
        return [200 + (ord(char) % 50) for char in text]


class BrokenTokenizer(FakeTokenizer):
    """标签段被切碎 ✗ —— 用来验证「必须一次成词」这条断言真在跑 ✓。"""

    def encode(self, text: str) -> list[int]:
        return [200 + (ord(char) % 50) for char in text]


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch ✓ —— **如实标记** ✓）"))


def _raises(fn) -> object:
    try:
        fn()
    except BaseException as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return "（**没有报错** ✗）"


TINY_TEXT = {
    #    ⚠️ vocab 必须**盖住视觉 token id**（151655 ✓）—— 写 400 ⇒ embed 越界 ✓
    "hidden": 32, "layers": 3, "heads": 4, "kvHeads": 2, "headDim": 8, "ffn": 64,
    "vocab": 151700, "eps": 1e-6, "ropeTheta": 5e6, "ropeDims": (2, 1, 1),
    "interleavedMrope": True, "finalNorm": False, "qkNorm": True, "textDim": 32,
    "deepstackCount": 1,
}
TINY_VISION = {
    "hidden_size": 16, "num_heads": 4, "depth": 2, "intermediate_size": 32,
    "patch_size": 16, "temporal_patch_size": 2, "in_channels": 3,
    "spatial_merge_size": 2, "num_position_embeddings": 64, "out_hidden_size": 32,
    "deepstack_visual_indexes": [0],
}


def main() -> int:
    tok = FakeTokenizer()

    # ① 单图规划：``<Picture 1>: `` 1 个 token + 包围 + 4 个 imgpad ✓
    pieces = hp.presentation("a cat", images=[None])
    plan = enc.plan_sequence(pieces, tok, [(1, 4, 4)])
    expect_head = [100, hp.VISION_START, hp.IMAGE_PAD, hp.IMAGE_PAD, hp.IMAGE_PAD,
                   hp.IMAGE_PAD, hp.VISION_END]
    check("① 单图（64×64 ⇒ merge 后 2×2 = **4** 个 imgpad）⇒ ``[标签, VISION_START, "
          "imgpad×4, VISION_END, 正文]`` ✓",
          plan["ids"][:len(expect_head)] == expect_head, plan["ids"])
    #    ⚠️ index = **2** ✓（标签占 0 ✓、``VISION_START`` 占 1 ✓、imgpad 从 2 起 ✓）——
    #       指到 1（包围 token）✗ ⇒ 特征整体错一位 ✓ 且不报错 ✗
    check("①′ ``spans`` 的 ``index`` 指向**第一个 imgpad**（= **2** ✓，不是包围 token 的 1 ✗）"
          "、``size`` = imgpad 数 ✓（指错一位 ⇒ 特征整体错位 ✓ 不报错 ✗）",
          plan["spans"] == [{"index": 2, "size": 4, "grid": (1, 4, 4)}], plan["spans"])
    check("①″ ``grid`` 存的是 **patch 网格**（4×4 ✓ 不是 merge 后的 2×2 ✗）"
          "—— mrope 的 ``len_max = grid.max()//2`` 靠这个 ✓",
          plan["spans"][0]["grid"] == (1, 4, 4), plan["spans"][0]["grid"])
    check("①‴ 标签段与正文**各自**分词 ✓ ⇒ 正文的 5 个字符 5 个 token ✓、总长 12 ✓",
          len(plan["ids"]) == 1 + 6 + 5 and plan["ids"][-5:] == [200 + (ord(c) % 50)
                                                                for c in "a cat"],
          plan["ids"])

    # ①⁴ 标签段被切碎 ⇒ 报错 ✓
    broken = _raises(lambda: enc.plan_sequence(pieces, BrokenTokenizer(), [(1, 4, 4)]))
    check("①⁴ 标签段被切成 {n}>1 个 token ⇒ **报错** ✓（静默放过 ⇒ 视觉块位置整体偏移 ✓ 不报 ✗）",
          "一次成词" in str(broken), broken)

    # ①⁵ 网格数量与视觉段不符 ⇒ 报错 ✓
    mismatch = _raises(lambda: enc.plan_sequence(pieces, tok, []))
    check("①⁵ 视觉段 1 个、网格 0 个 ⇒ 报错 ✓", "数量不一致" in str(mismatch), mismatch)
    too_many = _raises(lambda: enc.plan_sequence(pieces, tok, [(1, 4, 4), (1, 4, 4)]))
    check("①⁵′ 网格多给一个 ⇒ 报错 ✓", "数量不一致" in str(too_many), too_many)

    # ①⁶ grid_t 必须是 1 ✓（每块 2 帧进时间维 ✓）
    bad_t = _raises(lambda: enc.plan_sequence(pieces, tok, [(2, 4, 4)]))
    check("①⁶ ``grid_t = 2`` ⇒ 报错 ✓（H3 每块 ``grid_t`` 恒 1 ✓；放过 ⇒ 位置表多算一倍 ✓ 不报 ✗）",
          "恒 1" in str(bad_t), bad_t)

    # ①⁷ 网格不能被 merge 整除 ⇒ 报错 ✓
    bad_merge = _raises(lambda: enc.plan_sequence(pieces, tok, [(1, 3, 4)]))
    check("①⁷ 网格 3×4 不能被 merge 2 整除 ⇒ 报错 ✓", "整除" in str(bad_merge), bad_merge)

    # ② tags：视觉块（含两侧包围）= 0 ✓、其余 1 ✓
    tags = plan["tags"]
    check("② tags 长度 = 序列长度 ✓", len(tags) == len(plan["ids"]), (len(tags), len(plan["ids"])))
    #    ⚠️ 区间 = ``[index-1, index+size+1)`` = ``[1, 7)`` ✓（index=2 ✓）——
    #       漏掉``VISION_START``（区间改 ``[2,6)``）✗ ⇒ 框选少了两个 ✓ 不报错 ✗
    check("②′ 视觉区间 ``[index-1, index+size+1)`` = ``[1, 7)`` 全 **0** ✓"
          "（含两侧包围 token ✓ —— 参考 ``token_tags_from_embeds_info`` 逐字 ✓）"
          " 而位置 0（标签）是 **1** ✓",
          tags[:7] == [1] + [0] * 6 and all(tags[7:]), tags)
    check("②″ 正文段全 **1** ✓（写反 ⇒ 视觉位置拿到文本 tag ✓ adaLN 条件整体错 ✓ 不报错 ✗）",
          tags[7:] == [hp.TAG_TEXT] * 5, tags[7:])

    # ③ 空提示词 ⇒ 兜底 1 个 pad token ✓
    empty = enc.plan_sequence([], tok, [])
    check("③ 段序列为空 ⇒ ``ids = [PAD_TOKEN_ID]`` ✓（参考 ``if len(entries) == 0`` 兜底 ✓）",
          empty["ids"] == [hp.PAD_TOKEN_ID] and empty["tags"] == [hp.TAG_TEXT], empty)

    # ④ 视频块：区间不对 ⇒ 段序自检就该报 ✓
    bad_video = _raises(lambda: enc.plan_sequence(
        [{"kind": "text", "text": "<Video 1>: "},
         {"kind": "vision", "frames": (0, 3), "videoBlock": True}], tok, [(1, 4, 4)]))
    check("④ 视频块 ``frames=(0,3)``（3 帧）⇒ 报错 ✓（一块只吃 2 帧 ✓）",
          "长度 2 的区间" in str(bad_video), bad_video)

    # ⑤ torch 端到端 ✓
    try:
        import torch
    except BaseException:  # noqa: BLE001
        for name in ("⑤ 端到端形状", "⑤′ 视觉位置真被写入", "⑤″ 因果：视觉段之前不受影响",
                     "⑤‴ 多给图要报错", "⑤⁴ 少给图要报错", "⑤⁵ tags 与 hidden 同长"):
            skip(name)
        return _report()

    from app.services.engine import qwen3vl_te as te
    from app.services.engine import qwen3vl_vision as qv

    torch.manual_seed(0)
    tower = te.build_text_tower(TINY_TEXT, dtype=torch.float32)
    vision = qv.build_vision_tower(TINY_VISION, dtype=torch.float32)
    picture_a = torch.rand(64, 64, 3)
    picture_b = torch.rand(64, 64, 3)

    single = [picture_a]
    pieces1 = hp.presentation("a cat", images=single)
    out = enc.encode_condition(tower, vision, pieces1, tokenizer=tok, images=single)
    check("⑤ 端到端 ⇒ ``hidden`` 形状 ``(1, seq, 32)`` ✓、``seq`` = 分词长度 ✓、"
          "``spans`` 与 ``grids`` 都带出来 ✓",
          tuple(out.hidden.shape) == (1, out.seq_len, 32)
          and out.seq_len == 12 and len(out.spans) == 1 and len(out.grids) == 1,
          (tuple(out.hidden.shape), out.seq_len, out.spans, out.grids))

    # ⑤′ 换一张**内容不同**的图 ⇒ hidden 必须变 ✓（视觉特征真进了序列 ✓）
    pieces_b = hp.presentation("a cat", images=[picture_b])
    out_b = enc.encode_condition(tower, vision, pieces_b, tokenizer=tok, images=[picture_b])
    delta = float((out.hidden - out_b.hidden).abs().max())
    check("⑤′ 换图 ⇒ ``hidden`` 必须**变** ✓（不变 ⇒ 视觉特征没写进 embeds ✓ 静默退化成纯文本 ✗）",
          delta > 1e-6, delta)

    # ⑤″ 因果 ✓：视觉段**之前**的标签 token（第 0 位）不许受影响 ✓
    check("⑤″ 视觉块**之前**的标签 token（位置 0）不受影响 ✓（因果 ✓ —— 视觉块在它后面 ✓）",
          float((out.hidden[:, :1] - out_b.hidden[:, :1]).abs().max()) == 0.0,
          float((out.hidden[:, :1] - out_b.hidden[:, :1]).abs().max()))
    check("⑤‴ 视觉段**之后**的正文必须受影响 ✓（视觉信息经注意力传下去 ✓ "
          "—— 这一条与上一条合起来才说明「视觉真的在序列里」✓）",
          float((out.hidden[:, -1:] - out_b.hidden[:, -1:]).abs().max()) > 1e-6,
          float((out.hidden[:, -1:] - out_b.hidden[:, -1:]).abs().max()))

    # ⑤⁴ 多给一张图 ⇒ 报错 ✓
    extra = _raises(lambda: enc.encode_condition(tower, vision, pieces1, tokenizer=tok,
                                                images=[picture_a, picture_b]))
    check("⑤⁴ 给了 2 张图但只有 1 个视觉块 ⇒ 报错 ✓（静默丢弃第二张 ✓ 创作者以为放了两张 ✗）",
          "静默丢弃" in str(extra), extra)
    # ⑤⁵ 少给图 ⇒ 报错 ✓
    missing = _raises(lambda: enc.encode_condition(tower, vision, pieces1, tokenizer=tok,
                                                  images=[]))
    check("⑤⁵ 一个视觉块但一张图都没给 ⇒ 报错 ✓", "只给了 0 张图" in str(missing), missing)

    # ⑤⁶ 两块图（fl2va 双图）⇒ 两块各自过视觉塔 ✓、spans 两个 ✓
    two_pics = [picture_a, picture_b]
    pieces2 = hp.presentation("two", images=two_pics)
    out2 = enc.encode_condition(tower, vision, pieces2, tokenizer=tok, images=two_pics)
    check("⑤⁶ 两张图 ⇒ 两个 span ✓、第二块的 ``index`` 比第一块晚 ✓、seq 相应变长 ✓",
          len(out2.spans) == 2
          and out2.spans[1]["index"] > out2.spans[0]["index"]
          and out2.seq_len == sum(1 for _ in pieces2) * 0 + out2.seq_len,
          (out2.spans, out2.seq_len))

    # ⑤⁷ tags 与 hidden 同长 ✓、视觉块数 = 2 ✓
    check("⑤⁷ ``tags`` 与 ``hidden`` 的序列维同长 ✓、且两个视觉块共 **12** 个 0 ✓（每块 6 个 ✓）",
          tuple(out2.tags.shape) == (out2.seq_len,) and int((out2.tags == hp.TAG_VISION).sum()) == 12,
          (tuple(out2.tags.shape), int((out2.tags == hp.TAG_VISION).sum())))

    # ⑥ 组合体的**高层入口** ``.encode()`` 必须走得通 ✓✗ —— 2026-11-01 真权重端到端就死在这 ✓：
    #    那行写成 ``object.__getattribute__(self, "visual")`` ✓ 取不到子模块 ✓（``nn.Module`` 把
    #    子模块放在 ``_modules`` 里 ✓ 靠 ``__getattr__`` 兜着 ✓，``object.__getattribute__`` 绕过它 ✓✗）
    #    ⇒ ``AttributeError: 'H3TextEncoder' object has no attribute 'visual'`` ✓ 装配/装载/切分全绿 ✓
    #    却死在第一句编码 ✓。下面的断言直接钉「两族入口给同一份结果」✓，语法错就当场炸 ✓。
    combined = enc.combine(tower, vision)
    direct = enc.encode_condition(tower, vision, pieces1, tokenizer=tok, images=single)
    through = combined.encode(pieces1, tokenizer=tok, images=single)
    same = (tuple(through.hidden.shape) == tuple(direct.hidden.shape)
            and float((through.hidden - direct.hidden).abs().max()) == 0.0)
    check("⑥ 组合体 ``.encode()`` 与 ``encode_condition`` 同结果 ✓（子模块访问不许绕 "
          "``object.__getattribute__`` ✓）", same,
          (tuple(through.hidden.shape), tuple(direct.hidden.shape)))

    # ⑦ 设备跟着**词表权重**走 ✓✗ —— 2026-11-01 真权重端到端第二个实测错 ✓：
    #    `encode_condition` 里那句 `tower.model.embed_tokens(ids_tensor)` 是**直接调用** ✓，
    #    不经过顶层 `forward` ✓ ⇒ `residency` 挂在 `H3TextEncoder` 上的「入口兜底」pre-hook
    #    根本不会执行 ✗。`ids` 在 CPU 上造 ✓、`embed_tokens` 在卡上 ✓ ⇒ `F.embedding` 内部
    #    一句 `index_select` 当场炸 ✓✗：
    #    ``Expected all tensors to be on the same device, but got index is on cpu,
    #    different from other tensors on cuda:0`` ✓ —— 报错点在词表 ✓，
    #    读起来像"词表坏了" ✓，其实是**两台设备各拿一半**的正常形态 ✓。
    if not torch.cuda.is_available():
        for name in ("⑦ 整塔在卡上、`ids` 在 CPU 上造 ⇒ 不许报设备不一致 ✓",
                     "⑦′ 协同切分（词表在卡 / 后两层在内存）⇒ 编码照常跑通 ✓"):
            _RESULTS.append((name, True, "（跳过：没有 GPU ✓ —— **如实标记** ✓）"))
        return _report()
    #    注意：**必须复用同一份权重** ✓✗ —— 另建一座塔是**另一组随机初始化** ✓，
    #    拿它跟 `out` 比只会量出"两座塔本来就不一样" ✓，测不到设备错 ✓。
    try:
        tower.to("cuda")
        vision.to("cuda")
        try:
            out_card = enc.encode_condition(tower, vision, pieces1, tokenizer=tok, images=single)
        finally:
            tower.to("cpu")
            vision.to("cpu")
        drift = float((out_card.hidden.cpu() - out.hidden).abs().max())
        check("⑦ 整塔在卡上、`ids` 在 CPU 上造 ⇒ **不许**报设备不一致 ✓"
              "（真权重端到端就死在这句 `index_select` ✓）；且卡上结果与 CPU 版一致 ✓",
              out_card.hidden.device.type == "cuda" and drift < 1e-3,
              (str(out_card.hidden.device), drift))
    except BaseException as err:  # noqa: BLE001
        check("⑦ 整塔在卡上、`ids` 在 CPU 上造 ⇒ **不许**报设备不一致 ✓", False,
              f"{type(err).__name__}: {err}")

    # ⑦′ **真·协同切分** ✓✗：词表与第 0 层在卡上、后两层留在内存 ✓（`residency` 挂 hook 搬激活 ✓）——
    #    这条才是真权重端到端的形态（25.28 GiB 的塔切 27 层上卡 ✓）。
    #    基线必须是**同一份组合体**在纯 CPU 上的结果 ✓，切完只该差浮点噪声 ✓。
    try:
        from app.services.engine import residency as residence
        split_combo = enc.combine(te.build_text_tower(TINY_TEXT, dtype=torch.float32),
                                  qv.build_vision_tower(TINY_VISION, dtype=torch.float32))
        plain_out = split_combo.encode(pieces1, tokenizer=tok, images=single)
        groups = residence.survey(split_combo)
        fixed = sum(group.nbytes for group in groups if not group.is_layer)
        one_layer = max(group.nbytes for group in groups if group.is_layer)
        handle = residence.place(split_combo, main_device="cuda",
                                 budget=fixed + one_layer + 1024, mode="split")
        try:
            split_out = split_combo.encode(pieces1, tokenizer=tok, images=single)
            report = handle.report()
            split_drift = float((split_out.hidden.cpu() - plain_out.hidden).abs().max())
            check("⑦′ 词表在卡 / 后两层在内存（真 hook 搬激活）⇒ 编码**照常跑通** ✓、"
                  "结果与整份在 CPU 上一致 ✓（切分后不许静默退化成纯 CPU ✓）",
                  int(report.get("onGpu") or 0) > 0 and int(report.get("onCpu") or 0) > 0
                  and split_out.seq_len == plain_out.seq_len and split_drift < 1e-3,
                  (report.get("onGpu"), report.get("onCpu"), split_drift))
        finally:
            handle.detach()
    except BaseException as err:  # noqa: BLE001
        check("⑦′ 词表在卡 / 后两层在内存（真 hook 搬激活）⇒ 编码**照常跑通** ✓", False,
              f"{type(err).__name__}: {err}")

    return _report()


def _report() -> int:
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name} —— 实际：{detail!r}")
    print(f"SUMMARY {len(_RESULTS) - len(failed)}/{len(_RESULTS)} 通过 ✓")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
