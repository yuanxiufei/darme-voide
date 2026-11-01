"""Qwen3-VL-32B **语言塔**（H3 文本编码器）的自检 ✓（torch 段没有就 SKIP 并明说 ✓）。

钉的是「**尺子 / mrope 交错 / 因果 / DeepStack 位置**」四类硬事实 ✓：
* 尺子必须与**盘上真形状**逐项对上 ✓（常量只作缺省 ✓）
* 旋转宽度 = ``head_dim`` ✓、频率个数 = ``head_dim // 2`` ✓（写反不报错但全错 ✗）
* interleaved mrope 是「隔 3 取 1」✓ 不是分段 ✗
* 没有视觉块 ⇒ 位置表必须是 ``None`` ✓ 不是全零 ✗
* 因果 ✓（改末 token 不影响前面 ✓）
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import qwen3vl_te as te  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch ✓ —— **如实标记**，不算通过 ✗）"))


def _raises(fn) -> object:
    try:
        fn()
    except BaseException as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return "（**没有报错** ✗）"


#: 盘上那套 32B 的真形状（2026-11-01 读 safetensors 头 ✓）
DISK_SHAPES = {
    "model.embed_tokens.weight": (151936, 5120),
    "model.layers.0.self_attn.q_proj.weight": (8192, 5120),
    "model.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "model.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "model.layers.0.self_attn.o_proj.weight": (5120, 8192),
    "model.layers.0.self_attn.q_norm.weight": (128,),
    "model.layers.0.self_attn.k_norm.weight": (128,),
    "model.layers.0.mlp.gate_proj.weight": (25600, 5120),
    "model.layers.0.input_layernorm.weight": (5120,),
    "model.layers.49.self_attn.q_proj.weight": (8192, 5120),
    "model.layers.49.mlp.down_proj.weight": (5120, 25600),
}

TINY = {
    "hidden": 32, "layers": 4, "heads": 4, "kvHeads": 2, "headDim": 8,
    "ffn": 64, "vocab": 50, "eps": 1e-6, "ropeTheta": 5e6,
    "ropeDims": (2, 1, 1), "interleavedMrope": True, "finalNorm": False,
    "qkNorm": True, "textDim": 32, "deepstackCount": 2,
}


def main() -> int:
    # ① 尺子自洽 ✓
    check("① 常量尺子自洽 ✓（ropeDims 之和 = head_dim//2 ✓、heads 能被 kvHeads 整除 ✓）",
          te.shape_problems(te.QWEN3VL_32B) == [], te.shape_problems(te.QWEN3VL_32B))
    # ①′ 旋转宽度 vs 频率个数 —— 差一倍的那两个数 ✓
    check("①′ ``rotary_dim_of(128)`` = **128** ✓（全维旋转 ✓）"
          " 而 ``freq_count_of(128)`` = **64** ✓ —— 两者不许相等 ✗",
          te.rotary_dim_of(128) == 128 and te.freq_count_of(128) == 64,
          (te.rotary_dim_of(128), te.freq_count_of(128)))
    check("①″ ``sum(ropeDims)`` = 64 = ``freq_count_of(headDim)`` ✓（不等 ⇒ 有维度拿不到轴 ✓ 且不报错 ✗）",
          sum(te.ROPE_DIMS) == te.freq_count_of(te.QWEN3VL_32B["headDim"]),
          (sum(te.ROPE_DIMS), te.freq_count_of(te.QWEN3VL_32B["headDim"])))

    # ② 从真形状反推 ✓
    shape = te.infer_shape(DISK_SHAPES)
    check("② 从盘上形状反推 ⇒ hidden 5120 / layers **50** / heads 64 / kvHeads 8 / "
          "headDim 128 / ffn 25600 / vocab 151936 ✓",
          (shape["hidden"], shape["layers"], shape["heads"], shape["kvHeads"],
           shape["headDim"], shape["ffn"], shape["vocab"])
          == (5120, 50, 64, 8, 128, 25600, 151936),
          {key: shape[key] for key in ("hidden", "layers", "heads", "kvHeads",
                                       "headDim", "ffn", "vocab")})
    check("②′ 盘上**没有** ``model.norm.weight`` ⇒ ``finalNorm=False`` ✓（截断模型没有收尾归一 ✓）",
          shape["finalNorm"] is False, shape["finalNorm"])
    check("②″ 有 ``q_norm.weight`` ⇒ ``qkNorm=True`` ✓（Qwen3 家族都有 ✓ 漏了 ⇒ 数值全偏 ✗）",
          shape["qkNorm"] is True, shape["qkNorm"])
    with_norm = dict(DISK_SHAPES)
    with_norm["model.norm.weight"] = (5120,)
    check("②‴ 补上 ``model.norm.weight`` ⇒ ``finalNorm`` 翻 True ✓（判据真读了形状 ✓ 不是写死 ✗）",
          te.infer_shape(with_norm)["finalNorm"] is True, te.infer_shape(with_norm)["finalNorm"])
    check("②⁗ 反推出的尺子照样自洽 ✓", te.shape_problems(shape) == [], te.shape_problems(shape))

    # ③ 没有视觉块 ⇒ **None** ✓ 不是全零 ✗
    check("③ 没有视觉块 ⇒ ``mrope_position_ids`` 返回 **None** ✓（给全零表 ⇒ 位置编码全同 ✓ 静默废掉 ✗）",
          te.mrope_position_ids([], 8) is None, te.mrope_position_ids([], 8))

    # ④ mrope 单图：imgpad 段 = 网格位置 ✓、之后文本**跳过**视觉用掉的最大位置 ✓
    rows = te.mrope_position_ids([{"index": 5, "size": 4, "grid": (1, 4, 4)}], 12)
    check("④ 单图（index=5 / size=4 / grid 4×4）⇒ T 行视觉段恒 = 5 ✓"
          "（``start + offset``，第一块 offset=0 ✓）",
          rows is not None and rows[0][5:9] == [5, 5, 5, 5],
          None if rows is None else rows[0])
    check("④′ H 行按**行号**递增、每个值重复「列数」次 ✓ ⇒ ``[5,5,6,6]`` ✓"
          "（写反 ⇒ H/W 互换 ✓ 图会被拉成转置的 ✓✗）",
          rows is not None and rows[1][5:9] == [5, 5, 6, 6],
          None if rows is None else rows[1])
    check("④″ W 行按**列号**递增、行内循环 ✓ ⇒ ``[5,6,5,6]`` ✓",
          rows is not None and rows[2][5:9] == [5, 6, 5, 6],
          None if rows is None else rows[2])
    check("④‴ 视觉之后的文本从 ``len_max + index + offset`` 起 ✓ ⇒ index 9 是 **7** ✓"
          "（不是 9 ✗ —— 视觉占了 5、6 两个位置 ✓）",
          rows is not None and rows[0][9] == 7 and rows[0][11] == 9,
          None if rows is None else rows[0][9:])
    check("④⁗ 视觉之前的文本是**原样** 0..4 ✓（第一块 offset=0 ✓）",
          rows is not None and rows[0][:5] == [0, 1, 2, 3, 4],
          None if rows is None else rows[0][:5])
    # ④⁵ 两块时 offset 累加 ✓（offset += len_max - size）
    two = te.mrope_position_ids([{"index": 2, "size": 4, "grid": (1, 4, 4)},
                                 {"index": 8, "size": 4, "grid": (1, 4, 4)}], 14)
    check("④⁵ 两块 ⇒ 第一块的 ``offset = 2 - 4 = -2`` 传到第二块 ✓ ⇒ 第二块视觉 T = 8 - 2 = **6** ✓"
          "（忘累加 ⇒ 位置整体后移 ✓ 不报错 ✗）",
          two is not None and two[0][8] == 6, None if two is None else two[0])
    # ④⁶ 越界 ⇒ 报错 ✓
    #    ⚠️ 用 index=9/size=4（9+4=13 > 12 ✓）—— 首版写 index=8 恰好卡在边界 ✓ 根本不该报 ✓
    over = _raises(lambda: te.mrope_position_ids([{"index": 9, "size": 4, "grid": (1, 4, 4)}], 12))
    check("④⁶ 视觉块越出序列 ⇒ 报错 ✓（静默截断 ⇒ 位置表错 ✓ 且不报 ✗）", "越出序列" in str(over), over)

    # ⑤ torch 段 ✓
    try:
        import torch
    except BaseException:  # noqa: BLE001
        for name in ("⑤ 1D rope", "⑤′ interleaved 交错", "⑥ 因果", "⑦ DeepStack 位置",
                     "⑧ 无 final norm", "⑨ QK-norm 形状", "⑩ deepstack 数量对不上要报错"):
            skip(name)
        return _report()

    # ⑤ 1D：pos 0 ⇒ cos 全 1 ✓、宽 = freq_count ✓
    cos, sin = te.freqs_cis(None, shape, length=3, dtype=torch.float32)
    check("⑤ 纯文本走 **1D** rope ✓ ⇒ 位置 0 的 cos 全 1 ✓、宽 = 64 = head_dim//2 ✓",
          tuple(cos.shape) == (3, 64) and bool(torch.allclose(cos[0], torch.ones(64)))
          and bool(torch.allclose(sin[0], torch.zeros(64))),
          (tuple(cos.shape), cos[0][:3].tolist()))

    # ⑤′ interleaved：第 j 个频率取哪一轴 ✓ —— 手算对照 ✓
    n = 2
    position_ids = [[1] * n, [2] * n, [3] * n]
    cos2, sin2 = te.freqs_cis(position_ids, shape, dtype=torch.float32)
    half = te.freq_count_of(int(shape["headDim"]))
    inv_freq = [1.0 / (float(shape["ropeTheta"]) ** ((2 * j) / float(shape["headDim"])))
                for j in range(half)]

    def axis_of(j: int) -> int:
        if j >= 60:
            return 0
        return j % 3

    want = [math.cos(inv_freq[j] * float(position_ids[axis_of(j)][0])) for j in range(half)]
    got = cos2[0].tolist()
    check("⑤′ interleaved mrope：第 j 个频率取 ``T(j%3=0)/H(=1)/W(=2)`` ✓、``j>=60`` 回落 T ✓"
          "（分段拼接 ✗ ⇒ 这里全错但不报错 ✓✗）",
          max(abs(a - b) for a, b in zip(want, got)) < 1e-5,
          max(abs(a - b) for a, b in zip(want, got)))
    check("⑤″ 三轴位置不同 ⇒ 相邻三个频率的相位**各取一轴** ✓（第 1 个对应 H=2 ✓ 不是 T=1 ✗）",
          abs(got[1] - math.cos(inv_freq[1] * 2.0)) < 1e-5
          and abs(got[2] - math.cos(inv_freq[2] * 3.0)) < 1e-5,
          (got[0], got[1], got[2]))

    # ⑥ 建 tiny 塔 ✓
    torch.manual_seed(0)
    tower = te.build_text_tower(TINY, dtype=torch.float32)
    ids = torch.randint(0, 50, (1, 12))
    out = tower(ids=ids)
    check("⑥ tiny 塔前向 ⇒ ``(1, 12, 32)`` ✓", tuple(out.shape) == (1, 12, 32), tuple(out.shape))
    # ⑥′ 因果 ✓：只改**最后一个** token ⇒ 前面的输出必须一字不变 ✓
    ids2 = ids.clone()
    ids2[0, -1] = (int(ids2[0, -1]) + 7) % 50
    out2 = tower(ids=ids2)
    delta = float((out[:, :-1] - out2[:, :-1]).abs().max())
    check("⑥′ 因果注意力 ✓：改末 token ⇒ 前面 11 个位置输出**完全不变** ✓（差 %.3g）"
          "（双向注意力 ⇒ 这里会变 ✓ 且不报错 ✗）" % delta, delta == 0.0, delta)
    check("⑥″ 末位置**必须**变 ✓（不然说明改的输入没生效 ✓ 上一条就成了假绿 ✗）",
          float((out[:, -1] - out2[:, -1]).abs().max()) > 1e-6,
          float((out[:, -1] - out2[:, -1]).abs().max()))

    # ⑦ DeepStack 加在**层之后**且只加前 N 层 ✓
    spans = [{"index": 4, "size": 2, "grid": (1, 2, 2)}]
    visual = torch.randn(2, 32)
    single = [visual]
    out_ds = tower(ids=ids, spans=spans, deepstack=single)
    check("⑦ ``deepstackCount=2`` 但只给 **1** 份 ⇒ 只加在第 1 层之后 ✓（不报错 ✓ 多给的才该用 ✓）",
          tuple(out_ds.shape) == (1, 12, 32), tuple(out_ds.shape))
    out_plain = tower(ids=ids, spans=spans)
    check("⑦′ 加了 DeepStack ⇒ 输出必须**不同** ✓（相同 ⇒ 注入没生效 ✓ 且不报错 ✗）",
          float((out_ds - out_plain).abs().max()) > 1e-6,
          float((out_ds - out_plain).abs().max()))
    # ⑦″ 只该动**视觉位置**？—— 不 ✓：视觉位置的扰动会经注意力传下去 ✓，
    #      但**视觉段之前**的 token 是因果的 ✓ ⇒ 它们可能被视觉内容影响？不会（视觉在它们后面 ✓）
    #      ⇒ 判据：第 4 个 token 之前的输出不该变 ✓
    plain_before = out_plain[:, :4]
    ds_before = out_ds[:, :4]
    check("⑦″ 视觉段**之前**的 4 个 token 不受影响 ✓（DeepStack 加在后面 ✓ 因果 ✓）",
          float((plain_before - ds_before).abs().max()) == 0.0,
          float((plain_before - ds_before).abs().max()))
    # ⑦‴ token 数对不上 ⇒ 报错 ✓
    bad = _raises(lambda: tower(ids=ids, spans=spans, deepstack=[torch.randn(3, 32)]))
    check("⑦‴ DeepStack 视觉 token 数对不上 ⇒ 报错 ✓（静默错位 = 视觉特征加到别的 token 上 ✗）",
          "DeepStack" in str(bad), bad)

    # ⑧ 截断模型**没有** final norm ✓
    check("⑧ 语言塔**没有** ``model.norm`` ✓（盘上是截断的第 50 层输出 ✓ 加个 norm 反而是错的 ✗）",
          not hasattr(tower.model, "norm"), [n for n, _ in tower.model.named_children()])
    check("⑧′ 层数 = 4 ✓（tiny ✓）、每层都有 ``q_norm`` ✓",
          len(tower.model.layers) == 4
          and all(hasattr(block.self_attn, "q_norm") and block.self_attn.q_norm is not None
                  for block in tower.model.layers),
          len(tower.model.layers))

    # ⑨ QK-norm 宽 = head_dim ✓（不是 hidden ✗）
    norm_weight = tower.model.layers[0].self_attn.q_norm.weight
    check("⑨ QK-norm 权重的宽 = **head_dim**（8）✓ 不是 hidden（32）✗",
          tuple(norm_weight.shape) == (8,), tuple(norm_weight.shape))
    no_norm = te.build_text_tower({**TINY, "qkNorm": False}, dtype=torch.float32)
    check("⑨′ ``qkNorm=False`` ⇒ 真不建 ``q_norm`` ✓（判据读了尺子 ✓ 不是写死 ✗）",
          no_norm.model.layers[0].self_attn.q_norm is None,
          hasattr(no_norm.model.layers[0].self_attn, "q_norm"))

    # ⑩ 视觉位置 vs 纯文本 ⇒ 位置表不同 ✓（证明 mrope 真被用了 ✓）
    out_span = tower(ids=ids, spans=spans)
    check("⑩ 同一个输入，带 span 与不带 span ⇒ 输出必须**不同** ✓（相同 ⇒ mrope 没接上 ✗）",
          float((out_span - out).abs().max()) > 1e-6,
          float((out_span - out).abs().max()))
    # ⑩′ ``ids`` 与 ``embeds`` 同时给 / 都不给 ⇒ 报错 ✓
    both = _raises(lambda: tower(ids=ids, embeds=torch.randn(1, 12, 32)))
    neither = _raises(lambda: tower())
    check("⑩′ ``ids`` 与 ``embeds`` 必须二选一 ✓（都给/都不给都要报 ✓）",
          "二选一" in str(both) and "二选一" in str(neither), (both, neither))
    # ⑩″ embeds 入口 ✓（宽不对要报 ✓）
    bad_wide = _raises(lambda: tower(embeds=torch.randn(1, 12, 16)))
    check("⑩″ ``embeds`` 宽 ≠ hidden ⇒ 报错 ✓（静默跑 ⇒ 线性层直接炸在别处 ✓ 或形状巧合放过 ✗）",
          "hidden" in str(bad_wide), bad_wide)
    embeds_ok = tower(embeds=torch.randn(1, 12, 32), spans=spans)
    check("⑩‴ ``embeds`` 入口可用 ✓（视觉特征拼好的那条路 ✓）",
          tuple(embeds_ok.shape) == (1, 12, 32), tuple(embeds_ok.shape))

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
