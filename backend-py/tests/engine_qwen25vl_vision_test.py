"""Qwen2.5-VL 视觉塔自检（Qwen-Image **Edit 2511** 的参考图编码器）。

判据分四层，**逐层抓不同的静默错**：

1. **盘上对账**：真权重文件（`qwen_2.5_vl_7b_fp8_scaled.safetensors`）的 ``visual.*`` 键
   （714 张量）与 :func:`qwen25vl_vision.expected_keys` **逐键相等** —— 「少接一层」时多出来的键
   会被丢掉而不报错，只有逐键对账能抓住；
2. **窗口重排**：``window_index`` 必须是**置换**，且与测试里独立写死的「按合并格算窗口」
   结果一致（不是拿被测实现自己跟自己比）；
3. **数值**：把「窗口重排 + 分段注意力」与「不重排 + 全图掩码注意力」两条**结构不同**的路
   放在**同一份权重、同一张图**上对拍 —— 掩码版是教科书写法、没有任何置换，
   两者一致才证明置换/位置/还原三件事都对（参考 435-457 行那三步最容易各错各的）；
4. **真权重装机**：从盘上读真实 fp8_scaled 权重跑一次前向（存在性守卫，缺文件则明说跳过）。

⚠️ 参考 (`comfy/text_encoders/qwen_vl.py::Qwen2VLVisionTransformer`) 的**五条硬事实**见
:mod:`qwen25vl_vision` 抬头；本文件第 ② / ③ 层就是把其中第 2、3、4 条钉住。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))
sys.stdout.reconfigure(encoding="utf-8")

from app.services.engine import qwen25vl_vision as vis  # noqa: E402

_TOTAL = 0
_FAILED = 0

WEIGHTS = os.environ.get(
    "ENGINE_QWEN25VL_WEIGHTS",
    r"D:/Comfy-Desktop/ComfyUI-Shared/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
)
#: 量化伴随件（fp8_scaled 每层两个标量）—— 对账时要剥掉，它们不是权重键
_SCALE_KEYS = (".scale_input", ".scale_weight")


def check(label: str, ok: bool, detail: object = None) -> None:
    global _TOTAL, _FAILED
    _TOTAL += 1
    if ok:
        print(f"  ok   {label}")
        return
    _FAILED += 1
    print(f"  FAIL {label} -> {detail!r}")


def _visual_keys() -> dict[str, tuple[int, ...]]:
    """读**头部**（不读字节）：``{键: 形状}``，只留 ``visual.`` 且剥掉量化伴随件。"""
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(WEIGHTS, "pt") as handle:
        shapes = {}
        for key in handle.keys():
            if not key.startswith("visual."):
                continue
            if key.endswith(_SCALE_KEYS):
                continue
            shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def case_header(known: bool) -> dict | None:
    print("[1] 盘上对账：714 个 visual 张量逐键认领")
    if not known:
        print(f"  skip 盘上没有 {WEIGHTS}（设 ENGINE_QWEN25VL_WEIGHTS 指向真文件才跑）")
        return None
    shapes = _visual_keys()
    #: 714 = **390 个真权重**（32 层 × 12 + patch_embed 1 + merger 5）+ **324 个量化伴随件**
    #: （32 层 × 5 个线性 × ``scale_input``/``scale_weight`` ++ merger 的 2 个线性 × 2）
    check("① 剥掉 scale 伴随件后 390 个真权重（32×12 + 1 + 5）", len(shapes) == 390, len(shapes))
    shape = vis.infer_shape(shapes)
    check("① 形状反推：hidden 1280 / patch 14 / t 2",
          (shape["hidden_size"], shape["patch_size"], shape["temporal_patch_size"]) == (1280, 14, 2),
          (shape["hidden_size"], shape["patch_size"], shape["temporal_patch_size"]))
    check("① 形状反推：depth 32 / ffn 3420 / merge 2 / out 3584",
          (shape["depth"], shape["intermediate_size"], shape["spatial_merge_size"],
           shape["out_hidden_size"]) == (32, 3420, 2, 3584),
          (shape["depth"], shape["intermediate_size"], shape["spatial_merge_size"],
           shape["out_hidden_size"]))
    check("① 尺子自洽（head_dim 80 能被 4 整除 / qkv 行数 = 3×hidden）",
          vis.shape_problems(shape) == [], vis.shape_problems(shape))
    expected = vis.expected_keys(shape)
    check("① **零多余、零遗漏**：应占键集合 == 盘上键集合",
          expected == set(shapes),
          {"missing": sorted(expected - set(shapes))[:5],
           "extra": sorted(set(shapes) - expected)[:5]})
    check("① num_heads 只能按表登记（权重里没有这一维）—— 16 头",
          shape["num_heads"] == 16 and shape["hidden_size"] // shape["num_heads"] == 80,
          shape["num_heads"])
    return shape


def _expected_windows(grid: tuple[int, int], window: int, merge: int) -> list[int]:
    """**独立**算窗口重排（合并格为单位的行优先 → 窗口优先），不调被测实现。"""
    height, width = grid
    rows, cols = height // merge, width // merge
    order: list[int] = []
    for block_row in range(0, rows, window):
        for block_col in range(0, cols, window):
            for row in range(block_row, min(block_row + window, rows)):
                for col in range(block_col, min(block_col + window, cols)):
                    order.append(row * cols + col)
    return order


def case_window() -> None:
    print("[2] 窗口重排：必须是置换，且与独立算出的顺序一致")
    import torch  # noqa: PLC0415

    shape = dict(vis.VISION_25VL)
    window = shape["window_size"] // shape["spatial_merge_size"] // shape["patch_size"]
    check("② 窗口在合并域是 4 格（112 // 2 // 14）", window == 4, window)

    for grid in ((4, 4), (12, 8), (8, 12), (20, 16)):
        index, cu = vis.window_plan(torch.tensor([[1, grid[0], grid[1]]]), shape)
        total = (grid[0] // 2) * (grid[1] // 2)
        check(f"② {grid[0]}×{grid[1]}：重排是置换（{total} 个合并格一个不差）",
              sorted(index.tolist()) == list(range(total)),
              {"total": total, "got": index.tolist()[:8]})
        check(f"② {grid[0]}×{grid[1]}：顺序与独立算法一致",
              index.tolist() == _expected_windows(grid, window, 2),
              {"got": index.tolist()[:12], "want": _expected_windows(grid, window, 2)[:12]})
        check(f"② {grid[0]}×{grid[1]}：段表末值 = 合并格数 × 4（窗口内 token 数）",
              cu[-1] == total * 4, {"cu": cu[-3:], "want": total * 4})

    #: 4×4（合并域 2×2）放得下一个窗口 ⇒ **不该**有重排（顺序原样）
    index, _cu = vis.window_plan(torch.tensor([[1, 4, 4]]), shape)
    check("② 一窗装得下的图不重排（4×4 ⇒ 恒等）", index.tolist() == [0, 1, 2, 3],
          index.tolist())


def _tiny_shape() -> dict:
    shape = dict(vis.VISION_25VL)
    shape.update({"hidden_size": 32, "intermediate_size": 48, "depth": 4, "num_heads": 4,
                  "out_hidden_size": 16, "fullatt_block_indexes": (1, 3)})
    return shape


def _naive_forward(tower, patches, grid_thw, window: int | None = None):
    """教科书写法：**不重排**，窗口边界用全图布尔掩码表达（与「置换 + 分段」是两个独立结构）。

    ``window`` 只作**反向验证**用 ✓：给它一个错的窗口大小 ✓，输出必须偏离 ✓
    —— 否则 ③ 对「窗口边界划在哪」是瞎的 ✓✗（两条路都用同一个错的划分，照样一致 ✓）。
    """
    import torch  # noqa: PLC0415

    hidden = tower.patch_embed(patches)
    pos = tower.positions(grid_thw)
    pos = torch.cat((pos, pos), dim=-1)
    cos, sin = pos.cos(), pos.sin()
    merge = tower.merge
    if window is None:
        window = tower.shape["window_size"] // merge // tower.shape["patch_size"]
    _, height, width = (int(value) for value in grid_thw[0].tolist())
    rows, cols = height // merge, width // merge
    #: 每个合并格的窗口号（独立算）+ 每格展开成 merge² 个 token
    window_id = [(row // window) * ((cols + window - 1) // window) + col // window
                 for row in range(rows) for col in range(cols)]
    ids = torch.tensor(window_id).repeat_interleave(merge * merge)
    image_ids = torch.zeros_like(ids)
    seq = int(hidden.shape[0])
    for layer_index, block in enumerate(tower.blocks):
        key = ids if layer_index not in tower.fullatt else image_ids
        #: ⚠️ 掩码必须是 ``(1, seq, seq)`` **三维** ✗✗：写成 ``(1, 1, seq, seq)`` 时
        #: `masked_fill` 会把 ``(heads, seq, seq)`` 广播成 ``(1, heads, seq, seq)`` ✓，
        #: 随后 `transpose(0, 1).reshape(seq, -1)` 按四维压平 ✓✗ —— **形状全对** ✓、
        #: head 与 token 混着压 ✓，输出差 0.4~0.6 ✓ 而报错/日志看起来像「注意力算错了」✓✗。
        mask = (key.unsqueeze(0) == key.unsqueeze(1)).unsqueeze(0)
        normed = block.norm1(hidden)
        qkv = block.attn.qkv(normed)
        q, k, v = qkv.reshape(seq, 3, block.attn.heads,
                              block.attn.head_dim).permute(1, 0, 2, 3).unbind(0)
        q = (q * cos.unsqueeze(-2)) + (block.attn._rotate_half(q) * sin.unsqueeze(-2))
        k = (k * cos.unsqueeze(-2)) + (block.attn._rotate_half(k) * sin.unsqueeze(-2))
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) / (block.attn.head_dim ** 0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        merged = torch.softmax(scores, dim=-1) @ v          # (heads, seq, head_dim)
        hidden = hidden + block.attn.proj(merged.transpose(0, 1).reshape(seq, -1))
        hidden = hidden + block.mlp(block.norm2(hidden))
    return tower.merger(hidden)


def case_numeric() -> None:
    print("[3] 数值：置换+分段注意力 == 不置换+全图掩码（同一份权重）")
    import torch  # noqa: PLC0415

    shape = _tiny_shape()
    torch.manual_seed(7)
    tower = vis.build_vision_tower(shape, dtype=torch.float32)

    pixels = torch.rand(84, 84, 3)
    patches, grid = vis.patches_of(pixels, shape)
    got = tower(patches, torch.tensor([list(grid)]))
    check("③ 输出 token 数 = 合并格数（84/14 = 6 ⇒ 6×6/4 = 9）", tuple(got.shape) == (9, 16),
          tuple(got.shape))

    #: ⚠️ **必须用跨窗口的尺寸** ✗✗：84×84 只有 3×3 个合并格 ✓ 不满一窗（4 格 = 2×2）✓ ⇒
    #: `window_plan` 退化成**恒等置换** ✓、「重排对不对」这条**根本没被测到** ✓✗（旧版即如此 ✓）。
    #: ⚠️ 而且**宽度方向也要跨** ✗✗：112×224 ⇒ 8×16 patch ⇒ 合并域 4 行 × 8 列 ✓，
    #: 窗宽 4 ⇒ 横向 2 窗 ✓；换成 168×112（合并域 6×4 ✓ 列数**正好等于**窗宽 ✓）虽然段表
    #: 也是 2 窗 ✓，但行优先顺序**本来就等于**窗口序 ✓ ⇒ 置换仍是恒等 ✓✗、白测 ✓。
    wide = torch.rand(112, 224, 3)
    patches_w, grid_w = vis.patches_of(wide, shape)
    grid_w_thw = torch.tensor([list(grid_w)])
    index, cu = vis.window_plan(grid_w_thw, shape)
    check("③ 这张图真的跨窗口（段表 3 个端点、置换**不是**恒等）",
          len(cu) == 3 and index.tolist() != sorted(index.tolist()),
          {"cu": cu, "index": index.tolist()[:10]})

    got_w = tower(patches_w, grid_w_thw)
    want_w = _naive_forward(tower, patches_w, grid_w_thw)
    check("③ 输出 token 数 = 合并格数（4×8 = 32）", tuple(got_w.shape) == (32, 16),
          tuple(got_w.shape))
    diff = float((got_w - want_w).abs().max())
    check("③ 两条结构不同的路逐元素一致（< 1e-5，f32）—— 置换+分段 vs 不置换+掩码",
          diff < 1e-5, diff)

    #: 反向：把窗口换成「一个合并格一窗」⇒ 注意力范围变小 ⇒ 必须**偏离** ✓。
    #: 不偏离说明 ③ 只证明了「同一套划分前后一致」✓，对「划分划在哪」是瞎的 ✓✗。
    broken = _naive_forward(tower, patches_w, grid_w_thw, window=1)
    gap = float((broken - got_w).abs().max())
    check("③ 反向：窗口划分改成「一格一窗」⇒ 输出必须变（否则 ③ 对窗口边界是瞎的）",
          gap > 1e-3, gap)


def case_real(shape: dict | None) -> None:
    print("[4] 真权重装机：从盘上 fp8_scaled 读 visual.* 跑一次前向")
    if shape is None:
        print("  skip 权重文件不在盘上")
        return
    import torch  # noqa: PLC0415
    from app.services.engine import weights as weights_mod  # noqa: PLC0415

    tower = vis.build_vision_tower(shape, dtype=torch.bfloat16)
    report = weights_mod.load_module_weights(
        tower, WEIGHTS, dtype=torch.bfloat16, key_map={"^visual\\.": ""})
    check("④ 装载无错（fp8_scaled 走反量化；scale 伴随件被吃掉）", report.error == "",
          report.error)
    #: ⚠️ 判据盯的是**模块自有**的 390 个键 ✓✗（714 是盘上 ``visual.*`` 的张量数 ✓，
    #: 含 324 个 fp8 配套 ``scale_input``/``scale_weight`` ✓ —— 它们被**吃掉** ✓、不进模块 ✓）。
    #: 旧版写成 ``getattr(report, "loaded", None) == 714`` ✓：字段名不对 ⇒ 恒 ``None`` ⇒ 永远红 ✓✗，
    #: 而"红"看起来像真缺陷 ✓。字段真名是 ``loadedKeys`` ✓（见 `weights.WeightLoadReport` ✓）。
    check("④ 模块自有的 390 个键**全部**被填（一个都不许留随机初始化）",
          report.complete and report.targetKeys == 390 and report.loadedKeys == 390,
          {"complete": report.complete, "loadedKeys": report.loadedKeys,
           "targetKeys": report.targetKeys, "missing": report.missing[:3]})
    pixels = torch.linspace(0.0, 1.0, 84 * 84 * 3).reshape(84, 84, 3)
    patches, grid = vis.patches_of(pixels, shape)
    first = tower(patches, torch.tensor([list(grid)]))
    check("④ 输出形状 (9, 3584) 且全有限（(6×6)/4 = 9 个 token）",
          tuple(first.shape) == (9, 3584) and bool(torch.isfinite(first).all()),
          tuple(first.shape))
    second = tower(patches, torch.tensor([list(grid)]))
    check("④ 两次前向逐元素相同（确定性；不同 ⇒ 有隐藏状态串味）",
          bool(torch.equal(first, second)), None)
    spread = float(first.float().std())
    check("④ 输出不是常数/全零（真权重跑通的最低证据）", spread > 1e-3, spread)


def main() -> int:
    known = os.path.isfile(WEIGHTS)
    print(f"权重：{WEIGHTS}（{'在' if known else '不在'}）")
    shape = case_header(known)
    case_window()
    case_numeric()
    case_real(shape)
    print(f"SUMMARY: {_TOTAL - _FAILED}/{_TOTAL} passed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
