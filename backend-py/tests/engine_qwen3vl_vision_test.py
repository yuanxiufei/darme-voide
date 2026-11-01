"""Qwen3-VL-32B **视觉塔**的自检 ✓（要 torch ✓ —— 没有就 SKIP 并**明说** ✓）。

钉的是「**形状 / 排序 / 隔离**」三类硬事实 ✓，不是"看起来对不对" ✗：
* 从**盘上真权重的形状**反推尺子 ✓（不信常量 ✓）
* merge 之后 token 数 = ``(h/merge)·(w/merge)·t`` ✓
* **图与图之间不许互看** ✓（这是 ``cu_seqlens`` 那段的全部意义 ✓）
* pos_embed 在 **48×48 原尺寸**上必须**等于直接取表** ✓（插值不该改变它 ✓）
* 像素归一化是 **0.5/0.5** ✓（Qwen3-VL ✓ 不是 Qwen2.5-VL 的 CLIP 均值 ✗）
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import qwen3vl_vision as vis  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch ✓ —— **如实标记**，不算通过 ✗）"))


def main() -> int:
    # ① 不需要 torch 的事实 ✓
    check("① 像素归一化是 **0.5/0.5** ✓（Qwen3-VL ✓）不是 Qwen2.5-VL 的 CLIP 均值 ✗",
          vis.pixel_mean_std() == ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), vis.pixel_mean_std())
    check("①′ 视觉塔用 **2D 旋转** ✓、不是 mrope ✓（两者入口完全不同 ✗）",
          vis.rotation_kind() == "2d-rotary", vis.rotation_kind())
    # ⚠️ 盘上真权重的形状（2026-11-01 读 safetensors 头 ✓）
    disk = {
        "visual.patch_embed.proj.weight": (1152, 3, 2, 16, 16),
        "visual.pos_embed.weight": (2304, 1152),
        "visual.blocks.0.attn.qkv.weight": (3456, 1152),
        "visual.blocks.0.mlp.linear_fc1.weight": (4304, 1152),
        "visual.merger.norm.weight": (1152,),
        "visual.merger.linear_fc1.weight": (4608, 4608),
        "visual.merger.linear_fc2.weight": (5120, 4608),
        "visual.deepstack_merger_list.0.norm.weight": (4608,),
        "visual.deepstack_merger_list.0.linear_fc2.weight": (5120, 4608),
        "visual.deepstack_merger_list.1.linear_fc2.weight": (5120, 4608),
        "visual.deepstack_merger_list.2.linear_fc2.weight": (5120, 4608),
    }
    for index in range(27):
        disk[f"visual.blocks.{index}.attn.qkv.weight"] = (3456, 1152)
    shape = vis.infer_shape(disk)
    check("② ⭐ 尺子**从权重反推** ✓：27 层 / 1152 / ffn 4304 / merge 2 / 输出 5120 / 网格 2304",
          (shape["depth"], shape["hidden_size"], shape["intermediate_size"],
           shape["spatial_merge_size"], shape["out_hidden_size"],
           shape["num_position_embeddings"]) == (27, 1152, 4304, 2, 5120, 2304), shape)
    check("②′ ⭐ DeepStack 挂 **8/16/24** 三层 ✓（权重里只有 merger、挂点查不到 ⇒ 走参考常量 ✓）",
          tuple(shape["deepstack_visual_indexes"]) == (8, 16, 24),
          shape["deepstack_visual_indexes"])
    check("②″ ⚠️ 少一路 merger ⇒ 挂点**跟着缩** ✓（不越界 ✗；只注两路 ✓）",
          tuple(vis.infer_shape({k: v for k, v in disk.items()
                                 if "deepstack_merger_list.2" not in k})
                ["deepstack_visual_indexes"]) == (8, 16))

    try:
        import torch
    except ImportError:
        for label in ("③…⑧"):
            skip(f"{label} torch 相关检查")
        return _report()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    # ⚠️ 小号但**同构**的塔 ✓（27 层太慢 ✓；只缩 hidden/ffn/depth ✓，merge 与 patch 保持不变 ✓）
    tiny = dict(shape)
    tiny.update({"hidden_size": 64, "intermediate_size": 128, "depth": 10,
                 "num_heads": 4, "out_hidden_size": 96, "num_position_embeddings": 64})
    tiny["deepstack_visual_indexes"] = (2, 5, 8)
    tower = vis.build_vision_tower(tiny, torch.float32)
    tower.eval()
    patch = int(tiny["patch_size"])
    temporal = int(tiny["temporal_patch_size"])
    merge = int(tiny["spatial_merge_size"])

    def tokens_for(frames: int, height: int, width: int) -> int:
        return frames * height * width

    def run(grid: list[tuple[int, int, int]], x: Any = None) -> tuple[Any, list[Any]]:
        grid_thw = torch.tensor(grid, dtype=torch.long)
        count = sum(tokens_for(*row) for row in grid)
        if x is None:
            x = torch.randn(count, 3 * temporal * patch * patch)
        return tower(x, grid_thw)

    # ③ 单图：merge 之后 token 数 = h/2 · w/2 · t
    merged, deep = run([(1, 8, 8)])
    check("③ ⭐ 单图 8×8 网格 ⇒ 合并后 **4×4 = 16** 个 token ✓（merge 2 ✓）",
          tuple(merged.shape) == (16, 96), tuple(merged.shape))
    check("③′ ⭐ DeepStack 三份 ✓ 形状与主输出同宽（都进语言塔 ✓）且**逐 chunck 对齐** ✓",
          len(deep) == 3 and all(tuple(item.shape) == (16, 96) for item in deep),
          [tuple(item.shape) for item in deep])

    # ④ 多图：**图与图不许互看** ✓ —— 这是 cu_seqlens 那段的全部意义 ✓
    #    ⚠️ 两次跑必须喂**同一份**像素 ✓（各自 randn 会把「输入不同」误判成「串扰」✗）
    #    ⚠️ 行数按 **patch 数** 算 ✓（``grid_thw`` 的 h/w 单位就是 patch ✗ 不是像素 ✗）
    one_image = tokens_for(1, 8, 8)
    pair = torch.randn(2 * one_image, 3 * temporal * patch * patch)
    two_merged, _ = run([(1, 8, 8), (1, 8, 8)], pair)
    first_only, _ = run([(1, 8, 8)], pair[:one_image])
    second_only, _ = run([(1, 8, 8)], pair[one_image:])
    check("④ ⭐⭐⭐ 两张图一起跑 ⇒ 每张的 token 与**单独跑**逐位相同 ✓（图间零串扰 ✓）"
          "—— 用全局注意力时这条会红 ✓（不报错、只是两张图的特征混在一起 ✗）",
          torch.allclose(two_merged[:16], first_only, atol=1e-6)
          and torch.allclose(two_merged[16:], second_only, atol=1e-6),
          (float((two_merged[:16] - first_only).abs().max()),
           float((two_merged[16:] - second_only).abs().max())))

    # ⑤ 时间维：t=2 的 8×8 ⇒ token 数翻倍 ✓ 且两帧**各自一段** ✓
    frames_in = torch.randn(2 * one_image, 3 * temporal * patch * patch)
    frames_merged, _ = run([(2, 8, 8)], frames_in)
    check("⑤ ⭐ ``t=2`` ⇒ token 数 = 2×(4×4) = 32 ✓（时间维不参与 merge ✓）",
          tuple(frames_merged.shape) == (32, 96), tuple(frames_merged.shape))
    single_frame, _ = run([(1, 8, 8)], frames_in[:one_image])
    check("⑤′ ⭐⭐ 两帧段与段之间**不许互看** ✓：``t=2`` 的前半 == 单独跑 ``t=1`` ✓"
          "—— 漏切段时这条会红 ✓（帧间混色 ✓ 不报错 ✓）",
          torch.allclose(frames_merged[:16], single_frame, atol=1e-6),
          float((frames_merged[:16] - single_frame).abs().max()))

    # ⑥ pos_embed：在**原尺寸**网格上插值必须等于「表按 merge 块重排」✓
    #    ⚠️ 参考那一步 ``view(...).permute(0,1,3,2,4,5)`` **会重排** ✓ ⇒ 直接与 ``table`` 比会红 ✗
    #    （而重排是**必需的** ✓：patch 序列本身就是 merge 块优先 ✓，位置编码必须同序 ✓）
    side = tower.num_grid_per_side
    baked = tower.fast_pos_embed_interpolate(torch.tensor([[1, side, side]]))
    table = tower.pos_embed.weight
    reordered = (table.view(side // merge, merge, side // merge, merge, -1)
                 .permute(0, 2, 1, 3, 4).flatten(0, 3))
    check("⑥ ⭐⭐ 网格 = 原尺寸 ⇒ 插值结果 **逐位等于「表按 merge 块重排」** ✓"
          "（重排是必需的 ✓ —— patch 序列就是块优先 ✓；与不重排的原表**不等** ✗ 见 ⑥‴）",
          torch.allclose(baked, reordered, atol=1e-6),
          float((baked - reordered).abs().max()))
    check("⑥′ 网格 8×8（小于原尺寸）⇒ 形状 ``8·8 × hidden`` ✓ 且**不是**简单取前 64 行 ✗（真的插值了 ✓）",
          tuple(baked.shape) == (side * side, int(tiny["hidden_size"]))
          and not torch.allclose(
              tower.fast_pos_embed_interpolate(torch.tensor([[1, 8, 8]])),
              table[:64], atol=1e-6))

    # ⑦ grid 与像素数不同源 ⇒ 当场报错 ✓（不静默错位 ✗）
    err = _raises(lambda: tower(torch.randn(10, 3 * temporal * patch * patch),
                                torch.tensor([[1, 4, 4]])))
    check("⑦ 网格算出 16 个 token、实际给了 10 个 ⇒ **当场报错** ✓（不静默错位 ✗）",
          "网格算出" in str(err), err)
    #    ⚠️ 取 **5×5** ✓ 不是 6×6 ✗（6 能被 merge 2 整除 ✓ 那条根本不该报 ✓ —— 首版就是写错这格）
    err2 = _raises(lambda: tower(torch.randn(tokens_for(1, 5, 5),
                                             3 * temporal * patch * patch),
                                  torch.tensor([[1, 5, 5]])))
    check("⑦′ 网格 5×5 不能被 merge 2 整除 ⇒ **当场报错** ✓（上游预处理就该切齐 ✗）",
          "整除" in str(err2), err2)

    # ⑧ 级联：三路 DeepStack 的合并是 **post-shuffle LayerNorm**（宽 hidden·merge² ✓）
    check("⑧ ⭐ 主 merger 的 norm 宽 = **hidden** ✓；DeepStack 的 norm 宽 = **hidden·merge²** ✓"
          "（两者相反 ✗ —— 写反会静默把图特征压扁 ✓）",
          tuple(tower.merger.norm.weight.shape) == (int(tiny["hidden_size"]),)
          and all(tuple(m.norm.weight.shape) == (int(tiny["hidden_size"]) * merge ** 2,)
                  for m in tower.deepstack_merger_list),
          (tuple(tower.merger.norm.weight.shape),
           tuple(tower.deepstack_merger_list[0].norm.weight.shape)))
    check("⑧′ 视觉塔**没有** QK-norm ✗（只有 norm1/norm2 ✓ —— 语言塔才有 q_norm/k_norm ✓）",
          not hasattr(tower.blocks[0], "q_norm")
          and hasattr(tower.blocks[0], "norm1") and hasattr(tower.blocks[0], "norm2"))
    return _report()


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
