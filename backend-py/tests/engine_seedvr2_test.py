"""自检：**SEEDVR2 / NaDiT 主干的形状反推与前向口径**（2026-11-01 ✓）。

这一族四处「形状全对、只有数值错」的陷阱 ✗✗：``ada`` 的 ``emb`` 切片组号、
``mlp`` 的 SwiGLU 门控顺序、``attn_norm`` 有没有 affine、末块 ``vid_only`` 的文本分支。
本文件用**小号合成几何**（3 块 / dim=64）钉形状反推 ＋ 前向守门 ＋ 结构口径 ✓；
真实 3B 权重只读 safetensors **头**（不加载数据）核对几何与参数量 ✓。
⚠️ 与参考实现的逐位对拍（``maxdiff 0``，含真实 fp16 权重）是人为验收那一步 ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_seedvr2_test.py
"""
from __future__ import annotations

import math
import os
import sys
from functools import partial
from pathlib import Path

import torch

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

# ⚠️ cp936 控制台下第一个 ``✓``／``⑩`` 就 UnicodeEncodeError ⇒ **0 条判据**直接退 ✗
#    （``run_all`` 已重配 stdout 所以不露头 —— 与 `engine_win_io_test` 同一个坑 ✓）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.engine import dit_seedvr2 as m  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []

_REAL = Path(r"D:/Comfy-Desktop/ComfyUI-Shared/models/diffusion_models/SEEDVR2/"
             r"seedvr2_ema_3b_fp16.safetensors")

#: ⑮⑯ 要一份**真 mp4** 才能验「尺寸/帧数从源视频来」（720×480 ✓ / 24fps ✓ / 49 帧 ✓）。
#: ⚠️ 盘上没有就 **skip** ✓ 不编假尺寸 ✗ —— 编出来的 plan 只能证明算术自洽 ✓，
#: 证明不了「读的是这个文件」✓✗。
#: 原先取的是 `inference/gradio_composite_demo/example_videos/horse.mp4`
#: （443 KB 示例素材）；2026-11-04 清理 `reference/`（整棵都是 `.gitignore` 的学习用拷贝）时
#: 连同 CogVideo 一起删了 ⇒ 现在**不再指向任何 reference 路径**：给 `DRAMA_TEST_VIDEO` 就用它，
#: 没给就落到一个不存在的路径、⑮⑯ 如实 skip（要复跑那份真样本：从 CogVideo 仓取回 horse.mp4）✓✗。
_VIDEO = Path(os.environ.get("DRAMA_TEST_VIDEO")
              or r"D:/code/voide/darme-voide/tmp/__nope__.mp4")
_MISSING = r"D:/code/voide/darme-voide/tmp/__nope__.mp4"


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call) -> str:  # noqa: ANN001
    try:
        call()
    except Exception as error:  # noqa: BLE001
        return str(error)
    return ""


_TINY = dict(layers=3, mm_layers=1, vid_dim=64, txt_dim=64, txt_in_dim=32,
             in_channels=33, out_channels=16, heads=2, head_dim=16,
             ffn_dim=256, expand_ratio=4, patch=(1, 2, 2), rope_freqs_per_axis=2,
             sinusoidal_dim=256, emb_dim=384, emb_hidden_dim=64)


def _tiny_shape(**over) -> m.SeedVR2Shape:
    cfg = dict(_TINY)
    cfg.update(over)
    return m.SeedVR2Shape(**cfg)


def _key_shapes(shape: m.SeedVR2Shape, *, cond: bool = False) -> dict[str, tuple[int, ...]]:
    """手抄一份权重键集（不建模型 ⇒ 反推与建模型互不背书）。"""
    d, td, ff = shape.vid_dim, shape.txt_dim, shape.ffn_dim
    inner, vol = shape.heads * shape.head_dim, shape.patch_volume
    shapes: dict[str, tuple[int, ...]] = {
        "vid_in.proj.weight": (d, shape.in_channels * vol),
        "vid_out.proj.weight": (shape.out_channels * vol, d),
        "vid_out_norm.weight": (d,),
        "txt_in.weight": (td, shape.txt_in_dim),
        "emb_in.proj_in.weight": (shape.emb_hidden_dim, shape.sinusoidal_dim),
        "emb_in.proj_out.weight": (shape.emb_dim, shape.emb_hidden_dim),
    }
    for i in range(shape.layers):
        shared, is_last = i >= shape.mm_layers, i == shape.layers - 1
        key, pre = ("all" if shared else "vid"), f"blocks.{i}"
        shapes[f"{pre}.ada.{key}.attn_shift"] = (d,)
        shapes[f"{pre}.attn.norm_q.{key}.weight"] = (shape.head_dim,)
        shapes[f"{pre}.attn.proj_qkv.{key}.weight"] = (3 * inner, d)
        shapes[f"{pre}.mlp.{key}.proj_in.weight"] = (ff, d)
        shapes[f"{pre}.mlp.{key}.proj_in_gate.weight"] = (ff, d)
        shapes[f"{pre}.attn.rope.rope.freqs"] = (shape.rope_freqs_per_axis,)
        if not shared and not is_last:
            shapes[f"{pre}.ada.txt.attn_shift"] = (td,)
            shapes[f"{pre}.attn.proj_qkv.txt.weight"] = (3 * inner, td)
    if cond:
        shapes["positive_conditioning"] = m.SEEDVR2_POS_COND_SHAPE
        shapes["negative_conditioning"] = m.SEEDVR2_NEG_COND_SHAPE
    return shapes


def _tiny_model(shape: m.SeedVR2Shape | None = None) -> m.SeedVR2DiT:
    shape = shape or _tiny_shape()
    torch.manual_seed(11)
    model = m.build(shape, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight") and p.dim() == 1:
                p.uniform_(0.9, 1.1)
            else:
                p.normal_(0.0, 0.08)
    return model


def _inputs(batch: int = 2, frames: int = 2, size: int = 8, tokens: int = 7):
    torch.manual_seed(3)
    return (torch.randn(batch, 16, frames, size, size),
            torch.randn(batch, 17, frames, size, size),
            torch.randn(batch, tokens, 32))


# --------------------------------------------------------------------------- #
# ① 形状反推
# --------------------------------------------------------------------------- #


def case_infer_shape() -> None:
    got = m.infer_shape(_key_shapes(_tiny_shape(), cond=True))
    check("① 反推：块数 3 / 双塔块 1（按 ``.txt.`` 键数出来，末块没有 ✓）",
          (got.layers, got.mm_layers) == (3, 1), (got.layers, got.mm_layers))
    rows = {
        "vid_dim": (got.vid_dim, 64), "txt_dim": (got.txt_dim, 64),
        "txt_in_dim": (got.txt_in_dim, 32), "in_channels": (got.in_channels, 33),
        "out_channels": (got.out_channels, 16), "heads": (got.heads, 2),
        "head_dim": (got.head_dim, 16), "ffn_dim": (got.ffn_dim, 256),
        "patch": (tuple(got.patch), (1, 2, 2)),
        "rope_freqs_per_axis": (got.rope_freqs_per_axis, 2),
        "sinusoidal_dim": (got.sinusoidal_dim, 256),
        "emb_dim": (got.emb_dim, 384), "emb_hidden_dim": (got.emb_hidden_dim, 64),
        "mlp_type": (got.mlp_type, "swiglu"),
        "vid_out_norm_ada": (got.vid_out_norm_ada, True),
        "has_conditioning": (got.has_conditioning, True),
    }
    wrong = {k: v for k, v in rows.items() if v[0] != v[1]}
    check("① 反推：其余 16 个字段逐个对上（全部来自权重形状，没有一项是猜的 ✓）",
          not wrong, wrong)
    check("① 反推：条件通道 = 33 − 16 = 17；旋转维 6×2 = 12 ≤ head_dim 16 ✓",
          got.condition_channels == 17 and got.rot_dim == 12,
          (got.condition_channels, got.rot_dim))
    check("① 反推：没有 ``positive_conditioning`` 键 ⇒ has_conditioning=False ✓",
          m.infer_shape(_key_shapes(_tiny_shape())).has_conditioning is False, None)


def case_infer_guards() -> None:
    shape = _tiny_shape()
    base = _key_shapes(shape)

    text = _raises(lambda: m.infer_shape(
        {k: v for k, v in base.items() if k != "vid_in.proj.weight"}))
    check("② 守门：缺 ``vid_in.proj.weight`` ⇒ 拒且点名该键 ✓", "vid_in.proj.weight" in text, text)

    # ⚠️ 要真的「缺一块」就得把 ``blocks.1.`` **所有**键都删掉 ✓ —— 只删一个键的话
    #    `_blocks_index` 仍看得见 blocks.1 ⇒ 编号依旧连续、本来就该放行 ✗（早先这条判据
    #    是**假红**：测的是「少一个键」，不是「少一块」✓✗）。
    broken = {k: v for k, v in base.items() if not k.startswith("blocks.1.")}
    text = _raises(lambda: m.infer_shape(broken))
    check("② 守门：块编号不连续 ⇒ 拒（缺块会被当成「模型就这么深」✗）", "不连续" in text, text)
    text = _raises(lambda: m.infer_shape(base))
    check("② 守门：只掉一个键（块仍在）⇒ **不**报「不连续」（别把缺键误判成缺块 ✗）",
          "不连续" not in text, text[:80])

    moved = dict(base)
    moved["blocks.2.ada.txt.attn_shift"] = (shape.txt_dim,)
    text = _raises(lambda: m.infer_shape(moved))
    check("② 守门：双塔块不连续（``.txt.`` 跑到末块）⇒ 拒", "连续" in text, text)

    seven = _key_shapes(_tiny_shape(vid_dim=3072, ffn_dim=m._swiglu_hidden(3072),
                                    rope_freqs_per_axis=14))
    text = _raises(lambda: m.infer_shape(seven))
    check("② 守门：vid_dim=3072 ⇒ 直接判 7B 并拒（按 3B 几何建会跑通但出片全糊 ✗）",
          "7B" in text, text)

    text = _raises(lambda: m.infer_shape(_key_shapes(_tiny_shape(rope_freqs_per_axis=3))))
    check("② 守门：6×3=18 > head_dim 16 ⇒ 拒", "head_dim" in text, text)

    text = _raises(lambda: m.infer_shape(_key_shapes(_tiny_shape(ffn_dim=128))))
    check("② 守门：SwiGLU 隐层与 ``8/3·dim`` 公式不符 ⇒ 拒", "SwiGLU" in text, text)

    odd = dict(base)
    odd["vid_out.proj.weight"] = (shape.out_channels * 6, shape.vid_dim)
    odd["vid_in.proj.weight"] = (shape.vid_dim, shape.in_channels * 6)
    text = _raises(lambda: m.infer_shape(odd))
    check("② 守门：patch 体量不是 4 ⇒ 拒（(1,2,2) 的分解反推不出来，不许猜 ✓）",
          "patch" in text, text)


def case_param_count() -> None:
    shape = _tiny_shape()
    counted = sum(p.numel() for p in m.build(shape).state_dict().values())
    check("③ 参数量：``param_count(shape)`` 与建出来的 state_dict 求和相同 ✓",
          m.param_count(shape) == counted, (m.param_count(shape), counted))

    with_cond = _tiny_shape(has_conditioning=True)
    counted2 = sum(p.numel() for p in m.build(with_cond).state_dict().values())
    check("③ 参数量：内置文本条件那两块 buffer（58×5120×2）也计入 ✓",
          m.param_count(with_cond) == counted2, (m.param_count(with_cond), counted2))


# --------------------------------------------------------------------------- #
# ② 前向
# --------------------------------------------------------------------------- #


def case_forward() -> None:
    model = _tiny_model()
    lat, cond, ctx = _inputs()
    with torch.no_grad():
        out = model(lat, torch.tensor([0.7, 0.3]), ctx, cond)
        again = model(lat, torch.tensor([0.7, 0.3]), ctx, cond)
        # ⚠️ 判「广播」必须拿**两行同值**的那次做基准 ✗：`out` 是 `[0.7, 0.3]`（两行**不同** ✓），
        #    拿它和 `[0.7]` 比等于在比两个不同的 σ 序列 —— 差 0.46 是正常的，不是缺陷 ✓✗。
        same = model(lat, torch.tensor([0.7, 0.7]), ctx, cond)
        broadcast = model(lat, torch.tensor([0.7]), ctx, cond)
        other = model(lat, torch.tensor([1.0, 1.0]), ctx, cond)

    # ⚠️ 出口形状**与输入同尺寸** ✓：patch=(1,2,2) 只压主干内部的 token 数
    #    （patchify 后是 `T·H/2·W/2` 个 token ✓），`vid_out` 会把它**解回原尺寸** ✓ ——
    #    修复/超分任务要的是与输入逐元素对齐的残差 ✗，输出 patch 网格等于尺寸都不对 ✗。
    check("④ 前向：输出 ``(B,16,T,H,W)``（patch 只压内部 token，出口解回原尺寸 ✓）",
          tuple(out.shape) == (2, 16, 2, 8, 8), tuple(out.shape))
    check("④ 前向：全 finite（fp32 下不该出 NaN / inf ✓）",
          bool(torch.isfinite(out).all()), float(out.abs().max()))
    check("④ 前向：同输入两次逐位相同（无隐藏随机性 / 未就地污染输入 ✓）",
          torch.equal(out, again), float((out - again).abs().max()))
    check("④ 前向：单个 σ 广播整批 == 显式给两行同值 ✓",
          torch.equal(broadcast, same), float((broadcast - same).abs().max()))
    check("④ 前向：换 σ 输出真的变（时间步确实进了主干 ✓）",
          not torch.allclose(same, other), float((other - same).abs().max()))

    with torch.no_grad():
        two_dim = _tiny_model()(lat[:1], torch.tensor([0.7]), ctx[0], cond[:1])
    check("④ 前向：B=1 时 context 可以给二维 ``(L,D)`` ✓",
          tuple(two_dim.shape) == (1, 16, 2, 8, 8), tuple(two_dim.shape))


def case_dtype_align() -> None:
    """④′ **精度对齐** ✓：采样侧的潜变量是 **f32** ✓，权重是 bf16/fp16 ✓ ⇒ 前向入口必须自己对齐 ✓。

    ⚠️ 这条是被**真权重首跑**逼出来的 ✗✗：`forward` 原先写 `dtype = latents.dtype` ✓
    （跟随潜变量 ✓，看起来"更保精度" ✓），而 bf16 权重下第一个 `nn.Linear`（`emb_in`）当场
    `mat1 and mat2 must have the same dtype, but got Float and BFloat16` ✓ —— 报错指向
    `nn.Linear`，读起来像"权重装错档"✗，实际是精度没对齐 ✓。口径同
    :meth:`dit_qwen_image.QwenImageDiT.forward` ✓：**在模块入口对齐**，别把采样侧降精度 ✗。
    """
    shape = _tiny_shape()
    model = m.build(shape, dtype=torch.bfloat16).eval()
    lat, cond, ctx = _inputs()
    with torch.no_grad():
        out = model(lat, torch.tensor([0.7, 0.7]), ctx, cond)
    check("④′ f32 潜变量 ＋ bf16 权重 ⇒ 不报错 ✓ 且输出精度跟**权重**走（bf16 ✓ 不是 f32 ✗）",
          out.dtype == torch.bfloat16 and tuple(out.shape) == (2, 16, 2, 8, 8),
          (str(out.dtype), tuple(out.shape)))

    #: 条件那条也要跟着对齐 ✓（只对齐潜变量、漏条件 ⇒ `cat` 当场报 dtype 不一致 ✓）
    with torch.no_grad():
        cond_f64 = cond.to(torch.float64)
        out2 = model(lat, torch.tensor([0.7, 0.7]),
                     ctx.to(torch.float32), cond_f64)
    check("④′ 条件给 float64 ⇒ 也被对齐（`cat` 不报 dtype 不一致 ✓）",
          out2.dtype == torch.bfloat16 and bool(torch.isfinite(out2.float()).all()),
          (str(out2.dtype), float(out2.float().abs().max())))


def case_forward_guards() -> None:
    model = _tiny_model()
    sigma = torch.tensor([0.7, 0.7])
    lat, cond, ctx = _inputs()

    text = _raises(lambda: model(lat[0], sigma, ctx, cond))
    check("⑤ 守门：4 维潜变量 ⇒ 拒并点明要 5 维", "5 维" in text, text)

    text = _raises(lambda: model(torch.randn(2, 8, 2, 8, 8), sigma, ctx, cond))
    check("⑤ 守门：潜变量通道不是 16 ⇒ 拒", "16" in text, text)

    text = _raises(lambda: model(lat, sigma, ctx, torch.randn(2, 16, 2, 8, 8)))
    check("⑤ 守门：条件潜不是 17 通道 ⇒ 拒（16 通道是常见误传 ✗）", "17" in text, text)

    text = _raises(lambda: model(lat, sigma, ctx, cond[:, :, :1]))
    check("⑤ 守门：条件潜与潜变量 B/T/H/W 不一致 ⇒ 拒", "一致" in text, text)

    text = _raises(lambda: model(lat, sigma, ctx, None))
    check("⑤ 守门：不给 conditioning ⇒ 拒（通用出图管线不能直接调这一族 ✓）",
          "17" in text, text)

    bare = _tiny_model(_tiny_shape(has_conditioning=False))
    text = _raises(lambda: bare(lat, sigma, None, cond))
    check("⑤ 守门：该档无内置文本条件又没给 context ⇒ 拒（不拿脏 buffer 当条件 ✓）",
          "context" in text, text)

    text = _raises(lambda: model(lat, sigma, ctx[0], cond))
    check("⑤ 守门：B=2 却给二维 context ⇒ 拒（二维只服务 B=1 ✓）", "B=1" in text, text)

    text = _raises(lambda: model(lat, sigma, torch.randn(2, 7, 16), cond))
    check("⑤ 守门：context 宽度不是 txt_in_dim ⇒ 拒", "32" in text, text)

    text = _raises(lambda: model(lat, torch.tensor([0.7, 0.7, 0.7]), ctx, cond))
    check("⑤ 守门：σ 个数与批大小不一致 ⇒ 拒", "sigma" in text, text)


# --------------------------------------------------------------------------- #
# ③ 结构口径
# --------------------------------------------------------------------------- #


def case_structure() -> None:
    shape = _tiny_shape()
    model = _tiny_model(shape)
    keys = set(model.state_dict())

    norm_keys = [k for k in keys if "attn_norm" in k or "mlp_norm" in k]
    check("⑥ 结构：块内 norm 无 affine（state_dict 里一个键都不占 ✓ 多出来就是骨架建错 ✗）",
          isinstance(model.blocks[0].attn_norm, m._MMNorm)
          and not list(model.blocks[0].attn_norm.parameters())
          and not norm_keys, norm_keys)

    last = f"blocks.{shape.layers - 1}."
    check("⑥ 结构：末块 ``vid_only`` 没有任何 ``.txt.`` 键 ✓",
          not any(k.startswith(last) and ".txt." in k for k in keys),
          sorted(k for k in keys if k.startswith(last) and ".txt." in k)[:4])

    shared = f"blocks.{shape.mm_layers}."
    check("⑥ 结构：共享块只有 ``.all.``（没有 vid/txt 之分 ✓）",
          not any(k.startswith(shared) and (".vid." in k or ".txt." in k) for k in keys),
          sorted(k for k in keys if k.startswith(shared))[:4])

    check("⑥ 结构：双塔块有 ``.txt.`` 分支，末端有 ``vid_out_norm`` + ``vid_out_ada`` ✓",
          any(k.startswith("blocks.0.") and ".txt." in k for k in keys)
          and "vid_out_norm.weight" in keys and "vid_out_ada.out_scale" in keys,
          sorted(k for k in keys if k.startswith("vid_out")))


def case_ada() -> None:
    dim = 8
    torch.manual_seed(5)
    ada = m._AdaMM(dim, dim, ("attn", "mlp"), ("in", "out"), shared=False, vid_only=False)
    with torch.no_grad():
        for p in ada.parameters():
            p.normal_(0.0, 0.1)
    emb = torch.randn(1, 6 * dim)
    hid = torch.randn(5, dim)

    out1 = ada.apply_in(hid.clone(), emb, "attn", torch.tensor([5]), branch="vid")
    out2 = ada.apply_in(hid.clone(), emb.repeat(2, 1), "attn", torch.tensor([3, 2]), branch="vid")
    check("⑦ 调制：B=1 快路径与 B>1 的 repeat_interleave 路逐位一致 ✓"
          "（720p 下 L 十万级，B=1 要避开 3·dim·L 物化 ✓）",
          torch.equal(out1, out2), float((out1 - out2).abs().max()))

    gate1 = ada.apply_out(hid.clone(), emb, "mlp", torch.tensor([5]), branch="vid")
    gate2 = ada.apply_out(hid.clone(), emb.repeat(2, 1), "mlp", torch.tensor([3, 2]), branch="vid")
    check("⑦ 调制：``apply_out`` 走 gate 那一格，两条路同样一致 ✓",
          torch.equal(gate1, gate2), float((gate1 - gate2).abs().max()))

    idx = ada.layers.index("mlp")
    sel = emb.reshape(1, -1, len(ada.layers), 3)[:, :, idx, :]
    shift, scale, _ = sel[0].unbind(-1)
    want = hid * (scale + ada.vid.mlp_scale) + (shift + ada.vid.mlp_shift)
    got = ada.apply_in(hid.clone(), emb, "mlp", torch.tensor([5]), branch="vid")
    check("⑦ 调制：``mlp`` 组切的是 emb 的第 2 组（不是第 1 组 ✓）",
          torch.allclose(got, want, atol=1e-6), float((got - want).abs().max()))

    last = m._AdaMM(dim, dim, ("attn", "mlp"), ("in", "out"), shared=False, vid_only=True)
    plain = hid.clone()
    passed = last.apply_in(plain, emb, "attn", torch.tensor([5]), branch="txt")
    check("⑦ 调制：末块 ``vid_only`` 上文本分支原样透传（同一对象，不白算 ✓）",
          passed.data_ptr() == plain.data_ptr(), None)

    out_ada = m._OutAda(dim)
    with torch.no_grad():
        out_ada.out_shift.normal_(0.0, 0.1)
        out_ada.out_scale.normal_(0.0, 0.1)
    sel_o = emb.reshape(1, -1, 2, 3)[:, :, 0, :]
    shift_o, scale_o, _ = sel_o[0].unbind(-1)
    want_o = hid * (scale_o + out_ada.out_scale) + (shift_o + out_ada.out_shift)
    got_o = out_ada(hid.clone(), emb, torch.tensor([5]))
    check("⑦ 调制：``vid_out_ada`` 按 attn 组（2 层里的 idx=0）切 emb，与手算一致 ✓"
          "（按 1 层切会得 2·dim 宽 ⇒ 崩，参考靠 Cache 副作用才不崩 ✗）",
          torch.allclose(got_o, want_o, atol=1e-6), float((got_o - want_o).abs().max()))


def case_mlp() -> None:
    mlp = m._MLP(4, 6, kind="swiglu")
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.3)
    x = torch.randn(3, 4)
    want = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, mlp.proj_in_gate.weight))
        * torch.nn.functional.linear(x, mlp.proj_in.weight), mlp.proj_out.weight)
    swapped = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, mlp.proj_in.weight))
        * torch.nn.functional.linear(x, mlp.proj_in_gate.weight), mlp.proj_out.weight)
    check("⑧ 前馈：SwiGLU 是 ``silu(门) * 值``（反写成 ``silu(值) * 门`` 形状全对、数值全错 ✗）",
          torch.allclose(mlp(x), want, atol=1e-6) and not torch.allclose(want, swapped),
          float((mlp(x) - want).abs().max()))

    with torch.no_grad():
        text = _raises(lambda: m._MLP(4, 6, kind="relu"))
    check("⑧ 前馈：未知档当面拒 ✓", "只支持" in text, text)


def case_attention() -> None:
    torch.manual_seed(7)
    q, k, v = (torch.randn(6, 2, 4) for _ in range(3))
    got = m._var_attention(q, k, v, heads=2, cu_seqlens=[0, 3, 6])

    def seg(x):
        return x.permute(1, 0, 2).unsqueeze(0)

    want = torch.cat([segment.squeeze(0).permute(1, 0, 2) for segment in (
        torch.nn.functional.scaled_dot_product_attention(seg(q[:3]), seg(k[:3]), seg(v[:3])),
        torch.nn.functional.scaled_dot_product_attention(seg(q[3:]), seg(k[3:]), seg(v[3:])),
    )], dim=0)
    check("⑨ 注意力：两段各自算一次（窗口互不可见 ✓）", torch.allclose(got, want, atol=1e-6),
          float((got - want).abs().max()))

    q2 = q.clone()
    q2[3:] = torch.randn(3, 2, 4)
    other = m._var_attention(q2, k, v, heads=2, cu_seqlens=[0, 3, 6])
    check("⑨ 注意力：改第 2 段不影响第 1 段（拼成一个注意力就会串味 ✗）",
          torch.equal(got[:3], other[:3]) and not torch.allclose(got[3:], other[3:]), None)

    text = _raises(lambda: m._var_attention(q, k, v, heads=2, cu_seqlens=[0, 3]))
    check("⑨ 注意力：``cu_seqlens`` 不落在 0 与总长上 ⇒ 拒", "cu_seqlens" in text, text)


def _window_fn(name: str, num_windows=(1, 1, 1)):
    """把 ``_WINDOW_FNS`` 里的裸窗口算法包成 ``_Patcher._make_window`` 的口径 ✓。

    ⚠️ 裸函数签名是 ``(size=(t,h,w), num_windows) -> [slice]``，而 ``_window`` / ``_window_idx``
    要的 ``window_fn`` 是**吃张量、还切好的张量列表** ✓（生产侧唯一实现 `_Patcher._make_window` ✓）。
    直接递裸函数会炸 ``'torch.Size' object has no attribute 'shape'`` ✗ —— 那不是窗口算法错，
    是**口径混用** ✗。
    """
    raw = m._WINDOW_FNS[name]

    def apply(x: torch.Tensor):
        t, h, w = x.shape[:3]
        return [x[st, sh, sw] for (st, sh, sw) in raw((t, h, w), num_windows)]

    return apply


def case_window() -> None:
    vid_shape = torch.tensor([[4, 8, 8]])
    wfn = _window_fn("720pwin_by_size_bysize")
    part, restore, wshape, wlen, wcount = m._window_idx(vid_shape, wfn)
    flat = torch.arange(4 * 8 * 8, dtype=torch.float32).unsqueeze(-1)
    check("⑩ 窗口：分区再还原逐位回到原序（不重叠是 argsort 还原的前提 ✓）",
          torch.equal(restore(part(flat)), flat), int(wshape.prod()))

    shifted = _window_fn("720pswin_by_size_bysize")
    part2, restore2, *_ = m._window_idx(vid_shape, shifted)
    check("⑩ 窗口：平移版（``720pswin``）同样能无缝还原 ✓",
          torch.equal(restore2(part2(flat)), flat), None)
    check("⑩ 窗口：窗口长度之和 == 原 token 数 ✓",
          sum(wlen) == 4 * 8 * 8 and sum(wcount) == 1, (sum(wlen), wcount))


def case_timestep() -> None:
    t = torch.tensor([0.0, 700.0])
    emb = m._timestep_embedding(t, 8)
    half = 4
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000) / half))
    want = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=-1)
    check("⑪ 时间步：``[sin, cos]`` 顺序、分母 half_dim（常见的 cos 在前 / 减 1 都不对 ✗）",
          torch.allclose(emb, want, atol=1e-6), float((emb - want).abs().max()))
    check("⑪ 时间步：t=0 ⇒ 前一半全 0、后一半全 1 ✓",
          bool(torch.allclose(emb[0, :half], torch.zeros(half)))
          and bool(torch.allclose(emb[0, half:], torch.ones(half))), None)

    text = _raises(lambda: m._timestep_embedding(torch.zeros(1, 2), 8))
    check("⑪ 时间步：时间步不是一维 ⇒ 拒", "一维" in text, text)


def case_real_weight() -> None:
    if not _REAL.is_file():
        skip(f"盘上没有 {_REAL.name}，跳过真实几何核对")
        return
    from safetensors import safe_open

    with safe_open(str(_REAL), framework="pt") as handle:
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}

    got = m.infer_shape(shapes)
    facts = {
        "layers": (got.layers, 32), "mm_layers": (got.mm_layers, 10),
        "vid_dim": (got.vid_dim, 2560), "txt_in_dim": (got.txt_in_dim, 5120),
        "heads": (got.heads, 20), "head_dim": (got.head_dim, 128),
        "ffn_dim": (got.ffn_dim, 6912), "emb_dim": (got.emb_dim, 15360),
        "rope_freqs_per_axis": (got.rope_freqs_per_axis, 21),
        "has_conditioning": (got.has_conditioning, False),
    }
    wrong = {k: v for k, v in facts.items() if v[0] != v[1]}
    check(f"⑫ 真实权重：{len(shapes)} 个张量反推出 3B 几何（32 块 / 双塔 10 块 / 20 头 × 128）✓",
          not wrong, wrong)
    check("⑫ 真实权重：参数量 == 3,391,476,448（3.39B）✓",
          m.param_count(got) == 3391476448, m.param_count(got))
    check("⑫ 真实权重：末块（31）没有文本键、第 10 块起全是 ``.all.`` ✓",
          not any(k.startswith("blocks.31.") and ".txt." in k for k in shapes)
          and any(k.startswith("blocks.10.") and ".all." in k for k in shapes), None)


def case_family_table() -> None:
    """⑬ 族表接线：``dit_build`` 认这一族，且**不把它混进出图族** ✓。"""
    from types import SimpleNamespace

    from app.services.engine import dit_build

    shapes = _key_shapes(_tiny_shape())
    info = SimpleNamespace(tensors={k: SimpleNamespace(shape=v) for k, v in shapes.items()})
    module, config = dit_build.build("seedvr2", info, dtype=torch.float32)

    check("⑬ 族表：``dit_build.build('seedvr2', info)`` 建出 ``SeedVR2DiT`` ✓",
          isinstance(module, m.SeedVR2DiT), type(module).__name__)
    pairs = {"family": ("seedvr2",), "text_dim": (32,), "in_channels": (33,)}
    check("⑬ 族表：config 有 family / text_dim（← txt_in_dim）/ in_channels=33 ✓",
          all(config.get(k) == v[0] for k, v in pairs.items()),
          {k: config.get(k) for k in pairs})

    contract = dit_build.contract_of("seedvr2")
    check("⑬ 族表：契约 = 5 维潜变量 · σ 批形态 · 口径 seedvr2 ✓",
          (contract.latent_dims, contract.sigma_form, contract.latent_format)
          == (5, "batch", "seedvr2"), contract)
    check("⑬ 族表：``image_only_keys()`` **不含** seedvr2（混进去会按 4 维造噪声 ✗）",
          "seedvr2" not in dit_build.image_only_keys(), dit_build.image_only_keys())
    both = dit_build.latent_format_problem("seedvr2", "wan21") or ""
    check("⑬ 族表：口径对账 —— 自家族 VAE 通过 ✓、挂 Wan21 的 VAE 当场报 **偏色/发灰** ✓",
          dit_build.latent_format_problem("seedvr2", "seedvr2") is None and "wan21" in both,
          both)


def case_family_real() -> None:
    """⑭ 真装载：真实 3B fp16 权重走族表 ⇒ 建壳 → 装 → 前向 finite。

    默认跳过（约 40 秒 / 峰值 14 GiB 内存）；设 ``SEEDVR2_REAL=1`` 才跑。
    """
    if os.environ.get("SEEDVR2_REAL") != "1":
        skip("设 SEEDVR2_REAL=1 才跑真实 3B 权重装载（约 40 秒 / 峰值 14 GiB）")
        return
    if not _REAL.is_file():
        skip(f"盘上没有 {_REAL.name}")
        return

    from app.services.engine import dit_build, safetensors as st, weights

    info = st.inspect(_REAL)
    module, config = dit_build.build("seedvr2", info, path=_REAL, dtype=torch.float16)
    report = weights.load_module_weights(module, _REAL, dtype=torch.float16)
    check("⑭ 真装载：权重逐键写完，没有 missing / unexpected ✓",
          not report.missing and not report.unexpected,
          (len(report.missing), len(report.unexpected), report.missing[:3]))

    torch.manual_seed(0)
    lat = torch.randn(1, 16, 2, 8, 8, dtype=torch.float16)
    cond = torch.randn(1, 17, 2, 8, 8, dtype=torch.float16)
    ctx = torch.randn(1, 7, int(config["text_dim"]), dtype=torch.float16)
    with torch.no_grad():
        out = module(lat, torch.tensor([0.7]), ctx, cond)
    check("⑭ 真装载：前向出 ``(1,16,2,8,8)`` 且全 finite ✓",
          tuple(out.shape) == (1, 16, 2, 8, 8) and bool(torch.isfinite(out).all()),
          (tuple(out.shape), float(out.abs().max())))


def case_plan() -> None:
    """⑮ **修复 plan** ✓：帧数/尺寸都从**源视频**推 ✓、步数与 cfg 的口径都要**如实报** ✓。

    ⚠️ 这一族最容易照抄的错 ✗✗：拿 ``megapixels``/``seconds`` 算尺寸（那是"生成" ✓ 不是"修" ✓）、
    拿 ``GenerationRequest`` 默认的 **30 步**跑一步蒸馏档（不报错、只是慢 30 倍还更糊 ✓✗）。
    """
    from app.services.engine import guidance as g
    from app.services.engine import pipeline

    #: ① ``seedvr2_target`` 的吸附算术（纯函数 ✓ 不碰 torch / 文件 ✓）
    t = pipeline.seedvr2_target({"width": 720, "height": 480, "fps": 24, "nbFrames": 49})
    check("⑮ 720×480/49 帧：16 的倍数不动 ✓、49＝4×12+1 不动 ✓、潜帧 13 ✓",
          (t["width"], t["height"], t["frames"], t["latent_frames"], t["fps"])
          == (720, 480, 49, 13, 24), t)
    t = pipeline.seedvr2_target({"width": 1920, "height": 1080, "fps": 30, "nbFrames": 100})
    check("⑮ 1080 就近吸到 1088 ✓（不是 1072 ✗）、100 帧**向下**吸到 97 ✓（4×24+1）",
          (t["width"], t["height"], t["frames"]) == (1920, 1088, 97), t)
    check("⑮ 吸附/截断都写进 warnings ✓（不静默改素材 ✗）",
          any("1080" in w and "1088" in w for w in t["warnings"])
          and any("97" in w for w in t["warnings"]), t["warnings"])
    t = pipeline.seedvr2_target({"width": 512, "height": 512, "nbFrames": 40})
    check("⑮ 帧率读不到 ⇒ 退 24 并点名 ✓", t["fps"] == 24
          and any("24fps" in w for w in t["warnings"]), t)
    t = pipeline.seedvr2_target({"width": 512, "height": 512, "fps": 25,
                                 "durationSeconds": 8.0})
    check("⑮ 只有时长 ⇒ 25×8＝200 帧 ⇒ 超 121 上限**截到 121 ✓ 并报**（121＝4×30+1 ✓）",
          t["frames"] == 121 and any("121" in w for w in t["warnings"]), t)
    check("⑮ 帧数不足下限 ⇒ 当场拒 ✓（不是静默修 1 帧 ✗）",
          "至少要" in _raises(lambda: pipeline.seedvr2_target(
              {"width": 512, "height": 512, "fps": 24, "nbFrames": 2})))
    check("⑮ 尺寸读不到 ⇒ 拒 ✓", "尺寸" in _raises(
        lambda: pipeline.seedvr2_target({"width": 0, "height": 0, "nbFrames": 49})))

    #: ② ``build_plan`` 的门槛：**没有源视频就不许开工** ✓
    from app.services.engine import pipeline as P

    def _req(**over):
        #: ⚠️ 别写成 ``GenerationRequest(stage=..., prompt="", **over)`` ✗ —— ``over`` 里再给
        #: ``prompt`` 就 ``got multiple values for keyword argument`` ✓（本轮真踩 ✓）。
        kwargs = {"stage": "seedvr2", "prompt": ""}
        kwargs.update(over)
        return P.GenerationRequest(**kwargs)

    msg = _raises(lambda: P.build_plan(_req()))
    check("⑮ 没给 `referenceVideos` ⇒ 拒 ✓ 且说明「修复族不是生成族」✓",
          "referenceVideos" in msg and "修复" in msg, msg[:120])
    msg = _raises(lambda: P.build_plan(_req(reference_videos=("a.mp4", "b.mp4"))))
    check("⑮ 给两路 ⇒ 拒 ✓（不静默只取第一路 ✗）", "一次只修" in msg, msg[:120])
    msg = _raises(lambda: P.build_plan(_req(reference_videos=(str(_MISSING),))))
    check("⑮ 路径不存在 ⇒ 拒 ✓ 且回执带原路径 ✓",
          "读不了源视频" in msg and "不存在" in msg, msg[:140])

    if not _VIDEO.exists():
        skip(f"盘上没有 {_VIDEO.name}")
        return

    #: ③ 真素材：plan 的六个字段都跟着**源视频**走 ✓
    plan = P.build_plan(_req(reference_videos=(str(_VIDEO),)))
    #: ⚠️ 这份素材真是 **8fps** ✓（不是 24 ✗）：帧率**只许抄源视频** ✓ —— 写死 24 或取默认值
    #: 会让「修复」悄悄改掉时间轴 ✓✗（6.125 秒的片子出来变成 2.04 秒 ✓）。
    check("⑮ plan：尺寸/帧数/潜帧/帧率都来自源视频 ✓（含 8fps 这个非默认值 ✓）",
          (plan.width, plan.height, plan.frames, plan.latent_frames, plan.fps)
          == (720, 480, 49, 13, 8),
          (plan.width, plan.height, plan.frames, plan.latent_frames, plan.fps))
    check("⑮ plan：默认 30 步被换成**官方 1 步** ✓ 并如实报 ✓",
          plan.steps == 1 and any("30" in w and "1 步" in w for w in plan.warnings), plan.steps)
    check("⑮ plan：σ 首尾 1→0 ✓ 且逐步递减 ✓（shift=1.0 直线 ✓）",
          abs(plan.sigmas[0] - 1.0) < 1e-6 and plan.sigmas[-1] == 0.0
          and all(a > b for a, b in zip(plan.sigmas, plan.sigmas[1:])), plan.sigmas)
    check("⑮ plan：timestep 就是 σ ✓（不乘 1000 ✗）",
          [float(v) for v in plan.timesteps] == [float(v) for v in plan.sigmas])

    #: ④ 显式传步数 ⇒ **照跑**，只报不改 ✓（与 Krea2 的 megapixels 同一纪律 ✓）
    plan2 = P.build_plan(_req(reference_videos=(str(_VIDEO),), steps=4))
    check("⑮ 显式 steps=4 ⇒ 照跑 ✓ 但报「多步不会更好」✓",
          plan2.steps == 4 and any("不会更好" in w for w in plan2.warnings), plan2.steps)
    plan3 = P.build_plan(_req(reference_videos=(str(_VIDEO),), prompt="去掉水印",
                              negative="模糊", guidance=g.GuidanceConfig(scale=5.0)))
    check("⑮ 没有文本塔 ⇒ `negative` 与 `cfg>1` 都**不生效**且各有告警 ✓（不静默 ✗）",
          any("负向提示词被" in w for w in plan3.warnings)
          and any("白花一倍算力" in w for w in plan3.warnings), plan3.warnings[-2:])


def case_conditioning() -> None:
    """⑯ **17 通道条件** ✓：形状跟着条件走 ✓、缺件当场拒 ✓、plan 被真解码校正 ✓。

    ⚠️ 这三条都在防同一件事 ✗✗：**条件缺一格时静默退化成"凭噪声生成"** ✓
    （出一片灰/一片糊 ✓，一条报错都没有 ✓）。
    """
    from app.services.engine import pipeline as P
    from app.services.engine import torch_backend as tb

    backend = tb.TorchBackend(device="cpu")
    backend._ditFamily = "seedvr2"
    backend._config = {"family": "seedvr2", "latent_channels": 16}

    msg = _raises(lambda: backend._encode_seedvr2_conditioning(
        P.GenerationRequest(stage="seedvr2", prompt="", reference_videos=(str(_VIDEO),))))
    check("⑯ 没挂 SEEDVR2 的 VAE ⇒ 拒 ✓ 且点名 `attach_seedvr2_vae` ✓",
          "attach_seedvr2_vae" in msg and "自己的 VAE" in msg, msg[:140])

    backend._vae = object()
    backend._vaeKind = "seedvr2-real"
    msg = _raises(lambda: backend._encode_seedvr2_conditioning(
        P.GenerationRequest(stage="seedvr2", prompt="")))
    check("⑯ 没有 `referenceVideos` ⇒ 拒 ✓（不退化成凭空生成 ✓）", "referenceVideos" in msg,
          msg[:120])

    req = P.GenerationRequest(stage="seedvr2", prompt="", seed=7)
    msg = _raises(lambda: backend.init_latents(P.GenerationPlan(), req))
    check("⑯ 没有条件就造噪声 ⇒ 拒 ✓ 且说明「形状由条件决定」✓", "条件" in msg, msg[:120])

    #: 条件在 ⇒ 噪声形状**跟着条件走** ✓，并把 plan 校正成真解码的帧数 ✓
    torch.manual_seed(0)
    backend._repairCondition = torch.zeros(1, 17, 3, 2, 2)
    backend._repairSource = {"frames": 9}
    plan = P.GenerationPlan()
    plan.latent_frames, plan.frames = 13, 49          # 假装 ffprobe 估多了 ✓
    noise = backend.init_latents(plan, req)
    check("⑯ 噪声形状 = (1, 16, 条件潜帧, h, w) ✓（16 通道 ✓ 不是 33 ✗）",
          tuple(noise.shape) == (1, 16, 3, 2, 2), tuple(noise.shape))
    check("⑯ plan 被校正到**真解码**的 3 潜帧 / 9 像素帧 ✓ 并留一条告警 ✓",
          (plan.latent_frames, plan.frames) == (3, 9)
          and any("校正" in w for w in plan.warnings), (plan.latent_frames, plan.frames))

    #: 同一 seed 逐位复现 ✓（噪声生成不掺时间/随机源 ✓）
    again = backend.init_latents(P.GenerationPlan(), req)
    backend._repairSource = None
    check("⑯ 同 seed ⇒ 逐位相同的噪声 ✓（可复现 ✓）",
          torch.equal(noise, again), float((noise - again).abs().max()))

    #: 结构事实缺 ⇒ 不猜通道数 ✓
    backend._config = {"family": "seedvr2"}
    msg = _raises(lambda: backend.init_latents(P.GenerationPlan(), req))
    check("⑯ `latent_channels` 缺 ⇒ 拒 ✓（16 也**不猜** ✓）", "latent_channels" in msg,
          msg[:120])


CASES = (
    ("① 形状反推", case_infer_shape),

    ("② 反推守门", case_infer_guards),
    ("③ 参数量", case_param_count),
    ("④ 前向", case_forward),
    ("④′ 精度对齐", case_dtype_align),
    ("⑤ 前向守门", case_forward_guards),
    ("⑥ 结构口径", case_structure),
    ("⑦ 调制", case_ada),
    ("⑧ 前馈", case_mlp),
    ("⑨ 注意力", case_attention),
    ("⑩ 窗口", case_window),
    ("⑪ 时间步", case_timestep),
    ("⑫ 真实权重", case_real_weight),
    ("⑬ 族表接线", case_family_table),
    ("⑭ 族表真装载", case_family_real),
    ("⑮ 修复 plan 口径", case_plan),
    ("⑯ 条件通路", case_conditioning),
)


def main() -> int:
    for title, case in CASES:
        print(f"\n== {title} ==")
        try:
            case()
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"{title} 抛异常", False, f"{type(error).__name__}: {error}")

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        mark = "✓" if ok else "✗"
        tail = "" if ok else f"  ← {detail}"
        print(f"{mark} {name}{tail}")
    for reason in _SKIPS:
        print(f"- 跳过：{reason}")

    # ⚠️ 收尾行必须是 ``SUMMARY: n/m passed`` 这个**字面量** ✓（``run_all`` 只认它 ✓；
    # 写成 ``n/m 通过（跳过 k 项）`` ⇒ 全量表里本套**项数恒为 0** ✗，见 ``structure_test`` ㊈ ✓）。
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    print(f"  （跳过 {len(_SKIPS)} 项）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
