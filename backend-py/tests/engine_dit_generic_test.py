"""``dit_generic``（通用单流 DiT 骨架）的判据。

盯三类事：

1. **数学自洽**：``RMSNorm`` 是 ``scale+1``、RoPE 在原点恒等且保范数、三轴维度之和 == head_dim；
2. **结构对得上权重**：用**盘上真实 krea2 文件**的形状反推结构，再双向对账键集合与逐键形状
   —— 这是唯一能证明「骨架没少层、没错名、没错维」的判据；
3. **前向真的算**：小配置跑通、输出形状正确、改文本条件输出必变（文本支路没有装饰化）、
   形状自相矛盾时**当场报错**而不是建出半个模型。

⚠️ 本套件不读权重内容（只读 safetensors 头）⇒ 秒级完成；装载反量化属 ``quant`` 的事。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.services.engine import dit_generic as g  # noqa: E402
from app.services.engine.safetensors import inspect as inspect_safetensors  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

#: 盘上那份「盘上最多一族」的真文件 ✓ —— 存在才跑真对账，缺了要**明说跳过**而不是静默 ✓。
REAL_KREA2 = Path(
    r"D:/Comfy-Desktop/ComfyUI-Shared/models/diffusion_models/Krea2/"
    r"krea2_turbo_int8_convrot.safetensors")


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _fake_shapes(features: int = 64, layers: int = 2, heads: int = 4, head_dim: int = 16,
                 kvheads: int = 1, mlpdim: int = 96, patch: int = 2, channels: int = 4,
                 tdim: int = 16, txtdim: int = 32, txtlayers: int = 2, txt_heads: int = 2,
                 txt_head_dim: int = 16, txt_kvheads: int = 2, txt_mlpdim: int = 48,
                 refiner: int = 2, txt_blocks: int = 1) -> dict[str, tuple[int, ...]]:
    """``txt_blocks=1`` 而 ``txtlayers=2`` 是**故意**的：盘上正是「2 个块处理 12 层特征」
    （``projector`` 入维 12、``layerwise_blocks`` 只有 2 个）—— 同数造数据就测不出把
    「输入层数」当「块数」用的错。
    """
    """造一份**与真文件同构**的最小形状表（键名规则照权重：``blocks.N.attn.wq.weight``…）。

    ⚠️ 它和 :func:`g.expected_keys` 是同一个人写的 ✓，所以二者「同错」是可能的 ✗ ——
    真正的闭环靠 :func:`case_real_weights` 拿盘上 878 个张量的真名去核 ✓。
    """
    p2 = patch * patch
    shapes: dict[str, tuple[int, ...]] = {
        "first.weight": (features, channels * p2), "first.bias": (features,),
        "tmlp.0.weight": (features, tdim), "tmlp.0.bias": (features,),
        "tmlp.2.weight": (features, features), "tmlp.2.bias": (features,),
        "tproj.1.weight": (6 * features, features), "tproj.1.bias": (6 * features,),
        "txtmlp.0.scale": (txtdim,),
        "txtmlp.1.weight": (features, txtdim), "txtmlp.1.bias": (features,),
        "txtmlp.3.weight": (features, features), "txtmlp.3.bias": (features,),
        "txtfusion.projector.weight": (1, txtlayers),   # 无 bias（真权重里就没有）
        "last.norm.scale": (features,),
        "last.linear.weight": (channels * p2, features), "last.linear.bias": (channels * p2,),
        "last.modulation.lin": (2, features),
    }
    for i in range(layers):
        head = f"blocks.{i}."
        shapes |= {
            head + "prenorm.scale": (features,), head + "postnorm.scale": (features,),
            head + "attn.wq.weight": (heads * head_dim, features),
            head + "attn.wk.weight": (kvheads * head_dim, features),
            head + "attn.wv.weight": (kvheads * head_dim, features),
            head + "attn.wo.weight": (features, heads * head_dim),
            head + "attn.gate.weight": (features, features),
            head + "attn.qknorm.qnorm.scale": (head_dim,),
            head + "attn.qknorm.knorm.scale": (head_dim,),
            head + "mlp.gate.weight": (mlpdim, features),
            head + "mlp.up.weight": (mlpdim, features),
            head + "mlp.down.weight": (features, mlpdim),
            head + "mod.lin": (6 * features,),
        }
    for group, count in (("layerwise_blocks", txt_blocks), ("refiner_blocks", refiner)):
        for i in range(count):
            head = f"txtfusion.{group}.{i}."
            shapes |= {
                head + "prenorm.scale": (txtdim,), head + "postnorm.scale": (txtdim,),
                head + "attn.wq.weight": (txt_heads * txt_head_dim, txtdim),
                head + "attn.wk.weight": (txt_kvheads * txt_head_dim, txtdim),
                head + "attn.wv.weight": (txt_kvheads * txt_head_dim, txtdim),
                head + "attn.wo.weight": (txtdim, txt_heads * txt_head_dim),
                head + "attn.gate.weight": (txtdim, txtdim),
                head + "attn.qknorm.qnorm.scale": (txt_head_dim,),
                head + "attn.qknorm.knorm.scale": (txt_head_dim,),
                head + "mlp.gate.weight": (txt_mlpdim, txtdim),
                head + "mlp.up.weight": (txt_mlpdim, txtdim),
                head + "mlp.down.weight": (txtdim, txt_mlpdim),
            }
    return shapes


def case_math() -> None:
    """纯数学：``scale+1`` 约定、RoPE 原点恒等、保范数、三轴维度。"""
    torch.manual_seed(0)
    norm = g.RMSNorm(8)
    x = torch.randn(2, 5, 8, dtype=torch.float32)
    check("RMSNorm 在 `scale=0` 时等价于 `F.rms_norm(weight=None)` —— 即 `scale+1` 约定 "
          "（把 scale 当权重直接乘会差一倍）",
          torch.allclose(norm(x), F.rms_norm(x, (8,), weight=None, eps=g.RMS_EPS), atol=1e-6))
    with torch.no_grad():
        norm.scale.fill_(1.0)
    check("RMSNorm 的 `scale=1` ⇒ 权重 2（不是 1）", torch.allclose(
        norm(x), F.rms_norm(x, (8,), weight=torch.full((8,), 2.0), eps=g.RMS_EPS), atol=1e-6))

    pos = torch.zeros(1, 4, 3)
    freqs = g.rope_freqs(pos, [8, 4, 4], theta=1e3)
    q = torch.randn(1, 2, 4, 16)
    check("RoPE 在位置 (0,0,0) 上是恒等（角度全 0 ⇒ cos=1, sin=0）",
          torch.allclose(g.apply_rope(q, freqs.unsqueeze(1)), q, atol=1e-5),
          (g.apply_rope(q, freqs.unsqueeze(1)) - q).abs().max())
    pos2 = torch.tensor([[[0.0, 3.0, 7.0]]]).repeat(1, 4, 1)
    rot = g.apply_rope(q, g.rope_freqs(pos2, [8, 4, 4]).unsqueeze(1))
    check("RoPE 保范数（旋转是正交变换 ⇒ 每格 ||x|| 不变；错配成分量会掉范数）",
          torch.allclose(rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4),
          (rot.norm(dim=-1) - q.norm(dim=-1)).abs().max())
    shape = g.Krea2Shape(features=64, layers=1, heads=4, head_dim=128, kvheads=1, mlpdim=96,
                         patch=2, channels=4, tdim=16, txtdim=32, txtlayers=2, txt_blocks=1,
                         txt_heads=2, txt_head_dim=16, txt_kvheads=2, txt_mlpdim=48,
                         refiner_layers=2)
    check("三轴 RoPE 维度之和 == head_dim，且 head_dim=128 ⇒ [32,48,48]",
          sum(shape.axes_dim) == 128 and shape.axes_dim == [32, 48, 48], shape.axes_dim)
    # ⚠️ **精度**也是判据的一维 ✗：`freqs` 是 fp32 ✓，与 bf16 的 q/k 相乘会把它升成 fp32 ✓✗ ⇒
    #    SDPA 报 "query float32 / value bf16" ✗（2026-10-31 盘上真 Z-Image 权重上实测撞的 ✓）。
    #    小配置自检全 fp32 时这个错**看不出来** ✓✗ —— 所以必须单独钉：升精度后**降回** x.dtype ✓。
    q16, f16 = q.to(torch.bfloat16), freqs.unsqueeze(1)
    check("RoPE 保 dtype：bf16 进 ⇒ bf16 出（不许被 fp32 的频率升上去 ✗✗）",
          g.apply_rope(q16, f16).dtype == torch.bfloat16, g.apply_rope(q16, f16).dtype)
    check("RoPE 保 dtype 的同时数值不变（bf16 与 fp32 差在 bf16 的 3 位尾数内 ✓）",
          torch.allclose(g.apply_rope(q16, f16).float(),
                         g.apply_rope(q, f16).float(), atol=2e-2, rtol=2e-2),
          (g.apply_rope(q16, f16).float() - g.apply_rope(q, f16).float()).abs().max())


def case_tiny_forward() -> None:
    """小配置跑通：形状、有限值、确定性、非 patch 整数倍尺寸、文本支路必须被吃。"""
    torch.manual_seed(1)
    shapes = _fake_shapes()
    model = g.build_from_shapes(shapes, patch=2).eval()
    shape = model.shape_
    check("`infer_shape` 从形状反推的维度与造数据时给定的一致",
          (shape.features, shape.layers, shape.heads, shape.kvheads, shape.mlpdim,
           shape.channels, shape.txtlayers, shape.txt_blocks, shape.refiner_layers)
          == (64, 2, 4, 1, 96, 4, 2, 1, 2), shape)
    check("`txtlayers`（输入层数 2）与 `txt_blocks`（块模块数 1）**分开记**："
          "同数记成一个会把块数建错（盘上 12 vs 2）",
          shape.txtlayers == 2 and shape.txt_blocks == 1, (shape.txtlayers, shape.txt_blocks))
    check("骨架键集合 == `expected_keys`（不多不少）",
          set(model.state_dict()) == g.expected_keys(shape),
          sorted(set(model.state_dict()) ^ g.expected_keys(shape))[:6])

    x = torch.randn(1, 4, 8, 12)          # 8、12 都能被 patch=2 整除
    t = torch.tensor([0.4])
    ctx = torch.randn(1, 6, shape.txtlayers * shape.txtdim)
    with torch.no_grad():
        out = model(x, t, ctx)
        again = model(x, t, ctx)
    check("前向输出形状 == 输入形状（补零后要裁回去）", out.shape == x.shape, out.shape)
    check("前向输出全有限（NaN/Inf 说明调制或归一化写错）", torch.isfinite(out).all(),
          out.abs().max())
    check("同输入同输出（没有隐式随机）", torch.equal(out, again))
    with torch.no_grad():
        odd = model(torch.randn(1, 4, 7, 9), t, ctx)   # 7、9 不是 patch 的倍数 ⇒ 走补零
    check("非 patch 整数倍的 latent 也能跑并裁回原尺寸", odd.shape == (1, 4, 7, 9), odd.shape)
    with torch.no_grad():
        other = model(x, t, torch.randn(1, 6, shape.txtlayers * shape.txtdim))
        other_t = model(x, torch.tensor([0.9]), ctx)
    check("换文本条件 ⇒ 输出必变（文本支路真的接进图，不是摆设）",
          not torch.allclose(out, other), (out - other).abs().max())
    check("换 timestep ⇒ 输出必变（时间调制真的接进图）",
          not torch.allclose(out, other_t), (out - other_t).abs().max())


def case_reject() -> None:
    """形状自相矛盾必须当场报错 —— 建出「差一层」的模型比报错危险得多。"""
    def raised(mutate) -> bool:
        shapes = _fake_shapes()
        mutate(shapes)
        try:
            g.build_from_shapes(shapes, patch=2)
        except g.ShapeError:
            return True
        return False

    check("`patch` 不能整除 `first` 入维 ⇒ ShapeError",
          raised(lambda s: s.__setitem__("first.weight", (64, 6))))
    check("`first` 入维 ≠ `last` 出维 ⇒ ShapeError（首尾不是同一结构）",
          raised(lambda s: s.__setitem__("last.linear.weight", (12, 64))))
    check("缺 `blocks.1.mlp.gate.weight`（少一层）⇒ ShapeError，不是「静默建 1 层」",
          raised(lambda s: s.pop("blocks.1.mlp.gate.weight")))
    def _renumber_to_gap(s: dict[str, tuple[int, ...]]) -> None:
        """把块 1 整块改名成块 2（**先 pop 再写** —— 只复制不删的话块号仍是 0,1,2 连续）。"""
        for key in [k for k in s if k.startswith("blocks.1.")]:
            s[key.replace("blocks.1.", "blocks.2.")] = s.pop(key)

    check("块号不连续（0、2 跳 1）⇒ ShapeError", raised(_renumber_to_gap))
    check("层宽不一致（块 1 的 wq 换个宽度）⇒ ShapeError",
          raised(lambda s: s.__setitem__("blocks.1.attn.wq.weight", (80, 64))))

    model = g.build_from_shapes(_fake_shapes(), patch=2).eval()
    bad_ctx = torch.randn(1, 6, 99)
    try:
        with torch.no_grad():
            model(torch.randn(1, 4, 8, 8), torch.tensor([0.5]), bad_ctx)
    except g.ShapeError:
        ok = True
    else:
        ok = False
    check("文本条件宽度不对 ⇒ ShapeError（拿错 TE 时不许静默 reshape 成一堆噪声）", ok)
    try:
        with torch.no_grad():
            model(torch.randn(1, 5, 8, 8), torch.tensor([0.5]), torch.randn(1, 6, 64))
    except g.ShapeError:
        ok = True
    else:
        ok = False
    check("latent 通道数不对 ⇒ ShapeError", ok)


def case_real_weights() -> None:
    """拿盘上真实 krea2 文件（878 张量）的形状做双向对账 —— 环只有这里才真正闭上。"""
    if not REAL_KREA2.exists():
        check(f"真实权重对账（{REAL_KREA2.name} 不在盘上 ⇒ 跳过，判据未生效）", True,
              "skipped")
        return
    info = inspect_safetensors(str(REAL_KREA2))
    shapes = {k: tuple(e.shape) for k, e in info.tensors.items()}
    shape = g.infer_shape(shapes, patch=2)
    model = g.Krea2DiT(shape)
    mine = set(model.state_dict())
    real = g.weight_tensor_keys(shapes)
    check("真实 krea2：骨架键 ⊇ 权重参数键（权重里每个参数都得有落点）",
          not (real - mine), sorted(real - mine)[:8])
    check("真实 krea2：权重参数键 ⊇ 骨架键（骨架不许凭空多出参数）",
          not (mine - real), sorted(mine - real)[:8])
    mismatched = [(k, tuple(v.shape), shapes[k]) for k, v in model.state_dict().items()
                  if k in shapes and tuple(v.shape) != shapes[k]]
    check("真实 krea2：878 个张量逐键形状相等（维度全靠反推，没有一处对不上）",
          not mismatched, mismatched[:6])
    check("真实 krea2 反推的结构常量（28 层 / hidden 6144 / 48 头 / 12 kv 头 / SwiGLU 16384）",
          (shape.layers, shape.features, shape.heads, shape.kvheads, shape.mlpdim)
          == (28, 6144, 48, 12, 16384), shape)


def main() -> int:
    case_math()
    case_tiny_forward()
    case_reject()
    case_real_weights()

    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
