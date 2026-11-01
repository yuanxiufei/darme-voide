"""``dit_lumina``（Z-Image / NextDiT 骨架）的判据。

盯四类事：

1. **约定别抄错**：RMSNorm 是**直接乘**（不是 Krea2 的 ``scale+1``）、时间轴是
   ``(1 - t) * 1000``、三轴 RoPE ``theta=256`` 且维度之和 == head_dim；
2. **结构与权重对得上**：用盘上真实 ``z_image_turbo`` 的 453 个张量双向对账键集合与逐键形状；
3. **前向真的算**：小配置跑通、确定性、非 patch 整数倍、文本/时间支路都要各自生效；
   **补位是不生效的**（参考 ``pad_tokens_multiple=None`` ⇒ 改 pad token 不许影响输出）；
4. **矛盾当场报**：形状自相矛盾、refiner 调制接反、拿错 TE 一律 ``ShapeError``。

⚠️ 只读 safetensors 头，不读 12 GB 权重内容。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.services.engine import dit_lumina as m  # noqa: E402
from app.services.engine.safetensors import inspect as inspect_safetensors  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

REAL_ZIMAGE = Path(
    r"D:/Comfy-Desktop/ComfyUI-Shared/models/unet/z_image_turbo_bf16.safetensors")


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class _Spy(torch.nn.Module):
    """记录「真正喂给时间嵌入」的标量 —— 「先取反再乘 1000」只能这样验，形状看不出来。"""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: torch.Tensor | None = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        self.seen = t.detach().clone()
        return self.inner(t)


def _fake_shapes(features: int = 64, layers: int = 2, heads: int = 4, head_dim: int = 16,
                 kvheads: int = 4, mlpdim: int = 96, patch: int = 2, channels: int = 4,
                 cap_dim: int = 32, t_freq: int = 16, t_inner: int = 32,
                 t_out: int | None = None, noise: int = 2, context: int = 1,
                 pad: bool = True) -> dict[str, tuple[int, ...]]:
    """造一份与真文件同构的最小形状表（键名照权重：``layers.N.attention.qkv.weight``…）。

    ``kvheads`` 默认与 ``heads`` 相等（Z-Image 就是不做 GQA）；``t_out`` 是时间嵌入输出宽度，
    它**必须**等于块调制入维 ``min(dim, 256)`` —— 这里按 ``features`` 取（真文件 dim=3840 ⇒ 256）。
    """
    t_out = features if t_out is None else t_out
    p2 = patch * patch
    shapes: dict[str, tuple[int, ...]] = {
        "x_embedder.weight": (features, channels * p2), "x_embedder.bias": (features,),
        "t_embedder.mlp.0.weight": (t_inner, t_freq), "t_embedder.mlp.0.bias": (t_inner,),
        "t_embedder.mlp.2.weight": (t_out, t_inner), "t_embedder.mlp.2.bias": (t_out,),
        "cap_embedder.0.weight": (cap_dim,),
        "cap_embedder.1.weight": (features, cap_dim), "cap_embedder.1.bias": (features,),
        "final_layer.linear.weight": (channels * p2, features),
        "final_layer.linear.bias": (channels * p2,),
        "final_layer.adaLN_modulation.1.weight": (features, t_out),
        "final_layer.adaLN_modulation.1.bias": (features,),
    }
    if pad:
        shapes |= {"x_pad_token": (1, features), "cap_pad_token": (1, features)}

    def block(head: str, modulated: bool) -> None:
        shapes.update({
            head + "attention_norm1.weight": (features,),
            head + "attention_norm2.weight": (features,),
            head + "ffn_norm1.weight": (features,),
            head + "ffn_norm2.weight": (features,),
            head + "attention.qkv.weight": ((heads + 2 * kvheads) * head_dim, features),
            head + "attention.out.weight": (features, heads * head_dim),
            head + "attention.q_norm.weight": (head_dim,),
            head + "attention.k_norm.weight": (head_dim,),
            head + "feed_forward.w1.weight": (mlpdim, features),
            head + "feed_forward.w3.weight": (mlpdim, features),
            head + "feed_forward.w2.weight": (features, mlpdim),
        })
        if modulated:
            shapes.update({head + "adaLN_modulation.0.weight": (4 * features, t_out),
                           head + "adaLN_modulation.0.bias": (4 * features,)})

    for i in range(layers):
        block(f"layers.{i}.", True)
    for i in range(noise):
        block(f"noise_refiner.{i}.", True)
    for i in range(context):
        block(f"context_refiner.{i}.", False)
    return shapes


def case_math() -> None:
    """约定：RMSNorm 直乘、三轴维度、时间轴 ``(1-t)×1000``、调制顺序。"""
    torch.manual_seed(0)
    norm = m.PlainRMSNorm(8)
    x = torch.randn(2, 5, 8)
    check("PlainRMSNorm 在 weight=1 时等于 `F.rms_norm(weight=None)` —— 即**直接乘**，"
          "不是 Krea2 的 `scale+1`（抄错会让每层输出大一倍）",
          torch.allclose(norm(x), F.rms_norm(x, (8,), weight=None, eps=m.NORM_EPS), atol=1e-6))
    with torch.no_grad():
        norm.weight.fill_(2.0)
    check("PlainRMSNorm 的 weight=2 ⇒ 输出正好 ×2（不是 ×3）",
          torch.allclose(norm(x), F.rms_norm(
              x, (8,), weight=torch.full((8,), 2.0), eps=m.NORM_EPS), atol=1e-6))

    shape = m.LuminaShape(features=64, layers=1, heads=4, head_dim=128, kvheads=4,
                          mlpdim=96, patch=2, channels=4, cap_dim=32, t_freq=16,
                          t_inner=32, t_out=8, refiner_layers=2,
                          context_refiner_layers=1, mod_dim=64)
    check("三轴 RoPE：head_dim=128 ⇒ [32,48,48] 且和为 128", shape.axes_dim == [32, 48, 48]
          and sum(shape.axes_dim) == 128, shape.axes_dim)
    check("theta 是 256 而不是 Krea2 的 1e3（形状里看不出来，只能靠默认值锁住）",
          shape.theta == 256.0 and shape.time_scale == 1000.0, (shape.theta, shape.time_scale))

    model = m.build_from_shapes(_fake_shapes(), patch=2).eval()
    spy = _Spy(model.t_embedder)
    model.t_embedder = spy
    with torch.no_grad():
        model(torch.randn(1, 4, 8, 8), torch.tensor([0.25]), torch.randn(1, 5, 32))
    check("时间轴口径：进 embedder 的是 `(1 - t) * 1000`（写成 t*1000 形状一样、数值全错）",
          spy.seen is not None and torch.allclose(spy.seen, torch.tensor([0.75 * 1000.0])),
          spy.seen)


def case_tiny_forward() -> None:
    """小配置跑通：形状、确定性、补位语义、各支路都真的接进图。"""
    torch.manual_seed(1)
    shapes = _fake_shapes()
    model = m.build_from_shapes(shapes, patch=2).eval()
    shape = model.shape_
    check("`infer_shape` 反推的维度与造数据时给定的一致",
          (shape.features, shape.layers, shape.heads, shape.kvheads, shape.mlpdim,
           shape.channels, shape.cap_dim, shape.mod_dim, shape.refiner_layers,
           shape.context_refiner_layers) == (64, 2, 4, 4, 96, 4, 32, 64, 2, 1), shape)
    check("骨架键集合 == `expected_keys`（不多不少）",
          set(model.state_dict()) == m.expected_keys(shape),
          sorted(set(model.state_dict()) ^ m.expected_keys(shape))[:6])

    x = torch.randn(1, 4, 8, 12)
    t = torch.tensor([0.4])
    ctx = torch.randn(1, 6, shape.cap_dim)
    with torch.no_grad():
        out = model(x, t, ctx)
        again = model(x, t, ctx)
    check("前向输出形状 == 输入形状（补零后要裁回去）", out.shape == x.shape, out.shape)
    check("前向输出全有限", torch.isfinite(out).all(), out.abs().max())
    check("同输入同输出（没有隐式随机）", torch.equal(out, again))
    with torch.no_grad():
        odd = model(torch.randn(1, 4, 7, 9), t, ctx)
    check("非 patch 整数倍的 latent 也能跑并裁回原尺寸", odd.shape == (1, 4, 7, 9), odd.shape)

    with torch.no_grad():
        other_ctx = model(x, t, torch.randn(1, 6, shape.cap_dim))
        other_t = model(x, torch.tensor([0.9]), ctx)
        longer = model(x, t, torch.randn(1, 40, shape.cap_dim))
    check("换文本条件 ⇒ 输出必变（文本支路不是摆设）",
          not torch.allclose(out, other_ctx), (out - other_ctx).abs().max())
    check("换 timestep ⇒ 输出必变（调制真接进图）",
          not torch.allclose(out, other_t), (out - other_t).abs().max())
    check("文本长度变化（6 ⇒ 40，跨过补位边界）⇒ 输出必变且形状不变",
          longer.shape == x.shape and not torch.allclose(out, longer))

    with torch.no_grad():
        model.cap_pad_token.fill_(5.0)
        first = model(x, t, ctx)
        before = model.cap_pad_token.clone()
        model.cap_pad_token.fill_(-5.0)
        second = model(x, t, ctx)
        model.cap_pad_token.copy_(before)
    #: ⚠️ 这条断言上轮是**反的** ✗（当时断言「改 pad token 输出必须变」⇒ 那是在锁住"填位"这个错行为 ✓✗）。
    #: 参考真值是 ``pad_tokens_multiple=None`` ⇒ 前向**从不填** ✓ ⇒ 改 pad token **不许**影响输出 ✓。
    #: 这条还顺手管住「x/cap_pad_token 被误用」：两个参数只要有一个被喂进图，这里立刻红 ✓。
    check("**不填位** ✓：改 `cap_pad_token` 不影响输出（参考 `pad_tokens_multiple=None` ✓✗）",
          torch.allclose(first, second), (first - second).abs().max())

    with torch.no_grad():
        no_pad = m.build_from_shapes(_fake_shapes(pad=False), patch=2).eval()
        out_np = no_pad(x, t, ctx)
    check("`pad=False` 的权重（无补位 token）也能建也能跑、且不出 pad 相关键",
          out_np.shape == x.shape and not any("pad_token" in k for k in no_pad.state_dict()))


def case_reject() -> None:
    """矛盾当场报错 —— 建出「调制接反」的模型比报错危险得多。"""
    def raised(mutate, **kwargs) -> bool:
        shapes = _fake_shapes()
        mutate(shapes)
        try:
            m.build_from_shapes(shapes, patch=2, **kwargs)
        except m.ShapeError:
            return True
        return False

    check("`patch` 不能整除 `x_embedder` 入维 ⇒ ShapeError",
          raised(lambda s: s.__setitem__("x_embedder.weight", (64, 6))))
    check("`x_embedder` 入维 ≠ `final_layer` 出维 ⇒ ShapeError（首尾不是同一结构）",
          raised(lambda s: s.__setitem__("final_layer.linear.weight", (12, 64))))
    check("缺 `layers.1.feed_forward.w1.weight`（少一层）⇒ ShapeError，不许静默建 1 层",
          raised(lambda s: s.pop("layers.1.feed_forward.w1.weight")))

    def _renumber_to_gap(s: dict[str, tuple[int, ...]]) -> None:
        for key in [k for k in s if k.startswith("layers.1.")]:
            s[key.replace("layers.1.", "layers.2.")] = s.pop(key)

    check("块号不连续（0、2 跳 1）⇒ ShapeError", raised(_renumber_to_gap))
    check("层宽不一致（块 1 的 qkv 换个宽度）⇒ ShapeError",
          raised(lambda s: s.__setitem__("layers.1.attention.qkv.weight", (400, 64))))
    check("qkv 输出维分不出 k/v（多 1 个 head_dim 的一半）⇒ ShapeError",
          raised(lambda s: s.__setitem__("layers.0.attention.qkv.weight", (200, 64))))
    check("noise_refiner 缺调制 ⇒ ShapeError（它必须带 adaLN）",
          raised(lambda s: [s.pop(k) for k in list(s)
                            if k.startswith("noise_refiner.1.adaLN_modulation.")]))
    check("context_refiner 多出调制 ⇒ ShapeError（它按参考实现不带 adaLN）",
          raised(lambda s: s.__setitem__(
              "context_refiner.0.adaLN_modulation.0.weight", (256, 64))))
    check("有 `x_pad_token` 却没有 `cap_pad_token` ⇒ ShapeError（补位 token 必须成对）",
          raised(lambda s: s.pop("cap_pad_token")))
    check("块调制宽度 ≠ 4×hidden ⇒ ShapeError（不是 4 段调制就别 chunk 4）",
          raised(lambda s: s.__setitem__("layers.0.adaLN_modulation.0.weight", (255, 64))))
    check("时间嵌入输出维 ≠ 块调制入维 ⇒ ShapeError（真文件里两者都是 256，接不上就是拿错模型）",
          raised(lambda s: [s.__setitem__("t_embedder.mlp.2.weight", (8, 32)),
                            s.__setitem__("t_embedder.mlp.2.bias", (8,))]))
    check("调制是 SiLU 版（Linear 在 .1）却按 .0 找 ⇒ ShapeError（版式对不上就该报）",
          raised(lambda s: [s.__setitem__(
              k.replace("adaLN_modulation.0.", "adaLN_modulation.1."), s.pop(k))
              for k in list(s) if k.startswith("layers.0.adaLN_modulation.0.")]))

    model = m.build_from_shapes(_fake_shapes(), patch=2).eval()
    bad = ((torch.randn(1, 4, 8, 8), torch.tensor([0.5]), torch.randn(1, 6, 99), "文本宽度"),
           (torch.randn(1, 5, 8, 8), torch.tensor([0.5]), torch.randn(1, 6, 32), "latent 通道"),
           (torch.randn(1, 4, 8, 8), torch.tensor([0.5, 0.5]), torch.randn(1, 6, 32),
            "timesteps 个数"))
    for x, t, ctx, label in bad:
        try:
            with torch.no_grad():
                model(x, t, ctx)
        except m.ShapeError:
            ok = True
        else:
            ok = False
        check(f"{label}不对 ⇒ ShapeError（拿错 TE / 送错 batch 不许静默算成噪声）", ok)


def case_real_weights() -> None:
    """盘上真实 z_image_turbo（453 张量）双向对账 —— 环只有这里才真正闭上。"""
    if not REAL_ZIMAGE.exists():
        check(f"真实权重对账（{REAL_ZIMAGE.name} 不在盘上 ⇒ 跳过，判据未生效）", True,
              "skipped")
        return
    info = inspect_safetensors(str(REAL_ZIMAGE))
    shapes = {k: tuple(e.shape) for k, e in info.tensors.items()}
    shape = m.infer_shape(shapes, patch=2)
    model = m.LuminaDiT(shape)
    mine, real = set(model.state_dict()), m.weight_tensor_keys(shapes)
    check("真实 z_image：骨架键 ⊇ 权重参数键（每个参数都有落点）", not (real - mine),
          sorted(real - mine)[:8])
    check("真实 z_image：权重参数键 ⊇ 骨架键（骨架不许凭空多出参数）", not (mine - real),
          sorted(mine - real)[:8])
    mismatched = [(k, tuple(v.shape), shapes[k]) for k, v in model.state_dict().items()
                  if k in shapes and tuple(v.shape) != shapes[k]]
    check("真实 z_image：453 个张量逐键形状相等", not mismatched, mismatched[:6])
    check("真实 z_image 反推的结构常量（30 层 / hidden 3840 / 30 头 / 非 GQA / "
          "SwiGLU 10240 / cap 2560 / 调制 256 / **不补位**）",
          (shape.layers, shape.features, shape.heads, shape.kvheads, shape.mlpdim,
           shape.cap_dim, shape.mod_dim, shape.pad_tokens, shape.refiner_layers,
           shape.context_refiner_layers) == (30, 3840, 30, 30, 10240, 2560, 256, 0, 2, 2),
          shape)
    #: ⚠️ 「建了但不用」必须**分开断言** ✗✓：合成一格就分不清是「键少了」还是「填了不该填的」✓。
    #: 参考 `NextDiT.__init__(pad_tokens_multiple=None)` ⇒ 两个 pad token **在权重里**、
    #: 前向**从不使用** ✓（证据：`comfy/ldm/lumina/model.py` 的 `if self.pad_tokens_multiple is not None`）。
    check("真实 z_image：`has_pad_tokens` 必须是**真** ✓（权重里确实有 x/cap_pad_token ⇒ 必须建）",
          shape.has_pad_tokens is True, shape.has_pad_tokens)
    check("真实 z_image：`pad_tokens` 必须是 **0** ✓✗（上轮估成 32 ⇒ 每图多填 ≤31 个"
          "**不遮罩**的 token ⇒ 形状对、结果偏）", shape.pad_tokens == 0, shape.pad_tokens)


def _fake_info(shapes: dict[str, tuple[int, ...]], dtype: str = "BF16") -> object:
    """造一份**只有头部**的 ``SafetensorsInfo`` —— 判族/建骨架都只认这个 ✓（不读 12 GB 内容 ✓）。"""
    from app.services.engine.safetensors import SafetensorsInfo, TensorEntry

    return SafetensorsInfo(
        path="<假>.safetensors", metadata={},
        tensors={name: TensorEntry(name, dtype, list(shape), 0, 0)
                 for name, shape in shapes.items()})


class _Recorder(torch.nn.Module):
    """记下**真正喂给主干的时间参数** —— σ 的"形态"这件事，看形状才分得清 ✓。"""

    def __init__(self) -> None:
        super().__init__()
        self.seen: object = None

    def forward(self, x: torch.Tensor, t: object, context: torch.Tensor) -> torch.Tensor:
        self.seen = t.detach().clone() if isinstance(t, torch.Tensor) else t
        return torch.zeros_like(x)


def _bare_backend(family: str | None) -> tuple[object, _Recorder]:
    """不装真权重、只测**接线**的后端（绕过 `_gate`/设备校验 ⇒ 判据只盯"喂了什么" ✓）。"""
    from app.services.engine import torch_backend as tb

    backend = object.__new__(tb.TorchBackend)
    recorder = _Recorder()
    backend._model = recorder
    backend._ditFamily = family
    backend._form = "dit"
    #: ⚠️ `vae_scale` 由 `_build_zimage_module` 在**真装载**时补进 config ✓（同族事实 ✓）——
    #: 裸后端要显式给 ✓，不给就等于「config 里读不到」✓（那条反向断言见接线④′ ✓）。
    backend._config = ({"family": family, "vae_scale": 8} if family else None)
    backend._gate = lambda: None
    backend._require_own_device = lambda *a, **k: None
    backend._torch = lambda: torch
    return backend, recorder


def case_precision() -> None:
    """**精度**这一维：盘上真权重是 bf16 ⇒ 整条链必须能在 bf16 上跑完 ✓。

    ⚠️ 为什么单列一案 ✗：前面几条全用 fp32 ✓ —— 而 ``rope_freqs`` 返回的正是 fp32 ✓，
    在 fp32 模型里"把 q/k 升成 fp32"**看不出任何异常** ✓✗；换成 bf16 就当场撞 SDPA 的
    "query float32 / value bf16" ✗✗（2026-10-31 在 11.46 GiB 真权重上实测到的就是这条 ✓）。
    """
    shape = m.infer_shape(_fake_shapes(), patch=2)
    model = m.LuminaDiT(shape).to(torch.bfloat16).eval()
    x = torch.randn(1, shape.channels, 8, 8, dtype=torch.bfloat16)
    ctx = torch.randn(1, 5, shape.cap_dim, dtype=torch.bfloat16)
    t = torch.full((1,), 0.7, dtype=torch.bfloat16)
    out = None
    detail: object = ""
    try:
        with torch.no_grad():
            out = model(x, t, ctx)
    except RuntimeError as err:                     # ⚠️ 不许把异常直接抛出去 ✗：那样整套是"崩"，
        detail = str(err)                           #    而不是一条能指名道姓的 FAIL ✓（README 口径 ✓）
    check("精度：bf16 整链跑通 ✓ 且输出仍 bf16 ✓（RoPE 一升精度 ⇒ SDPA 当场报 dtype 不符 ✗）",
          out is not None and out.dtype == torch.bfloat16 and tuple(out.shape) == tuple(x.shape),
          detail or (out.dtype, tuple(out.shape)))
    check("精度：bf16 前向出的数是**有限值** ✓（不是 nan/inf 混过去 ✓）",
          out is not None and bool(torch.isfinite(out).all()),
          detail or float(out.abs().max()))


def case_wiring() -> None:
    """**接线**判据：判族 ⇒ 选骨架 ⇒ 前向口径 ⇒ 自述 —— 这四步任何一步没接，骨架就白写了。"""
    from app.services.engine import torch_backend as tb

    info = _fake_info(_fake_shapes())
    check("接线① 判族：Z-Image 键名表 ⇒ `zimage` ✓（只看键名 ✗不看文件名 ✓）",
          tb._dit_family_of(info) == tb.ZIMAGE_FAMILY, tb._dit_family_of(info))
    krea2 = _fake_info({"first.weight": (64, 4), "blocks.0.attn.wq.weight": (64, 64),
                        "last.linear.weight": (4, 64), "txtfusion.0.weight": (64, 64)})
    check("接线① 判族：认不出的键布局 ⇒ **不给**专用骨架（走老路 ✓，绝不硬塞 Z-Image ✓）",
          tb._dit_family_of(krea2) != tb.ZIMAGE_FAMILY, tb._dit_family_of(krea2))
    check("接线① 判族：空 info / 没 info ⇒ None ✓（认不出就**不猜** ✗，也不许炸 ✗）",
          tb._dit_family_of(_fake_info({})) is None and tb._dit_family_of(None) is None,
          (tb._dit_family_of(_fake_info({})), tb._dit_family_of(None)))

    module, config = tb._build_zimage_module(info, {})
    check("接线② 没给 `patch` ⇒ 建得出来，但报告必须写明**这是默认值**（推不出来 ✗）",
          config["family"] == tb.ZIMAGE_FAMILY
          and config["configSources"]["patch"] == "default", config.get("configSources"))
    _, config2 = tb._build_zimage_module(info, {"patch": 2})
    check("接线② 显式给 `patch` ⇒ 报告写 `explicit` ✓（同一个数值也要分清来源 ✓）",
          config2["configSources"]["patch"] == "explicit", config2.get("configSources"))
    check("接线② 文本宽度映射到 `cap_dim` ✓（后端问的是 `text_dim` ✗，不映射就悄悄回落 ⇒ 白装 ✓✗）",
          config["text_dim"] == config["cap_dim"] == 32, (config["text_dim"], config["cap_dim"]))
    check("接线② 建出来的是 `LuminaDiT` ✓ 且键集与形状表同构 ✓",
          isinstance(module, m.LuminaDiT)
          and not (set(module.state_dict()) - set(_fake_shapes())))

    x = torch.zeros(2, 4, 8, 8)
    ctx = torch.zeros(2, 5, 32)
    backend, recorder = _bare_backend(tb.ZIMAGE_FAMILY)
    backend.denoise(x, 0.7, ctx, None)
    check("接线③ Z-Image 前向拿到 **(b,) 张量** ✓ 且是**原样 σ**（先取反 = 错两次 ✗）",
          isinstance(recorder.seen, torch.Tensor) and tuple(recorder.seen.shape) == (2,)
          and torch.allclose(recorder.seen, torch.full((2,), 0.7)), recorder.seen)
    check("接线③ …且 dtype 跟 latent 走 ✓（bf16 latent 配 fp32 σ ⇒ 第一层调制就撞 dtype ✗）",
          recorder.seen.dtype == x.dtype, recorder.seen.dtype)
    backend, recorder = _bare_backend(None)
    backend.denoise(x, 0.7, ctx, None)
    check("接线③ 通用 DiT 仍拿**标量** ✓（老路一字未改 ✓）",
          isinstance(recorder.seen, float) and recorder.seen == 0.7, recorder.seen)

    backend, _ = _bare_backend(tb.ZIMAGE_FAMILY)
    check("接线④ 自述：装上 Z-Image（config 带 `vae_scale` ✓ 同真装载 ✓）⇒ `real-shape` ✓",
          backend._latent_mode() == "real-shape", backend._latent_mode())
    #: ⚠️ 反向：config 里**读不到** `vae_scale` ⇒ 必须老实报占位 ✓✗。
    #: 上轮这里两处口径打架 ✓（`_latent_mode` 的注释说"读不到就报占位" ✓、断言却要它报 real-shape ✗）——
    #: 正确的分工是：**真装载**（`_build_zimage_module` ✓）负责把 `vae_scale` 补进 config ✓，
    #: 自述只如实反映 config ✓；把"应该补"和"补没补"混成一条断言，就看不出哪一头坏了 ✓。
    stale, _ = _bare_backend(tb.ZIMAGE_FAMILY)
    stale._config = {"family": tb.ZIMAGE_FAMILY}
    check("接线④′ 反向：config 里**没有** `vae_scale` ⇒ 老实报 `placeholder-1d` ✓✗（不许自述成真形状 ✓）",
          stale._latent_mode() == "placeholder-1d", stale._latent_mode())
    backend, _ = _bare_backend(None)
    check("接线④ 自述：认不出族 ⇒ 回老路的占位口径 ✓（没有被这次接线改掉 ✓）",
          backend._latent_mode() == "placeholder-1d", backend._latent_mode())

    if REAL_ZIMAGE.exists():
        real = inspect_safetensors(str(REAL_ZIMAGE))
        check("接线① 盘上真文件 ⇒ `zimage` ✓（453 张量只读头部 ✓）",
              tb._dit_family_of(real) == tb.ZIMAGE_FAMILY, tb._dit_family_of(real))
    else:
        check("接线① 真文件判族（不在盘上 ⇒ 跳过，判据未生效）", True, "skipped")


def main() -> int:
    case_math()
    case_tiny_forward()
    case_reject()
    case_real_weights()
    case_precision()
    case_wiring()

    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
