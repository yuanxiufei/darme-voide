"""自检：**MiniMax Music3** 两段口径 —— 音频 DiT（`dit_music3`）与音频 VAE（`vae_music3`）✓。

跑法：``.venv/Scripts/python.exe tests/engine_music3_test.py``

这族是**全仓唯一不是图**的模型 ✓✗：潜变量 **3 维** ``(b, 128, T)`` ✓、``T`` 是**潜帧**
（86.13 Hz ✓）不是像素边；时间缩放是 **512 样本/潜帧** 而不是"边长 ÷8" ✓。三处
**不报错、只有歌不对** 的口径在这里逐条钉住：``t = 1 − σ`` ✓、``return -out`` ✓、
``潜帧 = 条件帧 × 3.4453125`` ✓。
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp936 下 `✓` 会 UnicodeEncodeError
except Exception:                              # pragma: no cover
    pass

_RESULTS: list[tuple[bool, str, object]] = []
_SKIPPED: list[str] = []


def check(name: str, condition: object, detail: object = "") -> bool:
    ok = bool(condition)
    _RESULTS.append((ok, name, detail))
    print(f"{'✓' if ok else '✗'} {name}"
          + (f"   {detail}" if detail and not ok else ""))
    return ok


def _error_of(call) -> str:
    """跑一段必炸的调用，返回异常文本 ✓（判据要"炸得对"✓）。"""
    try:
        call()
    except Exception as err:            # noqa: BLE001 —— 这里就是要接住任何异常 ✓
        return f"{type(err).__name__}: {err}"
    return ""


#: 小号合成结构（跑得动就行 ✓）。⚠️ ``heads`` **权重里没有** ✓ ⇒ 只能按
#: `DEFAULT_DIM_HEADS`(=64) 推 ⇒ ``dim`` 必须被 64 除尽 ✓✗。
_D, _FF, _C, _CL, _CH, _CP, _ROT, _FREQ = 128, 128, 4, 2, 8, 8, 8, 8
_IN_DIM = _C * 2 + _CP          # 16 ✓
_CTX_DIM = _CL * _CH            # 16 ✓
_LAYER = "diffusion_transformer.transformer.layers"


def _shapes() -> dict[str, tuple[int, ...]]:
    """一份**自洽**的合成形状表 ✓（键名后缀与真权重逐字一致 ✓）。"""
    return {
        "diffusion_transformer.postprocess_conv.weight": (_C, _D, 3),
        "diffusion_transformer.preprocess_conv.weight": (_D, _IN_DIM, 1),
        "diffusion_transformer.transformer.project_in.weight": (_D, _IN_DIM),
        "diffusion_transformer.transformer.project_out.weight": (_C, _D),
        "diffusion_transformer.to_timestep_embed.0.weight": (_D, _FREQ * 2),
        "diffusion_transformer.to_timestep_embed.2.weight": (_D, _D),
        "diffusion_transformer.timestep_features.weight": (_FREQ, 1),
        "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": (_ROT // 2,),
        "latent_conditioners.0.weight": (_CP, _CH, 3),
        "cond_layer_logits": (_CL, 1, _CH),
        f"{_LAYER}.0.ff.ff.0.proj.weight": (_FF * 2, _D),
        f"{_LAYER}.0.ff.ff.2.weight": (_D, _FF),
        f"{_LAYER}.1.attn.qkv.weight": (3 * _D, _D),
    }


def _shape(**override: int):
    """合成形状 → :class:`Music3Shape` ✓（override 用来做反向验证 ✓）。"""
    from app.services.engine import dit_music3  # noqa: PLC0415

    return dit_music3.infer_shape(_shapes(), **override)


# ── ① 帧率换算：条件 25 Hz → 潜帧 86.13 Hz ──────────────────────────────────

def case_rates() -> None:
    from app.services.engine import dit_music3, vae_music3  # noqa: PLC0415

    got = [(f, dit_music3.latent_length(f)) for f in (4, 100, 200, 1000)]
    check("①-1 200 条件帧 ⇒ **689** 潜帧 ✓（＝ 8 秒 × 86.13 Hz ✓）",
          got == [(4, 13), (100, 344), (200, 689), (1000, 3445)], dict(got))

    #: ⭐ 两个错误值的排除 ✓✗：漏 VAE hop ⇒ 367 ✓、漏重采样 ⇒ 375 ✓。
    #: 三个数彼此都不是"差不多" ✓ ⇒ 抄成一个乘数就会露 ✓。
    check("①-2 689 既不是 367（漏 960/512）也不是 375（漏 44100/24000）✓",
          int(200 * 44100 / 24000) == 367 and int(200 * 960 / 512) == 375
          and dit_music3.latent_length(200) == 689)

    raw = [int(f * 44100 / 24000 * 960 / 512) for f in range(1, 400)]
    check("①-3 逐字照抄（不先约分）与 ×3.4453125 恒等 ✓（1…399 帧逐帧核 ✓）",
          raw == [int(f * 3.4453125) for f in range(1, 400)])

    check("①-4 689 潜帧 = **352 768** 样本（512×689 ✓）= 7.9993 秒 ✓",
          vae_music3.samples_for_frames(689) == 352768
          and abs(vae_music3.seconds_for_frames(689) - 8.0) < 0.01,
          vae_music3.seconds_for_frames(689))

    #: 反向必须**下取整** ✓：10 秒 ⇒ 861 而不是 862 ✓（862 会多出 8 ms 尾巴 ⇒ 拼接爆音 ✓✗）。
    check("①-5 秒 → 潜帧下取整 ✓（8 秒⇒689 ✓、10 秒⇒861 ✓）",
          (vae_music3.latent_frames_for_seconds(8.0),
           vae_music3.latent_frames_for_seconds(10.0)) == (689, 861))

    check("①-6 潜帧率 44100/512 = 86.1328 Hz ✓（**不是** 25 Hz 条件帧率 ✓✗）",
          abs(vae_music3.MUSIC3_DAV_FACTS["latentHz"] - 86.1328125) < 1e-6,
          vae_music3.MUSIC3_DAV_FACTS["latentHz"])


# ── ② 前向契约与潜空间口径 ─────────────────────────────────────────────────

def case_contract() -> None:
    from app.services.engine import dit_build  # noqa: PLC0415
    from app.services.engine.latent_formats import get_latent_format  # noqa: PLC0415

    contract = dit_build.contract_of("music3")
    check("②-1 契约三格：**3 维** ／ σ 批形态 ／ 口径 `minimax_music3` ✓",
          (contract.latent_dims, contract.sigma_form, contract.latent_format)
          == (3, "batch", "minimax_music3"))

    spec = get_latent_format("minimax_music3")
    check("②-2 口径表：128 通道 ／ **1 维**（无空间维 ✓）／ 时间 ÷512 ／ scale 恒 1.0 ✓",
          (spec.latent_channels, spec.latent_dimensions, spec.temporal_downscale_ratio,
           spec.spatial_downscale_ratio, spec.scale_factor) == (128, 1, 512, 1, 1.0))

    check("②-3 别名 `music3` 指回同一条 ✓（不许有第二条同名口径 ✗）",
          get_latent_format("music3") is spec and contract.latent_spec() is spec)

    bad = [_error_of(lambda v=v: dit_build.ForwardContract(
        latent_dims=v, sigma_form="batch", latent_format="minimax_music3"))
        for v in (6, 2)]
    check("②-4 `latent_dims` 只收 3/4/5 ✓（6 与 2 当场拒 ✓ 不是「看起来也行」✗）",
          all("latent_dims" in text for text in bad), bad)

    #: readout 不许写成 `1×` ✓✗ —— 那会让人按边长算潜帧 ✓。
    text = dit_build.readout("music3")
    check("②-5 `readout` 明写「1 维音频」与「时间 ÷512」✓、不写「1×」✗",
          "1 维音频" in text and "时间 ÷512" in text and "1×" not in text,
          text)


# ── ③ 形状推断：自相矛盾即拒 ───────────────────────────────────────────────

def case_shape_infer() -> None:
    from app.services.engine import dit_music3  # noqa: PLC0415

    shape = _shape()
    check("③-1 自洽形状推得出 ✓（dim/ffn/latent/条件/层数/头数/旋转维/嵌宽 ✓）",
          (shape.dim, shape.ffn_dim, shape.latent_channels, shape.cond_layers,
           shape.cond_hidden, shape.cond_proj, shape.layers, shape.heads,
           shape.rotary_dim, shape.freq_dim)
          == (_D, _FF, _C, _CL, _CH, _CP, 2, 2, _ROT, _FREQ), shape)

    check("③-2 两个宽度是**两件事** ✓✗：入口 16（潜 4×2＋条件 8）／条件总宽 16（2×8）✓",
          (shape.in_dim, shape.context_dim) == (_IN_DIM, _CTX_DIM),
          (shape.in_dim, shape.context_dim))

    def broken(mutate) -> str:
        data = _shapes()
        mutate(data)
        return _error_of(lambda: dit_music3.infer_shape(data))

    cases = {
        "出口宽度与 project_out 不符": (lambda d: d.__setitem__(
            "diffusion_transformer.postprocess_conv.weight", (_C + 1, _D, 3)), "project_out"),
        "入口宽度 ≠ 潜×2＋条件": (lambda d: d.__setitem__(
            "diffusion_transformer.preprocess_conv.weight", (_D, _IN_DIM + 8, 1)), "入口宽度"),
        "层号断号": (lambda d: d.__setitem__(f"{_LAYER}.7.attn.qkv.weight", (3 * _D, _D)),
                 "层号不连续"),
        "旋转维超过头宽": (lambda d: d.__setitem__(
            "diffusion_transformer.transformer.rotary_pos_emb.inv_freq", (_D // 2,)), "旋转"),
        "GLU 不是 2 倍宽": (lambda d: d.__setitem__(
            f"{_LAYER}.0.ff.ff.0.proj.weight", (_FF, _D)), "GLU"),
        #: ⚠️ 只改 ``project_in`` 的 dim=96 ⇒ 先落在**宽度互不自洽**上 ✓（宽度问题先报 ✓），
        #: 「除不尽」那条走下面 ③-5 的显式 heads ✓ —— 两件是不同判据 ✗ 不该混着验 ✓。
        "dim 换成 96 而别处还是 128": (lambda d: d.__setitem__(
            "diffusion_transformer.transformer.project_in.weight", (96, _IN_DIM)), "≠ 96"),
    }
    detail = {why: broken(mutate) for why, (mutate, _) in cases.items()}
    check("③-3 六种自相矛盾**逐条点名**拒绝 ✓（不拿出厂形状硬装 ✓✗）",
          all(detail[why] and needle in detail[why]
              for why, (_, needle) in cases.items()), detail)

    #: 头数**权重里没有** ✓ ⇒ 默认走 ``dim // 64`` ✓；显式给 heads 且除不尽 ⇒ 当场拒 ✓✗
    #: （静默 ``dim // heads`` 会让 q/k 的 ``reshape`` 报「元素数不对」✓ —— 那是远处炸 ✓）。
    text = _error_of(lambda: dit_music3.infer_shape(_shapes(), heads=3))
    check("③-5 显式 heads 除不尽 dim ⇒ 当场点名拒 ✓（不静默整除 ✗）",
          bool(text) and "除不尽" in text, text)

    #: 后缀匹配必须**唯一** ✓：多命中说明形状表混了另一份结构 ✓，随手取第一个
    #: ⇒ 静默按错的那份建 ✓✗。
    dup = _shapes()
    dup["shadow.diffusion_transformer.timestep_features.weight"] = (_FREQ, 1)
    text = _error_of(lambda: dit_music3.infer_shape(dup))
    check("③-4 后缀多命中即拒 ✓（不随手取第一个 ✗）", bool(text), text)


# ── ④ RoPE：split-half 配对 + 正号（与 comfy 融合核逐位一致 ✓） ─────────────

def case_rope() -> None:
    import torch  # noqa: PLC0415

    from app.services.engine import dit_music3  # noqa: PLC0415

    rotary = dit_music3.Music3Rotary(rotary_dim=4)
    with torch.no_grad():
        rotary.inv_freq.copy_(torch.tensor([1.0, 0.5]))
    matrix = rotary.forward_from_seq_len(2, torch.float32)[:, :, 1:2]   # 位置 1 ⇒ θ=[1,.5]

    basis = torch.zeros(1, 1, 1, 4)
    basis[0, 0, 0, 0] = 1.0
    out = dit_music3._apply_rope(basis, matrix)[0, 0, 0].tolist()

    #: ⭐ 指纹在**第 3 个分量** ✓✗：split-half 把 (0,2) 配成一对 ⇒ ``[cos,0,sin,0]`` ✓；
    #: 交错配对会是 ``[cos,sin,0,0]`` ✓ —— 形状全对、数值全不同 ✓。
    check("④-1 是 **split-half** 配对（i ↔ i+d/2 ✓）不是交错 ✓（sin 落第 3 分量 ✓）",
          abs(out[0] - 0.5403023) < 1e-5 and abs(out[1]) < 1e-7
          and abs(out[2] - 0.8414710) < 1e-5 and abs(out[3]) < 1e-7, out)

    #: 符号必须**为正** ✓✗：comfy 环境实测 `comfy_kitchen.apply_rope_split_half` 与
    #: 参考 Python 分支**逐位相等**（maxDiff 0.0 ✓）；拿到 −sin ⇒ 等于 position 取负 ✓
    #: ⇒ 旋律整体反向却不报错 ✓✗。
    check("④-2 旋转方向与融合核一致 ✓（第 3 分量为 **+sin** ✓）", out[2] > 0)

    def reference(x, mat):              # 手写 rotate_half ✓
        cos, sin = mat[..., 0, 0], mat[..., 1, 0]
        head = x.shape[-1] // 2
        first, second = x[..., :head], x[..., head:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    x = torch.randn(2, 3, 5, 4)
    check("④-3 与手写 rotate_half 逐位相等 ✓（maxDiff == 0 ✓）",
          torch.equal(dit_music3._apply_rope(x, matrix), reference(x, matrix)))

    #: 只转**前 rotary 维** ✓、**v 不转** ✓✗（转了 ⇒ 值全错但不报错 ✓）。
    #: ``rotary`` 取自 ``rotation_matrix.shape[-3] * 2`` ✓（= inv_freq 的个数 ×2 ✓）。
    src = (BACKEND_PY / "app/services/engine/dit_music3.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    check("④-4 只给 q/k 的**前一段**加旋转 ✓、v 不转 ✓（读源码判据 ✓）",
          body.count("_apply_rope(q[..., :rotary], rotation_matrix)") == 1
          and body.count("_apply_rope(k[..., :rotary], rotation_matrix)") == 1
          and "_apply_rope(v" not in body
          and "rotary = rotation_matrix.shape[-3] * 2" in body)


# ── ⑤ 时间步翻转 + 输出取负（最贵的两处静默口径 ✓✗） ────────────────────────

def case_timestep_and_sign() -> None:
    import torch  # noqa: PLC0415

    from app.services.engine import dit_music3  # noqa: PLC0415

    torch.manual_seed(20261102)
    model = dit_music3.build(_shape(), dtype=torch.float32).eval()
    latents = torch.randn(1, _C, 13)
    context = torch.randn(1, 4, _CTX_DIM)
    sigma = torch.tensor([0.3])

    with torch.no_grad():
        out = model(latents, sigma, context)
        condition = model.condition_for(context, latents.shape[-1])
        flipped = model.velocity(latents, 1.0 - sigma, condition)
        raw = model.velocity(latents, sigma, condition)

    check("⑤-1 契约常量：``OUTPUT_SIGN == -1`` ✓（参考 ``return -out`` ✓）",
          dit_music3.OUTPUT_SIGN == -1.0, dit_music3.OUTPUT_SIGN)

    check("⑤-2 入口 σ **翻转**成 ``t = 1 − σ`` ✓：``forward(0.3) == −velocity(0.7)`` 逐位 ✓",
          torch.equal(out, -flipped), (out + flipped).abs().max().item())

    #: 两条口径**各自独立可辨** ✓✗：漏翻转 ⇒ 落在 `raw` ✓、漏取负 ⇒ 落在 `flipped` ✓，
    #: 两个错值都与正确值差得远 ✓（不报错 ✓，只是歌糊 ✓）。
    check("⑤-3 漏翻转／漏取负各自落在不同的错值上 ✓（都不是「差不多」✗）",
          not torch.allclose(out, raw) and not torch.allclose(out, flipped)
          and (out - raw).abs().max().item() > 1e-3
          and (out - flipped).abs().max().item() > 1e-3,
          {"vs raw": (out - raw).abs().max().item(),
           "vs flipped": (out - flipped).abs().max().item()})

    two = torch.randn(2, _C, 9)
    with torch.no_grad():
        shape = model(two, torch.tensor([0.5]), torch.randn(2, 4, _CTX_DIM)).shape
    text = _error_of(lambda: model(two, torch.tensor([0.5, 0.4, 0.3]),
                                   torch.randn(2, 4, _CTX_DIM)))
    check("⑤-4 σ 单值可广播 ✓、个数对不上当场拒 ✓（不静默错位 ✗）",
          shape == (2, _C, 9) and "σ" in text, {"shape": shape, "err": text})


# ── ⑥ 前向门禁与条件 pad/crop ───────────────────────────────────────────────

def case_forward_guards() -> None:
    import torch  # noqa: PLC0415

    from app.services.engine import dit_music3  # noqa: PLC0415

    torch.manual_seed(7)
    model = dit_music3.build(_shape(), dtype=torch.float32).eval()

    text = _error_of(lambda: model(torch.randn(1, _C, 2, 2), torch.tensor([0.5]),
                                   torch.randn(1, 4, _CTX_DIM)))
    check("⑥-1 四维 latent 当场拒 ✓ 且话里点名「音频」✓（不让人以为图族抄错 ✓）",
          "音频" in text, text)

    text = _error_of(lambda: model(torch.randn(1, _C, 13), torch.tensor([0.5]),
                                   torch.randn(1, 4, _CTX_DIM - 1)))
    check("⑥-2 条件宽度 ≠ ``cond_layers × cond_hidden`` 当场拒 ✓", bool(text), text)

    #: 4 帧条件 ⇒ latent_length(4)=13 ✓ 刚好；多要就 pad ✓、少要就 crop ✓
    #: （不 pad ⇒ 主干 `cat` 尺寸不符 ⇒ 只在长音频炸 ✓✗）。
    with torch.no_grad():
        exact = model.condition_for(torch.randn(1, 4, _CTX_DIM), 13).shape[-1]
        short = model.condition_for(torch.randn(1, 4, _CTX_DIM), 20).shape[-1]
        long = model.condition_for(torch.randn(1, 4, _CTX_DIM), 5).shape[-1]
    check("⑥-3 条件对齐：13⇒13 ✓、要 20 就 pad ✓、要 5 就 crop ✓",
          (exact, short, long) == (13, 20, 5), (exact, short, long))


# ── ⑦ 长音频滑窗：覆盖 + 分母不为零 ─────────────────────────────────────────

def case_sliding_window() -> None:
    import torch  # noqa: PLC0415

    from app.services.engine import dit_music3  # noqa: PLC0415

    torch.manual_seed(11)
    model = dit_music3.build(_shape(), dtype=torch.float32).eval()
    window = dit_music3.latent_length(dit_music3.MAX_CONDITION_FRAMES)
    hop = dit_music3.latent_length(dit_music3.CONDITION_HOP_FRAMES)

    check("⑦-1 窗口 689 潜帧 ✓、步长 344 ✓（＝ 200/100 条件帧 ✓）",
          (window, hop) == (689, 344), (window, hop))

    #: 长音频必须走滑窗 ✓：900 > 689 ⇒ 3 个窗口 ✓；分母每点 ≥1 ✓
    #: （有洞 ⇒ 该处恒 0 ⇒ 听感是"周期性掉音" ✓✗）。
    length = 900
    with torch.no_grad():
        out = model(torch.randn(1, _C, length), torch.tensor([0.4]),
                    torch.randn(1, 4, _CTX_DIM))
    check("⑦-2 长音频输出形状保持 ✓、无 NaN ✓（滑窗不改长度 ✓）",
          tuple(out.shape) == (1, _C, length) and torch.isfinite(out).all().item(),
          tuple(out.shape))

    #: 覆盖判据：最后一个窗口必须够到末尾 ✓（`start += hop` 而 hop<window ⇒ 重叠 ✓）；
    #: 写成 `start = end`（不重叠）会留缝 ✓✗。
    covered = [0]
    while covered[-1] + window < length:
        covered.append(covered[-1] + hop)
    check("⑦-3 窗口**重叠**覆盖末尾 ✓（步长 344 < 窗口 689 ✓；且不是 1 ⇒ 不慢 344 倍 ✗）",
          covered[-1] + window >= length and 1 < hop < window, {"starts": covered})


# ── ⑧ DAV（音频 VAE）：键名互认 / 少键必拒 / 形状自证 ────────────────────────

_DAV: dict[str, object] = {}


def _dav():
    import torch  # noqa: PLC0415

    from app.services.engine import vae_music3  # noqa: PLC0415

    if "m" not in _DAV:
        model = vae_music3.build_music3_dav(dtype=torch.float32)
        model.eval()
        _DAV["m"] = model
    return _DAV["m"]


def _dav_shapes() -> dict[str, tuple[int, ...]]:
    """``state_dict`` 那一套键名（**新式** parametrize ✓）× 形状 ✓。"""
    return {name: tuple(t.shape) for name, t in _dav().state_dict().items()}


def _legacy_shapes() -> dict[str, tuple[int, ...]]:
    """检查点那一套键名（**旧式** ``weight_g`` / ``weight_v`` ✓）× 形状 ✓。"""
    from app.services.engine import vae_music3  # noqa: PLC0415

    return {vae_music3.as_legacy_name(name): tuple(t.shape)
            for name, t in _dav().state_dict().items()}


def case_dav_keys() -> None:
    from app.services.engine import vae_music3  # noqa: PLC0415

    names = vae_music3.dav_key_names()
    check("⑧-1 真权重 **121** 键 ✓、新旧两种写法点数一致 ✓",
          len(names) == 121 == vae_music3.MUSIC3_DAV_FACTS["keyCount"]
          and len(vae_music3.dav_key_names(legacy=False)) == 121, len(names))

    check("⑧-2 `legacy=False` 与模块 ``state_dict`` **逐字相同** ✓（不是「能改写」就算 ✓）",
          list(vae_music3.dav_key_names(legacy=False)) == sorted(_dav().state_dict()))

    check("⑧-3 两种键名归一后**同一集合** ✓（weight_g/v ↔ parametrizations ✓、两向都有 ✓）",
          {vae_music3._canonical_key(n) for n in names} == set(_dav().state_dict())
          and {vae_music3.as_legacy_name(n) for n in _dav().state_dict()} == set(names)
          and vae_music3._canonical_key("up.0.weight_g")
          == "up.0.parametrizations.weight.original0"
          and vae_music3.as_legacy_name("up.0.parametrizations.weight.original0")
          == "up.0.weight_g")

    check("⑧-4 自证形状表零问题 ✓（两种键名写法都收 ✓）",
          vae_music3.shape_problems(_legacy_shapes()) == []
          and vae_music3.shape_problems(_dav_shapes()) == [],
          (vae_music3.shape_problems(_legacy_shapes())[:3],
           vae_music3.shape_problems(_dav_shapes())[:3]))

    broken = _dav_shapes()
    key = "dec_in_proj.weight"
    if key in broken:
        broken[key] = (broken[key][0], broken[key][1] + 1, broken[key][2])
    problems = vae_music3.shape_problems(broken)
    check("⑧-5 入宽改一个数 ⇒ 点名报 ✗（装载前核 ✓ 不等反量化炸 ✓）",
          problems == ["dec_in_proj 输入通道 65 ≠ 64 ✗（128 潜通道要折半成 2 × 64 ✓）"]
          or (len(problems) == 1 and "dec_in_proj" in problems[0] and "65" in problems[0]),
          problems[:2])

    #: 少键必须拒 ✓✗（``strict=False`` 会让该层留 ``ones`` 初始值 ⇒ 尺度全错 ⇒ 噪音 ✓）。
    payload = dict(_dav().state_dict())
    payload.pop(next(iter(payload)))
    text = _error_of(lambda: vae_music3.load_music3_dav_state_dict(_dav(), payload))
    check("⑧-6 少一个键装载当场拒 ✓（不许 ``strict=False`` 静默放过 ✗）", bool(text), text)

    #: ⭐ ``Snake1d.alpha`` 必须是**确定初值** ✓✗：写 ``torch.empty`` 会留未初始化内存 ✓
    #: （本机实测抽到 ``alpha ≈ 3.5e7`` ✓）⇒ 不装权重的壳会解出满是高频抖动的假声音 ✓。
    import torch  # noqa: PLC0415

    shells = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        shells.append([p.detach().clone() for n, p in
                       vae_music3.build_music3_dav(dtype=torch.float32).named_parameters()
                       if n.endswith(".alpha")])
    check("⑧-7 随机壳没有未初始化内存：``alpha`` 初值全 1 ✓、两次构建逐位相同 ✓",
          len(shells[0]) == 29
          and all(float(a.min()) == 1.0 and float(a.max()) == 1.0 for a in shells[0])
          and all(torch.equal(a, b) for a, b in zip(shells[0], shells[1], strict=True)),
          len(shells[0]))


# ── ⑨ DAV 解码：128 = 2 × 64 行主序折半 ────────────────────────────────────

def case_dav_decode() -> None:
    import torch  # noqa: PLC0415

    from app.services.engine import vae_music3  # noqa: PLC0415

    model = _dav()
    torch.manual_seed(3)
    latents = torch.randn(1, 128, 2)
    with torch.no_grad():
        audio = vae_music3.decode_latents(model, latents)

    check("⑨-1 ``(1,128,2)`` ⇒ ``(1,2,1024)`` ✓（2 声道 ✓ ×512 上采样 ✓）",
          tuple(audio.shape) == (1, 2, 1024), tuple(audio.shape))
    check("⑨-2 取值落在 (−1,1) ✓（末端 tanh ✓）且无 NaN ✓",
          torch.isfinite(audio).all().item() and audio.abs().max().item() <= 1.0,
          audio.abs().max().item())

    #: ⭐ 折半顺序的**行为判据** ✓✗ —— 判「**另一半逐位不变**」✓，不判「清零那半恒为常数」✗：
    #: 后者在本机实测**不成立** ✓（转置卷积对常数输入也只在**内部**给常数值 ✓，
    #: 而 ``T=2`` 经过 4 段 stride（8/8/4/2 ✓）后整段都是**边界** ✓ ⇒ 系数逐点不同 ✓，
    #: 零输入照样解出一条时变波形 ✓）—— 拿它当判据会误报 ✓。
    #: 真正的分界只有一条：128 通道**行主序折半**成两个**独立样本** ✓ ⇒
    #: 清零后 64 ⇒ 右声道**逐位不变**、左声道变 ✓；清零前 64 ⇒ 左声道**逐位不变**、右声道变 ✓。
    #: 写成交错声道 / 整 128 通道一锅卷积 ⇒ 两侧都会变 ✓✗（形状全对、只是左右串了 ✓）。
    tail_cut = latents.clone()
    tail_cut[:, 64:] = 0.0
    head_cut = latents.clone()
    head_cut[:, :64] = 0.0
    with torch.no_grad():
        tail_muted = vae_music3.decode_latents(model, tail_cut)
        head_muted = vae_music3.decode_latents(model, head_cut)

    left_delta = float((tail_muted[0, 0] - audio[0, 0]).abs().max())
    right_delta = float((head_muted[0, 1] - audio[0, 1]).abs().max())
    check("⑨-3 后 64 通道清零 ⇒ 左声道**逐位不变** ✓、右声道变 ✓（通道 0..63 → 左 ✓）",
          torch.equal(tail_muted[0, 0], audio[0, 0])
          and not torch.equal(tail_muted[0, 1], audio[0, 1]),
          {"leftDelta": left_delta, "rightStd": float(tail_muted[0, 1].std())})

    check("⑨-4 前 64 通道清零 ⇒ 右声道**逐位不变** ✓、左声道变 ✓（通道 64..127 → 右 ✓）",
          torch.equal(head_muted[0, 1], audio[0, 1])
          and not torch.equal(head_muted[0, 0], audio[0, 0]),
          {"rightDelta": right_delta, "leftStd": float(head_muted[0, 0].std())})


# ── ⑩ 装配表接线 ───────────────────────────────────────────────────────────

def case_build_table() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from app.services.engine import dit_build, dit_music3  # noqa: PLC0415

    spec = dit_build.BUILDERS["music3"]
    check("⑩-1 表项：`build_kwargs` 收窄成 ``(dtype,)`` ✓（音乐族没有 patchify ✓✗ 白传不算错但读不出）",
          spec.build_kwargs == ("dtype",) and spec.meta is True,
          spec.build_kwargs)
    check("⑩-2 表项：``in_channels`` 取**潜通道**（4 ✓）而不是入口宽（16 ✓✗）",
          spec.shortcuts == (("in_channels", "latent_channels"), ("text_dim", "context_dim")),
          spec.shortcuts)
    check("⑩-3 表项：``vae_scale`` 恒 1 ✓ + 44100/512 两个事实 ✓",
          dict(spec.facts).get("vae_scale") == 1
          and dict(spec.facts).get("latentSampleRate") == 44100
          and dict(spec.facts).get("latentHop") == 512, dict(spec.facts))

    #: 走**公共流程**建一次 ✓（不是直接调 `build()` 绕开表 ✓）——
    #: 顶层 `config` 必须把两个宽度分得开 ✓✗（混用 ⇒ 上游按 16 造条件 ✓ 静默错位 ✓）。
    shapes = _shapes()
    info = SimpleNamespace(tensors={k: SimpleNamespace(shape=tuple(v))
                                    for k, v in shapes.items()},
                           total_tensor_bytes=0, problems=())
    module, config = dit_build.build("music3", info, dtype=torch.float32)
    check("⑩-4 公共流程建得出 ✓：``in_channels``=4 ✓、``text_dim``=16（条件总宽 ✓）✓",
          config["in_channels"] == _C and config["text_dim"] == _CTX_DIM
          and config["family"] == "music3", {k: config.get(k) for k in
                                             ("in_channels", "text_dim", "family")})
    #: ⚠️ 公共流程走 ``materialize_meta`` ✓ ⇒ 权重是 ``torch.empty`` **未初始化** ✓
    #: （那边是**故意**的 ✓：值一律由装载填 ✓）。直接前向会随机摸到 NaN/inf ✓✗ ——
    #: 判据要验的是「接线跑得动」✓，所以先拿同构随机初始化把空内存填掉 ✓。
    module.load_state_dict(dit_music3.build(_shape(), dtype=torch.float32).state_dict())
    with torch.no_grad():
        out = module(torch.randn(1, _C, 13), torch.tensor([0.6]),
                     torch.randn(1, 4, _CTX_DIM))
    check("⑩-5 建出来的模块前向跑得动 ✓（形状保持、无 NaN ✓）",
          tuple(out.shape) == (1, _C, 13) and torch.isfinite(out).all().item(),
          tuple(out.shape))


def main() -> int:
    for case in (case_rates, case_contract, case_shape_infer, case_rope,
                 case_timestep_and_sign, case_forward_guards, case_sliding_window,
                 case_dav_keys, case_dav_decode, case_build_table):
        try:
            case()
        except Exception as err:       # noqa: BLE001 —— 一条炸掉不该吞掉后面的 ✓
            import traceback  # noqa: PLC0415

            traceback.print_exc()
            check(f"{case.__name__} 整体跑通 ✓", False, err)

    failed = [name for ok, name, _ in _RESULTS if not ok]
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    if failed:
        print("未通过：")
        for name in failed:
            print(f"  ✗ {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
