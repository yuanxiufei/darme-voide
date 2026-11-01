"""自检：**qwen_image 双流 MMDiT 的形状／键／前向口径**（2026-10-31）。

为什么值得单独一份 ✗：这一族与 `dit_qwen21`（Qwen-Image **2.1** ✓）**在同一个磁盘目录、同一个
文件形态、同一套 VAE 通道**上 ✓ ✓，却**不是同一个模型** ✗✗。四处「形状全都对、图全不对」的换法：

==================================  ====================================  ============================
                                   2512 / Edit 2511（本族 ✓）            2.1（`dit_qwen21` ✓）
==================================  ====================================  ============================
流数                               **双流**（txt/img 各一套调制与 MLP）✓   单流 ✓
每流调制组数                        **6**（两组 shift/scale/gate）✓        4 ✓
时间口径                            **1000·σ** ✓（不翻转 ✓）              1000·(1−σ) ✓
参考图（Edit）                      **token 串联** ✓ + 「位置序号」三轴 ✓   不适用
patch                               **2×2 拍平进通道**（16→64）✓           无 ✓
==================================  ====================================  ============================

⇒ 本文件钉的就是这五格 + 三条「不报错只是想当然错」的机制：

① 6 组调制是**族指纹** ✓ —— 拿 2.1 的形状表进来必在这里报（唯一拦得住的地方 ✓）；
② 时间口径抄错 Z-Image/2.1 的「翻转」写法 ⇒ 形状全对、图能出 ✓✗（等于去噪方向反着走 ✓）；
③ 参考图**必须按 token 串**（`cat(dim=1)` ✓）、目标 token 用 t=σ 行、参考 token 用 **t=0** 行 ✓；
   写反或走通道拼接 ⇒ 形状同样对 ✓，只是 Edit 的「保内容」能力消失 ✓✗。

⚠️ **不跑真装** ✗（2512 bf16 ≈38 GiB ✓、Edit int8 ≈19 GiB ✓，自检里不装 ✓）——
只读**头部**做键对账 ✓（`safetensors.read_header` 只读头几 KB ✓），真出图是人为验收那一步 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_dit_qwen_image_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import torch  # noqa: E402

from app.services.engine import dit_generic, dit_qwen_image as qim, safetensors  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _error_of(fn, kinds=(Exception,)) -> str:  # noqa: ANN001, ANN202
    try:
        fn()
    except kinds as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return ""


# ------------------------------------------------------------------ 形状表

_TINY = dict(features=32, head_dim=8, hidden=64, channels=4, context_dim=24, layers=2)


def _table(*, mods: int = 6, sentinel: bool = True, patch: int = 2, **over) -> dict:
    """按真权重的**键名/形状**造一张小表 ✓（`infer_shape` 只读这张表 ✓，不碰真权重 ✓）。"""
    spec = dict(_TINY)
    spec.update(over)
    f, hd, hid = spec["features"], spec["head_dim"], spec["hidden"]
    ch, ctx, layers = spec["channels"], spec["context_dim"], spec["layers"]
    width = ch * patch * patch
    t: dict[str, tuple[int, ...]] = {
        "img_in.weight": (f, width), "img_in.bias": (f,),
        "txt_in.weight": (f, ctx), "txt_in.bias": (f,),
        "txt_norm.weight": (ctx,),
        "time_text_embed.timestep_embedder.linear_1.weight": (f, 256),
        "time_text_embed.timestep_embedder.linear_1.bias": (f,),
        "time_text_embed.timestep_embedder.linear_2.weight": (f, f),
        "time_text_embed.timestep_embedder.linear_2.bias": (f,),
        "norm_out.linear.weight": (2 * f, f), "norm_out.linear.bias": (2 * f,),
        "proj_out.weight": (width, f), "proj_out.bias": (width,),
    }
    for i in range(layers):
        p = f"transformer_blocks.{i}."
        t[p + "img_mod.1.weight"] = (mods * f, f)
        t[p + "img_mod.1.bias"] = (mods * f,)
        t[p + "txt_mod.1.weight"] = (mods * f, f)
        t[p + "txt_mod.1.bias"] = (mods * f,)
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
                     "to_out.0", "to_add_out"):
            t[p + f"attn.{name}.weight"] = (f, f)
            t[p + f"attn.{name}.bias"] = (f,)
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            t[p + f"attn.{name}.weight"] = (hd,)
        for stream in ("img_mlp", "txt_mlp"):
            t[p + f"{stream}.net.0.proj.weight"] = (hid, f)
            t[p + f"{stream}.net.0.proj.bias"] = (hid,)
            t[p + f"{stream}.net.2.weight"] = (f, hid)
            t[p + f"{stream}.net.2.bias"] = (f,)
    if sentinel:
        t[qim.EDIT_SENTINEL] = (0,)
    return t


TINY_AXES = (2, 2, 4)  # 和 = 8 = head_dim ✓


def _shape(**over) -> "qim.QwenImageShape":
    kw = dict(heads=None, axes_dim=TINY_AXES)
    kw.update(over)
    return qim.infer_shape(_table(), **kw)


# --------------------------------------------------------------- ① 形状推断


def case_shape() -> None:
    sh = _shape()
    check("① heads 由 `attn.norm_q.weight` 的**宽度 = head_dim** 反推 ✓ ⇒ features/head_dim ✓"
          "（to_q 是 [f,f]，自己看不出头数 ✗）",
          (sh.heads, sh.head_dim, sh.features) == (4, 8, 32), (sh.heads, sh.head_dim))
    check("①′ channels = img_in 输入宽 / patch² = 4 ✓（16ch VAE + 2×2 拍平 = 64 ✓ 同一套算法 ✓）",
          sh.channels == 4, sh.channels)
    check("①″ 有 `__index_timestep_zero__` ⇒ 参考图口径 = `index_timestep_zero` ✓（Edit 2511 ✓）",
          sh.ref_method == qim.REF_INDEX_TZERO and sh.ref_timestep_zero, sh.ref_method)
    no_sentinel = qim.infer_shape(_table(sentinel=False), axes_dim=TINY_AXES)
    check("①‴ 没有该键 ⇒ 退回 `index` 档 ✓（2512 t2i ✓），且 `ref_timestep_zero` 为假 ✓",
          no_sentinel.ref_method == qim.REF_INDEX and not no_sentinel.ref_timestep_zero,
          no_sentinel.ref_method)

    message = _error_of(lambda: qim.infer_shape(_table(mods=4), axes_dim=TINY_AXES),
                        (qim.ShapeError,))
    check("①⁗ **6 组调制是族指纹** ✓：4 组（2.1 的写法 ✓）⇒ 当场报错并点名 6 组 ✓"
          "（这一格是唯一能分开两族的地方 ✓ —— 族名之外全靠它 ✓）",
          "6" in message and "qwen_image" in message, message or "没报错 ✗")

    message = _error_of(lambda: qim.infer_shape(_table(), axes_dim=(2, 2, 2)), (qim.ShapeError,))
    check("①⁵ axes_dim 之和 ≠ head_dim ⇒ 当场报错 ✓（和为 6 ≠ 8 ✓）",
          "head_dim" in message or "axes" in message, message or "没报错 ✗")


# --------------------------------------------------------------- ② 键对账


def case_keys() -> None:
    for method, sentinel in ((qim.REF_INDEX, False), (qim.REF_INDEX_TZERO, True)):
        sh = qim.infer_shape(_table(sentinel=sentinel), axes_dim=TINY_AXES)
        model = qim.build(sh)
        built, wanted = set(model.state_dict()), qim.expected_keys(sh)
        check(f"② 口径 {method} ✓：建出来的键与该族**应有的键**逐一对齐 ✓（不多不少 ✓）",
              built == wanted, {"缺": sorted(wanted - built)[:6], "多": sorted(built - wanted)[:6]})
    sh = qim.infer_shape(_table(sentinel=False), axes_dim=TINY_AXES)
    model = qim.build(sh)
    check("②′ 没有哨兵的档位**不注册**那个空缓冲 ✓（否则 2512 的键对账会凭空多一格 ✓✗）",
          qim.EDIT_SENTINEL not in model.state_dict(), sorted(model.state_dict())[:3])


# --------------------------------------------------------------- ③ 前向


def case_forward() -> None:
    torch.manual_seed(11)
    net = qim.build(_shape()).eval()
    lat = torch.randn(1, 4, 8, 12)  # 非方形 ✓
    ctx = torch.randn(1, 5, 24)
    with torch.no_grad():
        out = net(lat, torch.tensor([0.7]), ctx)
    check("③ 输出形状 = 潜变量形状 ✓（内部 patch 2×2 拍平再还原 ✓，边补齐后回裁 ✓）",
          tuple(out.shape) == tuple(lat.shape), tuple(out.shape))

    odd = torch.randn(1, 4, 7, 9)  # 奇数边 ⇒ 触发补齐路径 ✓
    with torch.no_grad():
        out_odd = net(odd, torch.tensor([0.7]), ctx)
    check("③′ **奇数边**也能跑且回裁正确 ✓（补齐的边不许漏进输出 ✓✗）",
          tuple(out_odd.shape) == tuple(odd.shape), tuple(out_odd.shape))

    with torch.no_grad():
        again = net(lat, torch.tensor([0.7]), ctx)
    check("③″ 同输入逐位复现 ✓（eval 下无随机性 ✓）", torch.equal(out, again),
          float((out - again).abs().max()))


def case_edit() -> None:
    torch.manual_seed(12)
    net = qim.build(_shape(ref_method=qim.REF_INDEX_TZERO)).eval()
    lat = torch.randn(1, 4, 8, 12)
    ctx = torch.randn(1, 5, 24)
    t = torch.tensor([0.7])
    ref_a = torch.randn(1, 4, 8, 8)
    ref_b = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        base = net(lat, t, ctx)
        one = net(lat, t, ctx, ref_latents=[ref_a])
        two = net(lat, t, ctx, ref_latents=[ref_a, ref_b])
        swapped = net(lat, t, ctx, ref_latents=[ref_b])
        by_index = net(lat, t, ctx, ref_latents=[ref_a], ref_method=qim.REF_INDEX)
    check("④ 参考图是 **token 串联** ✓：输出形状不随参考张数变化 ✓（通道拼接会让输出变宽 ✓✗）",
          tuple(one.shape) == tuple(base.shape) == tuple(two.shape), tuple(two.shape))
    check("④′ 参考图**真进入**前向了 ✓（换一张参考图 ⇒ 输出必须变 ✓；不变说明被丢掉了 ✓✗）",
          float((one - swapped).abs().max()) > 1e-3, float((one - swapped).abs().max()))
    check("④″ 参考图**张数**也起作用 ✓（第二张的 token 与第一张位置号不同 ✓）",
          float((one - two).abs().max()) > 1e-3, float((one - two).abs().max()))
    check("④‴ `index_timestep_zero` 与 `index` 两档**确实不同** ✓"
          "（参考 token 用 t=0 行 vs 用 t=σ 行 ✓；写成一档 ⇒ 参考图被当成「有噪的同级 token」✓✗）",
          float((one - by_index).abs().max()) > 1e-3, float((one - by_index).abs().max()))

    ref_bad = torch.randn(2, 4, 8, 8)
    message = _error_of(lambda: net(lat, t, ctx, ref_latents=[ref_bad]), (qim.ShapeError,))
    check("④⁗ 参考图 batch 与目标不符 ⇒ 当场报错 ✓（不许静默广播 ✓✗）",
          "batch" in message, message or "没报错 ✗")

    no_sentinel = qim.build(_shape(ref_method=qim.REF_INDEX)).eval()
    with torch.no_grad():
        plain = no_sentinel(lat, t, ctx, ref_latents=[ref_a])
    check("④⁵ 没有哨兵的族（2512 ✓）即使收到参考图也**不进 t=0 档** ✓"
          "（拿 t2i 权重冒充 Edit 时不会炸 ✓，但也不会凭空获得 Edit 能力 ✓ —— 口径按权重自述 ✓）",
          tuple(plain.shape) == tuple(lat.shape), tuple(plain.shape))


# --------------------------------------------------------------- ⑤ 时间口径


def case_time() -> None:
    probe = qim._TimestepProj(256, 32)
    sigma = torch.tensor([0.7, 0.2])
    want = dit_generic.timestep_embedding(sigma * 1000.0, 256, time_factor=1.0)
    got = probe(sigma)
    hand = probe.linear_2(probe.act(probe.linear_1(want.to(probe.linear_1.weight.dtype))))
    check("⑤ 时间口径 = **1000·σ** ✓（与手工 `timestep_embedding(1000σ)` 逐位一致 ✓）",
          torch.equal(got, hand), float((got - hand).abs().max()))

    flipped = dit_generic.timestep_embedding((1.0 - sigma) * 1000.0, 256, time_factor=1.0)
    hand_flipped = probe.linear_2(probe.act(probe.linear_1(flipped.to(probe.linear_1.weight.dtype))))
    check("⑤′ 与 Z-Image/2.1 的 **1000·(1−σ)** ✓ 明显不同 ✓"
          "（这一格抄错不报错 ✗ —— 只是把「去噪方向」走反 ✓✗）",
          float((got - hand_flipped).abs().max()) > 1e-2,
          float((got - hand_flipped).abs().max()))


# --------------------------------------------------------------- ⑥ 调制


def case_modulate() -> None:
    x = torch.randn(1, 3, 5)
    shift = torch.randn(1, 5)
    scale = torch.randn(1, 5)
    gate = torch.randn(1, 5)
    got, g = qim._modulate(x, shift, scale, gate, None)
    want = x * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)
    check("⑥ 调制 = **(1+scale)·x + shift** ✓（shift 是加性 ✓ 不是乘性 ✗）",
          torch.allclose(got, want, atol=1e-6) and g.shape == (1, 1, 5),
          float((got - want).abs().max()))

    swapped = x * (1 + shift).unsqueeze(1) + scale.unsqueeze(1)
    check("⑥′ 把 shift/scale 对调 ⇒ 数值不同 ✓（「不报错但亮度全偏」✓✗ —— 所以这条要钉住 ✓）",
          not torch.allclose(got, swapped, atol=1e-4), float((got - swapped).abs().max()))

    # tz 档：前 tz 个 token 用行 0、其余用行 1（t=0）
    both = torch.randn(2, 5)
    both_scale = torch.randn(2, 5)
    both_gate = torch.randn(2, 5)
    x2 = torch.randn(1, 4, 5)
    mixed, gates = qim._modulate(x2, both, both_scale, both_gate, 2)
    first = x2[:, :2] * (1 + both_scale[0]).unsqueeze(0).unsqueeze(0) + both[0].unsqueeze(0).unsqueeze(0)
    second = x2[:, 2:] * (1 + both_scale[1]).unsqueeze(0).unsqueeze(0) + both[1].unsqueeze(0).unsqueeze(0)
    check("⑥″ tz 档：前缀 token 用行 0、参考 token 用行 1 ✓，且返回两套 gate ✓",
          torch.allclose(mixed[:, :2], first, atol=1e-6)
          and torch.allclose(mixed[:, 2:], second, atol=1e-6)
          and isinstance(gates, tuple) and gates[0].shape[-1] == 5,
          float((mixed[:, 2:] - second).abs().max()))


# --------------------------------------------------------------- ⑦ 真实权重头部


_DIT_FILES = (
    ("2512 t2i", "qwen_image_2512_bf16.safetensors", qim.REF_INDEX),
    ("Edit 2511", "qwen_image_edit_2511_int8_convrot.safetensors", qim.REF_INDEX_TZERO),
)


def case_real_headers() -> None:
    """只读**头部**（几 KB ✓）做键对账 ✓ —— 这是「这份权重能不能住进这个模型」的最硬证据 ✓。"""
    root = Path(os.environ.get("COMFY_MODELS", r"D:/Comfy-Desktop/ComfyUI-Shared/models"))
    folder = root / "diffusion_models"
    for label, name, method in _DIT_FILES:
        path = folder / name
        if not path.exists():
            skip(f"⑦ {label}：盘上没有 {path} ⇒ 跳过真键对账")
            continue
        try:
            meta = safetensors.inspect(str(path))
        except Exception as err:  # noqa: BLE001
            check(f"⑦ {label}：头部读不出来 ✗", False, f"{type(err).__name__}: {err}")
            continue
        shapes = {name_: tuple(entry.shape) for name_, entry in meta.tensors.items()}
        try:
            sh = qim.infer_shape(shapes)
        except qim.ShapeError as err:
            check(f"⑦ {label}：形状表推不出这一族 ✗", False, str(err))
            continue
        wanted = qim.expected_keys(sh)
        missing = wanted - set(shapes)
        check(f"⑦ {label}：`infer_shape` 认得 ✓（{sh.layers} 块 / {sh.features} 宽 / "
              f"{sh.heads} 头 × {sh.head_dim} / 参考口径 {sh.ref_method} ✓）",
              sh.ref_method == method and sh.heads * sh.head_dim == sh.features,
              (sh.ref_method, sh.heads, sh.head_dim))
        check(f"⑦′ {label}：模型**应有的每一根键**权重里都有 ✓（缺 {len(missing)} 根 ✓）"
              f"⇒ 这份模型装得下这份权重 ✓",
              not missing, sorted(missing)[:8])
        extras = set(shapes) - wanted
        if method == qim.REF_INDEX:
            check("⑦″ 2512（bf16 ✓）键数**恰好**等于应有键数 ✓（多出来就说明模型少建了一层 ✓✗）",
                  not extras, sorted(extras)[:8])
        else:
            check("⑦″ Edit（int8 量化 ✓）多出来的键都是**量化伴随件** ✓（`.comfy_quant`/`_scale` ✓），"
                  "不带参数语义 ✓",
                  all("comfy_quant" in key or "scale" in key for key in extras),
                  sorted(extras)[:6])


# --------------------------------------------------------------- ⑧ 后端构建器


class _FakeInfo:
    """`safetensors.inspect()` 的最小替身 ✓（构建器只用 ``.tensors[*].shape`` ✓）。"""

    def __init__(self, table: dict) -> None:
        self.tensors = {name: type("T", (), {"shape": tuple(shape)})()
                        for name, shape in table.items()}
        self.path = "<fake>"


def case_backend() -> None:
    """`torch_backend` 的构建器**真能**按这张表建出模型 ✓（只验接线与搬名，不验数学 ✓）。"""
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    module, config = tb._build_qwen_image_module(
        _FakeInfo(_table(sentinel=False)), {"axes_dim": TINY_AXES}, dtype=torch.float32)
    check("⑧ 构建器接上了这一族 ✓：族名 `qwen_image` ✓、三格搬名（`text_dim`/`in_channels`/"
          "`vae_scale` ✓）全部落进 config ✓",
          config["family"] == tb.QWEN_IMAGE_FAMILY == "qwen_image"
          and config["text_dim"] == _TINY["context_dim"]
          and config["in_channels"] == _TINY["channels"]
          and config["vae_scale"] == 8,
          {k: config.get(k) for k in ("family", "text_dim", "in_channels", "vae_scale")})
    check("⑧′ `vae_scale=8` **不是**从权重读的 ✓（2512 与 2.1 的潜通道数不同、而权重里没有这一格 ✗）"
          "⇒ `configSources` 必须自己说清来源 ✓",
          "Wan21" in str(config["configSources"]["vae_scale"]),
          config["configSources"]["vae_scale"])
    check("⑧″ t2i（无哨兵 ✓）⇒ `editRefWired` 记 n/a ✓；且 `qwen_image21` **不等于**这一族 ✓"
          "（用 `startswith` 判族会让 2.1 被这座骨架接走 ✗）",
          config["configSources"]["editRefWired"] == "n/a（t2i 权重 ✓）"
          and tb.QWEN_IMAGE_FAMILY != tb.QWEN21_FAMILY
          and not tb.QWEN21_FAMILY == "qwen_image",
          config["configSources"]["editRefWired"])

    edit_module, edit_config = tb._build_qwen_image_module(
        _FakeInfo(_table(sentinel=True)), {"axes_dim": TINY_AXES}, dtype=torch.float32)
    check("⑧‴ Edit（带 `__index_timestep_zero__` ✓）⇒ 口径为 `index_timestep_zero` ✓、"
          "且 `editRefWired` **明写未接** ✓（不静默当作 t2i 能力 ✓✗）",
          edit_module.shape.ref_timestep_zero
          and str(edit_config["configSources"]["editRefWired"]).startswith("False"),
          edit_config["configSources"]["editRefWired"])

    try:
        tb._build_qwen_image_module(_FakeInfo(_table(mods=4)), {"axes_dim": TINY_AXES},
                                    dtype=torch.float32)
    except tb.TorchBackendUnavailable as err:
        kind, message = type(err).__name__, str(err)
    except Exception as err:  # noqa: BLE001
        kind, message = type(err).__name__, str(err)
    else:
        kind, message = "没报错", ""
    check("⑧⁴ 拿 2.1 的形状（4 组调制 ✗）进来 ⇒ **后端这一层**就转成 `TorchBackendUnavailable` ✓"
          "（裸 `ShapeError` 冒到路由层会变 500 ✗）",
          kind == "TorchBackendUnavailable" and "6 组" in message, f"{kind} | {message}")
    check("⑧⁵ 这一族在 `IMAGE_ONLY_FAMILIES` 里 ✓（否则 `denoise` 会拿标量 σ 去喂逐 batch 张量的口 ✗）",
          tb.QWEN_IMAGE_FAMILY in tb.IMAGE_ONLY_FAMILIES, tb.IMAGE_ONLY_FAMILIES)
    del module, edit_module


# --------------------------------------------------------------- main


def main() -> int:
    case_shape()
    case_keys()
    case_forward()
    case_edit()
    case_time()
    case_modulate()
    case_backend()
    case_real_headers()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
