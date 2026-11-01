"""自检：**H3 真视频 VAE** ✓ —— 权重落地了，能不能**真接上**（2026-10-29 ✓）。

前面几轮把"真权重"这件事拆成了三段，这一段钉的是**第三段（解码侧）** ✓：

1. `vae_h3.py` 的实现与权重键对齐 ✓（562 个键逐个对上 —— 本轮**不重验**装载，重验的是**接线**）；
2. `TorchBackend.attach_vae()` **认得出**这一族 ✓（报告里 `kind="h3-real"` ✓，`decode()` 的
   ``synthetic`` 随之**真的变 False** ✓ —— 不是还写死 True ✗）；
3. `runtime` 那条装配路上**有口子** ✓（``real_video_vae`` ✓）且**发现不到权重就当场拒** ✗，
   **绝不悄悄回落**成参考 VAE ✗✗。

分两档跑 ✓：

* **常跑档**（零 torch 也行 ✓，秒级 ✓）：架构事实、时间几何、配置守卫、**拒绝语义**；
* **真权重档**（``DARME_H3_VAE_LIVE=1`` ✓ 且盘上找得到权重 ✓）：真装载 → 编码 → 解码，
  逐条判 **形状 / 像素值域 / 还原度** ✓；没开就**显式 SKIP** ✓（不装作跑过 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_vae_h3_test.py

    # 再带真权重那一档（约 40 秒 + 5 GiB 权重）
    $env:DARME_H3_VAE_LIVE=1; ./.venv/Scripts/python.exe tests/engine_vae_h3_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import runtime as runtime_mod  # noqa: E402
from app.services.engine import vae_h3 as vae_h3_mod  # noqa: E402
from app.services.engine import vae as vae_mod  # noqa: E402
from app.services.engine.torch_backend import TorchBackendUnavailable  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ① 架构事实：与 H3 口径**同源** ✓（一个倍率、三个数，谁都别自己再写一份 ✗）
# ─────────────────────────────────────────────────────────────────────────────
def case_facts() -> None:
    facts = vae_h3_mod.H3_VIDEO_VAE_ARCH_FACTS
    h3 = vae_mod.H3_VIDEO_VAE_FACTS
    check("事实: spatialScale=16 且与 `vae.H3_VIDEO_VAE_FACTS['vaeScale']` **逐值相同**",
          int(facts["spatialScale"]) == 16
          and int(facts["spatialScale"]) == int(h3["vaeScale"]),
          (facts["spatialScale"], h3.get("vaeScale")))
    check("事实: temporalScale=4 / latentChannels=24（H3 双流主干的潜形态）",
          int(facts["temporalScale"]) == 4 and int(facts["latentChannels"]) == 24,
          (facts["temporalScale"], facts["latentChannels"]))
    # ⚠️ 配置里的三个属性**必须**与事实表一致 ✗ —— 两边各写一份就是漂移的来路 ✓✗
    cfg = vae_h3_mod.H3VideoVAEConfig()
    check("事实: 配置属性与事实表**同源**（spatial_scale / temporal_scale / latent_channels）",
          (cfg.spatial_scale, cfg.temporal_scale, cfg.latent_channels)
          == (int(facts["spatialScale"]), int(facts["temporalScale"]),
              int(facts["latentChannels"])),
          (cfg.spatial_scale, cfg.temporal_scale, cfg.latent_channels))


# ─────────────────────────────────────────────────────────────────────────────
# ② 时间几何：17k+5 帧 ⇔ 5k+2 潜帧 ✓（编码侧与解码侧**共用一套算式** ✓）
# ─────────────────────────────────────────────────────────────────────────────
def case_geometry() -> None:
    lat = vae_h3_mod.latents_for_frames
    fra = vae_h3_mod.frames_for_latents
    check("几何: 单帧是特例（1 帧 ⇒ 1 潜格 ⇒ 1 帧）",
          lat(1) == 1 and fra(1) == 1, (lat(1), fra(1)))
    exact = [(k, 17 * k + 5) for k in range(0, 8)]
    ok = all(fra(lat(frames)) == frames for _k, frames in exact)
    check("几何: `5+17k` 帧上**严格互逆**（这条不成立就说明两个方向各写了一套算式）",
          ok, [(frames, lat(frames), fra(lat(frames))) for _k, frames in exact])
    # ⚠️ 其余长度**不保证**互逆 ✗（编码末尾那 3 个潜格是**整段丢**的 ✓：13 帧编成 2 潜格 ⇒ 解回 5 帧 ✓）——
    #    这条**不许**写成"只许多" ✗：口径本来就是"上游按 5+17k 请求" ✓，其余长度由上层裁剪 ✓。
    short = [(frames, lat(frames), fra(lat(frames))) for frames in (2, 13, 18, 40, 97)]
    check("几何: 非 `5+17k` 长度上**不静默编成 0 / 负数**（`ceil` 与 `-3` 的边界别写反）",
          all(0 <= got <= frames + 17 for frames, _z, got in short), short)
    check("几何: 潜帧数**单调不减**（帧数涨 ⇒ 潜格不缩）",
          all(lat(n) <= lat(n + 1) for n in range(1, 200)),
          [lat(n) for n in (1, 17, 18, 34, 35, 100)])
    plan = vae_h3_mod.decode_temporal_plan(lat(39))
    check("几何: 解码分段计划与帧数换算**对得上**（39=5+17·2 ⇒ 12 潜格 ⇒ 2 段 ⇒ 解回 39 帧）",
          lat(39) == 12 and plan["chunks"] >= 1 and 0 <= plan["padTokens"] < 5
          and fra(lat(39)) == 39, (lat(39), plan, fra(lat(39))))


# ─────────────────────────────────────────────────────────────────────────────
# ③ 配置守卫：分块参数必须是**空间倍率的整数倍** ✓（否则潜空间对不齐 ⇒ 画面错位 ✗）
# ─────────────────────────────────────────────────────────────────────────────
def case_config() -> None:
    cfg = vae_h3_mod.H3VideoVAEConfig()
    check("配置: 时间口径（17 帧 / 每段 5 潜格 / 多带 2 格 / 叠 5 帧 / 开头丢 3 帧）",
          (cfg.clip_length, cfg.token_chunk, cfg.token_overlap, cfg.frame_overlap,
           cfg.frame_pre_padding) == (17, 5, 2, 5, 3),
          (cfg.clip_length, cfg.token_chunk, cfg.token_overlap, cfg.frame_overlap,
           cfg.frame_pre_padding))
    raisers: list[str] = []
    for kwargs in ({"tile_size": 100}, {"tile_overlap_min": 0}, {"tile_size": 0}):
        try:
            vae_h3_mod.H3VideoVAEConfig(**kwargs)
        except vae_h3_mod.H3VideoVAEError:
            raisers.append(str(kwargs))
    check("配置: 分块参数**不是 16 的倍数 / ≤0 ⇒ 当场拒**（静默放过 ⇒ 画面错位且不报错 ✗）",
          len(raisers) == 3, raisers)
    report = cfg.to_dict()
    need = {"weights", "device", "latentChannels", "spatialScale", "temporalScale",
            "tokenChunk", "tokenOverlap", "frameOverlap", "framePrePadding", "strict"}
    check("配置: 报告口径**含解码端要用的每一项**（少一项调用方就只能猜 ✗）",
          need <= set(report), sorted(need - set(report)))
    check("配置: 默认**严格** 且默认落 **cpu**（真权重 5.2 GiB 不抢 DiT 那 19.62 GiB ✗）",
          report["strict"] is True and report["device"] == "cpu",
          (report["strict"], report["device"]))
    # ⭐ `frame_overlap` 减的是 **`frame_pre_padding`** ✓（参考逐字 ✓），不是 `token_drop` ✗。
    #    ⚠️ 默认档两者**都是 3** ✓ ⇒ 写错也照样算出 5 ✓ **不报错** ✓ ⇒ 只能靠换一个 clip 长度来分岔 ✓
    #    （20 帧 ⇒ pre_padding 0 ✓、drop 仍 3 ✓ ⇒ 正解 8 ✓ / 写错给 5 ✗）。
    odd = vae_h3_mod.H3VideoVAEConfig(clip_length=20)
    check("配置: ⭐⭐ 帧重叠按 ``token_overlap·4 − frame_pre_padding`` ✓ 算（**不是**减 `token_drop` ✗）"
          "—— 默认档两者同为 3 ⇒ 写错也不报错 ✓✗，换 clip=20 才分岔 ✓",
          (odd.frame_pre_padding, odd.frame_overlap) == (0, 8),
          (odd.frame_pre_padding, odd.frame_overlap, odd.token_drop))


# ─────────────────────────────────────────────────────────────────────────────
# ④ 拒绝语义：不给权重 / 文件不存在 ⇒ **明确报错** ✗（不返回半个模型 ✓）
# ─────────────────────────────────────────────────────────────────────────────
def case_refuse() -> None:
    try:
        vae_h3_mod.build_h3_video_vae(vae_h3_mod.H3VideoVAEConfig())
    except vae_h3_mod.H3VideoVAEError as err:
        check("拒绝: 不给权重 ⇒ `H3VideoVAEError`（随机初始化只会骗自己 ✗）",
              "权" in str(err) or "weights" in str(err), str(err)[:120])
    except Exception as err:  # noqa: BLE001
        check("拒绝: 不给权重 ⇒ `H3VideoVAEError`", False, f"{type(err).__name__}: {err}")
    else:
        check("拒绝: 不给权重 ⇒ `H3VideoVAEError`", False, "没抛")

    missing = str(Path(os.getcwd()) / "no_such_vae_weights.safetensors")
    try:
        vae_h3_mod.build_h3_video_vae(vae_h3_mod.H3VideoVAEConfig(weights=missing))
    except vae_h3_mod.H3VideoVAEError:
        check("拒绝: 权重文件不存在 ⇒ `H3VideoVAEError`（不是 FileNotFoundError 那种裸抛 ✗）", True)
    except Exception as err:  # noqa: BLE001
        check("拒绝: 权重文件不存在 ⇒ `H3VideoVAEError`", False, f"{type(err).__name__}: {err}")
    else:
        check("拒绝: 权重文件不存在 ⇒ `H3VideoVAEError`", False, "没抛")


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ 装配路：发现不到权重 ⇒ **拒装** ✗（绝不回落参考族 ✗✗ —— 那正是"以为在出真画面"的来路）
# ─────────────────────────────────────────────────────────────────────────────
def case_no_silent_fallback() -> None:
    original = vae_h3_mod.h3_video_vae_weights_on_disk
    vae_h3_mod.h3_video_vae_weights_on_disk = lambda **_kw: {  # type: ignore[assignment]
        "key": None, "path": None, "candidates": [], "catalogError": None, "notes": []}
    try:
        try:
            runtime_mod._attach_video_vae(None, {"latents_dim": 24}, real=True)  # noqa: SLF001
        except TorchBackendUnavailable as err:
            check("装配: 要真 VAE 但盘上没有 ⇒ **拒装**（`backend=None` 也没被碰过 ⇒ 拒在发现那一步）",
                  "没发现" in str(err) or "没" in str(err), str(err)[:160])
        except Exception as err:  # noqa: BLE001
            check("装配: 要真 VAE 但盘上没有 ⇒ `TorchBackendUnavailable`", False,
                  f"{type(err).__name__}: {err}")
        else:
            check("装配: 要真 VAE 但盘上没有 ⇒ **拒装**", False, "没抛（= 悄悄回落了 ✗）")
    finally:
        vae_h3_mod.h3_video_vae_weights_on_disk = original  # type: ignore[assignment]

    # ⚠️ 通道数对不上也要拒 ✓ —— 但它发生在**发现之后** ⇒ 这里给一个假路径 + 假发现 ✓
    vae_h3_mod.h3_video_vae_weights_on_disk = lambda **_kw: {  # type: ignore[assignment]
        "key": "fake", "path": __file__, "candidates": [], "catalogError": None, "notes": []}
    try:
        try:
            runtime_mod._attach_video_vae(None, {"latents_dim": 32}, real=True)  # noqa: SLF001
        except TorchBackendUnavailable as err:
            check("装配: 主干 latent 通道 32 ✗ ≠ 24 ⇒ 拒装（不拿「差不多」顶上 ✗）",
                  "24" in str(err), str(err)[:120])
        except Exception as err:  # noqa: BLE001
            check("装配: 主干 latent 通道 ≠ 24 ⇒ 拒装", False, f"{type(err).__name__}: {err}")
        else:
            check("装配: 主干 latent 通道 ≠ 24 ⇒ 拒装", False, "没抛")
    finally:
        vae_h3_mod.h3_video_vae_weights_on_disk = original  # type: ignore[assignment]

    # ⚠️ 参考族那条老路**一字未动** ✓（默认 `real=False` ⇒ 还是 `VideoVAEConfig` ✓）
    import inspect  # noqa: PLC0415

    sig = inspect.signature(runtime_mod._attach_video_vae)  # noqa: SLF001
    check("装配: `real` 默认 **False** ⇒ 不点名就还是参考族 ✓（老路一字未动 ✓）",
          sig.parameters["real"].default is False, sig.parameters["real"].default)
    loaded = inspect.signature(runtime_mod.EngineRuntime.ensure_loaded)
    check("装配: `ensure_loaded` 三个新口**都在签名上** ✓ 且默认关 ✓",
          {"real_video_vae", "video_vae_device", "video_vae_weights"} <= set(loaded.parameters)
          and loaded.parameters["real_video_vae"].default is False,
          sorted({"real_video_vae", "video_vae_device", "video_vae_weights"}
                 - set(loaded.parameters)))


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ 真权重档（门控）：装载 → 编码 → 解码 → 后端接线
# ─────────────────────────────────────────────────────────────────────────────
def case_live() -> None:
    import torch  # noqa: PLC0415 —— 门控进来的 ✓

    found = vae_h3_mod.h3_video_vae_weights_on_disk()
    path = str((found or {}).get("path") or "")
    if not path:
        skip("真权重档：清单里没发现 H3 视频 VAE ⇒ 跳过（这不是「通过」✗）")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    # ⚠️ 后端与 VAE **钉同一个设备** ✓（`decode` 会拒跨设备 ✓ —— 自检也别指望它替人搬 ✗）；
    #    有卡就走卡 ✓（生产就是这样 ✓：fp16 5.2 GiB ✓），没卡退回 cpu fp32 ✓（慢但等价 ✓）。
    import time  # noqa: PLC0415

    backend = TorchBackend(device=device)
    mark = time.perf_counter()
    report = backend.attach_vae(vae_h3_mod.H3VideoVAEConfig(
        weights=path, device=device, tiling=False))
    t_load = time.perf_counter() - mark
    check("真权重: 报告 `kind=h3-real` 且 `complete=True`（装不齐不许当能用 ✗）",
          report.get("kind") == "h3-real" and report.get("complete") is True,
          {"kind": report.get("kind"), "complete": report.get("complete"),
           "missing": (report.get("missing") or [])[:4]})
    check("真权重: 报告 `spatialScale=16` / `latentChannels=24`（与主干潜形态同源 ✓）",
          int(report.get("spatialScale") or 0) == 16
          and int(report.get("latentChannels") or 0) == 24,
          (report.get("spatialScale"), report.get("latentChannels")))
    # ⚠️ 比**设备族**不比字面 ✗：`"cuda"` 落进报告会写成 `"cuda:0"` ✓（同一张卡 ✓ 不是不一致 ✗）
    check("真权重: 权重落的设备**如实报**（默认 cpu ✓；cuda 需 DiT 先下卡 ✓）",
          str(report.get("weightDevice")).split(":")[0] == device.split(":")[0],
          report.get("weightDevice"))

    module = backend._vae  # noqa: SLF001 —— 编码侧后端没有独立入口 ✓
    torch.manual_seed(20261029)
    # 结构化输入（**不是纯噪声** ✗）：中心一块亮、四周暗 + 一条斜纹 ⇒ 还原度有可比性 ✓
    frames, size = 5, 64
    x = torch.full((1, 3, frames, size, size), -0.8)
    x[..., size // 4:3 * size // 4, size // 4:3 * size // 4] = 0.9
    for step in range(frames):
        x[:, :, step, step * 8:(step + 1) * 8, :] = 0.3

    mark = time.perf_counter()
    with torch.no_grad():
        z = module.encode(x.to(device))
    t_enc = time.perf_counter() - mark
    want_hw = size // 16
    check("真权重: 编码形状 `[1,24,T_L,H/16,W/16]`（倍率现算，不是猜的 ✗）",
          tuple(z.shape) == (1, 24, vae_h3_mod.latents_for_frames(frames), want_hw, want_hw),
          tuple(z.shape))

    stats = vae_h3_mod.verify_latents_stats(z)
    check("真权重: 潜变量量级体检**跑得通**（`ok` 只作参考 ✓，关键是它认得这 24 个通道 ✓）",
          isinstance(stats, dict) and stats.get("channels") == 24
          and isinstance(stats.get("frames"), int),
          {key: stats.get(key) for key in ("ok", "tolerance", "channels", "frames")}
          if isinstance(stats, dict) else stats)

    mark = time.perf_counter()
    out = backend.decode(z, None, None)
    t_dec = time.perf_counter() - mark
    pix = out["frames"]
    check("真权重: **`decode()` 不再自称合成**（`synthetic=False` + `vaeKind=h3-real` ✓）",
          out.get("synthetic") is False and out.get("vaeKind") == "h3-real",
          (out.get("synthetic"), out.get("vaeKind")))
    check("真权重: 解码形状 `[B,3,T,16·H_L,16·W_L]` 且 T ≥ 输入帧数",
          tuple(pix.shape)[0] == 1 and tuple(pix.shape)[1] == 3
          and tuple(pix.shape)[3] == size and tuple(pix.shape)[4] == size
          and int(pix.shape[2]) >= frames, tuple(pix.shape))
    check("真权重: 像素落在 **[0,1]**（`media.write_video` 的契约 ✓）",
          float(pix.min()) >= 0.0 and float(pix.max()) <= 1.0
          and str(pix.dtype) == "torch.float32",
          (float(pix.min()), float(pix.max()), str(pix.dtype)))

    # ⚠️ 还原度：不要求逐位（VAE 本就有损 ✓）—— 要求"**比错配的输入更像我**" ✓
    #    这是能证伪的那一档 ✓：接错权重 / 归一化常量翻车 ⇒ 两条相关性会**一样烂** ✓✗
    got = pix[0, :, :frames].float()
    ref = ((x[0] + 1) / 2).float()
    shifted = torch.roll(ref, shifts=size // 2, dims=-1)
    corr_ref = _pearson(got, ref)
    corr_shift = _pearson(got, shifted)
    check("真权重: 还原度**指向自己的输入**（相关性 > 错配一半的图，且 > 0.5）",
          corr_ref > corr_shift and corr_ref > 0.5,
          {"corrRef": round(corr_ref, 4), "corrShift": round(corr_shift, 4)})

    # ⚠️ 单帧那条路（= 首帧条件用的就是它 ✓）：换实现最容易在这里错位 ✓✗
    with torch.no_grad():
        z1 = module.encode(x[:, :, :1].to(device))
        one = backend.decode(z1, None, None)
    check("真权重: 单帧编码只留 1 个潜格、解码回 1 帧（首帧条件走的正是这条路 ✓）",
          tuple(z1.shape)[2] == 1 and int(one["frames"].shape[2]) == 1,
          (tuple(z1.shape), tuple(one["frames"].shape)))

    # 事实行（不进判据 ✓ —— 只为把"这台机器上到底多快 / 多像"记下来 ✓）
    print(f"FACT  {device} · {Path(path).name} · 装载 {t_load:.1f}s / "
          f"编码 {frames}帧@{size}² {t_enc:.2f}s / 解码 {tuple(pix.shape)[2]}帧 {t_dec:.2f}s · "
          f"corr(自己)={corr_ref:.4f} vs corr(错配)={corr_shift:.4f} · "
          f"潜变量体检 ok={stats.get('ok')}")


def _pearson(a: Any, b: Any) -> float:
    import torch  # noqa: PLC0415

    left = a.reshape(-1).double()
    right = b.reshape(-1).double()
    left = left - left.mean()
    right = right - right.mean()
    denom = float(left.norm() * right.norm())
    return 0.0 if denom == 0 else float((left * right).sum()) / denom


def main() -> int:
    case_facts()
    case_geometry()
    case_config()
    if not _have_torch():
        skip("缺 torch ⇒ 装载拒绝那两组（`build_h3_video_vae` 要真张量）跳过 ✓")
    else:
        case_refuse()
    case_no_silent_fallback()

    live = str(os.environ.get("DARME_H3_VAE_LIVE") or "").strip() not in ("", "0", "false")
    if not live:
        skip("真权重档未开（`DARME_H3_VAE_LIVE=1` 才跑 ✓ —— 要真读 5 GiB 权重）")
    elif not _have_torch():
        skip("真权重档：缺 torch")
    else:
        case_live()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
