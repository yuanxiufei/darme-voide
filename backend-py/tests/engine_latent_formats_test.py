"""自研引擎·潜空间口径表 ✓（表自洽 / 已知数值 / **跨来源逐值** / 别名归一 / 五种换算语义 / 报错口径 ✓）。

口径（本套盯的就是「**同一事实只写一份**」✓）：
* H3 的 24 通道**不是** `latent_formats` 发明的 ✓ ⇒ 与 `latent_container.H3_VIDEO_CHANNELS` ✓、
  `h3_form.H3_TRUNK_DEFAULTS["latents_dim"]` ✓ **逐值比对** ✓（谁改了另一边这里会红 ✗）；
* 音频 32 通道同理 ✓，且**必须区别于** `geometry.AUDIO_LATENT_CHANNELS`（那个是**立体声声声道数 = 2** ✓✗）；
* 值域换算要 torch ✓（**查表不需要** ✓）⇒ 没装 torch 时那几条**显式 SKIP** ✓（打摘要 ✓，不静默消失 ✗）。

跑法::

    ./.venv/Scripts/python.exe tests/engine_latent_formats_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import geometry  # noqa: E402
from app.services.engine import h3_form  # noqa: E402
from app.services.engine import latent_container as lc  # noqa: E402
from app.services.engine import latent_formats as lf  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def case_table() -> None:
    """① 表自洽：非空 / 键与规范名一致 / 换算标签合法 ✓。"""
    names = lf.list_latent_formats()
    check("① 表非空且成规模 ✓（≥ 30 条口径 ✓）", len(names) >= 30, len(names))
    check("①′ 列名**已排序** ✓（稳定输出 ✓ 便于比对 ✓）", names == sorted(names))
    check("①″ 每条 spec 的 ``name`` 与表键**一致** ✓",
          all(spec.name == key for key, spec in lf.LATENT_FORMATS.items()),
          [k for k, s in lf.LATENT_FORMATS.items() if s.name != k])
    check("①‴ 换算标签全在 :data:`PROCESS_MODES` 里 ✓",
          all(spec.process in lf.PROCESS_MODES for spec in lf.LATENT_FORMATS.values()))


def case_known_values() -> None:
    """② 已知数值：SD 系 / SD3-Flux 的 shift / H3 的下采样比 ✓（写错一位就红 ✓）。"""
    sd15 = lf.get_latent_format("sd15")
    sdxl = lf.get_latent_format("sdxl")
    sd3 = lf.get_latent_format("sd3")
    flux = lf.get_latent_format("flux")
    h3 = lf.get_latent_format("h3_video")
    check("② SD15 = 4 通道 2 维 × ``0.18215`` ✓ 且是**纯缩放** ✓",
          (sd15.latent_channels, sd15.latent_dimensions, sd15.scale_factor, sd15.process)
          == (4, 2, 0.18215, lf.PROCESS_SCALE), sd15.to_dict())
    check("②′ SDXL = 4 通道 × ``0.13025`` ✓（与 SD15 **不是**同一个数 ✓✗）",
          (sdxl.latent_channels, sdxl.scale_factor) == (4, 0.13025) and sdxl.scale_factor != sd15.scale_factor)
    check("②″ SD3 / Flux = 16 通道 + **先减 shift 再乘 scale** ✓（二者语义同类 ✓ 数值不同 ✓）",
          (sd3.process, sd3.shift_factor, sd3.scale_factor, sd3.latent_channels)
          == (lf.PROCESS_AFFINE, 0.0609, 1.5305, 16)
          and (flux.process, flux.shift_factor) == (lf.PROCESS_AFFINE, 0.1159))
    check("②‴ H3 视频 = 3 维 + 下采样 **16× 空间 / 4× 时间** ✓（实测口径 ✓ 见 ``vae_h3`` ✓）",
          (h3.latent_dimensions, h3.spatial_downscale_ratio, h3.temporal_downscale_ratio)
          == (3, 16, 4), h3.to_dict())
    check("②⁗ 像素空间族 = **恒等** ✓（3 通道 ✓ 不改数值 ✓）",
          lf.get_latent_format("pixel_space").process == lf.PROCESS_IDENTITY
          and lf.get_latent_format("pixel_space").latent_channels == 3)
    check("②⁵ 均值/标准差族有 6 条 ✓（Wan21 / Wan22 / Mochi / SDXL-Playground / Qwen-Image 2.1 / "
          "Qwen-Image 2512 ✓）",
          sorted(n for n, s in lf.LATENT_FORMATS.items() if s.process == lf.PROCESS_MEANSTD)
          == ["mochi", "qwen_image", "qwen_image21", "sdxl_playground_2_5", "wan21", "wan22"])
    check("②⁵′ ⭐ Qwen-Image 2.1 = **64 通道 / 16×** ✓ 且与 ``vae_wan`` 的结构事实**逐值相同** ✓"
          "（两处一起漂才会漏 ✓✗）",
          (lf.get_latent_format("qwen_image21").latent_channels,
           lf.get_latent_format("qwen_image21").spatial_downscale_ratio) == (64, 16))


#: 参考实现 ``comfy/latent_formats.py`` **逐类声明**的空间/时间下采样比 ✓（2026-10-31 抄录 ✓）。
#: ⚠️ 只抄「**不等于基类默认 8 / 1**」的那些 ✓ —— 参考实现没写的族就是 8 / 1 ✓✗。
#: 这张表就是判据：我们表里少抄一格 ⇒ 拿它算潜形状的尺寸/帧数**全错却不报错** ✓✗
#: （这类错最贵：图/视频出来只是"尺寸不对"或"抖动"，看不出根因 ✓✗）。
REF_DOWNSCALE = {
    "h3_video": (16, 4),
    "wan21": (8, 4),
    "wan22": (16, 4),       # Wan22 **继承** Wan21 的时间 4 ✓、空间改 16 ✓
    "hunyuan_video": (8, 4),
    "hunyuan_video_15": (16, 4),
    "hunyuan_image_21": (32, 1),
    "ltxv": (32, 8),
    "ltxav": (32, 8),       # LTXAV **继承** LTXV ✓
    "mochi": (8, 6),        # **不是** 4 ✓✗
    "cosmos_1_c_v8x8x8": (8, 8),
    "cogvideox": (8, 4),
    "cogvideox_1_5": (8, 4),  # 继承 CogVideoX ✓
    "flux2": (16, 1),
    "sc_prior": (42, 1),
    "sc_b": (4, 1),
    "pixel_space": (1, 1),
    "stable_audio_1": (1, 2048),
    "stable_audio_3": (1, 4096),
    "ace_audio_15": (1, 1764),
    "yue2": (1, 1920),
    "minimax_music3": (1, 512),
    "qwen_image21": (16, 1),  # 本仓另有 ``vae_wan.WanVaeShape.scale`` 佐证 ✓
}


def case_downscale() -> None:
    """②⁶ ⭐ 下采样比逐族与参考实现**逐值一致** ✓（2026-10-31 补 ✓✗）。

    ⚠️ 这条防的是「**统一按 8 / 1**」那一整片静默错值 ✓✗：原先除 H3 外全是基类默认 ✓，
    注释还写着「未采集、别当实测值」✓ —— 可这两格是参考实现**明文声明**的 ✓，
    真用起来（算潜形状 / 推帧数）就是一错到底且**不报错** ✓✗。
    """
    got = {name: (spec.spatial_downscale_ratio, spec.temporal_downscale_ratio)
           for name, spec in lf.LATENT_FORMATS.items()}
    wrong = {name: (want, got.get(name)) for name, want in REF_DOWNSCALE.items()
             if got.get(name) != want}
    check("②⁶ 逐族下采样比与参考实现一致 ✓（少抄/抄错一格都在这里红 ✓✗）", not wrong, wrong)
    check("②⁶′ Qwen-Image **2512** 走参考实现基类默认 **8 / 1** ✓（它在 comfy 里就是 "
          "``LatentFormat`` 基类 ✓ —— 与 2.1 的 16× / Wan21 的 16ch 三族分开 ✓）",
          got["qwen_image"] == (8, 1), got["qwen_image"])
    #: 参考实现里这两对是**继承**关系 ✓ ⇒ 子类必须继承父类那两格 ✓✗（"改了一个忘了另一个" ✓✗）。
    pairs = (("ltxav", "ltxv"), ("cogvideox_1_5", "cogvideox"))
    drifted = {child: (got[child], got[parent]) for child, parent in pairs
               if got[child] != got[parent]}
    check("②⁶″ 继承族的那两格必须与父类**逐值相同** ✓（``LTXAV(LTXV)`` / ``CogVideoX1_5(CogVideoX)`` ✓）",
          not drifted, drifted)
    check("②⁶‴ ``Wan22(Wan21)`` 只改了**空间**那一格 ✓（时间仍 4 ✓、空间 8→16 ✓）",
          got["wan22"][1] == got["wan21"][1] and got["wan22"][0] != got["wan21"][0],
          (got["wan21"], got["wan22"]))


def case_cross_source() -> None:
    """③ ⭐ 跨来源：通道数**不是这里发明的** ✓（逐值比对 ⇒ 两处都不会漂 ✓）。"""
    trunk = h3_form.H3_TRUNK_DEFAULTS
    chan = lf.get_latent_format("h3_video").latent_channels
    check("③ ⭐ 三处**逐值相同** ✓：`latent_formats` == `latent_container.H3_VIDEO_CHANNELS` == "
          "`h3_form.H3_TRUNK_DEFAULTS['latents_dim']`（同一个事实 ✓）",
          chan == lc.H3_VIDEO_CHANNELS == int(trunk["latents_dim"]),
          (chan, lc.H3_VIDEO_CHANNELS, trunk["latents_dim"]))
    audio = lf.get_latent_format("h3_audio").latent_channels
    check("③′ ⭐ **音频** 32 通道与主干默认值**逐值相同** ✓（``audio_latents_dim`` ✓）",
          audio == lf.H3_AUDIO_LATENT_CHANNELS == int(trunk["audio_latents_dim"]),
          (audio, lf.H3_AUDIO_LATENT_CHANNELS, trunk["audio_latents_dim"]))
    check("③″ ⚠️ 32 **不是** ``geometry.AUDIO_LATENT_CHANNELS`` ✓ —— 那个是**立体声声声道数 = 2** ✓✗"
          "（两码事 ✓ 混了就会把声道数当潜通道 ✓✗）",
          audio != int(geometry.AUDIO_LATENT_CHANNELS) and int(geometry.AUDIO_LATENT_CHANNELS) == 2,
          (audio, geometry.AUDIO_LATENT_CHANNELS))


def case_aliases() -> None:
    """④ 别名/大小写归一 ✓；认不出 ⇒ **报错** ✗（不回退默认 ✗✗）。"""
    check("④ 大小写不敏感 ✓（``SDXL`` ⇒ ``sdxl`` ✓）", lf.normalize_latent_format("SDXL") == "sdxl")
    check("④′ 连字符/下划线变体 ✓（``MiniMax_H3`` / ``chroma_radiance`` / ``wAn2.1`` ✓）",
          (lf.normalize_latent_format("MiniMax_H3"), lf.normalize_latent_format("chroma_radiance"),
           lf.normalize_latent_format("wAn2.1")) == ("h3_video", "pixel_space", "wan21"))
    check("④″ 传 :class:`LatentFormatSpec` **原样返回** ✓（便于内部复用 ✓）",
          lf.get_latent_format(lf.LATENT_FORMATS["sd15"]) is lf.LATENT_FORMATS["sd15"])
    text = _raises(lambda: lf.normalize_latent_format("no_such_format"), "认不出")
    check("④‴ 认不出 ⇒ 报错 ✓，且**明说不回退默认 4 通道** ✗✗（那等于当 SD1.5 用 ✓✗）",
          text is not None and "不回退默认" in text, text)
    check("④⁗ 空名字 ⇒ 也报错 ✓（不静默当默认 ✓）",
          _raises(lambda: lf.normalize_latent_format(""), "空的") is not None)


def case_validation() -> None:
    """⑤ 构造期校验：标签与 ``shift_factor`` **必须自洽** ✓（填错就报 ✓ 不带走 ✓✗）。"""
    check("⑤ ``affine`` 少给 ``shift_factor`` ⇒ 报错 ✓（少减一个数 = **偏移的**潜变量 ✓✗）",
          _raises(lambda: lf.LatentFormatSpec("x", process=lf.PROCESS_AFFINE), "shift_factor") is not None)
    check("⑤′ 纯 ``scale`` 却填了 ``shift_factor`` ⇒ 报错 ✓（标签选错 ✓✗）",
          _raises(lambda: lf.LatentFormatSpec("x", shift_factor=0.1), "shift_factor") is not None)
    check("⑤″ 未知换算标签 ⇒ 报错 ✓（不猜一个最像的 ✓）",
          _raises(lambda: lf.LatentFormatSpec("x", process="magic"), "不合法") is not None)
    check("⑤‴ 通道/维数非正 ⇒ 报错 ✓；**空名字** ⇒ 也报错 ✓（不静默当默认 ✓）",
          _raises(lambda: lf.LatentFormatSpec("x", latent_channels=0), "必须为正") is not None
          and _raises(lambda: lf.LatentFormatSpec(""), "不能为空") is not None)


def case_process() -> None:
    """⑥ 五种换算语义的真张量行为 ✓（往返恒等 ✓ / 缺 mean-std 就报 ✗ / 重排型拒 ✗）。"""
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        skip("没装 torch ⇒ 值域换算那几条不跑 ✓（查表/归一那几条照跑 ✓）")
        return

    once = torch.tensor([1.0])
    check("⑥ 纯缩放：``sd15`` 往返 ⇒ **恒等** ✓（乘 0.18215 再除回来 ✓）",
          torch.allclose(lf.process_out("sd15", lf.process_in("sd15", once)), once, atol=1e-6))
    x = torch.tensor([0.1])
    expect = (x - 0.0609) * 1.5305
    check("⑥′ ``affine``：**先减 shift 再乘 scale** ✓（SD3 ✓）且往返恒等 ✓",
          torch.allclose(lf.process_in("sd3", x), expect, atol=1e-6)
          and torch.allclose(lf.process_out("sd3", lf.process_in("sd3", x)), x, atol=1e-6))
    check("⑥″ ``identity``：**原样返回** ✓（像素空间族不做换算 ✓ —— 别顺手乘一个数 ✗）",
          torch.equal(lf.process_in("pixel_space", x), x)
          and torch.equal(lf.process_out("pixel_space", x), x))
    text = _raises(lambda: lf.process_in("wan21", x), "mean")
    check("⑥‴ ``meanstd`` **不带** mean/std ⇒ 报错 ✓（均方差属**权重** ✓ 本表不假装有 ✗✗）",
          text is not None and "mean" in text, text)
    mean, std = torch.tensor([0.5]), torch.tensor([2.0])
    y = torch.tensor([1.5])
    check("⑥⁗ ``meanstd`` 给了 mean/std ⇒ 往返恒等 ✓（``(x−mean)·scale/std`` ✓）",
          torch.allclose(lf.process_in("wan21", y, mean=mean, std=std),
                         (y - mean) * 1.0 / std, atol=1e-6)
          and torch.allclose(lf.process_out("wan21", lf.process_in("wan21", y, mean=mean, std=std),
                                            mean=mean, std=std), y, atol=1e-6))
    text2 = _raises(lambda: lf.process_in("hunyuan_image_21_refiner", x), "重排")
    check("⑥⁵ ``rearrange`` 型 ⇒ **具名拒绝** ✗（只乘个数 = 少了一半重排的错值 ✓✗）",
          text2 is not None and "重排" in text2, text2)
    check("⑥⁶ 非张量**可转**就转 ✓（``[1.0]`` 这种列表也认 ✓）",
          lf.process_in("sd15", [1.0]) is not None)


def main() -> int:
    case_table()
    case_known_values()
    case_downscale()
    case_cross_source()
    case_aliases()
    case_validation()
    case_process()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
