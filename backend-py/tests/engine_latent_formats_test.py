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
    check("②⁵ 均值/标准差族有 4 条 ✓（Wan21 / Wan22 / Mochi / SDXL-Playground ✓）",
          sorted(n for n, s in lf.LATENT_FORMATS.items() if s.process == lf.PROCESS_MEANSTD)
          == ["mochi", "sdxl_playground_2_5", "wan21", "wan22"])


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
