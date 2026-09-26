"""S7 自检：**联合 AV 潜变量的容器互操作**（零依赖 ✓ **不需要 torch** ✓ 2026-09-24）。

⚠️ 本套**只用假容器/假张量** ✓（鸭子类型 ✓）—— 本仓规矩：**测试不许依赖真机装了什么** ✗，
而真 torch 张量与真 ComfyUI 容器天然满足同一套最小协议 ✓ ⇒ 用假的验判据、判据本身与真货同源 ✓。

钉的是两条会**静默出错**的判据 ✗✗：

* ⭐ **认视频流**：0 条 ⇒ 报错并**印出各流形状** ✓；≥2 条 ⇒ 报错「**歧义**」✗ 且**不许挑第一条** ✗✗
  （挑错 = 把音频流当画面解 ✓✗）；
* ⭐ **换流保类型**：重建失败 ⇒ **报错** ✗ —— **不静默退化成 list** ✗✗（那会让下游炸得莫名其妙 ✓）；
* ⭐ 通道数**不是本模块发明的** ✓：与 `dit.H3_SHAPE_FACTS` 的 `latents_dim` 逐值比对 ✗
  （两处各写一份必然漂 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_latent_container_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import h3_form  # noqa: E402
from app.services.engine import latent_container as lc  # noqa: E402
from app.services.engine import vae as vae_mod  # noqa: E402

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


class FakeTensor:
    """假张量 ✓（只满足最小协议：``shape`` + ``ndim`` ✓）。"""

    def __init__(self, *shape: int) -> None:
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def __repr__(self) -> str:  # pragma: no cover - 只为报错好看 ✓
        return f"FakeTensor{self.shape}"


class FakeNested:
    """假容器 ✓（``is_nested`` + ``unbind`` ✓ —— 就是 ComfyUI ``NestedTensor`` 的最小协议 ✓）。"""

    is_nested = True

    def __init__(self, tensors: Any) -> None:
        self.tensors = list(tensors)

    def unbind(self) -> tuple[Any, ...]:
        return tuple(self.tensors)


class WrapperNested:
    """**另一种**容器 ✓（不是 ``FakeNested`` 的子类 ✓）—— 用来验「鸭子类型真的认别人」✓。"""

    def __init__(self, tensors: Any) -> None:
        self._streams = tuple(tensors)

    is_nested = True

    def unbind(self) -> tuple[Any, ...]:
        return self._streams


class OldNested:
    """**老容器** ✓：构造签名与新版不同（要额外参数 ✓）⇒ 按原类型重建会失败 ✓。

    ⚠️ 这才是真实场景 ✓（真 ComfyUI 的新老容器**签名不同** ✓）—— 本套第一次用的是「只吃
    ``tuple`` 的容器」✗，那是**我想当然造的** ✗（真容器吃的是可迭代 ✓）。
    """

    is_nested = True

    def __init__(self, tensors: Any, version: str = "") -> None:
        if version != "v1":
            raise TypeError("老容器只认 version='v1' ✗")
        self.tensors = list(tensors)

    def unbind(self) -> tuple[Any, ...]:
        return tuple(self.tensors)


class NewNested(FakeNested):
    """**新容器** ✓：真 ComfyUI 那种「吃一个可迭代」的签名 ✓。"""


class Stubborn:
    """怎么构造都失败 ✓（验「失败要报错、**不退化**」✓✗）。"""

    is_nested = True

    def __init__(self, tensors: Any) -> None:  # noqa: ARG002
        raise TypeError("我就是不让构造 ✓")

    def unbind(self) -> tuple[Any, ...]:
        return ()


VIDEO = FakeTensor(1, 24, 5, 32, 32)
AUDIO = FakeTensor(1, 8, 40)


def case_bare_tensor() -> None:
    """① 裸张量 ✓：形状对 ⇒ 直接当视频 ✓；形状不对 ⇒ **报错**（不是将就用 ✗）。"""
    got = lc.split_video_stream(VIDEO)
    check("① 裸张量且形状是 H3 视频流 ⇒ ``is_container=False`` ✓、拿到的就是它本身 ✓",
          got.is_container is False and got.video is VIDEO and got.index is None,
          got.to_dict())
    wrong = _raises(lambda: lc.split_video_stream(AUDIO), "24")
    check("①′ ⭐ 裸张量形状不对（音频 3 维 ✓）⇒ 报错且**点名通道数 24** ✓（拿它当视频解 = 一堆噪声 ✓✗）",
          wrong is not None and "B×24×T×H×W" in wrong, wrong)
    check("①″ 既不是容器也不是张量（比如 int ✓）⇒ 报错并说清「缺什么」✓",
          _raises(lambda: lc.split_video_stream(42), "shape") is not None)


def case_container() -> None:
    """② 容器 ✓：**恰好一条**视频流才行 ✓（0 条 / ≥2 条都报错 ✓）。"""
    got = lc.split_video_stream(FakeNested([VIDEO, AUDIO]))
    check("② 视频 + 音频 ⇒ 认出下标 0 ✓、视频是**原对象** ✓",
          got.is_container and got.index == 0 and got.video is VIDEO, got.to_dict())
    later = lc.split_video_stream(FakeNested([AUDIO, VIDEO]))
    check("②′ 视频排在后面也认得出 ✓（下标 1 ✓ —— 不是「永远取第 0 条」✗）",
          later.index == 1 and later.video is VIDEO, later.to_dict())
    none_msg = _raises(lambda: lc.split_video_stream(FakeNested([AUDIO, AUDIO])), "(1, 8, 40)")
    check("②″ ⭐ 没有视频流 ⇒ 报错且**印出实际各流形状** ✓（不印就只能靠猜 ✓✗）、"
          "并写明**不猜**哪条是视频 ✓",
          none_msg is not None and "不猜" in none_msg, none_msg)
    two = _raises(lambda: lc.split_video_stream(FakeNested([VIDEO, FakeTensor(1, 24, 5, 16, 16)])),
                  "歧义")
    check("②‴ ⭐⭐ 两条形状都像视频 ⇒ 报「**歧义**」✗✗ 且**不许挑第一条** ✗（挑错就是解错流 ✓✗）",
          two is not None and "不猜第一条" in two, two)
    check("②⁴ ``describe_shapes``：容器 ⇒ 列出**每条流**的形状 ✓；裸张量 ⇒ 一个 ✓",
          lc.describe_shapes(FakeNested([VIDEO, AUDIO])) == [(1, 24, 5, 32, 32), (1, 8, 40)]
          and lc.describe_shapes(VIDEO) == [(1, 24, 5, 32, 32)])
    duck = lc.split_video_stream(WrapperNested([AUDIO, VIDEO]))
    check("②⁵ ⭐ **鸭子类型**：不是本套那个类、只满足 ``is_nested``+``unbind`` 也认 ✓"
          "（否则真容器一升级就全瞎 ✗）",
          duck.index == 1 and duck.video is VIDEO, duck.to_dict())


def case_replace() -> None:
    """③ 换流 ✓：**保类型** ✓、只换一槽 ✓、失败**报错不退化** ✗✗。"""
    nested = FakeNested([VIDEO, AUDIO])
    result = lc.split_video_stream(nested)
    new_video = FakeTensor(1, 24, 9, 32, 32)
    rebuilt = lc.replace_video_stream(result, new_video)
    check("③ 换流后**类型不变** ✓、视频槽换成新对象 ✓、另一端还是**原对象** ✓、顺序不变 ✓",
          isinstance(rebuilt, FakeNested) and rebuilt.tensors[0] is new_video
          and rebuilt.tensors[1] is AUDIO, [t.shape for t in rebuilt.tensors])
    check("③′ 裸张量入参 ⇒ 换流就是**直接返回新视频** ✓（没有容器要重建 ✓）",
          lc.replace_video_stream(lc.split_video_stream(VIDEO), new_video) is new_video)

    old = OldNested([VIDEO, AUDIO], "v1")
    fixed = lc.replace_video_stream(lc.split_video_stream(old), new_video,
                                    container_type=NewNested)
    check("③″ **老容器**（构造签名不同 ✓）按原类型重建会失败 ⇒ 走**声明的容器类型**这条 fallback ✓"
          "且成功 ✓（这就是现实里「老/新 ComfyUI 容器不同」那条路 ✓）",
          isinstance(fixed, NewNested) and fixed.tensors[0] is new_video, type(fixed).__name__)

    # ⚠️ 用 ``object.__new__`` 绕过 ``__init__`` ✓：要的就是「**能拆、不能建**」这种容器 ✓
    #    （直接 ``Stubborn()`` 会在**造夹具**那一步就抛 ✗ —— 本套真撞过一次 ✓）。
    stubborn = object.__new__(Stubborn)
    stubborn.unbind = lambda: (VIDEO, AUDIO)          # noqa: SLF001 —— 造一个「能拆不能建」的容器 ✓
    split = lc.split_video_stream(stubborn)
    msg = _raises(lambda: lc.replace_video_stream(split, new_video), "Stubborn")
    check("③‴ ⭐⭐ 重建失败 ⇒ **报错** ✗ 且说明**不静默退化成 list** ✗✗、并把试过的构造列出来 ✓",
          msg is not None and "list" in msg and "详情" in msg, msg)

    stale = lc.SplitResult(video=VIDEO, streams=(VIDEO,), index=5, container=FakeNested([VIDEO]))
    check("③⁴ 用**过期/越界**的 SplitResult 换流 ⇒ 报错 ✓（别把别人的流换串 ✓✗）",
          _raises(lambda: lc.replace_video_stream(stale, new_video), "不在容器范围内") is not None)


def case_cross_source() -> None:
    """④ ⭐ 跨来源：视频流通道数**不是本模块发明的** ✓（逐值比对 ✓ 两处都不会漂 ✓）。"""
    # ⚠️ 权威**不在** `dit.H3_SHAPE_FACTS` ✗（那张表里没有 `latents_dim` 这个键 ✓✗ ——
    #    本套第一次就把它写错成那个 ✓）。真权威是：主干默认值 ✓ + VAE 那 24 个统计的长度 ✓。
    trunk = int(h3_form.H3_TRUNK_DEFAULTS["latents_dim"])
    stats = len(vae_mod.H3_VIDEO_VAE_FACTS["latentsMean"])
    check("④ ⭐ 三处**逐值相同** ✓：`latent_container.H3_VIDEO_CHANNELS` == "
          "`h3_form.H3_TRUNK_DEFAULTS['latents_dim']` == len(`vae.H3_VIDEO_VAE_FACTS['latentsMean']`) "
          "（同一个事实 ✓ —— 谁改了任一边这里都会红 ✗）",
          lc.H3_VIDEO_CHANNELS == trunk == stats,
          (lc.H3_VIDEO_CHANNELS, trunk, stats))
    custom = FakeTensor(1, 6, 4, 8, 8)
    check("④′ 通道/维数**可覆盖** ✓（证明上面那句不是「写死 24」的套套逻辑 ✓）",
          lc.split_video_stream(custom, channels=6).video is custom
          and lc.is_video_stream(custom, channels=6) is True
          and lc.is_video_stream(custom) is False)


def main() -> int:
    case_bare_tensor()
    case_container()
    case_replace()
    case_cross_source()
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
