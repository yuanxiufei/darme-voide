"""**联合音视频潜变量的容器互操作** ✓（2026-09-24 补 ✓，口径来自逆向 ✓ 零依赖 ✓）。

## 解决什么
H3 的联合 AV 潜变量在 ComfyUI 侧**不是一个裸张量** ✗：它是个**容器**（``NestedTensor`` 一类 ✓），
``unbind()`` 出若干条流 ✓ —— 其中**视频流**是 ``B×24×T×H×W``（5 维、24 通道 ✓）、音频流是 3 维 ✓。
于是「把视频流换掉」有**两个坑** ✗：

1. **认出哪条是视频流** ✓：按形状认 ✓ —— ⚠️ 认不出来 / 认出**多条**都必须**报错** ✗，
   不许「挑第一条看着像的」✗✗（挑错就是把音频流当视频解 ✓✗）；
2. **换回去要保住容器类型** ✓：老/新 ComfyUI 的容器实现不同 ✓，重建时若退化成裸 ``list`` ✗
   会让下游炸得莫名其妙 ✓✗ ⇒ 重建失败要**报错并说清** ✓。

## 为什么写成**鸭子类型**（不 ``import torch`` ✗）
本仓规矩：**测试不许依赖真机装了什么** ✗ ⇒ 本模块只要求对象具备**最小协议** ✓
（``shape`` / ``ndim`` / ``unbind()`` ✓，真张量与真容器天然满足 ✓），自检用**假容器**跑 ✓✓。

## 不猜
* 视频流的通道数**不是本模块发明**的 ✓：它与 ``h3_form.H3_TRUNK_DEFAULTS["latents_dim"]`` ✓、
  以及 ``vae.H3_VIDEO_VAE_FACTS["latentsMean"]`` 的长度（24 个统计 ✓）**是同一个事实** ✓
  ⇒ 自检里**逐值比对**这几处 ✓✗（两处各写一份必然漂 ✓）；
* 认不出来 ⇒ 报错并**印出各流的形状** ✓（否则排查只能靠猜 ✓✗）；
* 容器重建失败 ⇒ 报错 + 说明「这种容器类型不支持这样重建」✓ —— **不静默退化成 list** ✗✗。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["H3_VIDEO_CHANNELS", "H3_VIDEO_RANK", "LatentContainerError", "SplitResult",
           "describe_shapes", "is_video_stream", "replace_video_stream", "split_video_stream"]

#: H3 **视频流**的通道数 ✓ —— ⚠️ 与 ``h3_form.H3_TRUNK_DEFAULTS["latents_dim"]`` 是**同一个事实** ✗
#: （自检逐值比对 ✓：主干默认值 ✓ + VAE 那 24 个统计的长度 ✓；谁改了另一边这里会红 ✓）。
H3_VIDEO_CHANNELS = 24
#: H3 **视频流**的维数 ✓（``B×C×T×H×W`` ⇒ 5 ✓；音频流是 3 ✓）。
H3_VIDEO_RANK = 5


class LatentContainerError(ValueError):
    """容器**认不出 / 换不回去** ✓ ⇒ 当场报 ✗（猜一条流出来比失败更贵 ✓✗）。"""


@dataclass(frozen=True)
class SplitResult:
    """拆出来的东西 ✓：``streams is None`` 表示**入参本来就是裸张量** ✓（不是容器 ✓）。"""

    video: Any
    streams: tuple[Any, ...] | None
    index: int | None
    container: Any

    @property
    def is_container(self) -> bool:
        return self.streams is not None

    def to_dict(self) -> dict[str, Any]:
        return {"isContainer": self.is_container, "index": self.index,
                "streams": describe_shapes(self.streams) if self.streams is not None else None,
                "videoShape": _shape_of(self.video)}


def _shape_of(value: Any) -> tuple[int, ...] | None:
    """鸭子类型的形状 ✓（``shape`` 能转成整型元组就算 ✓）。"""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _is_tensorish(value: Any) -> bool:
    return _shape_of(value) is not None and getattr(value, "ndim", None) is not None


def _is_container(value: Any, container_type: type | None = None) -> bool:
    """容器判定 ✓：真类型 / 或**鸭子类型**（``is_nested`` 为真 **且** 有 ``unbind`` ✓）。"""
    if container_type is not None and isinstance(value, container_type):
        return True
    return bool(getattr(value, "is_nested", False)) and callable(getattr(value, "unbind", None))


def describe_shapes(samples: Any, *, container_type: type | None = None) -> list[tuple[int, ...] | None]:
    """把「各流长什么样」列出来 ✓（**报错时必带** ✗ —— 不带就只能靠猜 ✓✗）。"""
    if _is_container(samples, container_type):
        return [_shape_of(stream) for stream in samples.unbind()]
    return [_shape_of(samples)]


def is_video_stream(value: Any, *, channels: int = H3_VIDEO_CHANNELS,
                    rank: int = H3_VIDEO_RANK) -> bool:
    """这条流是不是 **H3 视频流** ✓（5 维 + 24 通道 ✓）。"""
    shape = _shape_of(value)
    return bool(shape) and len(shape) == rank and shape[1] == channels


def split_video_stream(samples: Any, *, channels: int = H3_VIDEO_CHANNELS,
                       rank: int = H3_VIDEO_RANK,
                       container_type: type | None = None) -> SplitResult:
    """从（可能是容器的）潜变量里**认出视频流** ✓ ⇒ :class:`SplitResult` ✓。

    ⚠️ **裸张量**：直接当视频流用 ✓（但**形状不对就报错** ✗ —— 拿音频张量当视频解是灾难 ✓✗）；
    ⚠️ **容器**：恰好一条视频流才行 ✓ —— 0 条 ⇒ 报错（印出各流形状 ✓）、≥2 条 ⇒ 报错（**歧义** ✓✗）。
    """
    if not _is_container(samples, container_type):
        if not _is_tensorish(samples):
            raise LatentContainerError(
                f"看不出这是什么 ✗：既不是容器（缺 ``unbind`` / ``is_nested`` ✓）"
                f"也不是张量（缺 ``shape`` ✓）—— 收到 {type(samples).__name__} ✓")
        if not is_video_stream(samples, channels=channels, rank=rank):
            raise LatentContainerError(
                f"裸张量不是 H3 视频流 ✗：形状 {_shape_of(samples)} ✓，"
                f"而视频流必须是 {rank} 维、第 2 维 = {channels} ✓（``B×{channels}×T×H×W`` ✓）"
                f" —— 拿别的流当视频解会得到一堆噪声 ✓✗")
        return SplitResult(video=samples, streams=None, index=None, container=None)

    streams = tuple(samples.unbind())
    found = [index for index, stream in enumerate(streams)
             if is_video_stream(stream, channels=channels, rank=rank)]
    shapes = [_shape_of(stream) for stream in streams]
    if not found:
        raise LatentContainerError(
            f"容器里没有 H3 视频流 ✗（要 {rank} 维、第 2 维 = {channels} ✓）—— "
            f"实际各流形状：{shapes} ✓。⚠️ 本仓**不猜**哪条是视频 ✗（挑错 = 把音频当画面解 ✓✗）")
    if len(found) > 1:
        raise LatentContainerError(
            f"容器里有 {len(found)} 条形状都像 H3 视频流 ✗（下标 {found} ✓）⇒ **歧义** ✗✗："
            f"各流形状：{shapes} ✓。⚠️ 不猜第一条 ✗ —— 请先确认哪条才是视频 ✓")
    return SplitResult(video=streams[found[0]], streams=streams, index=found[0],
                       container=samples)


def replace_video_stream(result: SplitResult, new_video: Any,
                         *, container_type: type | None = None) -> Any:
    """换掉视频流 ✓ ⇒ **保住容器类型** ✓（裸张量入参 ⇒ 直接返回新视频 ✓）。

    ⚠️ 只动视频那一槽 ✓（别的流**原对象**放回去 ✓）；⚠️ 重建失败 ⇒ **报错** ✗ 不退化 ✗✗。
    """
    if result.streams is None:
        return new_video
    rebuilt = list(result.streams)
    if result.index is None or not 0 <= result.index < len(rebuilt):
        raise LatentContainerError(
            f"视频流下标 {result.index} 不在容器范围内 ✗（共 {len(rebuilt)} 条流 ✓）—— "
            f"别拿过期/别处的 SplitResult 来换流 ✓✗")
    rebuilt[result.index] = new_video
    attempts: list[tuple[str, Any]] = [("原类型", type(result.container).__call__)]
    if container_type is not None and container_type is not type(result.container):
        attempts.append(("声明的容器类型", container_type))
    errors: list[str] = []
    for label, factory in attempts:
        try:
            return factory(rebuilt)
        except Exception as err:  # noqa: BLE001 —— 各家容器构造契约不同 ✓ 只记不吞 ✗
            errors.append(f"{label}（{getattr(factory, '__self__', factory)} ✓）: "
                          f"{type(err).__name__}: {err}")
    raise LatentContainerError(
        f"换流失败 ✗：容器类型 {type(result.container).__name__} 不支持这样重建 ✓（试过 "
        f"{len(attempts)} 种构造 ✓）。⚠️ **不静默退化成 list** ✗✗ —— 那会让下游炸得莫名其妙 ✓；"
        f"详情：{errors}")
