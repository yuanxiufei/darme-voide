"""自检：**编码侧**（像素 ⇒ 潜变量 ✓ 2026-11-01）。

## 为什么单独一组

解码侧的口径对账已经有人守 ✓（`engine_dit_build_test.py` ⑦′ ✓）；**编码侧一直是空的** ✓✗：
`latent_formats.process_in` 写好了却**全仓零调用** ✓，于是三种 VAE 的 ``encode`` 返回值
（H3 张量·自带归一化 / Flux 二元组·无缩放 / Wan 张量·无缩放 / 参考 VAE·要另过
``normalize_latents``）被**同一行裸调用** consumers 混着用 ✓✗：

* Wan 族 ⇒ ``AttributeError: 'WanVAE' object has no attribute 'encode'``（2026-11-01 前**这个方法不存在** ✓）；
* Flux 族 ⇒ ``(mean, logvar)`` 元组被当张量 ⇒ ``'tuple' object has no attribute 'shape'``；
* 参考 VAE ⇒ 拿到**未归一化**潜变量 ⇒ 首帧条件差一个量级、**形状全对、不报错** ✓✗。

判据四类，都是**可判真假的事实** ✓：

* **算术类**：用**恒等桩**把 ``process_in`` / ``process_out`` 的互逆关系钉死 ✓
  （漏了不报错、只偏色 ✓ ⇒ 只能这么验 ✓）；
* **形状类**：Wan 编码 5 维进 5 维出 ✓、4 维进 4 维出 ✓、通道 = ``z_dim`` ✓；
* **错配类**：``family`` 填成别的一族 ⇒ 当场人话 ✓（不许等广播时静默错 ✗）；
* **接线类**：`torch_backend` 里 ``self._vae.encode(`` **只许出现在** :meth:`_encode_pixels` 内 ✓
  （散在外面的每一处都是一颗静默雷 ✓）。

⚠️ 没装 torch ⇒ 显式 SKIP ✓（**没跑 ≠ 绿** ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_encode_test.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
ENGINE = BACKEND_PY / "app" / "services" / "engine"
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import latent_formats as lf  # noqa: E402
from app.services.engine import sdxl_vae as sdxl_mod  # noqa: E402
from app.services.engine import vae_flux as flux_mod  # noqa: E402
from app.services.engine import vae_wan as wan_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def src_backend() -> str:
    return (ENGINE / "torch_backend.py").read_text(encoding="utf-8")


def _small_wan() -> Any:
    """最小的 Wan VAE（``z_dim=4`` ✓ —— 通道少 ⇒ CPU 上也秒出 ✓）。"""
    from tests.engine_vae_wan_test import SMALL  # noqa: PLC0415

    return wan_mod.build(SMALL).double().eval(), SMALL


def _has_torch() -> bool:
    """``vae_wan`` **硬 import torch** ✓（没有 ``has_torch()`` 那种软降级 ✓）⇒ 这里判导入本身 ✓。"""
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def case_wan_encode_shape() -> None:
    """① 形状类：``Encoder3d`` 真前向 ✓、``WanVAE.encode`` 5 维/4 维都认 ✓。"""
    if not _has_torch():
        skip("没装 torch ⇒ Wan 编码形状不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    module, spec = _small_wan()
    torch.manual_seed(0)
    frames = torch.randn(1, 3, 1, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        encoded = module.encode(frames)
    want = (1, int(spec.z_dim), 1, 16 // int(spec.scale), 16 // int(spec.scale))
    check(f"① Wan 编码 {tuple(encoded.shape)} ✓（应 {want} ✓）", tuple(encoded.shape) == want,
          (tuple(encoded.shape), want))
    with torch.no_grad():
        single = module.encode(frames.squeeze(2))
    check("①′ 4 维进 ⇒ 4 维出 ✓（别自己多加/少一维 ✗）",
          single.ndim == 4 and int(single.shape[1]) == int(spec.z_dim), tuple(single.shape))
    check("①″ `Encoder3d` 有 `forward` ✓（2026-11-01 前只有 `__init__` ⇒ 编码必 AttributeError ✗）",
          callable(getattr(wan_mod.Encoder3d, "forward", None)))


def case_norm_symmetry() -> None:
    """② 算术类：``encode_latents`` = ``process_in(encode(x))`` ✓、与 ``decode_latents`` 互逆 ✓。"""
    if not _has_torch():
        skip("没装 torch ⇒ 归一化算术不跑 ✓")
        return
    import torch  # noqa: PLC0415

    class _Identity:
        """编码/解码都原样返回 ⇒ 只剩下**缩放**这一层可看 ✓。"""

        shape = type("S", (), {"z_dim": 4, "scale": 8, "patch_size": 1})()

        def encode(self, frames: Any) -> Any:
            return frames

        def decode(self, latents: Any, **_knobs: Any) -> Any:
            #: ``WanVAE.decode`` 的契约带三个 tiling 旋钮（``tile`` / ``halo`` / ``attn_chunk`` ✓，
            #: 默认全 ``None`` ⇒ 走旧路径 ✓）⇒ 桩也得收 ✓；恒等桩不做滑窗，收下即丢 ✓。
            #: ⚠️ 桩只收位置参数会让「产品扩了接口」表现为**桩报 TypeError** ✓✗ ——
            #: 那是测试跟不上契约，不是编码侧算错 ✓。
            return latents

    mean = (0.5, -0.5, 0.25, 1.0)
    std = (2.0, 0.5, 4.0, 1.5)
    frames = torch.tensor([[[[[0.3]]], [[[-0.2]]], [[[0.7]]], [[[1.1]]]]], dtype=torch.float32)
    encoded = wan_mod.encode_latents(_Identity(), frames, family="qwen_image21",
                                     mean=mean, std=std)
    want = torch.tensor([(0.3 - 0.5) / 2.0, (-0.2 + 0.5) / 0.5,
                         (0.7 - 0.25) / 4.0, (1.1 - 1.0) / 1.5], dtype=torch.float32)
    check("② ⭐ `encode_latents` 做 `(x − mean) / std` ✓（= `process_in` ✓；漏了不报错 ✓ 只是首帧不生效 ✓）",
          torch.allclose(encoded.reshape(-1), want, atol=1e-6),
          (encoded.reshape(-1).tolist(), want.tolist()))
    back = wan_mod.decode_latents(_Identity(), encoded, family="qwen_image21",
                                  mean=mean, std=std, to_unit=False)
    check("②′ 与 `decode_latents` **互逆** ✓（两边同用一张口径表 ⇒ 往返该复原 ✓）",
          torch.allclose(back.reshape(-1), frames.reshape(-1), atol=1e-6),
          (back.reshape(-1).tolist(), frames.reshape(-1).tolist()))

    #: ⚠️ 「编码侧没接」的**铁证**以前是：`process_in` 全仓零调用 ✓✗。这条钉住它必须被用上 ✓。
    consumers = sum(
        1 for path in (ENGINE / "vae_wan.py", ENGINE / "sdxl_vae.py", ENGINE / "latent_formats.py")
        if "process_in(" in path.read_text(encoding="utf-8")
    )
    check("②″ `process_in` 有真消费者 ✓（≥2：定义 + 编码入口 ✓；零调用 = 编码侧又断了 ✓✗）",
          consumers >= 2, consumers)


def case_family_mismatch() -> None:
    """③ 错配类：``family`` 与模型通道数对不上 ⇒ 当场人话 ✓（别等广播时出现莫名的形状错 ✗）。"""
    if not _has_torch():
        skip("没装 torch ⇒ 错配判据不跑 ✓")
        return
    import torch  # noqa: PLC0415

    module, spec = _small_wan()
    torch.manual_seed(0)
    frames = torch.randn(1, 3, 1, 16, 16, dtype=torch.float64)
    detail: object = None
    try:
        with torch.no_grad():
            wan_mod.encode_latents(module, frames, family="wan21")
    except ValueError as err:
        detail = str(err)
    check("③ `family` 与 ``z_dim`` 不符 ⇒ ValueError 人话 ✓（`wan21` 是 16 通道 ✓、小模型是 4 ✓）",
          isinstance(detail, str) and "16" in detail and "4" in detail, detail)


def case_flux_scale() -> None:
    """④ Flux / SDXL：``encode_latents`` 用 ``(mean − shift) * scale`` ✓、配置口径错了当场拒 ✓。"""
    if not sdxl_mod.has_torch():
        skip("没装 torch ⇒ Flux 编码算术不跑 ✓")
        return
    import torch  # noqa: PLC0415

    class _MeanIdentity:
        """``encode`` 返回 ``(mean, logvar)`` 二元组 ✓（照真 VAE 的形状 ✓），mean 恒等于输入 ✓。"""

        config = flux_mod.flux_vae_config()

        def encode(self, images: Any) -> Any:
            return images, None

    images = torch.full((1, 16, 2, 2), 0.5, dtype=torch.float32)
    out = flux_mod.encode_latents(_MeanIdentity(), images)
    want = (0.5 - 0.1159) * 0.3611
    check("④ ⭐ Flux 编码做 `(x − 0.1159) * 0.3611` ✓（= `Flux.process_in` ✓；漏了不报错 ✓ 只是条件不生效 ✓）",
          abs(float(out[0, 0, 0, 0]) - want) < 1e-5, (float(out[0, 0, 0, 0]), want))
    check("④′ 二元组**被正确拆开** ✓（直接当张量用 ⇒ `'tuple' object has no attribute ...` ✗）",
          out.ndim == 4 and int(out.shape[1]) == 16, tuple(out.shape))

    #: ⚠️ 拿 SDXL 的 ``0.13025`` 建 Flux VAE ⇒ 权重装得上、形状全对、**每张图偏色** ✓✗ ⇒ 建模型时就拦 ✓。
    bad = sdxl_mod.SdxlVaeConfig(latent_channels=16, quant_conv=False,
                                 scaling_factor=0.13025, shift_factor=0.0)
    problems = flux_mod.config_problems(bad)
    check("④″ 用 SDXL 的缩放建 Flux VAE ⇒ 两条都报（`scaling_factor` / `shift_factor`）✓",
          len(problems) == 2 and any("0.3611" in p for p in problems)
          and any("0.1159" in p for p in problems), problems)
    check("④‴ 默认 `flux_vae_config()` ⇒ 预检**空** ✓（别把自己家的默认值拦了 ✗）",
          not flux_mod.config_problems(flux_mod.flux_vae_config()))


def case_wiring() -> None:
    """⑤ 接线类：裸 ``self._vae.encode(`` **只许**在 :meth:`_encode_pixels` 里 ✓✗。"""
    source = src_backend()
    tree = ast.parse(source)
    bounds: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_encode_pixels":
            bounds.append((node.lineno, max(getattr(node, "end_lineno", node.lineno),
                                            node.lineno)))
    check("⑤ `torch_backend` 有 `_encode_pixels` ✓（统一编码口子 ✓）", bool(bounds), bounds)

    stray: list[int] = []
    seen: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "encode"):
            continue
        inner = func.value
        if not (isinstance(inner, ast.Attribute) and inner.attr == "_vae"):
            continue
        seen.append(node.lineno)
        inside = any(start <= node.lineno <= end for start, end in bounds)
        if not inside:
            stray.append(node.lineno)
    #: ⚠️ 先钉"扫描器**真看得见**" ✓：一处都扫不到 ⇒ 下面那条就是空断言 ✓✗（现在应有 2 处：
    #: H3 自带归一化 ✓ / 参考 VAE 要走 `normalize_latents` ✓）。
    check("⑤′ 扫描器**真扫得到** `self._vae.encode(` ✓（空集 ⇒ 下一条等于没判 ✗）",
          len(seen) == 2, seen)
    check("⑤′ 裸调 `self._vae.encode(` **一处都不许散在外面** ✓✗（散在外面 = 三种口径混着用 ✓）",
          not stray, stray)

    #: 首帧条件的形状自检以前用滑窗 `shape[1::2]` ✓✗ —— 注释写着"不比 T"、代码却在比 T ✓。
    check("⑤″ 首帧形状自检**逐位点名** ✓（`shape[1::2]` 会把 T 夹进去 ✓✗）",
          "for i in (1, 3, 4)" in source and "encoded.shape[1::2]" not in source)


def main() -> int:
    case_wan_encode_shape()
    case_norm_symmetry()
    case_family_mismatch()
    case_flux_scale()
    case_wiring()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    #: ⚠️ 收尾口径是**硬要求** ✗✗（`run_all.py` 抬头那条 ✓）：只认 `PASS  ` / `FAIL  ` 前缀 ✓
    #: 与 `SUMMARY: n/m passed` ✓ —— 写成「15/15 项通过」会在全量表里显示**空摘要** ✓✗
    #: （同族套件已被这条坑过 5 次 ✓，`run_all` 的账就是靠 `SUMMARY:` 行收敛项数 ✓）。
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}\n      实际: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {passed}/{len(_RESULTS)} passed"
          f"{f'（{len(_SKIPS)} 项 SKIP ✓）' if _SKIPS else ''}")
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    #: cp936 控制台下 `✓` / `✗` 会 UnicodeEncodeError 把整套炸掉 ✓✗（同族套件都自兜 ✓）。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
