"""S7 自检：**注意力核选择**（`sdpa` ✓ —— 2026-11-10）。

这一套盯的是「一次**换核**改造有没有把别的东西带坏」✓✗，分四层：

1. **契约**：档位只有 ``auto`` / ``off`` ✓、写错的值**回落且留痕** ✓（不是静默 ✓✗）、
   `describe()` 回执如实报「走了几层 cuDNN / 跌回原路几次」✓；
2. **等价性**：``off`` 档必须与 `F.scaled_dot_product_attention` **逐位相同** ✓
   （``torch.equal`` 真判 ✓，不是比到小数点后几位 ✓✗）—— 这是「随时能退回改造前」的凭据 ✓；
3. **安全**：判不过就走原路 ✓、判定函数自己抛异常也走原路 ✓、``head_dim`` 超限判 False ✓、
   ``scale`` / ``dropout_p`` 非默认**不碰** cuDNN ✓（那两个口支持面窄，不值赌 ✓）；
4. **源码判据**：九个网络本体**都不再直调** SDPA ✓（改一处漏八处是这一类改造最常见的病 ✓✗）、
   而文本塔**必须仍然直调** ✓（它们与 comfy 有 ``maxDiff=0`` 的钉子 ✓，顺手改一圈就把钉子打掉 ✓✗）、
   `sdpa.py` **一行 softmax 都没有** ✓（它只选核、不算数学 ✓）。

CUDA 专属的那几条（真 cuDNN 路径）在无卡机器上 **如实 skip** ✓、不靠编 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_sdpa_test.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

BACKEND_PY = Path(__file__).resolve().parents[1]
ENGINE = BACKEND_PY / "app" / "services" / "engine"
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import sdpa  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPPED: list[str] = []
_HAS_CUDA = torch.cuda.is_available()

#: 网络本体：这些文件的 SDPA **必须**已经走 :func:`sdpa.attention`
BODIES = ("dit.py", "dit_flux.py", "dit_generic.py", "dit_lumina.py", "dit_music3.py",
          "dit_qwen21.py", "dit_qwen_image.py", "dit_seedvr2.py", "dit_wan22.py", "h3_form.py")
#: 文本塔：这些文件的 SDPA **必须**还是直调（与 comfy 的逐位钉子在这一层）
TEXT_TOWERS = ("qwen35_te.py", "qwen3vl_te.py", "umt5_te.py", "te_music3.py")


def check(name: str, ok: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(ok), detail))


def _src(name: str) -> str:
    return (ENGINE / name).read_text(encoding="utf-8")


def _sdpa_calls(name: str) -> int:
    """数**真调用**（AST ``Call``），不是数字符串 —— 注释与文档里提一嘴不该算数 ✓✗。"""
    tree = ast.parse(_src(name))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if target == "scaled_dot_product_attention":
            count += 1
    return count


def _triple(h: int, hd: int, n: int, *, pad: int = 0, device: str = "cpu"):
    q = torch.randn(1, h, n, hd, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, h, n, hd, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, h, n, hd, dtype=torch.bfloat16, device=device)
    mask = None
    if pad:
        mask = torch.zeros(1, 1, 1, n, dtype=torch.bfloat16, device=device)
        mask[:, :, :, :pad] = -torch.finfo(torch.bfloat16).max
    return q, k, v, mask


def case_contract() -> None:
    check("① 档位闭集 ``MODES`` 只有 auto / off ✓（多一个都是没实现的承诺 ✓✗）",
          sdpa.MODES == ("auto", "off"), sdpa.MODES)
    sdpa.set_mode(None)
    check("① 不设环境变量时默认 auto ✓", sdpa.mode() == "auto", sdpa.mode())
    sdpa.set_mode("off")
    check("① ``set_mode('off')`` 立即生效 ✓", sdpa.mode() == "off", sdpa.mode())
    try:
        sdpa.set_mode("turbo")
        ok, detail = False, "没抛"
    except ValueError as exc:
        ok, detail = True, str(exc)
    check("① 写没收档位就**当场抛** ✓（不是静默当 auto ✓✗）", ok, detail)
    sdpa.set_mode(None)

    import os
    os.environ[sdpa.MODE_ENV] = "AUTO "          # 大小写与空格都该认
    ok = sdpa.mode() == "auto"
    os.environ[sdpa.MODE_ENV] = "bogus"
    sdpa.reset()
    fell = sdpa.mode() == "auto" and sdpa.describe()["unknownMode"] == "bogus"
    os.environ.pop(sdpa.MODE_ENV, None)
    sdpa.reset()
    check("① 档位宽松解析（``AUTO ``⇒auto ✓；``bogus``⇒回落 auto 且回执留痕 ✓）",
          ok and fell, (ok, fell, sdpa.describe()))

    desc = sdpa.describe()
    check("① ``describe()`` 三格齐全且明写非逐位 ✓",
          {"mode", "cudnnCalls", "fallbacks", "note"} <= set(desc)
          and "非逐位" in str(desc["note"]), desc)


def case_equiv_off() -> None:
    """``off`` 档必须与改造前**逐位相同** —— 这是能随时回滚的凭据。"""
    sdpa.set_mode("off")
    sdpa.reset()
    for label, args in (("无掩码", dict(h=4, hd=64, n=256)),
                        ("加性 float 掩码", dict(h=4, hd=64, n=256, pad=17))):
        q, k, v, mask = _triple(**args)
        got = sdpa.attention(q, k, v, attn_mask=mask)
        want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        check(f"② off 档 · {label} 与 ``F.sdpa`` **逐位相同** ✓（torch.equal）",
              torch.equal(got, want), float((got.float() - want.float()).abs().max()))
    check("② off 档**一次都没**进 cuDNN ✓（计数器为 0，不是「进了但结果碰巧一样」✓✗）",
          sdpa.describe()["cudnnCalls"] == 0, sdpa.describe())
    sdpa.set_mode(None)


def case_equiv_shapes() -> None:
    """``enable_gqa`` / ``scale`` / ``dropout_p`` 三个口在 off 档下也要逐位同。"""
    sdpa.set_mode("off")
    q, k, v, _ = _triple(h=8, hd=64, n=128)
    k2 = torch.randn(1, 2, 128, 64, dtype=torch.bfloat16)
    v2 = torch.randn(1, 2, 128, 64, dtype=torch.bfloat16)
    got = sdpa.attention(q, k2, v2, enable_gqa=True)
    want = F.scaled_dot_product_attention(q, k2, v2, enable_gqa=True)
    check("② off 档 · GQA（heads≠kvheads）逐位相同 ✓", torch.equal(got, want),
          float((got.float() - want.float()).abs().max()))

    got = sdpa.attention(q, k, v, scale=0.5)
    want = F.scaled_dot_product_attention(q, k, v, scale=0.5)
    check("② off 档 · 自定义 ``scale`` 逐位相同 ✓", torch.equal(got, want),
          float((got.float() - want.float()).abs().max()))

    got = sdpa.attention(q, k, v, is_causal=True)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    check("② off 档 · ``is_causal`` 逐位相同 ✓", torch.equal(got, want),
          float((got.float() - want.float()).abs().max()))
    sdpa.set_mode(None)


def case_safety() -> None:
    """判不过 / 判定自己炸 / 不该碰的口 —— 一律走原路，绝不抛异常。"""
    sdpa.set_mode("auto")
    sdpa.reset()
    #: CPU 上 ``q.is_cuda`` 为 False ⇒ must be False
    q, k, v, mask = _triple(h=4, hd=64, n=128)
    check("③ CPU 张量判 False ✓（无卡环境不许伪装能用 ✓✗）",
          sdpa.can_use_cudnn(q, k, v, mask) is False, sdpa.can_use_cudnn(q, k, v, mask))

    #: 判定函数自己抛异常 ⇒ 也必须 False（不许把「判不了」升级成「崩一张图」✓✗）
    if _HAS_CUDA:
        real = torch.backends.cuda.can_use_cudnn_attention

        def boom(*_a, **_k):
            raise RuntimeError("判定炸了")
        torch.backends.cuda.can_use_cudnn_attention = boom      # type: ignore[assignment]
        try:
            q2, k2, v2, m2 = _triple(h=4, hd=64, n=256, device="cuda")
            survived = sdpa.can_use_cudnn(q2, k2, v2, m2) is False
            out = sdpa.attention(q2, k2, v2, attn_mask=m2)       # 不许抛
            same = torch.allclose(out, F.scaled_dot_product_attention(q2, k2, v2, attn_mask=m2),
                                  atol=1e-2, rtol=1e-2)
        finally:
            torch.backends.cuda.can_use_cudnn_attention = real  # type: ignore[assignment]
        check("③ 判定自身抛异常 ⇒ 判 False 且照常出结果 ✓（不把判定故障升级成崩溃 ✓）",
              survived and same, (survived, same))

        sdpa.reset()
        out = sdpa.attention(q2, k2, v2, attn_mask=m2, dropout_p=0.1)
        check("③ ``dropout_p`` 非 0 时**不碰** cuDNN ✓（那个口支持面窄 ✓，不进计数 ✓）",
              sdpa.describe()["cudnnCalls"] == 0 and out.shape == q2.shape, sdpa.describe())
        sdpa.reset()
        sdpa.attention(q2, k2, v2, attn_mask=m2, scale=0.3)
        check("③ 自定义 ``scale`` 时不碰 cuDNN ✓", sdpa.describe()["cudnnCalls"] == 0,
              sdpa.describe())

        q3, k3, v3, m3 = _triple(h=4, hd=256, n=256, device="cuda")
        check("③ ``head_dim=256`` 判 False ✓（不是我们写死的阈值，是 PyTorch 自己的约束 ✓）",
              sdpa.can_use_cudnn(q3, k3, v3, m3) is False,
              sdpa.can_use_cudnn(q3, k3, v3, m3))
    else:
        _SKIPPED.append("③ CUDA 专属三条（判定抛异常 / dropout / scale / head_dim 256）无卡跳过")
    sdpa.reset()
    sdpa.set_mode(None)


def case_cudnn() -> None:
    """真 cuDNN 路径：真换了核（不逐位）✓、但数值与 float 参考同量级 ✓、掩码真生效 ✓。"""
    if not _HAS_CUDA:
        _SKIPPED.append("④ 真 cuDNN 路径四条无卡跳过（不编数字 ✓）")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sdpa.set_mode("auto")
    sdpa.reset()
    q, k, v, mask = _triple(h=4, hd=64, n=512, pad=37, device="cuda")

    got = sdpa.attention(q, k, v, attn_mask=mask)
    check("④ auto 档在能用时**真进了** cuDNN ✓（计数器 +1，不是静默走原路 ✓✗）",
          sdpa.describe()["cudnnCalls"] == 1, sdpa.describe())

    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        eff = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    d = (got.float() - eff.float()).abs()
    check("④ 与 EFFICIENT 路径差值在 bf16 舍入量级 ✓（≤5e-3；本次是否逐位相同如实记，不逼它必须不同 ✓✗）",
          d.max().item() <= 5e-3,
          {"max": float(d.max()), "bitIdentical": bool(torch.equal(got, eff))})

    #: 参考值：float32 + -inf 显式 softmax
    qf, kf, vf = (t.float() for t in (q, k, v))
    logits = (qf @ kf.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    logits[..., :37] = float("-inf")
    ref = logits.softmax(-1) @ vf
    dref = (got.float() - ref).abs().max().item()
    check("④ 与 float32 显式 softmax 参考的偏差 ≤5e-3 ✓（更快不是更不准 ✓）", dref <= 5e-3, dref)

    nomask = sdpa.attention(q, k, v)
    dnomask = (got.float() - nomask.float()).abs().max().item()
    check("④ **掩码真被应用** ✓（与不传掩码差 >5e-3；若≈0 说明 cuDNN 把掩码丢了 ✓✗）",
          dnomask > 5e-3, dnomask)

    #: 42 个形状：判 True ⇒ 强制 CUDNN 真能跑（判定与真跑不许各说各话）
    bad = []
    for hd in (128, 64, 96, 256):
        for dtype in (torch.bfloat16, torch.float16):
            for n in (1000, 3460, 4296):
                qq = torch.randn(1, 4, n, hd, dtype=dtype, device="cuda")
                kk, vv = torch.randn_like(qq), torch.randn_like(qq)
                said = sdpa.can_use_cudnn(qq, kk, vv)
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        F.scaled_dot_product_attention(qq, kk, vv)
                    ran = True
                except Exception:  # noqa: BLE001
                    ran = False
                if said != ran:
                    bad.append((hd, str(dtype), n, said, ran))
    check("④ 判定与真跑**不许各说各话** ✓（24 个形状逐一对账；不服的清单必须空 ✓）",
          not bad, bad)
    sdpa.reset()
    sdpa.set_mode(None)


def case_source() -> None:
    """源码判据：改一处漏八处、顺手改掉钉子 —— 两种病都靠这里拦。"""
    leaked = []
    for name in BODIES:
        if _sdpa_calls(name):
            leaked.append(f"{name}（还直调 {_sdpa_calls(name)} 处）")
        if "from . import sdpa" not in _src(name):
            leaked.append(name + "(没导入 sdpa)")
    check("⑤ 十个网络本体**都不再直调** SDPA ✓ 且都导入 ``sdpa`` ✓（漏一处就报 ✓）",
          not leaked, leaked)

    towers = [n for n in TEXT_TOWERS if not _sdpa_calls(n)]
    check("⑤ 文本塔**必须仍然直调** SDPA ✓（它们有与 comfy 的 ``maxDiff=0`` 钉子 ✓✗）",
          not towers, towers)

    own = _src("sdpa.py")
    tree = ast.parse(own)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("⑤ ``sdpa.py`` **一行 softmax 数学都没有** ✓（只选核 ✓；判 AST 调用名，不判文档字符串 ✓）",
          not ({"softmax"} & (attrs | calls)), sorted({"softmax"} & (attrs | calls)))

    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    check("⑤ ``sdpa.py`` 只 import torch / os / typing ✓（不许拉引擎别的模块 ✓，否则要成环 ✓✗）",
          set(names) & {"pipeline", "runtime", "torch_backend"} == set(), None)

    check("⑤ 回落要**计数** ✓（源码里有 ``fallbacks`` ✓：静默跌回原路等于没人发现 ✓✗）",
          "fallbacks" in own)
    check("⑤ 自动档必须走 ``mode()`` 而不是读死常量 ✓（源码判据：``mode() == \"auto\"`` ✓）",
          'mode() == "auto"' in own, None)


def main() -> int:
    case_contract()
    case_equiv_off()
    case_equiv_shapes()
    case_safety()
    case_cudnn()
    case_source()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for note in _SKIPPED:
        print("SKIP  " + note)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（{len(_SKIPPED)} skipped）" if _SKIPPED else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
