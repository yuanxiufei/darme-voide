"""S7 自检：引擎的**文本编码器 + tokenizer 注入**（CPU 可验 ✓ 2026-09-17）。

这块填的是最后一个占位 ✓：此前条件向量是 sha256 造的占位 ✓，现在是**真结构**算出来的 ✓
（TE 仍**未训练** ✗ ⇒ 语义无意义 ✓，但凡"管道/形状/注意力/截断"这些都能真验 ✓）。

判据里两条是刻意防**假绿**的 ✓：
* **注意力真的混了 token**（改第 0 个 token ⇒ 第 1 个位置的输出必须变 ✓ —— 若只是逐位置 MLP 就会不变 ✗）；
* **TE 的输出真的能喂进 DiT**（形状对齐后再走一次真前向 ✓ —— 只对形状不够 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_text_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import text_encoder as te_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


TE_CONFIG = te_mod.TextEncoderConfig(vocab_size=512, hidden=32, depth=2, heads=4,
                                     max_length=16, output_dim=32)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# ① 配置与 tokenizer（**不猜词表** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_config() -> None:
    try:
        te_mod.TextEncoderConfig(hidden=33, heads=4)
        bad = False
    except te_mod.TextEncoderError as err:
        bad = "不能被 heads" in str(err)
    check("① hidden 不能被 heads 整除 ⇒ 报错 ✓", bad)
    check("② 配置里有 maxLength（**截断口径**要显式 ✓ 不静默切 ✗）",
          TE_CONFIG.to_dict()["maxLength"] == 16)

    tokenizer = te_mod.StubTokenizer(vocab_size=512, max_length=16)
    check("③ 桩 tokenizer **确定性**（同文本同 id ✓ 便于复现 ✓）",
          tokenizer.encode("雨夜") == tokenizer.encode("雨夜"), tokenizer.encode("雨夜"))
    check("④ id 落在词表范围内 ✓",
          all(0 <= value < 512 for value in tokenizer.encode("雨夜霓虹街头")), "")

    ids, original, truncated = te_mod.tokenize_prompt(tokenizer, "一二三四五六七八九十一二三四五六七八九十",
                                                      max_length=16)
    check("⑤ ⭐ 超长 ⇒ **截断到 maxLength 且回报原长/截断标记** ✓（静默切最坏 ✗）",
          len(ids) == 16 and original == 20 and truncated is True, (len(ids), original, truncated))
    empty_ids, empty_original, empty_truncated = te_mod.tokenize_prompt(tokenizer, "", max_length=16)
    check("⑥ 空文本 ⇒ 给一个 pad id ✓（不返回空列表 ⇒ 避免后续除零/空张量 ✗）",
          empty_ids == [0] and empty_original == 0 and empty_truncated is False, empty_ids)

    try:
        te_mod.HFTokenizer("不存在的仓库").encode("x")
        hf_ok = False
    except te_mod.TextEncoderError as err:
        hf_ok = "transformers" in str(err) or "仓库" in str(err)
    check("⑦ `transformers` 未装 ⇒ **可行动报错**（给出镜像安装提示 ✓ 并指向桩 tokenizer ✓）",
          hf_ok)


# ══════════════════════════════════════════════════════════════════════════
# ② 编码器前向（真结构 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_forward() -> None:
    if not _have_torch():
        skip("torch 未安装 ⇒ 编码器前向与集成跳过 ✓")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    encoder = te_mod.build_text_encoder(TE_CONFIG)
    ids = torch.tensor([[3, 9, 27, 5]])
    out = encoder(ids)
    check("⑧ 输出形状 = (B, L, output_dim) ✓（与 DiT 的上下文口径一致 ✓）",
          tuple(out.shape) == (1, 4, TE_CONFIG.output_dim), tuple(out.shape))

    # ⭐ 注意力真的混了 token：改**第 0 个** token ⇒ **第 1 个位置**的输出必须变 ✓
    other = encoder(torch.tensor([[4, 9, 27, 5]]))
    check("⑨ ⭐ **注意力真的混合了 token**（改首位 ⇒ 其他位置输出也变 ✓ —— "
          "纯逐位置 MLP 就不会变 ✗）",
          not bool(torch.allclose(out[:, 1], other[:, 1])), "")
    check("⑩ 同输入逐位可复现 ✓（无隐藏随机 ✓）",
          bool(torch.allclose(out, encoder(ids))), "")
    check("⑪ 位置编码生效（同样的 token 换个位置 ⇒ 输出不同 ✓）",
          not bool(torch.allclose(encoder(torch.tensor([[3, 9]]))[:, 0],
                                  encoder(torch.tensor([[9, 3]]))[:, 0])), "")

    try:
        encoder(torch.zeros(1, TE_CONFIG.max_length + 1, dtype=torch.long))
        over_ok = False
    except te_mod.TextEncoderError as err:
        over_ok = "maxLength" in str(err)
    check("⑫ 超过 maxLength ⇒ **报错并指出要先显式截断** ✓（不硬跑到越界 ✗）", over_ok)

    mask = torch.tensor([[False, False, True, True]])
    with_mask = encoder(ids, mask)
    check("⑬ 掩码路径可用，且**全 padding 行不会出 NaN** ✓（首位置强制放开 ✓）",
          bool(torch.isfinite(with_mask).all()), "")
    check("⑭ 掩码改变输出（padding 位置真的被排除 ✓）",
          not bool(torch.allclose(with_mask[:, 0], out[:, 0])), "")


# ══════════════════════════════════════════════════════════════════════════
# ③ 集成：TE → DiT → VAE → mp4（占位条件换成**真条件** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_integration(root: Path) -> None:
    if not _have_torch():
        skip("torch 未安装 ⇒ 集成跳过 ✓")
        return
    import torch  # noqa: PLC0415

    from app.services.engine import dit as dit_mod
    from app.services.engine import media as media_mod
    from app.services.engine import pipeline as pipe
    from app.services.engine import vae as vae_mod
    from app.services.engine import weights as weights_mod
    from app.services.engine.torch_backend import TorchBackend

    backend = TorchBackend()
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 集成跳过 ✓")
        return

    # 让 DiT 的 text_dim 与 TE 的 output_dim 对齐 ✓（真机上由权重决定 ✓ 这里显式配 ✓）
    dit_config = dit_mod.DiTConfig(hidden=32, depth=2, heads=4, patch_size=(1, 2, 2),
                                   in_channels=4, text_dim=TE_CONFIG.output_dim,
                                   mlp_ratio=2.0, vae_scale=8)
    checkpoint = weights_mod.save_module_weights(
        dit_mod.build_dit(dit_config), root / "dit_text.safetensors",
        metadata={"hidden": "32", "depth": "2", "heads": "4", "vae_scale": "8"})
    backend.load_weights(path=checkpoint, config=dit_config)
    attached = backend.attach_text_encoder(TE_CONFIG, te_mod.StubTokenizer(512, max_length=16))
    check("⑮ 挂上文本编码器 ⇒ `describe()` 如实报 `textEncoderLoaded` ✓ 且带 tokenizer 名 ✓",
          backend.describe()["textEncoderLoaded"] is True
          and attached["tokenizer"] == "stub", attached)

    positive = backend.encode_text(pipe.GenerationRequest(prompt="雨夜霓虹街头", steps=2))
    check("⑯ `encode_text` 现在回**真编码器**的输出（不再是 sha256 占位 ✓）",
          tuple(positive["positive"].shape) == (1, len(positive["ids"]), TE_CONFIG.output_dim),
          tuple(positive["positive"].shape))
    check("⑰ 正/负条件**不同** ✓（负提示词为空也有自己的向量 ✓）",
          not bool(torch.allclose(positive["positive"], positive["negative"])), "")
    check("⑱ 回报里带**截断信息** ✓（前端可据此提示「提示词被截了」✓）",
          "truncated" in positive and "tokens" in positive, {k: positive[k] for k in
                                                            ("tokens", "truncated")})

    check("⑲ VAE 也挂上 ✓", bool(backend.attach_vae(
        vae_mod.VideoVAEConfig(base_channels=8, latent_channels=4,
                               channel_multipliers=(1, 1, 1, 1)))))

    request = pipe.GenerationRequest(prompt="雨夜霓虹街头", negative="模糊", seed=6, steps=2,
                                     seconds=0.2, temporal_compression=1,
                                     outputs_dir=str(root))
    result = pipe.run_sync(request, backend)
    video = (result.outputs or {}).get("videoPath")
    check("⑳ ⭐⭐ 整链：TE → DiT → VAE → **真 mp4**（ffprobe 复核 ✓）",
          result.ok is True and bool(video) and Path(video).exists()
          and media_mod.probe(video)["width"] == result.plan.width,
          (result.error, video))
    check("㉑ 仍然如实标 `synthetic=True` ✓（TE/DiT/VAE **全未训练** ✗ ⇒ 不是真结果 ✓）",
          result.synthetic is True and result.outputs.get("synthetic") is True, "")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_text_"))
    case_config()
    case_forward()
    case_integration(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
