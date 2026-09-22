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

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def _raises(call: Any, needle: str | None = None) -> str | None:
    """**能触发**的反向证明 ✓：调它、看报错里有没有那个词 ✓（没报错 ⇒ None ⇒ 断言红 ✓）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 就是来看它报什么的 ✓
        return str(err) if needle is None or needle in str(err) else None
    return None


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

    # ⚠️ **两种世界都成立** ✓（本仓纪律：断言不许依赖「真机装没装」✗）：
    #    未装 ⇒ 报「未安装 transformers ✓ + 镜像提示 ✓ + 指向 stub / 自研 BPE ✓」；
    #    已装但仓库名坏 ⇒ 报「装载失败 ✓ + 核对仓库名 ✓ + 指向自研 BPE ✓」（**外部库的裸异常
    #    不许漏出去** ✗ —— 2026-09-20 装上之后实测：HF 抛的是 `OSError: Repo id must use …` ✓✗）。
    hf_message = _raises(lambda: te_mod.HFTokenizer("不存在的仓库").encode("x"))
    check("⑦ 好坏两种世界都给**可行动**的 `TextEncoderError` ✓（安装提示 / 仓库核对 ✓，"
          "且都指向 `stub` 或**自研 BPE** ✓ —— 不把外部库的裸异常漏出来 ✗）",
          hf_message is not None
          and ("transformers" in hf_message or "装载 HF 分词器失败" in hf_message)
          and ("StubTokenizer" in hf_message or "自研 BPE" in hf_message),
          hf_message)


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

    # ── ⭐ 真词表（**本仓自研 BPE** ✓ 2026-09-20）────────────────────────────
    #    ⚠️ 这里用**合成词表**验接线 ✓（真词表随权重来 ✓ —— 自检不依赖它 ✓）。
    from app.services.engine import tokenizer_bpe as tbp  # noqa: PLC0415
    space = tbp.bytes_to_unicode()[32]           # ⚠️ 空格在 byte 映射里不是 `" "` ✓
    vocab = {char: index for index, char in enumerate(sorted(tbp.bytes_to_unicode().values()))}
    extra = 400
    for token in ("he", "ll", "hell", "hello",
                  f"{space}w", f"{space}wo", f"{space}wor", f"{space}worl", f"{space}world"):
        vocab[token] = extra
        extra += 1
    merges = ["h e", "l l", "he ll", "hell o",
              f"{space} w", f"{space}w o", f"{space}wo r", f"{space}wor l", f"{space}worl d"]
    vocab["<|endoftext|>"] = 999                 # 特殊符 id 故意**很大** ✓（守嵌入表尺寸 ✓）
    vocab_dir = root / "tokenizer"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    (vocab_dir / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "added_tokens": [{"id": 999, "content": "<|endoftext|>", "special": True}],
        "pre_tokenizer": {"type": "ByteLevel"},
    }), encoding="utf-8")

    real = backend.attach_text_encoder(tokenizer_path=str(vocab_dir))
    check("⑲⁰ ⭐ **真词表**（HF 格式 ✓）直接挂上 ✓：`tokenizer=\"bpe\"` ✓ "
          "（不再是 `stub` 假桩 ✗ —— `describe()` 一眼可辨 ✓）且词表口径随报告给出 ✓",
          real["tokenizer"] == "bpe"
          and real["tokenizerDetail"]["vocabSize"] == len(vocab)
          and real["tokenizerDetail"]["backend"] == "own-bpe"       # ⭐ 走的是**自研**那条 ✓
          and real["tokenizerDetail"]["detail"]["merges"] == len(merges)
          and backend.describe()["tokenizer"] == "bpe", real)
    check("⑲⁰′ ⭐ 不给 config 时 TE 的 `vocab_size` **自动取 `required_vocab_size`** ✓"
          "（含特殊符 id 999 ⇒ ≥ 1000 ✓ —— 拿 `vocab_size`=262 建表会**越界** ✓✗）",
          real["config"]["vocabSize"] >= 1000, real["config"])

    encoded = backend.encode_text(pipe.GenerationRequest(prompt="hello world", steps=2))
    check("⑲¹ ⭐ 走**真 BPE**：`hello world` ⇒ 2 个 token ✓（合并真的发生 ✓ 不是按字符 11 个 ✗）",
          encoded["tokens"] == 2 and tuple(encoded["positive"].shape) == (1, 2, TE_CONFIG.output_dim),
          (encoded["tokens"], tuple(encoded["positive"].shape)))

    check("⑲² 嵌入表**装不下**分词器（含特殊符 id 999 ✓）⇒ **明确报错** ✓"
          "（不静默截断 ✗ —— 越界往往到真跑才炸 ✓）",
          _raises(lambda: backend.attach_text_encoder(
              te_mod.TextEncoderConfig(vocab_size=100, output_dim=TE_CONFIG.output_dim),
              tokenizer_path=str(vocab_dir)), "越界") is not None, None)

    # ⭐⭐ 2026-09-22 补：**显式给不一致的 TE 配置 ⇒ 当场报** ✓（DiT 那侧同样守 ✓ —— 期望值来自
    #    `DiTConfig.text_dim` ✓）。此前只在**编码之后**由形状错兜住 ✗：报的是「输出应为 (1,L,n) ✓
    #    收到 (1,L,m) ✗」⇒ **指不到**"是把 TE 配错了" ✓✗。
    #    ⚠️ 顺带钉住**先核后改** ✓：核不过时**先前挂好的** TE 必须原封不动 ✓（不是被清掉 ✗✗）。
    mismatch_reason = _raises(lambda: backend.attach_text_encoder(
        te_mod.TextEncoderConfig(vocab_size=100, output_dim=TE_CONFIG.output_dim * 2)), "对不上")
    check("⑲³ ⭐⭐ 显式给错的 `output_dim` ⇒ 挂载**当场报** ✓ 且点名 `condition_proj` ✓；"
          "⚠️ **先核后改** ✓ —— 先前挂好的 TE **原封不动**（`textEncoderLoaded` 仍 True ✓ 且仍是对的那个 ✓）",
          mismatch_reason is not None and "condition_proj" in mismatch_reason
          and backend.describe()["textEncoderLoaded"] is True
          and backend.describe()["textEncoderCheck"]["agrees"] is True,
          (mismatch_reason, backend.describe()["textEncoderCheck"]))

    check("⑲ VAE 也挂上 ✓", bool(backend.attach_vae(
        vae_mod.VideoVAEConfig(base_channels=8, latent_channels=4,
                               channel_multipliers=(1, 1, 1, 1)))))

    # ⚠️⚠️ 2026-09-22 补这道门 ✗：这一族套件（`engine_io_test` / `engine_dual_stream_test` ✓）原本就都有
    #    「ffmpeg 可用」的显式门 ✓，**只有本套没有** ✗ ⇒ 在缺 ffmpeg 的环境里 **⑳/㉑ 会直接报两条红** ✗✗
    #    （报的是 `找不到 ffmpeg ✗` ⇒ 看着像代码坏了 ✓✗，实际只是本机没装 / 没进当前进程的 PATH ✓）。
    #    ⇒ 按同族口径补 ✓：**可用性本身红一条**（ffmpeg 是硬依赖 ✓ 该被看见 ✓），
    #      依赖它的两条**跳过**（SKIP ✓ 不是「通过」✗ —— 本仓规矩：没跑 ≠ 绿 ✓）。
    #    ⚠️ 门必须在 `run_sync` **之前** ✗（跑完再判，链已经拿 MediaError 炸过一遍了 ✓✗）。
    check("⑳前置 本机 ffmpeg/ffprobe 可用 ✓（下面两条的硬依赖 ✓ —— 缺了它们只能跳过 ✗）",
          media_mod.have_ffmpeg(), media_mod.ffmpeg_version())
    if not media_mod.have_ffmpeg():
        skip("本机没有可用的 ffmpeg/ffprobe ⇒ ⑳/㉑ **没跑** ✓（不是通过 ✗）—— "
             "⚠️ 确认装过就 `where.exe ffmpeg` 复核当前进程能否解析 ✓"
             "（WinGet 的 alias 是重解析点 ✓ 可能 `lexists=True` 但 `exists=False` ✓✗）")
        return

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
