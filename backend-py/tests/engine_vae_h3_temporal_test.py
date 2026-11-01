"""自检：**H3 视频 VAE 的时间分块解码** ✓ —— 这段算法"形状全对但不报错"的重灾区（2026-11-01）。

为什么单开一份 ✗（`engine_vae_h3_test.py` 已经验过几何 ✓）：那一份只验到**算式** ✓，
① 非门控档只比数字 ✓ 从不真跑 `decode_temporal` ✗；
② 门控档只喂 **5 帧** ✓ ⇒ 2 个潜格 ⇒ 恰好落进「不足一段就补一段」那条特例 ✓
   ⇒ **跨段拼接（dec_overlap 混进下一段）一次都没被执行过** ✗✗。

而这段算法最险的地方正是跨段那一步 ✓：算错 `frame_pre_padding` / `frame_overlap` / 少写一段，
**所有形状都还是"对的"** ✓（画布是按同一套算式算的 ✓），只是画面在段边界上错位 ✓✗。

所以本份用**桩解码器**（不装 5 GiB 权重 ✓）把"每个像素帧来自哪个潜格"直接编码进像素值 ✓，
逐帧对账 ✓：
* 画布有没有**真的写满**（少写 ⇒ 尾部是 `torch.empty` 的垃圾 ✓ 不会报错 ✗）；
* 帧序有没有**倒退**（拼接错位 ⇒ 帧号回跳 ✓）；
* 段边界是不是**正好** `frame_overlap` 帧在混合 ✓（数错 ⇒ 该是整数的地方分数化 ✓ 反之亦然）；
* 分段计划与**参考实现逐字转写**的 oracle 在 1..60 个潜格上逐值相同 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_vae_h3_temporal_test.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import vae_h3 as vae_h3_mod  # noqa: E402

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
# ① oracle：把参考实现那两段算式**逐字转写**成纯函数 ✓（唯一目的就是"换个手写一遍"✓）
#    ⚠️ 转写**不 import** 被测模块 ✗ —— 否则等于拿自己验自己 ✓✗
# ─────────────────────────────────────────────────────────────────────────────
def _oracle_params(clip_length: int, temporal_scale: int, token_drop: int) -> dict[str, int]:
    chunk = math.ceil(clip_length / temporal_scale)
    return {
        "chunk": chunk,
        "overlap": (-token_drop) % chunk,
        "pre": (-clip_length) % temporal_scale,
        "chunkDec": chunk * temporal_scale,
        "splits": (1 if token_drop > 0 else 0) + 1,
    }


def _oracle_chunks(z_len: int, p: dict[str, int], token_drop: int) -> tuple[int, int]:
    """参考 ``_decode_temporal_chunks`` ✓ —— 补格数 / 段数。"""
    pseudo = z_len + token_drop
    pad = (-pseudo) % p["chunk"]
    pseudo += pad
    chunks = pseudo // p["chunk"] - (1 if token_drop > 0 else 0)
    if chunks < 1:
        pad += p["chunk"]
        chunks += 1
    return pad, chunks


def _oracle_pad_frames(z_len: int, pad_tokens: int, p: dict[str, int],
                       clip_length: int, temporal_scale: int) -> int:
    """参考 ``_decode_temporal_pad_frames`` ✓ —— 补出来的潜格一共多几帧。"""
    if pad_tokens <= 0:
        return 0
    intra_tail = clip_length % temporal_scale
    if intra_tail == 0:
        return pad_tokens * temporal_scale
    before = z_len - pad_tokens
    return sum(intra_tail if (before + k) % p["chunk"] == 0 else temporal_scale
               for k in range(pad_tokens))


def _oracle_frames(z_len: int, pad: int, chunks: int, p: dict[str, int],
                   clip_length: int, temporal_scale: int) -> int:
    """参考 ``_decode_temporal_frame_plan`` ✓ —— 实际会写出几帧。"""
    total = 0
    tail = 0
    for i in range(chunks):
        start = i * p["chunk"]
        end = start + p["chunk"] + p["overlap"]
        clip_tokens = max(0, min(end, z_len) - min(start, z_len))
        clip_frames = clip_tokens * temporal_scale
        for j in range(p["splits"]):
            f_start = j * p["chunkDec"]
            f_end = min(f_start + p["chunkDec"], clip_frames)
            got = max(0, f_end - f_start - p["pre"])
            if j == 0:
                total += got
            else:
                tail = got
    total += tail
    return total - _oracle_pad_frames(z_len, pad, p, clip_length, temporal_scale)


def _oracle(latents: int, clip_length: int, temporal_scale: int,
            token_drop: int) -> dict[str, int]:
    #: ⚠️ ``T_lat == 1`` 是**特例** ✓（参考 ``decode_output_shape`` 明写 ``if t == 1: frames = 1`` ✓）——
    #: oracle 漏掉这条会在"单潜格"上误报 ✓✗（本份第一版就是这么误报的 ✓）。
    p = _oracle_params(clip_length, temporal_scale, token_drop)
    pad, chunks = _oracle_chunks(latents, p, token_drop)
    frames = 1 if latents == 1 else _oracle_frames(
        latents + pad, pad, chunks, p, clip_length, temporal_scale)
    return {"padTokens": pad, "chunks": chunks, "frames": frames}


def case_plan_vs_oracle() -> None:
    """分段计划 / 帧数换算 ⇄ oracle ✓（换 4 套时间口径 ✓ —— 单套口径验不出"抄错常量"✓✗）。"""
    combos = [(17, 4, 3), (20, 4, 3), (17, 4, 0), (33, 8, 5)]
    bad: list[Any] = []
    for clip_length, temporal_scale, token_drop in combos:
        for latents in range(1, 61):
            want = _oracle(latents, clip_length, temporal_scale, token_drop)
            got_plan = vae_h3_mod.decode_temporal_plan(
                latents, clip_length=clip_length, temporal_scale=temporal_scale,
                token_drop=token_drop)
            got_frames = vae_h3_mod.frames_for_latents(
                latents, clip_length=clip_length, temporal_scale=temporal_scale,
                token_drop=token_drop)
            if (got_plan["padTokens"], got_plan["chunks"], got_frames) != (
                    want["padTokens"], want["chunks"], want["frames"]):
                bad.append(((clip_length, temporal_scale, token_drop), latents,
                            (got_plan["padTokens"], got_plan["chunks"], got_frames),
                            (want["padTokens"], want["chunks"], want["frames"])))
    check("计划: 1..60 潜格 × 4 套口径上，补格数/段数/出帧数与参考转写**逐值相同** ✓",
          not bad, bad[:3])

    # ⚠️ 特例单列 ✗：``T_lat`` 小到装不满一段时**要额外套一段**（否则段数=0 ⇒ 一帧都不出 ✓
    #    画布却已按"1 帧"分好 ⇒ 静默出黑片 ✓✗）。
    tiny = [(_ := n, vae_h3_mod.decode_temporal_plan(n)["chunks"]) for n in (1, 2, 3)]
    check("计划: 潜格 1/2/3 都**至少有一段**（段数 0 ⇒ 静默出空片，形状还不报错 ✗）",
          all(chunks >= 1 for _n, chunks in tiny), tiny)


def case_frame_overlap() -> None:
    """重叠帧数必须按 **`frame_pre_padding`** 算 ✓（写成 `token_drop` 在 clip=17 时**一模一样** ✓✗）。

    ⚠️ 这就是"改一个字、全绿、只有画面错位"那一类 ✓：``clip_length=17`` 时 ``(-17)%4 == 3 == token_drop`` ✓
    ⇒ 两种写法算出的重叠帧**都是 5** ✓；换 ``clip_length=20`` 才分岔 ✓（正确 8 ✓／错误 5 ✓）
    ⇒ 只有真换长度来验，才抓得住 ✓。
    """
    #: ⚠️ ``temporal_scale`` **不是**构造参数 ✗（这是个常量 4 ✓，参考实现里就是写死的 ✓）
    cases = [(17, 3, 5), (20, 3, 8), (33, 5, 13), (17, 0, 0)]
    got = []
    for clip_length, token_drop, want in cases:
        config = vae_h3_mod.H3VideoVAEConfig(clip_length=clip_length, token_drop=token_drop)
        got.append((clip_length, token_drop, config.frame_overlap, want))
    check("重叠帧数: 按 `frame_pre_padding` 算（写 `token_drop` ⇒ clip=17 全绿、clip=20 少 3 帧 ✗）",
          all(actual == want for _clip, _drop, actual, want in got), got)


# ─────────────────────────────────────────────────────────────────────────────
# ② 桩解码器：把「像素帧来自哪个潜格」编进像素值 ✓（不装权重 ✓ 也能验拼接 ✓）
# ─────────────────────────────────────────────────────────────────────────────
def _stub_module(torch: Any, config: Any) -> Any:
    """**meta 设备**上空建结构 ✓（5 GiB 权重一个字节都不占 ✓），再把两头换成桩 ✓。

    ⚠️ 必须真用被测类 ✗ 不许在测试里另写一遍算法 ✓ —— 要验的就是 ``decode_temporal`` 本体 ✓。
    """
    with torch.device("meta"):
        module = vae_h3_mod._impl().VideoVAE(config)  # noqa: SLF001 —— 故意取被测实现 ✓

    state: dict[str, Any] = {"wrote": 0, "calls": 0, "truncated": None, "lateWrite": None}

    def stub_decode(z: Any) -> Any:
        """潜格 ⇒ 像素；通道 0 = **来源潜格号** ✓、通道 1 = **本次调用流水号** ✓。

        ⚠️ 为什么非要第二路 ✗：真解码器在段边界**会重新起步** ✓（因果网络从零开始），同一个潜格在两段里
        出来的像素**本来就不一样** ✓；而"潜格号"这种确定性映射两边**完全相同** ✓ ⇒ 只靠它看混合
        ⇒ 混出来的还是整数 ✓ **永远看不出 blend 有没有生效** ✗✗（本份第一版就栽在这 ✓）。
        """
        state["calls"] += 1
        stamp = float(state["calls"] - 1)
        batch, _channels, frames, height, width = z.shape
        ids = z[:, 0, :, 0, 0]  # [B,T] —— 来源潜格号 ✓
        #: 一个潜格 ⇒ 连着的 4 个像素帧 ✓（复制的轴是**帧**这一层 ✓，别先 view 再 expand ✗）
        rep = ids.view(batch, frames, 1).expand(batch, frames, 4).reshape(
            batch, 1, frames * 4, 1, 1)
        out = rep.expand(batch, 3, frames * 4, height * 16, width * 16).contiguous()
        out[:, 1].fill_(stamp)
        out[:, 2].zero_()
        return out

    def stub_finalize(part: Any) -> Any:
        return part.float()

    module._adaptive_decode = stub_decode  # noqa: SLF001
    module._finalize_pixels = stub_finalize  # noqa: SLF001
    original_write = module._write_part  # noqa: SLF001

    def counting_write(canvas: Any, part: Any, write_pos: int) -> int:
        before = part.shape[2]
        after = original_write(canvas, part, write_pos)
        written = after - write_pos
        if written < before:
            #: ⚠️ 画布不够大时 ``_write_part`` 会**截断** ✓ 不报错 ✗。
            #: 尾块截断是**合规**的（补出来的潜格本来就要削掉 ✓）；但**截断之后还继续写** ✗
            #: ⇒ 前面少写了 / 段数算多了 ✓ —— 那正是"尾部一截是 `torch.empty` 垃圾"的症状 ✓✗。
            if state["truncated"] is not None and written > 0:
                state["lateWrite"] = (write_pos, before, written)
            state["truncated"] = (write_pos, before, written)
        state["wrote"] = after
        return after

    module._write_part = counting_write  # noqa: SLF001
    return module, state


def _channel(torch: Any, canvas: Any, index: int) -> list[float]:
    """画布某通道 ⇒ 每帧数值（混合过的帧是**分数** ✓）。"""
    return [round(float(v), 6) for v in canvas[0, index, :, 0, 0].detach().cpu().tolist()]


def _zeros(torch: Any, latents: int) -> Any:
    z = torch.zeros((1, 24, latents, 1, 1), dtype=torch.float32)
    for index in range(latents):
        z[:, 0, index, 0, 0] = float(index + 1)  # 1-based：避开 0 与"没写过"混淆 ✓
    return z


def case_stitch(torch: Any) -> None:
    config = vae_h3_mod.H3VideoVAEConfig()
    module, state = _stub_module(torch, config)
    latents_cases = [2, 4, 5, 7, 12, 17, 24, 39]
    shape_bad: list[Any] = []
    write_bad: list[Any] = []
    order_bad: list[Any] = []
    for latents in latents_cases:
        state["truncated"] = None
        state["lateWrite"] = None
        state["wrote"] = 0
        out = module.decode_temporal(_zeros(torch, latents))
        want = (1, 3, vae_h3_mod.frames_for_latents(latents), 16, 16)
        if tuple(out.shape) != want:
            shape_bad.append((latents, tuple(out.shape), want))
        if state["lateWrite"] is not None or state["wrote"] != out.shape[2]:
            write_bad.append((latents, state["lateWrite"], state["wrote"], out.shape[2]))
        ids = [round(v) for v in _channel(torch, out, 0)]
        if any(b < a for a, b in zip(ids, ids[1:])) or min(ids) <= 0:
            order_bad.append((latents, ids))
    check("拼接: 画布形状 = `[1,3,frames_for_latents,16,16]`（多段也不偏 ✓）", not shape_bad,
          shape_bad[:3])
    check("拼接: 画布**每一帧都被写过**、截断只准发生在**末尾那一次**"
          "（中途截断 ⇒ 尾部是 `torch.empty` 垃圾 ✗ 且不报错 ✗）", not write_bad, write_bad[:3])
    check("拼接: 帧序**只增不减**（拼接错位 ⇒ 帧号回跳 ✓，这是「分段拼错了」最直接的证据 ✓）",
          not order_bad, order_bad[:2])


def case_blend(torch: Any) -> None:
    """段边界必须**正好** ``frame_overlap`` 帧混合 ✓，且权重是**线性**的 ✓（可逐帧算出来 ✓）。

    ⚠️ 每段推进 ``chunkDec - frame_pre_padding`` = **17** 帧 ✓（不是 12 ✗ —— 12 是"扣掉 5 帧重叠"
    的错觉 ✓：那 5 帧是**上一段 split-1 扣着没写**的 ✓ 不是从画布上再扣一次 ✗）。
    """
    config = vae_h3_mod.H3VideoVAEConfig()
    module, _state = _stub_module(torch, config)
    overlap = config.frame_overlap
    advance = config.token_chunk * config.temporal_scale - config.frame_pre_padding
    detail: list[Any] = []
    ok = True
    for latents in (7, 12, 17, 24, 39):
        _state["calls"] = 0  # ⚠️ 流水号要**每次从 0 起** ✓（跨用例累加 ⇒ 期望值整体错位 ✓✗）
        stamps = _channel(torch, module.decode_temporal(_zeros(torch, latents)), 1)
        chunks = vae_h3_mod.decode_temporal_plan(latents)["chunks"]
        #: 画布分两段：前 ``chunks·17`` 帧是各段的 split-0 ✓；剩下的是**最后一段 split-1 扣着的尾巴** ✓
        #: —— 尾巴是**直接写**的 ✓ 不参与混合 ✓（把它当"第 chunks 段"去算期望 ⇒ 假报警 ✓✗）
        chunk_frames = min(len(stamps), chunks * advance)
        expected: list[float] = []
        for index in range(chunk_frames):
            chunk, offset = divmod(index, advance)
            expected.append(float(chunk - 1) + offset / overlap
                            if chunk >= 1 and offset < overlap else float(chunk))
        expected += [float(chunks - 1)] * (len(stamps) - chunk_frames)
        mismatched = [(i, got, want) for i, (got, want) in enumerate(zip(stamps, expected))
                      if abs(got - want) > 1e-4]
        if mismatched:
            ok = False
            detail.append((latents, len(stamps), chunks, mismatched[:4]))
    check(f"混合: 段边界**恰好 {overlap} 帧**、且逐帧等于线性权重 `(s-1)+k/{overlap}`"
          f"（权重次序/帧数错 ⇒ 边界糊或错位，画面尺寸照样对 ✗）", ok, detail[:2])


def main() -> int:
    case_plan_vs_oracle()
    case_frame_overlap()
    if not _have_torch():
        skip("缺 torch ⇒ 拼接/混合两组（要真张量与真类）跳过 ✓")
    else:
        import torch  # noqa: PLC0415

        case_stitch(torch)
        case_blend(torch)

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
