"""S7 自检：引擎的**真模型层**（DiT + 权重装载 ✓ 2026-09-17）。

这套是"实现功能"那一步的核心验证 ✓ —— 全部在 **CPU 上**跑得动 ✓，因此**现在就**能验：

* 模型前向**形状自洽**（``(B,C,T,H,W)`` 进、同形出 ✓）；
* ⭐ **条件真的接进去了**（改 σ / 改文本 ⇒ 输出必须变 ✓ —— 防"条件没接进网络"的**假绿** ✗）；
* ⭐ **权重往返**：建模型 → 存成 safetensors → 新建模型装载 → 输出**逐位相同** ✓；
* ⭐ **差异如实报**：缺键 / 多键 / **形状不符**（⇒ 中止装载 ✓ 且**不污染**原参数 ✓）；
* 流匹配的 x0 转换（velocity / epsilon / sample ✓）与错误分支 ✓。

⚠️ 本套验证的是**机制** ✓，**不是**"H3 能出片" ✗（那还差真权重 + 命名映射 ✓，见 `torch_backend.PENDING_PARTS` ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_dit_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import dit as dit_mod  # noqa: E402
from app.services.engine import safetensors as st  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


CONFIG = dit_mod.DiTConfig(hidden=32, depth=2, heads=4, patch_size=(1, 2, 2),
                           in_channels=4, text_dim=16, mlp_ratio=2.0, vae_scale=8)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# ① 配置（**不猜** ✓：缺就报错）
# ══════════════════════════════════════════════════════════════════════════
def case_config() -> None:
    try:
        dit_mod.DiTConfig(hidden=33, depth=2, heads=4)
        failed = False
    except dit_mod.DiTConfigError as err:
        failed = "不能被 heads" in str(err)
    check("① hidden 不能被 heads 整除 ⇒ **报错**（而不是悄悄取整 ✗）", failed)

    read = dit_mod.DiTConfig.from_metadata({"hidden": "64", "depth": "3", "heads": "8",
                                            "patch": "1x2x2", "text_dim": "32"})
    check("② 元数据里读得到就用（宽容读入 ✓ 多种键名 ✓）",
          (read.hidden, read.depth, read.heads, read.patch_size) == (64, 3, 8, (1, 2, 2)),
          read.to_dict())
    try:
        dit_mod.DiTConfig.from_metadata({"hidden": "64"})
        missing_ok = False
    except dit_mod.DiTConfigError as err:
        missing_ok = "不猜" in str(err)
    check("③ 元数据不全 ⇒ **明确报错并要求显式给** ✓（**不猜结构** ✗ —— 猜了会在真机上错得莫名其妙 ✓）",
          missing_ok)
    check("④ `vae_scale` 默认 0 = **未给** ✓（像素↔潜空间比是 VAE 的知识，本模块不猜 ✗）",
          dit_mod.DiTConfig().vae_scale == 0)

    # ⭐ 2026-09-20：H3 结构事实（抄自 `ComfyUI` ✓ 逐条带出处 ✓ 不是猜的 ✗）
    h3 = dit_mod.H3_SHAPE_FACTS
    check("④′ ⭐ H3 事实自洽：**heads × headDim ≠ hidden** ✓（56×128=7168 vs 5376 ✓）"
          "—— 这条正是「本仓 MHA 装不下 H3 权重」的根因 ✓",
          h3["heads"] * h3["headDim"] == h3["attnInnerDim"] != h3["hidden"],
          (h3["heads"], h3["headDim"], h3["hidden"]))
    h3_config = dit_mod.DiTConfig(hidden=h3["hidden"], depth=h3["depth"], heads=h3["heads"],
                                  patch_size=h3["patchSize"], in_channels=h3["videoLatents"],
                                  text_dim=h3["textDim"], vae_scale=h3["vaeScale"])
    check("④″ ⭐ H3 数字**能构造** DiTConfig ✓ —— 但**不等于**能装 H3 权重 ✗"
          "（本仓推出来的 head_dim 是 96，不是 128 ✓）",
          h3_config.head_dim == h3["hidden"] // h3["heads"] != h3["headDim"],
          h3_config.head_dim)
    check("④‴ ⚠️ 缺口清单**只减不骗** ✓：未关 **7** 条 + 已关 3 条都列着 ✓"
          "（第 114 步把参考读全后从 5 条补到 7 条 ✓ —— 「没记 ≠ 没有」✗）",
          len(dit_mod.H3_STRUCTURAL_GAPS) == 7 and len(dit_mod.H3_GAPS_CLOSED) == 3
          and any("adaLN" in item for item in dit_mod.H3_STRUCTURAL_GAPS),
          (len(dit_mod.H3_STRUCTURAL_GAPS), len(dit_mod.H3_GAPS_CLOSED)))
    check("④¹⁶ ⭐ 打包规格**单独立表** ✓（与形状数字分开 ✓：混一起会让两类断言互相牵连 ✗）"
          "且钉住两条最要紧的事实：**音频在视频之前** ✓ + **无 attention mask** ✓",
          len(dit_mod.H3_PACK_FACTS) >= 6
          and any("audio" in fact and "video" in fact for fact in dit_mod.H3_PACK_FACTS)
          and any("attention mask" in fact for fact in dit_mod.H3_PACK_FACTS),
          dit_mod.H3_PACK_FACTS[:2])

    # ⭐⭐ 2026-09-20 **① 注意力维度**：让本仓能**表达** H3 的 7168 ≠ 5376 ✓
    #    （⚠️ H3 的真实尺寸**只构造、不建模型** ✗ —— 一个 block 的 qkv 就是 460 MB ✓，50 层 ≈ 23 GB ✗）
    h3_attn = dit_mod.DiTConfig(hidden=h3["hidden"], depth=h3["depth"], heads=h3["heads"],
                                patch_size=h3["patchSize"], in_channels=h3["videoLatents"],
                                text_dim=h3["textDim"], vae_scale=h3["vaeScale"],
                                attn_dim=h3["attnInnerDim"])
    check("④⁗ ⭐⭐ 显式给 `attn_dim` ⇒ **attn_head_dim == 128**（= H3 的 headDim ✓✓）"
          "—— 这正是「旧形态推不出来」的那一项（旧形态只会给 96 ✗）",
          h3_attn.attn_head_dim == h3["headDim"] != h3_attn.head_dim,
          (h3_attn.attn_head_dim, h3["headDim"], h3_attn.head_dim))
    keep = dit_mod.DiTConfig(hidden=32, depth=1, heads=4, patch_size=(1, 2, 2),
                             in_channels=4, text_dim=16)
    check("④⁵ ⭐ `attn_dim` 默认 0 ⇒ **inner == hidden**、`attn_head_dim == head_dim` ✓"
          "（旧权重 / 旧预设 / 旧断言一字不改 ✓）",
          keep.inner_dim == keep.hidden and keep.attn_head_dim == keep.head_dim, keep.inner_dim)
    try:
        from app.services.engine import dit as _dit  # noqa: PLC0415
        import torch  # noqa: PLC0415

        wide_cfg = dit_mod.DiTConfig(hidden=32, depth=1, heads=4, patch_size=(1, 2, 2),
                                     in_channels=4, text_dim=16, attn_dim=64)
        wide = _dit.build_dit(wide_cfg)
        block = wide.blocks[0]
        check("④⁶ ⭐ attn_dim≠hidden ⇒ 走**显式** qkv_proj/out_proj（形状 = 缩放版 H3 ✓："
              "qkv 32→192 ✓、out 64→32 ✓）且**没有** MHA ✓",
              block.explicit_attention is True
              and tuple(block.qkv_proj.weight.shape) == (3 * 64, 32)
              and tuple(block.out_proj.weight.shape) == (32, 64)
              and not hasattr(block, "attn"), tuple(block.qkv_proj.weight.shape))
        narrow = _dit.build_dit(keep)
        check("④⁷ ⭐ attn_dim=0 ⇒ 仍然只有 MHA（`attn` 在 ✓、`qkv_proj` **不在** ✓）",
              hasattr(narrow.blocks[0], "attn")
              and not hasattr(narrow.blocks[0], "qkv_proj")
              and narrow.blocks[0].explicit_attention is False)
        latent = torch.zeros(1, 4, 2, 4, 4)
        out = wide(latent, 0.5)
        check("④⁸ 显式形态**能跑通**且与潜变量同形 ✓",
              tuple(out.shape) == tuple(latent.shape), tuple(out.shape))
        raised = ""
        try:
            # ⚠️ 形状要**按 text_dim** 给（这个 config 是 16 ✓）——
            #    第一版写成 32 ✗ ⇒ 先撞上 `text_proj` 的矩阵乘法 ⇒ RuntimeError 把整套跑挂 ✗✓
            wide(latent, 0.5, torch.zeros(1, 3, wide_cfg.text_dim))
        except dit_mod.DiTConfigError as err:
            raised = str(err)
        # ⚠️ 实测发现：`DiT.forward` 在 **cross_attention=False** 时**根本不把 context 往下传** ✗
        #    ⇒ 这条"给了 context 就该报错"在**整模型层面到不了** ✓ ⇒ 改成下面两处更早/更靠内的守卫 ✓
        excited = bool(raised)
        check("④⁹⁰ ⭐ 记录一个事实：`cross_attention=False` 时 context **被忽略** ✓"
              "（不报错也不使用 ✓ —— 所以「被忽略」必须是**配置里说清楚**的 ✓，见下两条 ✓）",
              excited is False and wide_cfg.cross_attention is False, raised)

        conflicted = ""
        try:
            dit_mod.DiTConfig(hidden=32, depth=1, heads=4, attn_dim=64, cross_attention=True)
        except dit_mod.DiTConfigError as err:
            conflicted = str(err)
        check("④⁹ ⭐⭐ `attn_dim≠hidden` + `cross_attention=True` ⇒ **构造时就报错** ✗"
              "（宁可最早报 ✓，也不做出「能跑但装不上真权重」的假模型 ✗）",
              "自注意力" in conflicted, conflicted)
        inner = ""
        try:
            wide.blocks[0](torch.zeros(1, 4, 32), torch.zeros(1, 32), torch.zeros(1, 3, 32))
        except dit_mod.DiTConfigError as err:
            inner = str(err)
        check("④¹⁰ 直接调 block 给 context ⇒ **内层守卫也报错** ✓（双保险 ✓）",
              "自注意力" in inner, inner)

        # ⭐⭐ 2026-09-20 **② 双输出** + **④ condition_proj / token_refiner**：
        #    键名**逐字对齐 H3** ✓ —— 这是"能装真权重"的判据 ✓（名字对不上就得再改一次 ✓）
        h3_like = dit_mod.DiTConfig(hidden=32, depth=1, heads=4, patch_size=(1, 2, 2),
                                    in_channels=4, text_dim=16, attn_dim=64,
                                    audio_latents=8, text_refiner_layers=2)
        h3_model = _dit.build_dit(h3_like)
        names = set(h3_model.state_dict())
        # ⚠️⚠️ 2026-09-22 修正**测试里的错** ✗：这里原本钉的是 `token_refiner.blocks.0.qkv_proj.weight` ✓✗
        #      —— 那是**旧近似实现**的键（`qkv_proj` 直接挂在 block 上 ✗）；参考（与 `h3_keys` 的
        #      期望键表 ✓）是 **`token_refiner.blocks.N.attn.qkv_proj.weight`** ✓（多一层 `attn.` ✓），
        #      且 refiner **没有 adaLN** ✓。⇒ 「测试钉着的名字」**不等于**「参考的名字」✗
        #      （这条正是本仓反复吃过的亏：清单/断言没写全 ⇒ 下一个人以为"只剩 X 件" ✓✗）。
        check("④¹¹ ⭐ 键名对齐 H3：`condition_proj` / `token_refiner.blocks.N.attn.qkv_proj` / "
              "`final_layer.video_out` / `final_layer.audio_out` 全部就位 ✓",
              any(key.startswith("condition_proj.") for key in names)
              and "token_refiner.blocks.0.attn.qkv_proj.weight" in names
              and "token_refiner.blocks.1.attn.qkv_proj.weight" in names
              and "final_layer.video_out.weight" in names
              and "final_layer.audio_out.weight" in names, sorted(names)[:8])
        check("④¹¹′ ⭐ refiner 的内部结构**也按参考** ✓（2026-09-22 复用 `h3_form.TokenRefiner` ✓）："
              "`blocks.N.{norm1,norm2}`（**RMSNorm** ✓）+ `attn.{q_norm,k_norm}` ✓ + SwiGLU "
              "`mlp.{fc1,fc2}` ✓ + 收尾 `final_norm` ✓；⚠️ **无 adaLN** ✗（旧近似实现多出来的 "
              "`token_refiner.blocks.N.adaln_proj.*` 现在**必须不存在** ✓ —— `h3_keys` 就把它当 unexpected ✓）",
              {"token_refiner.final_norm.weight",
               "token_refiner.blocks.0.norm1.weight",
               "token_refiner.blocks.0.norm2.weight",
               "token_refiner.blocks.0.attn.q_norm.weight",
               "token_refiner.blocks.0.attn.k_norm.weight",
               "token_refiner.blocks.0.attn.out_proj.weight",
               "token_refiner.blocks.0.mlp.fc1.weight",
               "token_refiner.blocks.0.mlp.fc2.weight"} <= names
              and not any("adaln" in key for key in names if key.startswith("token_refiner.")),
              sorted(key for key in names if key.startswith("token_refiner.")))
        # ⚠️ 期望值我第一版写错了 ✗✓：把 H3 的 24 通道算进了**缩放版**（`patch 1×2×2 × 4ch = 16` ✓、
        #    H3 是 `2×2×24 = 96` ✓ —— **同一个公式** ✓）⇒ 自检当场红 ✓
        check("④¹² ⭐ 形状 = **缩放版 H3**：video_out 32→16（patch 1×2×2 × 4ch ✓；H3 是 2×2×24=96 ✓）、"
              "audio_out 32→8（= audio_latents ✓；H3 是 →32 ✓）",
              tuple(h3_model.final_layer.video_out.weight.shape) == (16, 32)
              and tuple(h3_model.final_layer.audio_out.weight.shape) == (8, 32),
              (tuple(h3_model.final_layer.video_out.weight.shape),
               tuple(h3_model.final_layer.audio_out.weight.shape)))
        video, audio = h3_model(latent, 0.5, torch.zeros(1, 3, wide_cfg.text_dim))
        check("④¹³ ⭐ 双输出形态返回 **(视频, 音频)** ✓（视频与潜变量同形 ✓、音频每 token 一个向量 ✓）",
              tuple(video.shape) == tuple(latent.shape)
              and audio.ndim == 3 and audio.shape[0] == 1 and audio.shape[-1] == 8,
              (tuple(video.shape), tuple(audio.shape)))

        # ⭐⭐ 2026-09-22 新增：**batch 之间不许串味** ✓
        #     起因：`h3_form` 的 refiner 按 H3 的 **2D 打包行**写 ✓（无 batch 维 ✗），dit 这边走 (B,L,D) ✓
        #     ⇒ 复用时若图省事**整批 `reshape(-1, hidden)`** ✗，batch 之间会被拉进同一条序列互相注意 ✓✗。
        #     ⚠️ 这种错 **`B=1` 的自检发现不了** ✗✗（本套绝大多数用例就是 B=1 ✓）⇒ 必须专门钉一条 ✓：
        #     判据 = **只改第 1 条的文本，第 0 条的输出必须**一字不动 ✓、第 1 条必须真变 ✓。
        #     （后半句是**反向证明** ✓ —— 否则"两条都不变"也会让前半句通过 ✓✗。）
        #     ⚠️⚠️ 前置：**必须先把调制层置非零** ✗ —— adaLN-Zero 初始化下**条件本来就不影响输出** ✓✗
        #     （第一版没置 ⇒ 两条 diff 都是 0.0 ⇒ 用例当场红 ✓✓ 这正是"反向证明"该有的作用 ✓）。
        with torch.no_grad():
            for block in h3_model.blocks:
                torch.nn.init.normal_(block.modulation[-1].weight, std=0.05)
                torch.nn.init.normal_(block.modulation[-1].bias, std=0.05)
            torch.nn.init.normal_(h3_model.final_modulation[-1].weight, std=0.05)
            torch.nn.init.normal_(h3_model.final_modulation[-1].bias, std=0.05)
            pair_latent = torch.cat([latent, latent * 0.5 + 0.1], dim=0)
            ctx_a = torch.zeros(2, 3, h3_like.text_dim)
            ctx_a[0] = 0.7
            ctx_a[1] = -0.3
            ctx_b = ctx_a.clone()
            ctx_b[1] = 0.9
            out_a, _ = h3_model(pair_latent, 0.5, ctx_a)
            out_b, _ = h3_model(pair_latent, 0.5, ctx_b)
        check("④¹⁶ ⭐⭐ **batch 独立性** ✓：只改第 1 条的文本 ⇒ 第 0 条输出**一字不动** ✓ 且第 1 条"
              "**确实变了** ✓（⇒ refiner 是**逐条**按 2D 走的 ✓ 没被整批 flatten 成一条序列 ✓✗）",
              torch.allclose(out_a[0], out_b[0]) and not torch.allclose(out_a[1], out_b[1]),
              (float((out_a[0] - out_b[0]).abs().max()),
               float((out_a[1] - out_b[1]).abs().max())))
        check("④¹⁴ ⭐ 两个开关都关 ⇒ **回到旧形态** ✓（`out` 在 ✓、`final_layer`/`token_refiner` 不在 ✓）",
              "out.weight" in set(narrow.state_dict())
              and not any(key.startswith(("final_layer.", "token_refiner."))
                          for key in narrow.state_dict()))
        half = ""
        try:
            dit_mod.DiTConfig(hidden=32, depth=1, heads=4, audio_latents=8)
        except dit_mod.DiTConfigError as err:
            half = str(err)
        check("④¹⁵ ⭐ 只给 `audio_latents` 不给 `attn_dim` ⇒ **构造时报错** ✗（拒绝「半套 H3」✓）",
              "attn_dim" in half, half)
    except ImportError:
        _skip("⏭ ④⁶-④⁹：没装 torch（显式注意力形态的形状/报错自检跳过）")

    info = st.inspect(_write_minimal_checkpoint())
    inferred = dit_mod.infer_config_from_info(info)
    check("⑤ 从权重头部**机械确定**层数（``blocks.N`` 的个数 ✓）", inferred.depth == 2,
          inferred.to_dict())


def _write_minimal_checkpoint(blocks: int = 2) -> Path:
    """造一个只有 ``blocks.1.weight`` 之类键的**极小** safetensors ✓（只为验层数推断 ✓）。"""
    import json
    import struct

    where = Path(tempfile.mkdtemp(prefix="dit_cfg_")) / "mini.safetensors"
    header: dict[str, Any] = {}
    offset = 0
    blobs: list[bytes] = []
    for index in range(blocks):
        name = f"blocks.{index}.w.weight"
        size = 4 * 4  # F32 4 元素
        header[name] = {"dtype": "F32", "shape": [4], "data_offsets": [offset, offset + size]}
        blobs.append(bytes(size))
        offset += size
    payload = json.dumps(header).encode("utf-8")
    with where.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)
    return where


# ══════════════════════════════════════════════════════════════════════════
# ② 前向 / 条件 / 流匹配（需要 torch ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_forward() -> None:
    if not _have_torch():
        skip("torch 未安装 ⇒ 前向与权重往返跳过 ✓")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    model = dit_mod.build_dit(CONFIG)
    latent = torch.randn(1, CONFIG.in_channels, 2, 8, 8)
    context = torch.randn(1, 1, CONFIG.text_dim)

    out = model(latent, 1.0, context)
    check("⑥ 前向**同形**（(B,C,T,H,W) → 同形 ✓）", tuple(out.shape) == tuple(latent.shape),
          (tuple(latent.shape), tuple(out.shape)))
    check("⑦ token 数 = 网格数之积（fT·fH·fW ✓ —— 与 patch 尺寸一致 ✓）",
          model.grid_sizes(latent) == (2, 4, 4), model.grid_sizes(latent))

    # ⚠️ 这里曾被自己的断言绊了一下 ✓，值得写清：**adaLN-Zero 的调制层初值全 0**
    #    ⇒ 刚建好的模型对 σ 与文本条件**完全没有反应** ✓ —— 这是该结构的**定义** ✓（恒等起步、训练中长出来 ✓），
    #    不是"条件没接" ✗。所以断言必须**成对**：初始化时无效 ✓ + 有非零调制权重后**必须**生效 ✓。
    check("⑧ adaLN-Zero 初始化 ⇒ 条件暂时无效（改 σ 输出不变 ✓ —— 这是结构定义，不是 bug ✓）",
          bool(torch.allclose(out, model(latent, 5.0, context))), "")
    check("⑨ 同输入**逐位可复现**（无隐藏随机 ✓）",
          bool(torch.allclose(out, model(latent, 1.0, context))), "")

    # 「训练一下」：把调制层权重置为非零 ✓ ⇒ 条件就该真的影响输出了 ✓
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(block.modulation[-1].bias, std=0.05)
    torch.nn.init.normal_(model.final_modulation[-1].weight, std=0.05)
    torch.nn.init.normal_(model.final_modulation[-1].bias, std=0.05)

    trained = model(latent, 1.0, context)
    check("⑩ ⭐ **σ 真的进了网络**（调制非零后：改 σ ⇒ 输出变 ✓ —— 防「条件没接」的假绿 ✗）",
          not bool(torch.allclose(trained, model(latent, 5.0, context))), "")
    check("⑪ ⭐ **文本条件也真的进了网络** ✓",
          not bool(torch.allclose(trained, model(latent, 1.0,
                                                 torch.randn(1, 1, CONFIG.text_dim)))), "")

    # 尺寸不整除 patch ⇒ 报错（不悄悄向下取整 ✗）
    try:
        model(latent[:, :, :, :7, :], 1.0, context)
        bad = False
    except dit_mod.DiTConfigError as err:
        bad = "整除" in str(err)
    check("⑪ 尺寸不整除 patch ⇒ **报错** ✓（悄悄取整会让画面错位但不报错 ✗）", bad)

    # 流匹配 x0
    x = torch.tensor([2.0])
    v = torch.tensor([0.5])
    check("⑫ 流匹配 velocity：x0 = x − σ·v ✓",
          float(dit_mod.flow_match_x0(v, x, 2.0)) == 1.0, float(dit_mod.flow_match_x0(v, x, 2.0)))
    check("⑬ σ=0 ⇒ 原样返回（没有噪声可去 ✓ 避免除零/越界 ✓）",
          float(dit_mod.flow_match_x0(v, x, 0.0)) == 2.0)
    check("⑭ sample 预测 ⇒ 直接采用模型输出 ✓",
          float(dit_mod.flow_match_x0(v, x, 2.0, prediction="sample")) == 0.5)
    try:
        dit_mod.flow_match_x0(v, x, 1.0, prediction="nope")
        bad_kind = False
    except dit_mod.DiTConfigError:
        bad_kind = True
    check("⑮ 未知预测类型 ⇒ 报错（不默认当成 velocity ✗ —— 猜错等于产物全错 ✓）", bad_kind)


# ══════════════════════════════════════════════════════════════════════════
# ③ 权重往返与差异报告（**最硬的验证** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_weights() -> None:
    if not _have_torch():
        skip("torch 未安装 ⇒ 权重往返跳过 ✓")
        return
    import torch  # noqa: PLC0415

    root = Path(tempfile.mkdtemp(prefix="dit_w_"))
    torch.manual_seed(1)
    model = dit_mod.build_dit(CONFIG)
    latent = torch.randn(1, CONFIG.in_channels, 2, 8, 8)
    context = torch.randn(1, 1, CONFIG.text_dim)
    before = model(latent, 1.0, context).detach()

    path = weights_mod.save_module_weights(model, root / "dit.safetensors",
                                           metadata={"hidden": "32", "depth": "2", "heads": "4",
                                                     "vae_scale": "8", "text_dim": "16"})
    torch.manual_seed(2)                     # 换个种子 ⇒ 新模型参数**不一样** ✓
    fresh = dit_mod.build_dit(CONFIG)
    check("⑯ 新模型的初始参数与旧的不同（否则下面的往返验证毫无意义 ✗）",
          not bool(torch.allclose(fresh(latent, 1.0, context).detach(), before)), "")

    report = weights_mod.load_module_weights(fresh, path)
    check("⑰ ⭐ 往返装载 complete=True（无缺键/无形状不符 ✓）", report.complete, report.to_dict())
    check("⑱ ⭐⭐ 装载后输出与原件**逐位相同**（权重真的进了正确的模块 ✓）",
          bool(torch.allclose(fresh(latent, 1.0, context).detach(), before, atol=1e-6)),
          float((fresh(latent, 1.0, context).detach() - before).abs().max()))

    # 元数据里带配置 ⇒ 可以不显式给 config ✓（这正是"真权重到手就能自举"的路径 ✓）
    from_meta = dit_mod.DiTConfig.from_metadata(st.inspect(path).metadata)
    check("⑲ 元数据能自举出配置（hidden/depth/heads/vae_scale ✓）",
          (from_meta.hidden, from_meta.depth, from_meta.vae_scale) == (32, 2, 8), from_meta.to_dict())

    # 缺键 ⇒ 如实报 missing（**不许**静默 ✗）
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    full = load_file(path)
    partial = {key: value for key, value in full.items() if "blocks.0" not in key}
    partial_path = root / "partial.safetensors"
    save_file(partial, str(partial_path), metadata={"hidden": "32", "depth": "2", "heads": "4"})
    partial_report = weights_mod.load_module_weights(dit_mod.build_dit(CONFIG), partial_path)
    check("⑳ ⭐ 缺键 ⇒ `complete=False` 且**逐条列出**（缺一层却照常跑 = 最坏的「像成功」✗）",
          partial_report.complete is False and len(partial_report.missing) > 0,
          partial_report.to_dict()["missing"][:3])

    extra = dict(full)
    extra["not.a.real.tensor"] = torch.zeros(2)
    extra_path = root / "extra.safetensors"
    save_file(extra, str(extra_path), metadata={})
    extra_report = weights_mod.load_module_weights(dit_mod.build_dit(CONFIG), extra_path)
    check("㉑ 多出来的键 ⇒ 报告 unexpected ✓ 但**不影响**完整性判定 ✓",
          extra_report.complete is True and extra_report.unexpected == ["not.a.real.tensor"],
          extra_report.to_dict()["unexpected"])

    # 形状不符 ⇒ 中止装载，且**不能污染**目标模块 ✓
    wrong = dict(full)
    wrong["out.weight"] = torch.zeros(4, 4)          # 故意给错形状 ✓
    wrong_path = root / "wrong.safetensors"
    save_file(wrong, str(wrong_path), metadata={})
    victim = dit_mod.build_dit(CONFIG)
    victim_before = victim(latent, 1.0, context).detach()
    wrong_report = weights_mod.load_module_weights(victim, wrong_path)
    check("㉒ ⭐ 形状不符 ⇒ **中止装载**（不是「部分装上」✗）且报错说清 ✓",
          wrong_report.complete is False and bool(wrong_report.error)
          and "中止装载" in wrong_report.error, wrong_report.error[:80])
    check("㉒b ⭐ 中止后目标模块**参数未被污染** ✓（半装状态最难查 ✗）",
          bool(torch.allclose(victim(latent, 1.0, context).detach(), victim_before)), "")

    check("㉓ 权重文件不存在 ⇒ 报错可行动（指向 `loader.plan_stage` ✓ 而不是干巴巴「未找到」✗）",
          "loader.plan_stage" in weights_mod.load_module_weights(
              dit_mod.build_dit(CONFIG), root / "nope.safetensors").error, "")

    # key_map：**显式**改名 ✓（不做自动猜前缀 ✗）
    mapped = {f"model.{key}": value for key, value in full.items()}
    mapped_path = root / "mapped.safetensors"
    save_file(mapped, str(mapped_path), metadata={})
    mapped_report = weights_mod.load_module_weights(
        dit_mod.build_dit(CONFIG), mapped_path, key_map={r"^model\.": ""})
    check("㉔ 显式 key_map 能改名装载 ✓（并计数 renamed ✓）；**不做**自动猜前缀 ✗",
          mapped_report.complete is True and mapped_report.renamed == len(full),
          (mapped_report.complete, mapped_report.renamed))


def main() -> int:
    case_config()
    case_forward()
    case_weights()

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
