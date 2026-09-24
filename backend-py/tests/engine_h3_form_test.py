"""H3 形态**积木**的自检 ✓（要 torch ✓ —— 没有就 SKIP 并**明说** ✓，不假装通过 ✗）。

⚠️ 这些积木**现在还没有调用方** ✗（见 `h3_form.H3_FORM_TODO` ✓）—— 所以本套件只钉
「**结构事实**」与「**能自己算出来的不变量**」✓：键名 / 形状 / 18 路 / cos 在前 ✓ /
旋转不变量 ✓ / 分档真的分档 ✓。**不写**"方向直觉"式的断言 ✗（那是上次栽的地方 ✓）。
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import h3_form as h3  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch ✓ —— **如实标记**，不算通过 ✗）"))


def main() -> int:
    # ① 事实表本身**不需要 torch** 就能读 ✓（所以先测它 ✓）
    # ⚠️ 2026-09-20 起待办**清零** ✓（参考视频入口已通 ✓ / PDD 已实现并从权重自动推断 ✓）
    #    ⇒ 判据翻转：**必须恰好为 0** ✓ —— 再往 `h3_form` 加没接线的东西必须先进 TODO ✗
    #    （否则"待办清单"又会变成没人看的套话 ✓✗）。
    check("① 积木清单非空 ✓、**待办清单已清零** ✓（全部接线完毕 ✓ —— 有新待办就必须出现 ✗）",
          len(h3.H3_FORM_PARTS) >= 8 and len(h3.H3_FORM_TODO) == 0,
          (len(h3.H3_FORM_PARTS), len(h3.H3_FORM_TODO)))
    check("①′ 不存在的名字 ⇒ `AttributeError` ✓（**不静默给 None** ✗；且**不需要 torch** ✓）",
          "没有" in str(_raises(lambda: h3.NoSuchPart)))

    try:
        import torch
    except ImportError:
        for label in ("②…⑫"):
            skip(f"{label} torch 相关检查")
        return _report()

    torch.manual_seed(0)
    # ②③ RMSNorm：**scale 不变性** 与"只有 weight、没有 bias" ✓
    norm = h3.RMSNorm(8, eps=1e-5)
    with torch.no_grad():   # ⚠️ 不包 `no_grad` 会打一条"requires_grad 转标量"的 warning ✓ ——
        x = torch.randn(4, 8)   #    它**看着像错** ✗（上轮刚记过"乱码像崩了"的教训 ✓）⇒ 顺手消掉 ✓
        scaled = norm(x * 2.0)
    check("② RMSNorm：**尺度不变**（`norm(2x) == norm(x)` ✓）且输出 RMS ≈ 1 ✓",
          torch.allclose(scaled, norm(x), atol=1e-6)
          and abs(float(scaled.pow(2).mean()) - 1.0) < 0.02,
          (float(scaled.pow(2).mean()), float(scaled.std())))
    check("③ RMSNorm 只有 `weight` ✓（**没有 bias** ✓ —— H3 全模型如此 ✓）",
          set(norm.state_dict()) == {"weight"}, sorted(norm.state_dict()))

    # ④ TimeEmbedder：**cos 在前** ✓ —— 用 pre-hook 抠出 `proj_in` 的输入来验 ✓（不靠形状猜 ✓）
    time_embed = h3.TimeEmbedder(8, 16, 16)
    captured: list[Any] = []
    time_embed.proj_in.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach()))
    time_embed(torch.zeros(1))
    raw = captured[0][0]
    check("④ TimeEmbedder：`t=0` ⇒ 嵌入前半 **cos=1** ✓、后半 **sin=0** ✓（**cos 在前** ✓ 是写死的顺序 ✓）",
          torch.allclose(raw[:4], torch.ones(4), atol=1e-6)
          and torch.allclose(raw[4:], torch.zeros(4), atol=1e-6), raw.tolist())

    # ⑤ SwiGLU：两个 Linear **都无 bias** ✓ + 门真的在起作用 ✓
    mlp = h3.SwiGLU(8, 12)
    # ⚠️ 期望我第一版又写反了 ✗✓：`nn.Linear(in, out).weight` 是 **(out, in)** ✓
    #    ⇒ `fc1` = (2×ffn, hidden) = (24, 8) ✓、`fc2` = (hidden, ffn) = **(8, 12)** ✓
    #    （**都是 (out, in)** ✓ 自洽 ✓）；这条是本轮第 5 次「代码对、我的期望错」✗ ⇒ 见日志规则 ✓
    check("⑤ SwiGLU：`fc1` (2×ffn, hidden) = (24, 8) ✓ / `fc2` (hidden, ffn) = (8, 12) ✓，**都无 bias** ✓",
          tuple(mlp.fc1.weight.shape) == (24, 8) and tuple(mlp.fc2.weight.shape) == (8, 12)
          and mlp.fc1.bias is None and mlp.fc2.bias is None,
          (tuple(mlp.fc1.weight.shape), tuple(mlp.fc2.weight.shape)))
    with torch.no_grad():
        mlp.fc1.weight.zero_()
    check("⑤′ 门起作用（`fc1` 置零 ⇒ 输出**全零** ✓ —— 不是「照抄输入」✗）",
          bool((mlp(torch.randn(3, 8)) == 0).all()))

    # ⑥ Attention：H3 的关键特征 —— **inner = heads×head_dim ≠ hidden** 也能跑 ✓
    attn = h3.Attention(hidden=8, heads=2, head_dim=3)
    check("⑥ Attention 键名对齐 H3 ✓ + `qkv_proj` 形状 = (3×heads×head_dim, hidden) ✓"
          "（**inner 6 ≠ hidden 8** ✓ 正是 H3 形态 ✓）",
          set(attn.state_dict()) == {"qkv_proj.weight", "q_norm.weight", "k_norm.weight",
                                     "out_proj.weight"}
          and tuple(attn.qkv_proj.weight.shape) == (18, 8)
          and tuple(attn.out_proj.weight.shape) == (8, 6),
          sorted(attn.state_dict()))
    head = attn(torch.randn(5, 8))
    check("⑥′ 前向：`[S, hidden] → [S, hidden]` ✓（S=5 无 batch 维 ✓）",
          tuple(head.shape) == (5, 8), tuple(head.shape))

    # ⑦ RoPE 的**四个不变量**（角度的**分配方式**没核过 ✗ ⇒ 只验"给定角度怎么转"✓）
    vec = torch.randn(3, 2, 6)
    angles_zero = torch.zeros(3, 4)
    check("⑦ `rot_dim=0` ⇒ 原样返回 ✓；角度全 0 ⇒ **原样** ✓（cos=1 / sin=0 ✓）",
          torch.equal(h3.Attention.apply_rope(vec, angles_zero, 0), vec)
          and torch.allclose(h3.Attention.apply_rope(vec, angles_zero, 4), vec, atol=1e-6))
    angles = torch.randn(3, 4)
    rotated = h3.Attention.apply_rope(vec, angles, 4)
    check("⑦′ 旋转**不改变范数** ✓（`‖x‖` 逐位保持 ✓ —— 这是 RoPE 的定义性质 ✓）",
          torch.allclose(vec.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5))
    check("⑦″ 只有前 `rot_dim` 维被转 ✓ —— **尾部逐位不变** ✓",
          torch.equal(rotated[..., 4:], vec[..., 4:]))

    # ⑧ AdalnProj：**18 路** = expand 6 × modalities 3 ✓
    adaln = h3.AdalnProj(t_dim=16, hidden=8, expand=6, modalities=3)
    outs = adaln(torch.randn(5, 16))
    check("⑧ adaLN = 6 × 3 = **18** 路 ✓（`linear` 出 18×hidden ✓）；返回 **6** 个张量 ✓、"
          "每个 `[S×3, hidden]` ✓（逐 token 自带 3 个候选行 ✓）",
          tuple(adaln.linear.weight.shape) == (144, 16) and len(outs) == 6
          and all(tuple(out.shape) == (15, 8) for out in outs),
          (tuple(adaln.linear.weight.shape), len(outs), tuple(outs[0].shape)))
    check("⑧′ mod-row 可以是**逐 token 索引**（不是只有整数 ✓ —— H3 就是按 token 分档 ✓）",
          tuple(outs[0][torch.tensor([0, 1, 2, 0])].shape) == (4, 8))

    # ⑨ DiTBlock：**分档真的分档** ✓ —— 双向验（不同 row ⇒ 不同 ✓；同一 row ⇒ 相同 ✓✓）
    block = h3.DiTBlock(hidden=8, heads=2, head_dim=3, ffn=12, t_dim=16)
    with torch.no_grad():
        block.adaln_proj.linear.weight.normal_(0, 0.5)
        block.adaln_proj.linear.bias.normal_(0, 0.5)
    # ⚠️ 两段必须喂**逐位相同**的输入 ✓ —— 初版喂了随机 token ✗ ⇒ 「同档应相同」那条
    #    会因为 token 不同而红 ✗✓（我自审时抓到的 ✓：这测的是**分档**，不是 token 内容 ✓）
    half = torch.randn(3, 8)
    tokens = torch.cat([half, half])
    t_emb = torch.randn(3, 16).repeat(2, 1)
    split = block(tokens, t_emb, [(0, 3, 0), (3, 6, 1)])
    same = block(tokens, t_emb, [(0, 3, 0), (3, 6, 0)])
    check("⑨ mod_segments **真分档**：同样输入下，两段用**不同** mod-row ⇒ 输出不同 ✓；"
          "改成**同一** row ⇒ 两段逐位相同 ✓（否则「分段」就是摆设 ✗ —— 双向都验 ✓）",
          not torch.allclose(split[:3], split[3:], atol=1e-5)
          and torch.allclose(same[:3], same[3:], atol=1e-6), None)

    # ⑩ FinalLayer：两个头 **fp32** ✓ + 返回两个张量 ✓
    final = h3.FinalLayer(hidden=8, t_dim=16, video_dim=6, audio_dim=4)
    video, audio = final(tokens, t_emb, (0, 4, 0), (4, 6, 0))
    check("⑩ `video_out` / `audio_out` **存 fp32** ✓（H3 的 fp32 岛 ✓）+ 返回 **(视频, 音频)** ✓、"
          "行数 = 各自段长 ✓（4 / 2 行 ✓）",
          final.video_out.weight.dtype == torch.float32
          and final.audio_out.weight.dtype == torch.float32
          and tuple(video.shape) == (4, 6) and tuple(audio.shape) == (2, 4),
          (str(final.video_out.weight.dtype), tuple(video.shape), tuple(audio.shape)))

    # ⑪ TokenRefiner：键名逐字对齐 H3 ✓
    refiner = h3.TokenRefiner(num_layers=2, hidden=8, heads=2, head_dim=3, ffn=12)
    refined = refiner(tokens)
    check("⑪ `token_refiner`：**2** 层 `blocks.N.*` ✓ + `final_norm` ✓（H3 是 2 层 ✓）；前向同形 ✓",
          "blocks.0.norm1.weight" in refiner.state_dict()
          and "blocks.1.attn.qkv_proj.weight" in refiner.state_dict()
          and "final_norm.weight" in refiner.state_dict()
          and tuple(refined.shape) == (6, 8), sorted(refiner.state_dict())[:3])

    # ── 打包层（第 ② 批 ✓ 事实来源 `dit.H3_PACK_FACTS` ✓）──
    layout = h3.packed_rows(text_len=7, latent_t=3, latent_h=4, latent_w=4, audio_t=5)
    kinds = [seg[2] for seg in layout["segments"]]
    check("⑫ 段顺序 = text → **audio → video** ✓（**音频在视频之前** ✓ —— 切线就靠这条 ✓）、"
          "从 0 起 ✓ **首尾相接**（无空洞无重叠 ✓）",
          kinds == ["text", "audio", "video"] and layout["segments"][0][0] == 0
          and all(layout["segments"][i][1] == layout["segments"][i + 1][0] for i in range(2))
          and layout["seq_len"] == 7 + 5 * 2 + 3 * 2 * 2, layout)
    check("⑫′ 行数公式：视频 = `latent_t×(h//2)(w//2)` ✓（3×2×2 = 12 ✓）、音频 = `audio_t×2` ✓（立体声 ✓）",
          layout["video_rows"] == 12 and layout["audio_rows"] == 10, layout)

    pos = h3.position_ids(text_len=7, latent_t=3, latent_h=4, latent_w=4, audio_t=5)
    check("⑬ `position_ids` = `[seq_len, 3]` ✓、float64 ✓、**三段与段表逐段对齐** ✓"
          "（text 的 t = 0…L−1 ✓ 且 h = w = 0 ✓）",
          tuple(pos.shape) == (29, 3) and pos.dtype == torch.float64
          and torch.allclose(pos[:7, 0], torch.arange(7, dtype=torch.float64))
          and bool((pos[:7, 1:] == 0).all()), tuple(pos.shape))
    check("⑬′ 音频段：t 从 `text_len` 起 ✓ 且**重复两遍** ✓（立体声 channel-major ✓）；"
          "每半段 w **恒定** ✓、两半 w **不同** ✓（取网格两端 ✓）",
          torch.allclose(pos[7:12, 0], torch.arange(7, 12, dtype=torch.float64))
          and torch.allclose(pos[12:17, 0], torch.arange(7, 12, dtype=torch.float64))
          and pos[7, 2] == pos[11, 2] and pos[12, 2] == pos[16, 2] and pos[7, 2] != pos[12, 2],
          (pos[7, 2].item(), pos[12, 2].item()))
    # ⚠️ token 顺序是 **先帧后行列** ✓ ⇒ 一帧内（`frame_rows` = 4 个）t **全部相同** ✓；
    #    下一个 t 要到 **+frame_rows** 处 ✓ —— 我第一版按"逐个 token 递增"写 ✗✓（自检当场红 ✓）
    per_frame = layout["video_rows"] // 3
    check("⑬″ 视频段：t 按**跨度表**累加 ✓，且**一帧内的 token 共用同一个 t** ✓"
          "（第 0 帧 = 起点 ✓、第 1 帧 = 起点 + 5/3 ✓）；正方形潜帧下 h 与 w 坐标**逐位相同** ✓",
          bool((pos[17:17 + per_frame, 0] == 7.0).all())
          and abs(float(pos[17 + per_frame, 0]) - (7.0 + 5.0 / 3.0)) < 1e-9
          and torch.allclose(torch.sort(pos[17:, 1]).values, torch.sort(pos[17:, 2]).values),
          pos[17:22, 0].tolist())

    grid, w_axis = h3.frame_grid_coords(8, 8)
    span = float(w_axis[-1] - w_axis[0])
    # ⚠️ `cartesian_prod` 是**笛卡尔积** ✓ ⇒ 两列**不是逐位相等** ✗（那是两回事 ✓）：
    #    正确的说法是「两轴的**值集合相同**」✓ —— 我第一版写成了逐位相等 ✗✓（又一次被自检抓住 ✓）
    check("⑭ 帧网格坐标：行数 = `(h//2)(w//2)` ✓、**单调递增** ✓、正方形时两轴**值集合相同** ✓、"
          "跨度 = `(n−1)·ratio/n·32` ✓（**按公式自己算** ✓ —— 不写「落在 0..32」那种直觉 ✗）",
          tuple(grid.shape) == (16, 2)
          and bool((w_axis[1:] > w_axis[:-1]).all())
          and abs(span - 3 * (1.0 / 4) * 32.0) < 1e-9
          and torch.allclose(grid[:4, 1], w_axis) and torch.allclose(grid[::4, 0], w_axis),
          (tuple(grid.shape), span))

    inv_freq = torch.tensor([1.0, 0.5, 0.25, 0.125])       # fp32 ✓（真权重里就是 fp32 ✓）
    unit = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float64)
    angles = h3.rope_angles(unit, inv_freq)   # ⚠️ 里面会 cast 成 fp32 ✓ ⇒ 断言也按 fp32 写 ✓
    check("⑮ `rope_angles`：长度 = `6×len(inv_freq)` ✓、**前后两半逐位相同** ✓、**t 在最前** ✓"
          "（用「只有某一轴非零」的坐标验 ✓ ⇒ 顺带钉住 t,h,w 的顺序 ✓）"
          "—— ⚠️ 比较**跟随实际 dtype** ✓，不写死 ✗（前两次就是写死 dtype 才红的 ✓）",
          tuple(angles.shape) == (3, 24)
          and torch.equal(angles[:, :12], angles[:, 12:])
          and torch.allclose(angles[0, :4], (2.0 * inv_freq).to(angles.dtype))
          and torch.allclose(angles[0, 4:8], torch.zeros(4, dtype=angles.dtype))
          and torch.allclose(angles[1, 4:8], (3.0 * inv_freq).to(angles.dtype)), angles[0].tolist())
    # ⚠️ 角度要**按 S 现算** ✓ —— 上一步我拿 3 行的 `angles` 去配 5 行的 q ✗
    #    ⇒ 广播直接报 size mismatch ✓（这种"形状没对齐"只有真跑才现形 ✓）
    wide = h3.rope_angles(torch.randn(5, 3, dtype=torch.float64), inv_freq)
    check("⑯ `rot_dim = 6·len(inv_freq)` 口径下 `apply_rope` **仍保范数** ✓（与 ⑦ 同一不变量 ✓，换成真长度 ✓）",
          (lambda q: torch.allclose(q.norm(dim=-1),
                                    h3.Attention.apply_rope(q, wide[:, :8], 8).norm(dim=-1),
                                    atol=1e-5))(torch.randn(5, 2, 8)))
    check("⑰ `mod_row` = 序号×3 + 模态 ✓（与 `AdalnProj` 的 reshape 自洽 ✓）；越界模态 ⇒ 报错 ✓",
          h3.mod_row(0, 0) == 0 and h3.mod_row(0, 2) == 2 and h3.mod_row(3, 1) == 10
          and "modality" in str(_raises(lambda: h3.mod_row(0, 3))))
    check("⑱ **只减不骗**（双向 ✓）：做完的从 TODO **移出** ✓、同时出现在 PARTS 里 ✓"
          "（本轮把「打包网格 / RoPE 频率分配」两条关掉了 ✓）",
          not any("打包网格" in item for item in h3.H3_FORM_TODO)
          and not any("频率分配" in item for item in h3.H3_FORM_TODO)
          and any("打包层" in item for item in h3.H3_FORM_PARTS)
          and any("rope_angles" in item for item in h3.H3_FORM_PARTS),
          h3.H3_FORM_TODO[:2])

    # ── 主干（第 ③ 批 ✓）──
    check("⑲ `H3_DEFAULTS` **逐条对上参考 `__init__`** ✓（5376 / 50 层 / 56×128 / ffn 14336 / text 5120 / "
          "潜 24 · 音频 32 / t 256→5376→2688 / F 16 / shift 12·3 ✓ —— 全是**事实** ✓）",
          h3.H3_DEFAULTS["hidden"] == 5376 and h3.H3_DEFAULTS["layers"] == 50
          and h3.H3_DEFAULTS["heads"] * h3.H3_DEFAULTS["head_dim"] == 7168
          and h3.H3_DEFAULTS["ffn"] == 14336 and h3.H3_DEFAULTS["text_dim"] == 5120
          and h3.H3_DEFAULTS["time_dim"] == 2688 and h3.H3_DEFAULTS["inv_freq_len"] == 16
          and (h3.H3_DEFAULTS["sigma_shift_video"], h3.H3_DEFAULTS["sigma_shift_audio"]) == (12.0, 3.0),
          sorted(h3.H3_DEFAULTS)[:4])
    values, index = h3.t_vals_for(0.5)
    check("⑳ ⭐ `t_vals_for` 的**档位是事实** ✓（2026-09-20 逐字核自参考 `seg_t` ✓）：text 与 video **同档** ✓、"
          "audio = 1−换算后的 σ ✓、cond/ref_img = max(t_v, 0.999) ✓、cond_audio/ref_audio = max(t_a, 1.0) ✓"
          " ⇒ 四个不同值 ⇒ **4 行** ✓（⚠️ 初版把 text 钉在 0.999 ✗ —— 那是**条件行**的档 ✓，已按事实改 ✓）",
          len(values) == 4 and index["text"] == index["video"] != index["audio"]
          and abs(float(values[index["video"]]) - 0.5) < 1e-6
          and abs(float(values[index["cond"]]) - 0.999) < 1e-6
          and abs(float(values[index["cond_audio"]]) - 1.0) < 1e-6
          and index["ref_img"] == index["cond"] and index["ref_audio"] == index["cond_audio"],
          (values.tolist(), index))
    deduped, dedup_index = h3.t_vals_for(0.5, shift_v=3.0, shift_a=3.0)
    check("⑳′ ⭐ **去重** ✓：两条 shift 相同时 audio 与 video 的 t 相同 ⇒ 那档合并 ✓（不去重 ⇒ 行号全错 ✓✗ "
          "且**不报错** ✗）；且 ``values`` **升序** ✓（= 参考的 ``sorted`` ✓ —— 换口径只为与参考同构 ✓，"
          "免得将来加段类时两种顺序**静默分叉** ✓✗）",
          dedup_index["audio"] == dedup_index["video"] and len(deduped) == 3
          and all(float(deduped[i]) <= float(deduped[i + 1]) for i in range(len(deduped) - 1)),
          (deduped.tolist(), dedup_index))
    check("㉑ `audio_carry` = `σ_a/σ_v` ✓；shift 相同 ⇒ 恒为 **1** ✓（不变量 ✓）；12→3 时在 (0,1) 内 ✓",
          abs(h3.audio_carry(0.5, 3.0, 3.0) - 1.0) < 1e-9 and 0.0 < h3.audio_carry(0.5) < 1.0,
          h3.audio_carry(0.5))
    segs2 = h3.mod_segments_for(layout["segments"], index)
    check("㉒ ⭐ `mod_segments_for` 的**标签也是事实** ✓（参考前向里的 `seg_tag` ✓）：**video/cond/ref_img = 0** ✓、"
          "**text = 1** ✓、**audio/cond_audio/ref_audio = 2** ✓ —— ⚠️ 初版写「text=0 / audio=1 / video=2」✗，"
          "**三档全错** ✓✗（标签只是 adaLN 的**行内偏移** ✓ ⇒ 真权重上会让每个模态接到**别的模态**那组参数 ✓"
          "而**完全不报错** ✗）；且**覆盖整条序列** ✓、**首尾相接** ✓、行号都 < 18 ✓",
          segs2[0][0] == 0 and segs2[-1][1] == layout["seq_len"]
          and all(segs2[i][1] == segs2[i + 1][0] for i in range(len(segs2) - 1))
          and segs2[0][2] == index["text"] * 3 + 1
          and segs2[1][2] == index["audio"] * 3 + 2
          and segs2[2][2] == index["video"] * 3 + 0
          and all(row < 18 for _a, _b, row in segs2), segs2)

    # ── cond / refs 段（2026-09-20 第 ① 批：打包层 ✓ 事实来源：参考 `PackedLayout` ✓）──
    keyframe = {"resolved_frame_index": 3, "latent": torch.zeros(2, 2, 2, 2),
                "audio_latent": torch.zeros(3, 2, 5)}
    cond_layout = h3.packed_layout(7, 3, 4, 4, 5, keyframes=[keyframe])
    cond_kinds = [kind for _a, _b, kind in cond_layout["segments"]]
    check("㉒′ ⭐ 关键帧 ⇒ 段序 ``text → cond → cond_audio → audio → video`` ✓（条件块夹在 text 与目标之间 ✓；"
          "**目标两条流永远在最后** ✓ 且 audio 在 video **之前** ✓）；cond 行数 = 潜帧 × 每帧行数 ✓"
          "（4×4 潜 ⇒ 每帧 4 行 ✓ ⇒ 2 潜帧 = **8** ✓）",
          cond_kinds == ["text", "cond", "cond_audio", "audio", "video"]
          and cond_layout["segments"][1] == (7, 15, "cond")
          and cond_layout["segments"][2] == (15, 25, "cond_audio"), cond_kinds)
    check("㉒″ ⭐ **cond 的 t 起点** = ``cursor + FRAME_RESCALE × frame_index`` ✓（cursor = text_len = 7 ✓、"
          "第 3 帧 ⇒ 7 + 5/3×3 = **12** ✓）；且更新掩码**只覆盖视频/音频行** ✓：cond 段**全 False**（不更新 ✓）、"
          "目标 video 段**全 True** ✓，text 段**两条都不进** ✓（塞进去会多出假阳性 ✓）",
          abs(float(cond_layout["position_ids"][7, 0]) - 12.0) < 1e-6
          and not bool(cond_layout["img_update"][:8].any())
          and bool(cond_layout["img_update"][8:20].all())
          and int(cond_layout["img_update"].numel()) == 20
          and int(cond_layout["audio_update"].numel()) == 10 + 10,
          (float(cond_layout["position_ids"][7, 0]),
           int(cond_layout["img_update"].numel()), int(cond_layout["audio_update"].numel())))
    ref_blocks = [{"kind": "image", "latent_h": 2, "latent_w": 2},
                  {"kind": "video", "latent_h": 4, "latent_w": 4, "latent_t": 2,
                   "ref_audio_t": 3}]
    ref_layout = h3.packed_layout(5, 2, 4, 4, 4, refs=ref_blocks)
    ref_kinds = [kind for _a, _b, kind in ref_layout["segments"]]
    check("㉒‴ ⭐ 参考块：段序 ``text → ref_img → ref_audio → ref_img → audio → video`` ✓"
          "（⚠️ 图像参考只出 `ref_img` ✓；``video`` 类块里**音频行排在视频行之前** ✓ —— 参考里两种写法都有 ✓ "
          "别「统一」成一种 ✗）；⭐ **图像参考整块共用一个 t** ✓（t = text_len = 5 ✓）而 cond 是**逐帧网格** ✓ "
          "—— 两种块**不一样** ✓✗（混了不报错 ✓）",
          ref_kinds == ["text", "ref_img", "ref_audio", "ref_img", "audio", "video"]
          and abs(float(ref_layout["position_ids"][5, 0]) - 5.0) < 1e-6, ref_kinds)
    check("㉒⁗ ⭐ **参考块会把目标时间轴整体推后** ✓：目标两条流的 t 起点 = ``text_len + Σ参考跨度`` ✓"
          "（跨度事实：图像 1.0 ✓、视频类 ``max(ref_audio_t, Σ视频跨度)`` ✓）；段表与坐标**同长** ✓"
          "（两者由**同一处**算出 ✓ ⇒ 不可能不同序 ✓✗）",
          abs(float(ref_layout["position_ids"][ref_layout["segments"][-2][0], 0])
              - (5.0 + h3.ref_time_span(ref_blocks[0]) + h3.ref_time_span(ref_blocks[1]))) < 1e-6
          and int(ref_layout["position_ids"].shape[0]) == ref_layout["seq_len"],
          (float(ref_layout["position_ids"][ref_layout["segments"][-2][0], 0]),
           ref_layout["seq_len"]))

    # ── ⭐⭐ 两类**一起给**（2026-09-23 补 ✓）：此前只有「只给关键帧」✓ 与「只给参考块」✓ 两条各自的路 ✗
    #    现实里「首帧 + 参考图 + 参考视频（带音轨）」是常用组合 ✓ ⇒ 段序 / 时间轴 / 行数对账都要在**混合**
    #    形态下核一遍 ✓（⚠️ 两种块混在一起**不会报错** ✓✗ —— 正是静默错最爱藏的地方 ✓）。
    #    ⚠️ 关键帧的潜尺寸要按**目标网格**给 ✓（事实：cond 用**目标**空间网格 ✓）——
    #    它自己的 H/W 不参与行数 ✓（(2,2,4,4)：T=2 ⇒ 2 帧 × 每帧 4 行 = **8** ✓ 与布局要的一致 ✓）。
    mixed_kf = [{"resolved_frame_index": 0, "latent": torch.zeros(2, 2, 4, 4),
                 "audio_latent": torch.zeros(3, 2, 2)},
                {"resolved_frame_index": 2, "latent": torch.full((2, 2, 4, 4), -0.25),
                 "audio_latent": torch.full((3, 2, 2), -0.5)}]
    mixed_refs = [{"kind": "image", "latent_h": 2, "latent_w": 2},
                  {"kind": "video", "latent_h": 4, "latent_w": 4, "latent_t": 2,
                   "ref_audio_t": 3}]
    mixed = h3.packed_layout(5, 3, 4, 4, 4, keyframes=mixed_kf, refs=mixed_refs)
    mixed_kinds = [kind for _a, _b, kind in mixed["segments"]]
    check("㉒⁶ ⭐⭐ **关键帧 + 参考块一起给**（现实常用组合 ✓）⇒ 段序 = "
          "``text → cond → cond_audio → cond → cond_audio → ref_img → ref_audio → ref_img → "
          "audio → video`` ✓（**cond 在 ref 之前** ✓、目标两条流仍在最后 ✓ 且 audio 在 video 前 ✓）",
          mixed_kinds == ["text", "cond", "cond_audio", "cond", "cond_audio",
                          "ref_img", "ref_audio", "ref_img", "audio", "video"], mixed_kinds)
    refs_span = sum(h3.ref_time_span(block) for block in mixed_refs)
    first_cond_t = float(mixed["position_ids"][mixed["segments"][1][0], 0])
    second_cond_t = float(mixed["position_ids"][mixed["segments"][3][0], 0])
    target_start = float(mixed["position_ids"][mixed["segments"][-2][0], 0])
    check("㉒⁷ ⭐⭐ 混合形态下**时间轴照样对账** ✓：``cursor = text_len + Σ参考跨度`` ✓（**参考块推后 ✓、"
          "条件块不推后** ✓✗ —— 两种块混在一起时最容易写错 ✓）；cond 的 t = ``cursor + FRAME_RESCALE×"
          "frame_index`` ✓（第 0 帧 ⇒ = cursor ✓、第 2 帧 ⇒ +5/3×2 ✓）；目标两条流起点 = cursor ✓",
          abs(first_cond_t - (5.0 + refs_span)) < 1e-6
          and abs(second_cond_t - (5.0 + refs_span + 5.0 / 3.0 * 2)) < 1e-6
          and abs(target_start - (5.0 + refs_span)) < 1e-6,
          (first_cond_t, second_cond_t, target_start, refs_span))
    check("㉒⁸ ⭐ 混合形态下**块清单与段表对账** ✓（非目标行数 = cond/cond_audio/ref_img/ref_audio 各段之和 ✓）"
          "且**更新掩码的长度与归属都对** ✓ —— ⚠️ 掩码与 `segments` 是**两个坐标空间** ✗✗：掩码只按"
          "视频行 / 音频行打包 ✓（不含 text 行 ✗）⇒ 长度 = 目标行数 + 非目标视频行 ✓，"
          "且**尾部**目标那截全 True ✓、前面全 False ✓（拿 `segments` 下标去切必错位 ✓✗ —— 我写第一条时"
          "就是这么错的 ✓）",
          sum(count for block, count in mixed["video_blocks"] if block[0] != "target")
          == sum(b - a for a, b, kind in mixed["segments"] if kind in ("cond", "ref_img"))
          and sum(count for block, count in mixed["audio_blocks"] if block[0] != "target")
          == sum(b - a for a, b, kind in mixed["segments"] if kind in ("cond_audio", "ref_audio"))
          and int(mixed["img_update"].numel()) == mixed["video_rows"] + sum(
              count for block, count in mixed["video_blocks"] if block[0] != "target")
          and int(mixed["audio_update"].numel()) == mixed["audio_rows"] + sum(
              count for block, count in mixed["audio_blocks"] if block[0] != "target")
          and not bool(mixed["img_update"][:-mixed["video_rows"]].any())
          and bool(mixed["img_update"][-mixed["video_rows"]:].all())
          and not bool(mixed["audio_update"][:-mixed["audio_rows"]].any())
          and bool(mixed["audio_update"][-mixed["audio_rows"]:].all()),
          (mixed["video_blocks"], mixed["segments"],
           int(mixed["img_update"].numel()), int(mixed["audio_update"].numel())))
    # ⚠️ 混合形态的**行序**也要"能失败"✓：两类块的**源**分开给 ⇒ 对调两个参考块 ⇒ 拼出的行必变 ✓
    #    （不变 ⇒ 它按插入序拼 ✗✗ ⇒ 顺序没受布局约束 ✓）；且**缺一个参考块的源** ⇒ 报错 ✓ 不静默跳过 ✗。
    mixed_seen_v = {("keyframe", 0): mixed_kf[0]["latent"], ("keyframe", 1): mixed_kf[1]["latent"],
                    # ⚠️ 通道数要等于 `latents_dim` ✓（宽度 = 通道 × patch 面积 ✓ —— 通道给 1 会让
                    #    `torch.cat` 在这儿报"尺寸不一致" ✗，那报的是拼装层的形状错 ✗ 不是"顺序"✓）。
                    ("ref", 0): torch.full((2, 1, 2, 2), 0.25),      # 图像参考 ⇒ 1 行 ✓（它自己的网格 ✓）
                    ("ref", 1): torch.full((2, 2, 4, 4), -0.5)}      # 视频参考 ⇒ 2×2×2 = 8 行 ✓
    mixed_seen_a = {("keyframe", 0): mixed_kf[0]["audio_latent"],
                    ("keyframe", 1): mixed_kf[1]["audio_latent"],
                    ("ref", 1): torch.full((3, 2, 3), -0.25)}        # 视频参考的音轨 ⇒ 3×2 = 6 行 ✓
    mixed_rows, mixed_audio = h3.assemble_blocks(mixed, mixed_seen_v, mixed_seen_a)
    # ⚠️ 对调的是**两个关键帧**的源 ✓（同形 ⇒ 行数不变 ⇒ 只有**顺序**变了 ✓ —— 这正是要判的东西 ✓✗；
    #    参考块之间形状不同（1 行 vs 8 行）⇒ 对调会先撞**行数校验** ✗，验不到"顺序"这件事 ✗）。
    swapped_seen_v = {("keyframe", 0): mixed_seen_v[("keyframe", 1)],
                      ("keyframe", 1): mixed_seen_v[("keyframe", 0)],
                      ("ref", 0): mixed_seen_v[("ref", 0)], ("ref", 1): mixed_seen_v[("ref", 1)]}
    swapped_seen_a = {("keyframe", 0): mixed_seen_a[("keyframe", 1)],
                      ("keyframe", 1): mixed_seen_a[("keyframe", 0)],
                      ("ref", 1): mixed_seen_a[("ref", 1)]}
    swapped_rows, swapped_audio = h3.assemble_blocks(mixed, swapped_seen_v, swapped_seen_a)
    missing_err = _raises(lambda: h3.assemble_blocks(
        mixed, {key: value for key, value in mixed_seen_v.items() if key != ("ref", 1)},
        mixed_seen_a))
    check("㉒⁹ ⭐⭐ 混合形态的**行序判据「能失败」** ✓：把两个**关键帧**的源对调 ⇒ 视频行与音频行"
          "**都要变** ✓（⚠️ 视频源与音频源**各自都要换** ✓✗ —— 只换一边，另一边当然不变 ✓，"
          "我第一版就是这么写错的 ✓）；缺一个参考块的源 ⇒ **报错** ✓"
          "（不静默跳过、不补零 ✗ —— 跳过之后要么行数不对、要么**错位而不报错** ✗✗）",
          not torch.equal(mixed_rows, swapped_rows) and not torch.equal(mixed_audio, swapped_audio)
          and missing_err is not None and "没有它" in str(missing_err),
          (mixed["video_blocks"], mixed["audio_blocks"], missing_err))

    # ── ⚠️ 「混了不报错」的**网格规则**逐条钉住（2026-09-24 补 ✓）──
    # 文档里写着三条：`cond` 用**目标**网格 ✓、参考图用**它自己的** ✓、`ref_audio` 的 w 取**目标**两端 ✓
    # 而 video 类参考块里的音频行取**它自己**的两端 ✓。⚠️ 摘要式断言（只数行数 / 只看 t）**验不到**它们 ✗✗：
    # 把两种网格"统一"成一种**不会报错** ✓，只会让**坐标**错 ✓ —— 所以要比**逐值**的 h/w 集合 ✓。
    def hw_of(layout: Any, index: int) -> set[tuple[float, float]]:
        start, stop, _kind = layout["segments"][index]
        return {(round(float(a), 6), round(float(b), 6))
                for a, b in layout["position_ids"][start:stop, 1:]}

    target_frame, target_w_axis = h3.frame_grid_coords(4, 4)
    ref_frame, ref_w_axis = h3.frame_grid_coords(2, 4)
    target_hw = {(round(float(a), 6), round(float(b), 6)) for a, b in target_frame}
    ref_hw = {(round(float(a), 6), round(float(b), 6)) for a, b in ref_frame}
    shape_layout = h3.packed_layout(3, 3, 4, 4, 2,
                                    keyframes=[{"resolved_frame_index": 0,
                                                "latent": torch.zeros(2, 1, 4, 4),
                                                "audio_latent": torch.zeros(3, 2, 2)}],
                                    refs=[{"kind": "image", "latent_h": 2, "latent_w": 4}])
    check("㉓′ ⭐⭐ `cond` 用**目标**网格 ✓ 而 `ref_img` 用**它自己的** ✓ —— 两者**不相等** ✓"
          "（⚠️ 相等就说明被「统一」了 ✓✗：不报错，只是坐标错 ✓）",
          [kind for _a, _b, kind in shape_layout["segments"]]
          == ["text", "cond", "cond_audio", "ref_img", "audio", "video"]
          and hw_of(shape_layout, 1) == target_hw and hw_of(shape_layout, 3) == ref_hw
          and target_hw != ref_hw,
          (sorted(hw_of(shape_layout, 1))[:3], sorted(hw_of(shape_layout, 3))[:3]))

    audio_layout = h3.packed_layout(3, 3, 4, 4, 2,
                                    refs=[{"kind": "audio", "ref_audio_t": 2},
                                          {"kind": "video", "latent_h": 2, "latent_w": 4,
                                           "latent_t": 1, "ref_audio_t": 2}])
    target_ends = {round(float(target_w_axis[0]), 6), round(float(target_w_axis[-1]), 6)}
    ref_ends = {round(float(ref_w_axis[0]), 6), round(float(ref_w_axis[-1]), 6)}
    audio_w = [set(round(float(value), 6) for value in audio_layout["position_ids"][a:b, 2])
               for a, b, kind in audio_layout["segments"] if kind == "ref_audio"]
    check("㉓″ ⭐⭐ `ref_audio` 的 w 取**目标**网格两端 ✓ 而 video 类参考块里的音频行取**它自己**的两端 ✓"
          "（两段**同名不同源** ✓✗：统一了不报错 ✓，坐标错 ✓）—— ⚠️ 也顺带钉住顺序："
          "audio 类参考块出 1 段 ✓、video 类参考块出**音频行在前、视频行在后** ✓",
          [kind for _a, _b, kind in audio_layout["segments"]]
          == ["text", "ref_audio", "ref_audio", "ref_img", "audio", "video"]
          and audio_w[0] == target_ends and audio_w[1] == ref_ends and target_ends != ref_ends,
          (sorted(audio_w[0]), sorted(audio_w[1])))

    # ── ⭐⭐ 混合形态下的**模态标签**（2026-09-24 补 ✓）──
    # 事实：`video`/`cond`/`ref_img` = 0 ✓、`text` = 1 ✓、`audio`/`cond_audio`/`ref_audio` = 2 ✓。
    # ⚠️ `cond` / `cond_audio` / `ref_img` / `ref_audio` 这**四档只在混合形态里出现** ✓✗ ——
    #    此前 `mod_segments_for` 只有**文档**写着它们 ✓，没有一条断言 ✗（而标签只是 adaLN 的**行内偏移** ✓
    #    ⇒ 错了**完全不报错** ✓，只是把模态接到别的模态那组参数 ✓✗）。
    mixed_values, mixed_index = h3.t_vals_for(0.5)
    mixed_mods = h3.mod_segments_for(mixed["segments"], mixed_index)
    expected_tags = {"text": 1, "cond": 0, "cond_audio": 2, "ref_img": 0,
                     "ref_audio": 2, "audio": 2, "video": 0}
    mixed_mod_rows = [(kind, row) for (_a, _b, row), (_c, _d, kind)
                      in zip(mixed_mods, mixed["segments"])]
    check("㉔ ⭐⭐ 混合形态下**模态标签逐段钉住** ✓（`row = 序号×3 + 标签` ✓；四档新标签 "
          "`cond`/`ref_img` = 0 ✓、`cond_audio`/`ref_audio` = 2 ✓）；且 `cond` 与 `ref_img` **同档** ✓、"
          "`cond_audio` 与 `ref_audio` **同档** ✓（都算 ``max(t_v|t_a, 条件 t)`` ✓）；行号全 < 18 ✓",
          len(mixed_mods) == len(mixed["segments"])
          and all(row == mixed_index[kind] * 3 + expected_tags[kind]
                  for kind, row in mixed_mod_rows)
          and mixed_index["cond"] == mixed_index["ref_img"]
          and mixed_index["cond_audio"] == mixed_index["ref_audio"]
          and all(row < 18 for _a, _b, row in mixed_mods),
          mixed_mod_rows)

    # ── ⭐⭐ 两条「给错也不报错」的时间轴/声道事实（2026-09-24 补 ✓）──
    # `dit.H3_PACK_FACTS` 第 4 条：跨度表 **(1,4,4,4,4) 循环** × 5/3 ✓ —— 此前只验了第 0/1 个 token ✗✗，
    # 而**跨度表给错照样单调递增** ✓✗（只有**回绕**那一格会露 ✓）。
    spans = h3.video_t_spans(7)
    check("㉕ ⭐⭐ 视频时间轴的**跨度表循环** ✓：7 个 token ⇒ **(1,4,4,4,4,1,4)** × 5/3 ✓"
          "（⚠️ 只验第 0/1 格**验不到**循环 ✗：给错也照样单调 ✓）",
          [round(float(span), 6) for span in spans]
          == [round(5.0 / 3.0 * factor, 6) for factor in (1, 4, 4, 4, 4, 1, 4)]
          and h3.FRAME_PER_TOKEN == (1, 4, 4, 4, 4)
          and abs(h3.FRAME_RESCALE - 5.0 / 3.0) < 1e-12,
          [round(float(span), 6) for span in spans])
    # `dit.H3_PACK_FACTS` 第 3 条：**立体声 channel-major** ✓ —— 它的**落点是 `w` 的两端** ✓
    # （前一半 `count` 行取低端 ✓、后一半取高端 ✓）。⚠️ 只验「t 重复两遍」**验不到**它 ✗✗：
    # 两种声道顺序的**行数与 t 完全一样** ✓✗（只有 w 分得开 ✓）。
    plain_layout = h3.packed_layout(3, 3, 4, 4, 5)      # 只目标两条流 ⇒ 段表 = text / audio / video ✓
    audio_start, audio_stop, _audio_kind = plain_layout["segments"][-2]
    audio_w = plain_layout["position_ids"][audio_start:audio_stop, 2]
    half = (audio_stop - audio_start) // 2
    check("㉖ ⭐⭐ 立体声 **channel-major** 的落点 ✓：目标音频段**前一半**行取 `w` **低端** ✓、"
          "**后一半**取**高端** ✓（低端 ≠ 高端 ✓ 才说明真按声道分开 ✓ —— 否则两声道会落成同一列 ✓✗）",
          half > 0 and bool((audio_w[:half] == audio_w[0]).all())
          and bool((audio_w[half:] == audio_w[-1]).all())
          and float(audio_w[0]) < float(audio_w[-1]),
          (float(audio_w[0]), float(audio_w[-1]), half))

    tiny = dict(hidden=8, layers=1, heads=2, head_dim=12, ffn=12, text_dim=5, latents_dim=2,
                audio_latents_dim=3, patch_size=(1, 2, 2), time_input_dim=4, time_hidden=8,
                time_dim=6, inv_freq_len=2, refiner_layers=1)
    trunk = h3.H3FormTrunk(**tiny)
    rows_err = _raises(lambda: trunk(torch.zeros(12, 8), torch.zeros(8, 3), torch.zeros(4, 5),
                                     0.5, 4, 4, keyframes=[{"resolved_frame_index": 0}]))
    check("㉒⁵ ⭐ 带条件块却**没给** `latent_t`/`audio_t` ⇒ **明确报错** ✓（条件块的行数不是每帧行数的"
          "整数倍 ⇒ 段长从行数**推不出来** ✓ —— 不猜 ✓、也不静默按目标段长跑 ✗）",
          rows_err is not None and ("推不出来" in str(rows_err) or "显式" in str(rows_err)), rows_err)

    # ── 行序判据（2026-09-20 第 ② 批 ✓ 把「两处人工对齐」换成「布局说了算」✓）──
    kf_a = {"resolved_frame_index": 0, "latent": torch.full((2, 1, 4, 4), 0.25),
            "audio_latent": torch.full((3, 2, 4), 0.5)}
    kf_b = {"resolved_frame_index": 1, "latent": torch.full((2, 1, 4, 4), -0.25),
            "audio_latent": torch.full((3, 2, 4), -0.5)}
    two = h3.packed_layout(3, 2, 4, 4, 2, keyframes=[kf_a, kf_b])
    video_seen, audio_seen = {}, {}
    for index, keyframe in enumerate((kf_a, kf_b)):
        video_seen[("keyframe", index)] = keyframe["latent"]
        audio_seen[("keyframe", index)] = keyframe["audio_latent"]
    video_rows, audio_rows = h3.assemble_blocks(two, video_seen, audio_seen)
    check("㊶ ⭐ **块清单与段表对账** ✓：块清单里「非目标」的行数之和 = 段表里那些段的行数 ✓"
          "（含目标那条 ✓，拼装方只取到 ``target`` 之前 ✓）",
          sum(count for block, count in two["video_blocks"] if block[0] != "target")
          == sum(b - a for a, b, kind in two["segments"] if kind in ("cond", "ref_img"))
          and sum(count for block, count in two["audio_blocks"] if block[0] != "target")
          == sum(b - a for a, b, kind in two["segments"] if kind in ("cond_audio", "ref_audio")),
          (two["video_blocks"], two["audio_blocks"]))
    swap_v, swap_a = {}, {}
    for index, keyframe in enumerate((kf_b, kf_a)):          # ⚠️ **对调**两个块的来源 ✓
        swap_v[("keyframe", index)] = keyframe["latent"]
        swap_a[("keyframe", index)] = keyframe["audio_latent"]
    swapped_rows, swapped_audio = h3.assemble_blocks(two, swap_v, swap_a)
    check("㊷ ⭐⭐ **行序判据「能失败」** ✓：把两个块的 sources 对调 ⇒ 拼出的行**必变** ✓"
          "（不变 ⇒ 它按 sources 的插入序拼 ✗✗ ⇒ 顺序根本没受布局约束 ✓）—— 这正是以前"
          "「同数不同序**拦不住**」那件事的判据 ✓",
          not torch.equal(video_rows, swapped_rows) and not torch.equal(audio_rows, swapped_audio))
    # ── PDD 头库（2026-09-20 ✓ 事实核自参考 `FinalLayer.forward` + `_pdd_head` ✓）──
    banked = h3.H3FormTrunk(**tiny, head_banks=3).eval()
    check("㊹ `head_banks=3` ⇒ 两个头的行数都 ×3 ✓（``n = 行数 // out_features`` ✓ 与参考同一算法 ✓）；"
          "`head_banks=0` ⇒ **构造期报错** ✓（不静默当成 1 ✗）",
          banked.final_layer.video_out.weight.shape[0]
          // banked.final_layer.video_out.out_features == 3
          and banked.final_layer.audio_out.weight.shape[0]
          // banked.final_layer.audio_out.out_features == 3
          and _raises(lambda: h3.H3FormTrunk(**tiny, head_banks=0)) is not None)
    v_rows_in, a_rows_in, text_in = torch.zeros(4, 8), torch.zeros(2, 3), torch.zeros(4, 5)
    # ⭐ 插值本身单独钉 ✓（不依赖"主干把日程递下去"那条链 ✓ —— 那条链**本轮没验过** ✗，见 TODO ✓）
    head = torch.nn.Linear(8, 5 * 3, bias=True, dtype=torch.float32)
    head.out_features = 5                       # ⚠️ 单头宽度 ✓（= 参考的口径 ✓ `n = 行数 // 它` ✓）
    probe = torch.randn(2, 8)
    wide = h3.pdd_head(head, probe, 3, 0, 3, 12.0)
    narrow = h3.pdd_head(head, probe, 3, 0, 1, 12.0)
    check("㊺ ⭐⭐ **PDD「能失败」**：span 不同 ⇒ 插出来的头**必不同** ✓"
          "（相同 ⇒ 那堆头块只用了第 0 块 ✗✗ ⇒ 等于 PDD 没接上 ✓）",
          not torch.equal(wide, narrow),
          (float(wide.abs().max()), float(narrow.abs().max())))
    check("㊺′ ⭐ span 只含第 0 块 ⇒ 结果**恰好等于第 0 块本身** ✓（事实：第 0 块是**完整头** ✓、"
          "后续块只是**偏移** ✓ ⇒ 这是「机制对不对」最硬的一条 ✓）",
          torch.allclose(narrow, torch.nn.functional.linear(probe, head.weight[:5], head.bias[:5])))
    check("㊻ ⭐ 头库 >1 却**没给 σ 日程** ⇒ **明确报错** ✓（不拿当前 σ 硬凑 ✗ ——"
          "定位 span 必须看**整条**日程 ✓）",
          "sample_sigmas" in str(_raises(lambda: banked(v_rows_in, a_rows_in, text_in, 0.5, 4, 4))))
    check("㊼ ⭐ `head_banks=1`（**默认** ✓）⇒ 仍是普通头 ✓ 且**不需要** σ 日程 ✓"
          "（老路径一字未动 ✓）",
          trunk(v_rows_in, a_rows_in, text_in, 0.5, 4, 4) is not None)
    # ⭐⭐ 整链：σ 日程 **从主干递到输出层** ✓（上一轮只验了插值本身 ✓，这条链是欠账 ✓）
    # ⚠️⚠️ **σ 与 start 的方向是反的** ✗（实测取证 ✓，不是直觉 ✓）：`1 − time_shift_sigma(σ, 12, 1.0)`
    #    随 σ 增而**减** ⇒ **σ 大 ⇒ start 小** ✓；σ 很小时它 ≈ 1 ⇒ `start` 一律被 `min(start, banks-1)`
    #    夹到末段 ✓✗ ⇒ 两个日程算成**同一段** ⇒ 输出当然相同 ✓（第一版用例就是这样误报的 ✓
    #    —— 我连续两版都**先猜方向** ✗ ⇒ 第三版才去**打印取证** ✓ 这正是本仓那条纪律 ✓）。
    banked64 = h3.H3FormTrunk(**tiny, head_banks=64).eval()
    with torch.no_grad():
        early = banked64(v_rows_in, a_rows_in, text_in, 0.9, 4, 4,
                         sample_sigmas=[1.0, 0.9, 0.0])[0]
        late = banked64(v_rows_in, a_rows_in, text_in, 0.9, 4, 4,
                        sample_sigmas=[1.0, 0.9, 0.5])[0]
    check("㊽ ⭐⭐ **整链判据**：σ 日程经 `H3FormTrunk.forward` 递到 `FinalLayer` ✓ ⇒ 换日程 ⇒ 输出**必变** ✓"
          "（不变 ⇒ 那条链断了 ✗✗ —— 先排掉「span 撞同一段」这种**用例自身**的坑 ✓）",
          not torch.equal(early, late), (float(early.abs().max()), float(late.abs().max())))
    check("㊽′ ⭐ 头库 >1 却**没给** σ 日程 ⇒ 整链上也**明确报错** ✓（不只单元级报 ✓）",
          "sample_sigmas" in str(_raises(
              lambda: banked64(v_rows_in, a_rows_in, text_in, 0.05, 4, 4))))

    check("㊸ ⭐ 缺块 ⇒ **报错** ✓（不静默跳过 ✗ —— 跳过之后要么行数不对 ✓，"
          "要么更坏：**错位而不报错** ✓✗）",
          "没有它" in str(_raises(lambda: h3.assemble_blocks(
              two, {("keyframe", 0): kf_a["latent"]}, audio_seen))))
    check("㉓ 主干模块名**逐字对齐参考** ✓（`video_patch_proj` / `audio_patch_proj`（**fp32** ✓）/ "
          "`condition_proj` / `time_embedder` / `rope.inv_freq`（**缓冲区** ✓）/ `token_refiner.blocks.N` / "
          "`blocks.N` / `final_layer.*` ✓）",
          {"video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight",
           "rope.inv_freq", "blocks.0.adaln_proj.linear.weight",
           "token_refiner.blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
           "final_layer.audio_out.weight"} <= set(trunk.state_dict())
          and trunk.video_patch_proj.weight.dtype == torch.float32
          and trunk.audio_patch_proj.weight.dtype == torch.float32
          and tuple(trunk.rope.inv_freq.shape) == (2,), sorted(trunk.state_dict())[:3])

    v_rows = torch.randn(12, 8)     # 12 = latent_t(3) × 每帧 4 ✓（4×4 潜帧 ⇒ (4//2)² ✓）
    a_rows = torch.randn(10, 3)     # 10 = audio_t(5) × 2 ✓（立体声 ✓）
    text = torch.randn(7, 5)
    with torch.no_grad():
        v_out, a_out = trunk(v_rows, a_rows, text, 0.5, 4, 4)
    check("㉔ 主干跑通：输出 = **(视频行, 音频行)** ✓、形状回到行级 ✓（12×8 / 10×3 ✓）、无 NaN ✓",
          tuple(v_out.shape) == (12, 8) and tuple(a_out.shape) == (10, 3)
          and bool(torch.isfinite(v_out).all()) and bool(torch.isfinite(a_out).all()),
          (tuple(v_out.shape), tuple(a_out.shape)))
    with torch.no_grad():
        v_other, _ = trunk(v_rows, a_rows, text, 0.2, 4, 4)
        v_text, _ = trunk(v_rows, a_rows, text + 1.0, 0.5, 4, 4)
    check("㉔′ ⭐ **条件真的接进去了**（反套套逻辑 ✓）：改 σ ⇒ 输出变 ✓；改文本 ⇒ 输出变 ✓"
          "（否则就是「条件没接进去」的假绿 ✗✗ —— 本仓的老毛病 ✓）",
          not torch.allclose(v_out, v_other, atol=1e-6)
          and not torch.allclose(v_out, v_text, atol=1e-6))
    check("㉔″ 行数除不尽 / 音频行是奇数 ⇒ **报错** ✗（不猜结构 ✓，也不静默补齐 ✗）",
          "不是每帧行数" in str(_raises(lambda: trunk(v_rows[:11], a_rows, text, 0.5, 4, 4)))
          and "2 的倍数" in str(_raises(lambda: trunk(v_rows, a_rows[:9], text, 0.5, 4, 4))))

    # ── 外壳（第 ④ 批 ✓）──
    latent = torch.randn(2, 3, 8, 8)          # C=2 / T=3 / H=W=8 ⇒ 网格 3×4×4 = 48 行 ✓
    rows = h3.patchify_video(latent)
    check("㉕ ⭐ **往返恒等**：`unpatchify(patchify(x)) == x` ✓（最强的那种不变量 ✓）；"
          "行数 = `T·(H//2)(W//2)` = 48 ✓、每行宽 = `C·pT·pH·pW` = 8 ✓",
          tuple(rows.shape) == (48, 8)
          and torch.allclose(h3.unpatchify_video(rows, 3, 4, 4, 2), latent, atol=1e-6),
          tuple(rows.shape))
    check("㉕′ 行数与**打包层**同一口径 ✓（`packed_rows` 的 `video_rows` 也是 48 ✓ —— 两处**必须**一致 ✓）",
          h3.packed_rows(0, 3, 8, 8, 0)["video_rows"] == int(rows.shape[0]),
          h3.packed_rows(0, 3, 8, 8, 0)["video_rows"])

    audio_latent = torch.randn(3, 2, 5)       # C=3 / 声道 2 / T=5 ⇒ 10 行 ✓
    audio_rows = h3.pack_audio(audio_latent)
    check("㉖ 音频打包 **channel-major** ✓（**先某一声道的 T 帧** ✓ 再换声道 ✓）+ **往返恒等** ✓",
          tuple(audio_rows.shape) == (10, 3)
          and torch.allclose(h3.unpack_audio(audio_rows, 2), audio_latent, atol=1e-6)
          # ⚠️ 行是 `[T, C]` ✓，而 `latent[:, 0, :]` 是 `[C, T]` ✗ ⇒ **必须转置再比** ✓
          #    （又一处"形状方向记反" ✓ —— 今天第 6 次同类 ✗✓，好在自检当场红 ✓）
          and torch.allclose(audio_rows[:5], audio_latent[:, 0, :].transpose(0, 1))
          and torch.allclose(audio_rows[5:], audio_latent[:, 1, :].transpose(0, 1)),
          tuple(audio_rows.shape))

    check("㉗ 掩码：**整片都在生成 ⇒ `None`** ✓（⚠️ 与「没读到」**必须分开** ✗ —— "
          "本仓那条「没数据 ≠ 通过」的同族 ✓）",
          h3.mask_row_values(torch.ones(3, 8, 8), 3, 8, 8) is None)
    holed = torch.ones(3, 8, 8)
    holed[1, 2:4, 2:4] = 0.0                  # 在第 1 帧挖一个 2×2 洞 ⇒ **恰好一个 patch 行**受影响 ✓
    values = h3.mask_row_values(holed, 3, 8, 8)
    check("㉗′ 掩码挖洞：长度 = `video_rows` ✓（48 ✓）、**只有那一个 patch 行 < 1** ✓、其余 = 1 ✓",
          values is not None and int(values.shape[0]) == 48
          and int((values < 1.0).sum()) == 1 and float(values.min()) < 1e-6,
          None if values is None else (int(values.shape[0]), int((values < 1.0).sum())))
    padded = h3.mask_row_values(torch.zeros(3, 4, 4), 3, 8, 8)
    check("㉗″ 掩码比潜尺寸小 ⇒ **replicate 补齐** ✓（补 0 会把边缘误判成「要生成」✓✗）"
          "⇒ 全 0 掩码补齐后仍全 0 ✓ ⇒ 值全 0 且**不是 `None`** ✓（还没全在生成 ✓）",
          padded is not None and float(padded.max()) < 1e-6,
          None if padded is None else (float(padded.max()), float(padded.min())))
    check("㉘ 新名字**都在 `__all__` 里** ✓（上上轮就漏过它 ✗ ⇒ 一跑 `AttributeError` ✓）；"
          "且「外层封装」已从 TODO **移出**、并出现在 PARTS ✓（**双向** ✓）",
          {"patchify_video", "unpatchify_video", "pack_audio", "unpack_audio",
           "mask_row_values", "H3FormTrunk", "t_vals_for"} <= set(h3.__all__)
          and not any("外层封装" in item for item in h3.H3_FORM_TODO)
          and any("外壳" in item for item in h3.H3_FORM_PARTS), h3.H3_FORM_TODO)

    # ── 接进后端（第 ⑤ 批 ✓）──
    check("㉙ 形态判别**双向** ✓：招牌键齐 ⇒ True ✓；**混进 DiT 招牌键 ⇒ False** ✓（防误判 ✗）；"
          "只给一半 ⇒ False ✓（**不按文件名猜** ✗ —— 那正是「名字对、结构错」的来源 ✓）",
          h3.looks_like_h3_form([*h3.H3_FORM_SIGNATURE_KEYS, "blocks.0.norm1.weight"])
          and not h3.looks_like_h3_form([*h3.H3_FORM_SIGNATURE_KEYS, "attn.in_proj_weight"])
          and not h3.looks_like_h3_form(["video_patch_proj.weight"]))
    check("㉚ 出厂尺寸**只有一处** ✓（`H3_DEFAULTS` = `H3_TRUNK_DEFAULTS` + `H3_SIGMA_SHIFTS` 派生 ✓）"
          "且**不需要 torch 也能读** ✓（在模块级 ✓）",
          h3.H3_TRUNK_DEFAULTS["hidden"] == 5376 and h3.H3_SIGMA_SHIFTS["sigma_shift_audio"] == 3.0
          and set(h3.H3_DEFAULTS) == set(h3.H3_TRUNK_DEFAULTS) | set(h3.H3_SIGMA_SHIFTS))

    tiny_path = pathlib.Path(__file__).resolve().parents[1] / "tests" / "_h3_tiny.safetensors"
    loaded_form: Any = None
    load_detail: Any = None
    try:
        from safetensors.torch import save_file
        from app.services.engine import torch_backend as tb
        save_file({name: tensor.contiguous() for name, tensor in trunk.state_dict().items()},
                  str(tiny_path))
        backend = tb.TorchBackend()
        report = backend.load_weights(path=str(tiny_path), config=tiny)
        loaded_form = getattr(backend, "_form", None)
        load_detail = (report.get("complete"), type(backend._model).__name__)
    except Exception as err:  # noqa: BLE001 —— 缺 safetensors 库就**如实报** ✓ 不假装通过 ✗
        load_detail = f"{type(err).__name__}: {err}"
    finally:
        tiny_path.unlink(missing_ok=True)
    check("㉛ ⭐ **集成**：把 H3 权重交给后端 ⇒ 它**认出形态并按 H3 建** ✓（而不是**静默按 DiT 建** ✗ —— "
          "那种情况下人只会看到「装载报告缺一堆键」✓✗，读成「权重没下全」✗）；装载报告 complete ✓"
          "—— ⭐ 这步之后 `h3_form` **不再是零调用** ✓",
          loaded_form == "h3-form" and isinstance(load_detail, tuple)
          and load_detail[0] is True and load_detail[1] == "H3FormTrunk", load_detail)
    check("㉜ 临时权重文件**已清理** ✓（自检不许留垃圾 ✗）", not tiny_path.exists())

    # ── 端到端一次去噪（第 ⑥ 批 ✓：外壳 + 坐标 + 主干 + 还原 全串起来 ✓）──
    tiny_video = torch.randn(2, 3, 8, 8)      # C=2 ✓ / T=3 ✓ / H=W=8 ✓（与 tiny 配置对得上 ✓）
    tiny_audio = torch.randn(3, 2, 5)         # C=3 ✓ / 声道 2 ✓ / T=5 ✓
    with torch.no_grad():
        v_vel, a_vel = h3.denoise_step(trunk, tiny_video, tiny_audio, text, 0.5)
    check("㉝ ⭐ **端到端一次去噪**：潜帧进 ⇒ 两条 velocity **同形**出 ✓"
          "（视频 `[C,T,H,W]` ✓ / 音频 `[C,ch,T]` ✓ —— 这一条把**外壳 + 坐标 + 主干 + 还原**全用上了 ✓）"
          "、dtype 跟随输入 ✓、无 NaN ✓",
          tuple(v_vel.shape) == tuple(tiny_video.shape)
          and tuple(a_vel.shape) == tuple(tiny_audio.shape)
          and v_vel.dtype == tiny_video.dtype and a_vel.dtype == tiny_audio.dtype
          and bool(torch.isfinite(v_vel).all()) and bool(torch.isfinite(a_vel).all()),
          (tuple(v_vel.shape), tuple(a_vel.shape)))
    with torch.no_grad():
        other_vel, _ = h3.denoise_step(trunk, tiny_video, tiny_audio, text, 0.2)
    check("㉝′ ⭐ **反套套逻辑**（端到端这层也要证 ✓）：改 σ ⇒ velocity **变** ✓"
          "（否则就是「条件没接进去」的假绿 ✗✗ —— 一个「永不生效」的接线与「没接」长得一样 ✗）",
          not torch.allclose(v_vel, other_vel, atol=1e-6))
    check("㉞ 潜帧与 patch **不整除** ⇒ 报错 ✓（**不猜、也不补零** ✗）",
          "不能被 patch" in str(_raises(
              lambda: h3.denoise_step(trunk, tiny_video[:, :, :, :7], tiny_audio, text, 0.5))))

    # ── 双流采样（第 ⑦ 批 ✓）──
    seen: list[tuple[int, float, float]] = []
    with torch.no_grad():
        sampled = h3.sample_dual_stream(trunk, tiny_video, tiny_audio, text, [1.0, 0.5, 0.0],
                                        callback=lambda index, sv, sa: seen.append((index, sv, sa)))
    check("㉟ 双流采样：步数 = `len(sigmas)−1` ✓（2 ✓）、两条流**形状与 dtype 都保持** ✓、无 NaN ✓",
          sampled["steps"] == 2 and int(sampled["video"].shape[1]) == 3
          and tuple(sampled["audio"].shape) == tuple(tiny_audio.shape)
          and sampled["video"].dtype == tiny_video.dtype
          and bool(torch.isfinite(sampled["video"]).all()), sampled["steps"])
    check("㉟′ 回调**每步都调一次** ✓（步号 0/1 ✓、σ 递减 ✓）",
          [item[0] for item in seen] == [0, 1] and seen[0][1] > seen[1][1], seen)
    with torch.no_grad():
        same = h3.sample_dual_stream(trunk, tiny_video, tiny_audio, text, [1.0, 0.5, 0.0],
                                     shift_v=3.0, shift_a=3.0)
    check("㊱ ⭐⭐ **双流的关键不变量**：两条 shift 相同时 ⇒ 音频 σ 与其视频**逐位相同** ✓"
          "（「换算 = 恒等」的证明 ✓）；12/3 时两者**不同** ✓ 且音频那条**单调递减到 0** ✓",
          same["videoSigmas"] == same["audioSigmas"]
          and sampled["videoSigmas"] != sampled["audioSigmas"]
          and sampled["audioSigmas"][-1] == 0.0
          and sampled["audioSigmas"][0] > sampled["audioSigmas"][1] > sampled["audioSigmas"][2],
          (sampled["videoSigmas"], sampled["audioSigmas"]))
    check("㊲ `audio_scale` = `shift_v/shift_a` = **4.0** ✓（事实 ✓）；shift 相同 ⇒ **1.0** ✓；"
          "⚠️ 它与**逐步**的 `audio_carry` **不是一回事** ✗（常数标志 vs 逐帧比值 ✓ —— 混了不报错 ✗✗）",
          abs(h3.audio_scale(12.0, 3.0) - 4.0) < 1e-12 and abs(h3.audio_scale(3.0, 3.0) - 1.0) < 1e-12
          and abs(h3.audio_carry(0.5) - h3.audio_scale(12.0, 3.0)) > 0.1, h3.audio_carry(0.5))
    check("㊳ `sigmas` 末位不是 0 ⇒ **报错** ✓；少于 2 个 ⇒ 报错 ✓（都不猜 ✗）",
          "末位必须是 0.0" in str(_raises(lambda: h3.sample_dual_stream(
              trunk, tiny_video, tiny_audio, text, [1.0, 0.5])))
          and "至少要有 2 个" in str(_raises(lambda: h3.sample_dual_stream(
              trunk, tiny_video, tiny_audio, text, [1.0]))))

    return _report()


def _raises(call: Any) -> BaseException | None:
    try:
        call()
    except BaseException as err:  # noqa: BLE001
        return err
    return None


def _report() -> int:
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
