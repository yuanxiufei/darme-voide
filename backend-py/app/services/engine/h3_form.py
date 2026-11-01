"""H3 形态的 DiT **积木**（2026-09-20 起 ✓）—— ⚠️ **与 `dit.py` 不是同一个模型** ✗。

为什么**单开一个文件** ✓：H3 是 **2D 打包序列**（``[总行, 通道×patch]`` ✓ **连 batch 维都没有** ✗）
+ **RoPE**（**没有位置表** ✗）+ **每 token 独立 timestep** ✓ + **18 路 adaLN 按 per-token mod-row 分段** ✓；
而 `dit.py` 是 ``(B, N, D)`` + 学习式 `pos_embed` + 单标量 σ ✓ —— 两者**不共用结构** ✗，
硬塞进一个类只会两边都读不懂 ✓。

⚠️⚠️ 本文件的**调用方是 `TorchBackend`（H3 双流路径）** ✓（2026-09-20 起接线完毕 ✓）：
主干前向 ✓ / 打包布局 ✓ / 采样循环 `sample_dual_stream` ✓ / PDD 头库 ✓ / 参考块四类 ✓
（图 ✓ 音 ✓ 视频 ✓ 带音轨视频 ✓ ⇒ `H3_FORM_TODO` 已清零 ✓）。
**还没核过真权重** ✗（机制已实现 ✓ ⇒ 缺口的权威清单在 `torch_backend.PENDING_PARTS` ✓）。

事实来源：`dit.H3_SHAPE_FACTS` / `dit.H3_PACK_FACTS`（2026-09-20 从参考实现读全后核出 ✓）。
⚠️ 本文件的**代码是自己写的** ✓（不抄 ✗），只在**结构事实**（键名 / 形状 / 18 路 / 打包顺序 ✓）上对齐 ✓。
"""
from __future__ import annotations

import math
from typing import Any

__all__ = [
    "H3_FORM_PARTS", "H3_FORM_SIGNATURE_KEYS", "H3_FORM_TODO", "H3_SIGMA_SHIFTS",
    "H3_TRUNK_DEFAULTS", "AdalnProj", "Attention", "DIT_SIGNATURE_KEYS", "DiTBlock",
    "FinalLayer", "FRAME_PER_TOKEN", "FRAME_RESCALE", "RefinerBlock", "RMSNorm", "SwiGLU",
    "TimeEmbedder", "TokenRefiner", "frame_grid_coords", "looks_like_h3_form", "mod_row",
    "packed_rows", "position_ids", "rope_angles", "video_t_grid", "AUDIO_COND_TIMESTEP",
    "H3_DEFAULTS", "H3FormTrunk", "VISUAL_COND_TIMESTEP", "audio_carry", "audio_scale",
    "denoise_step", "mask_row_timesteps", "mask_row_values", "mask_rows_for",
    "mod_segments_for", "pack_audio", "pad_to_patch_size", "patchify_video",
    "rows_unique_index",
    "sample_dual_stream", "t_vals_for", "unpack_audio", "unpatchify_video",
    "audio_grid", "packed_layout", "ref_time_span", "video_grid", "video_t_spans",
    "assemble_blocks", "cond_noise_augment", "pdd_head", "curve_t_emb",
    "H3_VIDEO_OUT_KEY", "video_patch_dim", "head_banks_from_shape",
    "Int8Linear",
]

#: ✅ 本文件**已经有**的东西（每条都**能独立自检** ✓）
H3_FORM_PARTS: tuple[str, ...] = (
    "RMSNorm（自己实现 ✓；含 per-head `q_norm`/`k_norm` ✓）",
    "SwiGLU MLP（`fc1: hidden→2×ffn` ✓ / `fc2: ffn→hidden` ✓ **都无 bias** ✓）",
    "Attention：显式 `qkv_proj`（hidden→3×heads×head_dim ✓）+ `out_proj` ✓ + partial split-half RoPE 接口 ✓",
    "TimeEmbedder：**正弦**（freq_dim→hidden→out ✓ **cos 在前** ✓）",
    "AdalnProj：`expand 6 × modalities 3` = **18** 路 ✓（返回 **6 个**张量 ✓ 按 `[S×3, hidden]` 切 ✓）",
    "DiTBlock：RMSNorm + adaLN **按 mod_segments 分段** ✓ + 注意力 + SwiGLU ✓",
    "TokenRefiner：`blocks`（H3 是 **2** 层 ✓）+ `final_norm` ✓",
    "FinalLayer：RMSNorm + **2 路** adaLN + `video_out` / `audio_out`（**存 fp32** ✓）",
    "打包层（无 refs/关键帧的 t2va 口径 ✓）：`packed_rows` 段表 ✓ + `position_ids`（"
    "text/audio/video 三段坐标 ✓）+ `frame_grid_coords`（面积归一化 ✓ 坐标 **×32** ✓）+ "
    "`video_t_grid`（跨度表 (1,4,4,4,4)×5/3 ✓）",
    "`rope_angles`：**3 轴各占 `inv_freq` 一份** ✓ 按 **t,h,w** 拼 ✓ 再**复制一份** ✓ "
    "（⇒ `rot_dim = 6·len(inv_freq)` ✓，H3 默认 = **96** ✓）；`inv_freq` 是**权重里的缓冲区** ✓ 不自己算 ✗",
    "`mod_row(唯一时间戳序号, 模态)` ⇒ ``序号×3 + 模态`` ✓（与 `AdalnProj` 的 reshape 自洽 ✓）",
    "**主干** `H3FormTrunk` ✓（模块名与参考 `__init__` 逐字对齐 ✓：`video_patch_proj` / "
    "`audio_patch_proj`（**都 fp32** ✓）/ `condition_proj` / `time_embedder` / `rope.inv_freq`"
    "（**缓冲区** ✓ 不自己算 ✗）/ `token_refiner` / `blocks` / `final_layer` ✓）"
    "—— 行级前向 ⇒ **(视频行, 音频行)** ✓ 且**按参考取负** ✓",
    "`H3_DEFAULTS`（H3 出厂尺寸 ✓ 全从参考 `__init__` 抄的**事实** ✓）+ `t_vals_for`"
    "（**唯一时间戳去重** ✓ —— 不去重会让行号全错且不报错 ✗✗）+ `mod_segments_for` ✓ + "
    "`audio_carry`（**`schedules.time_shift_sigma` 的真消费者** ✓）",
    "外壳 `patchify_video` / `unpatchify_video` / `pack_audio` / `unpack_audio` ✓"
    "（**往返恒等** ✓ 是自检里最强的那种不变量 ✓；行序与 `position_ids` **必须同序** ✓ 否则"
    "画面错位而**不报错** ✗）",
    "`mask_row_values` ✓（去噪掩码 → per-2×2 行 ∈[0,1] ✓；**全在生成 ⇒ 返回 `None`** ✓ —— "
    "「没有掩码」与「掩码全是 1」**必须分开** ✗，别混成一个全 1 张量 ✓）",
    "形态判别 `looks_like_h3_form` ✓（**双向**：招牌键齐 ⇒ True ✓、混进 DiT 招牌键 ⇒ False ✓）"
    "+ 出厂尺寸 `H3_TRUNK_DEFAULTS`/`H3_SIGMA_SHIFTS`（**模块级** ✓ 无 torch 也能读 ✓、**只有一处** ✓）",
    "⭐ **后端接线** ✓：`torch_backend.load_weights` 先判形态 ✓ ⇒ H3 权重按 `H3FormTrunk` 建 ✓ "
    "（**不再零调用** ✓✓；在此之前它会被**静默**按 DiT 建 ✗ —— 表面只表现为「装载报告缺一堆键」✗）",
    "⭐⭐ **`refs` 段也接进管线** ✓（2026-09-20 ✓）：参考图 → `TorchBackend._dual_references` ✓"
    "（⚠️ 用**图片自己的**网格 ✓ —— 与 `cond` 用**目标**网格**不同** ✓）；行拼装**收敛到一处** ✓"
    "（`_dual_extra_rows` ✓ 照本文件的段序拼 ✓ ⇒ 顺带把「只敢收一块关键帧」那条限制**解掉**了 ✓）；"
    "管线侧：后端没自述 `references` ⇒ **明确拒绝** ✗（不静默忽略参考图 ✓）",
    "⭐⭐ **cond 段已接进管线** ✓（2026-09-20 ✓）：首帧 → `TorchBackend._dual_keyframes` ✓（图片按 plan 尺寸"
    "编码成潜变量 ✓ ⇒ 用**目标**空间网格 ✓ 与参考一致 ✓）→ `sample_dual` 把 cond 行**前插** ✓ → "
    "`denoise_step`/`sample_dual_stream` 透传 `keyframes` ✓（⚠️ 只传行、不传结构 ⇒ 总行数校验**当场红** ✓"
    "—— 实测抓到过 ✓）⇒ 双流 + 首帧**已跑通** ✓（`engine_dual_stream` ✓ 40/40 ✓）",
    "⭐⭐ **打包布局一处算全** ✓：`packed_layout` ✓（段表 + 坐标 + `img_update`/`audio_update` ✓ —— "
    "`packed_rows`/`position_ids` 都**委托**它 ✓ ⇒ 不可能出现「段表与坐标不同序」✓✗）+ **cond/refs 段** ✓"
    "（关键帧 ✓ / 图像 ✓ / 音频 ✓ / 视频 四类块 ✓；时间轴 cursor 规则逐条核自参考 ✓）"
    "+ ⚠️ **两处静默错的修正** ✓：`t_vals_for` 的 text 档（应跟 video ✓ 不是 0.999 ✗）、"
    "`mod_segments_for` 的模态标签（video/cond/ref_img=0 ✓、text=1 ✓、audio 系=2 ✓ —— 初版**三点全错** ✓✗ "
    "且都**不会报错** ✓）",
    "✅ **条件行噪声增广已补齐** ✓（2026-11-01 ✓，原「已知偏差」销账 ✓）：`cond_noise_augment` ✓ —— "
    "``r = aug·z + (1−aug)·noise`` ✓，``aug`` 默认 ``VISUAL_COND_TIMESTEP = 0.999`` / "
    "``AUDIO_COND_TIMESTEP = 1.0`` ✓；**每个 cond 块都重启同一 RNG 流** ✓（同形块拿到**逐位相同**的噪声 ✓ "
    "—— 这不是巧合 ✓ 是参考的语义 ✓）；音频流的种子是 ``seed + 1`` ✓（两条流**不共用**噪声 ✓）。"
    "入口是 `assemble_blocks` ✓（块序在那里已知 ⇒ 与参考「逐块加噪」同构 ✓；行拼完再加就**不对** ✗）。"
    "⚠️ ``aug >= 1`` ⇒ **原样返回、且不消耗任何随机数** ✓（不是「乘 1 再算一遍」 ✗）。"
    "另：``q/k`` 的 **norm 与 rope 先后**在 2026-10-31 前是写反的 ✗（见 `Attention.forward` ✓）。",
    "⭐⭐ **双流接进管线** ✓（2026-09-20 ✓ 同一天 ✓）：`TorchBackend` 的 "
    "`init_dual_latents` ✓（两条初始噪声 ✓，共用 seed ⇒ 逐位可复现 ✓）+ `sample_dual` ✓"
    "（**直接调本文件的 `sample_dual_stream`** ✓ —— 采样循环只有一份 ✓）+ `_decode_dual` ✓"
    "（视频 / 音频各走各的 VAE ✓）+ `write`（真 mp4 + 真 wav ✓）+ `pipeline` **按后端自述分流** ✓"
    "⇒ 一次 `run_sync` 出**两条真文件** ✓（`engine_dual_stream_test` ✓）",
)

#: ⚠️ **还没有的东西**（**只增不减** ✓ —— 关掉一条就删一条 ✓，与 `dit.H3_GAPS_CLOSED` 同一纪律 ✓）。
#: ✅ **2026-09-20 清零** ✓：参考视频块（``video`` / ``video_audio``）入口已通 ✓
#: （`reference_videos` ✓ + `media.load_video_tensor` / `extract_wav` ✓）；
#: PDD 头库已实现并验证 ✓（`pdd_head` ✓ + `head_banks` 从权重形状自动推断 ✓ ⇒ 不需要「事先知道 n」✓）。
#: ⇒ 再往本文件加**没接线**的东西，必须**先进这张表** ✗（不假装它已经是功能 ✓）。
H3_FORM_TODO: tuple[str, ...] = ()


#: H3 形态的**招牌键**（少了任何一个 ⇒ 就不是它 ✓）；用它们判形态 ✓ 而不是"按文件名猜" ✗
H3_FORM_SIGNATURE_KEYS: tuple[str, ...] = (
    "video_patch_proj.weight", "audio_patch_proj.weight", "rope.inv_freq",
    "final_layer.audio_out.weight", "token_refiner.blocks.0.attn.qkv_proj.weight",
)

#: **DiT 形态的招牌键** ✓ —— 只要有它们，就**一定不是** H3 形态 ✓（防误判 ✓）
DIT_SIGNATURE_KEYS: tuple[str, ...] = (
    "attn.in_proj_weight", "out.weight", "patch.proj.weight", "pos_embed",
)


def looks_like_h3_form(key_names: Any) -> bool:
    """这堆权重键名**是不是** H3 形态 ✓（纯函数 ✓ 不需要 torch ✓）。

    ⚠️ 判据要**双向** ✓：既要**认得出** ✓（招牌键全在 ✓），也要**不误判** ✗
    （出现 DiT 的招牌键 ⇒ 直接 False ✓）。**只按文件名 / 只按大小**判形态都不行 ✗ ——
    那正是"**名字对、结构错**"的来源 ✓（本项目已被它咬过好几次 ✓）。
    """
    names = {str(name) for name in key_names}
    if names & set(DIT_SIGNATURE_KEYS):
        return False
    return set(H3_FORM_SIGNATURE_KEYS) <= names


#: **H3 主干的出厂尺寸** ✓（= `H3FormTrunk` 的构造参数 ✓）—— 事实来源：参考实现的 `__init__` 签名 ✓。
#: ⚠️ 放在**模块级**（而不是工厂里 ✓）：这样**不需要 torch 也能读到** ✓，
#: 而且它是**唯一一处** ✓ —— 装了 torch 之后 `H3_DEFAULTS` 由它派生（避免两份会漂移 ✗）。
H3_TRUNK_DEFAULTS: dict[str, Any] = {
    "hidden": 5376, "layers": 50, "heads": 56, "head_dim": 128, "ffn": 14336,
    "text_dim": 5120, "latents_dim": 24, "audio_latents_dim": 32, "patch_size": (1, 2, 2),
    "time_input_dim": 256, "time_hidden": 5376, "time_dim": 2688, "inv_freq_len": 16,
    "refiner_layers": 2, "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
    #: ⚠️ **时间嵌入的两种变体** ✓（参考实现里是互斥的两支 ✓，见 `h3_keys` 的 `variant` ✓）：
    #: ``False`` = 标准（`time_embedder.proj_in/proj_out` 两层 MLP ✓）；
    #: ``True`` = **曲线**（`adaln_t_table` 查表 ✓）—— 厂商真权重 `minimax_h3_*_int8_convrot`
    #: 就是这一支 ✓（2026-10-29 实测：按标准建 ⇒ 缺 4 个 `time_embedder.*` 键 ✗ + 多一个
    #: `adaln_t_table` ✓✗ ⇒ `complete=False` ⇒ 装载被判「不是完整模型」✗）。
    "use_adaln_curves": False, "adaln_curve_grid": 1025,
}

#: 采样侧的 shift ✓（不在主干构造参数里 ✗ —— 所以单独列 ✓）
H3_SIGMA_SHIFTS: dict[str, float] = {"sigma_shift_video": 12.0, "sigma_shift_audio": 3.0}

#: ``video_out`` 头的键名 ✓（**权重里的事实** ✓ —— 本仓模块名与参考逐字对齐 ✓ ⇒ 键名一致 ✓）。
H3_VIDEO_OUT_KEY = "final_layer.video_out.weight"


def video_patch_dim(latents_dim: int, patch_size: tuple[int, int, int]) -> int:
    """``latents_dim × pT·pH·pW`` ✓（= 视频**行**的宽度 ✓，也就是 ``video_out`` 的单头宽度 ✓）。"""
    return int(latents_dim) * math.prod(int(value) for value in patch_size)


def head_banks_from_shape(shape: Any, single_dim: int) -> int:
    """从 ``video_out`` 的**权重形状**推断 PDD 头库大小 ✓（事实：行数 = ``banks × 单头宽度`` ✓）。

    ⚠️⚠️ 为什么必须**从权重读** ✗：``banks`` 是**检查点的事实** ✓ —— 本仓不知道 ✓ 也不许猜 ✓：
    * 拿默认 ``1`` 去装 ``banks>1`` 的权重 ⇒ **形状不符** ✗（运气好当场报错 ✓，运气不好在别处错 ✓）；
    * 反过来说死一个数 ⇒ 换检查点就错 ✓✗。
    ⚠️ 行数**不是整数倍** ⇒ **报错** ✓（不静默回落到 1 ✗ —— 那会让 PDD **静默失效** ✓✗，
    本文件已经在这上面栽过一次 ✓：见 `FinalLayer.__init__` 里 `out_features` 那条注释 ✓）。
    """
    if int(single_dim) < 1:
        raise ValueError(f"单头宽度必须 ≥ 1（收到 {single_dim} ✗）")
    rows = int(shape[0])
    if rows % int(single_dim):
        raise ValueError(
            f"`video_out` 行数 {rows} 不是单头宽度 {single_dim} 的整数倍 ✗ ⇒ 推断不了头库大小 ✓"
            f"（不猜 ✗ —— 猜错就是「装上了但头库用错」✓✗）")
    return max(1, rows // int(single_dim))


#: 积木缓存 ✓（见 :func:`_parts` ✓ —— 没有它，每次 `h3_form.H3FormTrunk` 都是**新类** ✗）
_PARTS_CACHE: dict[str, Any] | None = None
#: :mod:`quant` 的延迟缓存 ✓（本模块要能「没有 torch 也导入」✓）
_QUANT_CACHE: Any = None


def _build_torch_parts() -> dict[str, Any]:
    """建出全部积木 ✓（**要 torch** ✓ —— 没有就抛 `ImportError` ✓，由调用方决定怎么报 ✓）。"""
    import torch  # noqa: PLC0415 —— 只有真要用 torch 时才 import ✓
    from torch import nn

    class RMSNorm(nn.Module):  # noqa: D101 —— 公开名见 __all__ ✓
        """**RMSNorm**（自己实现 ✓）：``x / sqrt(mean(x²) + eps) * weight`` ✓。

        与 LayerNorm 的差别（H3 全模型用它 ✓）：**不减均值** ✓、**只有 weight 没有 bias** ✓。
        ⚠️ 一律**在 float32 里算** ✓ 再转回原 dtype（与参考实现的"norm 用 fp32"一致 ✓）。
        """

        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Any) -> Any:
            dtype = x.dtype
            value = x.to(torch.float32)
            value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (value * self.weight.to(torch.float32)).to(dtype)

    class TimeEmbedder(nn.Module):  # noqa: D101
        """**正弦时间嵌入**（H3 用这个 ✓，不是 `Linear(1, hidden)` ✗）。

        ``t ∈ [0, 1]`` ✓ ⇒ 频率 ``exp(−ln(10000)·i/(freq_dim/2))`` ✓ ⇒ **cos 在前、sin 在后** ✓
        （顺序在参考实现里是写死的 ✓）⇒ 再 `Linear` + SiLU + `Linear` ✓。
        """

        def __init__(self, freq_dim: int, hidden: int, out: int) -> None:
            super().__init__()
            if freq_dim % 2:
                raise ValueError(f"freq_dim 必须是偶数（收到 {freq_dim} ✗）")
            self.freq_dim = int(freq_dim)
            self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
            self.proj_out = nn.Linear(hidden, out, bias=True)

        def forward(self, t: Any) -> Any:
            half = self.freq_dim // 2
            freqs = torch.exp(-math.log(10000.0)
                              * torch.arange(half, dtype=torch.float32, device=t.device) / half)
            args = t.to(torch.float32).reshape(-1, 1) * freqs.reshape(1, -1)
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # ⚠️ cos 在前 ✓
            return self.proj_out(nn.functional.silu(self.proj_in(emb)))

    class SwiGLU(nn.Module):  # noqa: D101
        """**SwiGLU** MLP ✓：``fc1`` 一次出 ``2×ffn`` ✓ ⇒ 一半当门 ✓ 一半当值 ✓。

        ⚠️ 两个都是**无 bias** ✓（与参考一致 ✓）。
        """

        def __init__(self, hidden: int, ffn: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
            self.fc2 = nn.Linear(ffn, hidden, bias=False)

        def forward(self, x: Any) -> Any:
            gate, value = self.fc1(x).chunk(2, dim=-1)
            return self.fc2(nn.functional.silu(gate) * value)

    class Attention(nn.Module):  # noqa: D101
        """显式注意力 ✓：``qkv_proj`` 一次出三份 ✓ + **per-head** `q_norm`/`k_norm` ✓ + `out_proj` ✓。

        ⚠️ **inner = heads × head_dim** ✓（H3 里 56×128 = **7168 ≠ hidden 5376** ✗ —— 这正是
        不能用 `nn.MultiheadAttention` 的理由 ✓，见 `dit.H3_GAPS_CLOSED` 第 ① 条 ✓）。
        """

        def __init__(self, hidden: int, heads: int, head_dim: int, qk_eps: float = 1e-5) -> None:
            super().__init__()
            self.heads = int(heads)
            self.head_dim = int(head_dim)
            inner = self.heads * self.head_dim
            self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
            self.q_norm = RMSNorm(self.head_dim, qk_eps)
            self.k_norm = RMSNorm(self.head_dim, qk_eps)
            self.out_proj = nn.Linear(inner, hidden, bias=False)

        @staticmethod
        def apply_rope(x: Any, angles: Any, rot_dim: int) -> Any:
            """对每个 head 的**前 ``rot_dim`` 维**做 **split-half** 旋转 ✓（标准做法 ✓）。

            ``x``：``[S, heads, head_dim]`` ✓；``angles``：``[S, rot_dim]`` ✓
            （**前后两半相同** ✓ —— 与参考实现 `rope_rotation_table` 只取 `angles[:, :half]` 一致 ✓）。
            ⚠️ **每个轴分多少对**没核过 ✗ ⇒ 本函数只负责"给定角度怎么转" ✓。
            """
            if rot_dim <= 0:
                return x
            if rot_dim % 2:
                raise ValueError(f"rot_dim 必须是偶数（收到 {rot_dim} ✗）")
            rotated, tail = x[..., :rot_dim], x[..., rot_dim:]
            half = rot_dim // 2
            pair = rotated.reshape(*rotated.shape[:-1], half, 2)
            # ⚠️ 三角函数在 **fp32** 里算 ✓ 再降到 `x.dtype` ✗ —— `angles` 恒 fp32 ✓（见
            #    :func:`rope_angles` ✓），而主干可能是 **bf16** ✓（真权重装载后 ✓）：不降的话
            #    `bf16 × fp32` 会把结果 promote 回 fp32 ✗ ⇒ 下一层线性立刻报
            #    「mat1 and mat2 must have the same dtype, but got Float and BFloat16」✓✗
            #    （报错位置在**下一层** ✓，指不到这里 ✓）。
            cos = torch.cos(angles[:, :half]).reshape(angles.shape[0], 1, half).to(x.dtype)
            sin = torch.sin(angles[:, :half]).reshape(angles.shape[0], 1, half).to(x.dtype)
            first, second = pair[..., 0], pair[..., 1]
            out = torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1)
            return torch.cat([out.reshape(*rotated.shape[:-1], rot_dim), tail], dim=-1).to(x.dtype)

        def forward(self, x: Any, rope_angles: Any = None, rot_dim: int = 0) -> Any:
            count = x.shape[0]
            packed = self.qkv_proj(x)
            inner = self.heads * self.head_dim
            q, k, v = packed.split(inner, dim=-1)
            q = q.reshape(count, self.heads, self.head_dim)
            k = k.reshape(count, self.heads, self.head_dim)
            v = v.reshape(count, self.heads, self.head_dim)
            #: ⚠️ **顺序是 norm → rope** ✓✗（2026-10-31 改 ✓，原先是 rope → norm ✗）。
            #:    参考实现那一步是**融合核** ``ck.rms_rope_split_half`` ✓（名字即顺序 ✓），
            #:    纯 python 的同族实现也写死了 ``q = apply_rope_split_half1(self.norm_q(q), freqs)``
            #:    （``comfy/ldm/ideogram4/model.py`` ✓）。RMS 因子本身对旋转不变 ✓，所以「先转后 norm」
            #:    **只在 norm 权重逐维不等于 1 时**才露出来 ✓：旋转把相邻两维混在一起 ✓，
            #:    先乘 ``w`` 再转 ≠ 先转再乘 ``w`` ✓✗ —— 训过的权重都不可能全 1 ✓，
            #:    症状是视频画质整体发糊、无任何报错 ✓✗。
            q, k = self.q_norm(q), self.k_norm(k)
            if rope_angles is not None:
                q = self.apply_rope(q, rope_angles, rot_dim)
                k = self.apply_rope(k, rope_angles, rot_dim)
            # [S, heads, head_dim] → [heads, S, head_dim] ⇒ 注意力在 S 上（**无 mask** ✓）
            out = nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            return self.out_proj(out.transpose(0, 1).reshape(count, inner))

    class AdalnProj(nn.Module):  # noqa: D101
        """**adaLN 投影** ✓：``expand × hidden × modalities`` 一次出全 ✓（H3：6×3 = **18** ✓）。

        ⚠️ 返回 `expand` 个张量 ✓，每个形状 ``[S × modalities, hidden]`` ✓ —— 即**每个 token
        自带 modalities 个候选行** ✓，由调用方按 `mod_segments` 的 **mod-row** 取用 ✓
        （H3 真形态就是这样按 token 分档的 ✓ —— 不是整批一个 shift/scale ✗）。
        """

        def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int,
                     apply_silu: bool = True, dtype: Any = None) -> None:
            super().__init__()
            self.expand = int(expand)
            self.modalities = int(modalities)
            #: ⚠️ **曲线档必须 ``False``** ✗✗（参考：``apply_silu=not use_adaln_curves`` ✓）——
            #: 曲线档的 ``t_emb`` 是**曲线基系数**（宽 8 ✓）不是时间嵌入 ✗ ⇒ 再套 SiLU 就是把
            #: 系数**非线性压扁** ✓ ⇒ 调制全偏 ✓ 而**不报错** ✓✗（2026-11-01 查出并修 ✓）。
            self.apply_silu = bool(apply_silu)
            self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype)

        def forward(self, t_emb: Any) -> tuple[Any, ...]:
            value = nn.functional.silu(t_emb) if self.apply_silu else t_emb
            value = self.linear(value)
            value = value.reshape(value.shape[0] * self.modalities, -1)
            return tuple(value.chunk(self.expand, dim=-1))

    def _mod_row(vecs: Any, row: Any, dtype: Any) -> Any:
        """取某一行 ✓（``row`` 可以是**整数**也可以是一整条**逐 token 索引** ✓）。

        ⚠️ 这个函数差点被我"编辑掉"✗：上一版替换 `_mod_*` 那两块时，old_str **把它一起圈进来了**
        ✗ ⇒ 定义没了 ✓，而 `compile()` **照样过** ✗（名字是运行期才查的 ✓）⇒ 一跑到 forward 就
        `NameError` ✓。**规则**（本仓第 85 步那条，这次是第三次 ✓）：改完模块级引用，
        **一定要真的 import 一次并跑到那条路径** ✓，只看语法不算 ✓。
        """
        return vecs[row].to(dtype)

    def _segmented(x: Any, segments: Any, other: Any, apply_fn: Any) -> Any:
        """**按段应用** ✓（段外原样 ✓）：``apply_fn(x_seg, other_seg, row)`` ✓，顺序拼接 ✓ **不原地改** ✓。

        ⚠️ 初版我用的是「原地切片赋值」✗ ⇒ 改成拼装 ✓：既不改坏入参 ✓，
        也让「**段外一个字节都没动**」这件事看得见 ✓（顺序拼接天然如此 ✓）。
        """
        pieces: list[Any] = []
        cursor = 0
        for start, stop, row in segments:
            pieces.append(x[cursor:start])
            pieces.append(apply_fn(x[start:stop], None if other is None else other[start:stop], row))
            cursor = stop
        pieces.append(x[cursor:])
        return torch.cat(pieces)

    def _mod_scale_shift(x: Any, shift: Any, scale: Any, segments: Any) -> Any:
        """adaLN 调制 ✓：``h = x·(1+scale) + shift`` ✓（按段取 shift/scale ✓）。"""
        return _segmented(
            x, segments, None,
            lambda h, _other, row: h * (1.0 + _mod_row(scale, row, h.dtype))
            + _mod_row(shift, row, h.dtype))

    def _mod_gate(x: Any, gate: Any, other: Any, segments: Any) -> Any:
        """门控残差 ✓：``out = x + gate ⊙ other`` ✓（按段取门 ✓）。"""
        return _segmented(
            x, segments, other,
            lambda h, other_seg, row: h + _mod_row(gate, row, h.dtype) * other_seg)

    class RefinerBlock(nn.Module):  # noqa: D101
        """refiner 的一层 ✓：**RMSNorm → 自注意力 → 残差** ✓ ×2 ✓（**无 adaLN** ✓ 按参考 ✓）。"""

        def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int,
                     eps: float = 1e-5, qk_eps: float = 1e-5) -> None:
            super().__init__()
            self.norm1 = RMSNorm(hidden, eps)
            self.norm2 = RMSNorm(hidden, eps)
            self.attn = Attention(hidden, heads, head_dim, qk_eps)
            self.mlp = SwiGLU(hidden, ffn)

        def forward(self, x: Any) -> Any:
            # ⚠️ 初版这里被我自己写成一串废算式 ✗✓（`a + x - (a + x) + …` ✗）—— 自审时抓到 ✓，
            #    改成参考的两行 ✓（refiner 层**无 adaLN** ✓、只走 norm→子层→残差 ✓）
            x = self.attn(self.norm1(x)) + x
            return self.mlp(self.norm2(x)) + x

    class TokenRefiner(nn.Module):  # noqa: D101
        """`token_refiner` ✓（键名逐字对齐 H3 ✓：`blocks.N.*` + `final_norm` ✓）。"""

        def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, ffn: int,
                     eps: float = 1e-5, qk_eps: float = 1e-5,
                     final_eps: float | None = None) -> None:
            # ⚠️ `final_eps` 是参考里**独立的一个 eps** ✓（`final_norm_eps` ✓）——
            #    初版我省掉了它 ✗ ⇒ 调用处按参考传 8 个参数时直接 `TypeError` ✓✓（**响亮** ✓）
            super().__init__()
            self.blocks = nn.ModuleList([
                RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
                for _ in range(num_layers)])
            self.final_norm = RMSNorm(hidden, eps if final_eps is None else final_eps)

        def forward(self, x: Any) -> Any:
            for block in self.blocks:
                x = block(x)
            return self.final_norm(x)

    class DiTBlock(nn.Module):  # noqa: D101
        """主干一层 ✓：RMSNorm → adaLN 分段调制 → 注意力 → 门控残差 ✓ ×2 ✓。

        ⚠️ 这里体现 H3 的**逐 token 分档** ✓：`mod_segments` = ``[(start, stop, row)]`` ✓
        （`row` 可为整数或逐 token 索引 ✓）—— 于是**文本 / 条件 / 音频 / 视频**各用各的档 ✓。
        """

        def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, t_dim: int,
                     eps: float = 1e-5, qk_eps: float = 1e-5, modalities: int = 3,
                     apply_silu: bool = True, adaln_dtype: Any = None) -> None:
            super().__init__()
            self.norm1 = RMSNorm(hidden, eps)
            self.norm2 = RMSNorm(hidden, eps)
            self.attn = Attention(hidden, heads, head_dim, qk_eps)
            self.mlp = SwiGLU(hidden, ffn)
            self.adaln_proj = AdalnProj(t_dim, hidden, 6, modalities,
                                        apply_silu=apply_silu, dtype=adaln_dtype)

        def forward(self, x: Any, t_emb: Any, mod_segments: Any,
                    rope_angles: Any = None, rot_dim: int = 0) -> Any:
            # 顺序照参考 ✓：`norm1 → adaLN(shift/scale) → 注意力 → 门控残差` ✓，再对 MLP 来一遍 ✓
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
            h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
            x = _mod_gate(x, gate_msa, self.attn(h, rope_angles, rot_dim), mod_segments)
            h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
            return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    def pdd_head(head: Any, hidden: Any, banks: int, start: int, stop: int,
                 flow_shift: float) -> Any:
        """**PDD 头库**（事实 ✓ 2026-09-20 逐条核自参考 `_pdd_head` ✓）—— **自己实现** ✓ 不抄 ✓。

        机制（事实 ✓）：头权重的行数是 ``banks × out_features`` ✓；**第 0 块是完整头** ✓、
        **后续块是相对它的偏移** ✓ ⇒ 当前这一步落在 ``[start, stop)`` 这几块上 ✓，
        按 ``dt`` 加权求和 ✓（``dt`` 来自把 ``linspace(1, 0, banks+1)`` 经 ``flow_shift`` 变形后的**差分** ✓）。

        ⚠️ ``dt`` 用 **fp64** 算 ✓（插值权重对精度敏感 ✓）再转回张量 dtype ✓。
        ⚠️ 这里的 ``flow_shift`` 是**各自那条流**的 shift ✓（视频 ``shifts[0]`` / 音频 ``shifts[1]`` ✓）；
        而**定位 span** 的 ``time_shift_sigma(σ, shift_v, **1.0**)`` 用的是 **1.0** ✗ —— 两处**别混** ✓
        （混了不报错 ✓✗）。
        """
        grid = torch.linspace(1.0, 0.0, banks + 1, dtype=torch.float64)
        span = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
        weights = (span / span.sum()).to(hidden.dtype)
        rows = head.weight.reshape(banks, -1, head.weight.shape[1])
        biases = head.bias.reshape(banks, -1)
        first = max(start, 1)                       # ⚠️ 第 0 块**恒为基** ✓（不含权重 ✓）
        weight = rows[0] + torch.einsum("n,noi->oi", weights[first - start:], rows[first:stop])
        bias = biases[0] + torch.einsum("n,no->o", weights[first - start:], biases[first:stop])
        return torch.nn.functional.linear(hidden, weight, bias)

    class FinalLayer(nn.Module):  # noqa: D101
        """输出层 ✓：RMSNorm + **2 路** adaLN ✓ + 两个头 ✓（**存 fp32** ✓ 与参考一致 ✓）。

        ⚠️ 视频与音频两个头**形状不同** ✓：视频头出 ``视频行维度`` ✓、音频头出 ``audio_dim`` ✓。
        ⚠️ 头可能是 **PDD 头库**（``head_banks > 1`` ✓ 见 :func:`pdd_head` ✓）—— ``head_banks=1`` 时
        就是普通头 ✓（**本仓默认** ✓ 行为与以前完全一致 ✓）。
        """

        def __init__(self, hidden: int, t_dim: int, video_dim: int, audio_dim: int,
            eps: float = 1e-5, head_banks: int = 1,
            apply_silu: bool = True, adaln_dtype: Any = None) -> None:
            super().__init__()
            if int(head_banks) < 1:
                raise ValueError(f"head_banks 必须 ≥ 1（收到 {head_banks} ✗）")
            self.head_banks = int(head_banks)
            self.norm = RMSNorm(hidden, eps)
            # ⚠️ 曲线档的 `apply_silu` 必须是 **False** ✗（理由见 `AdalnProj.__init__` ✓）——
            #    参考对**输出层**也照传同一个 `curve` 字典 ✓（只改块不改头 ⇒ 头那一路仍是被压扁的 ✓✗）。
            self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1,
                                        apply_silu=apply_silu, dtype=adaln_dtype)
            self.video_out = nn.Linear(hidden, video_dim * self.head_banks, bias=True,
                                       dtype=torch.float32)
            self.audio_out = nn.Linear(hidden, audio_dim * self.head_banks, bias=True,
                                       dtype=torch.float32)
            # ⚠️⚠️ `out_features` 必须**改回「单头宽度」** ✗ —— 参考的算法是
            #    ``n = 行数 // out_features`` ✓，而它构造时给的就是**单头**宽度 ✓
            #    （权重里才有 n× 行 ✓）。本仓若照 `nn.Linear(hidden, vd*banks)` 留下的默认值 ✓
            #    就会恒得 ``n == 1`` ✗ ⇒ **PDD 静默失效** ✓✗（2026-09-20 实测踩到：用例报"没走 PDD" ✓
            #    —— 而且那现象**完全不报错** ✓，只是头库白装了 ✓）。
            if self.head_banks > 1:
                self.video_out.out_features = int(video_dim)
                self.audio_out.out_features = int(audio_dim)

        def forward(self, x: Any, t_emb: Any, video_seg: Any, audio_seg: Any,
                    sigma_v: float | None = None, sample_sigmas: Any = None,
                    shifts: Any = None) -> tuple[Any, Any]:
            shift, scale = self.adaln_proj(t_emb)
            video_start, video_stop, video_row = video_seg
            audio_start, audio_stop, audio_row = audio_seg
            video = (self.norm(x[video_start:video_stop])
                     * (1.0 + _mod_row(scale, video_row, scale.dtype))
                     + _mod_row(shift, video_row, shift.dtype)).to(torch.float32)
            audio = (self.norm(x[audio_start:audio_stop])
                     * (1.0 + _mod_row(scale, audio_row, scale.dtype))
                     + _mod_row(shift, audio_row, shift.dtype)).to(torch.float32)
            banks = int(self.video_out.weight.shape[0] // self.video_out.out_features)
            if banks == 1:
                return self.video_out(video), self.audio_out(audio)
            # ── PDD 头库（事实 ✓ 见模块里的 `pdd_head` ✓）────────────────────────
            audio_banks = int(self.audio_out.weight.shape[0] // self.audio_out.out_features)
            if audio_banks != banks:
                raise ValueError(
                    f"视频/音频两个头库大小不一致 ✗（{banks} vs {audio_banks} ✓）—— 不猜 ✗")
            if sample_sigmas is None:
                raise ValueError(
                    "PDD 头库要按 σ 在**整条日程**里定位 span ✗ ⇒ 必须给 `sample_sigmas` ✓"
                    "（不给就报错 ✓ —— 不拿当前 σ 硬凑一个 ✗）")
            if shifts is None or sigma_v is None:
                raise ValueError("PDD 头库还需要当前 σ 与两条 shift ✗")
            from app.services.engine import schedules  # noqa: PLC0415 —— 同 audio_carry ✓
            schedule = torch.as_tensor(sample_sigmas, dtype=torch.float64)
            index = int((schedule - float(sigma_v)).abs().argmin())
            sigma_next = float(schedule[min(index + 1, int(schedule.shape[0]) - 1)])
            shift_v, shift_a = float(shifts[0]), float(shifts[1])
            # ⚠️ 定位 span 用 **1.0**（不是 shift_a ✗ —— 两处不同 ✓，混了不报错 ✓✗）
            start, stop = (int(round((1.0 - schedules.time_shift_sigma(s, shift_v, 1.0)) * banks))
                           for s in (float(sigma_v), sigma_next))
            start = min(start, banks - 1)
            stop = max(stop, start + 1)
            return (pdd_head(self.video_out, video, banks, start, stop, shift_v),
                    pdd_head(self.audio_out, audio, banks, start, stop, shift_a))

    # ────────────────────── 打包层（2026-09-20 第 ② 批 ✓ 事实来源见 `dit.H3_PACK_FACTS` ✓）──────
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)      # 视频时间轴：每 token 占的帧跨度（循环 ✓）
    FRAME_RESCALE = 5.0 / 3.0              # 该表的缩放 ✓

    def frame_grid_coords(latent_h: int, latent_w: int) -> tuple[Any, Any]:
        """一个潜帧里各 **2×2 patch 行**的 ``(h, w)`` 坐标 ✓ —— **面积归一化** ✓。

        公式（事实 ✓）：``ratio = dim / sqrt(h·w)`` ✓、``n = dim / 2`` ✓、
        ``coord_i = (i·ratio/n + (1−ratio)/2) · 32`` ✓（``endpoint=False`` ✓ ⇒ **不含右端** ✓）。
        ⇒ 正方形潜帧下 **h 与 w 两轴坐标逐位相同** ✓（自检钉住 ✓）；坐标均值略小于 16 ✓（因不入右端 ✓）。
        """
        area = math.sqrt(latent_h * latent_w)
        axes = []
        for dim in (latent_h, latent_w):
            ratio, count = dim / area, dim // 2
            axis = [(index * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0
                    for index in range(count)]
            axes.append(torch.tensor(axis, dtype=torch.float64))
        grid = torch.cartesian_prod(axes[0], axes[1])       # [rows, 2] ✓（外层 h、内层 w ✓）
        return grid, axes[1]

    def video_t_spans(count: int) -> list[float]:
        """视频各 token 的**时间跨度** ✓（跨度表**循环** ✓ —— 见 `FRAME_PER_TOKEN` ✓）。"""
        return [FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(int(count))]

    def video_t_grid(count: int, origin: float) -> Any:
        """视频各 token 的**时间轴起点** ✓：``origin + 前缀和(exclusive)`` ✓（跨度表循环 ✓）。"""
        values, running = [], float(origin)
        for span in video_t_spans(count):
            values.append(running)
            running += span
        return torch.tensor(values, dtype=torch.float64)

    def audio_grid(origin: float, count: int, w_low: float, w_high: float) -> Any:
        """音频坐标 ``[count×2, 3]`` ✓：``t`` 逐潜帧推进且**重复两遍** ✓（立体声 channel-major ✓）、
        ``h = 0`` ✓、``w`` 取给定两端 ✓（**两个声道各一端** ✓）。"""
        grid = torch.zeros(int(count) * 2, 3, dtype=torch.float64)
        grid[:, 0] = (float(origin) + torch.arange(int(count), dtype=torch.float64)).repeat(2)
        grid[:int(count), 2] = float(w_low)
        grid[int(count):, 2] = float(w_high)
        return grid

    def video_grid(count: int, frame: Any, origin: float) -> Any:
        """视频坐标 ``[count×每帧行数, 3]`` ✓：``t`` 用 :func:`video_t_grid` ✓、``h/w`` 用给定网格 ✓。

        ⚠️ 网格**由调用方给** ✓ —— 参考块用的是**它自己的**尺寸网格 ✓（可以不等于目标尺寸 ✓），
        目标两条流用的才是目标网格 ✓。两者混用**不会报错** ✓✗。
        """
        block = torch.empty(int(count), int(frame.shape[0]), 3, dtype=torch.float64)
        block[:, :, 0] = video_t_grid(count, origin)[:, None]
        block[:, :, 1:] = frame[None]
        return block.reshape(-1, 3)

    def ref_time_span(block: Any) -> float:
        """参考块在时间轴上**先占掉**的跨度 ✓（目标两条流的起点要排在所有参考块之后 ✓）。

        事实 ✓（逐条核自参考 `_ref_t_span` ✓）：``image`` ⇒ **1.0** ✓；``audio`` ⇒ ``ref_audio_t`` ✓；
        ``video``/``video_audio`` ⇒ ``max(ref_audio_t, Σ视频跨度)`` ✓。
        """
        kind = str(block.get("kind") or "")
        if kind == "image":
            return 1.0
        if kind == "audio":
            return float(block.get("ref_audio_t") or 0.0)
        if kind in ("video", "video_audio"):
            return max(float(block.get("ref_audio_t") or 0.0),
                       sum(video_t_spans(int(block.get("latent_t") or 0))))
        return 0.0

    def packed_layout(text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int,
                      keyframes: Any = None, refs: Any = None) -> dict[str, Any]:
        """**整条打包布局** ✓（段表 + 坐标 + 更新掩码 ✓）—— `packed_rows` / `position_ids` 都委托它 ✓。

        ⚠️ 为什么要**一处算全** ✗：段表与坐标**必须逐行同序** ✓（不同序 ⇒ 画面错位而**不报错** ✗✗）
        —— 分两个函数各拼一遍迟早漂移 ✓（本文件早先那条教训 ✓）。

        段顺序（**事实** ✓ 2026-09-20 逐字核自参考 `PackedLayout` ✓）::

            text → [cond* / cond_audio*] → [ref_img* / ref_audio*] → audio → video

        ⇒ **目标两条流永远在最后** ✓（本仓早先钉住的 ✓），且**目标时间轴起点排在所有参考块之后** ✓
        （``cursor = text_len + Σ参考跨度`` ✓）。

        三条"混了不报错"的规则（都按事实来 ✓）：
        * ``cond``（关键帧）用**目标空间网格** ✓、``t`` 起点 = ``cursor + FRAME_RESCALE × frame_index`` ✓；
          而 ``ref_img``（图像参考）**整块共用一个 t**（``cursor`` ✓ 每块让 cursor **+1.0** ✓），
          空间网格用**它自己的**尺寸 ✓ —— 两种块**不一样** ✓✗；
        * ``ref_audio`` 的 ``w`` 取**目标**网格两端 ✓；而 ``video``/``video_audio`` 参考块里的音频行
          取**它自己**网格两端 ✓；
        * ``cond_audio`` 排在 ``cond`` **之后** ✓，而 ``video`` 类参考块里音频行排在视频行**之前** ✓
          （参考里就是两种写法 ✓ —— 别"统一"成一种 ✗）。

        更新掩码的**归属**（事实 ✓）：只有 ``img_update`` / ``audio_update`` 两条 ✓，分别只覆盖
        **视频行** 与 **音频行** ✓；``text`` 行**两条都不进** ✓（文本有自己的标签机制 ✓，
        塞进去会让"哪些行不更新"多出一批假阳性 ✓）。

        ⚠️⚠️ **两条掩码与 ``segments`` 不在同一个坐标空间** ✗✗（2026-09-23 补注 ✓ —— 写自检时
        自己就把两者混着用了 ✓✗，值得写清 ✗）：
        ``segments`` 的区间是**序列坐标** ✓（含 text ✓、长度 = ``seq_len`` ✓）；
        而 ``img_update`` 只按**视频行**打包 ✓（长度 = 目标 ``video_rows`` ✓ **加** 非目标里的视频行 ✓）、
        ``audio_update`` 只按**音频行**打包 ✓ —— **都不含 text 行** ✗ ⇒ 拿 ``segments`` 的下标去切掩码
        **必错位** ✓✗（而且不会报错：只会"某些行该更新却没更新" ✓✗）。
        ⇒ 用法：掩码的**尾部** ``video_rows`` / ``audio_rows`` 项对应**目标**两条流 ✓（全 True ✓），
        其余按块序排列 ✓；完整掩码由 :meth:`H3FormTrunk.forward` 里**再拼一次目标**得到 ✓。
        """
        frame, w_axis = frame_grid_coords(latent_h, latent_w)
        frame_rows = int(frame.shape[0])
        target_w = (float(w_axis[0]), float(w_axis[-1]))
        text_len, audio_t, latent_t = int(text_len), int(audio_t), int(latent_t)

        text = torch.zeros(text_len, 3, dtype=torch.float64)
        text[:, 0] = torch.arange(text_len, dtype=torch.float64)
        segments: list[tuple[str, int]] = [("text", text_len)]
        positions: list[Any] = [text]
        img_update: list[Any] = []
        audio_update: list[Any] = []
        #: ⭐ **行序的权威来源** ✓：``[(块标识, 行数), …]`` ✓ —— 拼 rows 的那一侧**照它取** ✓
        #: （块标识 = ``("keyframe", i)`` / ``("ref", i)`` / ``("target", 0)`` ✓）。
        #: ⚠️ 为什么必须有它 ✗：以前「布局一处、拼装另一处」⇒ 同数不同序时总行数校验**拦不住** ✓✗，
        #: 只能靠两处代码人工对齐 ✓（`H3_FORM_TODO` 第 3 条 ✓）⇒ 现在**布局说了算** ✓。
        video_blocks: list[tuple[tuple[str, int], int]] = []
        audio_blocks: list[tuple[tuple[str, int], int]] = []
        #: 当前段属于哪个块 ✓（关键帧/参考块循环里改它 ✓ —— 目标段用初值 ✓）
        current: tuple[str, int] = ("target", 0)
        row = text_len

        def push(kind: str, count: int, grid: Any, stream: str, *, update: bool) -> None:
            """加一段 ✓（``stream`` ∈ ``video`` / ``audio`` / ``none`` ✓ —— 决定进哪条更新掩码 ✓）。"""
            nonlocal row
            segments.append((kind, int(count)))
            positions.append(grid)
            if stream == "video":
                img_update.append(torch.full((int(count),), bool(update)))
                video_blocks.append((current, int(count)))
            elif stream == "audio":
                audio_update.append(torch.full((int(count),), bool(update)))
                audio_blocks.append((current, int(count)))
            row += int(count)

        # 关键帧（cond ✓）与参考块都夹在 text 与目标之间 ✓
        cursor = float(text_len) + sum(ref_time_span(block) for block in (refs or ()))
        for kf_index, keyframe in enumerate(keyframes or ()):
            current = ("keyframe", kf_index)      # ⚠️ 进循环就改块标识 ✓（下面每段都记到它名下 ✓）
            cond_t = cursor + FRAME_RESCALE * float(keyframe["resolved_frame_index"])
            latent = keyframe.get("latent")
            if latent is not None:
                # ⚠️⚠️ 维数**必须显式校验** ✗：参考那边是 **5 维**（``[B,C,T,H,W]`` ✓ ⇒ 帧数是
                #     ``shape[2]`` ✓），而本仓是 **4 维**（``[C,T,h,w]`` ✓ 无 batch ✓ ⇒ 帧数是
                #     ``shape[1]`` ✓）。2026-09-20 实测踩到 ✓：照抄参考取 ``shape[2]`` ⇒ 取到的是
                #     **h**（14 ✓）⇒ 段长算成 686 ✗ —— 好在总行数校验**当场红** ✓✓（要是没有它，
                #     就是"行数静默错位"那种最难查的错 ✗✗）。
                shape = tuple(latent.shape)
                if len(shape) != 4:
                    raise ValueError(
                        f"关键帧潜变量应为 ``(C, T, h, w)``（**无 batch** ✓ 本仓口径 ✓），"
                        f"收到 {shape} ✗ —— 5 维是**参考**的口径 ✓ 别混 ✗")
                frames = int(shape[1])
                push("cond", frames * frame_rows, video_grid(frames, frame, cond_t),
                     "video", update=False)
            audio_latent = keyframe.get("audio_latent")
            if audio_latent is not None:
                count = int(audio_latent.shape[-1])
                push("cond_audio", count * 2, audio_grid(cond_t, count, *target_w),
                     "audio", update=False)

        cursor = float(text_len)
        for ref_index, block in enumerate(refs or ()):
            current = ("ref", ref_index)          # ⚠️ 同理：参考块也带上自己的身份 ✓
            kind = str(block.get("kind") or "")
            if kind == "image":
                ref_frame, _ = frame_grid_coords(int(block["latent_h"]), int(block["latent_w"]))
                count = int(ref_frame.shape[0])
                grid = torch.empty(count, 3, dtype=torch.float64)
                grid[:, 0] = cursor                      # ⚠️ 整块**同一个 t** ✓（不是逐帧网格 ✓）
                grid[:, 1:] = ref_frame
                push("ref_img", count, grid, "video", update=False)
                cursor += 1.0
            elif kind == "audio":
                ref_t = int(block.get("ref_audio_t") or 0)
                if ref_t > 0:
                    push("ref_audio", ref_t * 2, audio_grid(cursor, ref_t, *target_w),
                         "audio", update=False)
                cursor += float(ref_t)
            elif kind in ("video", "video_audio"):
                ref_frame, ref_w = frame_grid_coords(int(block["latent_h"]),
                                                     int(block["latent_w"]))
                ref_t, ref_v = int(block.get("ref_audio_t") or 0), int(block["latent_t"])
                if ref_t > 0:        # ⚠️ 音频行**排在视频行之前** ✓（参考如此 ✓）
                    push("ref_audio", ref_t * 2,
                         audio_grid(cursor, ref_t, float(ref_w[0]), float(ref_w[-1])),
                         "audio", update=False)
                push("ref_img", ref_v * int(ref_frame.shape[0]),
                     video_grid(ref_v, ref_frame, cursor), "video", update=False)
                cursor += max(float(ref_t), sum(video_t_spans(ref_v)))

        # ⚠️ 进目标段前**必须重置块标识** ✗：不然目标两条流会被记到"最后一个关键帧/参考块"名下 ✓✗
        #    ⇒ 拼装方就会去要**那块**的潜变量来当目标 ✓（错得不响 ✓）。
        current = ("target", 0)

        # ── 目标两条流：**audio 在 video 之前** ✓ 且**永远在最后** ✓ ──
        push("audio", audio_t * 2, audio_grid(cursor, audio_t, *target_w), "audio", update=True)
        push("video", latent_t * frame_rows, video_grid(latent_t, frame, cursor),
             "video", update=True)
        # ⚠️ 段表对外是**绝对区间** ``(start, stop, kind)`` ✓（`mod_segments_for` 与 forward 都吃这种 ✓）
        #    —— 内部用 ``(kind, count)`` 只是为了拼装好写 ✓；转换只有**这一处** ✓（别再各自推 ✗）。
        absolute: list[tuple[int, int, str]] = []
        offset = 0
        for kind, count in segments:
            absolute.append((offset, offset + count, kind))
            offset += count
        return {
            "seq_len": row, "segments": tuple(absolute),
            # ⭐ 行序（拼 rows 时照它取 ✓）—— 含目标那条 ✓（拼装方只取到 target 之前 ✓）
            "video_blocks": tuple(video_blocks), "audio_blocks": tuple(audio_blocks),
            "position_ids": torch.cat(positions),
            "img_update": (torch.cat(img_update) if img_update
                           else torch.zeros(0, dtype=torch.bool)),
            "audio_update": (torch.cat(audio_update) if audio_update
                             else torch.zeros(0, dtype=torch.bool)),
            "video_rows": latent_t * frame_rows, "audio_rows": audio_t * 2,
        }

    #: 条件行的 timestep（**事实** ✓ 参考里是写死的常量 ✓；text 归到 visual 那一档 ✓）。
    #: ⚠️ 定义必须**早于** `assemble_blocks` ✗（它的默认值在 def 时求值 ✓ —— 放到下面会
    #: `UnboundLocalError: cannot access local variable` ✓ 实测踩过 ✓）。
    VISUAL_COND_TIMESTEP = 0.999
    AUDIO_COND_TIMESTEP = 1.0

    def cond_noise_augment(rows: Any, aug: float, seed: int) -> Any:
        """条件行的**噪声增广** ✓ ⇒ ``aug·rows + (1−aug)·noise`` ✓（事实来源：参考
        ``_cond_video_rows`` / ``_cond_audio_rows`` ✓）。

        ⚠️ **为什么要做** ✗？条件行（首帧 / 参考图 / 参考音轨）本来是"干净"的 ✓ ⇒ 模型在训练里从没见过
        "干净条件" ⇒ 在推理里直接喂干净行会让它把条件当**已生成内容**来用 ✓✗（症状：首帧与后续帧接不上、
        参考图只被"看一眼"不被驱动 ✓，**不报错** ✓）。参考的做法是把条件行**按 timestep 混一点噪声** ✓
        （``aug`` 就是那个 timestep ✓ 同时用于 `t_vals_for` 的 cond 档 ✓ —— **两者必须用同一个数** ✗✗）。
        ⚠️ **每个块各自重启同一 RNG 流** ✓（参考注释原话 "every condition intentionally restarts the same
        RNG stream" ✓）：同形的两个块因此拿到**逐位相同**的噪声 ✓ —— 这不是写错 ✓，自检把这条钉住 ✓。
        ⚠️ ``aug >= 1`` ⇒ **原样返回、不消耗随机数** ✓（不是"乘 1 再算" ✗ —— 那会白算一遍、还会把
        `Generator` 的状态推进 ✓✗，一旦调用方复用生成器就**静默错** ✓）。
        ⚠️ ``aug`` 越界 ⇒ **当场报错** ✗（它不是步数/σ ✓，是"保留多少原信号" ✓；给 12.0 那种会得到
        12 倍放大的行 ✓✗ 且**不报错** ✓）。
        """
        aug = float(aug)
        if not 0.0 <= aug <= 1.0:
            raise ValueError(
                f"条件行噪声增广的 aug 必须在 [0, 1] 内（收到 {aug} ✗）—— 它是「保留多少原信号」"
                f"（= 参考的 ``VISUAL_COND_TIMESTEP`` / ``AUDIO_COND_TIMESTEP`` ✓），不是步数也不是 σ ✗✗")
        if aug >= 1.0:
            return rows
        # ⚠️ 生成器在 **CPU** ✓、噪声算在 **fp32** ✓（参考逐字如此 ✓）—— 设备靠 ``.to`` 搬 ✓ 不是让
        #    `randn` 直接生在卡上 ✗（那样同 seed 在不同卡/不同 torch 版本上**不保证同值** ✓✗）。
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(tuple(rows.shape), generator=generator, dtype=torch.float32)
        mixed = aug * rows.to(torch.float32) + (1.0 - aug) * noise.to(rows.device)
        return mixed.to(dtype=rows.dtype)

    def assemble_blocks(layout: Any, video_sources: Any, audio_sources: Any,
                        patch_size: tuple[int, int, int] = (1, 2, 2), *,
                        seed: int = 0,
                        visual_cond_aug: float = VISUAL_COND_TIMESTEP,
                        audio_cond_aug: float = AUDIO_COND_TIMESTEP) -> tuple[Any, Any]:
        """按**布局的块序**把条件/参考潜变量拼成行 ✓ ⇒ ``(视频行, 音频行)`` ✓（**不含目标** ✓）。

        ``video_sources`` / ``audio_sources``：``{("keyframe", 0): 潜变量, ("ref", 0): …}`` ✓ ——
        **只提供块**，顺序**不问**调用方 ✓（顺序由 `packed_layout` 的块清单说了算 ✓）。
        ⚠️ 这就是"行序判据"的落点 ✓：以前布局一处、拼装另一处 ⇒ 同数不同序时**总行数校验拦不住** ✗✗
        （只会"画面不对" ✓）⇒ 现在**布局说了算、这里照抄** ✓。
        ⚠️ 缺块 / 块行数对不上 ⇒ **报错** ✗（不静默跳过 ✓ —— 少一截或错位都是最难查的那种 ✓）。

        ⭐ **条件行噪声增广在这里做** ✓（2026-11-01 ✓ 对齐参考 `_cond_video_rows` / `_cond_audio_rows` ✓）：
        因为**块序只有这里知道** ✓ —— 参考是**逐块**加噪、每块重启同一 RNG 流 ✓；行拼成一整条之后
        再加就变成"一条连续流" ✓✗（同形块不再拿到相同噪声 ✓ 而且**不报错** ✓）。
        ``seed`` 默认 0 ✓ 与参考 ``payload.get("seed", 0)`` 同口径 ✓；**音频流的种子是 ``seed + 1``** ✓✗
        （两条流共用种子 ⇒ 音视频条件行吃到相关噪声 ✓ 不报错 ✓）。
        """
        def gather(blocks: Any, sources: Any, convert: Any, label: str,
                   aug: float, aug_seed: int) -> Any:
            rows: list[Any] = []
            for block, count in blocks:
                if block[0] == "target":
                    continue                    # ⚠️ 目标行由调用方自己拼 ✓（这里只管条件/参考 ✓）
                if block not in sources:
                    raise ValueError(
                        f"布局要 {block} 的 {count} 行（{label}）✓，但 sources 里没有它 ✗"
                        f"（不静默跳过 ✗ —— 跳过之后行数对不上，或更坏：**错位而不报错** ✗✗）")
                converted = convert(sources[block])
                if int(converted.shape[0]) != int(count):
                    raise ValueError(
                        f"块 {block} 实际 {int(converted.shape[0])} 行 ✓，布局要 {count} 行（{label}）✗")
                # ⚠️ **逐块**加噪 ✓（每次重新 `manual_seed(aug_seed)` ⇒ 同形块噪声逐位相同 ✓）
                rows.append(cond_noise_augment(converted, aug, aug_seed))
            return torch.cat(rows, dim=0) if rows else None

        return (gather(layout["video_blocks"], video_sources,
                       lambda latent: patchify_video(latent, patch_size), "视频",
                       visual_cond_aug, int(seed)),
                gather(layout["audio_blocks"], audio_sources, pack_audio, "音频",
                       audio_cond_aug, int(seed) + 1))

    def packed_rows(text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int,
                    keyframes: Any = None, refs: Any = None) -> dict[str, Any]:
        """打包序列的**行数**与**段表** ✓（事实来源 `dit.H3_PACK_FACTS` ✓ + `packed_layout` ✓）。

        ⚠️ 段顺序**固定**：``text`` → （条件/参考块）→ ``audio`` → ``video`` ✓
        （**目标音频在视频之前** ✓ —— "最要紧的两条事实"之一 ✓，自检**钉方向** ✓）。
        """
        layout = packed_layout(text_len, latent_t, latent_h, latent_w, audio_t,
                               keyframes=keyframes, refs=refs)
        return {"seq_len": layout["seq_len"], "segments": layout["segments"],
                "video_rows": layout["video_rows"], "audio_rows": layout["audio_rows"]}

    def position_ids(text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int,
                     keyframes: Any = None, refs: Any = None) -> Any:
        """``[S, 3]`` 的 ``(t, h, w)`` 坐标 ✓（与 `packed_rows` **同一处算出** ✓ 不会不同序 ✓）。

        * ``text``：``t = 0…L−1`` ✓、``h = w = 0`` ✓；
        * ``audio``：``t`` 逐潜帧推进 ✓ 且**重复两遍** ✓、``w`` 取两端 ✓（两声道各一端 ✓）；
        * ``video``：``t`` 用 :func:`video_t_grid` ✓、``h/w`` 用 :func:`frame_grid_coords` ✓；
        * ``cond`` / ``ref_img`` / ``ref_audio``：见 :func:`packed_layout` 的三条规则 ✓。

        ⚠️ 时间轴起点：无参考块时 = ``text_len`` ✓；有参考块 ⇒ 目标两条流**整体后移** ✓。
        """
        return packed_layout(text_len, latent_t, latent_h, latent_w, audio_t,
                             keyframes=keyframes, refs=refs)["position_ids"]

    def rope_angles(positions: Any, inv_freq: Any) -> Any:
        """``[S, 3]`` 坐标 → ``[S, 6·len(inv_freq)]`` 旋转角 ✓（**3 轴各占一份** ✓ 顺序 t,h,w ✓）。

        ⚠️ **前后两半逐位相同** ✓ —— 这不是笔误 ✓：split-half 旋转只需前一半 ✓
        （`Attention.apply_rope` 同样只取前一半 ✓），复制一份是为了把张量铺成
        ``forward`` 里 fused kernel 要的长度 ✓。**自检把"两半相同"钉住** ✓，免得有人"顺手去掉"✗。
        ⚠️ ``inv_freq`` 是**权重里的缓冲区** ✓（H3 默认 16 个 ✓）—— 本项目**不自己算**它 ✗
        （不是 ``1/10000^(2i/d)`` 那种通式 ✓ 由检查点给 ✓）。
        """
        # ⚠️ 先 cast 到 **fp32** ✓（事实：参考实现里 `position_ids` 是 float64、进这里先 `.to(float32)` ✓）
        #    —— 不做的话会变成 "Double did not match Float" ✗（自检 ⑮ 当场抓到 ✓），
        #    而且真权重那边 `inv_freq` 本来就是 fp32 ✓ ⇒ 混算没有意义 ✗。
        positions = positions.to(torch.float32)
        inv_freq = inv_freq.to(torch.float32)
        # ⚠️⚠️ **设备必须对齐** ✗（2026-09-25 在 A5000 上实测抓到 ✓）：`positions` 是
        #    `packed_layout` 用**纯常量**拼的坐标表 ✓（`frame_grid_coords` / `video_t_grid` /
        #    `audio_grid` 里那几处 `torch.tensor(...)` 都没有 `device=` ✓）⇒ 它**天然在 CPU** ✓；
        #    而 `inv_freq` 是**模型缓冲区** ✓（`self.rope` 的 buffer ✓ ⇒ 跟着 `.to("cuda")` 走 ✓）。
        #    两者一乘 ⇒ CUDA 上当场 "Expected all tensors to be on the same device, but found
        #    at least two devices, cuda:0 and cpu!" ✗✗（**CPU 上一切正常** ⇒ 此前挡不住 ✓✗，
        #    而报错文案是"设备混用" ✗ —— 指不到"是布局坐标没搬" ✓）。
        #    以**模型那侧**（`inv_freq`）为准 ✓：坐标表是派生物 ✓，跟随模型 ✓ 而不是反过来 ✗。
        positions = positions.to(device=inv_freq.device)
        per_axis = positions.unsqueeze(-1) * inv_freq.reshape(1, 1, -1)   # [S, 3, F] ✓
        time_axis, height_axis, width_axis = per_axis.unbind(dim=1)
        half = torch.cat((time_axis, height_axis, width_axis), dim=-1)    # [S, 3F] ✓ t,h,w ✓
        return torch.cat((half, half), dim=-1)                            # [S, 6F] ✓ 复制 ✓

    def mod_row(unique_t_index: int, modality: int, modalities: int = 3) -> int:
        """adaLN 输出的行号 ✓：``唯一时间戳序号 × modalities + 模态`` ✓。

        事实来源：参考实现里 ``rows_to_mod_index(...) // 3`` ✓ ⇒ 行布局就是
        ``[t0_m0, t0_m1, t0_m2, t1_m0, …]`` ✓（与 `AdalnProj` 的
        ``reshape(S × modalities, -1)`` **自洽** ✓）。
        """
        if not 0 <= modality < modalities:
            raise ValueError(f"modality 必须在 [0, {modalities}) 内（收到 {modality} ✗）")
        return unique_t_index * modalities + modality

    # ─────────────── 主干（2026-09-20 第 ③ 批 ✓ 模块名与参考 `__init__` 逐字对齐 ✓）───────────
    # ⚠️ 由**模块级**那份派生 ✓（**同一规则只留一处** ✗ —— 两份写死迟早漂移 ✓）
    H3_DEFAULTS: dict[str, Any] = {**H3_TRUNK_DEFAULTS, **H3_SIGMA_SHIFTS}
    # ⚠️ ``VISUAL_COND_TIMESTEP`` / ``AUDIO_COND_TIMESTEP`` 在**上面**（`assemble_blocks` 之前）✓ ——
    #    那是唯一一处 ✓；这里再写一遍就是第二份 ✓✗（长得一样也会漂 ✓）。

    def audio_carry(sigma_v: float, shift_v: float = 12.0, shift_a: float = 3.0) -> float:
        """采样器把音频按 ``σ_a/σ_v`` 缩放带着走的倍数 ✓（进出都要还原 ✓）。

        ⚠️ 这就是 `schedules.time_shift_sigma` 的**真消费者** ✓ —— 视频与音频各跑一条 shift
        （H3 = 12.0 / 3.0 ✓），采样器只喂**视频 σ** ✓ ⇒ 音频那一路必须**逐帧换算** ✓。
        不变量（自检钉住 ✓）：``shift_v == shift_a`` ⇒ 倍数恒为 **1** ✓（不换算也不出错的情形 ✓）。
        """
        from app.services.engine import schedules  # noqa: PLC0415 —— 同包内引，避免顶层循环 ✓
        sigma_a = schedules.time_shift_sigma(max(sigma_v, 1e-6), shift_v, shift_a)
        return sigma_a / max(sigma_v, 1e-6)

    def t_vals_for(sigma_v: float, shift_v: float = 12.0, shift_a: float = 3.0,
                   visual_cond_t: float = VISUAL_COND_TIMESTEP,
                   audio_cond_t: float = AUDIO_COND_TIMESTEP, *,
                   video_t: float | None = None, audio_t: float | None = None,
                   extra: Any = ()) -> tuple[Any, dict[str, int]]:
        """各段的时间戳 ``t = 1 − σ`` ✓ ⇒ 去重成 ``t_vals`` ✓ + **段类 → 行号** 的映射 ✓。

        ⚠️⚠️ **档位与排序都是事实** ✓（2026-09-20 逐字核自参考 `MiniMaxH3Model.forward` 的
        ``seg_t`` 与 ``unique_t = sorted(...)`` ✓ —— 本仓初版**两处都猜错**了 ✗✓）：

        * ``text`` 与 ``video`` **同档**（``t_v = 1 − σ_v`` ✓）—— ⚠️ 初版把 text 钉在
          ``VISUAL_COND_TIMESTEP``（0.999）✗✗：那是**条件行**的档 ✓ 不是文本的 ✓；
        * ``audio`` = ``1 − time_shift(σ_v, shift_v, shift_a)`` ✓（见 :func:`audio_carry` ✓）；
        * ``cond`` / ``ref_img`` = ``max(t_v, visual_cond_t)`` ✓；
          ``cond_audio`` / ``ref_audio`` = ``max(t_a, audio_cond_t)`` ✓（条件行**钉在接近 1** ✓）；
        * ⭐ 唯一时间戳列表按 **值升序** ✓（``sorted(set(...))`` ✓）—— ⚠️ 初版写的是"首次出现顺序"✗：
          ⚠️ **老实说**：在本仓现有的段类集合下两者**恰好等价** ✓（因为 ``t_v ≤ t_a ≤ cond`` ✓
          且段类首次出现的顺序也满足这个大小序 ✓ —— 我原以为"正好相反"✗，**没验证过** ✓✗，不再那么说 ✓）。
          换成 ``sorted`` 只为**与参考同构** ✓：将来加段类时两种口径会**静默分叉** ✓✗
          （行号错位不会报错 ✓ —— 与下面那条「模态标签」是同一族 ✓）。

        ⚠️ **去重是必须的** ✗：`AdalnProj` 的行数是「**唯一时间戳** × 模态」✓ ⇒ 不去重会把同一个
        时间戳算成两行 ✓ ⇒ 行号全错 ✗（同样**不报错** ✗）。
        """
        from app.services.engine import schedules  # noqa: PLC0415
        sigma_a = schedules.time_shift_sigma(max(sigma_v, 1e-6), shift_v, shift_a)
        t_v, t_a = 1.0 - float(sigma_v), 1.0 - sigma_a
        wanted = {
            "text": t_v, "video": t_v, "audio": t_a,
            "cond": max(t_v, float(visual_cond_t)),
            "ref_img": max(t_v, float(visual_cond_t)),
            "cond_audio": max(t_a, float(audio_cond_t)),
            "ref_audio": max(t_a, float(audio_cond_t)),
        }
        # ⚠️ **逐行档位的覆盖口** ✓（`mask_row_timesteps` 的产物 ✓）：只改 `video` / `audio` 两档 ✓，
        #    `text` **仍跟** ``t_v`` ✓（参考 `seg_t["text"]` 从不被掩码改 ✓）。
        if video_t is not None:
            wanted["video"] = float(video_t)
        if audio_t is not None:
            wanted["audio"] = float(audio_t)
        #: ⚠️ ``extra``：**逐行**时间戳的全部取值 ✓ —— 不进 `wanted`（那不是段档 ✗）但**必须进
        #: `values` ✓：`AdalnProj` 的行数是「唯一时间戳 × 模态」✓ ⇒ 漏登一个取值 ⇒ 行号全错 ✓✗
        #: （不报错 ✓）。
        values = sorted(set(wanted.values()) | {float(value) for value in extra})
        index = {kind: values.index(value) for kind, value in wanted.items()}
        return torch.tensor(values, dtype=torch.float32), index

    def rows_unique_index(t_vals: Any, rows_t: Any) -> Any:
        """**逐行**时间戳 → **唯一时间戳序号** ✓（``[行]`` 的 ``long`` ✓；参考 ``rows_to_mod_index`` 的
        去标签那一半 ✓）。

        ⚠️ 参考在同处写的是 ``rows_to_mod_index(rows_t, 0) // 3`` ✓ —— ``3`` 是**模态数** ✓，
        ``//3`` 就是"把 ``序号×3+0`` 还原成序号" ✓ ⇒ 这里直接出序号 ✓（少一次乘除 ✓ 语义不变 ✓）。

        ⚠️ 查表口径是 **python float 精确相等** ✓（不是 ``searchsorted`` 近似 ✓）：``rows_t`` 的每个
        取值都由 :func:`t_vals_for` 的 ``extra`` **原样**登记过 ✓（见 `mask_row_timesteps` 的调用点 ✓）
        ⇒ 查不到就是调用方漏登 ✓ ⇒ ``KeyError`` **当场炸** ✓（不静默取近似值 ✗）。
        """
        levels = rows_t.unique()
        table = {float(value): i for i, value in enumerate(t_vals.tolist())}
        try:
            bases = [table[float(value)] for value in levels.tolist()]
        except KeyError as err:
            raise ValueError(
                f"逐行时间戳 {err.args[0]} 不在唯一时间戳表里 ✗（表 = {sorted(table)} ✓）—— "
                "带逐行档位就必须把它的取值登记进 `t_vals_for(extra=...)` ✓（漏登会让行号整体错位 ✓ 不报错 ✗）"
            ) from None
        base = torch.tensor(bases, dtype=torch.long, device=levels.device)
        return base[torch.searchsorted(levels, rows_t)]

    def mod_segments_for(segments: Any, index: dict[str, int],
                         modalities: int = 3, video_rows_t: Any = None,
                         audio_rows_t: Any = None, t_vals: Any = None,
                         text_tags: Any = None) -> Any:
        """把段表变成 adaLN 的 ``mod_segments`` ✓（``row = 唯一序号 × modalities + 标签`` ✓）。

        ⚠️⚠️ **模态标签**（事实 ✓ 2026-09-20 逐字核自参考前向里的 ``seg_tag`` ✓）：
        **video / cond / ref_img = 0** ✓、**text = 1** ✓、**audio / cond_audio / ref_audio = 2** ✓。

        ⚠️ 本仓初版写的是「text=0 ✓ / audio=1 ✓ / video=2」✗ —— **三档全错** ✓✗：标签只是 adaLN 的
        **行内偏移** ✓ ⇒ 错法**完全不报错** ✓，只是在真权重上把每个模态接到**别的模态**那一组参数
        （画面"就是不对" ✓✗）。⇒ 现在按"音频在视频之前"那条事实的同一份来源钉住 ✓。

        ⚠️⭐ ``video_rows_t`` / ``audio_rows_t``（去噪掩码的**逐行**档位 ✓ 见
        :func:`mask_row_timesteps` ✓）：给了就把**那一路目标段**从"整段一个整数行号"换成
        "**逐 token** 的一串行号" ✓ —— 这正是参考 ``rows_to_mod_index`` 的用法 ✓。
        ``t_vals`` 必须同时给 ✓（逐行取值要按它查序号 ✓ 不给就报错 ✓ 不猜 ✗）。

        ⚠️⚠️ ``text_tags``（``h3_prompt.token_tags`` 的产物 ✓）**必须喂** ✓ 否则文本段里那些
        ``IMAGE_PAD`` 会拿到 **text 那一行 adaLN** ✗（它们在参考里是 **video 行（标签 0）** ✓ ——
        见参考 ``for i in range(1, b - a + 1)`` 那段"按标签把文本段切成若干 run"✓）。
        错法**一个数都不错** ✓：段数、行数、形状全对 ✓，只是每个视觉占位 token 的调制参数来自
        文本那组 ⇒ fl2va / ref2va 只要有图/帧就整段偏 ✓✗（2026-11-01 查出：本仓到那天为止
        ``token_tags`` **没有任何消费者** ✗，H3 全线喂的一直是"整段一个 text 行" ✓）。
        """
        tags = {"video": 0, "cond": 0, "ref_img": 0, "text": 1,
                "audio": 2, "cond_audio": 2, "ref_audio": 2}
        per_row = {"video": video_rows_t, "audio": audio_rows_t}
        if any(value is not None for value in per_row.values()):
            if t_vals is None:
                raise ValueError(
                    "给了逐行时间戳就必须同时给 ``t_vals`` ✗（行号要按唯一时间戳表查 ✓ 不猜 ✗）")

        def text_runs(start: int, stop: int) -> list[Any]:
            """按 ``text_tags`` 把文本段切成**标签一致的若干 run** ✓（参考原样的切法 ✓）。

            ⚠️ 参考里判界用的是 ``tags[i]`` —— ``i`` 是**段内偏移** ✓ 不是 ``start + i`` ✗（原样照抄 ✓）。
            段从 0 开始时两者相同 ✓；**段不从 0 起就不同** ✗ ⇒ 那种布局我们这里**直接拒** ✓（不猜 ✗：
            照抄会读到别人的标签位、避开又和参考算得不一样 ✓✗，两边都是错 ⇒ 只好当场报错 ✓）。
            """
            count = stop - start
            values = (list(text_tags.view(-1).tolist()) if hasattr(text_tags, "view")
                      else list(text_tags))
            if start != 0:
                raise ValueError(
                    f"文本段从第 {start} 行起 ✗ —— ``text_tags`` 的参考切法按**段内偏移**取标签 ✓，"
                    "段不从 0 起就会读到别人的标签位 ⇒ 这里不做猜测 ✓（要么 0 起 ✓，要么改布局 ✓）")
            if len(values) < count:
                raise ValueError(
                    f"``text_tags`` 只有 {len(values)} 个 ✗，而文本段是 {count} 行 ✓ —— "
                    "标签与文本必须同源同长 ✓（短了会把后半段全当标签 0 ⇒ 静默偏 ✓✗）")
            bad = sorted({v for v in values[:count] if v not in (0, 1)})
            if bad:
                raise ValueError(
                    f"``text_tags`` 出现 {bad} ✗ —— 只允许 0（video 行 ✓）/ 1（text 行 ✓）；"
                    "2 是 audio 行 ✗（文本段里不该有音频占位 ✓）")
            runs: list[Any] = []
            run_start = 0
            for offset in range(1, count + 1):
                #: ⚠️ 顺序不能换 ✓：``offset == count`` 必须先判 ✓（否则会越界读 ``values[count]`` ✗）
                if offset == count or values[offset] != values[run_start]:
                    runs.append((start + run_start, start + offset,
                                 mod_row(index["text"], values[run_start], modalities)))
                    run_start = offset
            return runs

        out: list[Any] = []
        for start, stop, kind in segments:
            if kind == "text" and text_tags is not None:
                out.extend(text_runs(start, stop))
                continue
            rows_t = per_row.get(kind)
            if rows_t is None:
                out.append((start, stop, mod_row(index[kind], tags[kind], modalities)))
                continue
            if int(rows_t.numel()) != stop - start:
                raise ValueError(
                    f"``{kind}`` 段的逐行时间戳有 {int(rows_t.numel())} 个 ✗，"
                    f"而该段是 {stop - start} 行 ✓ —— 一段一行 ✓ 对不上就是掩码与布局不同源 ✓（不猜 ✗）")
            out.append((start, stop,
                        rows_unique_index(t_vals, rows_t) * modalities + tags[kind]))
        return tuple(out)

    class H3FormTrunk(nn.Module):  # noqa: D101
        """**H3 形态主干** ✓ —— 模块名与参考的 ``__init__`` **逐字对齐** ✓（可直接装它的权重 ✓）。

        ⚠️ 它**不是** `dit.py` 那个 DiT 的变体 ✗ —— 输入是 **2D 行**（``[S, 通道×patch]`` ✓ **无 batch 维** ✗）、
        时间条件是**每 token 的 t** ✓、注意力是 RoPE ✓。已接线进管线 ✓（`TorchBackend` 双流路径 ✓）。

        ⚠️ ``cond``/``refs`` 段 ✓、denoise mask ✓、输出头 bank（PDD ✓）都已实现 ✓；
        ``unpatchify``/``unpack_audio`` 的**外层封装**也已有 ✓（`pipeline` → `write` ✓）。
        ⚠️ 仍未核过**真权重** ✗（机制已实现 ✓ ⇒ 见 `torch_backend.PENDING_PARTS` ✓）。
        """

        def __init__(self, hidden: int = H3_DEFAULTS["hidden"], layers: int = H3_DEFAULTS["layers"],
                     heads: int = H3_DEFAULTS["heads"], head_dim: int = H3_DEFAULTS["head_dim"],
                     ffn: int = H3_DEFAULTS["ffn"], text_dim: int = H3_DEFAULTS["text_dim"],
                     latents_dim: int = H3_DEFAULTS["latents_dim"],
                     audio_latents_dim: int = H3_DEFAULTS["audio_latents_dim"],
                     patch_size: tuple[int, int, int] = H3_DEFAULTS["patch_size"],
                     time_input_dim: int = H3_DEFAULTS["time_input_dim"],
                     time_hidden: int = H3_DEFAULTS["time_hidden"],
                     time_dim: int = H3_DEFAULTS["time_dim"],
                     inv_freq_len: int = H3_DEFAULTS["inv_freq_len"],
                     refiner_layers: int = H3_DEFAULTS["refiner_layers"],
                     norm_eps: float = H3_DEFAULTS["norm_eps"],
                     qk_norm_eps: float = H3_DEFAULTS["qk_norm_eps"],
                     final_norm_eps: float = H3_DEFAULTS["final_norm_eps"],
            modalities: int = 3, head_banks: int = 1,
                    use_adaln_curves: bool = False,
                    adaln_curve_grid: int = 1025) -> None:
            super().__init__()
            #: ⚠️ PDD 头库大小 ✓（``> 1`` 才走 `pdd_head` ✓）—— **默认 1 = 普通头** ✓
            #: ⚠️ 真权重里它是几 **还没核到** ✗ ⇒ 先显式给、由调用方决定 ✓（不猜 ✓）
            self.head_banks = int(head_banks)
            self.patch_size = tuple(patch_size)
            self.latents_dim = int(latents_dim)
            self.audio_latents_dim = int(audio_latents_dim)
            self.modalities = int(modalities)
            video_patch_dim = self.latents_dim * math.prod(self.patch_size)
            self.video_patch_dim = video_patch_dim
            # ⚠️ 两个 patch 投影都**存 fp32** ✓（事实 ✓ —— 与输出头一样是 fp32 岛 ✓）
            self.video_patch_proj = nn.Linear(video_patch_dim, hidden, bias=True,
                                              dtype=torch.float32)
            self.audio_patch_proj = nn.Linear(self.audio_latents_dim, hidden, bias=True,
                                              dtype=torch.float32)
            self.condition_proj = nn.Linear(text_dim, hidden, bias=True)
            #: ⚠️ 两种时间嵌入**互斥** ✓（参考实现就是 if/else ✓）：曲线检查点里**没有**
            #: `time_embedder.*` 那四把权重 ✗，改由 `adaln_t_table` 查表 ✓ ⇒ 按标准建就会
            #: 「缺 4 键 + 多 1 键」✗✓。表和 `rope.inv_freq` 同类 ✓：都是**顶层 buffer** ✓、
            #: 都是 **fp32** ✓（`_prepare_h3_module` 的 fp32 岛扳回逻辑覆盖 buffers ✓）。
            self.use_adaln_curves = bool(use_adaln_curves)
            if self.use_adaln_curves:
                self.register_buffer("adaln_t_table",
                                     torch.empty(int(adaln_curve_grid), int(time_dim),
                                                 dtype=torch.float32))
                self.time_embedder = None            # type: ignore[assignment]
            else:
                self.time_embedder = TimeEmbedder(time_input_dim, time_hidden, time_dim)
            # ⚠️ `inv_freq` 是**权重里的缓冲区** ✓（H3 默认 16 ✓）—— 本项目**不自己算**它 ✗
            self.rope = nn.Module()
            self.rope.register_buffer("inv_freq", torch.ones(inv_freq_len))
            self.rot_dim = 6 * int(inv_freq_len)         # 3 轴 × F × 复制 2 份 ✓
            self.token_refiner = TokenRefiner(refiner_layers, hidden, heads, head_dim, ffn,
                                              norm_eps, qk_norm_eps, final_norm_eps)
            # ⚠️⚠️ **曲线档的 adaLN 必须关 SiLU、且按 fp32 建** ✗✗（参考：`curve = {"apply_silu":
            #    not use_adaln_curves, "adaln_dtype": float32 if curves else dtype}` ✓ 同一份字典
            #    **同时喂块与输出层** ✓）。写错**不报错** ✓ —— 8 维曲线基系数被 SiLU 压扁 ⇒
            #    每个块、每一层的调制全偏 ✓✗（2026-11-01 查出 ✓ 真权重 `ref2va_pruned_int8_convrot`
            #    就是曲线档 ✓：`adaln_t_table [1025, 8]` ✓、无 `time_embedder.*` ✓）。
            curve_style: dict[str, Any] = {
                "apply_silu": not self.use_adaln_curves,
                "adaln_dtype": (torch.float32 if self.use_adaln_curves else None)}
            self.curve_style = dict(curve_style)
            self.blocks = nn.ModuleList([
                DiTBlock(hidden, heads, head_dim, ffn, time_dim, norm_eps, qk_norm_eps,
                         modalities, **curve_style)
                for _ in range(layers)])
            self.final_layer = FinalLayer(hidden, time_dim, video_patch_dim,
                                          self.audio_latents_dim, final_norm_eps,
                                          head_banks=self.head_banks, **curve_style)

        def forward(self, video_rows: Any, audio_rows: Any, text_states: Any, sigma_v: float,
                    latent_h: int, latent_w: int, shift_v: float = 12.0, shift_a: float = 3.0,
                    visual_cond_t: float = VISUAL_COND_TIMESTEP, *,
                    audio_cond_t: float = AUDIO_COND_TIMESTEP,
                    latent_t: int | None = None, audio_t: int | None = None,
                    keyframes: Any = None, refs: Any = None,
                    sample_sigmas: Any = None, denoise_mask: Any = None,
                    text_token_tags: Any = None) -> tuple[Any, Any]:
            """行级前向 ✓ ⇒ ``(视频行, 音频行)`` ✓（**已按参考的约定取负** ✓ velocity ✓）。

            ``video_rows``：``[行, latents_dim×pT·pH·pW]`` ✓（**含条件/参考块的行** ✓ 若有 ✓）、
            ``audio_rows``：``[行, audio_dim]`` ✓、``text_states``：``[L, text_dim]`` ✓
            （**都是 fp32 ✓、都没有 batch 维 ✓**）。

            ``text_token_tags``：``h3_prompt.token_tags`` 的产物 ✓（长度**必须**等于 ``text_states`` 行数 ✓）。
            不给 ⇒ 整段文本用 text 那一行 adaLN ✓（**纯文本提示词就是这样** ✓）；给了 ⇒ 文本段按标签
            切成若干 run ✓（视觉占位 token 走 video 行 ✓ —— 不给而提示词里**真有**视觉块 ⇒ 静默偏 ✓✗）。

            ⚠️ **无** keyframes/refs 时段长从行数推出来 ✓（``latent_t = 视频行 / 每帧行数`` ✓ 除不尽报错 ✓）；
            ⚠️ **有**条件/参考块时**推不出来** ✗（那些块的行数不是每帧行数的整数倍 ✓）⇒ ``latent_t`` /
            ``audio_t`` **必须显式给** ✓（不猜 ✗），然后按 `packed_layout` 算出的**总行数**反向校验 ✓
            （对不上就报错 ✓ —— 这正是"行数与布局脱节"那条静默错的守卫 ✓）。
            """
            text_len = int(text_states.shape[0])
            if text_token_tags is not None:
                got = int(text_token_tags.view(-1).numel() if hasattr(text_token_tags, "view")
                          else len(text_token_tags))
                #: ⚠️ 长度不符**必须当错** ✗：``mod_segments_for`` 只判"够不够" ✓，多出来的那段标签
                #: 会被**静默丢掉** ✓ ⇒ 文本与标签不同源（换了提示词没重算 tags ✓✗）。
                if got != text_len:
                    raise ValueError(
                        f"``text_token_tags`` 有 {got} 个 ✗，而文本是 {text_len} 行 ✓ —— "
                        "标签必须与**这一份**文本同源 ✓（换了提示词就得重算 ✓）")
            video_count = int(video_rows.shape[0])
            audio_count = int(audio_rows.shape[0])
            if audio_count % 2:
                raise ValueError(f"音频行数必须是 2 的倍数（收到 {audio_count} ✗ —— 立体声 ✓）")
            per_frame = (latent_h // 2) * (latent_w // 2)
            if (keyframes or refs) and (latent_t is None or audio_t is None):
                raise ValueError(
                    "带 keyframes/refs 时必须**显式**给 latent_t 与 audio_t ✗"
                    "（条件/参考块的行数不是每帧行数的整数倍 ⇒ 从行数**推不出来** ✓ 不猜 ✗）")
            if latent_t is None:
                if video_count % per_frame:
                    raise ValueError(
                        f"视频行数 {video_count} 不是每帧行数 {per_frame} 的整数倍 ✗"
                        f"（latent {latent_h}×{latent_w} ✓）—— 不猜 ✗")
                latent_t = video_count // per_frame
            if audio_t is None:
                audio_t = audio_count // 2

            layout = packed_layout(text_len, int(latent_t), latent_h, latent_w, int(audio_t),
                                   keyframes=keyframes, refs=refs)
            want_video = int(layout["img_update"].numel())
            want_audio = int(layout["audio_update"].numel())
            if (video_count, audio_count) != (want_video, want_audio):
                raise ValueError(
                    f"行数与布局对不上 ✗：收到 视频 {video_count} / 音频 {audio_count} ✓，"
                    f"布局要 视频 {want_video} / 音频 {want_audio} ✓"
                    f"（条件/参考块的行算上再传 ✓ —— 不猜、也不静默补齐 ✗）")
            # ⚠️⚠️ **去噪掩码 ⇒ 逐行时间戳** ✓✗（2026-11-01 补 ✓，口径逐字核自参考 ✓ 见
            #    :func:`mask_row_timesteps` ✓）：此前本仓**只有采样器那半边**（缩放增量 ✓）⇒
            #    被掩掉的行（如二采的音频流 ✓）档位仍是 ``1−σ`` ✓ ⇒ 主干把它们当**含噪**行 ✓✗
            #    ⇒ 条件变软 ✓ 而日志全绿 ✓。``denoise_mask`` 恒 1 的路径取值**完全不变** ✓。
            segments = layout["segments"]
            video_t = video_rows_t = audio_t_override = audio_rows_t = None
            extra_t: tuple[float, ...] = ()
            if denoise_mask is not None:
                if not isinstance(denoise_mask, (tuple, list)) or len(denoise_mask) != 2:
                    raise ValueError(
                        "denoise_mask 要给 ``(mask_video, mask_audio)`` 两路 ✗（口径同 "
                        "`sample_dual_stream` ✓：超清二采 = video 1 / audio 0 ✓）")
                mask_video, mask_audio = denoise_mask
                video_t, video_rows_t, audio_t_override, audio_rows_t = mask_row_timesteps(
                    sigma_v, mask_video, mask_audio,
                    shift_v=shift_v, shift_a=shift_a,
                    visual_cond_t=visual_cond_t, audio_cond_t=audio_cond_t,
                    latent_t=int(latent_t), latent_h=latent_h, latent_w=latent_w,
                    # ⚠️ 声道数从**目标**音频行数反推 ✓（不能拿 `audio_count` ✗ —— 它含条件/参考块 ✓）
                    audio_channels=want_audio // int(audio_t))
            # ⚠️ 逐行档位里出现过的**每一个**取值都得进唯一时间戳表 ✓（否则行号整体错位 ✓ 不报错 ✗）
            if video_rows_t is not None or audio_rows_t is not None:
                extra_t = tuple(sorted({float(value)
                                        for rows in (video_rows_t, audio_rows_t) if rows is not None
                                        for value in rows.unique().tolist()}))
            t_vals, index = t_vals_for(sigma_v, shift_v, shift_a, visual_cond_t, audio_cond_t,
                                       video_t=video_t, audio_t=audio_t_override,
                                       extra=extra_t)
            mod_segments = mod_segments_for(segments, index, self.modalities,
                                            video_rows_t=video_rows_t,
                                            audio_rows_t=audio_rows_t,                                             t_vals=t_vals,
                                            text_tags=text_token_tags)

            # 组装：文本走 refiner 那一支 ✓；视频/音频行**按段顺序**贴回去 ✓（与参考一致 ✓）
            text_embed = self.token_refiner(self.condition_proj(text_states.to(
                next(self.condition_proj.parameters()).dtype)))
            video_embed = self.video_patch_proj(video_rows.float()).to(text_embed.dtype)
            audio_embed = self.audio_patch_proj(audio_rows.float()).to(text_embed.dtype)
            h = torch.empty(layout["seq_len"], text_embed.shape[-1], dtype=text_embed.dtype,
                            device=text_embed.device)
            voff = aoff = 0
            for start, stop, kind in segments:
                count = stop - start
                if kind == "text":
                    h[start:stop] = text_embed
                elif kind in ("cond", "ref_img", "video"):
                    h[start:stop] = video_embed[voff:voff + count]
                    voff += count
                else:
                    h[start:stop] = audio_embed[aoff:aoff + count]
                    aoff += count

            # ⚠️ 坐标**直接用布局产出的那份** ✓（与段表同一处算出 ✓ ⇒ 不可能"不同序"✓✗）
            angles = rope_angles(layout["position_ids"], self.rope.inv_freq)
            # ⚠️⚠️ **设备要对齐** ✗（2026-09-25 在 A5000 上实测抓到 ✓）：`t_vals` 是
            #    :func:`t_vals_for` 用**纯数值**拼的时间戳表 ✓ ⇒ **天然在 CPU** ✓；
            #    而 `time_embedder` 的权重在**模型设备**上 ✓ ⇒ 直接喂进去在 CUDA 上当场
            #    "Expected all tensors to be on the same device ... mat1 is on cpu, different
            #    from other tensors on cuda:0" ✗✗（**CPU 上一切正常** ✓✗ ⇒ 此前挡不住 ✓✗）。
            #    以 `h`（已在计算设备上 ✓）为准 ✓：时间戳表是派生物 ✓，跟随计算设备 ✓。
            #    ⚠️ **dtype 也要对齐** ✗（2026-10-29 ✓）：`t_vals` 恒 fp32 ✓，而 `time_embedder`
            #    在真权重装载后是 **bf16** ✓ ⇒ 只对齐 device 不对齐 dtype 会在 `proj_in` 报
            #    「mat1 and mat2 must have the same dtype」✗✗。走"喂给它的那层的 dtype"✓
            #    （不是写死 bf16 ✗：synthetic 主干是 fp32 ✓，写死就把那条路弄坏 ✓✗）。
            if self.use_adaln_curves:
                t_emb = curve_t_emb(self.adaln_t_table, t_vals.to(device=h.device)).to(h.dtype)
            else:
                time_dtype = next(self.time_embedder.parameters()).dtype
                t_emb = self.time_embedder(t_vals.to(device=h.device,
                                                     dtype=time_dtype)).to(h.dtype)
            for block in self.blocks:
                h = block(h, t_emb, mod_segments, angles, self.rot_dim)

            # ⚠️⚠️ 输出层的行号与主干**不是一回事** ✗✓：主干是 **18 路**（`序号×3 + 模态` ✓），
            #     而 `FinalLayer` 的 adaLN 是 **1 路**（`AdalnProj(2, 1)` ✓）⇒ 行号就是
            #     **唯一时间戳序号** ✓。初版我把 18 路的行号直接喂过来 ✗ ⇒ `IndexError` ✓✓
            #     （同一份数据、两套编号 ✓ —— 这种错**只有真跑才现形** ✓，而且它**会报错** ✓ 算是走运 ✓）
            video_seg = next((start, stop, index["video"])
                             for start, stop, kind in segments if kind == "video")
            audio_seg = next((start, stop, index["audio"])
                             for start, stop, kind in segments if kind == "audio")
            # ⚠️ 逐行档位时输出层也**按行**取 ✓（参考：``(va, vb, rows_to_mod_index(rows_t, 0) // 3)`` ✓
            #    —— 输出层的 AdalnProj 是 ``modalities=1`` ✓ ⇒ 行号就是**唯一时间戳序号** ✓，
            #    与主干那套 ``序号×3+标签`` 差一个 ``//3`` 的除法 ✓ 这里直接出序号 ✓）
            if video_rows_t is not None:
                video_seg = (video_seg[0], video_seg[1], rows_unique_index(t_vals, video_rows_t))
            if audio_rows_t is not None:
                audio_seg = (audio_seg[0], audio_seg[1], rows_unique_index(t_vals, audio_rows_t))
            # ⚠️ σ 日程与两条 shift 都递给输出层 ✓：`head_banks > 1` 时要靠它们定位 PDD 的 span ✓
            video, audio = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v,
                                            sample_sigmas, (shift_v, shift_a))
            return -video, -audio          # ⚠️ **取负** ✓（参考的 velocity 约定 ✓，不是笔误 ✗）

    # ───────── 外层封装：潜帧 ⇄ 行（2026-09-20 第 ④ 批 ✓ 事实来源见模块 docstring ✓）─────────
    def pad_to_patch_size(latent: Any, patch_size: tuple[int, int, int] = (1, 2, 2),
                          padding_mode: str = "circular") -> Any:
        """``[C, T, H, W]`` 尾部补到 **patch 的整数倍** ✓（口径逐字照参考 ✓ 2026-11-01 读全 ✓）。::

            pad = ()
            for i in range(img.ndim - 2):                       # 每个空间维 ✓
                p = (patch_size[i] - img.shape[i+2] % patch_size[i]) % patch_size[i]
                pad = (0, p) + pad                              # ⚠️ **前置**拼接 ⇒ 倒序 ✓
            F.pad(img, pad, mode="circular")

        ⚠️ 参考吃 ``[B, C, T, H, W]``（``ndim-2 == 3`` ✓），本仓没有 batch 维 ✓ 是 ``[C, T, H, W]``
        （``ndim-1 == 3`` ✓）—— 空间维**个数一样** ✓，只是下标起点差一格 ✓。
        ⚠️ **补边不裁**：参考在 `_forward` **末尾**才 ``video_out[:, :, :orig_t, :orig_h, :orig_w]``
        裁回（`model.py` 781 行 ✓）⇒ 补出来的那些行**真的参与了注意力与 RoPE** ✓，
        不是"补完就丢"✓✗ —— 只补不参与等于白补 ✓（而只裁不补则尺寸对不上 ⇒ 会当场报错 ✓）。
        ⚠️ 这里必须 **circular** ✓（参考默认值 ✓）：replicate 会在补出来的那一列/行上把边缘
        **整条拖出去** ⇒ 画面边缘一条竖痕 ✓✗；而 mask 那边参考**有意用 replicate** ✓
        （`model.py` 的 mask 补齐 ✓）—— **两处模式不同，别统一** ✗。
        ⚠️ 追踪/脚本化时 circular 不可用 ⇒ 参考降级成 ``reflect`` ✓（照抄 ✓）。
        """
        if not patch_size:
            return latent
        pads: tuple[int, ...] = ()
        for axis in range(int(latent.dim()) - 1):
            patch = int(patch_size[axis])
            if patch <= 1:
                extra = 0
            else:
                extra = (patch - int(latent.shape[axis + 1]) % patch) % patch
            pads = (0, extra) + pads
        if not any(pads[1::2]):
            return latent
        if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
            padding_mode = "reflect"
        return torch.nn.functional.pad(latent, pads, mode=padding_mode)

    def patchify_video(latent: Any, patch_size: tuple[int, int, int] = (1, 2, 2)) -> Any:
        """``[C, T, H, W]`` → ``[T·h·w, C·pT·pH·pW]`` ✓（行序 = **先帧、后行列** ✓）。

        ⚠️ 本仓的行级表示**没有 batch 维** ✓（与 `H3FormTrunk` 一致 ✓）—— 参考实现是把 batch
        折进行数的（``B·t·h·w`` ✓），但它内部处处只取第 0 条 ✓ ⇒ 这里**只收单条** ✓ 不假装支持批 ✗。
        ⚠️ 维度命名（免得看错 ✓）：``pT/pH/pW`` = **patch 尺寸** ✓；``fT/fH/fW`` = **网格数** ✓。
        行序必须与 :func:`position_ids` 的 video 段一致（``fT·fH·fW`` ✓），否则画面**错位但不报错** ✗。
        """
        channels, frames, height, width = latent.shape
        patch_t, patch_h, patch_w = patch_size
        grid_t, grid_h, grid_w = frames // patch_t, height // patch_h, width // patch_w
        view = latent.reshape(channels, grid_t, patch_t, grid_h, patch_h, grid_w, patch_w)
        # → (grid_t, grid_h, grid_w, channels, patch_t, patch_h, patch_w) ✓ 与参考同序 ✓
        view = view.permute(1, 3, 5, 0, 2, 4, 6)
        return view.reshape(grid_t * grid_h * grid_w, -1)

    def unpatchify_video(rows: Any, frames: int, height: int, width: int, channels: int,
                         patch_size: tuple[int, int, int] = (1, 2, 2)) -> Any:
        """`:func:`patchify_video` 的**逆** ✓（自检用「往返恒等」钉住 ✓✓ 最强的那种不变量 ✓）。

        ``height/width`` 传的是**潜帧网格数** ✓（``H // pH`` ✓ —— 参考的调用就是 ``lat_h // 2`` ✓）。
        """
        patch_t, patch_h, patch_w = patch_size
        view = rows.reshape(frames, height, width, channels, patch_t, patch_h, patch_w)
        view = view.permute(3, 0, 4, 1, 5, 2, 6)
        return view.reshape(channels, frames * patch_t, height * patch_h, width * patch_w)

    def curve_t_emb(table: Any, t_vals: Any) -> Any:
        """**曲线变体**的时间嵌入 ✓：`adaln_t_table` 查表 + **线性插值** ✓。

        口径**逐字**照参考实现 ✓（``comfy/ldm/minimax/model.py`` ✓ 2026-10-29 读全 ✓）::

            pos = clamp(t, 0, 1) * (grid - 1)          # 落在 [0, grid-1] 的**分数**格点 ✓
            i0  = floor(pos).clamp(max=grid - 2)       # t=1.0 时不许把 i0+1 读出去 ✓
            row = lerp(table[i0], table[i0+1], pos-i0) # 两行按小数部分**混** ✓

        ⚠️⚠️ 必须**线性插值** ✗ 不许四舍五入到最近格 ✓：曲线在相邻格之间是连续的 ✓，
        取整会让**同一个 σ 在不同步数下落到不同行** ✓✗（症状＝相邻帧亮度跳变、成片闪烁 ✓）；
        而且取整后「两个不同 σ ⇒ 同一行」在数值上是**静默**的 ✗✗（不报错、只是画质掉 ✓）。

        ⚠️ `t` 的语义是 **σ 归一化到 [0,1]** ✓（不是步序号 ✗）—— 由 :func:`t_vals_for` 产出 ✓。
        """
        grid = int(table.shape[0])
        pos = t_vals.to(dtype=torch.float32).clamp(0.0, 1.0) * (grid - 1)
        low = pos.floor().long().clamp(max=grid - 2)
        return torch.lerp(table[low], table[low + 1],
                          (pos - low).unsqueeze(1).to(table.dtype))

    def pack_audio(latent: Any) -> Any:
        """``[C, ch, T]`` → ``[ch·T, C]`` ✓ **channel-major** ✓（``ch0`` 的 T 帧在前 ✓ 再 ``ch1`` ✓）。

        参考里是 ``latent[0].permute(1, 2, 0).reshape(ch·t, c)`` ✓ —— 即 ``[C, ch, T] → [ch, T, C]`` ✓
        ⇒ 顺序 = **先声道、后时间** ✓（这一点决定了音频行与坐标的对应 ✓ 弄反了**不会报错** ✗）。
        """
        channels, channel_count, frames = latent.shape
        return latent.permute(1, 2, 0).reshape(channel_count * frames, channels)

    def unpack_audio(rows: Any, channel_count: int = 2) -> Any:
        """`:func:`pack_audio` 的**逆** ✓ ⇒ ``[C, ch, T]`` ✓（往返恒等 ✓）。"""
        frames = int(rows.shape[0]) // channel_count
        return rows.reshape(channel_count, frames, rows.shape[-1]).permute(2, 0, 1)

    def mask_row_values(mask: Any, latent_t: int, latent_h: int, latent_w: int) -> Any:
        """去噪掩码 → **每个 2×2 patch 行**一个 ``[0, 1]`` 值 ✓；**全都在生成 ⇒ 返回 ``None``** ✓。

        事实 ✓：掩码是 ``[T, H, W]``（1 = 生成 ✓）；先 **replicate 补齐**到潜尺寸 ✓，
        再按 ``(t, h//2, 2, w//2, 2)`` 取 **amax** ✓ ⇒ 平坦化成 ``[video_rows]`` ✓
        （行数**必然等于** `packed_rows` 的 ``video_rows`` ✓ —— 自检交叉验证 ✓）。
        ⚠️ ``None`` 的语义是「**整片都在生成** ⇒ 不需要掩码」✓ —— 不是"没读到"✗
        （本仓那条"**没读到 ≠ 通过**"的同族 ✓：这里必须**分清**「没有掩码」与「掩码全是 1」✓，
        所以返回 ``None`` 而不是一张全 1 的张量 ✓，免得下游把两者混为一谈 ✗）。
        """
        padded = torch.nn.functional.pad(
            mask, (0, latent_w - int(mask.shape[-1]), 0, latent_h - int(mask.shape[-2])),
            mode="replicate")
        folded = padded.reshape(latent_t, latent_h // 2, 2, latent_w // 2, 2).amax(dim=(2, 4))
        values = folded.reshape(-1)
        if bool((values >= 1.0 - 1e-3).all()):
            return None
        return values

    def mask_rows_for(mask: Any, want: int, shape: tuple[int, ...], name: str) -> Any:
        """把**可 broadcast 的**掩码归一到 ``want`` 维 ✓（``[C,T,H,W] → [T,H,W]`` ✓ / ``[C,ch,T] → [ch,T]`` ✓）。

        ⚠️ 口径**照参考**：它取的是 ``denoise_mask[0, 0]`` ✓（= **第 0 条 batch 的第 0 个通道** ✓）
        —— 即「掩码逐通道一致」是上游约定 ✓。本仓没有 batch 维 ✓ ⇒ 只剥**通道**那几维 ✓
        （剥到 ``want`` 维为止 ✓）。``[T]`` / ``[ch]`` 这种**少一维**的写法先补 1 维 ✓。
        剩下的「帧/高」「声道/时间」广播到 `shape` ✓（``[1,H,W]`` → ``[T,H,W]`` ✓）。
        """
        if not hasattr(mask, "shape"):
            raise ValueError(f"{name} 掩码不是张量（{type(mask).__name__} ✗）")
        value = mask
        while int(value.dim()) > want:
            value = value[0]
        while int(value.dim()) < want:
            value = value.unsqueeze(0)
        try:
            return torch.broadcast_to(value, shape)
        except RuntimeError as err:
            raise ValueError(
                f"{name} 掩码 broadcast 不到 {tuple(shape)} ✗（收到 {tuple(int(v) for v in mask.shape)} ✓）："
                f"{err} ✓") from None

    def mask_row_timesteps(sigma_v: float, mask_video: Any = None, mask_audio: Any = None, *,
                           shift_v: float = 12.0, shift_a: float = 3.0,
                           visual_cond_t: float = VISUAL_COND_TIMESTEP,
                           audio_cond_t: float = AUDIO_COND_TIMESTEP,
                           latent_t: int, latent_h: int, latent_w: int,
                           audio_channels: int = 2) -> tuple[Any, Any, Any, Any]:
        """去噪掩码 → **逐行时间戳** ✓ ⇒ ``(video_t, video_rows_t, audio_t, audio_rows_t)`` ✓。

        ## 为什么这一条必须做 ✗✗（口径逐字核自参考 ``MiniMaxH3Model.forward`` ✓ 2026-11-01）

        参考对 ``denoise_mask`` 做了**两件事** ✓，本仓此前只做了**第二件** ✗：

        ① **改行的 adaLN 时间戳** ✓ —— 掩码值 ``m`` 把该行放在 ``σ = m·σ_流`` 上 ✓ ⇒ 它的档位是
           ``1 − m·σ`` ✓（``clamp(max=条件档)`` 管住"完全保留"的那些行 ✓）；
        ② 把**增量**按掩码缩放 ✓（本仓 `sample_dual_stream` 早就做了 ✓）。

        只做 ② 的症状是**完全不报错** ✓✗：``audio 流 mask=0`` 时那一片**数值冻结**住了 ✓（②起了作用 ✓），
        但主干仍按 ``t_a = 1−σ_a`` 把它们当**含噪**行看待 ✗ ⇒ 二采里音频本该是「已定稿的干净音轨」
        （参考给的档位是 ``t = 1.0`` ✓），却被当成"当前 σ 下的噪声样本" ✓✗ ⇒ 口型对齐的
        条件信号是**软的** ✓（画质/口型都掉 ✓ 而日志全绿 ✓）。

        ⚠️ ``m`` 恒 1 的路（一采主路 ✓）**取值完全不变** ✓：``1 − 1·σ = t_v`` ✓ 恰好就是原来的档 ✓
        ⇒ 这条改动只对"真的有行被掩掉"的路径生效 ✓（= 二采 ✓）。
        ``audio_channels`` 是音频行数的一半（``pack_audio`` 的 ``ch`` ✓）。
        """
        from app.services.engine import schedules  # noqa: PLC0415 —— 同 `audio_carry` ✓

        sigma_a = schedules.time_shift_sigma(max(float(sigma_v), 1e-6), shift_v, shift_a)
        t_a = 1.0 - sigma_a
        t_pin_v = max(1.0 - float(sigma_v), float(visual_cond_t))
        t_pin_a = max(t_a, float(audio_cond_t))
        video_t: float | None = None
        video_rows_t: Any = None
        audio_t: float | None = None
        audio_rows_t: Any = None
        if mask_video is not None:
            rows_mask = mask_rows_for(mask_video, 3, (int(latent_t), int(latent_h), int(latent_w)),
                                      "视频")
            m = mask_row_values(rows_mask.to(torch.float32), latent_t, latent_h, latent_w)
            if m is not None:
                # ⚠️ σ 与掩码**必须同设备** ✓（掩码在 CPU、σ 是 python float ⇒ 用一个 float 乘 ✓
                #    天然不分设备 ✓；参考那边 σ 是张量、所以写了 ``.to(m.device)`` ✓）
                rows_t = (1.0 - m * float(sigma_v)).clamp(max=t_pin_v)
                if rows_t.unique().numel() == 1:
                    video_t = float(rows_t[0])
                else:
                    video_rows_t = rows_t
        if mask_audio is not None:
            # ⚠️ 音频的每帧行数 = ``ch`` ✓、帧数 = 掩码末维 ✓（`[..., ch, T]` ✓ 同 `pack_audio` ✓）
            frames = int(mask_audio.shape[-1])
            rows_mask = mask_rows_for(mask_audio, 2, (int(audio_channels), frames), "音频")
            m = rows_mask.to(torch.float32).reshape(-1)      # ⚠️ channel-major ✓ 同 `pack_audio` ✓
            # ⚠️ 与视频那路**不对称**是参考原样 ✓：这边要显式挡「全 1」✓（视频那路的
            #    `mask_row_values` 自己会返回 None ✓）—— 少了这个判断就多算一个无用的档 ✓（无害但不忠实 ✗）
            if not bool((m >= 1.0 - 1e-3).all()):
                rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
                if rows_t.unique().numel() == 1:
                    audio_t = float(rows_t[0])
                else:
                    audio_rows_t = rows_t
        return video_t, video_rows_t, audio_t, audio_rows_t

    def denoise_step(trunk: Any, video_latent: Any, audio_latent: Any, text_states: Any,
                     sigma_v: float, *, shift_v: float = 12.0, shift_a: float = 3.0,
                     visual_cond_t: float = VISUAL_COND_TIMESTEP,
                     audio_cond_t: float = AUDIO_COND_TIMESTEP,
                     extra_video_rows: Any = None, extra_audio_rows: Any = None,
                     keyframes: Any = None, refs: Any = None,
                     sample_sigmas: Any = None, denoise_mask: Any = None,
                     text_token_tags: Any = None) -> tuple[Any, Any]:
        """**一次去噪**：潜帧进 ⇒ **两条 velocity 出** ✓（形状与输入**同形** ✓）。

                     ⚠️ 这个函数是"**把整条链串起来**"的那一处 ✓ —— 它一次用掉了
                     `patchify_video` ✓ / `pack_audio` ✓ / `H3FormTrunk` ✓ / `unpatchify_video` ✓ / `unpack_audio` ✓：
                     潜帧 ⇄ 行、坐标与 RoPE 在主干内部 ✓、最后**还原回潜帧形状** ✓。

                     * ``video_latent``：``[C, T, H, W]`` ✓ —— ⚠️ **不必被 patch 整除** ✓：
                      进函数先按 :func:`pad_to_patch_size`（circular）补到整数倍 ✓、出函数末尾裁回原点 ✓
                      （与参考 606 / 781 行同义 ✓）；形状**永远**与输入一致 ✓；
                     * ``audio_latent``：``[C, ch, T]`` ✓（``ch`` 就是声道数 ✓ —— 只有它进 `unpack_audio` ✓）；
                     * ``text_states``：``[L, text_dim]`` ✓；
                     * ``denoise_mask``：``(mask_video, mask_audio)`` ✓ **只用来定"逐行时间戳"** ✓ ——
                       ⚠️ **增量**的掩码缩放**不在**这里 ✗（那是采样器的活 ✓ 见 `sample_dual_stream` ✓），
                       两处**必须都给**才等于参考语义 ✓（只给采样器那半边 ⇒ 掩掉的行档位错 ⇒ 不报错、条件变软 ✓✗）；
                     * ⚠️ 输出 dtype **跟随输入** ✓（主干内部那两个 `*_patch_proj` 与输出头是 **fp32** ✓ ⇒ 这里转回来 ✓，
                       与参考封装的 `.to(video_x.dtype)` 同义 ✓）；
                     * ⚠️ **双流日程的 carry**（``σ_a/σ_v`` ✓）由**采样器**负责 ✓ —— 本函数只管"一次" ✓，
                       见 :func:`audio_carry` ✓。**不做**就只是"少一个换算"✗（不会报错 ✓✗）。
                     """
        _, patch_h, patch_w = trunk.patch_size
        channels, frames, height, width = (int(value) for value in video_latent.shape)
        # ⚠️ **补边** ✓（不是报错 ✗）：参考 `_forward` 头一件事就是
        #    `pad_to_patch_size(video_x, patch_size)` ✓，末尾再 `[:, :, :orig_t, :orig_h, :orig_w]`
        #    裁回原点 ✓（`model.py` 606 / 781 行 ✓）—— 本仓曾在这里**直接 raise** ✗
        #    ⇒ 一样是"能跑、只是挡分辨率"✓✗：奇数 H/W 一个都进不来 ✓（而参考照跑 ✓）
        padded = pad_to_patch_size(video_latent, trunk.patch_size)
        lat_t, lat_h, lat_w = (int(value) for value in padded.shape[1:])
        if denoise_mask is not None:
            # ⚠️ 掩码要随目标**一起**补 ✓（replicate ✓ 与参考同口径 ✓）：不补的话
            #    `mask_rows_for` 会因 H/W 与补后尺寸不同而**当场报错** ✓（这一步不静默 ✓），
            #    但报出来的话补边就白做了 ✓✗ ⇒ 两件事必须成对 ✓
            if not isinstance(denoise_mask, (tuple, list)) or len(denoise_mask) != 2:
                raise ValueError(
                    f"denoise_mask 要给 ``(mask_video, mask_audio)`` 两路 ✗"
                    f"（收到 {type(denoise_mask).__name__} ✓）")
            mask_video = mask_rows_for(denoise_mask[0], 3, (frames, height, width),
                                       "denoise_mask[0]")
            if (lat_t, lat_h, lat_w) != (frames, height, width):
                mask_video = pad_to_patch_size(mask_video.unsqueeze(0), trunk.patch_size,
                                               "replicate")[0]
            denoise_mask = (mask_video, denoise_mask[1])
        video_rows = patchify_video(padded, trunk.patch_size)
        audio_rows = pack_audio(audio_latent)
        # ⚠️ **条件/参考块的行排在目标行之前** ✓（`packed_layout` 的段序就是如此 ✓ —— 无 refs 时
        #    就是「cond* 在前 ✓」）⇒ 这里只做**前插** ✓，**不排序、不猜顺序** ✗：
        #    顺序错了会被 `forward` 的**总行数校验**拦一部分 ✓，但**同数不同序**它拦不住 ✓✗
        #    ⇒ 顺序责任在**调用方** ✓（见 `TorchBackend.sample_dual` ✓）。
        if extra_video_rows is not None:
            video_rows = torch.cat([extra_video_rows, video_rows], dim=0)
        if extra_audio_rows is not None:
            audio_rows = torch.cat([extra_audio_rows, audio_rows], dim=0)
        out_video, out_audio = trunk(video_rows, audio_rows, text_states, sigma_v,
                                     lat_h, lat_w, shift_v, shift_a, visual_cond_t,
                                     audio_cond_t=audio_cond_t,
                                     # ⚠️ 段长**显式给** ✓：带 extra 行时从总行数**推不出来** ✗
                                     #    （会误判成"必须显式给"而报错 ✓ —— 这里就直接给对 ✓）
                                     # ⚠️ 布局**也得知道有 cond/ref 段** ✗：只多传行、不传结构 ⇒
                                     #    总行数校验会**当场红** ✓（2026-09-20 实测就是这样抓到的 ✓）
                                     # ⚠️ 这里给的是**补后**的格数 ✓（参考 613 行同样取补后值 ✓）——
                                     #    RoPE 与行数按补后的算 ✓，末尾再裁 ✓
                                     latent_t=lat_t, audio_t=int(audio_latent.shape[-1]),
                                     keyframes=keyframes, refs=refs,
                                     sample_sigmas=sample_sigmas,
                                     denoise_mask=denoise_mask,
                                     text_token_tags=text_token_tags)
        video_out = unpatchify_video(out_video, lat_t, lat_h // patch_h, lat_w // patch_w,
                                     channels, trunk.patch_size)
        # ⚠️ **裁回原点** ✓（与参考 781 行同义 ✓）—— 不裁的话调用方拿到的形状会变 ✓✗
        #    （`denoise_step` 的契约就是"与输入同形"✓），补出来的行到此为止 ✓
        #    ⚠️ 本仓**没有 batch 维** ✓ ⇒ 少一个冒号 ✓（参考是 `[:, :, :orig_t, :orig_h, :orig_w]` ✓）
        video_out = video_out[:, :frames, :height, :width].contiguous()
        return (video_out.to(video_latent.dtype),
                unpack_audio(out_audio, int(audio_latent.shape[1])).to(audio_latent.dtype))

    def audio_scale(shift_v: float = 12.0, shift_a: float = 3.0) -> float:
        """``shift_v / shift_a`` ✓（H3 = **4.0** ✓ 事实来源：参考的 `ModelSamplingAV.audio_scale` ✓）。

        ⚠️ 它是**采样器配置里的一个标志** ✓ —— **不是**逐步的 ``σ_a/σ_v`` ✗（那个见 :func:`audio_carry` ✓）。
        两者容易混 ✗：一个是常数 4.0 ✓、一个是逐帧变的比值 ✓ —— 混了**不会报错** ✗✗。
        """
        return float(shift_v) / float(shift_a)

    def sample_dual_stream(trunk: Any, video_latent: Any, audio_latent: Any, text_states: Any,
                           sigmas: Any, *, shift_v: float = 12.0, shift_a: float = 3.0,
                           callback: Any = None, extra_video_rows: Any = None,
                           extra_audio_rows: Any = None, keyframes: Any = None,
                           refs: Any = None, denoise_mask: Any = None,
                           pin: Any = None, pin_mask: Any = None,
                           text_token_tags: Any = None) -> dict[str, Any]:
        """**双流欧拉采样** ✓：视频走 `sigmas` ✓、音频走**换算后**的 σ ✓（各走各的日程 ✓）。

        更新式（按参考**取负**的 velocity 约定 ✓ —— 见 `CONST.calculate_denoised` 是 ``x − σ·v`` ✓）::

            x_{n+1} = x_n + (σ_{n+1} − σ_n) · v

        它等价于「先算 ``x0 = x − σ·v`` ✓ 再按新的 σ 重新加噪」✓ —— 两条路子是同一个式子 ✓。

        ⚠️⚠️ **音频的 σ 必须逐步换算** ✗（`schedules.time_shift_sigma` ✓）：不换就等于把
        「同一个 σ」当成「**两种不同的噪声水平**」✗，而且**不会报错** ✗✗ —— 这是双流最容易
        静默出错的一处 ✓，所以自检把「shift 相同时两条 σ 序列**逐位相同**」钉住 ✓。
        ⚠️ `sigmas` 末位必须是 **0** ✓（本仓 `schedules` 的约定 ✓：末尾即去噪到 x0 ✓）⇒ 否则**报错** ✓（不猜 ✗）。

        ⚠️ 这里**不做** carry ✗（把音频潜变量缩放带到视频日程上那种做法 ✓）—— 本函数让**每条流
        待在自己的 σ 上** ✓，与 `denoise_step` 的口径一致 ✓（它内部按各自的 σ 出 velocity ✓）。

        ``denoise_mask``（2026-09-25 补 ✓，口径**来自上游可读源码** ✓）：``(mask_video, mask_audio)`` ✓
        —— ⭐⭐ 超清**二采**就是靠它锁音频的 ✗✗：上游 ``_h3_build_denoise_mask`` 明写
        「video 流 mask=**1**（重采 ✓）、audio 流 mask=**0**（保持一采结果 ✓）」✓
        （⚠️ **不是**"不把音频送进主干" ✗ —— 音频照样参与前向 ✓，只是**这一步的更新量按掩码缩放** ✓）。
        ``mask=0`` ⇒ 该流**逐位不变** ✓（自检钉住 ✓）；``None`` ⇒ 与原来**逐位相同** ✓（默认路径不动 ✗）。

        ⚠️⚠️ 掩码有**两半个作用** ✓，本函数只负责**后一半** ✗：① **改逐行时间戳**（在主干里 ✓ ——
        掩掉的行按 ``1−m·σ`` 定档 ✓ 见 :func:`mask_row_timesteps` ✓）② **缩放增量**（下面这个循环 ✓）。
        两半缺一都**不报错** ✓✗：缺 ① 时被锁住的流仍按 ``1−σ`` 被当"含噪样本"处理 ✓ ⇒ 条件信号变软 ✓。
        本函数把这**同一份**掩码原样递给 `denoise_step` ✓ ⇒ 两半用的是同一张掩码 ✓（不会错配 ✓）。

        ⭐⭐ ``pin`` / ``pin_mask``（2026-09-26 补 ✓，口径**来自上游可读源码** ✓）：**掩码冻结的局部重绘** ✓
        —— 「续拍」与「重拍」共用这一个原语 ✓（`h3_edit` 那层只算窗口 ✓）。
        ``pin = (pin_video, pin_audio)`` 给**原始潜变量**（冻结区的**真值** ✓，与两条流同形状 ✓）；
        ``pin_mask = (mask_video, mask_audio)`` **1 = 重画** ✓ / **0 = 冻结** ✓（可 broadcast ✓）。

        ⚠️⚠️ 与 `denoise_mask` **不是一回事** ✗✗（差在最关键的一点上）：
        `denoise_mask` 缩放的是**增量** ✓ ⇒ ``mask=0`` 那片停在**第 0 步的含噪值**上 ✓；
        而 ``pin_mask=0`` 那片**收敛到原始潜变量** ✓ —— 上游 ``KSamplerX0Inpaint`` 的写法是
        ``x = x·m + noise_scaling(σ, noise, z0)·(1−m)`` ✓、**并且**把主干输出也
        ``out = out·m + z0·(1−m)`` ✓ —— 等价于「把冻结区那份 velocity 换成**它自己的噪声方向** ✓」
        （本函数就是这么实现的 ✓：``noise_dir = (x₀ − pin)/σ₀`` ✓，**不需要**随机数发生器 ✓
        因为 x₀ = pin + σ₀·noise 是构造出来的 ✓）。这样 ``σ→0`` 时冻结区**逐位等于 pin** ✓
        （自检钉住 ✓）—— 这正是重拍「窗外原样不动 ✓」与续拍「上一段尾巴逐位保留 ✓」要的语义 ✓。

        ⚠️ 两条掩码**不许同时给同一路** ✗（语义会叠在一起 ✗ ⇒ **当场报错** ✓ 不猜 ✗）。
        ⚠️ 要 pin 就必须 ``sigmas[0] > 0`` ✗（``σ₀=0`` 时反推不出噪声方向 ✓ ⇒ 报错 ✓ 不猜 ✗）。
        """
        from app.services.engine import schedules  # noqa: PLC0415
        steps = list(sigmas)
        if len(steps) < 2:
            raise ValueError(f"sigmas 至少要有 2 个（收到 {len(steps)} ✗）")
        if float(steps[-1]) != 0.0:
            raise ValueError(f"sigmas 末位必须是 0.0（收到 {steps[-1]} ✗ —— 本仓约定 ✓）")
        video_sigmas = [float(value) for value in steps]
        audio_sigmas = [schedules.time_shift_sigma(value, shift_v, shift_a)
                        for value in video_sigmas]
        state_video, state_audio = video_latent, audio_latent
        mask_video: Any = None
        mask_audio: Any = None
        if denoise_mask is not None:
            # ⚠️ 只认**二元组** ✗（不靠"能 unpack 就算对"✗ —— 张量恰好第一维是 2 时它能悄悄 unpack 成功 ✓✗）
            if not isinstance(denoise_mask, (tuple, list)) or len(denoise_mask) != 2:
                raise ValueError(
                    "denoise_mask 要给 ``(mask_video, mask_audio)`` 两路 ✗（口径见函数注释："
                    "超清二采 = video 1 / audio 0 ✓）—— ⚠️ 只给一路就没法表达「每条流各自锁不锁」✓✗")
            mask_video, mask_audio = denoise_mask
        # ⭐⭐ 冻结区（pin）：先反推**噪声方向**，再在循环里把那片 velocity 换成它 ✓（见 docstring ✓）
        pin_video: Any = None
        pin_audio: Any = None
        pin_mask_video: Any = None
        pin_mask_audio: Any = None
        noise_video: Any = None
        noise_audio: Any = None
        if (pin is None) != (pin_mask is None):
            raise ValueError(
                "`pin` 与 `pin_mask` 必须**一起给** ✗（只给 pin 不知道冻结哪一片 ✓、只给 pin_mask "
                "不知道那片要收敛到什么 ✓）")
        if pin is not None:
            if mask_video is not None or mask_audio is not None:
                raise ValueError(
                    "`denoise_mask` 与 `pin_mask` 不许同时用 ✗ —— 两者都是掩码，叠在一起语义不明 ✓"
                    "（要「窗外收敛到原始潜变量」就用 pin ✓；要「整条锁住不动」才用 denoise_mask ✓）")
            if not isinstance(pin, (tuple, list)) or len(pin) != 2:
                raise ValueError("`pin` 要给 ``(pin_video, pin_audio)`` 两路 ✗（只钉一路 ⇒ 那一路给 None ✓）")
            if not isinstance(pin_mask, (tuple, list)) or len(pin_mask) != 2:
                raise ValueError("`pin_mask` 要给 ``(mask_video, mask_audio)`` 两路 ✗（同上 ✓）")
            pin_video, pin_audio = pin
            pin_mask_video, pin_mask_audio = pin_mask
            for stream, state, pinned, pin_only_mask, sigma0 in (
                    ("视频", state_video, pin_video, pin_mask_video, video_sigmas[0]),
                    ("音频", state_audio, pin_audio, pin_mask_audio, audio_sigmas[0])):
                if pinned is None:
                    if pin_only_mask is not None:
                        raise ValueError(f"pin_mask 给了{stream}一路而 pin 没给 ✗（两路要成对 ✓）")
                    continue
                if pin_only_mask is None:
                    raise ValueError(f"pin 给了{stream}一路而 pin_mask 没给 ✗（两路要成对 ✓）")
                if not hasattr(pinned, "shape") or not hasattr(pin_only_mask, "shape"):
                    raise ValueError(
                        f"pin/pin_mask 的{stream}一路不是张量（{type(pinned).__name__} ✗）"
                        "—— 这两者只对**真后端**开放 ✓（干跑后端的占位张量没有 shape ✓ 语义 ✓）")
                if tuple(pinned.shape) != tuple(state.shape):
                    raise ValueError(
                        f"pin 的{stream}与**本条流的初始潜变量**形状不等 ✗（pin {tuple(pinned.shape)} "
                        f"vs 状态 {tuple(state.shape)} ✓）—— pin 是**同一条流**的原始潜变量 ✓ ⇒ "
                        "要钉头部就把 pin **补齐到整条长度**（计划层的事 ✓ 本函数不补齐 ✗ 也不猜 ✗）")
                if float(sigma0) <= 0.0:
                    raise ValueError(
                        f"要 pin{stream}就必须**首 σ > 0** ✗（收到 {sigma0} ✓）—— 冻结区的噪声方向靠 "
                        "``x₀ = pin + σ₀·noise`` 反推 ✓ ⇒ σ₀=0 时反推不出来 ✓（报错不猜 ✗）")
                try:
                    torch.broadcast_shapes(tuple(state.shape), tuple(pin_only_mask.shape))
                except RuntimeError as err:
                    raise ValueError(
                        f"pin_mask 的{stream}一路 broadcast 不到 {tuple(state.shape)} ✗（{err} ✓）"
                        "—— 掩码要能落到每一个槽位（如 ``(1,T,1,1)`` ✓）") from None
            if pin_video is not None:
                noise_video = (state_video - pin_video) / float(video_sigmas[0])
            if pin_audio is not None:
                noise_audio = (state_audio - pin_audio) / float(audio_sigmas[0])
        for index in range(len(video_sigmas) - 1):
            sigma_v, next_v = video_sigmas[index], video_sigmas[index + 1]
            sigma_a, next_a = audio_sigmas[index], audio_sigmas[index + 1]
            velocity_v, velocity_a = denoise_step(
                trunk, state_video, state_audio, text_states, sigma_v,
                shift_v=shift_v, shift_a=shift_a, extra_video_rows=extra_video_rows,
                extra_audio_rows=extra_audio_rows, keyframes=keyframes, refs=refs,
                # ⚠️ **整条**视频 σ 日程递下去 ✓（PDD 头库要靠它定位 span ✓）——
                #    只在"本条流"的语义上给 ✓（音频那条由 `time_shift_sigma` 现算 ✓）
                sample_sigmas=video_sigmas,
                # ⭐⭐ 掩码**两处都要给** ✓✗：这边只拿它定**逐行时间戳** ✓（参考的
                #    ``seg_t/video_rows_t`` ✓），增量的缩放在下面循环里 ✓ —— 少了这一格
                #    「掩掉的行」档位仍按 ``1−σ`` ✓ ⇒ 不报错、条件变软 ✓✗（2026-11-01 补 ✓）
                denoise_mask=(mask_video, mask_audio) if denoise_mask is not None else None,
                # ⚠️ 文本标签逐**步**都要给 ✓（每步的 adaLN 行号都靠它切 ✓）—— 漏了不报错 ✗，
                #    视觉占位 token 会退回 text 那组调制 ✓✗
                text_token_tags=text_token_tags)
            if pin_video is not None:
                # ⭐ 冻结区那份 velocity = **它自己的噪声方向** ✓（= 上游 `KSamplerX0Inpaint` 的
                #    ``out = out·m + z0·(1−m)`` 在「x₀ = x − σ·v」约定下的等价写法 ✓）
                velocity_v = velocity_v * pin_mask_video + noise_video * (1.0 - pin_mask_video)
            if pin_audio is not None:
                velocity_a = velocity_a * pin_mask_audio + noise_audio * (1.0 - pin_mask_audio)
            delta_v = (next_v - sigma_v) * velocity_v
            delta_a = (next_a - sigma_a) * velocity_a
            if mask_video is not None:
                delta_v = delta_v * mask_video          # ⭐ mask=0 ⇒ 该流**逐位不变** ✓✗
            if mask_audio is not None:
                delta_a = delta_a * mask_audio
            state_video = state_video + delta_v
            state_audio = state_audio + delta_a
            if pin_video is not None:
                # ⚠️ 逐位收口 ✓：浮点上 `a + b − b` 不保证等于 `a` ✗ ⇒ 冻结区显式写回 ✓
                state_video = (state_video * pin_mask_video
                               + (pin_video + next_v * noise_video) * (1.0 - pin_mask_video))
            if pin_audio is not None:
                state_audio = (state_audio * pin_mask_audio
                               + (pin_audio + next_a * noise_audio) * (1.0 - pin_mask_audio))
            if callback is not None:
                callback(index, sigma_v, sigma_a)
        return {"video": state_video, "audio": state_audio, "steps": len(video_sigmas) - 1,
                "videoSigmas": video_sigmas, "audioSigmas": audio_sigmas,
                "masked": denoise_mask is not None, "pinned": pin is not None}

    class Int8Linear(nn.Module):  # noqa: D101 —— 公开名见 __all__ ✓
        """**int8 权重常驻**的线性层 ✓ —— **前向时**才反量化 ✓（2026-10-29 加 ✓）。

        ## 为什么不能「先整份反量化再装」✗✗

        主 DiT 的 int8 权重是 **17.94 GiB** ✓；整份反量化成 bf16 = **35.9 GiB** ✗ ⇒
        A5000 可用显存 22.49 GiB 连**装**都装不进去 ✗（还没算激活 ✓）——旧路径必然 OOM ✓，
        这不是「跑得慢」而是「根本起不来」✓✗（`loader.residency_plan` 早就点过这一条 ✓）。

        ## 做法 ✓

        * ``weight`` 参数**就是 int8** ✓（17.94 GiB 原样 ✓，PyTorch 的 ``.to(dtype)`` **不会**
          把非浮点张量降精度 ✓ ⇒ 后面把整个模块 ``.to(bf16)`` 也不会动它 ✓）；
        * ``weight_scale`` 是文件里的 per-row 尺度 ✓（几 KB/层 ✓，**必须 fp32** ✓ ——
          降成 bf16 会丢掉 0.4% 的尺度精度 ✓✗，比 int8 本身的量化误差还大 ✗）；
        * 每次 ``forward`` **只反量化本层** ✓ ⇒ 瞬时 ≈ 单层 bf16 权重（最大 294 MiB ✓）。

        ## convrot 的旋转搬到**激活侧** ✓

        推导见 :func:`quant.dequantize_weight_rotated` ✓：权重侧反旋转要 **4.9 TMAC/步** ✗，
        激活侧只要 **0.32 GMAC/步** ✓（差 15 倍 ✓）。两者**必须成对** ✓ —— 本类把这一对
        锁在同一个 ``forward`` 里 ✓，从外面看不出有两个可错配的开关 ✓。
        """

        def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                     scale_shape: Any = (1,), convrot: int | None = None,
                     dtype: Any = None, device: Any = None) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.convrot = None if convrot is None else int(convrot)
            self.weight = nn.Parameter(
                torch.empty((self.out_features, self.in_features), dtype=torch.int8,
                            device=device), requires_grad=False)
            #: ⚠️ 形状**照文件**建 ✓（`[N]` 与 `[N,1]` 两种写法都有 ✓ —— 自己写死会被装载闸口按
            #: 「形状不符」拦下 ✓，而模板是 `loader` 那份「不猜」的口径 ✓）。
            self.register_buffer(
                "weight_scale",
                torch.ones(tuple(int(d) for d in scale_shape), dtype=torch.float32, device=device))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features,
                                                     dtype=dtype or torch.float32, device=device),
                                         requires_grad=False)
            else:
                self.register_parameter("bias", None)

        def extra_repr(self) -> str:
            rot = "" if self.convrot is None else f", convrot={self.convrot}"
            return (f"in_features={self.in_features}, out_features={self.out_features}, "
                    f"bias={self.bias is not None}, int8{rot}")

        def forward(self, x: Any) -> Any:
            quant = _quant_mod()
            weight = self.weight
            scale = self.weight_scale
            # ⚠️ 权重**驻 CPU** 那条路（`place_module(..., park_int8_on=...)` ✓）：
            #    本层**现搬** ✓ —— 上一层算完它的副本就释放了 ✓，所以峰值是「单层 int8 + 单层 bf16」✓，
            #    不是整份 17.94 GiB ✗（这是 A5000 上唯一跑得起来的排法 ✓）。
            #    ⚠️ 只在**设备不同**时才搬 ✗：常驻显卡时这里一次拷贝都不发生 ✓（老路一字未变 ✓）。
            if weight.device != x.device:
                weight = weight.to(x.device, non_blocking=weight.is_pinned())
            if scale.device != x.device:
                scale = scale.to(x.device, non_blocking=scale.is_pinned())
            weight = quant.dequantize_weight_rotated(weight, scale, dtype=x.dtype)
            if self.convrot is not None:
                x = quant.rotate_activations(x, self.convrot)
            bias = (None if self.bias is None
                    else self.bias.to(device=x.device, dtype=x.dtype))
            return nn.functional.linear(x, weight, bias)

    return {
        "torch": torch, "nn": nn, "RMSNorm": RMSNorm, "TimeEmbedder": TimeEmbedder,
        "SwiGLU": SwiGLU, "Attention": Attention, "AdalnProj": AdalnProj,
        "RefinerBlock": RefinerBlock, "TokenRefiner": TokenRefiner, "DiTBlock": DiTBlock,
        "FinalLayer": FinalLayer, "FRAME_PER_TOKEN": FRAME_PER_TOKEN,
        "FRAME_RESCALE": FRAME_RESCALE, "frame_grid_coords": frame_grid_coords,
        "video_t_grid": video_t_grid, "packed_rows": packed_rows, "position_ids": position_ids,
        "packed_layout": packed_layout, "assemble_blocks": assemble_blocks,
        "cond_noise_augment": cond_noise_augment, "pdd_head": pdd_head,
        "audio_grid": audio_grid, "video_grid": video_grid,
        "video_t_spans": video_t_spans, "ref_time_span": ref_time_span,
        "rope_angles": rope_angles, "mod_row": mod_row, "H3_DEFAULTS": H3_DEFAULTS,
        "VISUAL_COND_TIMESTEP": VISUAL_COND_TIMESTEP, "AUDIO_COND_TIMESTEP": AUDIO_COND_TIMESTEP,
        "H3FormTrunk": H3FormTrunk, "audio_carry": audio_carry,
        "mod_segments_for": mod_segments_for, "t_vals_for": t_vals_for,
        "denoise_step": denoise_step, "audio_scale": audio_scale,
        "sample_dual_stream": sample_dual_stream,
        "mask_row_values": mask_row_values, "mask_rows_for": mask_rows_for,
        "mask_row_timesteps": mask_row_timesteps, "rows_unique_index": rows_unique_index,
        "pack_audio": pack_audio,
        "curve_t_emb": curve_t_emb,
        "patchify_video": patchify_video, "unpack_audio": unpack_audio,
        "unpatchify_video": unpatchify_video, "pad_to_patch_size": pad_to_patch_size,
        "Int8Linear": Int8Linear,
    }


def _quant_mod() -> Any:
    """延迟拿到 :mod:`quant` ✓（**不**在模块顶部 import ✗ —— 本模块要能「没有 torch 也导入」✓）。"""
    global _QUANT_CACHE
    if _QUANT_CACHE is None:
        from app.services.engine import quant as module  # noqa: PLC0415
        _QUANT_CACHE = module
    return _QUANT_CACHE


def _parts() -> dict[str, Any]:
    """**缓存**建好的积木 ✓。

    ⚠️ 为什么要缓存 ✗：老写法每次 `h3_form.H3FormTrunk` 触发的 `__getattr__` 都**重新定义一遍
    全部类** ✓✗ ⇒ 同一次装载里拿 3 次就是 **3 个不同的类** ✓✗（`isinstance` 跨调用失效 ✓，
    `upgrade_linear_to_int8` 换上去的层与调用方手里的类不是同一个 ✓）。缓存后 identity 稳定 ✓。
    """
    global _PARTS_CACHE
    if _PARTS_CACHE is None:
        _PARTS_CACHE = _build_torch_parts()
    return _PARTS_CACHE


def upgrade_linear_to_int8(module: Any, int8_scales: dict[str, Any], *,
                           convrot: Any = None) -> list[str]:
    """把 ``int8_scales`` 点名的 ``nn.Linear`` **原地**换成 :class:`Int8Linear` ✓（返回换掉的名单 ✓）。

    ``int8_scales``：``{"blocks.0.mlp.fc1": (28672,), ...}`` ✓ —— 值是该权重**配套 scale 在文件里的
    形状** ✓。⚠️ 形状**不能**自己写死 ✗：``[N]`` / ``[N,1]`` / ``[1]`` 三种写法在文件里都真实存在 ✓，
    装错形状会被装载闸口按「形状不符」整份中止 ✓（那是好事 ✓，好过静默广播 ✓）。

    ⚠️ **调用时机只有一处合法** ✗：必须在「模块已经 ``.to(dtype)`` 成计算精度」之后、
    「开始装载」之前 ✓ —— 反过来会出两个错：① 先 upgrade 再 ``.to(bf16)`` ⇒ 把 ``weight_scale``
    从 fp32 降成 bf16 ✓✗（尺度精度比 int8 的量化误差还大 ✗）；② 装载后再 upgrade ⇒ 换上去的
    层权重是全 0 ✓✗（`torch.empty` 没填 ✓，不报错、出片全黑 ✓）。

    ``convrot`` **逐层解析** ✓：整文件同组给 ``256`` ✓，逐层不同给 ``{层名: group}`` ✓
    （实测 H3 全量版同文件里 64 与 256 并存 ✓）—— 一律经 :func:`quant.group_for` 取本层那一个 ✓，
    查不到的层 ⇒ 不旋转（不拿别层的 group 顶上 ✗）。
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    cls = _parts()["Int8Linear"]
    swapped: list[str] = []
    for name, scale_shape in int8_scales.items():
        parent_path, _, leaf = str(name).rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        old = getattr(parent, leaf, None)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"`{name}` 不是 nn.Linear（收到 {type(old).__name__} ✗）"
                            f"⇒ 不能换成 int8 常驻层 ✗")
        new = cls(old.in_features, old.out_features, bias=old.bias is not None,
                  scale_shape=tuple(int(d) for d in scale_shape),
                  convrot=_quant_mod().group_for(str(name), convrot),
                  dtype=old.weight.dtype, device=old.weight.device)
        new.train(old.training)
        setattr(parent, leaf, new)
        swapped.append(str(name))
    return swapped


def place_module(module: Any, device: Any, *, park_int8_on: Any = None) -> dict[str, Any]:
    """把模块状态搬到 ``device`` ✓ —— 但 **int8 常驻权重可以留在别处** ✓（``park_int8_on`` ✓）。

    ## 为什么不能直接 ``module.to(device)`` ✗

    整份 int8 是 **17.94 GiB** ✓，22.49 GiB 的 A5000 上「全塞进显存」之后只剩 **2.87 GiB**
    给激活 ✓✗ —— 而 720×1280×5s 那种长度的激活要十几 GiB ✓✗（2026-10-29 实测扫过三档 ✓：
    128 行 1.16 GiB ✓ / 5120 行 1.37 GiB ✓ / 20736 行**跑 30 分钟没完** ✗）。
    ⇒ 唯一走得通的排法是「int8 **钉在 CPU** ✓ + 前向逐层搬 ✓」—— 峰值降到 **单层** ✓
    （最大那层 147 MiB int8 + 294 MiB bf16 ✓），代价是每步多一趟 PCIe ✓。

    ## 它**不是** ``.to()`` 的替代品 ✓✗

    * **不改 dtype** ✗（dtype 是 :func:`_prepare_h3_module` 的事 ✓，这里只管"在哪个设备"✓）；
    * **只管参数与缓冲区** ✓（那就是模块的全部状态 ✓ —— 没有第三种东西 ✓）；
    * ``park_int8_on=None`` 时等价于「只搬设备」✓（本仓默认路径仍用 ``module.to()`` ✓，
      这里给"驻 CPU"那条路用 ✓）。

    返回的字典是**报告**（不猜 ✗）：``parked`` 层数 ✓、``parkedBytes`` 字节 ✓（17.94 GiB 那件事
    要能对上号 ✓）、``pinned`` 是否真的锁页 ✓（异步 H2D 的前提 ✓；锁不上**不算错** ✓，只是慢 ✓）。
    """
    import torch  # noqa: PLC0415

    target = torch.device(device) if not isinstance(device, torch.device) else device
    park = None if park_int8_on is None else (
        torch.device(park_int8_on) if not isinstance(park_int8_on, torch.device)
        else park_int8_on)
    moved = parked = 0
    parked_bytes = 0
    pinned = True
    for tensor in [*module.parameters(), *module.buffers()]:
        if tensor is None:  # pragma: no cover - 参数表里不会有 None ✓（`register_parameter(None)` 不会进表 ✓）
            continue
        keep = park is not None and tensor.dtype == torch.int8
        want = park if keep else target
        current = tensor.data if tensor.data.device != want else None
        if current is not None:
            tensor.data = current.to(want, non_blocking=True)
        if keep:
            parked += 1
            parked_bytes += tensor.numel() * tensor.element_size()
            # ⚠️ 锁页**只对 CPU 有意义** ✓：异步 H2D 要它才真异步 ✓（不锁也行 ✓，只是退化成同步 ✓）
            if want.type == "cpu" and not tensor.data.is_pinned():
                try:
                    tensor.data = tensor.data.pin_memory()
                except RuntimeError:       # pragma: no cover - 无 CUDA / 锁页配额满 ✓
                    pinned = False
        elif current is not None:
            moved += 1
    return {"device": str(target), "parkedOn": None if park is None else str(park),
            "applied": park is not None, "moved": moved, "parked": parked,
            "parkedBytes": parked_bytes, "pinned": pinned}


def __getattr__(name: str) -> Any:
    """让 `from ... import RMSNorm` 之类的写法**只在真要 torch 时才报错** ✓。

    没有 torch 的环境里 `import h3_form` **不该炸** ✓（本仓对 torch 一律"可选"✓）——
    但**也不能假装有** ✗：用到具体类时才抛 `ImportError` ✓。
    ⚠️ 顺序要紧 ✓：**先查 `__all__`** ✓ 再建积木 ✓ —— 初版反过来 ✗ ⇒ 问一个**不存在的名字**
    也会去 `import torch` ✗（没装 torch 的环境里就变成 `ImportError` 而不是 `AttributeError` ✗）。
    """
    if name.startswith("_") or name not in __all__:
        raise AttributeError(f"h3_form 没有 {name} ✓")
    return _parts()[name]
