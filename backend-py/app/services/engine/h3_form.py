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
    "denoise_step", "mask_row_values", "mod_segments_for", "pack_audio", "patchify_video",
    "sample_dual_stream", "t_vals_for", "unpack_audio", "unpatchify_video",
    "audio_grid", "packed_layout", "ref_time_span", "video_grid", "video_t_spans",
    "assemble_blocks", "pdd_head",
    "H3_VIDEO_OUT_KEY", "video_patch_dim", "head_banks_from_shape",
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
            cos = torch.cos(angles[:, :half]).reshape(angles.shape[0], 1, half)
            sin = torch.sin(angles[:, :half]).reshape(angles.shape[0], 1, half)
            first, second = pair[..., 0], pair[..., 1]
            out = torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1)
            return torch.cat([out.reshape(*rotated.shape[:-1], rot_dim), tail], dim=-1)

        def forward(self, x: Any, rope_angles: Any = None, rot_dim: int = 0) -> Any:
            count = x.shape[0]
            packed = self.qkv_proj(x)
            inner = self.heads * self.head_dim
            q, k, v = packed.split(inner, dim=-1)
            q = q.reshape(count, self.heads, self.head_dim)
            k = k.reshape(count, self.heads, self.head_dim)
            v = v.reshape(count, self.heads, self.head_dim)
            if rope_angles is not None:
                q = self.apply_rope(q, rope_angles, rot_dim)
                k = self.apply_rope(k, rope_angles, rot_dim)
            q, k = self.q_norm(q), self.k_norm(k)
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
                     apply_silu: bool = True) -> None:
            super().__init__()
            self.expand = int(expand)
            self.modalities = int(modalities)
            self.apply_silu = bool(apply_silu)
            self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

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
                     eps: float = 1e-5, qk_eps: float = 1e-5, modalities: int = 3) -> None:
            super().__init__()
            self.norm1 = RMSNorm(hidden, eps)
            self.norm2 = RMSNorm(hidden, eps)
            self.attn = Attention(hidden, heads, head_dim, qk_eps)
            self.mlp = SwiGLU(hidden, ffn)
            self.adaln_proj = AdalnProj(t_dim, hidden, 6, modalities)

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
            eps: float = 1e-5, head_banks: int = 1) -> None:
            super().__init__()
            if int(head_banks) < 1:
                raise ValueError(f"head_banks 必须 ≥ 1（收到 {head_banks} ✗）")
            self.head_banks = int(head_banks)
            self.norm = RMSNorm(hidden, eps)
            self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1)
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

    def assemble_blocks(layout: Any, video_sources: Any, audio_sources: Any,
                        patch_size: tuple[int, int, int] = (1, 2, 2)) -> tuple[Any, Any]:
        """按**布局的块序**把条件/参考潜变量拼成行 ✓ ⇒ ``(视频行, 音频行)`` ✓（**不含目标** ✓）。

        ``video_sources`` / ``audio_sources``：``{("keyframe", 0): 潜变量, ("ref", 0): …}`` ✓ ——
        **只提供块**，顺序**不问**调用方 ✓（顺序由 `packed_layout` 的块清单说了算 ✓）。
        ⚠️ 这就是"行序判据"的落点 ✓：以前布局一处、拼装另一处 ⇒ 同数不同序时**总行数校验拦不住** ✗✗
        （只会"画面不对" ✓）⇒ 现在**布局说了算、这里照抄** ✓。
        ⚠️ 缺块 / 块行数对不上 ⇒ **报错** ✗（不静默跳过 ✓ —— 少一截或错位都是最难查的那种 ✓）。
        """
        def gather(blocks: Any, sources: Any, convert: Any, label: str) -> Any:
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
                rows.append(converted)
            return torch.cat(rows, dim=0) if rows else None

        return (gather(layout["video_blocks"], video_sources,
                       lambda latent: patchify_video(latent, patch_size), "视频"),
                gather(layout["audio_blocks"], audio_sources, pack_audio, "音频"))

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
    #: 条件行的 timestep（**事实** ✓ 参考里是写死的常量 ✓；text 归到 visual 那一档 ⚠️ **未逐条核过** ✗）
    VISUAL_COND_TIMESTEP = 0.999
    AUDIO_COND_TIMESTEP = 1.0

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
                   audio_cond_t: float = AUDIO_COND_TIMESTEP) -> tuple[Any, dict[str, int]]:
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
        values = sorted(set(wanted.values()))          # ⚠️ **升序** ✓（= 参考的 sorted ✓ 不是首次出现 ✓）
        index = {kind: values.index(value) for kind, value in wanted.items()}
        return torch.tensor(values, dtype=torch.float32), index

    def mod_segments_for(segments: Any, index: dict[str, int],
                         modalities: int = 3) -> Any:
        """把段表变成 adaLN 的 ``mod_segments`` ✓（``row = 唯一序号 × modalities + 标签`` ✓）。

        ⚠️⚠️ **模态标签**（事实 ✓ 2026-09-20 逐字核自参考前向里的 ``seg_tag`` ✓）：
        **video / cond / ref_img = 0** ✓、**text = 1** ✓、**audio / cond_audio / ref_audio = 2** ✓。

        ⚠️ 本仓初版写的是「text=0 ✓ / audio=1 ✓ / video=2」✗ —— **三档全错** ✓✗：标签只是 adaLN 的
        **行内偏移** ✓ ⇒ 错法**完全不报错** ✓，只是在真权重上把每个模态接到**别的模态**那一组参数
        （画面"就是不对" ✓✗）。⇒ 现在按"音频在视频之前"那条事实的同一份来源钉住 ✓。
        """
        tags = {"video": 0, "cond": 0, "ref_img": 0, "text": 1,
                "audio": 2, "cond_audio": 2, "ref_audio": 2}
        return tuple((start, stop, mod_row(index[kind], tags[kind], modalities))
                     for start, stop, kind in segments)

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
            modalities: int = 3, head_banks: int = 1) -> None:
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
            self.time_embedder = TimeEmbedder(time_input_dim, time_hidden, time_dim)
            # ⚠️ `inv_freq` 是**权重里的缓冲区** ✓（H3 默认 16 ✓）—— 本项目**不自己算**它 ✗
            self.rope = nn.Module()
            self.rope.register_buffer("inv_freq", torch.ones(inv_freq_len))
            self.rot_dim = 6 * int(inv_freq_len)         # 3 轴 × F × 复制 2 份 ✓
            self.token_refiner = TokenRefiner(refiner_layers, hidden, heads, head_dim, ffn,
                                              norm_eps, qk_norm_eps, final_norm_eps)
            self.blocks = nn.ModuleList([
                DiTBlock(hidden, heads, head_dim, ffn, time_dim, norm_eps, qk_norm_eps, modalities)
                for _ in range(layers)])
            self.final_layer = FinalLayer(hidden, time_dim, video_patch_dim,
                                          self.audio_latents_dim, final_norm_eps,
                                          head_banks=self.head_banks)

        def forward(self, video_rows: Any, audio_rows: Any, text_states: Any, sigma_v: float,
                    latent_h: int, latent_w: int, shift_v: float = 12.0, shift_a: float = 3.0,
                    visual_cond_t: float = VISUAL_COND_TIMESTEP, *,
                    audio_cond_t: float = AUDIO_COND_TIMESTEP,
                    latent_t: int | None = None, audio_t: int | None = None,
                    keyframes: Any = None, refs: Any = None,
                    sample_sigmas: Any = None) -> tuple[Any, Any]:
            """行级前向 ✓ ⇒ ``(视频行, 音频行)`` ✓（**已按参考的约定取负** ✓ velocity ✓）。

            ``video_rows``：``[行, latents_dim×pT·pH·pW]`` ✓（**含条件/参考块的行** ✓ 若有 ✓）、
            ``audio_rows``：``[行, audio_dim]`` ✓、``text_states``：``[L, text_dim]`` ✓
            （**都是 fp32 ✓、都没有 batch 维 ✓**）。

            ⚠️ **无** keyframes/refs 时段长从行数推出来 ✓（``latent_t = 视频行 / 每帧行数`` ✓ 除不尽报错 ✓）；
            ⚠️ **有**条件/参考块时**推不出来** ✗（那些块的行数不是每帧行数的整数倍 ✓）⇒ ``latent_t`` /
            ``audio_t`` **必须显式给** ✓（不猜 ✗），然后按 `packed_layout` 算出的**总行数**反向校验 ✓
            （对不上就报错 ✓ —— 这正是"行数与布局脱节"那条静默错的守卫 ✓）。
            """
            text_len = int(text_states.shape[0])
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
            t_vals, index = t_vals_for(sigma_v, shift_v, shift_a, visual_cond_t, audio_cond_t)
            segments = layout["segments"]
            mod_segments = mod_segments_for(segments, index, self.modalities)

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
            t_emb = self.time_embedder(t_vals).to(h.dtype)
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
            # ⚠️ σ 日程与两条 shift 都递给输出层 ✓：`head_banks > 1` 时要靠它们定位 PDD 的 span ✓
            video, audio = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v,
                                            sample_sigmas, (shift_v, shift_a))
            return -video, -audio          # ⚠️ **取负** ✓（参考的 velocity 约定 ✓，不是笔误 ✗）

    # ───────── 外层封装：潜帧 ⇄ 行（2026-09-20 第 ④ 批 ✓ 事实来源见模块 docstring ✓）─────────
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

    def denoise_step(trunk: Any, video_latent: Any, audio_latent: Any, text_states: Any,
                     sigma_v: float, *, shift_v: float = 12.0, shift_a: float = 3.0,
                     visual_cond_t: float = VISUAL_COND_TIMESTEP,
                     audio_cond_t: float = AUDIO_COND_TIMESTEP,
                     extra_video_rows: Any = None, extra_audio_rows: Any = None,
                     keyframes: Any = None, refs: Any = None,
                     sample_sigmas: Any = None) -> tuple[Any, Any]:
        """**一次去噪**：潜帧进 ⇒ **两条 velocity 出** ✓（形状与输入**同形** ✓）。

        ⚠️ 这个函数是"**把整条链串起来**"的那一处 ✓ —— 它一次用掉了
        `patchify_video` ✓ / `pack_audio` ✓ / `H3FormTrunk` ✓ / `unpatchify_video` ✓ / `unpack_audio` ✓：
        潜帧 ⇄ 行、坐标与 RoPE 在主干内部 ✓、最后**还原回潜帧形状** ✓。

        * ``video_latent``：``[C, T, H, W]`` ✓（``T`` 要能被 ``pT`` 整除 ✓、``H/W`` 能被 2 整除 ✓）；
        * ``audio_latent``：``[C, ch, T]`` ✓（``ch`` 就是声道数 ✓ —— 只有它进 `unpack_audio` ✓）；
        * ``text_states``：``[L, text_dim]`` ✓；
        * ⚠️ 输出 dtype **跟随输入** ✓（主干内部那两个 `*_patch_proj` 与输出头是 **fp32** ✓ ⇒ 这里转回来 ✓，
          与参考封装的 `.to(video_x.dtype)` 同义 ✓）；
        * ⚠️ **双流日程的 carry**（``σ_a/σ_v`` ✓）由**采样器**负责 ✓ —— 本函数只管"一次" ✓，
          见 :func:`audio_carry` ✓。**不做**就只是"少一个换算"✗（不会报错 ✓✗）。
        """
        patch_t, patch_h, patch_w = trunk.patch_size
        channels, frames, height, width = (int(value) for value in video_latent.shape)
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError(
                f"潜帧 {tuple(video_latent.shape)} 不能被 patch {trunk.patch_size} 整除 ✗"
                f"（不猜、也不补零 ✗）")
        video_rows = patchify_video(video_latent, trunk.patch_size)
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
                                     height, width, shift_v, shift_a, visual_cond_t,
                                     audio_cond_t=audio_cond_t,
                                     # ⚠️ 段长**显式给** ✓：带 extra 行时从总行数**推不出来** ✗
                                     #    （会误判成"必须显式给"而报错 ✓ —— 这里就直接给对 ✓）
                                     # ⚠️ 布局**也得知道有 cond/ref 段** ✗：只多传行、不传结构 ⇒
                                     #    总行数校验会**当场红** ✓（2026-09-20 实测就是这样抓到的 ✓）
                                     latent_t=frames, audio_t=int(audio_latent.shape[-1]),
                                     keyframes=keyframes, refs=refs,
                                     sample_sigmas=sample_sigmas)
        return (unpatchify_video(out_video, frames, height // patch_h, width // patch_w,
                                 channels, trunk.patch_size).to(video_latent.dtype),
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
                           refs: Any = None, denoise_mask: Any = None) -> dict[str, Any]:
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
        for index in range(len(video_sigmas) - 1):
            sigma_v, next_v = video_sigmas[index], video_sigmas[index + 1]
            sigma_a, next_a = audio_sigmas[index], audio_sigmas[index + 1]
            velocity_v, velocity_a = denoise_step(
                trunk, state_video, state_audio, text_states, sigma_v,
                shift_v=shift_v, shift_a=shift_a, extra_video_rows=extra_video_rows,
                extra_audio_rows=extra_audio_rows, keyframes=keyframes, refs=refs,
                # ⚠️ **整条**视频 σ 日程递下去 ✓（PDD 头库要靠它定位 span ✓）——
                #    只在"本条流"的语义上给 ✓（音频那条由 `time_shift_sigma` 现算 ✓）
                sample_sigmas=video_sigmas)
            delta_v = (next_v - sigma_v) * velocity_v
            delta_a = (next_a - sigma_a) * velocity_a
            if mask_video is not None:
                delta_v = delta_v * mask_video          # ⭐ mask=0 ⇒ 该流**逐位不变** ✓✗
            if mask_audio is not None:
                delta_a = delta_a * mask_audio
            state_video = state_video + delta_v
            state_audio = state_audio + delta_a
            if callback is not None:
                callback(index, sigma_v, sigma_a)
        return {"video": state_video, "audio": state_audio, "steps": len(video_sigmas) - 1,
                "videoSigmas": video_sigmas, "audioSigmas": audio_sigmas,
                "masked": denoise_mask is not None}

    return {
        "torch": torch, "nn": nn, "RMSNorm": RMSNorm, "TimeEmbedder": TimeEmbedder,
        "SwiGLU": SwiGLU, "Attention": Attention, "AdalnProj": AdalnProj,
        "RefinerBlock": RefinerBlock, "TokenRefiner": TokenRefiner, "DiTBlock": DiTBlock,
        "FinalLayer": FinalLayer, "FRAME_PER_TOKEN": FRAME_PER_TOKEN,
        "FRAME_RESCALE": FRAME_RESCALE, "frame_grid_coords": frame_grid_coords,
        "video_t_grid": video_t_grid, "packed_rows": packed_rows, "position_ids": position_ids,
        "packed_layout": packed_layout, "assemble_blocks": assemble_blocks,
        "pdd_head": pdd_head,
        "audio_grid": audio_grid, "video_grid": video_grid,
        "video_t_spans": video_t_spans, "ref_time_span": ref_time_span,
        "rope_angles": rope_angles, "mod_row": mod_row, "H3_DEFAULTS": H3_DEFAULTS,
        "VISUAL_COND_TIMESTEP": VISUAL_COND_TIMESTEP, "AUDIO_COND_TIMESTEP": AUDIO_COND_TIMESTEP,
        "H3FormTrunk": H3FormTrunk, "audio_carry": audio_carry,
        "mod_segments_for": mod_segments_for, "t_vals_for": t_vals_for,
        "denoise_step": denoise_step, "audio_scale": audio_scale,
        "sample_dual_stream": sample_dual_stream,
        "mask_row_values": mask_row_values, "pack_audio": pack_audio,
        "patchify_video": patchify_video, "unpack_audio": unpack_audio,
        "unpatchify_video": unpatchify_video,
    }


def __getattr__(name: str) -> Any:
    """让 `from ... import RMSNorm` 之类的写法**只在真要 torch 时才报错** ✓。

    没有 torch 的环境里 `import h3_form` **不该炸** ✓（本仓对 torch 一律"可选"✓）——
    但**也不能假装有** ✗：用到具体类时才抛 `ImportError` ✓。
    ⚠️ 顺序要紧 ✓：**先查 `__all__`** ✓ 再建积木 ✓ —— 初版反过来 ✗ ⇒ 问一个**不存在的名字**
    也会去 `import torch` ✗（没装 torch 的环境里就变成 `ImportError` 而不是 `AttributeError` ✗）。
    """
    if name.startswith("_") or name not in __all__:
        raise AttributeError(f"h3_form 没有 {name} ✓")
    return _build_torch_parts()[name]
