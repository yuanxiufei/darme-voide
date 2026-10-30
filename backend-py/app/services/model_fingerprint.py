"""真实权重指纹 —— **读文件头判架构**：不看文件名、不加载权重、不需要 torch。

⭐⭐ 为什么必须这么做（本模块存在的唯一理由）：

1. **文件名是营销字段**。`wan2.2_i2v_14B_fp8.safetensors` 可以叫 `model_final_v3.safetensors`，
   也可以把名字贴到完全另一份权重上。仓里早先的「视频模型识别」靠 ``local_model_scan`` 的
   文件名规则表（`wan-video-gguf` 之类，confidence=high/medium）—— 那是**启发式**，
   命不中就漏、改名就瞎。真正硬的判据只有权重本身：**key 名 + 张量形状**。
2. **ComfyUI 自己就是这么判的**（真源：``reference/comfy-org/ComfyUI/comfy/model_detection.py``
   的 ``detect_unet_config``），本模块把那套判据**按字面量移植**，每条都能追到上游。
3. **代价是读几十 KB，不是读几 GB**。``safetensors`` 头是「8 字节小端长度 + JSON」，
   ``gguf`` 头是「magic + 版本 + kv 表」—— 只在前若干 MB。实测本机 52 个权重文件
   **0.12 秒**全判完（平均 2ms/个），比按文件名猜还快 ✓。

⚠️ 五条边界（写在这里，免得以后被当成 bug 报上来）：
* **只判架构与变体，不判「好不好用」**：能判出这是 Wan 的 i2v，判不出它有没有训崩；
* **Wan 2.1 / 2.2 不区分**：两者 key 结构同构（上游也只给一个 ``wan2.1`` 名），
  能判的是 **t2v / i2v**（由 ``patch_embedding`` 入通道数 16 vs 36 决定，实测核实 ✓）；
* **GGUF 读 kv 表 + tensor info，不读 tokenizer 数组**：巨量的是词表（十几万条字符串），
  kv 里的**数组值**一律跳过、tensor info 逐条读（一个 DiT 一两千条，实测仍毫秒级 ✓）。
  ⚠️ 必须读 tensor info：``general.architecture`` 是**量化工具写的标签、可能写错**
  （真机实例：``Krea2/krea2_turbo-Q4_K_M.gguf`` 的标签写的是 ``qwen_image``，见下）；
* **同 key 集的两个变体判不出**：实测 ``minimax_h3_fl2va`` 与 ``ref2va`` 的张量 key **完全一致**
  （1035 个，差集为空）⇒ 子类型只能给 ``unknown``，**不许拿文件名补** ✗；
* **未命中一律 ``unknown``**，绝不按文件名兜底成「大概是 wan」✗ —— 猜出来的结论比没结论更坏。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ARCHITECTURES",
    "FingerprintError",
    "SAFETENSORS_MAX_HEADER",
    "Architecture",
    "base_name",
    "detect_architecture",
    "fingerprint_file",
    "gguf_capabilities",
    "integrity",
    "is_video",
    "name_audit",
    "parameters",
    "quantization",
    "read_gguf_header",
    "read_gguf_tensors",
    "read_safetensors_header",
    "shape_digest",
    "summarize",
    "video_architectures",
    "video_subtype",
]

#: safetensors 头理论上限。真文件一般 1~10 MB；超过说明文件坏了或是别的格式 ✓
SAFETENSORS_MAX_HEADER = 96 * 1024 * 1024
GGUF_MAGIC = b"GGUF"

#: GGUF kv 值类型 → 定长字节数（``8``=string、``9``=array 特殊处理 ✓）
_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
#: 一个 kv 里数组超过这个长度就只记个数（词表有十几万条，解码它们毫无意义 ✓）
_GGUF_ARRAY_KEEP = 64
#: tensor info 条数上限（防损坏文件让我们读几十万条名子）
GGUF_MAX_TENSORS = 200_000

#: ``general.architecture`` 是 llama.cpp/量化脚本的**自报标签** —— 张量名认不出时用它兜底，
#: 但要标明 ``confidence="declared"``（自报 ≠ 核实 ✓）。
_GGUF_DECLARED_KIND: dict[str, tuple[str, str]] = {
    "wan": ("video", ""),
    "hunyuan-video": ("video", ""),
    "ltxv": ("video", ""),
    "ltxv-2": ("video", ""),
    "cogvideox": ("video", ""),
    "minimax_h3": ("video", ""),
    "minimax-music3": ("audio", ""),
    "seedvr": ("video", "restore"),
    "qwen_image": ("image", ""),
    "krea2": ("image", ""),
    "flux": ("image", ""),
    "sd": ("image", ""),
    "lumina2": ("image", ""),
    "hidream": ("image", ""),
    "hunyuan3d": ("3d", ""),
}


class FingerprintError(RuntimeError):
    """文件不是可识别的权重格式，或头已损坏。``reason`` 是给人看的短句。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Architecture:
    """一条「架构指纹」。

    ``signatures`` 是**若干组候选**，任意一组里的 key **全部**命中即成立 —— 因为同一架构
    在不同导出工具/不同版本下 key 命名会变（实测：FLUX 新旧档一个叫 ``img_mod`` 一个叫
    ``double_stream_modulation_img``；Qwen-Image 2.1 把 ``txt_norm`` 改成了 ``txt_in.text_norm``）。
    只用一组签名会把「改名后的新版权重」全判成 unknown ✗。
    """

    name: str
    family: str
    kind: str                       #: video / image / 3d / audio / detect / depth
    signatures: tuple[tuple[str, ...], ...]
    note: str = ""
    audio: bool = False             #: 该架构**同时**生成音频（H3 这类联合模型 ✓）
    subtype_hint: str = ""          #: 固定子类型（视频修复类模型没有 t2v/i2v 之分 ✓）


#: ⚠️ **顺序即特异性**：更独特的判据排前面，先命中先返回。
#: 每条签名都对着真机权重核实过（`tmp/_probe_assert.py` 的核验结果）。
ARCHITECTURES: tuple[Architecture, ...] = (
    # ── 视频生成（本模块主角）─────────────────────────────────────────────
    Architecture("minimax_h3", "MiniMax H3", "video",
                 (("video_patch_proj.weight", "audio_patch_proj.weight"),),
                 "音视频联合 DiT：一次出画面 + 声音，本项目短剧主线", audio=True),
    Architecture("hunyuan_video", "HunyuanVideo", "video",
                 (("txt_in.individual_token_refiner.blocks.0.norm1.weight",),
                  ("txt_in.individual_token_refiner.blocks.0.norm1.bias",)),
                 "腾讯混元视频 DiT"),
    Architecture("wan", "Wan 2.1/2.2", "video",
                 (("head.modulation", "head.head.weight", "patch_embedding.weight"),),
                 "阿里万相；t2v/i2v 由 patch_embedding 入通道数判（16/36，实测 ✓）"),
    Architecture("ltxv", "LTX-Video", "video",
                 (("adaln_single.emb.timestep_embedder.linear_1.bias",),
                  ("audio_adaln_single.linear.weight",)),
                 "Lightricks LTX：极快低步数，适合草稿"),
    Architecture("mochi_preview", "Genmo Mochi", "video",
                 (("t5_yproj.weight",),), "Genmo Mochi 1 preview"),
    Architecture("cogvideox", "CogVideoX", "video",
                 (("blocks.0.norm1.linear.weight",),), "智谱 CogVideoX"),
    Architecture("cosmos_predict2", "NVIDIA Cosmos Predict2", "video",
                 (("blocks.0.mlp.layer1.weight",),), "世界模型系（预测后续画面）"),
    Architecture("cosmos", "NVIDIA Cosmos", "video",
                 (("blocks.block0.blocks.0.block.attn.to_q.0.weight",),), "Cosmos 初代"),
    Architecture("seedvr2", "SeedVR2", "video",
                 (("vid_in.proj.weight", "vid_out.proj.weight", "emb_in.proj_out.weight"),
                  ("blocks.35.mlp.all.proj_in_gate.weight",),
                  ("blocks.31.mlp.all.proj_in_gate.weight",)),
                 "视频超分/修复（不是生成）；3B/7B 层数不同，故用 vid_in/vid_out 判",
                 subtype_hint="restore"),
    Architecture("kandinsky5", "Kandinsky 5", "video",
                 (("visual_transformer_blocks.0.cross_attention.key_norm.weight",),),
                 "Kandinsky 5（含 lite video 变体）"),
    # ── 音频生成 ──────────────────────────────────────────────────────────
    Architecture("minimax_music3", "MiniMax Music 3", "audio",
                 (("diffusion_transformer.transformer.project_in.weight",),),
                 "音乐生成 DiT"),
    # ── 3D ────────────────────────────────────────────────────────────────
    Architecture("hunyuan3d2_1", "Hunyuan3D 2.1", "3d",
                 (("t_embedder.mlp.2.weight", "latent_in.weight"),), "混元 3D 2.1"),
    Architecture("hunyuan3d2", "Hunyuan3D 2.0", "3d",
                 (("latent_in.weight", "cond_in.weight"),), "混元 3D 2.0"),
    # ── 图像生成（列出来是为了「认出它不是视频」，而不是全丢 unknown）────
    Architecture("flux2", "FLUX.2", "image",
                 (("double_stream_modulation_img.lin.weight", "cond_type_embedding.weight"),),
                 "黑森林 FLUX.2"),
    Architecture("flux", "FLUX.1", "image",
                 (("double_blocks.0.img_mod.lin.weight", "img_in.weight"),
                  ("double_stream_modulation_img.lin.weight", "img_in.weight")),
                 "黑森林 FLUX.1（新旧两种 key 命名都在，实测核实 ✓）"),
    Architecture("krea2", "Krea 2", "image",
                 (("txtfusion.projector.weight",),), "Krea 2（K2）"),
    Architecture("qwen_image", "Qwen-Image", "image",
                 (("img_in.weight", "txt_in.text_norm.weight"),
                  ("img_in.weight", "txt_norm.weight")),
                 "通义千问图像（2.1 改名为 txt_in.text_norm，实测 ✓）"),
    Architecture("sdxl", "SDXL", "image",
                 (("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight",
                   "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight"),),
                 "Stable Diffusion XL（双文本编码器是硬标志）"),
    Architecture("hidream", "HiDream", "image",
                 (("caption_projection.0.linear.weight", "t_embedder1.mlp.0.weight"),), "HiDream"),
    Architecture("lumina2", "Lumina 2", "image",
                 (("cap_embedder.1.weight",),), "Lumina 2 架构（Z-Image 同族）"),
    Architecture("pixart_sigma", "PixArt-Σ", "image",
                 (("t_block.1.weight",),), "PixArt Sigma"),
    Architecture("ernie", "ERNIE-Image", "image",
                 (("layers.0.mlp.linear_fc2.weight",),), "百度 ERNIE 图像"),
    # ── 视觉专用（检测/深度）───────────────────────────────────────────────
    Architecture("RT_DETR_v4", "RT-DETR v4", "detect",
                 (("encoder.pan_blocks.1.cv4.conv.weight",),), "目标检测"),
    Architecture("DepthAnything3", "Depth Anything 3", "depth",
                 (("backbone.embeddings.patch_embeddings.projection.weight",),), "单目深度"),
)

#: 视频家族的**能力语义**（不是判定依据，只用于给前端/智能体一句人话 ✓）
_VIDEO_FAMILY_NOTES = {
    "wan": "万相：t2v 文生视频 / i2v 图生视频 / vace 统一编辑 / s2v 语音驱动 / camera 运镜；"
           "版本（2.1/2.2）在权重里同构，判不出，别硬报 ✗",
    "minimax_h3": "MiniMax H3：音视频同出（一次生成画面 + 声音），本项目短剧主线用它",
    "hunyuan_video": "腾讯混元视频：t2v/i2v，中文语义强",
    "ltxv": "LTX-Video：极快、低步数，适合草稿与首帧验证",
    "cogvideox": "智谱 CogVideoX：老牌 t2v/i2v，生态工具多",
    "mochi_preview": "Genmo Mochi：开源 t2v 长镜头",
    "cosmos": "NVIDIA Cosmos：世界模型系，偏物理一致性",
    "cosmos_predict2": "NVIDIA Cosmos Predict2：世界模型系（预测下一段画面）",
    "seedvr2": "SeedVR2：视频超分/修复（修既有片子，不生成）",
    "kandinsky5": "Kandinsky 5：含 lite video 变体",
}


def _prefix_of(name: str, suffix: str) -> str | None:
    """`"model.diffusion_model.head.modulation"` + `"head.modulation"` → `"model.diffusion_model"`。"""
    if name == suffix:
        return ""
    if not name.endswith(suffix):
        return None
    head = name[: -len(suffix)]
    if not head.endswith("."):
        return None
    return head[:-1]


def _join(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _match_signature(tensors: Mapping[str, Any], names: Sequence[str], signature: Sequence[str]) -> list[str] | None:
    """试一组签名：所有 key 必须落在**同一个前缀**上，返回命中清单（含前缀）。"""
    for first in signature:
        prefix = None
        for name in names:
            prefix = _prefix_of(name, first)
            if prefix is not None:
                break
        if prefix is None:
            continue
        evidence = [_join(prefix, key) for key in signature]
        if all(item in tensors for item in evidence):
            return evidence
    return None


def detect_architecture(tensors: Mapping[str, Any]) -> dict[str, Any]:
    """从 ``{key: …}`` 判定架构。返回统一结论（未命中给 ``unknown``，**不猜** ✓）。

    ``prefix`` 的处理：上游会先用 ``unet_prefix_from_state_dict`` 切掉
    ``model.diffusion_model.`` 这类外壳；这里等价地**对每组签名反推前缀**，
    并要求同组的所有 key 落在**同一个前缀**上 —— 否则「A 的 key 在前缀 1、B 的 key 在前缀 2」
    会被误拼成一个架构 ✗。
    """
    names = list(tensors)
    if not names:
        return {"architecture": "unknown", "kind": "unknown", "family": "", "prefix": "",
                "evidence": [], "confidence": "none", "reason": "空 state_dict"}
    for arch in ARCHITECTURES:
        for signature in arch.signatures:
            evidence = _match_signature(tensors, names, signature)
            if evidence is None:
                continue
            prefix = evidence[0][: len(evidence[0]) - len(signature[0])].rstrip(".") if signature[0] in evidence[0] else ""
            return {
                "architecture": arch.name,
                "kind": arch.kind,
                "family": arch.family,
                "audio": arch.audio,
                "prefix": prefix,
                "evidence": evidence,
                "confidence": "exact" if len(evidence) > 1 else "single-key",
                "subtype": (arch.subtype_hint or video_subtype(tensors, prefix))
                if arch.kind == "video" else "",
                "note": _VIDEO_FAMILY_NOTES.get(arch.name, arch.note),
            }
    return {"architecture": "unknown", "kind": "unknown", "family": "", "prefix": "",
            "evidence": [], "confidence": "none",
            "reason": "没有任何已知指纹命中（不是猜，是真的不认识）"}


def video_subtype(tensors: Mapping[str, Any], prefix: str) -> str:
    """**视频子类型** —— 决定这个权重能不能吃参考图/参考视频/音频。

    判据分两层，先 key 后 **shape**：

    * key 层照抄上游 Wan 分支的 ``model_type`` 赋值顺序（vace → camera → s2v → … → i2v/t2v）；
      ⚠️ 顺序不能重排（camera 也带 img_emb）✗；
    * **shape 层是本模块补的真判据**：实测 Wan2.2 的 i2v 权重里**没有** ``img_emb``（上游那套
      key 判据是 2.1 时代的），但 ``patch_embedding.weight`` 的第二维是 **36**（16 路 latent
      + 20 路参考帧），t2v 是 **16**。所以看通道数才判得出 2.2 的 i2v ✓ ——
      只靠 key 会把 `wan2.2_i2v_14B` 判成 `t2v`（本机实测就是错的，已修）。
    """
    def has(suffix: str) -> bool:
        return _join(prefix, suffix) in tensors

    if has("vace_patch_embedding.weight"):
        return "vace"
    if has("control_adapter.conv.weight"):
        return "camera" if has("img_emb.proj.0.bias") else "camera_2.2"
    if has("casual_audio_encoder.encoder.final_linear.weight"):
        return "s2v"
    if has("patch_embedding_audio.weight") or has("audio_patch_embedding.weight"):
        return "animate"
    # Wan-Animate 2（真机 ``wan_animate_2_int8_convrot`` 实测）：没有上面那些 ``patch_embedding_*``，
    # 但它给 cross_attn **多开了一路参考图支路** ``k_img``/``v_img`` + ``norm_k_img``（普通 i2v 没有）⇒
    # 这才是「参考图驱动人物」的硬标志；只看入通道数 36 会把它误判成普通 i2v ✗
    if has("blocks.0.cross_attn.k_img.weight"):
        return "animate"
    if has("patch_embedding_mask.weight"):
        return "scail2"
    if has("patch_embedding_pose.weight"):
        return "scail"
    if has("patch_embedding_global.weight"):
        return "wandancer"
    if has("img_emb.proj.0.bias"):
        # 首尾帧能力：上游用 ``img_emb.emb_pos`` 的 shape[1] 记 token 数（flf 专用位置嵌入 ✓）
        return "flf2v" if has("img_emb.emb_pos") else "i2v"

    channels = _in_channels(tensors.get(_join(prefix, "patch_embedding.weight")))
    if channels == 36:
        return "i2v"
    if channels == 16:
        return "t2v"
    return "unknown"


def _in_channels(tensor: Any) -> int:
    """``patch_embedding.weight`` 的 ``[out, in, kt, kh, kw]`` 取 ``in``（拿不到返回 0 ✓）。"""
    # 两种形态都收：读头回来的是 ``{"dtype":…, "shape":[…]}``；调用方手搓的常直接给 ``(out, in, …)`` 元组
    shape: Any
    if isinstance(tensor, Mapping):
        shape = tensor.get("shape")
    else:
        shape = tensor
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return 0
    try:
        return int(shape[1])
    except (TypeError, ValueError):
        return 0


def is_video(detection: Mapping[str, Any]) -> bool:
    return str(detection.get("kind") or "") == "video"


# ── safetensors ──────────────────────────────────────────────────────────────
def read_safetensors_header(path: str) -> dict[str, Any]:
    """读 safetensors 头：``{tensors: {名: {"dtype":…, "shape":[…]}}, metadata: {...}, bytes}``。

    格式：``u64 小端 header_len`` ＋ ``header_len`` 字节 JSON ＋ 张量数据。
    所以**只用读前 header_len+8 字节**，几 GB 的权重也一样快 ✓。
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise FingerprintError(f"读不到文件：{exc}") from exc
    with open(path, "rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise FingerprintError("文件太小，不是 safetensors")
        (header_len,) = struct.unpack("<Q", head)
        if header_len <= 2 or header_len > SAFETENSORS_MAX_HEADER:
            raise FingerprintError(f"safetensors 头长度异常（{header_len} 字节）")
        if header_len + 8 > size:
            raise FingerprintError("safetensors 头超出文件长度（文件被截断）")
        raw = handle.read(header_len)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FingerprintError(f"safetensors 头不是合法 JSON：{exc}") from exc
    if not isinstance(payload, Mapping):
        raise FingerprintError("safetensors 头不是对象")
    metadata = payload.get("__metadata__")
    tensors: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if key == "__metadata__" or not isinstance(value, Mapping):
            continue
        offsets = value.get("data_offsets")
        tensors[key] = {
            "dtype": str(value.get("dtype") or ""),
            "shape": list(value.get("shape") or []),
            # ⭐ data_offsets 是 safetensors 规范里的**数据区坐标**（不是我们算的）：
            # 拿它对账文件大小，就能当场判出「权重被截断/尾巴多了东西」 ✓
            "dataOffsets": list(offsets) if isinstance(offsets, (list, tuple)) else [],
        }
    return {
        "format": "safetensors",
        "tensors": tensors,
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        "bytes": size,
        "dataStart": 8 + header_len,
    }


# ── GGUF ────────────────────────────────────────────────────────────────────
def _read_exact(handle: Any, size: int, what: str) -> bytes:
    """精确读 ``size`` 字节 —— **不许把 ``struct.error`` 漏给调用方**。

    真机踩过：GGUF 的 tensor info 被截断（量化中断）时 ``handle.read(8)`` 只回 2 字节，
    ``struct.unpack`` 抛的是 ``struct.error``，而 ``fingerprint_file`` 只兜 ``FingerprintError``
    ⇒ **一个坏文件就能让整轮扫描崩掉** ✗。读不全一律转成可报的 FingerprintError ✓。
    """
    raw = handle.read(size)
    if len(raw) != size:
        raise FingerprintError(f"GGUF {what} 被截断（要 {size} 字节，只读到 {len(raw)}）")
    return raw


def _read_gguf_string(handle: Any) -> str:
    (length,) = struct.unpack("<Q", _read_exact(handle, 8, "字符串长度"))
    if length > 1 << 20:
        raise FingerprintError(f"GGUF 字符串过长（{length} 字节）")
    return _read_exact(handle, length, "字符串").decode("utf-8", "replace")


def _read_gguf_value(handle: Any, value_type: int) -> Any:
    if value_type == _GGUF_STRING:
        return _read_gguf_string(handle)
    if value_type == _GGUF_ARRAY:
        (item_type,) = struct.unpack("<I", _read_exact(handle, 4, "数组元素类型"))
        (count,) = struct.unpack("<Q", _read_exact(handle, 8, "数组长度"))
        if item_type == _GGUF_STRING:
            kept: list[str] = []
            for index in range(count):
                item = _read_gguf_string(handle)
                if index < _GGUF_ARRAY_KEEP:
                    kept.append(item)
            return kept if len(kept) == count else {"count": count, "head": kept}
        width = _GGUF_FIXED.get(item_type)
        if width is None:
            raise FingerprintError(f"GGUF 数组元素类型未知（{item_type}）")
        # 数值数组一个都不留：我们只关心「声明了没有」，数值本身对判架构无用 ✓
        handle.seek(width * count, os.SEEK_CUR)
        return {"count": count}
    width = _GGUF_FIXED.get(value_type)
    if width is None:
        raise FingerprintError(f"GGUF 值类型未知（{value_type}）")
    raw = handle.read(width)
    if len(raw) != width:
        raise FingerprintError("GGUF 头被截断")
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}[value_type]
    return struct.unpack(fmt, raw)[0]


def read_gguf_tensors(handle: Any, tensor_count: int) -> dict[str, dict[str, Any]]:
    """从 kv 表之后接着读 **tensor info**（名 + 形状），读到即停，不碰数据区。

    每条：``name(string) | n_dims(uint32) | dims(uint64×n) | type(uint32) | offset(uint64)``。
    DiT 一两千条、LLM 也只有几百条 ⇒ 仍是毫秒级 ✓。
    """
    if tensor_count < 0 or tensor_count > GGUF_MAX_TENSORS:
        raise FingerprintError(f"GGUF 张量数量异常（{tensor_count}）")
    tensors: dict[str, dict[str, Any]] = {}
    for _ in range(tensor_count):
        name = _read_gguf_string(handle)
        # ⚠️ 记住这个零字节陷阱：tensor_count 写大了（或数据区提前开始）时，
        # 读指针会吃进数据区的 ``\x00``，解出「空名字 + 0 维」这种**合法但无意义**的条目 ——
        # 不拦的话整份权重会被当成"读成功"，只是架构/字节账全是垃圾 ✗
        if not name or not name.strip("\x00").strip():
            raise FingerprintError("GGUF tensor info 损坏（解析出空张量名）")
        raw = handle.read(4)
        if len(raw) != 4:
            raise FingerprintError("GGUF tensor info 被截断")
        (n_dims,) = struct.unpack("<I", raw)
        if n_dims < 1 or n_dims > 8:
            raise FingerprintError(f"GGUF 张量维数异常（{n_dims}）")
        raw = handle.read(8 * n_dims + 12)
        if len(raw) != 8 * n_dims + 12:
            raise FingerprintError("GGUF tensor info 被截断")
        dims = list(struct.unpack(f"<{n_dims}Q", raw[: 8 * n_dims]))
        ggml_type, = struct.unpack("<I", raw[8 * n_dims: 8 * n_dims + 4])
        (data_offset,) = struct.unpack("<Q", raw[8 * n_dims + 4: 8 * n_dims + 12])
        tensors[name] = {"dtype": f"ggml:{ggml_type}", "shape": dims,
                         "ggmlType": ggml_type, "dataOffset": data_offset}
    return tensors


def read_gguf_header(path: str, *, with_tensors: bool = True) -> dict[str, Any]:
    """读 GGUF 的 kv 元数据（＋ 默认顺带读 tensor info）。

    Ollama 的 blob、ComfyUI 的 `*-GGUF` 视频权重都是这个格式。里面有的是真判据：
    ``general.architecture``、``*.block_count``、``clip.has_vision_encoder``、
    ``*.pooling_type``、``tokenizer.chat_template`` —— 比「看文件名像不像 vl」硬得多 ✓。

    ⚠️ 为什么要读 tensor info：``general.architecture`` 是**量化脚本自己写的标签**，
    真机上有写错的（``krea2_turbo-Q4_K_M.gguf`` 标成 ``qwen_image``）。张量名才是权重本体 ✗。
    ``with_tensors=False`` 供只关心 kv 的调用方（能力判定）省一次读盘。
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise FingerprintError(f"读不到文件：{exc}") from exc
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise FingerprintError("不是 GGUF（magic 不匹配）")
        (version,) = struct.unpack("<I", _read_exact(handle, 4, "版本号"))
        if version < 2 or version > 3:
            raise FingerprintError(f"GGUF 版本不支持（{version}）")
        tensor_count, kv_count = struct.unpack("<QQ", _read_exact(handle, 16, "头部计数"))
        if kv_count > 100_000:
            raise FingerprintError(f"GGUF kv 数量异常（{kv_count}）")
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(handle)
            (value_type,) = struct.unpack("<I", _read_exact(handle, 4, "kv 值类型"))
            metadata[key] = _read_gguf_value(handle, value_type)
        offset = handle.tell()
        tensors: dict[str, dict[str, Any]] = {}
        if with_tensors:
            try:
                tensors = read_gguf_tensors(handle, tensor_count)
            except (FingerprintError, struct.error):
                # tensor info 坏了不影响 kv 结论（架构回落成 declared ✓）——
                # 这里连 ``struct.error`` 一起兜：宁可拿不到张量指纹，也不许炸掉整轮扫描 ✗
                tensors = {}
        # 数据区起点 = tensor info 结束处按 general.alignment 对齐（llama.cpp 的写入口径）✓
        data_start: int | None = None
        if tensors:
            data_start = _align(handle.tell(), _gguf_alignment(metadata))
    return {
        "format": "gguf",
        "version": version,
        "tensorCount": tensor_count,
        "metadata": metadata,
        "tensors": tensors,
        "bytes": size,
        "metadataEnd": offset,
        "dataStart": data_start,
    }


# ── 量化账与完整性（只能靠读权重本体算） ──────────────────────────────────────
# safetensors 的 dtype → 每元素字节（规范里就这几个；不认识的一律**不猜**）
_SAFE_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
}

# ggml type → (名字, 每块元素数, 每块字节数)。数值取自 ggml.c 的 type_traits ✓
# ⚠️ 表里没有的类型（各代 IQ*）**不给字节数**：宁可这条算不精确，也不许拿近似值冒充 ✓
_GGML_BLOCKS: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}

# 文件名里常见的量化自称 → 能确认它成立的 dtype（**最长优先**，避免 q4_k 抢了 q4_k_m）
_NAME_CLAIMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("q4_k_m", ("Q4_K",)), ("q4_k_s", ("Q4_K",)), ("q5_k_m", ("Q5_K",)),
    ("q5_k_s", ("Q5_K",)), ("q8_k", ("Q8_K",)), ("q6_k", ("Q6_K",)),
    ("q8_0", ("Q8_0",)), ("q5_0", ("Q5_0",)), ("q4_0", ("Q4_0",)),
    ("q2_k", ("Q2_K",)), ("q3_k", ("Q3_K",)),
    ("bf16", ("BF16",)), ("fp16", ("F16",)), ("fp32", ("F32",)),
    ("fp8", ("F8_E4M3", "F8_E5M2")), ("f8", ("F8_E4M3", "F8_E5M2")),
    ("int8", ("I8", "U8")),
)

# 这些自称**从 dtype 上根本定不下来**：4bit 权重在 safetensors 里也是按 U8 字节打包存的，
# dtype 只暴露「打包容器」，不暴露位宽 ⇒ 只能如实标 unverifiable，不许判 verified/refuted ✗
# ⚠️ ``convrot`` 有意不在此列：它是 ComfyUI 的**权重旋转技法**名，不是位宽声明 ——
# 把它当"自称"会额外产出一堆 unverifiable 噪音（真机 154 份里有一半带这后缀）✗
_UNVERIFIABLE_CLAIMS = ("int4", "w4a8", "w4a4", "w8a8", "nvfp4")

# 文件名里的「参数量自称」：`wan2.2_..._14B_fp8`、`qwen3-vl-4b-heretic`、`seedvr2_ema_7b`
# 前后边界都用「非字母数字」卡住，免得把 `fp16` 的 `16`、`Q8_K` 的 `8` 当成参数量 ✗
_PARAM_CLAIM_RE = re.compile(r"(?<![0-9a-zA-Z])(\d+(?:\.\d+)?)[bB](?![0-9a-zA-Z])")

# **量化标记**（判「同一模型的哪几份只是精度不同」时要从文件名里剥掉的词）。
# ⚠️ 只收**确指精度/打包方式**的词 —— `sharp`（锐化蒸馏，权重真的不同）、`pruned`、`turbo`、
# `distill` 一律不收：真机 `seedvr2_ema_7b` 与 `seedvr2_ema_7b_sharp` 剥离后必须仍是**两个基名** ✓
_QUANT_MARKERS: tuple[str, ...] = tuple(sorted((
    "fp8_e4m3fn", "fp8_e5m2", "fp8_e8m0", "fp8", "fp16", "bf16", "fp32", "f16", "f32",
    "e4m3fn", "e5m2", "int8", "int4", "w4a8", "w4a4", "w8a8", "nvfp4", "convrot",
    "q2_k", "q3_k", "q4_0", "q4_k_m", "q4_k_s", "q5_0", "q5_k_m", "q5_k_s",
    "q6_k", "q8_0", "q8_k", "mixed_block35", "mixed", "scaled",
), key=len, reverse=True))


def base_name(filename: str) -> str:
    """**剥掉量化标记后的基名** —— `seedvr2_ema_7b_fp8_e4m3fn` → `seedvr2_ema_7b`。

    它的用途只有一个：判「这几份是不是**同一个模型的不同精度**」。真机依据（本机 154 份）：
    `seedvr2_ema_7b` 一族剥完正是 3 份（fp8 7.67 GiB / fp8-mixed-block35 7.88 GiB / fp16 15.35 GiB），
    而 `seedvr2_ema_7b_sharp` 因 `sharp` **不在**剥离表里，被正确分成另一组 ✓。

    ⚠️ 长词先剥（表按长度倒序）：否则 `fp8` 会先吃掉 `fp8_e4m3fn` 的一半、留下 `e4m3fn` 残渣 ✗
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    for marker in _QUANT_MARKERS:
        stem = stem.replace(marker, "")
    return re.sub(r"[-_.]{2,}", "_", stem).strip("_-. ")


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _gguf_alignment(metadata: Mapping[str, Any]) -> int:
    """GGUF 数据区对齐（``general.alignment``，默认 32）。异常值一律退回 32 ✓。"""
    try:
        alignment = int(metadata.get("general.alignment", 32) or 32)
    except (TypeError, ValueError):
        return 32
    return alignment if 1 <= alignment <= (1 << 20) else 32


def _elements(tensor: Mapping[str, Any]) -> int:
    count = 1
    for dim in tensor.get("shape") or []:
        try:
            count *= int(dim)
        except (TypeError, ValueError):
            return 0
    return count


def parameters(tensors: Mapping[str, Any]) -> int:
    """**张量元素总数** —— 只由形状决定，与 dtype/量化无关，所以是精确值 ✓。

    ⚠️ 一个必须讲清的语义边界：4bit 权重在盘上是**按 U8 打包**存的（两个 4bit 挤一字节），
    所以这类文件的元素总数 ≈ **真实参数的一半**（真机实证：``seedvr2_3b_int4`` 1.87B、
    ``seedvr2_3b_int8`` 3.39B、``qwen3vl_4b_int4`` 2.42B、``qwen3vl_4b_int8`` 4.44B）。
    因此这里只报"存了多少个元素"这个事实，**不替调用方换算成参数量** ✗。
    """
    total = 0
    for tensor in tensors.values():
        if isinstance(tensor, Mapping):
            total += _elements(tensor)
    return total


def shape_digest(tensors: Mapping[str, Any]) -> str:
    """**(张量名, 形状) 的指纹** —— 「这是不是同一个模型的另一份量化/副本」的机器判据 ✓。

    真机实证（本机 154 份权重里最典型的一组）：``seedvr2_ema_7b`` 一族有 **11 份**
    （fp16 / fp8_e4m3fn / fp8-mixed-block35-fp16 / Q4_K_M / Q6_K_M / Q8_K_M / sharp-…），
    张量名与形状**逐条一致、只有 dtype 不同** ⇒ 形状指纹相同、字节数与量化名不同。

    ⚠️ 边界写清楚：只覆盖**名字 + 形状**，**不含 dtype、不含一个字节的数据** ⇒
    认「同模型的不同变体」可靠，认「内容相同」**不可靠**（同架构同形状的两份不同权重
    也会撞上）✗ —— 要判内容相同必须全量哈希，那是另一件事。
    """
    if not tensors:
        return ""
    lines = []
    for name in sorted(tensors):
        tensor = tensors[name]
        shape = list(tensor.get("shape") or []) if isinstance(tensor, Mapping) else []
        lines.append(f"{name}:{'x'.join(str(int(d)) for d in shape)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def quantization(tensors: Mapping[str, Any], fmt: str = "") -> dict[str, Any]:
    """**权重本体称出来的量化账** —— 文件名说 ``fp8``/``Q4_K_M`` 时，这儿才是真相。

    两条算法都只用头里的信息（形状 ＋ dtype/ggml type），**不读一个字节的数据区** ✓：
    * safetensors：``元素数 × dtype 字节``；
    * GGUF：``元素数 ÷ 每块元素 × 每块字节``（Q4_K 是 256 个元素 144 字节，不是 0.5 字节/元素 ✗）。

    ``exact=False`` 表示**有张量算不出来**（dtype/ggml type 不在表里，如各代 IQ*）——
    此时 ``weightsBytes`` 是**已知部分的下界**，调用方不许把它当总量报 ✗。
    """
    by_type: dict[str, int] = {}
    total = 0
    unknown = 0
    for info in tensors.values():
        if not isinstance(info, Mapping):
            unknown += 1
            continue
        elems = _elements(info)
        if elems <= 0:
            unknown += 1
            continue
        ggml_type = info.get("ggmlType")
        if isinstance(ggml_type, int) and ggml_type >= 0:
            spec = _GGML_BLOCKS.get(ggml_type)
            if spec is None:
                unknown += 1
                continue
            label, block, block_bytes = spec
            if block > 1 and elems % block:
                unknown += 1  # 行宽不是整块 ⇒ 结构已坏，不硬算
                continue
            size = (elems // block) * block_bytes
        else:
            label = str(info.get("dtype") or "")
            width = _SAFE_DTYPE_BYTES.get(label)
            if width is None:
                unknown += 1
                continue
            size = elems * width
        total += size
        by_type[label] = by_type.get(label, 0) + size
    ratios = ({k: round(v / total, 4) for k, v in by_type.items()} if total else {})
    dominant = max(ratios.items(), key=lambda kv: kv[1])[0] if ratios else "unknown"
    scheme = dominant if ratios.get(dominant, 0) >= 0.95 else "mixed"
    return {
        "scheme": scheme,
        "types": dict(sorted(ratios.items(), key=lambda kv: -kv[1])),
        "weightsBytes": total,
        "exact": unknown == 0,
        "unknownTensors": unknown,
        "format": fmt,
    }


def integrity(header: Mapping[str, Any]) -> dict[str, Any]:
    """**用头里自带的坐标对账文件大小** —— 判「这份权重是不是被截断了」。

    为什么这是真事实而不是推断：
    * safetensors 的每条张量都写了 ``data_offsets: [begin, end]``（规范强制），
      数据区总长 = ``max(end)``，必须与 ``文件大小 - 头长度`` **相等**；
    * GGUF 的每条 tensor info 写了 ``offset``，末条的 ``offset + 本张量字节`` 必须与
      ``文件大小 - 数据区起点`` 相等（数据区起点按 ``general.alignment`` 对齐）。

    所以：``截断``（下不动）、``尾部多出``（多半是拼接/传输损坏）都能当场点名，
    而且**不用读数据区**，几十 GB 的权重一样毫秒级 ✓。
    """
    fmt = str(header.get("format") or "")
    size = int(header.get("bytes") or 0)
    data_start = header.get("dataStart")
    tensors = header.get("tensors") or {}
    declared: int | None = None
    if isinstance(data_start, int) and tensors:
        if fmt == "safetensors":
            ends = [int(t["dataOffsets"][-1]) for t in tensors.values()
                    if isinstance(t, Mapping) and len(t.get("dataOffsets") or []) == 2]
            if len(ends) == len(tensors):
                declared = max(ends)
        else:
            ends = []
            for info in tensors.values():
                if not isinstance(info, Mapping) or "dataOffset" not in info:
                    continue
                elements = _elements(info)
                ggml_type = info.get("ggmlType")
                # ⚠️ 别写 ``int(x or -1)``：ggml **type 0 就是 F32**，会被 ``or`` 吞成 -1 ✗
                # （真机踩过：含 F32 范数的 GGUF 全部对账失败、被判 unknown）
                spec = _GGML_BLOCKS.get(ggml_type) if isinstance(ggml_type, int) else None
                if spec is None or (spec[1] > 1 and elements % spec[1]):
                    ends = []
                    break
                ends.append(int(info["dataOffset"]) + (elements // spec[1]) * spec[2])
            if ends:
                declared = max(ends)
    if declared is None or not isinstance(data_start, int):
        return {"ok": True, "status": "unknown", "declaredBytes": None,
                "actualBytes": None, "delta": None,
                "note": "头里没有可用的数据区坐标（tensor info 读不全），无法对账"}
    actual = size - data_start
    delta = actual - declared
    if delta == 0:
        status, ok, note = "exact", True, ""
    elif delta < 0:
        status, ok = "truncated", False
        note = f"数据区少了 {-delta} 字节（文件被截断或拷贝中断）"
    elif delta <= 4096:
        status, ok, note = "padded", True, f"尾部多 {delta} 字节（对齐/填充，通常无害）"
    else:
        status, ok = "trailing", False
        note = f"尾部多 {delta} 字节（不是对齐量级，可能拼接过别的数据）"
    return {"ok": ok, "status": status, "declaredBytes": declared,
            "actualBytes": actual, "delta": delta, "note": note}


def name_audit(name: str, quant: Mapping[str, Any], params: int = 0) -> list[dict[str, Any]]:
    """**文件名自称 vs 权重本体** —— 逐条给 verified/refuted/unverifiable，两个维度：

    * ``dimension="quant"``：``*_fp8_e4m3fn_mixed_block35_fp16`` / ``*_Q4_K_M`` / ``*_int4_convrot``
      —— 前两类能核实，第三类**核不了**（4bit 按 U8 打包存，dtype 只看到容器）⇒ 如实标
      ``unverifiable``，不许为了好看判成 verified ✗；
    * ``dimension="parameters"``：``_14B_`` / ``_7b_`` / ``_4b_`` 这类**规模自称**，
      拿元素总数比。真机实证的容差是必要的：`7B` 实际 8.24B（×1.18）、`3B` 实际 3.39B（×1.13）、
      `4B` 实际 4.44B（×1.11）、`14B` 实际 14.29B（×1.02）—— 名字里的 B 指的是**底模档位**，
      本体常多出视觉塔/文本塔 ⇒ 判据是「同一档」而不是「相等」✓。

    ⚠️ 三条**一律不判**的边界（真机各有一例，判了就是假警报 ✗）：
    ① 名字带 ``int4/w4a8/nvfp4`` 的：4bit 打包存 ⇒ 元素数只有真实参数**一半**
       （`seedvr2_3b_int4` 1.87B vs `seedvr2_3b_int8` 3.39B）；
    ② 比值 <0.5 或 >2 的：`lightx2v_I2V_14B_…_rank64_distill` 实际 0.37B —— 那是 **LoRA**，
       名字里的 14B 是它要挂的**底模**，不是它自己；
    ③ 比值落在 0.5–0.75 / 1.25–2 的：可能是裁剪/蒸馏版，只标 ``unverifiable`` 并写明实际值。
    """
    lower = os.path.basename(name).lower()
    found = set((quant.get("types") or {}).keys())
    exact = bool(quant.get("exact"))
    claims: list[dict[str, Any]] = []
    claimed_spans: list[str] = []
    packed = any(token in lower for token in _UNVERIFIABLE_CLAIMS)

    def record(token: str, expect: Sequence[str], verdict: str, note: str = "", *,
               found_text: str | None = None, dimension: str = "quant") -> None:
        claims.append({"claim": token, "dimension": dimension, "expect": list(expect),
                       "found": found_text if found_text is not None
                       else (quant.get("scheme") or "unknown"),
                       "verdict": verdict, "note": note})

    for token in _UNVERIFIABLE_CLAIMS:
        if token in lower:
            claimed_spans.append(token)
            record(token, (), "unverifiable",
                   "4bit/nvfp4 类权重在容器里按 U8 打包存盘，dtype 只暴露容器、不暴露位宽")
    for token, expect in _NAME_CLAIMS:
        if token not in lower or any(token in span for span in claimed_spans):
            continue
        claimed_spans.append(token)
        if found & set(expect):
            record(token, expect, "verified")
        elif exact:
            record(token, expect, "refuted", f"本体里没有 {'/'.join(expect)}"
                                            f"（实为 {quant.get('scheme')}）")
        else:
            record(token, expect, "unverifiable", "有张量 dtype 不在已知表里，无法定论")

    if params > 0 and not packed:
        for match in _PARAM_CLAIM_RE.finditer(lower):
            billions = float(match.group(1))
            if billions < 0.1:
                continue
            ratio = (params / 1e9) / billions
            actual = f"{params / 1e9:.2f}B"
            if 0.75 <= ratio <= 1.25:
                record(f"{billions:g}b", (), "verified", f"本体 {actual} 个元素（同档）",
                       found_text=actual, dimension="parameters")
            elif 0.5 <= ratio <= 2.0:
                record(f"{billions:g}b", (), "unverifiable",
                       f"本体 {actual}，可能被裁剪/蒸馏过，不硬判",
                       found_text=actual, dimension="parameters")
            # 带宽外：LoRA/适配器等，名字里的 B 指底模 ⇒ 一个字都不说（说就是假警报）✗
    return claims


def gguf_capabilities(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """从 GGUF kv 判**真能力**（读权重，不是看模板猜）。

    对照上游 ``ggufCapabilities``（``server/images.go``）的口径：
    ``clip.has_vision_encoder`` ⇒ vision（真视觉塔在场）、``pooling_type`` ⇒ embedding。
    ⚠️ 上游那段 Go 要**解 GGUF 张量**才判，这里只读 kv 就能拿到同样两个关键位，
    因此能力名一致、证据更强（kv 是写在头里的**声明**，不是推断 ✓）。
    """
    capabilities: list[str] = []
    sources: dict[str, list[str]] = {}

    def add(capability: str, source: str) -> None:
        if capability not in capabilities:
            capabilities.append(capability)
            sources.setdefault(capability, []).append(source)

    if metadata.get("clip.has_vision_encoder") is True:
        add("vision", "gguf")
    if metadata.get("pooling_type") not in (None, 0, ""):
        add("embedding", "gguf")
    template = str(metadata.get("tokenizer.chat_template") or "")
    if ".Tools" in template or "tools" in template:
        add("tools", "gguf")
    architecture = str(metadata.get("general.architecture") or "")
    return {
        "capabilities": capabilities,
        "sources": sources,
        "architecture": architecture,
        "contextLength": metadata.get(f"{architecture}.context_length")
        or metadata.get("general.context_length"),
        "blockCount": metadata.get(f"{architecture}.block_count"),
        "embeddingLength": metadata.get(f"{architecture}.embedding_length"),
        "vision": metadata.get("clip.has_vision_encoder") is True,
        "hasChatTemplate": bool(template),
    }


def fingerprint_file(path: str) -> dict[str, Any]:
    """**一个文件 → 一份可断言的结论**（按扩展名分派，失败也返回结构 ✓）。

    返回：``path / format / architecture / kind / subtype / family / evidence /
    confidence / size / error``。⚠️ 任何异常都折成 ``error`` 而不抛 —— 扫描器要对上百个
    文件连续跑，一个坏文件不该中断整轮（但必须**记下来**，不能假装它不存在 ✓）。
    """
    name = os.path.basename(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    result: dict[str, Any] = {
        "path": path, "name": name, "size": size, "format": "", "architecture": "unknown",
        "kind": "unknown", "subtype": "", "family": "", "evidence": [], "confidence": "none",
        "error": "",
    }
    try:
        lower = path.lower()
        if lower.endswith(".safetensors"):
            header = read_safetensors_header(path)
            detection = detect_architecture(header["tensors"])
            result.update(detection)
            result["format"] = "safetensors"
            result["tensorCount"] = len(header["tensors"])
            result["metadata"] = _short_metadata(header["metadata"])
            result["quant"] = quantization(header["tensors"], "safetensors")
            result["integrity"] = integrity(header)
            result["parameters"] = parameters(header["tensors"])
            result["shapeDigest"] = shape_digest(header["tensors"])
        elif lower.endswith(".gguf"):
            header = read_gguf_header(path)
            caps = gguf_capabilities(header["metadata"])
            declared = str(caps["architecture"] or "")
            result["format"] = "gguf"
            result["tensorCount"] = header["tensorCount"]
            result["capabilities"] = caps["capabilities"]
            result["declaredArchitecture"] = declared or "unknown"
            result["gguf"] = {
                "contextLength": caps["contextLength"],
                "blockCount": caps["blockCount"],
                "vision": caps["vision"],
            }
            # ⭐ 先用**张量名**判（量化后的 DiT 张量名原样保留 ⇒ 与 safetensors 同一套判据 ✓）。
            # 真机实例：`krea2_turbo-Q4_K_M.gguf` 的 `general.architecture` 自报 `qwen_image`，
            # 而张量里的 `txtfusion.projector.weight` 说明它其实是 Krea2 ⇒ 张量名优先 ✓
            detection = detect_architecture(header["tensors"]) if header["tensors"] else {}
            if detection.get("architecture") not in (None, "", "unknown"):
                result.update(detection)
                result["evidence"] = list(detection.get("evidence") or [])
            else:
                kind, subtype = _GGUF_DECLARED_KIND.get(declared, ("unknown", ""))
                result["architecture"] = declared or "unknown"
                result["family"] = declared
                result["kind"] = kind
                result["subtype"] = subtype
                result["evidence"] = [f"general.architecture={declared}"] if declared else []
                result["confidence"] = "declared" if declared else "none"
                result["note"] = ("按 GGUF 自报的 general.architecture 判定（未核实到张量指纹）"
                                  if declared else "")
            result["metadata"] = _short_metadata(header["metadata"])
            result["quant"] = quantization(header["tensors"], "gguf")
            result["integrity"] = integrity(header)
            result["parameters"] = parameters(header["tensors"])
            result["shapeDigest"] = shape_digest(header["tensors"])
        else:
            raise FingerprintError("扩展名不是 .safetensors / .gguf")
    except FingerprintError as exc:
        result["error"] = exc.reason
    except OSError as exc:
        result["error"] = f"读文件失败：{exc}"
    if not result["error"]:
        result["nameAudit"] = name_audit(name, result.get("quant") or {},
                                        int(result.get("parameters") or 0))
    return result


def _short_metadata(metadata: Mapping[str, Any], limit: int = 16) -> dict[str, Any]:
    """只留前 ``limit`` 个**标量/短值**的 kv（头里可能有巨大数组，别一路搬出去 ✓）。"""
    short: dict[str, Any] = {}
    for key, value in metadata.items():
        if len(short) >= limit:
            break
        if isinstance(value, (str, int, float, bool)):
            text = value if not isinstance(value, str) else value[:120]
            short[str(key)] = text
    return short


def video_architectures() -> tuple[str, ...]:
    """全部已知视频架构名（按表顺序去重 ✓）。"""
    seen: list[str] = []
    for arch in ARCHITECTURES:
        if arch.kind == "video" and arch.name not in seen:
            seen.append(arch.name)
    return tuple(seen)


def summarize(detections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一批指纹 → 按架构聚合（前端要的是「我能拍什么」，不是文件清单 ✓）。

    ⚠️ 读不动的（``error``）单独进 ``unreadable``，**不混进** ``architectures.unknown`` ✗ ——
    「文件坏了」与「认不出架构」是两件事，混一起会让「这台机器认不出几个模型」无法定位 ✓。
    """
    counts: dict[str, int] = {}
    unreadable = 0
    for item in detections:
        if item.get("error"):
            unreadable += 1
            continue
        name = str(item.get("architecture") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    video = [d for d in detections if is_video(d)]
    subtypes: dict[str, int] = {}
    for item in video:
        key = str(item.get("subtype") or "unknown")
        subtypes[key] = subtypes.get(key, 0) + 1
    kinds: dict[str, int] = {}
    quantizations: dict[str, int] = {}
    broken = 0
    weights_bytes = 0
    refuted = 0
    families: dict[str, list[Mapping[str, Any]]] = {}
    for item in detections:
        if item.get("error"):
            continue
        key = str(item.get("kind") or "unknown")
        kinds[key] = kinds.get(key, 0) + 1
        quant = item.get("quant") or {}
        scheme = str(quant.get("scheme") or "")
        if scheme:
            quantizations[scheme] = quantizations.get(scheme, 0) + 1
        weights_bytes += int(quant.get("weightsBytes") or 0)
        check = item.get("integrity") or {}
        if check.get("status") in ("truncated", "trailing"):
            broken += 1
        refuted += sum(1 for c in item.get("nameAudit") or [] if c.get("verdict") == "refuted")
        digest = str(item.get("shapeDigest") or "")
        if digest:
            families.setdefault(digest, []).append(item)
    # ⭐ **同一模型的多份精度副本** —— 两个条件必须**同时**成立：
    #   ① 张量名与形状逐条一致（`shapeDigest` 相同）；
    #   ② 文件名剥掉量化标记后等于同一个基名（`base_name` 相同）。
    # 只用①会当场造假警报（真机实测）：`Qwen-Image-Edit-2509-*` 那 5 份是**不同的 LoRA**
    # （angles/fusion/relight…）、`minimax_h3` 那 4 份是 **4step/8step × fl2v/ref2v**、
    # `*_VAE` 那 4 份是 **Qwen 与 Wan 两家不同的 VAE** —— 它们形状一模一样、权重毫不相干 ✗。
    # 加上②之后真机成 3 组、零假阳性：`seedvr2_ema_7b`(3) / `seedvr2_ema_7b_sharp`(3) / `seedvr2_ema_3b`(2) ✓
    variants: list[dict[str, Any]] = []
    for items in families.values():
        if len(items) < 2:
            continue
        by_base: dict[str, list[Mapping[str, Any]]] = {}
        for item in items:
            by_base.setdefault(base_name(str(item.get("name") or "")), []).append(item)
        for basename, same in by_base.items():
            if len(same) < 2:
                continue
            ordered = sorted(same, key=lambda it: int((it.get("quant") or {}).get("weightsBytes") or 0))
            variants.append({
                "baseName": basename,
                "shapeDigest": str(ordered[0].get("shapeDigest") or ""),
                "architecture": str(ordered[0].get("architecture") or "unknown"),
                "subtype": str(ordered[0].get("subtype") or ""),
                "parameters": int(ordered[0].get("parameters") or 0),
                "count": len(ordered),
                "bytes": sum(int((it.get("quant") or {}).get("weightsBytes") or 0) for it in ordered),
                # 小份在前（"跑得动的那份"排第一）；8 条封顶，免得扫描响应被清单撑爆
                "variants": [{"name": str(it.get("name") or ""),
                              "scheme": str((it.get("quant") or {}).get("scheme") or ""),
                              "weightsBytes": int((it.get("quant") or {}).get("weightsBytes") or 0)}
                             for it in ordered[:8]],
            })
    variants.sort(key=lambda g: (-g["count"], -g["bytes"]))
    return {
        "total": len(detections),
        "unreadable": unreadable,
        "broken": broken,
        "video": len(video),
        "architectures": dict(sorted(counts.items())),
        "kinds": dict(sorted(kinds.items())),
        "videoSubtypes": dict(sorted(subtypes.items())),
        "quantizations": dict(sorted(quantizations.items(), key=lambda kv: -kv[1])),
        "weightsBytes": weights_bytes,
        "nameRefuted": refuted,
        # 去重后的「模型」数（按 张量名+形状 指纹）—— 154 份文件里其实只有 100 来个模型 ✓
        "models": len(families),
        "variantGroups": variants[:12],
    }
