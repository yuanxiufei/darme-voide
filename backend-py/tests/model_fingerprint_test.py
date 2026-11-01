"""自检：**真实权重指纹**（``services/model_fingerprint.py``）＋ 扫描接入 ＋ 2 个自研端点。

被测的主张只有一句：**模型是什么，看权重本身，不看文件名**。分三层：

1. **读得对**：合成 safetensors / GGUF（真格式字节，不是 mock），坏文件必须报错、不许抛异常；
2. **判得对**：用合成 state_dict 验签名与视频子类型，重点是真机踩出来的坑
   （Wan2.2 i2v 没有 ``img_emb``、Wan-Animate 靠 ``cross_attn.k_img``、
   GGUF 的 ``general.architecture`` 可能写错 —— 真机 krea2 标成 qwen_image）；
3. **接得上**：``local_model_scan`` 真的用指纹改写按名字判错的结论，端点真的通。

⚠️ 全程不依赖本机真实权重（tempfile 现造字节），不写任何机器路径 ✓。
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="fingerprint_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import local_model_scan as ls  # noqa: E402
from app.services import model_fingerprint as fp  # noqa: E402

TOTAL = 0
FAILED = 0


def check(name: str, condition: object, detail: object = "") -> None:
    global TOTAL, FAILED
    TOTAL += 1
    if condition:
        print(f"PASS  {name}")
    else:
        FAILED += 1
        text = f"  <{detail}>" if detail else ""
        print(f"FAIL  {name}{text}")


# ============================================================
# 合成文件（真字节）
# ============================================================
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1,
               "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
               "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}

# ggml type → (每块元素, 每块字节)，与被测模块独立的一份（自检不抄被测的表 ✗）
GGML_BYTES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 14: (256, 210), 30: (1, 2)}


def write_safetensors(path: str | Path, tensors: dict[str, tuple[int, ...]],
                      metadata: dict[str, str] | None = None, *,
                      dtypes: dict[str, str] | None = None,
                      truncate: int = 0, extra: int = 0,
                      omit_data: bool = False) -> str:
    """造一份**真格式** safetensors。``truncate``/``extra`` 用来造「数据区对不上」的坏文件。

    ``omit_data=True``：只写头、不写数据区 —— 为了造「14B 元素」这种**头合法但数据区放不下**
    的样本测参数量（真写 28GB 会爆盘，而参数量只由头里的形状决定 ✓）。
    """
    header: dict[str, Any] = {}
    offset = 0
    for name, shape in tensors.items():
        count = 1
        for dim in shape:
            count *= int(dim)
        dtype = (dtypes or {}).get(name, "F16")
        size = count * DTYPE_BYTES.get(dtype, 1)  # 未知 dtype 按 1 字节占位（只为造出结构）
        header[name] = {"dtype": dtype, "shape": [int(d) for d in shape],
                        "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = dict(metadata)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    body = b"" if omit_data else b"\x00" * max(0, offset - truncate)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + body + b"\x00" * extra)
    return str(path)


def _gs(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv_str(key: str, value: str) -> bytes:
    return _gs(key) + struct.pack("<I", 8) + _gs(value)


def kv_u32(key: str, value: int) -> bytes:
    return _gs(key) + struct.pack("<I", 4) + struct.pack("<I", int(value))


def kv_bool(key: str, value: bool) -> bytes:
    return _gs(key) + struct.pack("<I", 7) + struct.pack("<?", bool(value))


def kv_arr_str(key: str, values: list[str]) -> bytes:
    return (_gs(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
            + struct.pack("<Q", len(values)) + b"".join(_gs(v) for v in values))


def kv_arr_u32(key: str, values: list[int]) -> bytes:
    body = b"".join(struct.pack("<I", int(v)) for v in values)
    return (_gs(key) + struct.pack("<I", 9) + struct.pack("<I", 4)
            + struct.pack("<Q", len(values)) + body)


def tensor_info(name: str, dims: tuple[int, ...], ggml_type: int = 1,
                data_offset: int = 0) -> bytes:
    return (_gs(name) + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims)}Q", *[int(d) for d in dims])
            + struct.pack("<I", ggml_type) + struct.pack("<Q", data_offset))


def ggml_size(dims: tuple[int, ...], ggml_type: int) -> int:
    block, block_bytes = GGML_BYTES[ggml_type]
    count = 1
    for dim in dims:
        count *= int(dim)
    return (count // block) * block_bytes


def write_gguf(path: str | Path, kv: list[bytes], tensors: dict[str, tuple[int, ...]],
               version: int = 3, declared_count: int | None = None, *,
               types: dict[str, int] | None = None, truncate: int = 0) -> str:
    """造一份**真格式** GGUF：tensor info 带真 offset，数据区按 32 字节对齐写满 ✓。

    ``truncate`` 用来造「数据区少了几字节」的坏文件（拷贝中断的典型样子）。
    """
    count = len(tensors) if declared_count is None else declared_count
    infos: list[bytes] = []
    data_bytes = 0
    for name, dims in tensors.items():
        ggml_type = (types or {}).get(name, 1)
        infos.append(tensor_info(name, dims, ggml_type, data_bytes))
        data_bytes += ggml_size(dims, ggml_type)
    head = b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", count, len(kv))
    body = b"".join(kv) + b"".join(infos)
    data_start = ((len(head) + len(body) + 31) // 32) * 32
    padding = b"\x00" * (data_start - len(head) - len(body))
    payload = b"\x00" * max(0, data_bytes - truncate)
    Path(path).write_bytes(head + body + padding + payload)
    return str(path)


def raises(fn, needle: str) -> tuple[bool, str]:
    try:
        fn()
    except fp.FingerprintError as exc:
        return (needle in exc.reason, exc.reason)
    return (False, "没有抛 FingerprintError")


WAN = {"head.modulation": (512,), "head.head.weight": (512, 512),
       "patch_embedding.weight": (5120, 16, 1, 2, 2)}
KREA2 = {"txtfusion.projector.weight": (4096, 2048)}


def main() -> int:  # noqa: C901
    tmp = Path(tempfile.mkdtemp(prefix="fp_"))

    # ① 架构表
    names = [a.name for a in fp.ARCHITECTURES]
    check("表: 架构名唯一", len(names) == len(set(names)))
    check("表: 每条都有非空签名",
          all(a.signatures and all(sig for sig in a.signatures) for a in fp.ARCHITECTURES))
    check("表: kind 落在已知集合",
          {a.kind for a in fp.ARCHITECTURES} <= {"video", "image", "3d", "audio", "detect", "depth"})
    check("表: 视频架构含 wan/h3/seedvr2 三条主线",
          {"wan", "minimax_h3", "seedvr2"} <= set(fp.video_architectures()))
    check("表: 同组签名内 key 不重复",
          all(len(set(sig)) == len(sig) for a in fp.ARCHITECTURES for sig in a.signatures))

    # ② safetensors 读取
    good = write_safetensors(tmp / "wan_t2v.safetensors", WAN, {"format": "pt"})
    header = fp.read_safetensors_header(good)
    check("safetensors: 读出 5 维 patch_embedding 形状",
          header["tensors"]["patch_embedding.weight"]["shape"] == [5120, 16, 1, 2, 2])
    check("safetensors: 读出 __metadata__", header["metadata"].get("format") == "pt")
    check("safetensors: 报出字节数", header["bytes"] == os.path.getsize(good))

    (tmp / "tiny.safetensors").write_bytes(b"\x01\x02")
    ok, why = raises(lambda: fp.read_safetensors_header(str(tmp / "tiny.safetensors")), "太小")
    check("safetensors: 太小的文件报错", ok, why)

    (tmp / "huge.safetensors").write_bytes(struct.pack("<Q", 10 ** 12) + b"{}")
    ok, why = raises(lambda: fp.read_safetensors_header(str(tmp / "huge.safetensors")), "异常")
    check("safetensors: 头长度异常报错（不吃 10^12 字节）", ok, why)

    (tmp / "trunc.safetensors").write_bytes(struct.pack("<Q", 500) + b'{"a": 1}')
    ok, why = raises(lambda: fp.read_safetensors_header(str(tmp / "trunc.safetensors")), "截断")
    check("safetensors: 头超出文件长度报错", ok, why)

    payload = b'{"a": '
    (tmp / "badjson.safetensors").write_bytes(struct.pack("<Q", len(payload)) + payload)
    ok, why = raises(lambda: fp.read_safetensors_header(str(tmp / "badjson.safetensors")), "JSON")
    check("safetensors: 非法 JSON 报错", ok, why)

    (tmp / "array.safetensors").write_bytes(struct.pack("<Q", 6) + b"[1, 2]")
    ok, why = raises(lambda: fp.read_safetensors_header(str(tmp / "array.safetensors")), "对象")
    check("safetensors: 头不是对象报错", ok, why)

    ok, why = raises(lambda: fp.read_safetensors_header(str(tmp / "nope.safetensors")), "读不到")
    check("safetensors: 文件不存在报错", ok, why)

    # ③ 架构判定
    got = fp.detect_architecture(WAN)
    check("判定: wan 三键 -> wan/video", got["architecture"] == "wan" and got["kind"] == "video")
    check("判定: in=16 -> t2v", got["subtype"] == "t2v", got["subtype"])
    check("判定: 证据含 3 条 key", len(got["evidence"]) == 3, got["evidence"])

    i2v = dict(WAN)
    i2v["patch_embedding.weight"] = (5120, 36, 1, 2, 2)
    check("判定: wan2.2 i2v 无 img_emb 也判 i2v（in=36，真机踩坑）",
          fp.detect_architecture(i2v)["subtype"] == "i2v")

    odd = dict(WAN)
    odd["patch_embedding.weight"] = (5120, 64, 1, 2, 2)
    check("判定: in 既非 16 也非 36 -> unknown（不硬猜）",
          fp.detect_architecture(odd)["subtype"] == "unknown")

    flf = dict(i2v)
    flf["img_emb.proj.0.bias"] = (1024,)
    flf["img_emb.emb_pos"] = (1, 514, 1024)
    check("判定: img_emb+emb_pos -> flf2v", fp.detect_architecture(flf)["subtype"] == "flf2v")
    plain = dict(i2v)
    plain["img_emb.proj.0.bias"] = (1024,)
    check("判定: img_emb 无 emb_pos -> i2v", fp.detect_architecture(plain)["subtype"] == "i2v")

    vace = dict(i2v)
    vace["vace_patch_embedding.weight"] = (5120, 96, 1, 2, 2)
    check("判定: vace_patch_embedding -> vace", fp.detect_architecture(vace)["subtype"] == "vace")

    cam = dict(plain)
    cam["control_adapter.conv.weight"] = (5120, 5120, 3, 3)
    check("判定: control_adapter+img_emb -> camera", fp.detect_architecture(cam)["subtype"] == "camera")
    cam22 = dict(i2v)
    cam22["control_adapter.conv.weight"] = (5120, 5120, 3, 3)
    check("判定: control_adapter 无 img_emb -> camera_2.2",
          fp.detect_architecture(cam22)["subtype"] == "camera_2.2")

    anim = dict(i2v)
    anim["blocks.0.cross_attn.k_img.weight"] = (5120, 5120)
    check("判定: cross_attn.k_img -> animate（别退化 i2v）",
          fp.detect_architecture(anim)["subtype"] == "animate")

    s2v = dict(i2v)
    s2v["casual_audio_encoder.encoder.final_linear.weight"] = (5120, 1024)
    check("判定: casual_audio_encoder -> s2v", fp.detect_architecture(s2v)["subtype"] == "s2v")

    for key, subtype in (("patch_embedding_mask.weight", "scail2"),
                         ("patch_embedding_pose.weight", "scail"),
                         ("patch_embedding_global.weight", "wandancer")):
        probe = dict(i2v)
        probe[key] = (5120, 1, 1, 2, 2)
        check(f"判定: {key.split('.')[0]} -> {subtype}",
              fp.detect_architecture(probe)["subtype"] == subtype)

    h3 = fp.detect_architecture({"video_patch_proj.weight": (5120, 64),
                                 "audio_patch_proj.weight": (5120, 64)})
    check("判定: minimax_h3 -> video 且 audio=True",
          h3["architecture"] == "minimax_h3" and h3["audio"] is True)
    check("判定: h3 子类型 unknown（ref2va/fl2va key 完全一致，判不出）", h3["subtype"] == "unknown")

    seed = fp.detect_architecture({"vid_in.proj.weight": (1,), "vid_out.proj.weight": (1,),
                                   "emb_in.proj_out.weight": (1,)})
    check("判定: seedvr2 -> video/restore", seed["architecture"] == "seedvr2"
          and seed["subtype"] == "restore")

    check("判定: flux 旧命名 img_mod 认得",
          fp.detect_architecture({"double_blocks.0.img_mod.lin.weight": (1,),
                                  "img_in.weight": (1,)})["architecture"] == "flux")
    check("判定: flux 新命名 double_stream_modulation 认得",
          fp.detect_architecture({"double_stream_modulation_img.lin.weight": (1,),
                                  "img_in.weight": (1,)})["architecture"] == "flux")
    check("判定: 带 cond_type_embedding 优先 flux2",
          fp.detect_architecture({"double_stream_modulation_img.lin.weight": (1,),
                                  "img_in.weight": (1,),
                                  "cond_type_embedding.weight": (1,)})["architecture"] == "flux2")
    check("判定: qwen_image 新命名 txt_in.text_norm 认得",
          fp.detect_architecture({"img_in.weight": (1,),
                                  "txt_in.text_norm.weight": (1,)})["architecture"] == "qwen_image")
    check("判定: qwen_image 旧命名 txt_norm 认得",
          fp.detect_architecture({"img_in.weight": (1,),
                                  "txt_norm.weight": (1,)})["architecture"] == "qwen_image")

    pre = fp.detect_architecture({f"model.diffusion_model.{k}": v for k, v in WAN.items()})
    check("前缀: model.diffusion_model. 被剥掉且证据仍带原文",
          pre["architecture"] == "wan" and pre["subtype"] == "t2v"
          and all(e.startswith("model.diffusion_model.") for e in pre["evidence"]), pre)
    check("前缀: 三键分落两个前缀 -> unknown（不许拼架构）",
          fp.detect_architecture({"a.head.modulation": (1,), "a.head.head.weight": (1,),
                                  "b.patch_embedding.weight": (5120, 16, 1, 2, 2)})
          ["architecture"] == "unknown")

    empty = fp.detect_architecture({})
    check("判定: 空 state_dict -> unknown 且说明原因",
          empty["architecture"] == "unknown" and "空" in empty["reason"], empty)
    stranger = fp.detect_architecture({"some.random": (1,)})
    check("判定: 不认识 -> unknown/confidence=none（不按名字兜底）",
          stranger["architecture"] == "unknown" and stranger["confidence"] == "none")

    # ④ GGUF 读取
    gguf = write_gguf(tmp / "wan_t2v.gguf",
                      [kv_str("general.architecture", "wan"),
                       kv_u32("wan.block_count", 30),
                       kv_u32("wan.context_length", 4096),
                       kv_bool("clip.has_vision_encoder", True),
                       kv_u32("wan.pooling_type", 2),
                       kv_str("tokenizer.chat_template", "{{ tools }}")],
                      WAN)
    gg = fp.read_gguf_header(gguf)
    check("gguf: version/tensorCount/kv 都读出",
          gg["version"] == 3 and gg["tensorCount"] == 3
          and gg["metadata"]["general.architecture"] == "wan")
    check("gguf: 默认顺带读 tensor info（含形状）",
          len(gg["tensors"]) == 3
          and gg["tensors"]["patch_embedding.weight"]["shape"] == [5120, 16, 1, 2, 2])
    check("gguf: metadataEnd 落在文件内", 0 < gg["metadataEnd"] < os.path.getsize(gguf))
    check("gguf: with_tensors=False 时不读 tensor info",
          fp.read_gguf_header(gguf, with_tensors=False)["tensors"] == {})

    big = fp.read_gguf_header(write_gguf(tmp / "big.gguf",
                                         [kv_arr_str("tokenizer.tokens",
                                                     [f"t{i}" for i in range(100)])], {}))
    kept = big["metadata"]["tokenizer.tokens"]
    check("gguf: 超长字符串数组只留 64 条 + 记总数（词表十几万条）",
          isinstance(kept, dict) and kept["count"] == 100 and len(kept["head"]) == 64, kept)
    check("gguf: 数值数组只记个数",
          fp.read_gguf_header(write_gguf(tmp / "num.gguf", [kv_arr_u32("x.values", [1, 2, 3])],
                                         {}))["metadata"]["x.values"] == {"count": 3})

    (tmp / "nomagic.gguf").write_bytes(b"NOPE" + b"\x00" * 32)
    ok, why = raises(lambda: fp.read_gguf_header(str(tmp / "nomagic.gguf")), "magic")
    check("gguf: magic 不对报错", ok, why)

    (tmp / "old.gguf").write_bytes(b"GGUF" + struct.pack("<I", 1) + struct.pack("<QQ", 0, 0))
    ok, why = raises(lambda: fp.read_gguf_header(str(tmp / "old.gguf")), "版本")
    check("gguf: 版本 1 不支持（报错，不猜着读）", ok, why)

    (tmp / "many.gguf").write_bytes(b"GGUF" + struct.pack("<I", 3)
                                    + struct.pack("<QQ", 0, 100_001))
    ok, why = raises(lambda: fp.read_gguf_header(str(tmp / "many.gguf")), "kv")
    check("gguf: kv 数量超限报错", ok, why)

    broken = fp.read_gguf_header(write_gguf(tmp / "broken.gguf",
                                            [kv_str("general.architecture", "wan")],
                                            {"only_one": (8,)}, declared_count=5))
    check("gguf: tensor info 被截断不抛（退回空，kv 结论仍可用）",
          broken["tensors"] == {} and broken["metadata"]["general.architecture"] == "wan")

    # ⑤ fingerprint_file（分派 + 自报标签 vs 张量名）
    safe = fp.fingerprint_file(good)
    check("文件: safetensors -> wan/t2v/exact",
          safe["architecture"] == "wan" and safe["subtype"] == "t2v"
          and safe["confidence"] == "exact" and not safe["error"], safe)
    check("文件: 报出 format/tensorCount/size",
          safe["format"] == "safetensors" and safe["tensorCount"] == 3
          and safe["size"] == os.path.getsize(good))

    mislabel = write_gguf(tmp / "krea2_mislabeled.gguf",
                          [kv_str("general.architecture", "qwen_image")], KREA2)
    mis = fp.fingerprint_file(mislabel)
    check("文件: 张量名优先于自报标签（真机 krea2 标成 qwen_image 的复现）",
          mis["architecture"] == "krea2" and mis["kind"] == "image", mis)
    check("文件: 矛盾仍如实保留 declaredArchitecture",
          mis.get("declaredArchitecture") == "qwen_image", mis.get("declaredArchitecture"))

    declared = fp.fingerprint_file(write_gguf(tmp / "seedvr.gguf",
                                              [kv_str("general.architecture", "seedvr")],
                                              {"mystery.weight": (8,)}))
    check("文件: 张量认不出时回落自报标签 seedvr -> video/restore",
          declared["architecture"] == "seedvr" and declared["kind"] == "video"
          and declared["subtype"] == "restore", declared)
    check("文件: 回落自报标签时 confidence=declared（自报≠核实）",
          declared["confidence"] == "declared")

    llm = fp.fingerprint_file(write_gguf(tmp / "qwen3-4b.gguf",
                                         [kv_str("general.architecture", "qwen3"),
                                          kv_bool("clip.has_vision_encoder", True)], {}))
    check("文件: 文本 LLM -> kind=unknown（本条只管媒体架构）",
          llm["kind"] == "unknown" and llm["declaredArchitecture"] == "qwen3", llm)
    check("文件: LLM 的 vision 能力从 kv 读出", "vision" in llm["capabilities"],
          llm["capabilities"])
    check("文件: 不支持的扩展名 -> error 不抛",
          fp.fingerprint_file(str(tmp / "x.pt"))["error"] != "")
    check("文件: 不存在的文件 -> error 不抛",
          fp.fingerprint_file(str(tmp / "missing.safetensors"))["error"] != "")

    # ⑥ summarize
    summary = fp.summarize([safe, mis, declared, {"error": "坏的"}, {"error": "也坏的"}])
    check("聚合: total 含坏文件", summary["total"] == 5, summary)
    check("聚合: 坏文件单独计 unreadable，不混进 architectures.unknown",
          summary["unreadable"] == 2 and "unknown" not in summary["architectures"], summary)
    check("聚合: video 只数视频", summary["video"] == 2, summary["video"])
    check("聚合: videoSubtypes 里 t2v/restore 各一条",
          summary["videoSubtypes"].get("t2v") == 1
          and summary["videoSubtypes"].get("restore") == 1, summary["videoSubtypes"])
    check("聚合: kinds 按模态分布",
          summary["kinds"].get("image") == 1 and summary["kinds"].get("video") == 2,
          summary["kinds"])

    # ⑦ 接入扫描
    check("接入: .pt 无结构头 -> None", ls.weight_fingerprint(str(tmp / "x.pt")) is None)
    check("接入: 同一文件第二次读走缓存（同一对象）",
          ls.weight_fingerprint(good) is ls.weight_fingerprint(good))

    hit_text = {"kind": "text", "runtime": "ollama", "confidence": "high",
                "role": "standalone", "matchedBy": ["gguf-llm"]}
    fixed, overridden = ls._reconcile_kind(hit_text, mis)
    check("接入: 名字判 text、权重判 image -> 以权重为准",
          fixed["kind"] == "image" and overridden)
    check("接入: 改写后 runtime 固定 comfyui（不注册假 H3/Ollama 服务）",
          fixed["runtime"] == "comfyui")
    check("接入: 改写留痕 matchedBy 追加 weight-fingerprint",
          fixed["matchedBy"][-1] == "weight-fingerprint", fixed["matchedBy"])

    kept, ov = ls._reconcile_kind(hit_text, declared)
    check("接入: 只有自报标签（declared）不够硬 -> 不改写", kept["kind"] == "text" and not ov)

    agree_hit = {"kind": "video", "runtime": "h3", "confidence": "high",
                 "role": "standalone", "matchedBy": ["h3-dit"]}
    agree, ov2 = ls._reconcile_kind(agree_hit, safe)
    check("接入: 名字与权重一致 -> 不动", agree["matchedBy"] == ["h3-dit"] and not ov2)

    kept3d, ov3 = ls._reconcile_kind(
        {"kind": "image", "runtime": "comfyui", "confidence": "high",
         "role": "standalone", "matchedBy": ["default-weights"]},
        {"architecture": "DepthAnything3", "kind": "depth", "confidence": "single-key"})
    check("接入: 扫描只认五种 kind -> depth 不改分类（避免 _KIND_ORDER 崩）",
          kept3d["kind"] == "image" and not ov3)

    scan_dir = tmp / "scan" / "diffusion_models"
    scan_dir.mkdir(parents=True, exist_ok=True)
    disguised = write_gguf(scan_dir / "qwen3-4b.gguf",
                           [kv_str("general.architecture", "qwen_image")], KREA2)
    model = ls._make_model(disguised, "qwen3-4b.gguf", "diffusion_models", ".gguf",
                           os.path.getsize(disguised), None)
    check("端到端: 名叫 qwen3-4b.gguf 的 Krea2 权重被纠正成 image",
          model is not None and model["kind"] == "image", model)
    check("端到端: 结果带 weight 明细与 weightVerified",
          model is not None and model["weight"]["architecture"] == "krea2"
          and model["weightVerified"] is True, (model or {}).get("weight"))
    check("端到端: kinds=image 能筛到它（按改写后的 kind 过滤）",
          ls._make_model(disguised, "qwen3-4b.gguf", "diffusion_models", ".gguf",
                         os.path.getsize(disguised), {"image"}) is not None)
    check("端到端: kinds=text 筛不到它（名字规则已失效）",
          ls._make_model(disguised, "qwen3-4b.gguf", "diffusion_models", ".gguf",
                         os.path.getsize(disguised), {"text"}) is None)

    scanned = ls.scan_local_models({"roots": [str(tmp / "scan")], "maxDepth": 3})
    check("扫描: weightSummary 统计到这份权重",
          scanned["weightSummary"]["total"] >= 1
          and scanned["weightSummary"]["architectures"].get("krea2") == 1,
          scanned["weightSummary"])
    check("扫描: 顶层 byKind 已是纠正后的 image", scanned["byKind"]["image"] >= 1,
          scanned["byKind"])

    # ⑦bis 量化账（文件名说 fp8/Q4_K_M，这里才是真相）
    plain16 = write_safetensors(tmp / "quant_f16.safetensors", {"a": (10,), "b": (3, 4)})
    q16 = fp.fingerprint_file(plain16)["quant"]
    check("量化: F16 按 元素数×2 算", q16["weightsBytes"] == (10 + 12) * 2, q16)
    check("量化: 单一 dtype -> scheme=F16 且 exact",
          q16["scheme"] == "F16" and q16["exact"] is True, q16)

    mixed = write_safetensors(tmp / "quant_mixed.safetensors",
                              {"a": (100,), "b": (100,), "c": (100,)},
                              dtypes={"a": "F8_E4M3", "b": "F8_E4M3", "c": "BF16"})
    qm = fp.fingerprint_file(mixed)["quant"]
    check("量化: 混合 dtype -> scheme=mixed，比例**按字节**（100+100 B vs 200 B）",
          qm["scheme"] == "mixed" and qm["types"] == {"F8_E4M3": 0.5, "BF16": 0.5}
          and qm["weightsBytes"] == 400, qm)

    odd_dtype = write_safetensors(tmp / "quant_f4.safetensors", {"a": (8,), "b": (8,)},
                                 dtypes={"a": "F4_E2M1"})
    q4 = fp.fingerprint_file(odd_dtype)["quant"]
    check("量化: dtype 不在已知表 -> exact=False、记下算不出的张量数（不许拿近似值冒充）",
          q4["exact"] is False and q4["unknownTensors"] == 1 and q4["weightsBytes"] == 16, q4)

    gq = write_gguf(tmp / "quant_q4k.gguf", [kv_str("general.architecture", "wan")],
                    {"blk.weight": (512, 256), "norm.weight": (256,)},
                    types={"blk.weight": 12, "norm.weight": 0})
    qg = fp.fingerprint_file(gq)["quant"]
    check("量化: GGUF 按块算 —— Q4_K 是 256 元素 144 字节（不是 0.5 B/元素）",
          qg["weightsBytes"] == 512 * 144 + 256 * 4, qg)
    check("量化: Q4_K 占 98.6% -> scheme=Q4_K（不是 mixed）", qg["scheme"] == "Q4_K", qg)

    # ⑦ter 完整性对账（头里自带的数据区坐标 vs 文件大小）
    exact = fp.fingerprint_file(plain16)["integrity"]
    check("完整性: 正常文件 -> exact、delta=0",
          exact["status"] == "exact" and exact["ok"] is True and exact["delta"] == 0, exact)

    short = fp.fingerprint_file(write_safetensors(tmp / "short.safetensors", {"a": (100,)},
                                                 truncate=8))["integrity"]
    check("完整性: 数据区少 8 字节 -> truncated/ok=False（拷贝中断的典型样子）",
          short["status"] == "truncated" and short["ok"] is False and short["delta"] == -8, short)

    padded = fp.fingerprint_file(write_safetensors(tmp / "padded.safetensors", {"a": (100,)},
                                                  extra=1024))["integrity"]
    check("完整性: 尾部多 1KB -> padded（对齐量级，仍可加载）",
          padded["status"] == "padded" and padded["ok"] is True, padded)

    trailing = fp.fingerprint_file(write_safetensors(tmp / "trailing.safetensors", {"a": (100,)},
                                                    extra=200_000))["integrity"]
    check("完整性: 尾部多 200KB -> trailing/ok=False（不是对齐量级）",
          trailing["status"] == "trailing" and trailing["ok"] is False, trailing)

    gexact = fp.fingerprint_file(gq)["integrity"]
    check("完整性: GGUF 按 offset+块字节对账 -> exact",
          gexact["status"] == "exact" and gexact["declaredBytes"] == 512 * 144 + 1024, gexact)
    gshort = fp.fingerprint_file(write_gguf(tmp / "gshort.gguf", [], {"blk.weight": (256, 256)},
                                           types={"blk.weight": 12}, truncate=64))["integrity"]
    check("完整性: GGUF 数据区少 64 字节 -> truncated",
          gshort["status"] == "truncated" and gshort["delta"] == -64, gshort)
    gnov = fp.fingerprint_file(write_gguf(tmp / "gnov.gguf", [], {"a": (32,)},
                                          declared_count=3))["integrity"]
    check("完整性: tensor info 读不全 -> unknown（不许谎报 ok）",
          gnov["status"] == "unknown" and gnov["declaredBytes"] is None, gnov)

    # ⑦quater 文件名自称 vs 本体（真机素材：*_Q8_K_M.gguf 实为 Q8_0）
    def audit(path: str) -> dict[str, str]:
        return {c["claim"]: c["verdict"] for c in fp.fingerprint_file(path)["nameAudit"]}

    check("名字: *_fp8*.safetensors 真为 F8_E4M3 -> verified",
          audit(write_safetensors(tmp / "x_fp8_e4m3.safetensors", {"a": (8,)},
                                  dtypes={"a": "F8_E4M3"})) == {"fp8": "verified"})
    check("名字: *_fp8*.safetensors 实为 F16 -> refuted（文件名撒谎）",
          audit(write_safetensors(tmp / "y_fp8_liar.safetensors", {"a": (8,)}))
          == {"fp8": "refuted"})
    check("名字: *_int4_convrot*.safetensors -> 只报 int4=unverifiable（convrot 是技法名、不是位宽自称）",
          audit(write_safetensors(tmp / "z_int4_convrot.safetensors", {"a": (8,)},
                                  dtypes={"a": "U8"})) == {"int4": "unverifiable"})
    check("名字: *-Q4_K_M.gguf 真为 Q4_K -> verified",
          audit(write_gguf(tmp / "ok-Q4_K_M.gguf", [], {"a": (256,)},
                           types={"a": 12})) == {"q4_k_m": "verified"})
    check("名字: *-Q8_K_M.gguf 实为 Q8_0 -> refuted（真机 seedvr2_7b 就是这种）",
          audit(write_gguf(tmp / "bad-Q8_K_M.gguf", [], {"a": (32,)},
                           types={"a": 8})) == {"q8_k": "refuted"})
    check("名字: 名字里没有量化自称 -> 不产出条目",
          fp.fingerprint_file(plain16)["nameAudit"] == [])

    # ⑦quinquies 聚合与扫描接线
    summary_q = fp.summarize([fp.fingerprint_file(p) for p in
                              (plain16, odd_dtype, gq, write_safetensors(
                                  tmp / "broken2.safetensors", {"a": (64,)}, truncate=16))])
    check("聚合: broken 只数截断/多余的，不算 dtype 认不出的",
          summary_q["broken"] == 1, summary_q)
    check("聚合: quantizations 按 scheme 计数（dtype 认不全的那份仍按dominant记 F16）",
          summary_q["quantizations"].get("F16") == 3
          and summary_q["quantizations"].get("Q4_K") == 1, summary_q["quantizations"])
    check("聚合: weightsBytes 是各份之和（含算不精确的那份的已知部分）",
          summary_q["weightsBytes"] == 44 + 16 + 512 * 144 + 1024 + 128, summary_q["weightsBytes"])

    broken_dir = tmp / "broken_scan"
    broken_dir.mkdir(parents=True, exist_ok=True)
    broken_path = write_safetensors(broken_dir / "half_downloaded.safetensors",
                                    {"a": (256,)}, truncate=64)
    broken_model = ls._make_model(broken_path, "half_downloaded.safetensors",
                                  "diffusion_models", ".safetensors",
                                  os.path.getsize(broken_path), None)
    check("扫描: 数据区不足的权重被标 weightBroken=True",
          broken_model is not None and broken_model["weightBroken"] is True,
          (broken_model or {}).get("weight"))
    check("扫描: 正常权重 weightBroken=False",
          ls._make_model(plain16, "quant_f16.safetensors", "diffusion_models",
                         ".safetensors", os.path.getsize(plain16), None)["weightBroken"] is False)
    broken_scan = ls.scan_local_models({"roots": [str(broken_dir)], "maxDepth": 2})
    check("扫描: weightSummary 里 broken 计数跟着出来",
          broken_scan["weightSummary"]["broken"] == 1, broken_scan["weightSummary"])

    # ⑦sexies 参数量 / 「同一模型的多份精度副本」
    check("参数: 元素总数只由形状决定（与 dtype 无关）",
          fp.parameters({"a": {"shape": [10]}, "b": {"shape": [3, 4]}}) == 22, )
    check("参数量: 非法形状不炸、按 0 计",
          fp.parameters({"a": {"shape": ["x"]}, "b": {"shape": [2]}}) == 2)

    check("基名: 剥掉量化标记（长词先剥，不留 e4m3fn 残渣）",
          fp.base_name("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors")
          == "seedvr2_ema_7b")
    check("基名: GGUF 的 -Q4_K_M 也剥（大小写不敏感）",
          fp.base_name("seedvr2_ema_7b-Q4_K_M.gguf") == "seedvr2_ema_7b")
    check("基名: sharp 不是量化标记 -> 与 7b 是**两个**基名（真机确实两份不同权重）",
          fp.base_name("seedvr2_ema_7b_sharp_fp16.safetensors") == "seedvr2_ema_7b_sharp")

    d1 = fp.shape_digest({"w": {"shape": [4, 4]}, "b": {"shape": [4]}})
    d2 = fp.shape_digest({"b": {"shape": [4]}, "w": {"shape": [4, 4]}})
    d3 = fp.shape_digest({"w2": {"shape": [4, 4]}, "b": {"shape": [4]}})
    d4 = fp.shape_digest({"w": {"shape": [4, 5]}, "b": {"shape": [4]}})
    check("形状指纹: 张量顺序无关", d1 == d2 and len(d1) == 16)
    check("形状指纹: 改名/改形状都会变（只覆盖名字+形状）", d1 != d3 and d1 != d4)

    # 参数量自称核对：头合法、数据区故意不放（14B 元素只存在于头里）
    big14 = write_safetensors(tmp / "m_14b.safetensors", {"w": (14_000_000_000,)}, omit_data=True)
    check("名字: 自称 14b、本体 14.0B 元素 -> verified（同档，不要求相等）",
          audit(big14) == {"14b": "verified"}, )
    check("参数: 大模型的元素数照样精确报出",
          fp.fingerprint_file(big14)["parameters"] == 14_000_000_000)

    lora = write_safetensors(tmp / "lightx2v_I2V_14B_rank64.safetensors",
                             {"w": (1_400_000_000,)}, omit_data=True)
    check("名字: 14B 的 rank64 LoRA（实际 1.4B）-> 一个字都不说（判 refuted 就是假警报）",
          audit(lora) == {}, audit(lora))

    pruned = write_safetensors(tmp / "m_7b.safetensors", {"w": (5_000_000_000,)}, omit_data=True)
    check("名字: 自称 7b 实际 5.0B（比值 0.71）-> unverifiable 并写明实际值",
          audit(pruned) == {"7b": "unverifiable"})
    entries = [c for c in fp.fingerprint_file(pruned)["nameAudit"] if c["dimension"] == "parameters"]
    check("名字: 参数条目带 dimension 与实际值", entries[0]["found"] == "5.00B", entries)

    packed = write_safetensors(tmp / "m_14b_int4_convrot.safetensors",
                               {"w": (7_000_000_000,)}, omit_data=True)
    check("名字: int4 打包文件不作文参数量核对（元素数只有真实参数一半）",
          audit(packed) == {"int4": "unverifiable"})

    only16 = write_safetensors(tmp / "only_fp16.safetensors", {"w": (8,)}, dtypes={"w": "F16"})
    check("名字: fp16 里的 16 不会被当成参数自称",
          [c["dimension"] for c in fp.fingerprint_file(only16)["nameAudit"]] == ["quant"])
    check("名字: 名字没提规模就不产出参数条目", audit(plain16) == {})

    fam_dir = tmp / "fams"
    fam_dir.mkdir(parents=True, exist_ok=True)
    alpha16 = write_safetensors(fam_dir / "alpha_1b_fp16.safetensors", {"w": (1000,)}, omit_data=True)
    alpha8 = write_safetensors(fam_dir / "alpha_1b_fp8.safetensors", {"w": (1000,)},
                               dtypes={"w": "F8_E4M3"}, omit_data=True)
    beta = write_safetensors(fam_dir / "beta_1b_fp16.safetensors", {"w": (1000,)}, omit_data=True)
    alpha2b = write_safetensors(fam_dir / "alpha_2b_fp16.safetensors", {"w": (2000,)}, omit_data=True)
    fam = fp.summarize([fp.fingerprint_file(p) for p in (alpha16, alpha8, beta, alpha2b)])
    groups = fam["variantGroups"]
    check("变体: 「基名 + 形状」同时相同才成组（同形状不同基名的 beta 不并进来）",
          len(groups) == 1 and groups[0]["baseName"] == "alpha_1b"
          and groups[0]["count"] == 2, groups)
    check("变体: 组内小份排在前、并给出各份 scheme 与字节",
          [v["scheme"] for v in groups[0]["variants"]] == ["F8_E4M3", "F16"]
          and groups[0]["variants"][0]["weightsBytes"] <= groups[0]["variants"][1]["weightsBytes"])
    check("变体: 组给合计字节（F8 1000B + F16 2000B）", groups[0]["bytes"] == 3000,
          groups[0]["bytes"])
    check("去重: models = 唯一形状指纹数（alpha_1b 与 beta_1b 形状相同 -> 算同一布局）",
          fam["models"] == 2, fam["models"])

    check("接线: 模型的 weight 明细里带参数与形状指纹",
          ls._make_model(alpha16, "alpha_1b_fp16.safetensors", "diffusion_models",
                         ".safetensors", os.path.getsize(alpha16), None)["weight"]["parameters"] == 1000)

    # ⑧ 端点
    client = TestClient(app)
    base = "/api/v1/local-models/fingerprint"
    resp = client.get(base)
    check("端点: GET 不带 path -> 400", resp.status_code == 400, resp.text)
    resp = client.get(base, params={"path": str(tmp / "missing.safetensors")})
    check("端点: GET 不存在的文件 -> 400", resp.status_code == 400, resp.text)
    (tmp / "x.pt").write_bytes(b"x")
    resp = client.get(base, params={"path": str(tmp / "x.pt")})
    check("端点: GET 不支持的结构头 -> 400", resp.status_code == 400, resp.text)
    resp = client.get(base, params={"path": mislabel})
    body = resp.json() if resp.status_code == 200 else {}
    check("端点: GET 单个权重 -> 200 且给出真架构",
          resp.status_code == 200 and body.get("data", {}).get("architecture") == "krea2",
          resp.text[:200])

    resp = client.post(base, json={"paths": [mislabel, str(tmp / "missing.safetensors")]})
    data = resp.json().get("data", {}) if resp.status_code == 200 else {}
    check("端点: POST paths -> 200、逐条给结果（坏的一条不影响好的）",
          resp.status_code == 200 and len(data.get("files", [])) == 2
          and data["summary"]["total"] == 2, resp.text[:200])
    check("端点: POST paths 的 summary 数到 unreadable=1",
          data.get("summary", {}).get("unreadable") == 1, data.get("summary"))

    resp = client.post(base, json={"paths": ["p"] * 2001})
    check("端点: POST paths 超上限 -> 400", resp.status_code == 400, resp.text)
    check("端点: MAX_FINGERPRINT_PATHS 与路由声明一致（2000）",
          __import__("app.routers.local_models", fromlist=["x"]).MAX_FINGERPRINT_PATHS == 2000)

    resp = client.post(base, json={"roots": [str(tmp / "scan")], "maxDepth": 3})
    data = resp.json().get("data", {}) if resp.status_code == 200 else {}
    check("端点: POST roots -> 扫描模式返回 summary",
          resp.status_code == 200 and data.get("mode") == "scan"
          and data["summary"]["architectures"].get("krea2") == 1, resp.text[:200])
    resp = client.post(base, json={"roots": [""]})
    check("端点: POST roots 全空 -> 400", resp.status_code == 400, resp.text)

    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    code = main()
    print(f"\nSUMMARY: {TOTAL - FAILED}/{TOTAL} 通过，{FAILED} 失败")
    sys.exit(1 if FAILED else code)
