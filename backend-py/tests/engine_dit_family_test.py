"""S27 自检：**判族 + 读拓扑**（`engine/dit_family.py`，2026-10-30）。

「所有模型都能跑」的第一层：盘上十来族的数学骨架是**同一套** pre-norm 残差块，
差别只在几处开关（qkv 分几块 / qk 是否归一 / MLP 是 SwiGLU 还是 GELU / 条件是 adaLN
还是 cross-attn / 单流还是双流）。所以先按**权重键名**判族、把开关读出来，再交给通用骨架。

本套盯的是**三类最容易静默错**的地方 ✓：
1. **键名归一与读形状必须用同一套名字** ✗✗（2026-10-30 抓到的真 bug：判族用剥了
   `model.diffusion_model.` 的逻辑名 ✓，读形状却拿逻辑名去查**原始**表 ✗ ⇒ 带前缀的文件
   `hidden=None` ⇒ 判成"读不到形状"、整族报不可建 ✓✗）；
2. **4bit 打包的列数是声明出来的** ✗（打包后 `(6144, 3072)` 与 per-row int8 **逐字相同** ✓；
   不读 `comfy_quant` 就永远分不出 ⇒ `hidden` 只有真值一半 ✓✗）；
3. **all-in-one 大包**（DiT 只占一部分，另有 `text_encoders.*` / `vae.*` ⇒ 按比例剥前缀
   **永远不成立** ✗ ⇒ 整份判不出族 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_dit_family_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import dit_family as fam  # noqa: E402
from app.services.engine import quant  # noqa: E402
from app.services.engine.safetensors import SafetensorsInfo, TensorEntry  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: 共享模型池（真文件那一段用；池子不在就跳过，**不把"本机没有"写成"必然缺"** ✗）
SHARED = Path("D:/Comfy-Desktop/ComfyUI-Shared/models")
DIFFUSION = SHARED / "diffusion_models"


# ══════════════════════════════════════════════════════════════════════════
# 夹具：**只造元信息**（`TensorEntry` 只有名字/ dtype / 形状 ✓，不写一个字节 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _info(pairs: dict[str, Any]) -> SafetensorsInfo:
    """``{名字: 形状}`` 或 ``{名字: (dtype, 形状)}`` ⇒ 一次 `inspect` 的结论 ✓。"""
    tensors: dict[str, TensorEntry] = {}
    for name, value in pairs.items():
        dtype, shape = value if isinstance(value, tuple) and isinstance(value[0], str) else (
            "BF16", value)
        tensors[name] = TensorEntry(name=name, dtype=dtype, shape=[int(x) for x in shape],
                                    begin=0, end=0)
    return SafetensorsInfo(path="<合成>", tensors=tensors)


def krea2_pairs(stem: str = "", *, blocks: int = 2, hidden: int = 512,
                cols: int | None = None, quanted: bool = False) -> dict[str, Any]:
    """Krea2 骨架（实测键名 ✓）：`wq/wo`、`qknorm.qnorm.scale`、`gate`、SwiGLU、
    调制是**裸参数** ``mod.lin``（键名就到 ``.lin`` ✓，没有 ``.weight`` ✗）。

    ``cols`` ⇐ **打包后的列数**（4bit 打包时只有 ``hidden`` 的一半 ✓）；
    ``quanted`` ⇐ 每层补一个 ``.comfy_quant`` + ``.weight_scale``（真文件都有 ✓）。
    """
    width = hidden if cols is None else cols
    pairs: dict[str, Any] = {}
    for n in range(blocks):
        head = f"{stem}blocks.{n}."
        pairs[head + "attn.wq.weight"] = ("I8" if quanted else "BF16", (hidden, width))
        pairs[head + "attn.wo.weight"] = ("I8" if quanted else "BF16", (hidden, width))
        pairs[head + "attn.gate.weight"] = ("I8" if quanted else "BF16", (hidden, width))
        pairs[head + "attn.qknorm.qnorm.scale"] = ("F32", (128,))
        pairs[head + "mlp.gate.weight"] = ("I8" if quanted else "BF16", (hidden, width))
        pairs[head + "mlp.up.weight"] = ("I8" if quanted else "BF16", (hidden * 4, width))
        pairs[head + "mlp.down.weight"] = ("I8" if quanted else "BF16", (hidden, hidden * 4))
        pairs[head + "prenorm.scale"] = ("BF16", (hidden,))
        # ⚠️ 裸参数：键名到此为止 ✓（写成 ``mod.lin.weight`` 就判不出族 ✗）
        pairs[head + "mod.lin"] = ("BF16", (hidden * 6,))
        if quanted:
            pairs[head + "attn.wq.comfy_quant"] = ("U8", (49,))
            pairs[head + "attn.wq.weight_scale"] = ("F32", (hidden,))
    return pairs


# ══════════════════════════════════════════════════════════════════════════
# ① 判族 + 拓扑：最朴素的 Krea2（无前缀、无量化）
# ══════════════════════════════════════════════════════════════════════════
def case_plain() -> None:
    spec = fam.detect(_info(krea2_pairs()))
    check("① 认得出 Krea2，层数按**不同 N 的个数**数（不是 max+1 ✓）",
          spec is not None and spec.key == "krea2" and spec.depth == 2, spec)
    check("① 拓扑开关全读对（RMSNorm / 分块 qkv / SwiGLU / adaLN / qknorm / attn 门控 / 单流）",
          spec is not None and (spec.norm, spec.attn, spec.mlp, spec.cond, spec.qk_norm,
                                spec.attn_gate, spec.dual_stream)
          == ("rms_scale", "separate_qkv", "swiglu_gate_up_down", "adaln_lin", True, True, False),
          spec.to_dict() if spec else None)
    check("① 形状读全：hidden / mlp_hidden / buildable",
          spec is not None and (spec.hidden, spec.mlp_hidden, spec.buildable) == (512, 2048, True),
          spec.to_dict() if spec else None)
    check("① 缺层看得出来（blocks 0,1,3 ⇒ depth=3 且不把 2 当有层）",
          (lambda s: s is not None and s.depth == 3)(fam.detect(_info({
              **krea2_pairs(blocks=2),
              "blocks.3.attn.wq.weight": (512, 512),
              "blocks.3.attn.wo.weight": (512, 512),
              "blocks.3.attn.gate.weight": (512, 512),
              "blocks.3.attn.qknorm.qnorm.scale": (128,),
              "blocks.3.mlp.gate.weight": (512, 512),
              "blocks.3.mlp.up.weight": (2048, 512),
              "blocks.3.mod.lin": (3072,),
          }))), None)


# ══════════════════════════════════════════════════════════════════════════
# ② 键名归一 ↔ 读形状必须同一套名字（**上一版真 bug 的回归判据** ✓✗）
# ══════════════════════════════════════════════════════════════════════════
def case_prefix() -> None:
    spec = fam.detect(_info(krea2_pairs("model.diffusion_model.")))
    check("② 带 `model.diffusion_model.` 前缀 ⇒ 判族照常（真文件 Krea2 int4 就是这样 ✓）",
          spec is not None and spec.key == "krea2" and spec.depth == 2, spec)
    check("② 带前缀时**形状也读得到**（上一版这里 hidden=None ✗ ⇒ 整族报不可建 ✗）",
          spec is not None and (spec.hidden, spec.mlp_hidden, spec.buildable) == (512, 2048, True),
          spec.to_dict() if spec else None)
    check("② 剥掉的前缀记在 spec 里（下游按逻辑名装权重要用 ✓）",
          spec is not None and spec.stripped_prefix == "model.diffusion_model.", spec)
    check("② 判决与无前缀版**逐字段一致**（只多一条 evidence）",
          spec is not None and _comparable(spec) == _comparable(fam.detect(_info(krea2_pairs()))),
          (spec.to_dict() if spec else None))


def _comparable(spec: fam.FamilySpec | None) -> dict[str, Any] | None:
    """去掉 evidence / strippedPrefix 后比（前缀只影响这两处 ✓）。"""
    if spec is None:
        return None
    out = spec.to_dict()
    out.pop("evidence", None)
    out.pop("strippedPrefix", None)
    return out


# ══════════════════════════════════════════════════════════════════════════
# ③ all-in-one 大包：DiT 只占一部分 ⇒ 按比例剥前缀**永远不成立**
# ══════════════════════════════════════════════════════════════════════════
def case_all_in_one() -> None:
    pairs = krea2_pairs("model.diffusion_model.", blocks=3)
    # 旁挂的 TE / VAE 让 DiT 只占 ~40%（实测 `krea2TurboNSFWAIO_v10` 是 2354 键里的一小半 ✓）
    for i in range(200):
        pairs[f"text_encoders.qwen3vl_4b.transformer.visual.blocks.{i}.attn.qkv.weight"] = (1280, 1280)
        pairs[f"vae.decoder.ups.{i}.conv.weight"] = (128, 128, 3, 3)
    spec = fam.detect(_info(pairs))
    check("③ all-in-one 大包也判得出族（DiT 前缀不占多数 ⇒ 旧实现整份 None ✗）",
          spec is not None and spec.key == "krea2" and spec.depth == 3, spec)
    check("③ 形状照样读得到（视图里的键名与查形状用的表是同一套 ✓）",
          spec is not None and spec.hidden == 512, spec)
    check("③ 旁挂部件要**说出来**（省一次外部装载；但不许混进 depth）",
          spec is not None and any("text_encoder" in note for note in spec.evidence)
          and any("VAE" in note for note in spec.evidence), spec.evidence if spec else None)


# ══════════════════════════════════════════════════════════════════════════
# ④ 4bit 打包：列数是**声明**出来的，形状分不出
# ══════════════════════════════════════════════════════════════════════════
def case_int4() -> None:
    packed = _info(krea2_pairs(quanted=True, cols=256))     # 真值 hidden=512、落盘只有 256 列
    specs = {f"blocks.{n}.attn.wq": {"format": "convrot_w4a4", "convrot_groupsize": 256,
                                     "linear_dtype": "int4"} for n in range(2)}
    spec = fam.detect(packed, specs=specs)
    check("④ 读了 `comfy_quant` ⇒ 隐藏维按**逻辑值**还原（512 而不是 256 ✓）",
          spec is not None and spec.hidden == 512, spec.to_dict() if spec else None)
    check("④ 还原这件事要在 evidence 里**说出来**（不静默 ×2 ✓）",
          spec is not None and any("4bit 打包" in note for note in spec.evidence),
          spec.evidence if spec else None)

    bare = fam.detect(packed)
    check("④ 没读说明时**不许按形状猜**：照实报列数（256 ✓）+ 在 unknown 里声明未核 ✓",
          bare is not None and bare.hidden == 256
          and any("4bit 打包未核" in note for note in bare.unknown),
          bare.to_dict() if bare else None)

    check("④ 「列数要不要 ×2」与「谁来还原」是**两个判据**（口径不许合并 ✗）",
          quant.packed_weight_of({"format": "nvfp4"}) is True
          and quant.int4_packed_of({"format": "nvfp4"}) is False
          and quant.packed_weight_of({"format": "asym_w4a8_int8"}) is True
          and quant.int4_packed_of({"format": "asym_w4a8_int8"}) is False,
          None)
    check("④ int8 / fp8 **不许**被当成打包（否则 hidden 会凭空 ×2 ✗）",
          quant.packed_weight_of({"format": "int8_tensorwise"}) is False
          and quant.packed_weight_of({"format": "float8_e4m3fn"}) is False
          and quant.packed_weight_of({}) is False and quant.packed_weight_of(None) is False,
          None)

    nv = fam.detect(_info(krea2_pairs(quanted=True, cols=256)),
                    specs={f"blocks.{n}.attn.wq": {"format": "nvfp4"} for n in range(2)})
    check("④ nvfp4 同属「每字节两个 4bit」⇒ 也按逻辑值读（漏一个 ⇒ 只有一半 ✗）",
          nv is not None and nv.hidden == 512, nv.to_dict() if nv else None)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 同族不同代：Qwen-Image 2512（MMDiT 双流）vs 2.1（单流 + SwiGLU）
# ══════════════════════════════════════════════════════════════════════════
def case_qwen_generations() -> None:
    h = 3072
    old = {
        "transformer_blocks.0.attn.to_q.weight": (h, h),
        "transformer_blocks.0.attn.add_q_proj.weight": (h, h),
        "transformer_blocks.0.attn.to_add_out.weight": (h, h),
        "transformer_blocks.0.img_mlp.net.0.proj.weight": (h * 4, h),
        "transformer_blocks.0.img_mod.1.weight": (h * 6, h),
        "txt_in.weight": (h, 3584),
    }
    spec = fam.detect(_info(old))
    check("⑤ 2512 判成 MMDiT 双流 + GELU MLP，text_dim 从 `txt_in` 读到 3584 ✓",
          spec is not None and spec.key == "qwen_image" and spec.dual_stream
          and spec.mlp == "gelu_linear" and (spec.text_dim, spec.mlp_hidden) == (3584, h * 4),
          spec.to_dict() if spec else None)

    new = {
        "transformer_blocks.0.attn.to_q.weight": (4096, 4096),
        "transformer_blocks.0.img_mlp.gate_up.weight": (4096 * 6, 4096),
        "transformer_blocks.0.img_mlp.out.weight": (4096, 4096 * 3),
        "txt_in.in_layer.weight": (4096, 4096),
        "txt_in.out_layer.weight": (4096, 4096),
        "modulation.1.weight": (4096 * 4, 4096),
    }
    spec2 = fam.detect(_info(new))
    check("⑤ 2.1 判成**另一族**（单流 + SwiGLU）—— 结构不同就不能塞进同一份前向 ✗",
          spec2 is not None and spec2.key == "qwen_image21" and not spec2.dual_stream
          and spec2.mlp == "swiglu_gate_up_down" and spec2.hidden == 4096,
          spec2.to_dict() if spec2 else None)
    check("⑤ 2.1 的文本原始宽度**读不到**（`in_layer` 是方阵 ✓）⇒ 进 unknown，不猜 ✗",
          spec2 is not None and spec2.text_dim is None
          and any("文本原始宽度" in note for note in spec2.unknown),
          spec2.unknown if spec2 else None)
    check("⑤ 两代的族名不同（`FAMILY_KEYS` 里都在册 ✓）",
          fam.detect_from_keys(old.keys()) == "qwen_image"
          and fam.detect_from_keys(new.keys()) == "qwen_image21"
          and {"qwen_image", "qwen_image21"} <= set(fam.FAMILY_KEYS), None)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 认不出 / 非 DiT / 只有键名
# ══════════════════════════════════════════════════════════════════════════
def case_edges() -> None:
    check("⑥ 认不出就返回 None（**不硬塞**最近的一族 ✗）",
          fam.detect(_info({"foo.weight": (8, 8), "bar.baz": (4,)})) is None, None)
    check("⑥ 空文件 ⇒ None（不是「认得出但形状缺失」✓）",
          fam.detect(SafetensorsInfo(path="<空>", tensors={})) is None, None)
    check("⑥ SDXL（UNet）认得出、输入通道从 4D conv 的 `shape[1]` 读（4 ✓）",
          (lambda s: s is not None and s.key == "sdxl" and s.hidden == 4)(fam.detect(_info({
              "input_blocks.0.0.weight": (320, 4, 3, 3),
              "middle_block.1.proj_in.weight": (320, 640, 1, 1),
          }))), None)
    check("⑥ 量化伴生键**不参与**判族（只有伴生键的一份 ⇒ 认不出 ✓）",
          fam.detect(_info({"blocks.0.attn.wq.comfy_quant": ("U8", (49,)),
                            "blocks.0.attn.wq.weight_scale": ("F32", (8,))})) is None, None)
    check("⑥ `readout(None)` 说人话（「认不出」而不是空白 ✓）",
          "认不出" in fam.readout(None), fam.readout(None))
    check("⑥ 只有键名（无形状）时的降级口径也认得出 Krea2",
          fam.detect_from_keys(krea2_pairs().keys()) == "krea2", None)


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 真文件（池子在就跑；同一族的不同量化版**必须**读出同一个 hidden ✓✗）
# ══════════════════════════════════════════════════════════════════════════
def case_real_files() -> None:
    pairs = (
        ("Krea2 int4 vs int8", DIFFUSION / "Krea2/krea2_turbo_int4_convrot.safetensors",
         DIFFUSION / "Krea2/krea2_turbo_int8_convrot.safetensors"),
        ("SEEDVR2 w4a8 vs int8", DIFFUSION / "SEEDVR2/seedvr2_3b_w4a8.safetensors",
         DIFFUSION / "SEEDVR2/seedvr2_3b_int8_convrot.safetensors"),
    )
    if not DIFFUSION.is_dir():
        return                                  # 池子不在 ⇒ 这一段不算 ✅/❌（不是"缺能力" ✗）
    for label, left, right in pairs:
        if not (left.exists() and right.exists()):
            continue
        a, b = fam.detect_file(left), fam.detect_file(right)
        check(f"⑦ 真文件·{label}：同族同 hidden（打包版不许读成一半 ✗）",
              a is not None and b is not None and a.key == b.key
              and a.hidden == b.hidden and a.depth == b.depth,
              (a.to_dict() if a else None, b.to_dict() if b else None))

    all_in_one = DIFFUSION / "Krea2/krea2TurboNSFWAIO_v10.safetensors"
    if all_in_one.exists():
        spec = fam.detect_file(all_in_one)
        check("⑦ 真文件·all-in-one（DiT + TE + VAE 同包）也判得出 Krea2 且形状齐全",
              spec is not None and spec.key == "krea2" and spec.depth == 28
              and spec.hidden == 6144, spec.to_dict() if spec else None)

    for name, expect in (("z_image_bf16.safetensors", "zimage"),
                         ("flux1-dev.safetensors", "flux"),
                         ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan"),
                         ("qwen_image_2.1_bf16.safetensors", "qwen_image21"),
                         ("qwen_image_2512_bf16.safetensors", "qwen_image"),
                         ("minimax_music3_dit_fp16.safetensors", "music3")):
        path = (DIFFUSION / name) if (DIFFUSION / name).exists() else (SHARED / "unet" / name)
        if not path.exists():
            continue
        spec = fam.detect_file(path)
        check(f"⑦ 真文件·{name} 判成 `{expect}` 且可建（hidden/depth 都读到了 ✓）",
              spec is not None and spec.key == expect and spec.buildable,
              spec.to_dict() if spec else None)


def main() -> int:
    case_plain()
    case_prefix()
    case_all_in_one()
    case_int4()
    case_qwen_generations()
    case_edges()
    case_real_files()
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + ("  ❌" if failed else "  ✅"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
