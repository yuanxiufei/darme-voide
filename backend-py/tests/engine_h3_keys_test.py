"""S24 自检：H3 检查点**键名核对器**（零依赖 ✓ torch-free ✓ 2026-09-20）。

此前「`H3FormTrunk` 模块名与参考逐字对齐 ✓」只是**代码侧声明** ✗ —— 真权重（19.53 GiB ✗）
到手之前，没有任何东西机械验证过「这份 checkpoint 的键全集 + 形状关系」能装进去 ✗。
本套验证新核对器：合成完整 H3 键表（小尺寸 ✓ / 出厂全尺寸 ✓ / curve 变体 ✓）→
`ok=True` ✓；故意弄坏（缺键 / 形状不符 / 非整数头库 / Refiner 多出 adaLN ✗）→ 当场红 ✓；
再验接线（`inventory.component_status` 对 H3 形态文件自动核对 ✓）。

结构事实来源：`reference/ComfyUI/comfy/ldm/minimax/model.py`（2026-09-20 读全 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_h3_keys_test.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import h3_keys  # noqa: E402
from app.services.engine import inventory as inv  # noqa: E402
from app.services.engine.h3_form import H3_TRUNK_DEFAULTS, video_patch_dim  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: 小尺寸自检用维度（**刻意与出厂不同** ✓ —— 核对器必须从检查点自己推尺寸 ✗ 不许硬编码 ✗）
SMALL_DIMS = {"hidden": 64, "head_dim": 16, "inner": 96, "ffn": 128, "t_dim": 32,
              "time_in": 8, "text_dim": 48, "video_dim": 24, "audio_dim": 8}


def h3_tensor_shapes(*, dims: dict[str, int] | None = None, depth: int = 3, refiner: int = 1,
                     variant: str = "standard", banks_video: int = 1, banks_audio: int = 1,
                     drop: tuple[str, ...] = (), override: dict[str, Any] | None = None,
                     extra: dict[str, Any] | None = None) -> dict[str, tuple[int, ...]]:
    """合成一份 H3 检查点的「键名 → 形状」清单 ✓（用核对器自己的期望表生成**合法**底版 ✓，
    再按需 drop / override / extra 弄坏 ✗）。"""
    d = dict(dims or SMALL_DIMS)
    table = dict(h3_keys.expected_h3_keys(d, depth=depth, refiner_layers=refiner, variant=variant))
    table["rope.inv_freq"] = (16,)
    table["final_layer.video_out.weight"] = (banks_video * d["video_dim"], d["hidden"])
    table["final_layer.video_out.bias"] = (banks_video * d["video_dim"],)
    table["final_layer.audio_out.weight"] = (banks_audio * d["audio_dim"], d["hidden"])
    table["final_layer.audio_out.bias"] = (banks_audio * d["audio_dim"],)
    if variant == "curve":
        table["adaln_t_table"] = (6, d["t_dim"])
    for key in drop:
        table.pop(key, None)
    if override:
        table.update(override)
    if extra:
        table.update(extra)
    return table


def default_h3_dims() -> dict[str, int]:
    """出厂尺寸 → 核对器维度表（与 `h3_keys._default_dims` 同一来源 `H3_TRUNK_DEFAULTS` ✓）。"""
    d = H3_TRUNK_DEFAULTS
    return {"hidden": int(d["hidden"]), "head_dim": int(d["head_dim"]),
            "inner": int(d["heads"]) * int(d["head_dim"]), "ffn": int(d["ffn"]),
            "t_dim": int(d["time_dim"]), "time_in": int(d["time_input_dim"]),
            "text_dim": int(d["text_dim"]),
            "video_dim": video_patch_dim(int(d["latents_dim"]), tuple(d["patch_size"])),
            "audio_dim": int(d["audio_latents_dim"])}


# ══════════════════════════════════════════════════════════════════════════
# ① 期望键表的结构事实（与参考 model.py 逐条对齐 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_expected_table() -> None:
    table = h3_keys.expected_h3_keys(default_h3_dims(), depth=50, refiner_layers=2)
    check("① 出厂全尺寸期望键数 = 19 固定 + 50×10 每层 + 2×8 refiner = 535 ✓",
          len(table) == 535, len(table))
    check("② DiT 每层 10 键含 18 路 adaLN（6×hidden×3 ✓）；Refiner 每层 8 键**没有 adaln** ✗ ✓",
          table["blocks.0.adaln_proj.linear.weight"] == (6 * 5376 * 3, 2688)
          and table["blocks.0.attn.qkv_proj.weight"] == (3 * 7168, 5376)
          and "token_refiner.blocks.0.adaln_proj.linear.weight" not in table,
          {k: v for k, v in table.items() if k.startswith("blocks.0.")})
    check("③ final_layer：2 路 adaLN（2×hidden ✓）+ 收尾 norm ✓",
          table["final_layer.adaln_proj.linear.weight"] == (2 * 5376, 2688)
          and table["final_layer.norm.weight"] == (5376,)
          and table["token_refiner.final_norm.weight"] == (5376,),
          {k: v for k, v in table.items() if "final_layer" in k or "final_norm" in k})
    check("④ curve 变体：`adaln_t_table` 替换 `time_embedder.*`（互斥 ✓）",
          "adaln_t_table" in h3_keys.expected_h3_keys(SMALL_DIMS, depth=1, refiner_layers=1,
                                                       variant="curve")
          and "time_embedder.proj_in.weight" not in h3_keys.expected_h3_keys(
              SMALL_DIMS, depth=1, refiner_layers=1, variant="curve"), None)


# ══════════════════════════════════════════════════════════════════════════
# ② 合法检查点 → ok ✓（小尺寸 / 出厂尺寸 / PDD / curve）
# ══════════════════════════════════════════════════════════════════════════
def case_valid() -> None:
    audit = h3_keys.audit_h3_checkpoint(h3_tensor_shapes())
    check("⑤ 小尺寸完整 H3 ⇒ ok ✓（尺寸从检查点自己推 ✓ 不依赖出厂值 ✗）",
          audit.ok and audit.depth == 3 and audit.refiner_layers == 1
          and audit.dims["hidden"] == 64 and audit.variant == "standard",
          (audit.ok, audit.depth, audit.dims, audit.problems, audit.missing, audit.shape_mismatch))
    check("⑥ 缺键 0 / 多键 0 / 形状不符 0 ✓",
          not audit.missing and not audit.unexpected and not audit.shape_mismatch
          and audit.expected_count == audit.present_count,
          (audit.expected_count, audit.present_count, audit.missing, audit.unexpected))

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(dims=default_h3_dims(), depth=50, refiner=2))
    check("⑦ 出厂全尺寸（50 层 / 2 refiner / 5376 维）⇒ ok 且 isDefaultSized ✓",
          audit.ok and audit.is_default_sized and audit.expected_count == 535,
          (audit.ok, audit.is_default_sized, audit.expected_count, audit.problems))

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(dims=default_h3_dims(), depth=50, refiner=2, banks_video=4))
    check("⑧ PDD 头库从权重形状推断（video_out 行数 = 4×96 ⇒ banks=4 ✓ 且仍 ok ✓）",
          audit.ok and audit.head_banks_video == 4, (audit.head_banks_video, audit.problems))

    audit = h3_keys.audit_h3_checkpoint(h3_tensor_shapes(variant="curve"))
    check("⑨ curve 变体（adaln_t_table 替换 time_embedder ✓）⇒ ok 且 variant=curve ✓",
          audit.ok and audit.variant == "curve", (audit.variant, audit.problems))


# ══════════════════════════════════════════════════════════════════════════
# ③ 反向证明（故意坏的 ✓ —— 检测不到才是 bug ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_broken() -> None:
    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(drop=("blocks.1.mlp.fc2.weight",)))
    check("⑩ 缺一个键 ⇒ missing 点名 + ok=False ✓",
          not audit.ok and audit.missing == ["blocks.1.mlp.fc2.weight"],
          (audit.missing, audit.problems))

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(override={"blocks.0.attn.qkv_proj.weight": (999, 64)}))
    check("⑪ 形状不符（qkv 行数 ≠ 3×out 列数 ✗）⇒ shapeMismatch 点名 ✓",
          not audit.ok and any(item["key"] == "blocks.0.attn.qkv_proj.weight"
                               for item in audit.shape_mismatch),
          (audit.shape_mismatch, audit.problems))

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(extra={"token_refiner.blocks.0.adaln_proj.linear.weight": (1152, 32)}))
    check("⑫ Refiner 多出 adaLN（**近似实现的老坑** ✗ 参考实现 refiner 无 adaLN ✓）⇒ unexpected ✓",
          "token_refiner.blocks.0.adaln_proj.linear.weight" in audit.unexpected,
          (audit.unexpected, audit.ok))

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(extra={"blocks.0.attn.to_gate_compress.weight": (96, 64)}))
    check("⑬ 可选的 VSA 门（to_gate_compress ✓）不算 unexpected ✓",
          audit.ok and not audit.unexpected,
          (audit.unexpected, audit.problems))

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(override={"final_layer.video_out.weight": (100, 64)}))
    check("⑭ 头库行数不是单头宽度整数倍 ⇒ 报错（不静默回落 1 ✗）✓",
          not audit.ok and any("整数倍" in p for p in audit.problems), audit.problems)

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(drop=("blocks.0.attn.q_norm.weight",)))
    check("⑮ 推导源缺失 ⇒ 点名回落出厂值 + 缺键照报 ✓",
          not audit.ok and not audit.dims_from_checkpoint
          and any("尺寸推导缺源" in p for p in audit.problems)
          and "blocks.0.attn.q_norm.weight" in audit.missing,
          (audit.problems, audit.missing))

    audit = h3_keys.audit_h3_checkpoint({})
    check("⑯ 空清单 ⇒ 结构问题点名（不是静默通过 ✗）",
          not audit.ok and len(audit.problems) >= 3, audit.problems)

    audit = h3_keys.audit_h3_checkpoint(
        h3_tensor_shapes(extra={"time_embedder.proj_in.weight": (64, 8)}, variant="curve"))
    check("⑰ standard 与 curve 两种时间嵌入**同时**在 ⇒ 互斥报错 ✓",
          not audit.ok and any("互斥" in p for p in audit.problems), audit.problems)


# ══════════════════════════════════════════════════════════════════════════
# ④ 结构推导：`H3FormTrunk` 构造参数**从权重读** ✓（2026-09-20 补的机制缺口 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_infer() -> None:
    inferred = h3_keys.infer_h3_trunk_config(h3_tensor_shapes())
    want = {"hidden": 64, "layers": 3, "head_dim": 16, "heads": 6, "ffn": 128,
            "text_dim": 48, "latents_dim": 6, "audio_latents_dim": 8, "patch_size": (1, 2, 2),
            "time_input_dim": 8, "time_hidden": 64, "time_dim": 32, "inv_freq_len": 16,
            "refiner_layers": 1, "modalities": 3, "head_banks": 1}
    got = {key: inferred.config.get(key) for key in want}
    check("㉑ 推导器：尺寸/层数/头数/ffn/modalities/banks 全从权重读出 ✓（不拿出厂常量 ✗）",
          inferred.ok and got == want, (inferred.ok, got, inferred.problems))
    check("㉒ 每个字段都带**来源**：真读出来的标 `权重` ✓、不可推的标默认 ✗（不许混 ✓）",
          inferred.derived_count >= 15
          and all(str(inferred.sources[k]).startswith("权重") for k in
                  ("hidden", "layers", "heads", "head_dim", "ffn", "text_dim", "latents_dim",
                   "audio_latents_dim", "time_dim", "inv_freq_len", "refiner_layers",
                   "modalities", "head_banks"))
          and all("参考默认" in inferred.sources[k] for k in
                  ("patch_size", "norm_eps", "qk_norm_eps", "final_norm_eps")),
          (inferred.derived_count, inferred.sources))

    default_sized = h3_keys.infer_h3_trunk_config(
        h3_tensor_shapes(dims=default_h3_dims(), depth=50, refiner=2))
    check("㉓ 出厂全尺寸 ⇒ 推导结果与 `H3_TRUNK_DEFAULTS` **逐字段一致** ✓"
          "（构造参数对得上 ✓ 不是「差不多」✓）",
          default_sized.ok
          and all(_same(default_sized.config.get(key), value)
                  for key, value in H3_TRUNK_DEFAULTS.items()
                  if key in default_sized.config)
          and default_sized.config["head_banks"] == 1
          and default_sized.config["modalities"] == 3,
          {k: (default_sized.config.get(k), v) for k, v in H3_TRUNK_DEFAULTS.items()
           if k in default_sized.config and not _same(default_sized.config.get(k), v)})

    banked = h3_keys.infer_h3_trunk_config(h3_tensor_shapes(banks_video=4, banks_audio=4))
    check("㉔ PDD 头库经推导器传出 ✓（banks=4 ✓ 与 `video_out` 行数一致 ✓）",
          banked.ok and banked.config["head_banks"] == 4, banked.config.get("head_banks"))

    mixed = h3_keys.infer_h3_trunk_config(h3_tensor_shapes(banks_video=3, banks_audio=1))
    check("㉕ 两头库**不一致** ⇒ 报错 ✓（主干只有一个 `head_banks` ✓ 不静默取大值 ✗）",
          not mixed.ok and any("两处头库不一致" in p for p in mixed.problems), mixed.problems)

    curve = h3_keys.infer_h3_trunk_config(h3_tensor_shapes(variant="curve"))
    check("㉖ curve 变体：`time_input_dim` 回落被**点名** ✗，`time_hidden` 回落 `hidden` ✓",
          curve.ok and "参考默认" not in curve.sources["time_hidden"]
          and "curve" in curve.sources["time_input_dim"],
          (curve.sources.get("time_input_dim"), curve.sources.get("time_hidden")))

    bad_patch = h3_keys.infer_h3_trunk_config(h3_tensor_shapes(), patch_size=(1, 3, 3))
    check("㉗ `video_patch_proj` 列数不是 patch 乘积整数倍 ⇒ 报错 ✓（不猜 latents_dim ✗）",
          not bad_patch.ok and any("整数倍" in p for p in bad_patch.problems), bad_patch.problems)

    bad_heads = h3_keys.infer_h3_trunk_config(
        h3_tensor_shapes(override={"blocks.0.attn.q_norm.weight": (5,)}))
    check("㉘ 头数推不出（inner ÷ head_dim 非整数 ✗）⇒ 报错 + 标注回落 ✓",
          not bad_heads.ok and any("推不出头数" in p for p in bad_heads.problems), bad_heads.problems)

    # ⚠️ 真 `torch` 才验得了构造：推导出的 config **必须能被 `H3FormTrunk` 接受** ✓
    # ⚠️⚠️ **只能用缩小版** ✓（2026-09-20 踩过 ✓：拿**出厂全尺寸** config 去真建模块 ⇒
    #     50 层 × 5376 维 ≈ 80 GB 参数 ⇒ 进程**当场访问越界崩掉** ✓✗（exit 0xC0000005 ✓）。
    #     键表侧的出厂全尺寸核对已由 ⑦ 覆盖 ✓ ⇒ 这里只要「同一条链在缩小版上通」✓。
    try:
        from app.services.engine import h3_form  # noqa: PLC0415
        built = h3_form.H3FormTrunk(**inferred.config)
        state = built.state_dict()
        # 期望表用**audit 推出的维度**（命名体系不同 ✗：`expected_h3_keys` 吃 `video_dim`/`t_dim` ✓）
        expected = h3_keys.expected_h3_keys(
            inferred.audit.dims, depth=inferred.config["layers"],
            refiner_layers=inferred.config["refiner_layers"])
        missing = sorted(set(expected) - set(state))
        wrong = [(key, list(state[key].shape), list(want)) for key, want in expected.items()
                 if want is not None and key in state and tuple(state[key].shape) != tuple(want)]
        check("㉙ ⭐⭐ **推导出的 config 真能建出模块** ✓ 且参数**名字 + 形状**与期望键表逐键一致 ✓"
              "（这条才是「插上就装」的整链判据 ✓ —— 只对名字不够 ✗）",
              not missing and not wrong, (missing[:6], wrong[:3]))
    except ImportError:
        check("㉙ 装了 torch 才验构造 ✓（本机无 torch ⇒ 跳过 ✓ 不是失败 ✗）", True, "skipped")


def _same(left: Any, right: Any) -> bool:
    """tuple ↔ list 视为相同 ✓（配置里两种写法都合法 ✓）。"""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    return left == right


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 接线：inventory.component_status 对 H3 形态文件自动核对 ✓
# ══════════════════════════════════════════════════════════════════════════
def write_safetensors(path: Path, tensors: dict[str, tuple[int, ...]]) -> None:
    """写一个**真的** safetensors 文件（只带头部 + 零数据 ✓ —— inventory 只读头 ✓）。"""
    offset = 0
    header: dict[str, dict[str, Any]] = {}
    for name, shape in tensors.items():
        count = 1
        for dim in shape:
            count *= int(dim)
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + count * 4]}
        offset += count * 4
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def case_wiring(root: Path) -> None:
    model_root = root / "models" / "diffusion_models"
    model_root.mkdir(parents=True, exist_ok=True)
    good = model_root / "h3.safetensors"
    write_safetensors(good, h3_tensor_shapes())
    entry = {"key": "dit_h3", "name": "H3 DiT", "kind": "diffusion_models",
             "filename": "h3.safetensors", "file_path": "diffusion_models/h3.safetensors",
             "size_gib": good.stat().st_size / 2 ** 30, "required": True}
    status = inv.component_status(entry, root=root / "models")
    check("⑱ H3 形态文件 ⇒ 自动核对（h3Audit 进报告 ✓）且 verified=True ✓",
          status["verified"] and status.get("h3Audit", {}).get("ok")
          and status.get("h3Audit", {}).get("depth") == 3,
          (status.get("verified"), status.get("h3Audit"), status.get("problems")))

    bad = model_root / "h3_broken.safetensors"
    write_safetensors(bad, h3_tensor_shapes(drop=("blocks.2.mlp.fc1.weight",)))
    broken_status = inv.component_status(
        {**entry, "filename": "h3_broken.safetensors",
         "file_path": "diffusion_models/h3_broken.safetensors"}, root=root / "models")
    check("⑲ 缺键的 H3 文件 ⇒ verified=False + problems 点名（阻断就绪 ✓）",
          not broken_status["verified"]
          and any("blocks.2.mlp.fc1.weight" in p for p in broken_status["problems"]),
          broken_status.get("problems"))

    # 非 H3 形态文件（DiT 形态 ✗）⇒ 不做 H3 核对（不该误伤 ✓）
    other = model_root / "dit.safetensors"
    write_safetensors(other, {"attn.in_proj_weight": (64, 64), "pos_embed": (1, 64)})
    other_status = inv.component_status(
        {**entry, "filename": "dit.safetensors", "file_path": "diffusion_models/dit.safetensors"},
        root=root / "models")
    check("⑳ 非 H3 形态文件 ⇒ 不跑 H3 核对（无 h3Audit 键 ✓ 不误伤 ✓）",
          "h3Audit" not in other_status and other_status["verified"],
          (other_status.get("h3Audit"), other_status.get("problems")))


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_expected_table()
        case_valid()
        case_broken()
        case_infer()
        case_wiring(root)
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"\n      ↳ {detail}"))
    # ⚠️ 汇总行**必须是 `SUMMARY: n/m passed`** ✓ —— `run_all.py` 按这个前缀收敛项数 ✓✗
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
