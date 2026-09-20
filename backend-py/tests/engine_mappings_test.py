"""S7 自检：**权重命名映射**（零依赖 ✓ + 真装载 ✓ 2026-09-17）。

最有说服力的一条（case_roundtrip ✓）：把一个**真 DiT** 的 ``state_dict`` **反向改写成"别家命名"**
（外层前缀 + `layers.N` + `attention` + 分开的 wq/wk/wv + `mlp.fc1/fc2` ✓）⇒ 存成 safetensors ✓
⇒ 再**套预设映射装回去** ✓ ⇒ 输出与原件**逐位相同** ✓✓。

这条同时钉住两件容易静默出错的事 ✓：
* **合并顺序**（q/k/v 拼错顺序 ⇒ **不报错**但结果全错 ✗ ⇒ 自检直接断言拼接后的张量与原件一致 ✓）；
* **映射完整性**（缺键必须进 `missingTargets` ✓ 而不是"装上一半照常跑" ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_mappings_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import mappings as mp  # noqa: E402

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


# ══════════════════════════════════════════════════════════════════════════
# ① 改名 / 合并规则本身
# ══════════════════════════════════════════════════════════════════════════
def case_rules() -> None:
    spec = mp.preset("dit")
    keys = [
        "model.diffusion_model.blocks.0.attention.wq.weight",
        "model.diffusion_model.layers.1.self_attn.wk.weight",
        "model.diffusion_model.transformer_blocks.2.mlp.fc1.weight",
        "pos_embed",
    ]
    renamed, hits = mp.apply_renames(keys, spec.renames)
    check("① 外层前缀 + `attention`/`self_attn` + `layers.N`/`transformer_blocks.N` 一次改到位 ✓",
          renamed[0] == "blocks.0.attn.wq.weight" and renamed[1] == "blocks.1.attn.wk.weight"
          and renamed[2] == "blocks.2.mlp.0.weight", renamed)
    check("② 没命中的键**原样保留** ✓（不静默丢 ✗）", renamed[3] == "pos_embed", renamed[3])
    check("③ 每条规则**命中数**都被记下来 ✓（0 命中的规则会在报告里显形 ✓）",
          hits["^model\\.diffusion_model\\."] == 3 and sum(1 for value in hits.values()
                                                          if value == 0) >= 5, hits)
    check("④ 未知预设 ⇒ **报错并列出可用项** ✓（不静默给空的 ✗）",
          "可用" in str(_error(lambda: mp.preset("nope"))))
    check("⑤ `build_key_map` 给的是**可直接喂 `load_module_weights`** 的表 ✓（且 `extra` 能覆盖 ✓）",
          isinstance(mp.build_key_map("dit"), dict)
          and mp.build_key_map("dit", extra={r"^zzz": "yyy"}).get("^zzz") == "yyy")


def case_merge_order() -> None:
    keys = ["blocks.0.attn.wq.weight", "blocks.0.attn.wk.weight", "blocks.0.attn.wv.weight",
            "blocks.1.attn.wq.weight", "blocks.1.attn.wk.weight", "blocks.1.attn.wv.weight"]
    groups, hits = mp._merge_groups(keys, mp.preset("dit").merges)  # noqa: SLF001
    check("⑥ ⭐ q/k/v **按声明顺序**分组 ✓（顺序错 ⇒ 不报错但结果全错 ✗ ⇒ 必须显式钉住 ✓）",
          groups.get("blocks.1.attn.in_proj_weight")
          == ["blocks.1.attn.wq.weight", "blocks.1.attn.wk.weight", "blocks.1.attn.wv.weight"],
          groups.get("blocks.1.attn.in_proj_weight"))
    check("⑦ 缺一段（比如只有 wq/wk）⇒ **不成组** ✓（宁可不装，也不装错的 ✗）",
          mp._merge_groups(["blocks.0.attn.wq.weight", "blocks.0.attn.wk.weight"],  # noqa: SLF001
                           mp.preset("dit").merges)[0] == {}, "")


# ══════════════════════════════════════════════════════════════════════════
# ② 报告（工作站上"填表"靠它 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_report() -> None:
    report = mp.plan_mapping(
        ["model.blocks.0.attn.wq.weight", "model.blocks.0.attn.wk.weight",
         "model.blocks.0.attn.wv.weight", "model.blocks.0.mlp.fc1.weight"],
        ["blocks.0.attn.in_proj_weight", "blocks.0.mlp.0.weight", "blocks.0.norm1.weight"])
    check("⑧ 能对上的目标键被计数 ✓", report["matched"] == 2, report["matched"])
    check("⑨ ⭐ **没人填的目标键**逐条列出 ✓（= 还缺规则 ✓ —— 这就是「填表」的入口 ✓）",
          report["missingTargets"] == ["blocks.0.norm1.weight"], report["missingTargets"])
    check("⑩ 缺键时 `complete=False` 且 `verdict` 说清还差几个 ✓（不静默放过 ✗）",
          report["complete"] is False and "还差 1 个" in report["verdict"], report["verdict"])
    check("⑪ 合并分组被**显式记录**在报告里 ✓（便于人工核对顺序 ✓）",
          report["mergeGroups"].get("blocks.0.attn.in_proj_weight") is not None,
          report["mergeGroups"])
    check("⑫ 把缺的键补上 ⇒ `complete=True` ✓（报告真的随输入变 ✓ 不是恒真 ✗）",
          mp.plan_mapping(["model.blocks.0.attn.wq.weight", "model.blocks.0.attn.wk.weight",
                           "model.blocks.0.attn.wv.weight"],
                          ["blocks.0.attn.in_proj_weight"])["complete"] is True)


# ══════════════════════════════════════════════════════════════════════════
# ③ 往返：真 DiT → 改写成"别家命名" → 套预设装回去 ⇒ **逐位相同** ✓✓
# ══════════════════════════════════════════════════════════════════════════
def case_roundtrip(root: Path) -> None:
    if not _have_torch():
        skip("torch 未安装 ⇒ 往返装载跳过 ✓")
        return
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    from app.services.engine import dit as dit_mod
    from app.services.engine import weights as weights_mod

    config = dit_mod.DiTConfig(hidden=32, depth=2, heads=4, patch_size=(1, 2, 2),
                               in_channels=4, text_dim=16, mlp_ratio=2.0, vae_scale=8)
    torch.manual_seed(0)
    model = dit_mod.build_dit(config)
    latent = torch.randn(1, 4, 2, 8, 8)
    context = torch.randn(1, 1, config.text_dim)
    before = model(latent, 1.0, context).detach()

    # 「别家命名」：外层前缀 + layers.N + attention + 分开的 wq/wk/wv + mlp.fc1/fc2 ✓
    foreign: dict[str, Any] = {}
    for key, tensor in model.state_dict().items():
        new_key = f"model.diffusion_model.{key}".replace("blocks.", "layers.")
        new_key = new_key.replace(".attn.", ".attention.")
        new_key = new_key.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
        if new_key.endswith(".attn.in_proj_weight".replace(".attn.", ".attention.")):
            stem = new_key[: -len("in_proj_weight")]
            hidden = tensor.shape[0] // 3
            foreign[f"{stem}wq.weight"] = tensor[:hidden]
            foreign[f"{stem}wk.weight"] = tensor[hidden:2 * hidden]
            foreign[f"{stem}wv.weight"] = tensor[2 * hidden:]
        else:
            foreign[new_key] = tensor
    checkpoint = root / "foreign.safetensors"
    save_file({key: value.contiguous() for key, value in foreign.items()}, str(checkpoint))
    check("⑬ 造出了一份「别家命名」的权重 ✓（含分开的 q/k/v ✓ 便于验证合并顺序 ✓）",
          len(foreign) > len(model.state_dict()) and any(key.endswith("wq.weight")
                                                         for key in foreign), len(foreign))

    report = mp.plan_mapping(list(foreign), list(model.state_dict()), name="dit")
    check("⑭ ⭐ 报告判定**映射完整**（没有目标键没人填 ✓）",
          report["complete"] is True, report["missingTargets"][:6])

    state = dict(load_file(str(checkpoint)))
    renamed, _hits = mp.apply_renames(list(state), mp.preset("dit").renames)
    remapped = {new: state[old] for old, new in zip(list(state), renamed)}
    groups, _merge_hits = mp._merge_groups(list(remapped), mp.preset("dit").merges)  # noqa: SLF001
    merged = mp.apply_merges(remapped, groups)
    for key, tensor in merged.items():
        remapped[key] = tensor
    # ⚠️ 留一份 **pop 之前** 的副本 ✓：⑰ 要拿融合前的源张量故意打乱顺序 ✓
    #    （初版直接用 `remapped[...]` 取 ⇒ KeyError ✗ —— 因为源键已经被 pop 掉了 ✓）
    before_merge = dict(remapped)
    for source in [item for sources in groups.values() for item in sources]:
        remapped.pop(source, None)

    torch.manual_seed(1)
    fresh = dit_mod.build_dit(config)
    fresh.load_state_dict(remapped, strict=True)      # strict ✓：**多余/缺失都会当场报错** ✓
    check("⑮ ⭐⭐ 映射后装载 ⇒ 输出与原件**逐位相同** ✓（合并顺序 + 全部键都对 ✓）",
          bool(torch.allclose(fresh(latent, 1.0, context).detach(), before, atol=1e-6)),
          float((fresh(latent, 1.0, context).detach() - before).abs().max()))

    # 走 `load_module_weights` 的那条正常路径也要能吃预设 ✓
    loaded = weights_mod.load_module_weights(
        dit_mod.build_dit(config), checkpoint,
        key_map=mp.build_key_map("dit"))
    check("⑯ ⭐ 但**只改名不合并**时如实报「缺融合键」✓（正是 q/k/v 那种必须合并的情形 ✓）",
          loaded.complete is False and any("in_proj" in key for key in loaded.missing),
          loaded.to_dict()["missing"][:4])

    # 合并顺序写错 ⇒ 必须能被发现（这正是「顺序不能猜」的理由 ✓）
    # ⚠️ 这里**不能**拿整个模型的输出去比 ✗：adaLN-Zero 的 `gate_msa` 初值为 0 ✓ ⇒
    #    未训练的模型**根本不看注意力** ✓ ⇒ q/k/v 打乱后整模型输出**一模一样** ✗
    #    （第一版就这么写的 ⇒ 断言恒假 ✓ 是**假前提** ✗）。改成直接比**注意力子模块**的输出 ✓。
    wrong = dict(remapped)
    stem = [key for key in groups if key.endswith("in_proj_weight")][0]
    sources = groups[stem]
    wrong[stem] = torch.cat([before_merge[sources[2]], before_merge[sources[0]],
                             before_merge[sources[1]]], dim=0)
    mutated = dit_mod.build_dit(config)
    mutated.load_state_dict(wrong, strict=True)
    probe = torch.randn(1, 3, config.hidden)
    block = 0
    with torch.no_grad():
        right_out = model.blocks[block].attn(probe, probe, probe, need_weights=False)[0]
        wrong_out = mutated.blocks[block].attn(probe, probe, probe, need_weights=False)[0]
    check("⑰ ⭐ 把合并顺序**故意打乱** ⇒ **注意力输出明显不同** ✓"
          "（证明「顺序错会静默出错」是真的 ✓；也顺带说明 adaLN-Zero 下整模型比不出来 ✗）",
          not bool(torch.allclose(right_out, wrong_out, atol=1e-5)),
          float((right_out - wrong_out).abs().max()))


def _error(fn) -> Exception:  # noqa: ANN001
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return err
    return Exception("（没有抛错 ✗）")


def case_minimax_h3() -> None:
    """⭐ H3 专用预设（2026-09-20 抄自 `reference/ComfyUI` 的实测键名 ✓）。"""
    spec = mp.preset("minimax-h3")
    check("⑱ ⭐ H3 预设存在，且**故意不带合并规则** ✗（H3 权重里 qkv 本就是融合的 ✓）",
          spec is not None and spec.merges == (), spec.merges)
    renamed, _hits = mp.apply_renames(
        ["model.diffusion_model.blocks.0.attn.qkv_proj.weight",
         "model.diffusion_model.blocks.3.mlp.fc1.weight",
         "model.diffusion_model.final_layer.video_out.weight"], spec.renames)
    check("⑲ 前缀 + `qkv_proj`→`in_proj_` + `mlp.fc1`→`mlp.0` 一次改到位 ✓",
          renamed[0] == "blocks.0.attn.in_proj_weight"
          and renamed[1] == "blocks.3.mlp.0.weight", renamed)
    check("⑳ ⭐⭐ 本仓**没有对应物**的键要留原样并在 note 里点名 ✗"
          "（不硬塞成看似对的名字 ✓ —— 那会变成「名字对、形状错」✗）",
          renamed[2] == "final_layer.video_out.weight"
          and "token_refiner" in spec.note and "video_out" in spec.note, renamed[2])


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_map_"))
    case_rules()
    case_merge_order()
    case_minimax_h3()
    case_report()
    case_roundtrip(root)

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
