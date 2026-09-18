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
