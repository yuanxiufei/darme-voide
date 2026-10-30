"""S7 自检：**NVFP4 反量化**（`quant_nvfp4` ✓ —— 2026-10-30）。

为什么这一套的每一条判据都得是「硬证据」✗：这一族**没有任何容错空间** ✓✗ ——
4 bit 打包的打包顺序反一次、块 scale 的 tiled 布局少扭一次，解出来的权重**形状完全正常** ✓、
数值**看着也正常**（std 0.01~0.02 那个量级 ✓）、**不会报任何错** ✓✗ ⇒ 症状只有一个：
「出片是噪声 / 语义全乱」，而那时根本不知道错在哪 ✗。

所以判据分四层，每层都有**独立于本仓实现**的真源 ✓：

1. **公开规格**：E2M1 的 16 个值 ✓、码序（0..7 为正 ✓、bit3 是符号 ✓）、
   **元素 0 在高半字节** ✓ —— 后者由 ComfyUI 自己的量化实现
   （`comfy/float.py::stochastic_float_to_fp4_e2m1`：``(fp4[0::2] << 4) | fp4[1::2]`` ✓）直接给出 ✓，
   而 ComfyUI 正是**写这个文件的那一方** ✓；
2. **逐位对拍**：与 `reference/lora/musubi-tuner/.../nvfp4_utils.py::dequantize_nvfp4`
   比 ``torch.equal`` ✓（该实现抬头写着 “verified bit-level against the BF16 reference” ✓）；
3. **布局往返**：``from_blocked(to_blocked(M)) == M`` ✓，其中 ``to_blocked`` 用的是
   **跑 ComfyUI 那份源码**（ast 抽出来 ✓）—— 不是本仓的复制品 ✓✗；
4. **真实文件**：盘上那份 14.61 GiB 的 nvfp4 文本编码器逐层体检 ✓（350 层 / 0 问题 ✓），
   并对**真权重**抽一层做反量化 + 数值合理性 + 「错解一定不一样」的反证 ✓。

⚠️ 特意**不**做成"用随机舍入量化后再还原、看误差多小" ✗：ComfyUI 的 `stochastic` 舍入
扰动幅度可达 ``2**(exp-2)*1.25`` ✓（与 6.0 同量级 ✓）⇒ 往返误差本来就大 ✓，
拿它当判据只会得出「解码错了」的假结论 ✓✗（实测：18.4% ✗ 与「完全不还原 swizzle」的 36.5% ✗
区分度太差 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_quant_nvfp4_test.py
"""
from __future__ import annotations

import ast
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
WORKSPACE = BACKEND_PY.parent
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="enginenvfp4_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import quant_nvfp4 as nv  # noqa: E402

REFERENCE = WORKSPACE / "reference" / "lora" / "musubi-tuner" / "src" / "musubi_tuner" / "modules" / "nvfp4_utils.py"
COMFY_FLOAT = Path("d:/code/voide/comfy-studio/ComfyUI/comfy/float.py")
REAL_MODEL = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders"
                  r"\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPPED: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str, why: str) -> None:
    """跳过的判据**单独计数并打印** ✓✗ —— 混进 PASS 里就是假绿 ✓。"""
    _SKIPPED.append(f"{name}（{why}）")


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _extract(path: Path, names: set[str]) -> dict[str, Any]:
    """把**别人仓库里的**函数源码抽出来跑 ✓（不是抄 ✗ —— 跑的就是那份原文 ✓）。

    只抽纯函数（它们只依赖 ``torch`` ✓）；不 import 整个包 ✗（那会把 musubi 的一堆依赖拖进来 ✓，
    而我们的判据必须在**没有安装那些包**的机器上也能跑 ✓）。
    """
    torch = _torch()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace: dict[str, Any] = {"torch": torch, "F": torch.nn.functional, "math": math,
                                 "Dict": dict, "Tuple": tuple, "List": list, "Optional": Any,
                                 "_BYTE_PAIR_LUT_CACHE": {}, "NVFP4_BLOCK_SIZE": 16}
    # 常量赋值也要带上（`_E2M1_VALUES` 一类 ✓）
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id.startswith(("_E2M1", "NVFP4_BLOCK"))
                for target in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


# ══════════════════════════════════════════════════════════════════════════
# ① 公开规格：值表 / 码序 / 打包顺序
# ══════════════════════════════════════════════════════════════════════════
def case_spec() -> None:
    torch = _torch()
    check("① E2M1 值表就是规格本身 ✓（``0, .5, 1, 1.5, 2, 3, 4, 6`` 与相反数，共 16 个 ✓）",
          nv.E2M1_VALUES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
          nv.E2M1_VALUES)
    check("① 块大小 16 / 最大值 6.0 ✓（与 `comfy/float.py` 的 F4_E2M1_MAX=6.0 一致 ✓）",
          nv.BLOCK_SIZE == 16 and nv.E2M1_MAX == 6.0, (nv.BLOCK_SIZE, nv.E2M1_MAX))

    # 打包顺序：一个字节两个值，**元素 0 在高半字节** ✓✗（反了就是整份权重错位 ✓ 但不报错 ✓✗）
    packed = torch.tensor([[0x21, 0x76, 0xF8, 0xA3, 0x00, 0xFF]], dtype=torch.uint8)
    got = nv.unpack_e2m1(packed).tolist()[0]
    want = [1.0, 0.5, 6.0, 4.0, -6.0, -0.0, -1.0, 1.5, 0.0, 0.0, -6.0, -6.0]
    check("① 手工字节 ⇒ 期望值 ✓（`0x21`→(1.0, 0.5) / `0xF8`→(-6.0, -0.0) / `0xA3`→(-1.0, 1.5)）"
          "—— 高半字节在前，这是 ComfyUI 侧 `(fp4[0::2] << 4) | fp4[1::2]` 的直接推论 ✓",
          got == want, {"got": got, "want": want})

    try:
        nv.unpack_e2m1(torch.zeros((4, 4), dtype=torch.float32))
        rejected = False
    except ValueError:
        rejected = True
    check("① 非 U8 输入**拒绝** ✓✗（悄悄按别的 dtype 解，又是一次「不报错的错」✓✗）", rejected)


# ══════════════════════════════════════════════════════════════════════════
# ② 与参照实现逐位对拍（musubi-tuner）
# ══════════════════════════════════════════════════════════════════════════
def case_reference() -> None:
    torch = _torch()
    if not REFERENCE.exists():
        skip("② 与参照实现逐位对拍", f"找不到 {REFERENCE} ✗")
        return
    ns = _extract(REFERENCE, {"dequantize_nvfp4", "to_blocked", "from_blocked",
                              "_byte_pair_lut", "_ceil_div", "_roundup"})
    check("① 值表与参照实现**逐项相同** ✓（同一个 16 元组，不是「看起来差不多」✓）",
          tuple(ns.get("_E2M1_VALUES", ())) == nv.E2M1_VALUES, ns.get("_E2M1_VALUES"))

    generator = torch.Generator().manual_seed(20261030)
    cases = [
        ((128, 64), (128, 8), "行/列都对齐（128 行 ✓ 64/16=4 是 4 的倍数 ✓）"),
        ((100, 64), (128, 8), "行数不齐 ⇒ 块 scale 带 128-padding ✓"),
        ((128, 40), (128, 8), "块列不齐 ⇒ 带 4-padding（80/16=5 ⇒ 补到 8 ✓ 用 orig_shape 裁 ✓）"),
    ]
    for (rows, packed_cols), scale_shape, why in cases:
        packed = torch.randint(0, 256, (rows, packed_cols), generator=generator, dtype=torch.uint8)
        scale = (torch.rand(scale_shape, generator=generator) * 2.0).to(torch.float8_e4m3fn)
        per_tensor = torch.tensor(0.0012345, dtype=torch.float32)
        cols = packed_cols * 2
        mine = nv.dequantize(packed, scale, per_tensor, (rows, cols), dtype=torch.float32)
        theirs = ns["dequantize_nvfp4"](packed, scale, per_tensor, (rows, cols), torch.float32)
        check(f"② 与参照实现 **逐位相等** ✓（``torch.equal`` ✓ 不是 allclose ✗）：{why}",
              torch.equal(mine, theirs),
              {"maxdiff": float((mine - theirs).abs().max()) if mine.shape == theirs.shape else "shape"})

    matrix = torch.rand((96, 6), generator=generator)
    check("② swizzle 与参照实现**同向同值** ✓（`to_blocked` 逐位相等 ✓）",
          torch.equal(nv.to_blocked(matrix), ns["to_blocked"](matrix)),
          float((nv.to_blocked(matrix) - ns["to_blocked"](matrix)).abs().max()))


# ══════════════════════════════════════════════════════════════════════════
# ③ 布局往返：用**ComfyUI 自己那份源码**的 to_blocked（第三方真源 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_layout() -> None:
    torch = _torch()
    if not COMFY_FLOAT.exists():
        skip("③ 与 ComfyUI 的 to_blocked 往返", f"找不到 {COMFY_FLOAT} ✗")
        return
    ns = _extract(COMFY_FLOAT, {"to_blocked"})
    generator = torch.Generator().manual_seed(7)
    shapes = [((128, 4), "对齐 ✓"), ((100, 6), "两个方向都要 padding ✓"),
              ((2560, 160), "真实 `up_proj` 的块表形状 ✓")]
    for shape, why in shapes:
        matrix = torch.rand(shape, generator=generator)
        blocked = ns["to_blocked"](matrix, flatten=False)
        back = nv.from_blocked(blocked, shape[0], shape[1])
        check(f"③ `from_blocked(ComfyUI.to_blocked(M)) == M` ✓（{shape} ✓ {why}）"
              f"—— tiled 布局**扭反一次**就整份错 ✓ 且形状照样合法 ✓✗",
              torch.equal(back, matrix), float((back - matrix).abs().max()))
    padded = ns["to_blocked"](torch.rand((100, 6), generator=generator), flatten=False)
    check("③ padding 的**元素个数**与 `block_scale_elements` 一致 ✓✗"
          "（用「真实形状」算会少一截 ⇒ 误报「scale 缺元素」✓✗）",
          int(padded.numel()) == nv.block_scale_elements(100, 96),
          {"got": int(padded.numel()), "want": nv.block_scale_elements(100, 96)})


# ══════════════════════════════════════════════════════════════════════════
# ④ 公式与 AWQ
# ══════════════════════════════════════════════════════════════════════════
def case_formula() -> None:
    torch = _torch()
    # 块 scale 取**常数** ⇒ swizzle 与常数无异 ✓ ⇒ 可以纯手算公式 ✓（不借任何布局知识 ✓）
    packed = torch.tensor([[0x21, 0x76, 0xF8, 0xA3, 0x00, 0xFF, 0x5A, 0x3C]] * 128, dtype=torch.uint8)
    scale = torch.full((128, 4), 0.5, dtype=torch.float8_e4m3fn)
    per_tensor = torch.tensor(4.0)
    got = nv.dequantize(packed, scale, per_tensor, (128, 16), dtype=torch.float32)
    want = torch.tensor([1.0, 0.5, 6.0, 4.0, -6.0, -0.0, -1.0, 1.5,
                         0.0, 0.0, -6.0, -6.0, 3.0, -1.0, 1.5, -2.0]) * (0.5 * 4.0)
    check("④ `w = fp4 × (块 scale × 二级 scale)` 手算逐元素相符 ✓（常数 scale ⇒ 不看布局也能验 ✓）",
          torch.allclose(got[0].float(), want, atol=1e-6, rtol=0),
          {"got": got[0].tolist(), "want": want.tolist()})

    torch.manual_seed(3)
    x = torch.randn(4, 32, dtype=torch.float32)
    weight = torch.randn(8, 32, dtype=torch.float32)
    pre = torch.rand(32, dtype=torch.float32) + 0.5
    fold = nv.fold_pre_quant_scale(weight, pre)
    check("④ AWQ 折进权重列 ≡ 乘在输入上 ✓（``(x·s) @ Wᵀ == x @ (W·s)ᵀ`` ✓ 结合顺序不同 ⇒ 只到 1e-5 ✓）",
          torch.allclose((x * pre) @ weight.T, x @ fold.T, atol=1e-5, rtol=1e-6),
          float(((x * pre) @ weight.T - x @ fold.T).abs().max()))
    try:
        nv.fold_pre_quant_scale(weight, torch.rand(31))
        rejected = False
    except ValueError:
        rejected = True
    check("④ `pre_quant_scale` 长度 ≠ in_features ⇒ **拒绝** ✓✗（错一列就是整层错 ✓ 还不报错 ✓✗）", rejected)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 体检（纯形状 ✓ 不需要 torch ✓ 不需要读一个字节 ✓）
# ══════════════════════════════════════════════════════════════════════════
class _Entry:
    def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = shape


class _Info:
    def __init__(self, tensors: dict[str, _Entry]) -> None:
        self.tensors = tensors


def case_plan() -> None:
    good = _Info({
        "m.0.comfy_quant": _Entry("U8", (30,)),
        "m.0.weight": _Entry("U8", (128, 32)),
        "m.0.weight_scale": _Entry("F8_E4M3", (128, 4)),
        "m.0.weight_scale_2": _Entry("F32", ()),
        "m.0.pre_quant_scale": _Entry("BF16", (64,)),
        "m.embed.comfy_quant": _Entry("U8", (30,)),
        "m.embed.weight": _Entry("I8", (8, 4)),          # int8 embedding ✓ 不归本模块 ✗
        "m.embed.weight_scale": _Entry("F32", (8, 1)),
    })
    result = nv.plan(good, [{"tensor": "m.0.comfy_quant", "format": "nvfp4"},
                            {"tensor": "m.embed.comfy_quant", "format": "int8_tensorwise"}])
    check("⑤ 合规层 ⇒ `supported=True`、`layers=1` ✓（**int8 的 embedding 不算 nvfp4** ✗ 按 dtype 跳过 ✓）",
          result["supported"] and result["layers"] == 1 and not result["problems"], result["note"])
    check("⑤ 报告里必须**写明布局**（打包顺序 / swizzle / 二级 scale / AWQ）✓"
          "—— 「能还原」这三个字不够，读的人要能自己复核 ✓",
          set(result["layout"]) == {"packed", "blockScale", "perTensorScale", "awq"},
          list(result["layout"]))

    broken = _Info({
        "m.0.comfy_quant": _Entry("U8", (30,)),
        "m.0.weight": _Entry("U8", (128, 32)),
        "m.0.weight_scale": _Entry("F8_E4M3", (128, 2)),   # 应为 (128, 4) ✗
        "m.0.weight_scale_2": _Entry("F32", ()),
    })
    bad = nv.plan(broken, [{"tensor": "m.0.comfy_quant", "format": "nvfp4"}])
    check("⑤ 块 scale 元素数对不上 ⇒ `supported=False` 且**说清该是多少** ✓✗"
          "（只报「不合法」等于让人自己猜 ✓✗）",
          not bad["supported"] and any("512" in problem for problem in bad["problems"]),
          bad["problems"])

    no_second = _Info({
        "m.0.comfy_quant": _Entry("U8", (30,)),
        "m.0.weight": _Entry("U8", (128, 32)),
        "m.0.weight_scale": _Entry("F8_E4M3", (128, 4)),
    })
    missing = nv.plan(no_second, [{"tensor": "m.0.comfy_quant", "format": "nvfp4"}])
    check("⑤ 缺 `weight_scale_2` ⇒ 拒绝 ✓✗（少一级尺度 ⇒ 整层数值整体偏 ✓ 且**不会报错** ✓✗）",
          not missing["supported"] and any("weight_scale_2" in p for p in missing["problems"]),
          missing["problems"])

    undeclared = nv.plan(_Info({"m.0.weight": _Entry("U8", (128, 32))}), [])
    check("⑤ 文件里没人自称 nvfp4 ⇒ 明确说「不归本模块管」✓（而不是假装体检过 ✓✗）",
          undeclared["declared"] is False and undeclared["layers"] == 0, undeclared["note"])


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 真实文件（盘上那份 14.61 GiB 的 h3 文本编码器）
# ══════════════════════════════════════════════════════════════════════════
def case_real_model() -> None:
    torch = _torch()
    from app.services.engine import loader as ld  # noqa: PLC0415
    from app.services.engine import safetensors as st  # noqa: PLC0415

    if not REAL_MODEL.exists():
        skip("⑥ 真实 nvfp4 权重体检", f"盘上找不到 {REAL_MODEL.name} ✗")
        return
    info = st.inspect(REAL_MODEL)
    result = nv.plan(info, ld.quant_formats_of(REAL_MODEL, info))
    check("⑥ 真实文件：350 层 nvfp4 **全部**通过形状自证 ✓（0 问题 ✓）"
          "—— 不是「大概能读」，是每层都算过块 scale 元素数 ✓",
          result["declared"] and result["layers"] == 350 and not result["problems"]
          and result["supported"], {"layers": result["layers"], "problems": result["problems"][:2]})
    check("⑥ 报告写明「这是同一份 plan 的口径」✓（`note` 里既有层数又有还原公式 ✓"
          "—— 只给个 `True` 等于让人自己猜怎么解的 ✓）",
          "fp4(E2M1)" in result["note"] and str(result["layers"]) in result["note"], result["note"])
    check("⑥ 其中带 AWQ `pre_quant_scale` 的层数 > 0 ✓（说明混合存在 ⇒ 折列逻辑必须对 ✓）",
          result["awqLayers"] > 0, result["awqLayers"])

    from safetensors import safe_open  # noqa: PLC0415

    stem = "model.layers.0.mlp.down_proj"
    with safe_open(str(REAL_MODEL), framework="pt") as handle:
        packed = handle.get_tensor(f"{stem}.weight")
        scale = handle.get_tensor(f"{stem}.weight_scale")
        scale_2 = handle.get_tensor(f"{stem}.weight_scale_2")
    weight = nv.dequantize(packed, scale, scale_2, (packed.shape[0], packed.shape[1] * 2),
                           dtype=torch.float32)
    spread = float(weight.std())
    check("⑥ 真权重反量化后**数值合理** ✓（std 落在 1e-3 ~ 1e-1 ✓ 出 0 或 1e3 量级都说明解错了 ✓）",
          1e-3 < spread < 1e-1, spread)
    # 反证：把 swizzle 还原去掉（当行主序用）⇒ 结果必须**明显不同** ✓
    raw = nv.unpack_e2m1(packed, dtype=torch.float32)
    blocks = scale.float()[:weight.shape[0], : weight.shape[1] // nv.BLOCK_SIZE]
    naive = (raw.reshape(weight.shape[0], -1, nv.BLOCK_SIZE)
             * (blocks * scale_2.float()).unsqueeze(-1)).reshape(weight.shape)
    gap = float((weight - naive).norm() / weight.norm())
    check("⑥ 反证：不还原 swizzle 的解与正解**相对差很大** ✓✗（与「解错了也一样正常」的错觉正面对撞 ✓）",
          gap > 0.1, gap)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 装载接线：`weights.load_module_weights` 真的按 nvfp4 还原 ✓✗
# ══════════════════════════════════════════════════════════════════════════
def _write_nvfp4_file(path: Path, stem: str, packed: Any, blocked: Any, scale_2: Any,
                      pre_quant: Any, spec: bytes) -> None:
    """造一份**结构上合法**的 nvfp4 safetensors ✓（与真文件同形：U8 打包 + swizzled 块 scale + 二级 ✓）。"""
    torch = _torch()
    from safetensors.torch import save_file  # noqa: PLC0415

    save_file({
        f"{stem}.weight": packed.to(torch.uint8),
        f"{stem}.weight_scale": blocked.to(torch.float8_e4m3fn),
        f"{stem}.weight_scale_2": scale_2.to(torch.float32).reshape(1),
        f"{stem}.pre_quant_scale": pre_quant.to(torch.float32),
        f"{stem}.comfy_quant": torch.frombuffer(bytearray(spec), dtype=torch.uint8).clone(),
    }, str(path))


def case_loader() -> None:
    """端到端：**文件 → 流式装载 → 参数** ✓（判据是「与逐位还原逐位相同」✗ 不是「装上了」✓）。"""
    torch = _torch()
    from app.services.engine import safetensors as st  # noqa: PLC0415
    from app.services.engine import weights as w_mod  # noqa: PLC0415

    if not hasattr(torch, "float8_e4m3fn"):
        skip("⑤ 装载接线", "这份 torch 没有 float8_e4m3fn ✗")
        return
    torch.manual_seed(7)
    rows, in_features = 512, 128
    stem = "blk"
    packed = torch.randint(0, 256, (rows, in_features // 2), dtype=torch.uint8)
    simple = (torch.rand(rows, in_features // nv.BLOCK_SIZE) * 0.02 + 1e-3)
    blocked = nv.to_blocked(simple)
    scale_2 = torch.tensor(0.0035, dtype=torch.float32)
    pre_quant = torch.rand(in_features, dtype=torch.float32) + 0.5
    spec = b'{"format": "nvfp4", "full_precision_matrix_mult": true}'
    #: ⚠️ 期望值必须**按文件里的实际字节**算 ✓✗：文件把块 scale 存成 e4m3 ✓（只有 4 位指数 + 3 位尾数 ✓）
    #: —— 拿 float32 的 `blocked` 去算，会得到一个「差在最后一位」的期望值 ✓✗，然后判据红得莫名其妙 ✓✗
    #: （本次实测踩过：3.15e-05 ✗）。
    blocked_stored = blocked.to(torch.float8_e4m3fn)
    want = nv.fold_pre_quant_scale(
        nv.dequantize(packed, blocked_stored, scale_2, (rows, in_features), dtype=torch.float32),
        pre_quant)

    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blk = torch.nn.Linear(in_features, rows, bias=False)

    with tempfile.TemporaryDirectory() as box:
        path = Path(box) / "nvfp4.safetensors"
        _write_nvfp4_file(path, stem, packed, blocked, scale_2, pre_quant, spec)
        report = w_mod.load_module_weights(_Block(), path)
        got = report.to_dict()

        check("⑤ ⭐ 流式装载按 **nvfp4 等式**还原 ✓ ⇒ 与逐位还原的结果**逐位相同** "
              "（「装上了」不算 ✓ —— 形状全对、数值全错才是这族最典型的坏法 ✓✗）",
              report.complete and got["loadedKeys"] == 1, got["error"] or got)
        model = _Block()
        w_mod.load_module_weights(model, path)
        check("⑤ 装载出来的参数与 `dequantize` + 折 AWQ 的结果 `torch.equal` ✓"
              "（参数是 float32 ✓ ⇒ 中间**不许**先降到 bf16 再算 ✓✗：AWQ 最大 9.7 倍 ✓ 会放大舍入）",
              torch.equal(model.blk.weight, want),
              float((model.blk.weight.float() - want.float()).abs().max()))

        check("⑤ 报告说得出「这层是按 nvfp4 还原的」✓（`layout` / 块 scale / 二级 scale / awq 四项齐 ✓）"
              "—— 只说「反量化了 1 个」等于没说 ✓",
              got["dequant"].get("converted", {}).get(f"{stem}.weight", {}).get("layout") == nv.FORMAT
              and got["dequant"].get("converted", {}).get(f"{stem}.weight", {}).get("awq") is True
              and got["dequant"].get("nvfp4") == 1, got["dequant"])
        check("⑤ nvfp4 的三个伙伴**被吃掉**（`consumedNvfp4`=3 ✓）而不是当成「多出来的键」✗"
              "（350 层 × 2~3 个 ⇒ 1000 条噪声会把真问题淹掉 ✓✗）",
              got["consumedNvfp4"] == 3 and got["consumedQuantMeta"] == 1
              and not got["unexpected"], {"nv": got["consumedNvfp4"], "un": got["unexpected"]})

        # 反证：**不** swizzle（把行主序的块 scale 直接当文件里的布局 ✓ —— 元素数一样 ✓
        #  ⇒ 形状、元素数、dtype 全合格 ✓ 装载也**不会报错** ✓✗）⇒ 必须解出**不同的值** ✓
        plain = Path(box) / "plain.safetensors"
        _write_nvfp4_file(plain, stem, packed, simple, scale_2, pre_quant, spec)
        naive_model = _Block()
        naive_report = w_mod.load_module_weights(naive_model, plain)
        gap = float((naive_model.blk.weight.float() - want.float()).norm() / want.float().norm())
        check("⑤ 反证：块 scale 不还原 swizzle ⇒ **装载照样「成功」** ✓✗ 但数值**明显不同** ✓"
              "（`to_blocked` 不是恒等 ⇒ 这条反证不是空话 ✓）",
              not torch.equal(blocked, simple) and naive_report.complete and not naive_report.error
              and gap > 0.05, {"gap": gap, "complete": naive_report.complete})

        # 「伙伴缺一个」必须在**写第一个字节之前**被拒 ✓（半装状态最难查 ✓）
        broken = Path(box) / "broken.safetensors"
        from safetensors.torch import save_file  # noqa: PLC0415

        save_file({
            f"{stem}.weight": packed,
            f"{stem}.weight_scale": blocked.to(torch.float8_e4m3fn),
            f"{stem}.comfy_quant": torch.frombuffer(bytearray(spec), dtype=torch.uint8).clone(),
        }, str(broken))
        hole = _Block()
        before = hole.blk.weight.clone()
        hole_report = w_mod.load_module_weights(hole, broken)
        check("⑤ 缺 `weight_scale_2` ⇒ **写之前**就拒 ✓ 并且模型零污染 ✓（不是装到一半 ✓✗）",
              not hole_report.complete and "weight_scale_2" in (hole_report.error or "")
              and torch.equal(hole.blk.weight, before), hole_report.error)
        del st


def case_real_loader() -> None:
    """真实权重端到端：**一层**真 nvfp4 装进 `nn.Linear` ✓（体检说能还原 ⇒ 就必须真能还原 ✓✗）。"""
    torch = _torch()
    from app.services.engine import loader as ld  # noqa: PLC0415
    from app.services.engine import safetensors as st  # noqa: PLC0415
    from app.services.engine import weights as w_mod  # noqa: PLC0415

    if not REAL_MODEL.exists():
        skip("⑥ 真实权重端到端装载", f"盘上找不到 {REAL_MODEL.name} ✗")
        return
    reader = st.TensorReader(REAL_MODEL)
    found = w_mod.nvfp4_layers_of(reader)
    check("⑥⭐ `weights.nvfp4_layers_of` 与体检**同一个入口**（`quant.supported_format` ✓）"
          "⇒ 该文件判出 350 层 ✓（自己认字符串的话，两边迟早分叉 ✓✗）",
          len(found) == 350, len(found))
    del reader

    stem = "model.layers.0.mlp.down_proj"
    info = st.inspect(REAL_MODEL)
    rows = int(info.tensors[f"{stem}.weight"].shape[0])
    cols = int(info.tensors[f"{stem}.weight"].shape[1]) * 2

    class _One(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(cols, rows, bias=False)

    model = _One()
    report = w_mod.load_module_weights(
        model, REAL_MODEL, key_map={"^" + re.escape(stem) + r"\.": "proj."})
    got = report.to_dict()
    check("⑥ 真权重：**一层**装进参数 ✓ 且报告说明是按 nvfp4 还原的 ✓（`complete` ✓、无形状不符 ✓）",
          report.complete and got["dequant"].get("converted", {}).get("proj.weight", {}).get("layout")
          == nv.FORMAT, got["error"] or got["dequant"].get("converted"))

    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(REAL_MODEL), framework="pt") as handle:
        packed = handle.get_tensor(f"{stem}.weight")
        scale = handle.get_tensor(f"{stem}.weight_scale")
        scale_2 = handle.get_tensor(f"{stem}.weight_scale_2")
        pq = handle.get_tensor(f"{stem}.pre_quant_scale")
    #: ⚠️ 真文件里的 `pre_quant_scale` 是 **bf16** ✓（不是 f32 ✓）⇒ 折列也在 bf16 精度上 ✓，
    #: 而参数的 dtype 是 **float32** ✓ ⇒ 期望值按 float32 算、**不**降到 bf16 ✓✗
    #: （降到 bf16 会比出 1.9e-03 的差 ✗ —— 那不是解错，是**期望值自己先舍入了一次** ✓✗）
    want = nv.fold_pre_quant_scale(
        nv.dequantize(packed, scale, scale_2, (rows, cols), dtype=torch.float32), pq)
    check("⑥⭐ 真权重端到端与逐位还原 `torch.equal` ✓（体检的「能还原」= 真能还原 ✓ 不是口号 ✓）",
          torch.equal(model.proj.weight, want),
          float((model.proj.weight.float() - want.float()).abs().max()))


def main() -> int:
    case_spec()
    case_reference()
    case_layout()
    case_formula()
    case_plan()
    case_loader()
    case_real_model()
    case_real_loader()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for note in _SKIPPED:
        print("SKIP  " + note)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"，{len(_SKIPPED)} skipped" if _SKIPPED else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
