"""LoRA 装配自检：**张量运算用假实现验算法** ✓（2026-10-29 ✓）。

这一套钉的是 `app/services/engine/lora.py` 的三条路与四条边界 ✓，全部**零 torch** ✓：
假张量（带 `.shape` 的二维 list ✓）+ 假算子（三重循环 ✓）就能把 `W + up@down·scale` 与
`base + up(down(x))·scale` 算清楚 ✓ —— 判据在测试里**另写一遍**（不复用被测代码 ✓），
算错就红 ✓。

⚠️ 不测"真权重能不能跑"✗：那要 torch + 真 checkpoint（`engine_dual_stream_test` 那条线 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_lora_test.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import lora as lora_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str) -> str | None:
    """**能触发**的反向证明 ✓：调它、看报错里有没有那个词 ✓（没报错 ⇒ None ⇒ 断言红 ✓）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 就是来看它报什么的 ✓
        return str(err) if needle in str(err) else None
    return None


@dataclass
class FakeTensor:
    """二维假张量 ✓（只提供 `.shape` ✓ —— 被测代码只认这一个属性 ✓）。"""

    rows: list[list[float]]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"FakeTensor{self.shape}"


class ListOps:
    """假算子 ✓（三重循环 ✓）—— 与 :class:`lora.TensorOps` 同签名 ✓。"""

    def shape(self, tensor: FakeTensor) -> tuple[int, ...]:
        return tensor.shape

    def matmul(self, left: FakeTensor, right: FakeTensor) -> FakeTensor:
        rows, inner = left.shape
        inner2, cols = right.shape
        assert inner == inner2, f"matmul 维度不匹配 {left.shape} @ {right.shape}"
        out = [[sum(left.rows[r][k] * right.rows[k][c] for k in range(inner))
                for c in range(cols)] for r in range(rows)]
        return FakeTensor(out)

    def linear(self, inputs: FakeTensor, weight: FakeTensor) -> FakeTensor:
        rows, cols = weight.shape
        transposed = FakeTensor([[weight.rows[r][c] for r in range(rows)] for c in range(cols)])
        return self.matmul(inputs, transposed)

    def add_scaled(self, base: FakeTensor, delta: FakeTensor, scale: float) -> FakeTensor:
        return FakeTensor([[base.rows[r][c] + delta.rows[r][c] * scale
                            for c in range(base.shape[1])] for r in range(base.shape[0])])


OPS = ListOps()


def tensor(rows: list[list[float]]) -> FakeTensor:
    return FakeTensor(rows)


def close(left: FakeTensor, right: FakeTensor, tol: float = 1e-9) -> bool:
    if left.shape != right.shape:
        return False
    return all(abs(left.rows[r][c] - right.rows[r][c]) < tol
               for r in range(left.shape[0]) for c in range(left.shape[1]))


#: rank=2 / in=3 的 down ✓ 与 out=4 / rank=2 的 up ✓（小到能手算 ✓）
DOWN = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
UP = tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 2.0]])
WEIGHT = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0]])
INPUTS = tensor([[1.0, 2.0, 4.0], [0.0, 3.0, 1.0]])


def ref_fold(weight: FakeTensor, down: FakeTensor, up: FakeTensor, scale: float) -> FakeTensor:
    """**另写一遍** ``W + up@down·scale`` ✓（三重循环，不复用被测代码 ✓）。"""
    update = OPS.matmul(up, down)
    return OPS.add_scaled(weight, update, scale)


def case_parse() -> None:
    report = lora_mod.parse_lora({
        "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": DOWN,
        "diffusion_model.blocks.0.mlp.fc1.lora_B.weight": UP,
        "diffusion_model.blocks.0.mlp.fc1.alpha": 4.0,
    })
    check("解析: 去掉 diffusion_model. 前缀，模块名对齐", list(report.deltas) == ["blocks.0.mlp.fc1"],
          list(report.deltas))
    delta = report.deltas["blocks.0.mlp.fc1"]
    check("解析: scale = alpha/rank = 4.0/2 = 2.0（不带 strength）",
          abs(delta.scale() - 2.0) < 1e-9, delta.scale())
    check("解析: strength 是乘上去的（0.5 ⇒ 1.0）", abs(delta.scale(0.5) - 1.0) < 1e-9, delta.scale(0.5))
    check("解析: 没有 alpha 时按约定取 1.0（再乘 strength=0.75）",
          abs(lora_mod.parse_lora({
              "x.lora_A.weight": DOWN, "x.lora_B.weight": UP}).deltas["x"].scale(0.75) - 0.75) < 1e-9)

    check("解析: lora_down/lora_up 与 .default 后缀都认",
          set(lora_mod.parse_lora({"m.lora_down.weight": DOWN, "m.lora_up.weight": UP,
                                   "n.lora_A.default.weight": DOWN,
                                   "n.lora_B.default.weight": UP}).deltas) == {"m", "n"})


def case_orphan_and_error() -> None:
    only_a = lora_mod.parse_lora({"blocks.9.mlp.fc1.lora_A.weight": DOWN})
    check("边界: 只有 A 没有 B -> 进 orphans、不装（不猜 ✗）",
          list(only_a.deltas) == [] and len(only_a.orphans) == 1, only_a.to_dict())
    check("边界: orphans 文案说清缺的是哪一半", "只有A" in only_a.orphans[0], only_a.orphans)
    only_b = lora_mod.parse_lora({"blocks.9.mlp.fc1.lora_B.weight": UP})
    check("边界: 反过来（只有 B）也要说清", only_b.orphans == ["blocks.9.mlp.fc1（只有B）"],
          only_b.orphans)

    mismatch = _raises(lambda: lora_mod.parse_lora({
        "m.lora_A.weight": tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),   # rank 2
        "m.lora_B.weight": tensor([[1.0, 0.0, 0.0]]),                     # rank 3 ✗
    }), "rank")
    check("边界: A/B 的 rank 对不上 -> 抛 LoraError（装上去会静默算错 ✗）", mismatch is not None, mismatch)

    unknown = lora_mod.parse_lora({"m.lora_A.weight": object(), "m.lora_B.weight": object()})
    check("边界: 形状信息不全 -> 进 orphans 而不是崩", unknown.orphans and not unknown.deltas,
          unknown.to_dict())


def case_paths() -> None:
    adaln = lora_mod.parse_lora({
        "diffusion_model.blocks.3.adaln_proj.lora_A.weight": DOWN,
        "diffusion_model.blocks.3.adaln_proj.lora_B.weight": UP})
    check("adaln: 标成 runtimeInject（pruned 底模已把 silu(t_emb) 空间塌成曲线表）",
          adaln.runtimeInject == ["blocks.3.adaln_proj"], adaln.to_dict())
    check("adaln: 仍然解析出 delta（注入时要用它的 down/up）", "blocks.3.adaln_proj" in adaln.deltas)

    path, reason = lora_mod.choose_path("blocks.0.mlp.fc2", layout="TensorWiseINT8")
    check("选路: int8 融合的 fc2 -> merge（bypass 钩子根本不会触发 ✗）",
          path == "merge" and "int8" in reason, (path, reason))
    check("选路: int8 但已转置（非融合布局）-> 回到 bypass",
          lora_mod.choose_path("blocks.0.mlp.fc2", layout="TensorWiseINT8", transposed=True)[0]
          == "bypass")
    check("选路: 普通 bf16/fp8 模块 -> bypass（merge 会被四舍五入吃掉 ✗）",
          lora_mod.choose_path("blocks.0.mlp.fc1", layout="bf16")[0] == "bypass")
    check("选路: adaln -> runtime（既不能 merge 也不该走 bypass）",
          lora_mod.choose_path("blocks.3.adaln_proj")[0] == "runtime")
    check("选路: 三种都给了非空理由（不许只给结论 ✓）",
          all(lora_mod.choose_path(m, layout=lay)[1]
              for m, lay in (("blocks.3.adaln_proj", ""), ("blocks.0.mlp.fc2", "TensorWiseINT8"),
                             ("blocks.0.mlp.fc1", ""))))


def case_math() -> None:
    delta = lora_mod.LoraDelta(module="m", down=DOWN, up=UP, alpha=4.0)  # scale 2.0

    folded = lora_mod.fold_weight(WEIGHT, delta, ops=OPS)
    check("merge: W + up@down·(alpha/rank)", close(folded, ref_fold(WEIGHT, DOWN, UP, 2.0)),
          folded.rows)
    check("merge: strength 线性作用（0.5 ⇒ 一半增量）",
          close(lora_mod.fold_weight(WEIGHT, delta, 0.5, ops=OPS),
                ref_fold(WEIGHT, DOWN, UP, 1.0)))
    check("merge: strength=0 ⇒ 一个比特都不改",
          close(lora_mod.fold_weight(WEIGHT, delta, 0.0, ops=OPS), WEIGHT))

    base = OPS.linear(INPUTS, WEIGHT)
    expected = OPS.add_scaled(base, OPS.linear(OPS.linear(INPUTS, DOWN), UP), 2.0)
    check("bypass: base(x) + up(down(x))·(alpha/rank)",
          close(lora_mod.bypass_output(base, INPUTS, delta, ops=OPS), expected))
    check("bypass: 与 merge 懒等价（同一输入下两条路必须落到同一个数 ✓）",
          close(lora_mod.bypass_output(base, INPUTS, delta, ops=OPS),
                OPS.linear(INPUTS, lora_mod.fold_weight(WEIGHT, delta, ops=OPS))),
          OPS.linear(INPUTS, folded).rows)

    check("报告: to_dict 给出模块数/孤儿数/注入数（不静默丢 ✓）",
          lora_mod.parse_lora({
              "a.lora_A.weight": DOWN, "a.lora_B.weight": UP,
              "b.lora_A.weight": DOWN,
              "diffusion_model.c.adaln_proj.lora_A.weight": DOWN,
              "diffusion_model.c.adaln_proj.lora_B.weight": UP}).to_dict()
          == {"modules": 2, "runtimeInject": ["c.adaln_proj"], "runtimeInjectCount": 1,
              "orphans": ["b（只有A）"], "orphanCount": 1, "error": ""})


class FakeLinear:
    """假 ``nn.Linear`` ✓ —— 只提供 :func:`lora.apply_to_model` 会碰的两样东西 ✓。"""

    def __init__(self, weight: FakeTensor) -> None:
        self.weight = SimpleNamespace(data=weight)
        self.hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> Any:
        self.hooks.append(hook)
        return hook


class FakeModel:
    def __init__(self, modules: dict[str, FakeLinear]) -> None:
        self._modules = modules

    def named_modules(self) -> Any:
        return self._modules.items()


def case_apply_model() -> None:
    delta = lora_mod.LoraDelta(module="blocks.0.mlp.fc1", down=DOWN, up=UP, alpha=4.0)
    parsed = lora_mod.LoraReport(deltas={"blocks.0.mlp.fc1": delta})

    linear = FakeLinear(WEIGHT)
    result = lora_mod.apply_to_model(FakeModel({"blocks.0.mlp.fc1": linear}), parsed, ops=OPS)
    check("装配: 默认走 bypass（挂前向钩子，不折权重）",
          result.bypassed == ["blocks.0.mlp.fc1"] and linear.hooks and result.merged == [],
          result.to_dict())
    check("装配: 权重**一个比特都没动**（量化底模上 merge 会被吃掉 ✗）",
          close(linear.weight.data, WEIGHT))
    hook = linear.hooks[0]
    base = OPS.linear(INPUTS, WEIGHT)
    check("装配: 钩子给出 base(x) + up(down(x))·scale",
          close(hook(None, (INPUTS,), base),
                OPS.add_scaled(base, OPS.linear(OPS.linear(INPUTS, DOWN), UP), 2.0)))

    int8 = FakeLinear(WEIGHT)
    merged = lora_mod.apply_to_model(FakeModel({"blocks.0.mlp.fc2": int8}),
                                     lora_mod.LoraReport(deltas={
                                         "blocks.0.mlp.fc2": lora_mod.LoraDelta(
                                             module="blocks.0.mlp.fc2", down=DOWN, up=UP, alpha=4.0)}),
                                     layout="TensorWiseINT8", ops=OPS)
    check("装配: int8 底模的 fc2 走 merge（钩子挂不上去 ✗）且数值 = W + up@down·scale",
          merged.merged == ["blocks.0.mlp.fc2"] and not int8.hooks
          and close(int8.weight.data, ref_fold(WEIGHT, DOWN, UP, 2.0)), merged.to_dict())

    none = lora_mod.apply_to_model(FakeModel({}), parsed, ops=OPS)
    check("装配: 模型里没这层 -> missing（不崩 ✗ 也不假装装好 ✗）",
          len(none.missing) == 1 and none.applied == 0, none.to_dict())

    adaln = lora_mod.apply_to_model(FakeModel({"blocks.3.adaln_proj": FakeLinear(WEIGHT)}),
                                    lora_mod.parse_lora({
                                        "blocks.3.adaln_proj.lora_A.weight": DOWN,
                                        "blocks.3.adaln_proj.lora_B.weight": UP}), ops=OPS)
    check("装配: adaln 没装就**明说没装**（error 非空），不许当成功 ✗",
          adaln.runtime and adaln.error and adaln.applied == 0, adaln.to_dict())

    renamed = FakeLinear(WEIGHT)
    moved = lora_mod.apply_to_model(FakeModel({"dit.blocks.0.mlp.fc1": renamed}), parsed,
                                    ops=OPS, transform=lambda name: f"dit.{name}")
    check("装配: 改名映射可用（默认恒等；H3 实测不需要 ✓）",
          moved.bypassed == ["blocks.0.mlp.fc1"] and renamed.hooks, moved.to_dict())

    two = lora_mod.apply_to_model(FakeModel({"blocks.0.mlp.fc1": FakeLinear(WEIGHT)}), parsed,
                                  strength=0.0, ops=OPS)
    zero = FakeModel({})
    _ = two
    check("装配: strength=0 的钩子等于直通（scale=0 ⇒ 输出不变）",
          close(lora_mod.bypass_output(base, INPUTS, delta, 0.0, ops=OPS), base))

    _ = zero


#: 盘上真实的 H3 turbo LoRA ✓（没有就 skip ✓ —— 这条测的是「真文件的键与形状长什么样」✓）
REAL_LORA = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\loras"
                 r"\minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")
#: 它要挂的底模 ✓（只读头部 ✓ 20 GiB 也不读数据区 ✓）
REAL_BASE = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models"
                 r"\minimax_h3_fl2va_pruned_int8_convrot.safetensors")


class ShapeOnly:
    """只有 ``.shape`` 的桩 ✓（真文件 1.8 GB 不读数据也能验键与 rank ✓）。"""

    def __init__(self, shape: Any) -> None:
        self.shape = tuple(int(size) for size in shape or ())


def case_kohya_names() -> None:
    """⑨ kohya 下划线分层键（2026-10-31 补 ✓）。

    为什么单独立一条 ✗：kohya 出的 SD1.5/SDXL/SD3 LoRA，键整串**一个点都没有** ✓
    （``lora_unet_down_blocks_0_attentions_0_attn1_to_q`` ✓），模型里是点分层 ✓ ——
    不翻译就 ``modules.get()`` 全 None ⇒ 报「一层都没装上」✓✗（报错好过静默 ✓，
    但能力等于没有 ✓✗）。
    """
    check("⑨ 下划线分层 → 点分层 ✓（SDXL 的 attn1.to_q ✓）",
          lora_mod.kohya_module_name(
              "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q")
          == "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q", "")
    check("⑨′ 已是点分层 ⇒ **原样返回** ✓✗（二次翻译会把 `to_q` 这类段拆坏 ✓）",
          lora_mod.kohya_module_name("blocks.0.mlp.fc1") == "blocks.0.mlp.fc1", "")
    check("⑨″ 文本塔的键也认 ✓（`lora_te1_text_model_encoder_layers_0_mlp_fc1` ✓）",
          lora_mod.kohya_module_name("lora_te1_text_model_encoder_layers_0_mlp_fc1")
          == "text_model.encoder.layers.0.mlp.fc1", "")

    parsed = lora_mod.parse_lora({
        "lora_unet_down_blocks_0_attentions_0_attn1_to_q.lora_A.weight": DOWN,
        "lora_unet_down_blocks_0_attentions_0_attn1_to_q.lora_B.weight": UP})
    linear = FakeLinear(WEIGHT)
    result = lora_mod.apply_to_model(
        FakeModel({"down_blocks.0.attentions.0.attn1.to_q": linear}), parsed, ops=OPS)
    check("⑨‴ kohya 键**靠翻译真装上** ✓✗（不翻译 ⇒ 「一层都没装上」⇒ 能力等于没有 ✓✗）",
          result.bypassed == ["lora_unet_down_blocks_0_attentions_0_attn1_to_q"]
          and bool(result.translated) and linear.hooks and result.ok, result.to_dict())

    blocked = lora_mod.apply_to_model(
        FakeModel({"down_blocks.0.attentions.0.attn1.to_q": FakeLinear(WEIGHT)}), parsed,
        ops=OPS, transform=lambda name: name)
    check("⑨⁴ 传了 `transform` ⇒ **不再**叠加翻译 ✓✗（两套口径打架 ⇒ 悄悄折进错的层 ✓✗）",
          blocked.applied == 0 and not blocked.translated and bool(blocked.error),
          blocked.to_dict())

    #: ⚠️ 段名**长得像**别的段，切开顺序/最长匹配一错就整份 LoRA 装不上 ✗（2026-10-31 实测补 ✓）。
    check("⑨⁵ ``self_attn_out_proj`` ⇒ ``self_attn.out_proj`` ✓（参考实现的 ``LORA_CLIP_MAP`` "
          "列的就是这个名 ✓；漏 ``out_proj`` ⇒ 被切成 ``out.proj`` ⇒ 整份 LoRA 一层都装不上 ✓✗）",
          lora_mod.kohya_module_name(
              "lora_te1_text_model_encoder_layers_0_self_attn_out_proj")
          == "text_model.encoder.layers.0.self_attn.out_proj", "")
    check("⑨⁶ ``ff_net_0_proj`` ⇒ ``ff.net.0.proj`` ✓（GEGLU 前馈的模块名**本身含点** ✓ —— "
          "光靠切段只能切出 ``ff_net.0.proj`` ⇒ 对不上 ✓✗）",
          lora_mod.kohya_module_name(
              "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj")
          == "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj", "")
    check("⑨⁷ 文本塔的 LayerNorm 名不许被拆成 ``layer.norm1`` ✓（``layer_norm1`` 是整段 ✓）",
          lora_mod.kohya_module_name(
              "lora_te1_text_model_encoder_layers_0_layer_norm1")
          == "text_model.encoder.layers.0.layer_norm1", "")


def case_real_file() -> None:
    from app.services.engine import safetensors as st  # noqa: PLC0415

    if not REAL_LORA.exists():
        skip(f"真文件核对（{REAL_LORA.name} 不在盘上）")
        return
    _metadata, header = st.read_header(REAL_LORA)
    tensors = {name: ShapeOnly(raw.get("shape"))
               for name, raw in header.items() if name != "__metadata__"}
    report = lora_mod.parse_lora(tensors)
    check("真文件: 624 键 -> 208 个模块、0 孤儿、0 adaln（turbo LoRA 不需要运行期注入 ✓）",
          (len(tensors), len(report.deltas), len(report.orphans), len(report.runtimeInject))
          == (624, 208, 0, 0),
          (len(tensors), len(report.deltas), report.orphans[:3], report.runtimeInject[:3]))

    # 模块名与**真底模**的参数名对账 ✓ —— 这就是「不需要 key_map」的证据 ✓
    lora_modules = set(report.deltas)
    if not REAL_BASE.exists():
        skip(f"底模名对账（{REAL_BASE.name} 不在盘上）")
    else:
        _meta, base_header = st.read_header(REAL_BASE)
        base_modules = {name[: -len(".weight")] for name in base_header
                        if name.endswith(".weight") and name != "__metadata__"}
        check("真文件: LoRA 的 208 个模块全部是真底模里的参数名（逐字一致，不用改名 ✓）",
              lora_modules <= base_modules, sorted(lora_modules - base_modules)[:5])

    ranks = {report.deltas[m].scale(): round(report.deltas[m].down.shape[0], 0)
             for m in ("blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1")}
    check("真文件: qkv rank=384、其余 rank=128（alpha/rank 各自算 ✓）",
          report.deltas["blocks.0.attn.qkv_proj"].down.shape[0] == 384
          and report.deltas["blocks.0.mlp.fc1"].down.shape[0] == 128, ranks)
    check("真文件: fc2 有 52 层 ⇒ 量化底模下这 52 层必须 merge",
          len([m for m in lora_modules if m.endswith(".mlp.fc2")]) == 52)
    check("真文件: 因子形状是 nn.Linear 约定（down=[rank,in]、up=[out,rank]，无需转置 ✓）",
          report.deltas["blocks.0.mlp.fc1"].down.shape == (128, 5376)
          and report.deltas["blocks.0.mlp.fc1"].up.shape == (28672, 128))


def case_wiring() -> None:
    """接线门 ✓ —— **不 import 重型模块**也能钉住的三件事 ✓（ast 读签名 ✓）。"""
    import ast  # noqa: PLC0415

    engine = BACKEND_PY / "app" / "services" / "engine"

    def params_of(filename: str, function: str) -> set[str]:
        tree = ast.parse((engine / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                names = {arg.arg for arg in node.args.args}
                names |= {arg.arg for arg in node.args.kwonlyargs}
                return names
        raise AssertionError(f"{filename} 里没有 {function}")

    check("接线: TorchBackend.load_weights 收 loras",
          "loras" in params_of("torch_backend.py", "load_weights"))
    check("接线: runtime.ensure_loaded / _ensure_loaded_internal 都收 loras",
          {"loras"} <= params_of("runtime.py", "ensure_loaded")
          and {"loras"} <= params_of("runtime.py", "_ensure_loaded_internal"))

    runtime_src = (engine / "runtime.py").read_text(encoding="utf-8")
    loads = [line for line in runtime_src.splitlines()
             if "load_weights(" in line and "report[" in line]
    check("接线: 真把 loras 传给 load_weights（不是收了参数就丢掉 ✗）",
          any("loras=loras" in line for line in loads), loads[:2])
    uses = runtime_src.count("loras=_lora_key(loras)")
    check("接线: 两处装载键都把 loras 算进去（否则换 LoRA 会复用旧模型 ✗✗）", uses == 2, uses)

    try:
        from app.services.engine import runtime as runtime_mod  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001
        skip(f"_lora_key 行为验证（runtime 在本环境导入不了：{type(err).__name__}）")
    else:
        one = lora_mod.LoraSpec(path=r"D:\a.safetensors", strength=0.8)
        two = lora_mod.LoraSpec(path=r"D:\b.safetensors", strength=1.5)
        check("装载键: 顺序无关（加法式增量可交换 ✓）",
              runtime_mod._lora_key([one, two]) == runtime_mod._lora_key([two, one]))
        check("装载键: 换强度 / 换文件 / 去掉 LoRA 都必变（不许复用旧模型 ✗）",
              len({runtime_mod._lora_key([one, two]), runtime_mod._lora_key([one]), 
                   runtime_mod._lora_key([lora_mod.LoraSpec(path=one.path, strength=0.5)]),
                   runtime_mod._lora_key(None)}) == 4)
        check("装载键: dict 写法和 LoraSpec 写法给出同一个键",
              runtime_mod._lora_key([{"path": one.path, "strength": 0.8}])
              == runtime_mod._lora_key([one]))
        #: ⚠️ 这两条盯的是**实测到的真缺陷** ✓✗（2026-10-29）——`_lora_key` 以前自己解析、
        #:    且**只认 `strength`** ✓✗ ⇒ `{"strength_model": 0.5}` 记成 **1.0** 而实际按 0.5 挂
        #:    ⇒ 「先按 1.0 装过 ⇒ 再来 0.5 直接复用旧模型、一声不响」✓✗（"不是那个档"查不出原因 ✓）。
        check("装载键: ``strength_model`` 别名与 ``strength`` 必给出同一个键（记错强度 = 静默复用 ✗✗）",
              runtime_mod._lora_key([{"path": one.path, "strength_model": 0.8}])
              == runtime_mod._lora_key([one])
              == runtime_mod._lora_key([{"path": one.path, "strength": 0.8}]))
        check("装载键: 显式 strength=0 记 0.0（抬成 1.0 就与实挂不一致 ✗✗）",
              runtime_mod._lora_key([{"path": one.path, "strength": 0}])
              == ((one.path, 0.0),), runtime_mod._lora_key([{"path": one.path, "strength": 0}]))

    try:
        from app.services.engine import bridge as bridge_mod  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001
        skip(f"settings→LoraSpec 验证（bridge 在本环境导入不了：{type(err).__name__}）")
        return
    if not REAL_LORA.exists():
        skip("settings→LoraSpec 用真文件验证（LoRA 不在盘上）")
        return
    specs = bridge_mod.lora_specs_from_settings({"loras": [{"path": str(REAL_LORA), "strength": 0.7}]})
    check("settings: {path, strength} -> LoraSpec(path, strength)",
          len(specs) == 1 and specs[0].strength == 0.7 and specs[0].path == str(REAL_LORA), specs)
    check("settings: 单条 ``lora`` 也认，缺省 strength=1.0",
          [s.strength for s in bridge_mod.lora_specs_from_settings({"lora": str(REAL_LORA)})] == [1.0])
    check("settings: 没配 -> 空（不算错误 ✓）",
          bridge_mod.lora_specs_from_settings({}) == [])
    relative = bridge_mod.lora_specs_from_settings({
        "models_dir": str(REAL_LORA.parent), "loras": [REAL_LORA.name]})
    check("settings: 相对路径按 models_dir 解析成绝对路径",
          relative and relative[0].path == str(REAL_LORA), relative)

    missing = _raises(lambda: bridge_mod.lora_specs_from_settings(
        {"loras": [r"D:\no-such-lora.safetensors"]}), "不存在")
    check("settings: 文件不存在 -> 当场报（不静默跳过 ✗）", missing is not None, missing)
    empty = _raises(lambda: bridge_mod.lora_specs_from_settings({"loras": [{"strength": 1.0}]}), "path")
    check("settings: 条目缺 path -> 当场报", empty is not None, empty)

    #: ⚠️ `lora_assembly` 存在的**唯一理由**：装配要能出门（回显）**且不取整** ✓✗ ——
    #:    `LoraSpec.to_dict()` 把 strength 抹到 4 位小数 ✓✗，而装载键（`runtime._lora_key` ✓）
    #:    比的是**原值** ✓ ⇒ 谁把这里"简化"成 `[spec.to_dict() for spec in specs]`，同一个 LoRA
    #:    就会在"回显"与"键"两侧算出**两个数** ⇒ 每换一次强度白装一遍模型 ✓✗，报告里的强度也
    #:    不再是实挂的那个数 ✓✗。这条以前**一条判据都没有** ✓✗（2026-10-29 补 ✓）。
    raw = lora_mod.LoraSpec(path=r"D:\长小数.safetensors", strength=0.123456789, name="turbo")
    assembled = bridge_mod.lora_assembly([raw])
    check("装配: 条目是 **dict** 且键名归一成 path/strength/name（回显要过裸 json.dumps ✗✗）",
          isinstance(assembled[0], dict)
          and set(assembled[0]) == {"path", "strength", "name"}, assembled)
    check("装配: strength 是**原值、不取整**（抹到 4 位 ⇒ 同一个 LoRA 算出两个键、白装一遍 ✗✗）",
          assembled[0]["strength"] == raw.strength
          and assembled[0]["strength"] != raw.to_dict()["strength"], assembled)
    check("装配: 回显与装载键**同一把尺**（`_lora_key(装配)` == `_lora_key([spec])` ✓）",
          runtime_mod._lora_key(assembled) == runtime_mod._lora_key([raw]), assembled)
    try:
        text = json.dumps(assembled, ensure_ascii=False)
    except TypeError as err:
        check("装配: 裸 json.dumps 能出去（塞 `LoraSpec` 就是 500 ✗✗）", False, repr(err))
    else:
        check("装配: 裸 json.dumps 能出去（塞 `LoraSpec` 就是 500 ✗✗）",
              '"strength"' in text and "LoraSpec" not in text, text[:90])


def case_coerce_spec() -> None:
    """``lora.coerce_spec`` —— 写法与 ``strength`` 口径的**唯一一处** ✓（2026-10-29 ✓）。"""
    spec = lora_mod.coerce_spec({"path": "/a/b.safetensors", "strength": 0.35, "name": "turbo"})
    check("coerce: {path, strength, name} -> LoraSpec",
          (spec.path, spec.strength, spec.name) == ("/a/b.safetensors", 0.35, "turbo"), spec)

    #: ⚠️ 这两条盯的是**真修过的缺陷** ✗✗：同一件事以前写了三遍（桥 / 装载键 / torch 后端 ✓），
    #:    其中后端那句是 ``float(item.get("strength", 1.0) or 1.0)`` ✓✗ —— 显式 ``0`` 被抬成
    #:    **1.0** ✓✗，而装载键按 ``0.0`` 记 ✓ ⇒ 「键说 0.0、实际按 1.0 挂」✓✗
    #:    ⇒ 换强度不重装、报告里的强度也是假的 ✓✗（本仓最恨的那类不对称 ✓）。
    check("coerce: 显式 strength=0 原样保留（不许被 `or 1.0` 抬成 1.0 ✗✗）",
          lora_mod.coerce_spec({"path": "/a/b.safetensors", "strength": 0}).strength == 0.0)
    check("coerce: 显式 strength=None -> 1.0（**不是** 0.0 ✗：那是把「没写」读成「关掉」✗）",
          lora_mod.coerce_spec({"path": "/a/b.safetensors", "strength": None}).strength == 1.0)
    check("coerce: 缺省 strength -> 1.0",
          lora_mod.coerce_spec({"path": "/a/b.safetensors"}).strength == 1.0)
    check("coerce: ``strength_model`` 别名也认",
          lora_mod.coerce_spec({"path": "/a/b.safetensors", "strength_model": 0.5}).strength == 0.5)
    check("coerce: 路径字符串 -> strength 1.0 / name 空",
          (lambda s: (s.path, s.strength, s.name))(
              lora_mod.coerce_spec("/a/b.safetensors")) == ("/a/b.safetensors", 1.0, ""))
    passes = lora_mod.LoraSpec(path="/a/b.safetensors", strength=0.9)
    check("coerce: 给 LoraSpec 原样返回（不复制、不改写）",
          lora_mod.coerce_spec(passes) is passes)
    check("coerce: 条目缺 path -> 报（不默认成空路径 ✗）",
          _raises(lambda: lora_mod.coerce_spec({"strength": 1.0}), "path") is not None)
    check("coerce: 写法不认识 -> 报",
          _raises(lambda: lora_mod.coerce_spec(123), "写法") is not None)


def case_reuse_gate() -> None:
    """复用门 ✓ —— LoRA 是**模型的一部分** ✓（2026-10-29 补 ✓）。

    这条盯**行为**，不是 `case_wiring` 那种"数出现次数"✗✗：键算得再对，只要没进
    `_loadKeyTuple` 的比较，换强度照样**静默复用** ✓✗（画面上只表现为"不是那个档" ✓），
    而那是本仓最恨的那种错 —— 一声不响 ✓。

    零 torch ✗ 也能验：复用判据排在"引擎正忙"检查**之前** ✓ ⇒ 把 `_current` 设上就能把两种
    结局分开 —— 复用 ⇒ 直接返回 `reused: True` ✓；没复用 ⇒ `EngineBusy` ✓（在碰 torch 之前就抛 ✓）。
    正例刻意**带 busy** ✓：同一套装配再要一次本来就不该因为忙而拒绝 ✓（那才是白等一分钟 ✓）。
    """
    try:
        from app.services.engine import runtime as runtime_mod  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001
        skip(f"复用门验证（runtime 在本环境导入不了：{type(err).__name__}）")
        return

    item = {"path": r"D:\m\turbo.safetensors", "strength": 0.5}
    dit = r"D:\m\dit.safetensors"

    def _key(*loras: Any) -> tuple[Any, ...]:
        # ⚠️ 装载键的**参数清单不许手抄** ✗✗：2026-10-31 实测正在这里翻车 ✓✗ —— 加了 `residency` 之后
        #    这里少抄两项 ⇒ 正例被判成"没复用" ⇒ 抛 `EngineBusy` ✓✗，而报错文字说的是"引擎正忙"，
        #    **完全看不出是这条测试漏了一行** ✓✗（红得对、指错地方 = 下一轮还得从头查一遍 ✓）。
        # ⇒ 取 `_ensure_loaded_internal` 的**签名默认值** ✓：`ensure_loaded` 把它的全部参数原样转给
        #    `_load_key` ✓（这条等价关系由 `engine_router_contract_test` ⑪′ 那侧兜着 ✓），
        #    以后再加装载参数**不用回来改这里** ✓。
        params = inspect.signature(
            runtime_mod.EngineRuntime._ensure_loaded_internal).parameters  # noqa: SLF001
        defaults = {name: param.default for name, param in params.items()
                    if name != "self" and param.default is not inspect.Parameter.empty}
        return runtime_mod._load_key(**{**defaults, "stage": "h3", "dit_path": dit,
                                        "loras": runtime_mod._lora_key(list(loras))})

    def _installed() -> Any:
        #: 假装"已按 strength=0.5 装好" ✓ —— `_backend` 只要非 None ✓（复用门不看它的类型 ✓）
        rt = runtime_mod.EngineRuntime()
        rt._backend = object()          # noqa: SLF001
        rt._loadKeyTuple = _key(item)   # noqa: SLF001
        rt._loadReport = {"stage": "h3", "weights": {"applied": 1}}  # noqa: SLF001
        rt._current = "task-1"          # noqa: SLF001 —— 忙：于是"没复用"当场抛 ✓
        return rt

    def _ask(loras: list[Any]) -> Any:
        return _installed().ensure_loaded(stage="h3", dit_path=dit, loras=loras)

    same = _ask([dict(item)])
    check("复用门: 同一份 LoRA 再要一次 ⇒ reused（**忙也不拒** ✓：本来就不用重装 ✓）",
          same.get("reused") is True, same)

    alias = _ask([{"path": item["path"], "strength_model": 0.5}])
    check("复用门: ``strength_model`` 写法命中同一份（记错强度 ⇒ **白装一遍** ✓✗）",
          alias.get("reused") is True, alias)

    #: ⚠️ 下面三条是同一个病的三个入口 ✓：**换了模型的一部分却复用旧模型** ✗
    stronger = _raises(lambda: _ask([{"path": item["path"], "strength": 0.8}]), "正在跑任务")
    check("复用门: 换强度 ⇒ **不复用**（0.5 → 0.8 必须重装 ✓✗）", stronger is not None, stronger)

    other = _raises(lambda: _ask([{"path": r"D:\m\other.safetensors", "strength": 0.5}]),
                    "正在跑任务")
    check("复用门: 换文件 ⇒ 不复用", other is not None, other)

    removed = _raises(lambda: _ask([]), "正在跑任务")
    check("复用门: 把 LoRA **摘掉** ⇒ 不复用（摘掉也是换模型 ✓✗）", removed is not None, removed)


def _error_of(call: Any) -> BaseException | None:
    """把异常对象本身拿出来 ✓（``_raises`` 只给文本 ✓，这里要比**类型** ✓）。"""
    try:
        call()
    except BaseException as err:  # noqa: BLE001 —— 就是来看它抛什么的 ✓
        return err
    return None


def _write_safetensors(path: Path, metadata: dict[str, str] | None) -> None:
    """造一个**最小** safetensors ✓（一个 F32、4 字节 ✓）—— 只为喂 ``read_tensors`` 的契约校验 ✓。"""
    header: dict[str, Any] = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header, ensure_ascii=False).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\x00\x00\x00\x00")


def case_training_gate() -> None:
    """训练产物契约门（E4.2 ✓ 2026-10-30 ✓）：**自称**训练产物的文件，元数据不全 ⇒ 读取时拒 ✓。

    ⚠️ 盯的三件事，缺一条这条判据就成摆设 ✓✗：

    * **真拦**：``format=training-v1`` 但缺 ``train_config``/``version`` ⇒ 抛 ``LoraError`` ✓
      （只报不拦＝把坏权重挂上去，画面只表现为"效果不对" ✓✗）；
    * **排位对**：校验必须在 ``import torch`` **之前** ✓ —— 不然"元数据不全"这种纯 stdlib 能判的事
      要先装几 GB 的 torch 才发现 ✓✗，而本仓自检大多零 torch ⇒ 这条根本测不到 ✓✗；
    * **不误伤**：普通 LoRA（``__metadata__`` 里没有 ``metadata`` 契约键 ✓）与三字段齐全的训练产物
      都必须**照旧放行** ✓✗（误伤的症状是「装不上」，比不拦更难查 ✓✗）。
    """
    import tempfile  # noqa: PLC0415 —— 只在造夹具时用 ✓

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        bad = root / "bad.safetensors"
        _write_safetensors(bad, {"metadata": json.dumps(
            {"format": "training-v1", "base_model": "minimax_h3_fl2va_pruned_int8.safetensors"})})
        err = _error_of(lambda: lora_mod.read_tensors(bad))
        check("契约门: 自称训练产物却缺 train_config/version ⇒ 当场拒 ✓（不补默认值 ✓✗）",
              isinstance(err, lora_mod.LoraError) and "元数据不完整" in str(err), err)
        check("契约门: 拒绝发生在碰 torch **之前** ✓（否则零 torch 自检测不到这条 ✗✗）",
              isinstance(err, lora_mod.LoraError) and "train_config" in str(err), err)

        good = root / "good.safetensors"
        _write_safetensors(good, {"metadata": json.dumps({
            "format": "training-v1", "base_model": "minimax_h3_fl2va_pruned_int8.safetensors",
            "train_config": {"rank": 128, "lr": 1e-4}, "version": "1.0.0"})})
        got = _error_of(lambda: lora_mod.read_tensors(good))
        check("契约门: 三字段齐 ⇒ 契约层**放行** ✓（后续若无 torch 只会是 ImportError ✗ 不是元数据错 ✗）",
              not (isinstance(got, lora_mod.LoraError) and "元数据" in str(got)), got)

        plain = root / "plain.safetensors"
        _write_safetensors(plain, {"format": "pt", "ss_network_dim": "128"})
        got_plain = _error_of(lambda: lora_mod.read_tensors(plain))
        check("契约门: 普通 LoRA（无 ``metadata`` 契约键）**照旧放行** ✓✗ —— 三字段要求不适用 ✓",
              not (isinstance(got_plain, lora_mod.LoraError) and "训练产物" in str(got_plain)), got_plain)


class Flat:
    """只有 ``reshape`` 的桩 ✓（DoRA 的 ``dora_scale`` 判「是不是全 1」要用 ✓）。"""

    def __init__(self, items: list[float]) -> None:
        self._items = list(items)

    def reshape(self, *shape: Any) -> list[float]:  # noqa: ARG002 —— 只服务 -1 拉平 ✓
        return self._items


def case_reference_parity() -> None:
    """对照参考实现（kohya / diffusers / ComfyUI）补齐的口径 ✓（2026-10-31 ✓）。"""
    zero_alpha = lora_mod.parse_lora({
        "m.lora_A.weight": DOWN, "m.lora_B.weight": UP, "m.alpha": 0.0})
    check("对照: alpha=0 ⇒ scale=1.0（kohya/diffusers 按 rank 抬 ✓；照字面算 = 静默乘 0 ✗✗）",
          abs(zero_alpha.deltas["m"].scale() - 1.0) < 1e-9, zero_alpha.deltas["m"].scale())

    dora = lora_mod.parse_lora({
        "m.lora_A.weight": DOWN, "m.lora_B.weight": UP,
        "m.dora_scale": Flat([2.0, 3.0, 4.0])})
    check("对照: DoRA（dora_scale 非全 1）⇒ 当场拒（只装 A/B 数值必错 ✗✗）",
          bool(dora.error) and "DoRA" in dora.error, dora.to_dict())

    identity = lora_mod.parse_lora({
        "m.lora_A.weight": DOWN, "m.lora_B.weight": UP,
        "m.dora_scale": Flat([1.0, 1.0, 1.0])})
    check("对照: DoRA 但 dora_scale 全 1 ⇒ 等价「没有 DoRA」，放行（不白拦 ✗）",
          identity.ok and list(identity.deltas) == ["m"], identity.to_dict())

    loha = lora_mod.parse_lora({"m.hada_w1_a": DOWN, "m.hada_w2_a": UP})
    check("对照: LoHa（hada_*）⇒ 当场拒、点明族名（静默少装 = 数值悄悄不对 ✗✗）",
          bool(loha.error) and "LoHa" in loha.error, loha.to_dict())

    te = lora_mod.apply_to_model(
        FakeModel({"blocks.0.mlp.fc1": FakeLinear(WEIGHT)}),
        lora_mod.LoraReport(deltas={
            "blocks.0.mlp.fc1": lora_mod.LoraDelta(module="blocks.0.mlp.fc1", down=DOWN, up=UP),
            "lora_te_text_model_encoder_layers_0_mlp_fc1": lora_mod.LoraDelta(
                module="lora_te_text_model_encoder_layers_0_mlp_fc1", down=DOWN, up=UP)}),
        ops=OPS)
    check("对照: 文本编码器层 ⇒ 记 skipped、**不算部分生效**（DiT 里本来没有它 ✗）",
          te.ok and len(te.skipped) == 1 and te.applied == 1, te.to_dict())

    partial = lora_mod.apply_to_model(
        FakeModel({"blocks.0.mlp.fc1": FakeLinear(WEIGHT)}),
        lora_mod.LoraReport(deltas={
            "blocks.0.mlp.fc1": lora_mod.LoraDelta(module="blocks.0.mlp.fc1", down=DOWN, up=UP),
            "blocks.7.mlp.fc1": lora_mod.LoraDelta(module="blocks.7.mlp.fc1", down=DOWN, up=UP)}),
        ops=OPS)
    check("对照: 部分层对不上 ⇒ 报「没有完整生效」（半装的 LoRA 只会像但不对 ✗✗）",
          bool(partial.error) and partial.applied == 1, partial.to_dict())

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        skip("因子对齐（本环境没装 torch）")
        return
    ref = torch.zeros(3, 3, dtype=torch.float64)
    aligned = lora_mod._align_delta(
        lora_mod.LoraDelta(module="m", down=torch.zeros(1, 2), up=torch.zeros(3, 1)), ref)
    check("对照: 因子的 dtype/设备跟目标层走（LoRA 存 fp32、底模 bf16 ⇒ 不搬会崩 ✗✗）",
          aligned.down.dtype == torch.float64
          and aligned.down.device == ref.device, (aligned.down.dtype, aligned.down.device))


def case_torch_ops_optional() -> None:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        skip("torch_ops 真机验证（本环境没装 torch：模块级零依赖 ✓ 已在上面全部验过 ✓）")
        return
    torch = sys.modules["torch"]
    ops = lora_mod.torch_ops()
    weight = torch.tensor(WEIGHT.rows)
    down = torch.tensor(DOWN.rows)
    up = torch.tensor(UP.rows)
    delta = lora_mod.LoraDelta(module="m", down=down, up=up, alpha=4.0)
    got = lora_mod.fold_weight(weight, delta, ops=ops)
    check("torch: fold 与假算子给出同一个数",
          bool(torch.allclose(got, torch.tensor(ref_fold(WEIGHT, DOWN, UP, 2.0).rows))),
          got.tolist())


def main() -> int:
    for label, case in (("解析", case_parse), ("边界", case_orphan_and_error),
                        ("选路", case_paths), ("数学", case_math),
                        ("装配", case_apply_model), ("kohya 键", case_kohya_names),
                   ("真文件", case_real_file),
                        ("接线", case_wiring), ("口径", case_coerce_spec),
                        ("复用门", case_reuse_gate),
                        ("契约门", case_training_gate),
                        ("对照参考", case_reference_parity),
                        ("torch", case_torch_ops_optional)):
        try:
            case()
        except Exception as err:  # noqa: BLE001 —— 用例自己崩了也要计红 ✓
            check(f"{label}: 用例抛异常", False, f"{type(err).__name__}: {err}")

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
