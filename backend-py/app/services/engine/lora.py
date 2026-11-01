"""**LoRA 装配**（自己实现 ✓）—— 从参考实现的语义搬过来，**不搬它的依赖** ✓。

## 为什么单独一层

参考实现里 LoRA 不是"读文件 + 加权"这么简单，它有三条**必须分开**的路 ✓，
选错不会报错、只会**悄悄变糊或悄悄丢掉** ✗（这两条正是参考实现踩出来的 ✓）：

1. **bypass（运行期加）** —— ``out = base(x) + up(down(x))·scale`` ✓。增量不折进权重 ✓：
   量化底模（bf16/fp8/int8）上把增量 merge 回去，会被**四舍五入/反量化吃干** ✗
   （参考原话：merging rounds it away in bf16 and requantizes it away in int8/fp8 ✓）；
2. **merge（折进权重）** —— ``W ← W + up@down·scale`` ✓。显存最省 ✓，但量化底模上会变软 ✗；
3. **runtime（连 merge/bypass 都做不了）** —— ``adaln_proj`` 这类：参考的 pruned 底模把
   ``silu(t_emb)`` 的 2688 维塌成了一小张曲线表 ✓ ⇒ LoRA 的 adaln 增量活在**已经不存在的
   空间**里 ✗，只能每次 forward 现算现加 ✓（本模块只如实**标出来** ✓，注入在运行时层做 ✓）。

还有一条**静默失效**的坑（参考实现量过：50 个 block 的 fc2 hook **触发 0 次** ✓）：
int8 融合 matmul 直接吃 ``linear.weight``、**不经过模块 forward** ⇒ 装在 ``fc2.forward`` 上的
bypass 钩子永远不响 ✗ ⇒ 那层 LoRA 被丢掉而**不报错** ✗。⇒ :func:`choose_path` 对此**强制 merge** ✓。

## 零依赖

本模块不 import torch ✓：张量运算走注入的 ``ops``（:class:`TensorOps` ✓），
真机传 :func:`torch_ops` ✓，自检传一个 list 版假实现就能验算法 ✓（与 :mod:`sampler` 同一纪律 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

__all__ = [
    "LoraApplyReport",
    "LoraDelta",
    "LoraError",
    "LoraReport",
    "LoraSpec",
    "TensorOps",
    "apply_specs",
    "apply_to_model",
    "bypass_output",
    "choose_path",
    "coerce_spec",
    "fold_weight",
    "kohya_module_name",
    "parse_lora",
    "read_tensors",
    "torch_ops",
]

#: LoRA 文件里低秩因子的后缀 ✓（A=down ``[rank, in]``、B=up ``[out, rank]`` ✓）
_DOWN_SUFFIXES: tuple[str, ...] = (".lora_A.weight", ".lora_down.weight", ".lora_A.default.weight")
_UP_SUFFIXES: tuple[str, ...] = (".lora_B.weight", ".lora_up.weight", ".lora_B.default.weight")
_ALPHA_SUFFIXES: tuple[str, ...] = (".alpha", ".lora_alpha")

#: 权重文件名里可能出现的外层前缀 ✓（参考实现的 key_map 只认 ``diffusion_model.<m>.weight`` ✓）
_STRIP_PREFIXES: tuple[str, ...] = ("model.diffusion_model.", "diffusion_model.", "model.")

#: 文本编码器侧的键 ✓ —— ``apply_to_model`` 的 ``model`` 是 **DiT** ✓，TE 的增量本来就不该
#: 在这里对上 ✗。认出来才不算「部分生效」✗（判据见 :func:`is_text_encoder_key` ✓）。
_TEXT_ENCODER_PREFIXES: tuple[str, ...] = (
    "lora_te_", "lora_te1_", "lora_te2_", "te.", "te1.", "te2.", "text_encoders.",
    "conditioner.", "text_model.", "text_encoder.", "cond_stage_model.",
)

#: 本模块**不支持**的适配器族 ✓（2026-10-31 补 ✗）—— 认得出就**必须说出来** ✓✗：
#: 当普通 LoRA 装 ⇒ 少装一堆张量、而且**不报错** ⇒ 数值悄悄不对 ✓✗（本仓最恨的那类 ✓）。
_UNSUPPORTED_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".dora_scale", "DoRA"),
    (".lora_mid.weight", "LoHa"), (".hada_w1_a", "LoHa"), (".hada_w2_a", "LoHa"),
    (".hada_t1", "LoHa"),
    (".lokr_w1", "LoKr"), (".lokr_w2", "LoKr"), (".lokr_w1_a", "LoKr"), (".lokr_w2_a", "LoKr"),
    (".oft_blocks", "OFT"),
    (".lora_embedding_A", "LoRA-Embedding"), (".lora_embedding_B", "LoRA-Embedding"),
)

#: 各族**为什么不支持** ✓（报错文案要用 ✓）
_UNSUPPORTED_REASONS: dict[str, str] = {
    "DoRA": "幅度向量 `dora_scale` 与 A/B 因子是**乘性**关系 ⇒ 只装 A/B 会算错 ✗",
    "LoHa": "Hadamard 积分解 ⇒ 不是一对低秩因子，`lora_mid`/`hada_*` 一个都装不了 ✗",
    "LoKr": "Kronecker 分解 ⇒ 需要 `lokr_w1/w2` 两半，不是 A/B 对 ✗",
    "OFT": "正交微调 ⇒ 作用在正交矩阵上，与低秩 A/B 完全不同 ✗",
    "LoRA-Embedding": "作用在 embedding 表上（按 id 取行），不是 Linear 的权重 ✗",
}


#: kohya / sd-scripts 那种**下划线分层**的键，靠这张表切回点分层 ✓（2026-10-31 补 ✓）。
#: ⚠️ 为什么必须有 ✗：kohya 出的 SD1.5/SDXL/SD3 LoRA，键长成
#: ``lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q`` ✓ —— 整串**一个点都没有** ✓，
#: 而模型里的模块名是点分层 ✓ ⇒ ``modules.get()`` 必然 None ⇒ 报「一层都没装上」✗。
#: HuggingFace/diffusers 口径（``down_blocks.0.attentions.0.…``）本来就有 ⇒ 不受影响 ✓。
_UNDERSCORE_TOKENS: tuple[str, ...] = (
    "lora_unet", "lora_te1", "lora_te2", "lora_te",
    "down_blocks", "up_blocks", "mid_block", "input_blocks", "middle_block", "output_blocks",
    "single_transformer_blocks", "transformer_blocks", "joint_blocks",
    "attentions", "self_attn", "cross_attn", "attn1", "attn2",
    "to_add_out", "to_out", "to_q", "to_k", "to_v", "to_gate",
    "qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "gate_proj",
    "up_proj", "down_proj",
    "proj_in", "proj_out", "proj_mlp", "proj", "mlp",
    "in_layers", "out_layers", "emb_layers", "skip_connection",
    "norm_added", "norm_out", "norm1", "norm2", "norm3", "norm",
    "layer_norm1", "layer_norm2", "layer_norm3", "final_layer_norm", "layer_norm",
    "ff_context", "ff_net", "ff_in", "ff_out", "ff",
    "time_embedding", "time_embed", "add_embedding", "label_emb",
    "conv_shortcut", "conv_norm_out", "conv_in", "conv_out", "conv1", "conv2", "conv",
    "linear_1", "linear_2", "linear", "mid",
    "text_model", "token_embedding", "position_embedding", "text_projection",
    "processor", "encoder", "layers", "fc1", "fc2",
)

#: 切开之后**段名 → 段名**的改写 ✓（段本身可能含点 ⇒ 拼回去才是真模块名 ✓）。
#: ⚠️ 为什么光靠切不行 ✗：kohya 键里的下划线来自「模块名把点换成下划线」✓，模块名本身**可以含点**
#: 之后的复合段 ✓ —— SDXL 的 GEGLU 前馈是 ``ff.net.0.proj`` ✓ ⇒ kohya 键 ``ff_net_0_proj`` ✓，
#: 切成 ``ff_net.0.proj`` 就对不上（整份 LoRA 报「没装全」⇒ 能力等于没有 ✓✗）。
#: 判据：:func:`kohya_module_name` 必须对**真模型**里每个模块名无损往返 ✓
#: （`tests/engine_sdxl_backend_test.py::case_kohya_coverage` ✓）。
_UNDERSCORE_ALIASES: dict[str, str] = {
    "ff_net": "ff.net",
}

#: 键首那段是**模型归属**而不是模块名 ✓（``lora_unet_`` ⇒ UNet、``lora_te1_`` ⇒ 第一个文本塔 ✓）
_UNDERSCORE_ROOTS: tuple[str, ...] = ("lora_unet", "lora_te", "lora_te1", "lora_te2")


def _split_underscore(text: str) -> list[str]:
    """把一串下划线名切成**已知段 + 数字段** ✓（最长优先 ✓，认不出的整段原样留 ✓ 不猜 ✗）。"""
    tokens = sorted(_UNDERSCORE_TOKENS, key=len, reverse=True)
    parts: list[str] = []
    index = 0
    while index < len(text):
        for token in tokens:
            end = index + len(token)
            if text.startswith(token, index) and (end == len(text) or text[end] == "_"):
                parts.append(token)
                index = end + 1 if end < len(text) else end
                break
        else:
            cut = text.find("_", index)
            if cut < 0:
                parts.append(text[index:])
                index = len(text)
            else:
                parts.append(text[index:cut])
                index = cut + 1
    return [part for part in parts if part]


def kohya_module_name(key: str) -> str:
    """kohya 的下划线分层键 → **点分层**模块名 ✓（已经是点分层就原样返回 ✓）。

    ``lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q``
    ⇒ ``down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`` ✓

    ⚠️ 这是**翻译**不是**匹配** ✓：翻完还得 ``modules.get()`` 真的查到才算数 ✓ ——
    查不到仍按「找不到」报错 ✓✗（认不出的段原样留着 ✓，不去模糊匹配 ✗：模糊配上错层
    ⇒ 权重折进错的模块 ⇒ 图「像但不对」✓✗，比没装上更难查 ✓）。
    """
    if "." in key and not key.startswith(_UNDERSCORE_ROOTS):
        return key
    parts = _split_underscore(key)
    if parts and parts[0] in _UNDERSCORE_ROOTS:
        parts = parts[1:]
    #: 段名改写（见 :data:`_UNDERSCORE_ALIASES` ✓）—— 段本身可以含点 ✓，拼回去才是真模块名 ✓。
    parts = [_UNDERSCORE_ALIASES.get(part, part) for part in parts]
    return ".".join(parts) if parts else key


class LoraError(ValueError):
    """LoRA 文件形状/配对不自洽 ✓ —— **拒绝**而不是猜 ✗（猜出来的是"看起来能跑" ✗）。"""


class TensorOps(Protocol):
    """张量算子（鸭子类型 ✓）—— 只要这五个动作就能装配 LoRA ✓。"""

    def shape(self, tensor: Any) -> tuple[int, ...]: ...

    def matmul(self, left: Any, right: Any) -> Any: ...

    def linear(self, inputs: Any, weight: Any) -> Any:
        """``inputs @ weight.T``（``nn.Linear`` 的语义 ✓）。"""
        ...

    def add_scaled(self, base: Any, delta: Any, scale: float) -> Any:
        """``base + delta·scale`` ✓（真机上**原地**加，省一次全尺寸临时 ✓）。"""
        ...


@dataclass
class LoraDelta:
    """**一个模块**的低秩增量 ✓（``down`` 与 ``up`` 成对，配不上就进 ``orphans`` ✗）。"""

    module: str
    down: Any
    up: Any
    alpha: float | None = None

    def scale(self, strength: float = 1.0) -> float:
        """``alpha/rank · strength`` ✓（无 alpha 时按参考实现的约定取 1.0 ✓）。

        ⚠️ **``alpha == 0`` 是「不缩放」而不是「强度 0」** ✗✗（2026-10-31 修 ✓）：
        参考实现（kohya ``LoRAUpDown`` / diffusers ``set_adapters`` ✓）一律把 ``alpha == 0``
        抬成 ``alpha = rank`` ⇒ 缩放因子 **1.0** ✓。照字面算 ``0/rank = 0.0`` ⇒ 增量被乘成 0
        ✗ —— **不报错、不告警、LoRA 等于没挂** ✓✗（本仓最恨的那类静默失效 ✓）。
        """
        rank = int(self.down.shape[0]) if hasattr(self.down, "shape") else 0
        alpha = float(self.alpha) if self.alpha is not None else None
        if alpha is not None and alpha == 0.0:
            alpha = float(rank) if rank else None       # kohya 约定：alpha=0 ⇒ 取 rank ⇒ scale 1.0 ✓
        base = (alpha / rank) if (alpha is not None and rank) else 1.0
        return base * float(strength)


@dataclass
class LoraReport:
    """装配结果 ✓ —— **每一条不装的原因都要说清** ✗（不静默丢 ✓，同 ``WeightLoadReport`` 的纪律 ✓）。"""

    deltas: dict[str, LoraDelta] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)
    runtimeInject: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "modules": len(self.deltas),
            "runtimeInject": self.runtimeInject[:8],
            "runtimeInjectCount": len(self.runtimeInject),
            "orphans": self.orphans[:8],
            "orphanCount": len(self.orphans),
            "error": self.error,
        }


def _identity_scalar(value: Any) -> bool:
    """这个张量是不是**全 1** ✓（DoRA 里全 1 的 ``dora_scale`` 等价于「没有 DoRA」✓ ⇒ 不必拦 ✗）。"""
    try:
        flat = value.reshape(-1) if hasattr(value, "reshape") else value
        items = list(flat)
    except Exception:
        return False
    if not items:
        return False
    try:
        return all(abs(float(item) - 1.0) <= 1e-6 for item in items)
    except Exception:
        return False


def _unsupported_families(tensors: Mapping[str, Any]) -> dict[str, list[str]]:
    """扫出**本模块装不了**的适配器族 ✓（族名 → 命中的张量名 ✓）。

    唯一豁免：DoRA 的 ``dora_scale`` **全是 1.0** ✓ —— 那在语义上就是「没有 DoRA」✓，
    拦下来只会让人白改文件 ✗（判据 :func:`_identity_scalar` ✓）。
    """
    found: dict[str, list[str]] = {}
    for name in tensors:
        for suffix, family in _UNSUPPORTED_SUFFIXES:
            if not name.endswith(suffix):
                continue
            if family == "DoRA" and _identity_scalar(tensors[name]):
                break
            found.setdefault(family, []).append(name)
            break
    return found


def is_text_encoder_key(module: str) -> bool:
    """这个 LoRA 模块名是不是**文本编码器**侧的 ✓（DiT 上本就找不到 ✓ ⇒ 不该算「部分生效」✗）。"""
    return str(module).startswith(_TEXT_ENCODER_PREFIXES)


def _align_delta(delta: LoraDelta, reference: Any) -> LoraDelta:
    """把两个因子搬到目标模块的**设备与精度**上 ✓（2026-10-31 修 ✗✗）。

    ⚠️ 为什么必须有 ✗：:func:`read_tensors` 用 ``torch.frombuffer`` 出的是 **CPU** 张量 ✓，
    而模块权重在卡上 ✓ ⇒ merge 路的 ``matmul(up, down)`` 在 CPU 算完、``add_`` 到卡上立刻
    ``Expected all tensors to be on the same device`` ✗；bypass 路更隐蔽 —— ``F.linear`` 在
    **首次前向**才炸 ✗（装载报告全绿 ✓✗）。``.to()`` 目标一致时返回自身 ✓ ⇒ 同设备同精度零拷贝 ✓，
    缓存里的 CPU 主本也不会被改 ✗。
    """
    like = reference
    if like is None or not hasattr(like, "dtype"):
        return delta
    device = getattr(like, "device", None)
    dtype = like.dtype
    moved: list[Any] = []
    for value in (delta.down, delta.up):
        move = getattr(value, "to", None)
        if move is None:
            moved.append(value)
            continue
        try:
            moved.append(move(device=device, dtype=dtype))
        except Exception:                       # 纯 CPU 假张量（自检）⇒ 只搬设备，不改精度 ✓
            moved.append(move(device=device) if device is not None else value)
    if moved[0] is delta.down and moved[1] is delta.up:
        return delta
    return replace(delta, down=moved[0], up=moved[1])


def _normalize(module: str) -> str:
    """去掉外层前缀 ✓（``diffusion_model.blocks.0.mlp.fc1`` → ``blocks.0.mlp.fc1`` ✓）。"""
    for prefix in _STRIP_PREFIXES:
        if module.startswith(prefix):
            return module[len(prefix):]
    return module


def parse_lora(tensors: Mapping[str, Any]) -> LoraReport:
    """把 LoRA 文件的张量表拆成 ``{模块名: LoraDelta}`` ✓（**不装载、不碰模型** ✓）。

    ⚠️ 三种"配不上"都要报出来，不能悄悄少装 ✗：
    只有 A 没有 B（或反之）⇒ ``orphans`` ✓；A/B 的 rank 不一致 ⇒ ``LoraError`` ✓；
    ``adaln_proj`` ⇒ ``runtimeInject`` ✓（它在 pruned 底模上折不进权重 ✓）。

    ⚠️ **别的适配器族**（DoRA / LoHa / LoKr / OFT / LoRA-Embedding ✓）⇒ ``error`` ✗：
    它们的分辨不了成一对低秩因子 ✓，当普通 LoRA 装等于**少装 + 不报错** ✓✗。
    """
    report = LoraReport()
    unsupported = _unsupported_families(tensors)
    if unsupported:
        report.error = "；".join(
            f"这是 {family} 适配器 ✗（{len(names)} 个张量，如 `{names[0]}` ✓）：{_UNSUPPORTED_REASONS[family]}"
            for family, names in unsupported.items())
        return report
    downs: dict[str, Any] = {}
    ups: dict[str, Any] = {}
    alphas: dict[str, float] = {}

    for name in tensors:
        for suffix in _DOWN_SUFFIXES:
            if name.endswith(suffix):
                downs[_normalize(name[: -len(suffix)])] = tensors[name]
                break
        else:
            for suffix in _UP_SUFFIXES:
                if name.endswith(suffix):
                    ups[_normalize(name[: -len(suffix)])] = tensors[name]
                    break
            else:
                for suffix in _ALPHA_SUFFIXES:
                    if name.endswith(suffix):
                        value = tensors[name]
                        try:
                            alphas[_normalize(name[: -len(suffix)])] = float(value)
                        except (TypeError, ValueError) as exc:  # pragma: no cover - 异常文件
                            report.error = f"{name} 的 alpha 不是标量（{exc}）"
                        break

    for module in sorted(set(downs) | set(ups)):
        down, up = downs.get(module), ups.get(module)
        if down is None or up is None:
            report.orphans.append(f"{module}（只有{'B' if down is None else 'A'}）")
            continue
        down_shape = tuple(getattr(down, "shape", ()) or ())
        up_shape = tuple(getattr(up, "shape", ()) or ())
        if len(down_shape) < 2 or len(up_shape) < 2:
            report.orphans.append(f"{module}（形状信息不全，无法确认 rank）")
            continue
        if int(down_shape[0]) != int(up_shape[-1]):
            raise LoraError(
                f"{module} 的 rank 对不上：down{down_shape} vs up{up_shape} ⇒ "
                f"{down_shape[0]} ≠ {up_shape[-1]} ✗（装上去会静默算错 ✗）")
        delta = LoraDelta(module=module, down=down, up=up, alpha=alphas.get(module))
        if ".adaln_proj" in module:
            # 折不进权重 ✓（底模的 silu(t_emb) 空间已被塌成曲线表 ✓）⇒ 由运行时逐 forward 注入 ✓
            report.runtimeInject.append(module)
        report.deltas[module] = delta
    return report


def choose_path(module: str, *, layout: str = "", transposed: bool = False) -> tuple[str, str]:
    """这个模块该走 ``"bypass"`` / ``"merge"`` / ``"runtime"`` ✓ —— 连**理由**一起给 ✓。

    规则全部来自参考实现的实测结论（不是偏好 ✓）：
    * ``adaln_proj`` ⇒ ``runtime`` ✓（空间已被 pruned 底模塌掉 ✓）；
    * int8 融合的 ``mlp.fc2`` ⇒ ``merge`` ✓（融合 kernel 绕过 forward，bypass 钩子**不触发** ✓，
      参考实现量到 50 层钩子触发 **0 次** ✓）；
    * 其余 ⇒ ``bypass`` ✓（量化底模上 merge 会把增量吃掉 ✓）。
    """
    if ".adaln_proj" in module:
        return "runtime", "adaln 增量活在 silu(t_emb) 的 2688 维空间里，pruned 底模已塌成曲线表"
    if module.endswith(".mlp.fc2") and layout.startswith("TensorWiseINT8") and not transposed:
        return "merge", "int8 融合 matmul 直接吃 linear.weight，不经过 forward ⇒ bypass 钩子不触发、LoRA 会静默失效"
    return "bypass", "量化底模上 merge 会把低秩增量四舍五入/反量化掉，运行期加才保得住"


def fold_weight(weight: Any, delta: LoraDelta, strength: float = 1.0,
                *, ops: TensorOps) -> Any:
    """**merge 路**：``W ← W + up@down·scale`` ✓（显存最省，量化底模上会变软 ✗）。"""
    update = ops.matmul(delta.up, delta.down)
    return ops.add_scaled(weight, update, delta.scale(strength))


def bypass_output(base_out: Any, inputs: Any, delta: LoraDelta, strength: float = 1.0,
                  *, ops: TensorOps) -> Any:
    """**bypass 路**：``base(x) + up(down(x))·scale`` ✓（原地加到 ``base_out`` ✓）。

    ``base_out`` 必须是模块**刚刚算出来的**输出（没有别的引用 ✓），否则原地加会踩到别人 ✗。
    """
    update = ops.linear(ops.linear(inputs, delta.down), delta.up)
    return ops.add_scaled(base_out, update, delta.scale(strength))


def torch_ops() -> TensorOps:
    """真机的算子实现 ✓（**函数内** import torch ✓ —— 模块级 import 会让无 torch 环境起不来 ✗）。"""
    import torch  # noqa: PLC0415

    class _TorchOps:
        def shape(self, tensor: Any) -> tuple[int, ...]:
            return tuple(int(size) for size in tensor.shape)

        def matmul(self, left: Any, right: Any) -> Any:
            return left @ right

        def linear(self, inputs: Any, weight: Any) -> Any:
            # ``F.linear`` 语义 ✓：x @ W.T ✓（LoRA 的 down/up 在盘上就是 nn.Linear 布局 ✓）
            return torch.nn.functional.linear(inputs.to(weight.dtype), weight)

        def add_scaled(self, base: Any, delta: Any, scale: float) -> Any:
            with torch.no_grad():
                target = delta.to(base.dtype)
                if hasattr(base, "add_") and base.is_floating_point():
                    return base.add_(target, alpha=float(scale))
                return base + target * float(scale)

    return _TorchOps()  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# 「挂哪个文件、多大强度」→ 真的装到模型上
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LoraSpec:
    """一条 LoRA 挂载 ✓（**路径必须是绝对路径** ✗ 相对路径换个 cwd 就静默指到别处 ✗）。"""

    path: str
    strength: float = 1.0
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or Path(self.path).stem

    def to_dict(self) -> dict[str, Any]:
        #: ⚠️ **回显专用** ✗✗：这里把 strength 抹到 4 位小数 ✓，而装载键（`runtime._lora_key` ✓）
        #:    比的是**原值** ✓ ⇒ **不许**拿它的结果去拼 ``loras`` 喂 ``ensure_loaded`` ✗：同一个
        #:    LoRA 会算出两个键 ⇒ 换强度不重装 + 报告里的强度与实挂不是一个数（本仓最恨的不对称 ✓）。
        #:    装配那条路要用 `bridge.lora_assembly`（原值、不取整 ✓，判据在 `engine_lora_test` ✓）。
        return {"path": self.path, "strength": round(float(self.strength), 4),
                "name": self.label}


def coerce_spec(item: Any) -> LoraSpec:
    """一条 LoRA 写法 ⇒ :class:`LoraSpec` ✓（``LoraSpec`` 本身 / 路径串 / ``{path, strength, name}`` ✓）。

    ⚠️ **``strength`` 的口径只有这一处** ✗✗（2026-10-29 ✓）：缺省或显式 ``None`` ⇒ **1.0** ✓；
    **显式 ``0.0`` 就是 ``0.0``** ✓✗ —— 不许抬成 1.0 ✗。
    为什么单独立这一处 ✓：此前同一件事写了三遍（``bridge.lora_specs_from_settings`` ✓、
    ``runtime._lora_key`` ✓、``torch_backend._apply_loras`` ✓），三处口径还**各不相同** ✓✗ ——
    其中 ``float(item.get("strength", 1.0) or 1.0)`` 会把 ``0`` 抬成 ``1.0`` ✓✗，而装载键按
    ``0.0`` 记 ✓ ⇒ **「键说 0.0、实际按 1.0 挂」** ✓✗ ⇒ 换强度不重装、报了强度也是假的 ✓✗
    （正是本仓最恨的那类不对称 ✓）。

    ⚠️ 只管**写法**，不解析路径 ✓✗：相对路径按谁拼（``modelsDir`` ✓）、文件在不在 ✓，
    是调用方各自的事 ✓（桥要拼相对路径 ✓，后端拿到手就该是绝对的 ✓）。
    """
    if isinstance(item, LoraSpec):
        return item
    if isinstance(item, str):
        path, strength, name = item, 1.0, ""
    elif isinstance(item, Mapping):
        path = str(item.get("path") or "").strip()
        raw = item.get("strength", item.get("strength_model", 1.0))
        strength = 1.0 if raw is None else float(raw)
        name = str(item.get("name") or "")
    else:
        raise LoraError(
            f"LoRA 条目写法不认识 ✗：{item!r} ✓（要路径字符串、``{{path, strength}}`` 或 "
            f"``LoraSpec`` ✓）")
    if not path:
        raise LoraError(
            f"LoRA 条目缺 ``path`` ✗：{dict(item) if isinstance(item, Mapping) else item!r}")
    return LoraSpec(path=path, strength=float(strength), name=name)


@dataclass
class LoraApplyReport:
    """装完的账 ✓ —— **每一层走哪条路、哪层没找到、哪层没生效**都要留下 ✓。"""

    merged: list[str] = field(default_factory=list)
    bypassed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    runtime: list[str] = field(default_factory=list)
    hooks: list[Any] = field(default_factory=list)
    #: ``missing`` 里**本来就该对不上**的那些 ✓（文本编码器层 ✓）—— 文案照进 ``missing`` ✓（可见 ✓），
    #: 但不参与「部分生效」判定 ✗。⚠️ **不进 ``to_dict``** ✗：``docs/engine-api.schema.json`` 里
    #: ``lora`` 的键集是逐字快照 ✓，加键＝破坏性改动 ✓（要加得连着环境文档一起改 ✓）。
    skipped: list[str] = field(default_factory=list)
    #: 靠 :func:`kohya_module_name` **翻译**才对上的那些层 ✓（键是下划线分层 ✓）。
    #: ⚠️ 同样**不进 ``to_dict``** ✗（键集是 schema 逐字快照 ✓）；它只用来回答
    #: 「这个 LoRA 是靠翻译才装上的吗」✓ —— 全 0 说明文件本来就是点分层 ✓。
    translated: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def applied(self) -> int:
        return len(self.merged) + len(self.bypassed)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def unapplied(self) -> int:
        """**真该装却没装上**的层数 ✓（扣掉文本编码器那些本来就不归这里管的 ✓）。"""
        return max(len(self.missing) - len(self.skipped), 0)

    def absorb(self, other: "LoraApplyReport") -> None:
        self.merged.extend(other.merged)
        self.bypassed.extend(other.bypassed)
        self.missing.extend(other.missing)
        self.runtime.extend(other.runtime)
        self.hooks.extend(other.hooks)
        self.skipped.extend(other.skipped)
        self.translated.extend(other.translated)
        if other.error and not self.error:
            self.error = other.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "merged": len(self.merged),
            "bypassed": len(self.bypassed),
            "missing": self.missing[:8],
            "missingCount": len(self.missing),
            "runtime": self.runtime[:8],
            "runtimeCount": len(self.runtime),
            "error": self.error,
        }


#: safetensors 里的 dtype 名 → torch 属性名 ✓（因子只可能是浮点 ✓，量化格式在这里是错文件 ✗）
_TORCH_DTYPES: dict[str, str] = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
}


def read_tensors(path: str | Path, *, torch_module: Any = None) -> dict[str, Any]:
    """读一个 LoRA 文件的**全部张量** ✓（自研 safetensors 读取器 ✓ 不装 ``safetensors`` 包 ✓）。

    ⚠️ 逐张量读、按头部自述的字节数封顶 ✓ —— 拿错文件（比如 19 GiB 的整权重）会**当场拒绝** ✗
    而不是把机器拖死 ✗。

    ⚠️ **自称训练产物**的文件（``format`` 以 ``training`` 开头）必须先过
    :func:`checkpoint_meta.read_training_contract` ✓：缺 ``base_model`` / ``train_config`` /
    ``version`` 任一 ⇒ 当场拒 ✗（E4.2：不补默认值 ✓✗）。校验放在 ``import torch`` **之前** ✓ ——
    否则「元数据不全」这种事也得先装 torch 才能发现 ✓✗，零 torch 的自检就测不到 ✓。
    """
    from . import checkpoint_meta as meta_mod  # noqa: PLC0415 —— 纯 stdlib ✓
    from . import safetensors as st  # noqa: PLC0415 —— 纯 stdlib ✓

    metadata, header = st.read_header(path)
    if meta_mod.is_training_contract(metadata):
        try:
            meta_mod.read_training_contract(metadata)
        except meta_mod.ContractError as err:
            raise LoraError(
                f"{Path(path).name} 自述是训练产物，但元数据不完整 ✗：{err}") from err

    import torch  # noqa: PLC0415 —— 模块级零依赖 ✓

    torch = torch_module or torch
    tensors: dict[str, Any] = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        dtype = str(raw.get("dtype", "")).upper()
        attribute = _TORCH_DTYPES.get(dtype)
        if attribute is None:
            raise LoraError(
                f"{name} 的 dtype 是 {dtype or '?'} ✗ —— LoRA 因子必须是浮点（F32/F16/BF16/F64 ✓）")
        shape = tuple(int(size) for size in raw.get("shape") or ())
        count = 1
        for size in shape:
            count *= size
        expected = count * st.DTYPE_ITEMSIZE.get(dtype, 0)
        payload = st.read_tensor_bytes(path, name, max_bytes=expected)
        flat = torch.frombuffer(bytearray(payload), dtype=getattr(torch, attribute))
        tensors[name] = flat.reshape(shape) if shape else flat.reshape(())
    return tensors


def _bypass_hook(_module: Any, inputs: Any, output: Any, *, delta: LoraDelta,
                 strength: float, ops: TensorOps) -> Any:
    """``nn.Linear`` 的前向钩子 ✓ —— ``out + up(down(x))·scale`` ✓（in-place 加到刚算出的 out ✓）。"""
    source = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    return bypass_output(output, source, delta, strength, ops=ops)


def apply_to_model(model: Any, report: LoraReport, *, strength: float = 1.0,
                   layout: str = "", ops: TensorOps | None = None,
                   transform: Callable[[str], str] | None = None) -> LoraApplyReport:
    """把 :func:`parse_lora` 的结果装进 ``model`` ✓（``nn.Module`` ✓，按模块名逐层对上 ✓）。

    ``transform``：模块名 → 模型里的模块名 ✓（默认恒等 ✓ —— H3 的 LoRA 键与
    ``h3_form`` 的模块路径**逐字一致** ✓，实测 208 个模块一一对上 ✓）。

    **不做的事**：``runtime``（adaln）这一路不装 ✗ —— 只如实记进 ``runtime`` 并写 ``error`` ✓，
    让调用方看见「这层没生效」✓ 而不是以为装好了 ✗。
    """
    ops = ops or torch_ops()
    modules = dict(model.named_modules())
    result = LoraApplyReport()
    for name, delta in report.deltas.items():
        target = transform(name) if transform else name
        #: 名字被**改写**过 ✓（不是恒等 ✓）⇒ 与下面 kohya 兜底同属「翻译」✓，找到模块后一并记账 ✓。
        #: ⚠️ 不记的后果：全走 ``transform`` 的塔（SDXL 的 TE 层 ✓）``translated`` 恒 0 ✓✗，
        #:    报告读起来像「一个键都没翻译」✓✗ —— 与实情相反 ✓。
        renamed = transform is not None and target != name
        path, reason = choose_path(name, layout=layout)
        module = modules.get(target)
        if module is None and transform is None:
            #: ⭐ 2026-10-31 补 ✓：kohya / sd-scripts 的 SD1.5·SDXL·SD3 LoRA 键是**下划线分层** ✓
            #: （整串无点 ✓），模型里是点分层 ⇒ 不翻译就「一层都没装上」✗。只在不传 ``transform``
            #: 时兜底 ✓ —— 传了 ``transform`` 说明调用方**自己**有映射口径 ✓，这里不许叠加猜测 ✗✗。
            translated = kohya_module_name(name)
            if translated != name:
                found = modules.get(translated)
                if found is not None:
                    module, target = found, translated
                    result.translated.append(name)
        if module is None:
            if is_text_encoder_key(name):
                # 文本编码器侧的增量 ✓ —— 传进来的 `model` 是 DiT ✗ ⇒ 本来就不该在这里对上 ✓。
                # 文案仍写进 `missing` ✓（**可见** ✓），但不参与「部分生效」判定 ✗。
                result.skipped.append(f"{name}（文本编码器层 ⇒ 本后端只装 DiT ✗）")
                result.missing.append(result.skipped[-1])
            else:
                result.missing.append(f"{name}（模型里没有 {target} ✗）")
            continue
        if renamed:
            result.translated.append(name)
        if path == "runtime":
            result.runtime.append(f"{name}（{reason} ✗）")
            continue
        # ⚠️ 因子必须**先搬到目标层**再做运算 ✗✗（显存路必崩 ✓，见 :func:`_align_delta` ✓）
        weight = getattr(module, "weight", None)
        reference = weight if weight is not None else next(iter(module.parameters(recurse=False)), None)
        delta = _align_delta(delta, reference)
        if path == "merge":
            if weight is None:
                result.missing.append(f"{name}（要 merge 但 {target} 没有 weight ✗）")
                continue
            weight.data = fold_weight(weight.data, delta, strength, ops=ops)
            result.merged.append(name)
            continue
        result.hooks.append(module.register_forward_hook(
            partial(_bypass_hook, delta=delta, strength=strength, ops=ops)))
        result.bypassed.append(name)
    reasons: list[str] = []
    if result.runtime:
        reasons.append(
            f"{len(result.runtime)} 层 adaln 增量没有装 ✗ —— 它们活在 silu(t_emb) 的塌缩空间里，"
            f"折不进权重也钩不到 forward ✓；真要支持得在 AdalnProj 里逐 forward 注入 ✓。"
            f"（盘上现有的 minimax_h3 turbo LoRA 实测 adaln 键为 **0** ✓ ⇒ 正常不会撞到这条 ✓）")
    if result.unapplied:
        # ⚠️ **部分生效 = 静默错值** ✗✗（2026-10-31 补 ✓）：只装上一半的 LoRA 不报错，
        #    画面/文本只会「像但不对」✓✗ —— 比一层都没装上更难查 ✓。
        #    一层都没装上通常是**命名口径**问题（kohya 原生 `lora_unet_down_blocks_0_…` 用下划线分层 ✓，
        #    模型里是 `down_blocks.0.…` ✓），必须当场说清 ✗。
        head = (f"{result.unapplied} 个模块在 DiT 里找不到 ✗"
                if result.applied == 0 else
                f"只装上 {result.applied} 层、另有 {result.unapplied} 层找不到 ✗")
        reasons.append(
            f"{head} ⇒ 这个 LoRA **没有完整生效** ✗（如 {result.missing[0]} ✓）。"
            f"本层认「点分层」（`down_blocks.0.…` ✓）与 kohya 下划线分层"
            f"（`lora_unet_down_blocks_0_…` ⇒ 自动翻译 ✓，见 `kohya_module_name` ✓）；"
            f"LoHa/LoKr/DoRA/OFT 这些**分解族**不认 ✗（见 `_UNSUPPORTED_REASONS` ✓）")
    if reasons:
        result.error = "；".join(reasons)
    return result


def apply_specs(model: Any, specs: Sequence[LoraSpec], *, layout: str = "",
                ops: TensorOps | None = None, reader: Callable[[str], Mapping[str, Any]] | None = None,
                transform: Callable[[str], str] | None = None) -> LoraApplyReport:
    """按顺序挂多个 LoRA ✓（后挂的叠在前面之上 ✓，与 ComfyUI 的链式行为一致 ✓）。

    ``reader`` 可注入 ✓（自检用假张量表 ✓，真机默认走 :func:`read_tensors` ✓）。
    """
    load = reader or read_tensors
    total = LoraApplyReport()
    for spec in specs:
        parsed = parse_lora(load(spec.path))
        if not parsed.ok:
            total.error = f"{spec.label}：{parsed.error}"
            continue
        total.absorb(apply_to_model(model, parsed, strength=spec.strength, layout=layout,
                                    ops=ops, transform=transform))
    return total
