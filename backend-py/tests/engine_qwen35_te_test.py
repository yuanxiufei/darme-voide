"""Qwen-Image **2.1** 的文本塔（Qwen3.5-9B 混合线性注意力）自检 —— 不需要显卡。

四层判据 ✓：

① **形状反推 ＋ 键对账**（读盘上真权重头 ✓）：``infer_shape`` 必须读出 4096/32/12288/16/4/256/64、
   24 线性 ＋ 8 完整、``layer_idx=-2 ⇒ 第 30 层`` ✓；``expected_keys`` 必须与文件里的语言塔键
   **逐字**相同 ✓（多一个报多余键 ✓、少一个静默漏参数 ✓✗）。
② **数值对拍**：小形状 fp32 塔 与 :func:`_ref_layer`（**另写一份**逐 token 递推的朴素实现 ✓，
   与本实现不共享执行路径 ✓）比数值 ✓ —— 同错不会同绿 ✓。
③ **反向验证**：把八处最容易错的口径各改坏一次 ✓，断言「坏法」与主实现的差 ≫「正确参照」
   与主实现的差 ✓ ⇒ 证明 ② 的分辨力真够 ✓。
④ **取层语义**：``layer_idx=-2`` 取倒数第 2 层之后的 hidden ✓、**不过** final norm ✓、batch>1 拒 ✓。
⑤ **后端接线**：``TorchBackend.attach_qwen35_te`` 装得上 ✓（``loadedKeys`` 逐字 = 塔的键数 ✓）、
   ``unexpected`` 里的视觉塔 / ``lm_head`` **单列**且不装 ✓、``textKind`` 说真话 ✓、
   ``output_dim ↔ text_dim`` 不符**先拒后不改** ✓、缺键 ⇒ **拒装**（不留随机层 ✓✗）。

⚠️ 反向验证的阈值**不许**写死绝对值 ✗：先量「没改坏时与参照的差」（float32 下 ~1e-7 ✓），
再要求坏法的差至少是它的 1e3 倍 ✓（别的族上吃过这个亏 ✓）。
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.services.engine import qwen35_te as q  # noqa: E402
from app.services.engine import quant, safetensors as st  # noqa: E402

FAILS: list[str] = []

#: 盘上真文件 ✓（不在就跳过 ①，②③④ 照跑 ✓）
REAL = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/text_encoders/"
            "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors")


def check(name: str, ok: object, detail: object = None) -> None:
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        FAILS.append(name)
        if detail is not None:
            print(f"      {detail}")


# --------------------------------------------------------------- 小形状塔


def small_shapes() -> dict[str, tuple[int, ...]]:
    """两层小塔 ✓：0 = 线性注意力 ✓、1 = 完整注意力 ✓（够验两种层型 ＋ 取层 ✓）。

    ⚠️ 刻意让 ``value_heads(4) != key_heads(2)`` ✓ —— 头复制那一步（``repeat_interleave`` ✓）
    只有不相等时才有效果 ✓；相等时 ``repeat`` 与它结果相同 ✓ ⇒ 那个坑**测不出来** ✗。
    ⚠️ ``head_dim=16`` ⇒ partial RoPE 4 维 ⇒ split-half 是 ``(0,2)(1,3)``、相邻是 ``(0,1)(2,3)``
    ⇒ 两种配对**不同** ✓（``head_dim=8`` 只有一对，测不出 ✗）。
    """
    vocab, hidden, ffn = 64, 32, 48
    heads, kv_heads, head_dim = 2, 1, 16
    key_heads, value_heads, head_width = 2, 4, 8
    kernel = 4
    key_dim, value_dim = key_heads * head_width, value_heads * head_width

    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        # ⚠️ 真文件里同存的视觉塔与 lm_head ✓ —— 反推必须**看不见**它们 ✓✗
        "model.visual.blocks.0.attn.qkv.weight": (hidden, hidden),
        "lm_head.weight": (vocab, hidden),
    }
    common = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (ffn, hidden),
        "mlp.up_proj.weight": (ffn, hidden),
        "mlp.down_proj.weight": (hidden, ffn),
    }
    linear = {
        "linear_attn.in_proj_qkv.weight": (2 * key_dim + value_dim, hidden),
        "linear_attn.in_proj_z.weight": (value_dim, hidden),
        "linear_attn.in_proj_b.weight": (value_heads, hidden),
        "linear_attn.in_proj_a.weight": (value_heads, hidden),
        "linear_attn.out_proj.weight": (hidden, value_dim),
        "linear_attn.conv1d.weight": (2 * key_dim + value_dim, 1, kernel),
        "linear_attn.A_log": (value_heads,),
        "linear_attn.dt_bias": (value_heads,),
        "linear_attn.norm.weight": (head_width,),
    }
    full = {
        "self_attn.q_proj.weight": (2 * heads * head_dim, hidden),
        "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
        "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
        "self_attn.o_proj.weight": (hidden, heads * head_dim),
        "self_attn.q_norm.weight": (head_dim,),
        "self_attn.k_norm.weight": (head_dim,),
    }
    for index, own in ((0, linear), (1, full)):
        for name, value in {**common, **own}.items():
            shapes[f"model.layers.{index}.{name}"] = value
    return shapes


SHAPES = small_shapes()
SMALL = q.infer_shape(SHAPES)


def seeded_module(layer_idx: int = -2) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """小塔 ＋ 一份**好脾气**的随机权重 ✓（两次建同一形状 ⇒ 权重逐字相同 ✓ 可直接换 ``layer_idx`` 比 ✓）。

    ⚠️ ``A_log`` **不能**跟着一起 ``+1`` ✗：``exp(A_log)≈2.7`` ⇒ 每步衰减 ``e^-3.5`` ⇒ 状态活不过
    一个 token ✓ ⇒ delta rule 退化成「只看最后一个 token」✓ 头复制/衰减符号这些坑全测不出来 ✗✗。
    这里取 ``A_log = -2``（每步留 ~0.84 ✓）⇒ 记忆横跨十几个 token ✓。
    """
    torch.manual_seed(20261031)
    module = q.build_from_shapes(SHAPES, layer_idx=layer_idx)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("A_log"):
                param.copy_(torch.randn_like(param) * 0.02 - 2.0)
            elif param.dim() == 1:
                param.copy_(torch.randn_like(param) * 0.05 + 1.0)
            else:
                param.copy_(torch.randn_like(param) * 0.05)
    return module, {k: v.detach().clone() for k, v in module.state_dict().items()}


# --------------------------------------------------------------- ② 独立参照


def _ref_rms(x: torch.Tensor, weight: torch.Tensor, add: bool) -> torch.Tensor:
    """``x/rms(x) * (weight + 1)`` ✓ —— 手写，**不走** ``F.rms_norm`` ✓。"""
    wide = weight + 1.0 if add else weight
    value = x.float()
    value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + SMALL.rms_eps)
    return value * wide.float()


def _ref_rope(x: torch.Tensor, rotary: int, mutate: frozenset[str]) -> torch.Tensor:
    """只转前 ``rotary`` 维 ✓；``rope_adjacent`` 时改成相邻配对 ✗（正法是 split-half ✓）。"""
    length = x.shape[0]
    inv = 1.0 / (SMALL.rope_theta ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary))
    angle = torch.outer(torch.arange(length, dtype=torch.float32), inv)      # (T, rotary/2)
    cos, sin = angle.cos(), angle.sin()
    out = x.clone()
    if "rope_adjacent" in mutate:
        first, second = x[..., 0:rotary:2], x[..., 1:rotary:2]
        out[..., 0:rotary:2] = first * cos[:, None] - second * sin[:, None]
        out[..., 1:rotary:2] = first * sin[:, None] + second * cos[:, None]
        return out
    half = rotary // 2
    co, si = cos[:, None], sin[:, None]                                      # (T,1,half)
    first, second = x[..., :half], x[..., half:rotary]
    out[..., :half] = first * co - second * si
    out[..., half:rotary] = first * si + second * co
    return out


def _ref_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """**逐 token 递推**的 delta rule ✓（= 块内求逆 ＋ 块间递推 的等价展开 ✓）。

    ``S_t = exp(g_t)·S_{t-1} + beta_t·k_t⊗(v_t − k_t·exp(g_t)·S_{t-1})``、``o_t = q_t·S_t`` ✓。
    """
    heads, k_dim, v_dim = query.shape[1], query.shape[-1], value.shape[-1]
    state = torch.zeros(heads, k_dim, v_dim)
    out = torch.zeros_like(value)
    for step in range(query.shape[0]):
        state = state * g[step].exp()[:, None, None]
        proj = torch.einsum("hk,hkv->hv", key[step], state)
        state = state + beta[step][:, None, None] * key[step].unsqueeze(-1) * (value[step] - proj).unsqueeze(-2)
        out[step] = torch.einsum("hk,hkv->hv", query[step], state)
    return out


def _ref_linear_attn(sd: dict[str, torch.Tensor], prefix: str, x: torch.Tensor,
                     mutate: frozenset[str]) -> torch.Tensor:
    """Gated DeltaNet ✓：因果卷积 → silu → 劈 q/k/v → 衰减+beta → delta rule → 门控 norm → out_proj ✓。"""
    w = lambda name: sd[f"{prefix}.linear_attn.{name}"]  # noqa: E731
    length = x.shape[0]
    mixed = x @ w("in_proj_qkv.weight").T
    z = x @ w("in_proj_z.weight").T
    b = x @ w("in_proj_b.weight").T
    a = x @ w("in_proj_a.weight").T
    kernel_w = w("conv1d.weight")[:, 0]
    kernel = SMALL.conv_kernel
    conv = torch.zeros_like(mixed)
    for step in range(length):
        for tap in range(kernel):
            # 正法：左补 kernel-1 再**切尾** ⇒ 第 step 个输出只吃 ≤ step 的输入 ✓
            source = step + tap if "conv_tail" in mutate else step - (kernel - 1) + tap
            if 0 <= source < length:
                conv[step] += mixed[source] * kernel_w[:, tap]
    mixed = F.silu(conv)
    query, key, value = mixed.split([SMALL.key_dim, SMALL.key_dim, SMALL.value_dim], dim=-1)
    decay = w("A_log").exp()
    if "decay_sign" not in mutate:
        decay = -decay
    g = decay * F.softplus(a + w("dt_bias"))
    beta = b.sigmoid()
    query = F.normalize(query.reshape(length, SMALL.linear_key_heads, SMALL.key_head_dim), dim=-1)
    key = F.normalize(key.reshape(length, SMALL.linear_key_heads, SMALL.key_head_dim), dim=-1)
    value = value.reshape(length, SMALL.linear_value_heads, SMALL.value_head_dim)
    repeat = SMALL.linear_value_heads // SMALL.linear_key_heads
    if "head_repeat" in mutate:
        query, key = query.repeat(1, repeat, 1), key.repeat(1, repeat, 1)
    else:
        query, key = query.repeat_interleave(repeat, 1), key.repeat_interleave(repeat, 1)
    core = _ref_delta_rule(query * (SMALL.key_head_dim ** -0.5), key, value, g, beta)
    core = _ref_rms(core.reshape(-1, SMALL.value_head_dim), w("norm.weight"),
                    "gated_norm_add" in mutate) * F.silu(z.reshape(-1, SMALL.value_head_dim))
    return core.reshape(length, SMALL.value_dim) @ w("out_proj.weight").T


def _ref_full_attn(sd: dict[str, torch.Tensor], prefix: str, x: torch.Tensor,
                   mutate: frozenset[str]) -> torch.Tensor:
    """GatedAttention ✓：q_proj 出 query+gate ✓、QK-norm ✓、partial RoPE ✓、因果 ✓、sigmoid 门 ✓。"""
    w = lambda name: sd[f"{prefix}.self_attn.{name}"]  # noqa: E731
    length = x.shape[0]
    packed = x @ w("q_proj.weight").T
    if "gate_chunk" in mutate:
        # 错法：整体前一半当 query、后一半当 gate ✗
        query, gate = packed[:, :SMALL.inner], packed[:, SMALL.inner:]
    else:
        # 正法：按头交错 `[q(hd) | gate(hd)]` ✓
        block = packed.reshape(length, SMALL.heads, 2 * SMALL.head_dim)
        query = block[..., : SMALL.head_dim].reshape(length, SMALL.inner)
        gate = block[..., SMALL.head_dim:].reshape(length, SMALL.inner)
    query = query.reshape(length, SMALL.heads, SMALL.head_dim)
    key = (x @ w("k_proj.weight").T).reshape(length, SMALL.kv_heads, SMALL.head_dim)
    value = (x @ w("v_proj.weight").T).reshape(length, SMALL.kv_heads, SMALL.head_dim)
    add = "qk_no_add" not in mutate
    query = _ref_rope(_ref_rms(query, w("q_norm.weight"), add), SMALL.rotary_dim, mutate)
    key = _ref_rope(_ref_rms(key, w("k_norm.weight"), add), SMALL.rotary_dim, mutate)
    group = SMALL.heads // SMALL.kv_heads
    out = torch.zeros(length, SMALL.heads, SMALL.head_dim)
    for step in range(length):
        for head in range(SMALL.heads):
            shared = head // group
            score = torch.stack([query[step, head] @ key[other, shared]
                                 for other in range(step + 1)]) / SMALL.head_dim ** 0.5
            weight = score.softmax(-1)
            out[step, head] = sum((weight[other] * value[other, shared]
                                   for other in range(step + 1)), torch.zeros(SMALL.head_dim))
    return (out.reshape(length, SMALL.inner) * gate.sigmoid()) @ w("o_proj.weight").T


def _ref_layer(sd: dict[str, torch.Tensor], index: int, x: torch.Tensor, kind: str,
               mutate: frozenset[str] = frozenset()) -> torch.Tensor:
    """pre-norm 一层 ✓：``norm → attn → 残差 → norm → mlp → 残差`` ✓。"""
    prefix = f"model.layers.{index}"
    add = "ln_no_add" not in mutate
    normed = _ref_rms(x, sd[f"{prefix}.input_layernorm.weight"], add)
    mixed = (_ref_linear_attn(sd, prefix, normed, mutate) if kind == "linear_attention"
             else _ref_full_attn(sd, prefix, normed, mutate))
    hidden = x + mixed
    normed = _ref_rms(hidden, sd[f"{prefix}.post_attention_layernorm.weight"], add)
    mlp = F.silu(normed @ sd[f"{prefix}.mlp.gate_proj.weight"].T) * (normed @ sd[f"{prefix}.mlp.up_proj.weight"].T)
    return hidden + mlp @ sd[f"{prefix}.mlp.down_proj.weight"].T


def _ref_tower(sd: dict[str, torch.Tensor], x: torch.Tensor, upto: int | None = None,
               mutate: frozenset[str] = frozenset()) -> torch.Tensor:
    for index, kind in enumerate(SMALL.layer_types):
        x = _ref_layer(sd, index, x, kind, mutate)
        if index == upto:
            break
    return x


def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """最大绝对差 ✓；溢出成 inf / nan 也算「差得离谱」✓（反向验证里 ``decay_sign`` 会炸 ✓）。"""
    value = (a - b).abs().max()
    return float(value) if bool(torch.isfinite(value).all()) else float("inf")


def _run_layers(module: torch.nn.Module, ids: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """依次跑前 ``layer + 1`` 个 block ✓ ⇒ 返回 (该层输入 ✓, 该层输出 ✓)。"""
    cos, sin = q._rope_tables(int(ids.shape[1]), SMALL.rotary_dim, SMALL.rope_theta, ids.device)
    with torch.no_grad():
        state = module.model.embed_tokens(ids)
        source = state
        for index, block in enumerate(module.model.layers):
            source = state
            state = block(state, cos, sin)
            if index == layer:
                break
    return source, state


# --------------------------------------------------------------- ① 真权重


def case_shape_from_real() -> None:
    """① 从盘上真权重反推结构 ✓ ＋ 语言塔键逐字对账 ✓。"""
    if not REAL.exists():
        print("  · 盘上没有 Qwen3.5 文本塔，跳过 ①")
        return
    info = st.inspect(str(REAL))
    shapes = {name: tuple(int(d) for d in tensor.shape) for name, tensor in info.tensors.items()}
    shape = q.infer_shape(shapes)
    got = (shape.hidden, shape.depth, shape.ffn, shape.heads, shape.kv_heads, shape.head_dim,
           shape.rotary_dim, shape.linear_key_heads, shape.linear_value_heads, shape.key_head_dim,
           shape.value_head_dim, shape.conv_kernel)
    # ⚠️ 线性注意力那边是 **16 key 头 / 32 value 头**（同头宽 128 ✓）✗ —— ``in_proj_qkv`` 出 8192 维
    # 拆成 ``key 2048 + key 2048 + value 4096`` ✓，别按「两边一样宽」猜成 32/32 ✗（会白算一遍才炸 ✗）
    want = (4096, 32, 12288, 16, 4, 256, 64, 16, 32, 128, 128, 4)
    check("① 真权重反推 hidden/depth/ffn/heads/kv/head_dim/rotary/线性头/头宽/卷积核", got == want,
          {"得到": got, "应为": want})
    check("①′ 层型表：24 线性 ＋ 8 完整，完整层落在第 3/7/…/31 层（每 4 层一个）",
          (shape.layer_types.count("linear_attention"), shape.layer_types.count("full_attention"),
           [i for i, kind in enumerate(shape.layer_types) if kind == "full_attention"])
          == (24, 8, list(range(3, 32, 4))))
    check("①″ `layer_idx=-2` ⇒ 0-based 第 30 层（不是 31、也不是 -1）", shape.target_layer() == 30,
          shape.target_layer())

    want_keys = q.expected_keys(shape)
    plain = q._language_shapes(shapes)
    have = {name for name in plain
            if not name.endswith(quant.COMPANION_SUFFIXES)
            and not name.endswith(quant.QUANT_META_SUFFIX)}
    check("①‴ `expected_keys` 与文件里的语言塔键**逐字**相同（视觉塔 / lm_head / 量化伴生键都已剥掉）",
          want_keys == have,
          {"只在期望里": sorted(want_keys - have)[:5], "只在文件里": sorted(have - want_keys)[:5],
           "期望": len(want_keys), "文件": len(have)})


def case_keys_match_module() -> None:
    """①⁴ ``build_from_shapes`` 建出来的键 == ``expected_keys`` ✓（9B 按 meta 建 ✓ 不占内存 ✓）。"""
    if not REAL.exists():
        print("  · 盘上没有 Qwen3.5 文本塔，跳过 ①⁴")
        return
    info = st.inspect(str(REAL))
    shapes = {name: tuple(int(d) for d in tensor.shape) for name, tensor in info.tensors.items()}
    shape = q.infer_shape(shapes)
    # ⚠️ 9B 的 bf16 权重是 18.5 GiB ✓ —— 只有 meta 建才敢在自检里跑 ✓
    with torch.device("meta"):
        module = q.build_from_shapes(shapes)
    names = set(module.state_dict())
    want = q.expected_keys(shape)
    check("①⁴ 9B 塔按 meta 建得出来，键与 `expected_keys` **逐字**相同", names == want,
          {"只在期望里": sorted(want - names)[:5], "只在模块里": sorted(names - want)[:5]})
    small, _ = seeded_module()
    check("①⁵ 小塔同理（不依赖盘上文件的那条通用防线）",
          set(small.state_dict()) == q.expected_keys(SMALL))


def case_layer_index() -> None:
    """④ ``layer_idx`` 语义 ✓：负索引折算 ✓ / 越界拒 ✓ / 前向取的就是那层 ✓ / 不过 final norm ✓。"""
    check("④ `target_layer()`：-2 ⇒ 倒数第 2 层、-1 ⇒ 最后一层",
          (SMALL.target_layer(), replace(SMALL, layer_idx=-1).target_layer()) == (0, 1),
          (SMALL.target_layer(), replace(SMALL, layer_idx=-1).target_layer()))
    for bad in (2, -3, 32):
        try:
            replace(SMALL, layer_idx=bad).target_layer()
            check(f"④ 越界的 layer_idx={bad} 必须当场拒", False, "没拒")
        except q.ShapeError:
            check(f"④ 越界的层下标 {bad} 当场拒 ✓", True)

    module, _ = seeded_module()
    ids = torch.randint(0, SMALL.vocab, (1, 7), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        got = module(ids)
    first = _run_layers(module, ids, 0)[1]
    check("④′ 前向吐的是第 0 层之后的 hidden（depth=2 ⇒ 倒数第 2 层 = 第 0 层）",
          torch.equal(got, first), (tuple(got.shape), tuple(first.shape)))
    with torch.no_grad():
        normalised = module.model.norm(first)
    check("④″ **没有**套 final norm（`model.norm` 建了但不参与前向）",
          not torch.allclose(got, normalised, atol=1e-5), _gap(got, normalised))
    check("④‴ 形状 (1, s, hidden) ✓ 宽 = 主干 text_dim", tuple(got.shape) == (1, 7, SMALL.hidden),
          tuple(got.shape))
    try:
        module(torch.zeros(2, 5, dtype=torch.long))
        check("④⁗ batch>1 必须拒（因果掩码那条路没有按条 pad）", False, "没拒")
    except q.ShapeError:
        check("④⁗ batch>1 当场拒 ✓（逐条调用）", True)


def case_runtime_dict() -> None:
    """运行时壳 ✓：``output_dim`` / ``max_length`` / ``to_dict()`` ✓（后端 ``encode_text`` 只认这几个 ✓）。"""
    payload = q.Qwen35TeRuntime(SMALL).to_dict()
    check("runtime：outputDim == hidden、vocabSize、layerIdx=-2、targetLayer=0、maxLength=512",
          (payload["outputDim"], payload["vocabSize"], payload["layerIdx"],
           payload["targetLayer"], payload["maxLength"])
          == (SMALL.hidden, SMALL.vocab, -2, 0, 512), payload)
    check("runtime：层型表原样交出去（24/8 那件事后端要看得见）",
          payload["layerTypes"] == list(SMALL.layer_types))
    text = q.tokens_of("一只猫")
    check("runtime：模板只在 `tokens_of` 这一处套（user 段 ＋ 关闭思考的 assistant 段 ✓ 没有 system）",
          text == q.QWEN35_TEMPLATE.format("一只猫")
          and text.startswith("<|im_start|>user\n一只猫<|im_end|>\n<|im_start|>assistant\n")
          and text.endswith("<think>\n</think>\n") and "system" not in text, repr(text))


# --------------------------------------------------------------- ⑤ 后端接线


def _tower_file(root: Path, name: str = "tower.safetensors",
                drop: str | None = None) -> Path:
    """写一份**仿真文件形态**的小塔 ✓：键带 ``model.language_model.`` 前缀 ✓ ＋ 视觉塔 ＋ ``lm_head`` ✓。

    ⚠️ 真文件（9B）里这三族**同存** ✓ —— 后端必须只装语言塔 ✓、另两族**单列**而不报错 ✓✗
    （`ignoredTowerKeys` 就是给这件事留的读数 ✓）；``drop`` 用来造「缺键」那份 ✓。
    """
    from safetensors.torch import save_file  # noqa: PLC0415 —— 只在这里用到 ✓

    _, sd = seeded_module()
    # ⚠️ 真文件里的语言塔键是 ``model.language_model.`` ＋ **塔内键剥掉 `model.`** ✓
    #    （即 ``model.language_model.embed_tokens.weight`` ✓ —— 与
    #    :func:`qwen35_te._language_shapes` 的剥法互逆 ✓）；写成
    #    ``model.language_model.model.embed_tokens.weight`` 会被它拼成 ``model.model.*`` ✗✗。
    tensors = {f"model.language_model.{key[len('model.'):]}": value
               for key, value in sd.items()}
    tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(
        SHAPES["model.visual.blocks.0.attn.qkv.weight"])
    tensors["lm_head.weight"] = torch.zeros(SHAPES["lm_head.weight"])
    if drop:
        # ⚠️ **不带默认值** ✗：`pop(drop, None)` 在名字写错时静默不删 ✓ ⇒ 缺键那条判据**假绿** ✓✗
        #    （本轮第一版就这么绿过 ✓ —— 前缀与 drop 名两处口径不一致，谁都没报 ✓）。
        del tensors[drop]
    path = root / name
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))
    return path


def case_backend_attach() -> None:
    """⑤ ``TorchBackend.attach_qwen35_te`` 的接线 ✓（装得对 / 拒绝得准 / 不留随机层 ✓）。"""
    import tempfile  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from app.services.engine import text_encoder as te_mod  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    backend = tb.TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        print("  · torch 不可用，跳过 ⑤")
        return
    want_keys = q.expected_keys(SMALL)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = _tower_file(root)
        backend._config = {"text_dim": SMALL.hidden}
        detail = backend.attach_qwen35_te(str(path), dtype=torch.float32, max_length=128)
        check("⑤ `attach_qwen35_te` 装上 ✓（kind / complete / textKind 三处都说真话 ✓）",
              detail.get("kind") == "qwen35-te" and detail.get("complete") is True
              and backend._textKind == "qwen35-te"
              and backend.describe()["textEncoderLoaded"] is True,
              {k: detail.get(k) for k in ("kind", "complete")})
        check(f"⑤′ loadedKeys == 塔的键数 {len(want_keys)} ✓、sourceKeys 含视觉塔与 lm_head ✓"
              "（少装一个键在这条判据上会立刻露头 ✓✗）",
              (detail.get("loadedKeys"), detail.get("sourceKeys"))
              == (len(want_keys), len(want_keys) + 2),
              (detail.get("loadedKeys"), detail.get("sourceKeys")))
        loaded = set(backend._textEncoder.state_dict())
        check("⑤″ 视觉塔 / lm_head **单列**（ignoredTowerKeys=2 ✓ unexpectedCount=2 ✓）"
              "且**没进模块** ✓（键集合 == `expected_keys` ✓）",
              (detail.get("ignoredTowerKeys"), detail.get("unexpectedCount")) == (2, 2)
              and loaded == want_keys,
              {"只在模块里": sorted(loaded - want_keys)[:5],
               "ignored": detail.get("ignoredTowerKeys")})
        check("⑤‴ 报告里的 outputDim 与主干 `text_dim` 一致 ✓（4096 那条在真机上由权重给出 ✓）",
              detail["config"]["outputDim"] == SMALL.hidden
              and detail["config"]["layerTypes"] == list(SMALL.layer_types), detail["config"])

        # ⭐ `encode_text` 的通用分支要吃得下它 ✓（模板只在 runtime 里套一次 ✓）
        backend._tokenizer = te_mod.StubTokenizer(SMALL.vocab, max_length=128)
        encoded = backend.encode_text(SimpleNamespace(prompt="一只猫", negative=""))
        check("⑤⁗ `encode_text` 走真塔 ⇒ (1, L, hidden) ✓、`textEncoder=\"qwen35-te\"` ✓、"
              "模板化如实上报 ✓",
              tuple(encoded["positive"].shape) == (1, encoded["tokens"], SMALL.hidden)
              and encoded["textEncoder"] == "qwen35-te"
              and encoded["byBranch"]["positive"]["templated"] is True,
              (tuple(encoded["positive"].shape), encoded.get("textEncoder")))

        # ⭐ **先核后改** ✓：宽度对不上 ⇒ 当场拒，先前挂好的塔原封不动 ✓
        wrong = tb.TorchBackend(device="cpu")
        wrong._config = {"text_dim": SMALL.hidden * 2}
        err = None
        try:
            wrong.attach_qwen35_te(str(path), dtype=torch.float32)
        except Exception as error:  # noqa: BLE001
            err = error
        check("⑤⁵ `text_dim` 对不上 ⇒ 挂载**当场拒** ✓ 且点名 `condition_proj` ✓ / 没有半装状态 ✓",
              err is not None and "condition_proj" in str(err)
              and wrong._textEncoder is None and wrong._textKind is None,
              str(err)[:120] if err else "没拒")

        # ⭐ 缺键 ⇒ 拒装 ✓，而且**分两道闸**都钉住（⚠️ 缺键其实先被 `infer_shape` 拦 ✗ ——
        #    那条闸比装载器更早、更指得准 ✓；`load_module_weights` 的 missing 分支是**兜底** ✓，
        #    它的前提是「模块键 ≠ 期望键」✗，而 ①⁴ 已钉住两者逐字相等 ✓ ⇒ 这里造不出来 ✓）
        for name, drop, want_text in (
                ("no_attn.safetensors",
                 "model.language_model.layers.1.self_attn.q_proj.weight", "不是这一族的层"),
                ("no_mlp.safetensors",
                 "model.language_model.layers.1.mlp.up_proj.weight", "权重没下全")):
            broken = _tower_file(root, name=name, drop=drop)
            half = tb.TorchBackend(device="cpu")
            half._config = {"text_dim": SMALL.hidden}
            err = None
            try:
                half.attach_qwen35_te(str(broken), dtype=torch.float32)
            except Exception as error:  # noqa: BLE001
                err = error
            check(f"⑤⁶ 缺 `{drop.rsplit('.', 2)[-2]}.{drop.rsplit('.', 1)[-1]}` ⇒ **拒装** ✓"
                  f"（反推阶段就报「{want_text}」✓）且不留随机层 ✓（`_textEncoder`/`_textKind` 仍 None ✓）",
                  err is not None and want_text in str(err)
                  and half._textEncoder is None and half._textKind is None,
                  str(err)[:140] if err else "没拒")


# --------------------------------------------------------------- ② 数值对拍


def case_parity_with_reference() -> None:
    """② 小塔 fp32 与逐 token 递推的独立参照比数值 ✓（两种长度 ✓ 覆盖单块 / 跨块 ✓）。

    ⚠️ 「整塔」那条必须**另建一个** ``layer_idx=-1`` 的塔 ✓：默认 ``layer_idx=-2`` 在 depth=2 时
    口径是**第 0 层** ✓，拿它去比两层结果 ⇒ 一定对不上 ✓✗（第一版就这么错过 ✓）。
    """
    module, sd = seeded_module()
    whole, _ = seeded_module(layer_idx=-1)
    for length in (6, 150):
        ids = torch.randint(0, SMALL.vocab, (1, length), generator=torch.Generator().manual_seed(length))
        with torch.no_grad():
            got = module(ids)[0]
            embedded = module.model.embed_tokens(ids)[0]
            tower_out = whole(ids)[0]
        first = _ref_layer(sd, 0, embedded, SMALL.layer_types[0])
        tower = _ref_layer(sd, 1, first, SMALL.layer_types[1])
        scale = float(tower.abs().max())
        check(f"② 线性注意力层 T={length}（块内求逆 vs 逐 token 递推）", _gap(first, got) / scale < 1e-5,
              _gap(first, got) / scale)
        check(f"②′ 整塔 T={length}（两层串联 ＋ 跨块状态传递：150 = 64+64+22 ✓）",
              _gap(tower_out, tower) / scale < 1e-5, _gap(tower_out, tower) / scale)


# --------------------------------------------------------------- ③ 反向验证

#: (改坏谁, 在第几层上验, 说明) ✓ —— 每条都是「不报错、只是数不对」的口径 ✓
VARIANTS: tuple[tuple[str, int, str], ...] = (
    ("ln_no_add", 0, "层 pre-norm 丢掉 (1+w)（Qwen 系 RMSNorm 存的是 scale−1 ✓）"),
    ("gated_norm_add", 0, "门控 norm 误加 (1+w)（它存的就是权重本身 ✓）"),
    ("conv_tail", 0, "因果卷积不切尾（看到未来 ✓ 形状全对 ✓）"),
    ("head_repeat", 0, "key/value 头用 repeat 而不是 repeat_interleave ✓"),
    ("decay_sign", 0, "A_log 符号反（衰减变放大 ✓）"),
    ("gate_chunk", 1, "q_proj 出参按 chunk(2) 劈，而不是按头交错 [q|gate] ✓"),
    ("qk_no_add", 1, "QK-norm 丢掉 (1+w) ✓"),
    ("rope_adjacent", 1, "RoPE 用相邻配对而不是 split-half ✓"),
)


def case_reverse_variants() -> None:
    """③ 每条坑都要「与主实现的差 ≫ 正确参照与主实现的差」✓ —— 否则 ② 那条判据是假绿 ✓✗。"""
    module, sd = seeded_module()
    for mutation, layer, why in VARIANTS:
        ids = torch.randint(0, SMALL.vocab, (1, 12), generator=torch.Generator().manual_seed(300 + layer))
        source, got = _run_layers(module, ids, layer)
        kind = SMALL.layer_types[layer]
        base = _gap(got[0], _ref_layer(sd, layer, source[0], kind))
        harm = _gap(got[0], _ref_layer(sd, layer, source[0], kind, frozenset({mutation})))
        check(f"③ {mutation}：改坏后差 {harm:.3e} ≫ 没改坏的 {base:.3e}（{why}）",
              harm > max(base * 1e3, 1e-6), (harm, base))


# --------------------------------------------------------------- 入口


def main() -> int:
    cases = (
        ("① 形状反推（真权重）", case_shape_from_real),
        ("① 键对账（真权重 / meta 建）", case_keys_match_module),
        ("④ 取层与 batch", case_layer_index),
        ("运行时壳", case_runtime_dict),
        ("⑤ 后端接线", case_backend_attach),
        ("② 与独立参照对拍", case_parity_with_reference),
        ("③ 反向验证", case_reverse_variants),
    )
    for title, case in cases:
        print(f"\n{title}")
        try:
            case()
        except Exception:  # noqa: BLE001 - 自检要把异常收进汇总 ✓
            FAILS.append(title)
            traceback.print_exc()
    print()
    if FAILS:
        print(f"✗ {len(FAILS)} 项未过：")
        for name in FAILS:
            print(f"  · {name}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
