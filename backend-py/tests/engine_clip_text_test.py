"""自检：``app.services.engine.clip_text``（CLIP 文本塔：CLIP-L + OpenCLIP-bigG ✓）。

跑法（在 ``backend-py`` 下 ✓）：

    ./.venv/Scripts/python.exe tests/engine_clip_text_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import clip_text as ct  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []
_TORCH: Any = None
_PROBED = False


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def torch_or_skip() -> Any:
    """懒取 torch ✓（只里探一次 ✓ ⇒ SKIP 只打一条 ✓）。"""
    global _TORCH, _PROBED  # noqa: PLW0603
    if not _PROBED:
        _PROBED = True
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            _TORCH = None
            skip("没装 torch ⇒ 文本塔整层不跑 ✓（显式 SKIP ✓ 不是通过 ✗）")
        else:
            _TORCH = torch
    return _TORCH


def tiny_config(**overrides: Any) -> Any:
    """缩小版配置 ✓（走**同一条前向** ✓ 不下载真权重就能验 ✓）。"""
    params = dict(
        name="tiny",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=16,
        pad_token=0,
    )
    params.update(overrides)
    return ct.ClipTextConfig(**params)


def case_config(t: Any) -> None:
    """① 两塔口径钉死 ✓ + 判别性校验**必须报错** ✗。"""
    check("① CLIP-L：768 宽 / 12 层 / 12 头 / 3072 / quick_gelu / **无**投影 / pad=49407 ✓",
          (ct.CLIP_L.hidden_size, ct.CLIP_L.num_hidden_layers, ct.CLIP_L.num_attention_heads,
           ct.CLIP_L.intermediate_size, ct.CLIP_L.hidden_act, ct.CLIP_L.projection_dim,
           ct.CLIP_L.pad_token) == (768, 12, 12, 3072, "quick_gelu", 0, 49407),
          ct.CLIP_L.describe())
    check("① bigG：1280 宽 / 32 层 / 20 头 / 5120 / gelu / 投影 1280 / pad=**0** ✓",
          (ct.CLIP_G.hidden_size, ct.CLIP_G.num_hidden_layers, ct.CLIP_G.num_attention_heads,
           ct.CLIP_G.intermediate_size, ct.CLIP_G.hidden_act, ct.CLIP_G.projection_dim,
           ct.CLIP_G.pad_token) == (1280, 32, 20, 5120, "gelu", 1280, 0),
          ct.CLIP_G.describe())
    check("① 两塔 vocab=49408、位置=77、eps=1e-5 ✓",
          all((cfg.vocab_size, cfg.max_position_embeddings, cfg.layer_norm_eps)
              == (49408, 77, 1e-5) for cfg in (ct.CLIP_L, ct.CLIP_G)))
    check("① 头宽 = 宽 ÷ 头数 ✓（768/12=64、1280/20=64 ✓）",
          (ct.CLIP_L.head_dim, ct.CLIP_G.head_dim) == (64, 64))
    check("① SDXL 条件宽 = 768 + 1280 = 2048 ✓",
          ct.SDXL_CONTEXT_DIM == ct.CLIP_L.hidden_size + ct.CLIP_G.hidden_size)
    check("① 官方检查点前缀：L/PREFIX-0、G/PREFIX-1 ✓",
          ct.CLIP_L.load_prefix.endswith("embedders.0.") and ct.CLIP_G.load_prefix.endswith("embedders.1."),
          (ct.CLIP_L.load_prefix, ct.CLIP_G.load_prefix))
    check("① 空名字 ⇒ 报错 ✓", _raises(lambda: tiny_config(name="  "), "名字") is not None)
    check("① 头数不整除宽 ⇒ 报错 ✓",
          _raises(lambda: tiny_config(hidden_size=30, num_attention_heads=4), "整除") is not None)
    check("① 认不出的激活 ⇒ 报错 ✓（不静默当 gelu ✗）",
          _raises(lambda: tiny_config(hidden_act="relu"), "hidden_act") is not None)
    check("① pad 落在词表外 ⇒ 报错 ✓",
          _raises(lambda: tiny_config(vocab_size=64, pad_token=99), "pad_token") is not None)
    check("① 投影维度为负 ⇒ 报错 ✓",
          _raises(lambda: tiny_config(projection_dim=-1), "projection_dim") is not None)


def case_key_names(t: Any) -> None:
    """② 键名表：能**不建模型**算出应有键数 ✓（真权重实测 196 / 517 ✓）。"""
    l_names = ct.clip_text_key_names(ct.CLIP_L)
    g_names = ct.clip_text_key_names(ct.CLIP_G)
    check("② CLIP-L 应有 **196** 键 ✓（真权重实测 196 ✓：2 嵌入 + 12 层 × 16 + 2 末尾 ✓）",
          len(l_names) == 196 and len(set(l_names)) == 196, len(l_names))
    check("② bigG 应有 **517** 键 ✓（= 196 结构 + 32 层 × 10 + 投影 ✓）",
          len(g_names) == 517 and len(set(g_names)) == 517, len(g_names))
    check("② 键名带 ``transformer.text_model.`` 前缀 ✓、投影只在带投影那塔 ✓",
          all(n.startswith("transformer.text_model.") for n in l_names)
          and "text_projection.weight" not in l_names
          and "text_projection.weight" in g_names)
    check("② 自述字典带上关键口径 ✓",
          ct.CLIP_G.describe()["projection_dim"] == 1280
          and ct.CLIP_L.describe()["head_dim"] == 64)


def case_forward(t: Any) -> None:
    """③ 缩小版前向：形状 / 层数 / 选层 / norm 开关 ✓ —— 走的是**同一条**前向 ✓。"""
    cfg = tiny_config()
    model = ct.build_clip_text_model(cfg)
    model.eval()
    check("③ 自建模型的键集与 ``clip_text_key_names`` **逐个对上** ✓（不多不少 ✓）",
          set(model.state_dict()) == set(ct.clip_text_key_names(cfg)),
          (len(model.state_dict()), len(ct.clip_text_key_names(cfg))))
    ids = t.tensor([[1, 2, 3, 4, 63, 0, 0], [5, 6, 7, 63, 0, 0, 0]])
    out = model(ids, layer_idx=-2, layer_norm_hidden_state=False, return_all_hidden_states=True,
                eos_token_id=63)
    check("③ 三层张量形状：last/selected ``(2,7,32)``、pooled ``(2,32)`` ✓",
          tuple(out.last_hidden_state.shape) == (2, 7, 32)
          and tuple(out.selected.shape) == (2, 7, 32)
          and tuple(out.pooled.shape) == (2, 32),
          out.describe())
    check("③ ``hidden_states`` 长度 = 层数 + 1 = 3 ✓（含嵌入输出 ✓）",
          len(out.hidden_states) == 3)
    check("③ 2 层时 ``layer_idx=-2`` ⇒ 下标 **0** ✓（倒数第二 ✓）", out.layer_index == 0)
    check("③ ``layer_norm_hidden_state=False`` ⇒ selected **原样**是该层输出 ✓",
          t.equal(out.selected, out.hidden_states[1]),
          float((out.selected - out.hidden_states[1]).abs().max()))
    same = model(ids, layer_idx=-2, layer_norm_hidden_state=True, eos_token_id=63)
    check("③ ``layer_norm_hidden_state=True`` ⇒ selected **不一样**（过了末尾 LayerNorm ✓）",
          not t.allclose(same.selected, out.selected))
    check("③ 不带投影那塔：``projected_pooled is None`` ✓（**不假装有** ✗）",
          out.projected_pooled is None)
    g_model = ct.build_clip_text_model(tiny_config(name="tiny_g", projection_dim=8))
    g_out = g_model(ids, eos_token_id=63)
    check("③ 带投影那塔：``projected_pooled`` 形状 ``(2,8)`` ✓ 且 = ``pooled @ Wᵀ`` ✓",
          tuple(g_out.projected_pooled.shape) == (2, 8)
          and t.allclose(g_out.projected_pooled,
                         g_out.pooled @ g_model.text_projection.weight.T, atol=1e-5))
    check("③ 层号越界 ⇒ 报错 ✓（不是夹到范围内 ✗）",
          _raises(lambda: model(ids, layer_idx=5, eos_token_id=63), "越界") is not None)
    check("③ 序列长超过位置嵌入 ⇒ 报错 ✓",
          _raises(lambda: model(t.ones((1, 20), dtype=t.long), eos_token_id=63), "位置嵌入") is not None)
    check("③ 输入不是二维 ⇒ 报错 ✓",
          _raises(lambda: model(t.ones((7,), dtype=t.long), eos_token_id=63), "(批, 长度)") is not None)


def case_causal_pool(t: Any) -> None:
    """④ 因果掩码 + EOS 池化位置 ✓（这两条错了会**静默**给出错条件 ✗ ⇒ 必须有判据 ✓）。"""
    cfg = tiny_config()
    model = ct.build_clip_text_model(cfg)
    model.eval()
    ids = t.tensor([[1, 2, 3, 4, 63, 0, 0]])
    full = model(ids, layer_norm_hidden_state=False, eos_token_id=63)
    changed = ids.clone()
    changed[0, 6] = 9  # 只改**最后**一个 token ✓
    after = model(changed, layer_norm_hidden_state=False, eos_token_id=63)
    check("④ 改最后一个 token ⇒ **前面各位置**输出一字不变 ✓（因果掩码 ✓）",
          t.allclose(full.last_hidden_state[:, :6], after.last_hidden_state[:, :6], atol=1e-6))
    check("④ 改最后一个 token ⇒ 那个位置**变了** ✓（注意力确实在跑 ✓ 不是恒等 ✓）",
          not t.allclose(full.last_hidden_state[:, 6], after.last_hidden_state[:, 6], atol=1e-6))
    check("④ ``pooled`` = ``last_hidden_state`` 里 **EOS 位置**那一行 ✓",
          t.equal(full.pooled, full.last_hidden_state[t.arange(1), t.tensor([4])]),
          "EOS 在位置 4 ✓")
    check("④ 两个 EOS 时取**第一个** ✓",
          t.equal(model(t.tensor([[1, 63, 2, 63, 0]]), eos_token_id=63).pooled,
                  model(t.tensor([[1, 63, 2, 63, 0]]), eos_token_id=63).last_hidden_state[
                      t.arange(1), t.tensor([1])]))
    check("④ 输入里**没有** EOS ⇒ 报错 ✓（不拿最后一个位置凑 ✗）",
          _raises(lambda: model(t.tensor([[1, 2, 3, 0, 0]]), eos_token_id=63), "eos") is not None)


def fake_open_clip_state(t: Any, cfg: Any) -> dict[str, Any]:
    """造一份**OpenCLIP 命名**的假权重 ✓（形状/键名照真权重 ✓ ⇒ 换算路径能整条跑 ✓）。"""
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    sd: dict[str, Any] = {
        "model.token_embedding.weight": t.randn(cfg.vocab_size, hidden) * 0.02,
        "model.positional_embedding": t.randn(cfg.max_position_embeddings, hidden) * 0.02,
        "model.ln_final.weight": t.randn(hidden), "model.ln_final.bias": t.randn(hidden),
        "model.logit_scale": t.tensor(4.6052),
    }
    if cfg.has_projection:
        sd["model.text_projection"] = t.randn(hidden, cfg.projection_dim) * 0.02
    for index in range(cfg.num_hidden_layers):
        head = f"model.transformer.resblocks.{index}."
        sd[head + "ln_1.weight"] = t.randn(hidden)
        sd[head + "ln_1.bias"] = t.randn(hidden)
        sd[head + "attn.in_proj_weight"] = t.randn(3 * hidden, hidden) * 0.02
        sd[head + "attn.in_proj_bias"] = t.randn(3 * hidden)
        sd[head + "attn.out_proj.weight"] = t.randn(hidden, hidden) * 0.02
        sd[head + "attn.out_proj.bias"] = t.randn(hidden)
        sd[head + "ln_2.weight"] = t.randn(hidden)
        sd[head + "ln_2.bias"] = t.randn(hidden)
        sd[head + "mlp.c_fc.weight"] = t.randn(inter, hidden) * 0.02
        sd[head + "mlp.c_fc.bias"] = t.randn(inter)
        sd[head + "mlp.c_proj.weight"] = t.randn(hidden, inter) * 0.02
        sd[head + "mlp.c_proj.bias"] = t.randn(hidden)
    return sd


def case_open_clip(t: Any) -> None:
    """⑤ OpenCLIP 命名换算 ✓ —— ⚠️ 转置方向错了形状**一样**、数值全错 ✗ ⇒ 必须有自算判据 ✓。"""
    cfg = tiny_config(name="tiny_g", projection_dim=8)
    raw = fake_open_clip_state(t, cfg)
    out = ct.open_clip_state_dict(raw, cfg)
    expected = set(ct.clip_text_key_names(cfg))
    check("⑤ 换算后键集 = 应有键集 + ``logit_scale`` ✓（不多不少 ✓）",
          set(out) - expected == {"logit_scale"}, sorted(set(out) - expected))
    head = "transformer.text_model.encoder.layers.1.self_attn."
    src = raw["model.transformer.resblocks.1.attn.in_proj_weight"]
    hidden = cfg.hidden_size
    check("⑤ ``in_proj_weight`` **三等分** 成 q/k/v ✓（顺序 q,k,v ✓）",
          t.equal(out[head + "q_proj.weight"], src[:hidden])
          and t.equal(out[head + "k_proj.weight"], src[hidden:2 * hidden])
          and t.equal(out[head + "v_proj.weight"], src[2 * hidden:]))
    check("⑤ ``in_proj_bias`` 也三等分 ✓",
          t.equal(out[head + "v_proj.bias"], raw["model.transformer.resblocks.1.attn.in_proj_bias"][2 * hidden:]))
    check("⑤ ``ln_1``→``layer_norm1`` / ``mlp.c_fc``→``mlp.fc1``（**带后缀**的键也要认 ✓）",
          "transformer.text_model.encoder.layers.1.layer_norm1.bias" in out
          and "transformer.text_model.encoder.layers.1.mlp.fc1.weight" in out
          and "transformer.text_model.encoder.layers.1.layer_norm2.weight" in out)
    check("⑤ ``positional_embedding``→``embeddings.position_embedding.weight`` ✓",
          t.equal(out["transformer.text_model.embeddings.position_embedding.weight"],
                  raw["model.positional_embedding"]))
    # ⚠️ **转置方向**的自算判据 ✓：OpenCLIP 用 ``x @ W`` ✓ ⇒ 换算后 ``nn.Linear`` 必须复现它 ✓。
    weight = out["text_projection.weight"]
    probe = t.randn(3, cfg.hidden_size)
    got = probe @ weight.T
    want = probe @ raw["model.text_projection"]
    check("⑤ ``text_projection`` **必须转置** ✓ —— 换算后 ``Linear(x)`` 要等于 OpenCLIP 的 ``x @ W`` ✓",
          t.allclose(got, want, atol=1e-5), float((got - want).abs().max()))
    # ⚠️ 只在 ``projection_dim == hidden_size`` 时（真 bigG 就是这样 ✗）「不转置」才**装得进去** ✓
    #    ⇒ 那才是会**静默**给出错条件的场景 ✗ ⇒ 单独反证一次 ✓。
    square = tiny_config(name="square", projection_dim=32)
    w_square = t.randn(square.hidden_size, square.projection_dim)
    fixed = ct.open_clip_state_dict({"model.text_projection": w_square}, square,
                                    )["text_projection.weight"]
    check("⑤ 同维情况下「不转置」与「转置」结果**不一样** ✓（⇒ 转置不是可省的一步 ✓）",
          tuple(fixed.shape) == (32, 32)
          and not t.allclose(probe @ w_square.T, probe @ fixed.T, atol=1e-5))
    check("⑤ 认出不了的块内键 ⇒ 报错 ✓（不静默跳过 ✗）",
          _raises(lambda: ct.open_clip_state_dict(
              {"model.transformer.resblocks.0.attn.q_proj.weight": t.randn(4, 4)}, cfg), "认不出") is not None)
    check("⑤ 层号 ≥ 配置层数 ⇒ 报错 ✓（配置与权重不是一套 ✓）",
          _raises(lambda: ct.open_clip_state_dict(
              {"model.transformer.resblocks.9.ln_1.weight": t.randn(4)}, cfg), "resblocks.9") is not None)
    check("⑤ ``in_proj`` 第 0 维不是 3×宽 ⇒ 报错 ✓",
          _raises(lambda: ct.open_clip_state_dict(
              {"model.transformer.resblocks.0.attn.in_proj_weight": t.randn(10, 4)}, cfg), "三等分") is not None)
    check("⑤ 无投影那塔收到 ``text_projection`` ⇒ 报错 ✓",
          _raises(lambda: ct.open_clip_state_dict(
              {"model.text_projection": t.randn(4, 4)}, tiny_config()), "text_projection") is not None)
    check("⑤ 前缀没有 ``model.`` ⇒ 报错 ✓（认不出就说认不出 ✗）",
          _raises(lambda: ct.open_clip_state_dict({"token_embedding.weight": t.randn(4, 4)}, cfg), "model.") is not None)


def case_load(t: Any) -> None:
    """⑥ 装载：三种前缀都认 ✓；缺键/多键/形状不符/前缀不明一律**报错** ✗。"""
    cfg = tiny_config(name="tiny_g", projection_dim=8)
    source = ct.build_clip_text_model(cfg)
    sd = {k: v.clone() for k, v in source.state_dict().items()}
    probe = t.tensor([[1, 2, 3, 4, 63, 0, 0], [5, 6, 63, 0, 0, 0, 0]])

    fresh = ct.build_clip_text_model(cfg)
    check("⑥ 已剥前缀（HF 命名 ✓）⇒ 装得上 ✓、无未用键 ✓",
          ct.load_clip_text_state_dict(fresh, sd) == []
          and t.equal(fresh(probe, eos_token_id=63).pooled, source(probe, eos_token_id=63).pooled))

    comfy = ct.build_clip_text_model(cfg)
    check("⑥ ComfyUI 内部叫法（``tiny_g.`` 前缀 ✓）也认 ✓",
          ct.load_clip_text_state_dict(comfy, {f"tiny_g.{k}": v for k, v in sd.items()}) == []
          and t.equal(comfy(probe, eos_token_id=63).pooled, source(probe, eos_token_id=63).pooled))

    flat = ct.build_clip_text_model(cfg)
    flat_sd = {(k[len("transformer."):] if k.startswith("transformer.") else k): v
               for k, v in sd.items()}
    check("⑥ 少一层（``text_model.`` 开头 ✓ 真 ``clip_l.safetensors`` 就是这样 ✓）也认 ✓",
          ct.load_clip_text_state_dict(flat, flat_sd) == []
          and t.equal(flat(probe, eos_token_id=63).pooled, source(probe, eos_token_id=63).pooled))

    missing = {k: v for k, v in sd.items() if k != "transformer.text_model.final_layer_norm.bias"}
    check("⑥ 缺一个键 ⇒ 报错 ✓（**不装半个模型** ✗ —— 那会留下随机初始化的层 ✗）",
          _raises(lambda: ct.load_clip_text_state_dict(ct.build_clip_text_model(cfg), missing), "缺") is not None)
    tally = ct.load_clip_text_state_dict(ct.build_clip_text_model(cfg), missing, strict=False)
    check("⑥ ``strict=False`` ⇒ 缺键**如实回报** ✓（不假装完整 ✓）",
          "transformer.text_model.final_layer_norm.bias" in tally, tally)
    check("⑥ 多出认不得的键 ⇒ 报错 ✓",
          _raises(lambda: ct.load_clip_text_state_dict(
              ct.build_clip_text_model(cfg), {**sd, "transformer.text_model.mystery.weight": t.randn(3)}),
              "认不得") is not None)
    bad_shape = dict(sd)
    bad_shape["transformer.text_model.final_layer_norm.bias"] = t.randn(7)
    check("⑥ 形状不符 ⇒ **中文**报错并指出键名 ✓",
          (_raises(lambda: ct.load_clip_text_state_dict(ct.build_clip_text_model(cfg), bad_shape),
                   "形状不符") is not None))
    with_buffer = {**sd, "transformer.text_model.embeddings.position_ids": t.arange(7).view(1, 7)}
    check("⑥ 真权重里的常量缓冲 ``position_ids`` ⇒ 装得上且**如实回报**为未用键 ✓",
          ct.load_clip_text_state_dict(ct.build_clip_text_model(cfg), with_buffer)
          == ["transformer.text_model.embeddings.position_ids"])
    check("⑥ 前缀完全认不出 ⇒ 报错 ✓（不猜 ✗）",
          _raises(lambda: ct.load_clip_text_state_dict(
              ct.build_clip_text_model(cfg), {f"zzz.{k}": v for k, v in sd.items()}), "认不出") is not None)
    check("⑥ 没装进本模块建的模型 ⇒ 报错 ✓",
          _raises(lambda: ct.load_clip_text_state_dict(t.nn.Linear(3, 3), sd), "config") is not None)


def case_sdxl(t: Any) -> None:
    """⑦ SDXL 拼接：2048 宽 / 各自截到较短的 / pooled 只取 bigG 的**投影后** ✓。"""
    l_cfg = tiny_config(name="l", hidden_size=8, num_attention_heads=2, intermediate_size=16)
    g_cfg = tiny_config(name="g", hidden_size=8, num_attention_heads=2, intermediate_size=16,
                        projection_dim=8)
    l_model, g_model = ct.build_clip_text_model(l_cfg), ct.build_clip_text_model(g_cfg)
    ids7 = t.tensor([[1, 2, 3, 4, 5, 6, 63]])
    ids5 = t.tensor([[1, 2, 3, 63, 0]])
    l_out = l_model(ids7, layer_idx=-2, layer_norm_hidden_state=False, eos_token_id=63)
    g_out = g_model(ids5, layer_idx=-2, layer_norm_hidden_state=False, eos_token_id=63)
    context, pooled = ct.encode_sdxl_conditioning(l_out, g_out, expected_dim=16)
    check("⑦ ``context`` 宽 = 两塔宽之和（8+8=16）✓，长度 = **较短**那个（5 ✓）",
          tuple(context.shape) == (1, 5, 16), tuple(context.shape))
    check("⑦ ``context`` = ``cat([l[:, :5], g[:, :5]], -1)`` ✓ 逐值对 ✓",
          t.allclose(context, t.cat([l_out.selected[:, :5], g_out.selected[:, :5]], dim=-1), atol=1e-6))
    check("⑦ ``pooled`` 取的是 bigG 的**投影后**结果 ✓（不是 L 的、也没拿未投影的凑 ✗）",
          t.allclose(pooled, g_out.projected_pooled, atol=1e-6)
          and tuple(pooled.shape) == (1, 8))
    check("⑦ 不传 ``expected_dim`` ⇒ 按 **SDXL 的 2048** 核宽 ✓ ⇒ 缩小版**报错** ✓（默认是响亮的 ✓）",
          _raises(lambda: ct.encode_sdxl_conditioning(l_out, g_out), "2048") is not None)
    no_proj = ct.build_clip_text_model(tiny_config(name="np", hidden_size=8, num_attention_heads=2,
                                                   intermediate_size=16))
    np_out = no_proj(ids5, eos_token_id=63)
    check("⑦ bigG 那塔没有投影 ⇒ 报错 ✓（不拿未投影的 pooled 凑 ✗）",
          _raises(lambda: ct.encode_sdxl_conditioning(np_out, np_out, expected_dim=16),
                  "text_projection") is not None)
    check("⑦ 批大小不一致 ⇒ 报错 ✓",
          _raises(lambda: ct.encode_sdxl_conditioning(l_out, g_model(
              t.cat([ids5, ids5], 0), eos_token_id=63), expected_dim=16), "批大小") is not None)
    check("⑦ 传的不是 ``ClipTextOutput`` ⇒ 报错 ✓",
          _raises(lambda: ct.encode_sdxl_conditioning(t.randn(2, 3), g_out), "ClipTextOutput") is not None)


def case_token_ids(t: Any) -> None:
    """⑧ 定长 id 组装：BOS/EOS/pad ✓、**两塔 pad 不同** ✗、超长裁剪**如实报告** ✓。"""

    class FakeTokenizer:
        """只实现 ``encode`` ✓（引擎不内置词表 ✓ 与 `text_encoder` 同一条纪律 ✓）。"""

        vocab_size = 49408

        def encode(self, text: str) -> list[int]:
            return [100 + i for i in range(len(text))]

    ids, truncated = ct.sdxl_token_ids(FakeTokenizer(), "abc", ct.CLIP_L)
    check("⑧ 长度 = 77 ✓、首位 BOS(49406) ✓、末位 EOS(49407) ✓、中间是编码 ✓",
          len(ids) == 77 and ids[0] == 49406 and ids[1:4] == [100, 101, 102]
          and ids[4] == 49407 and not truncated, ids[:6])
    check("⑧ CLIP-L 的 pad = **49407**（= eos ✓）⇒ 尾部全 49407 ✓",
          set(ids[4:]) == {49407})
    g_ids, _ = ct.sdxl_token_ids(FakeTokenizer(), "abc", ct.CLIP_G)
    check("⑧ bigG 的 pad = **0** ✗（两塔口径不一样 ✓ 出处 `sdxl_clip.py` ✓）",
          g_ids[5:] == [0] * 72 and g_ids[0] == 49406 and g_ids[4] == 49407, g_ids[:6])
    long_ids, cut = ct.sdxl_token_ids(FakeTokenizer(), "x" * 200, ct.CLIP_L)
    check("⑧ 超长 ⇒ 裁到 75 个 token + BOS/EOS ✓ 且 **如实回报** ``truncated=True`` ✓",
          len(long_ids) == 77 and cut is True and long_ids[76] == 49407
          and long_ids[0] == 49406 and long_ids[75] == 100 + 74)
    check("⑧ ``truncate=False`` 时超长 ⇒ 报错 ✓（不静默改输入 ✗）",
          _raises(lambda: ct.sdxl_token_ids(FakeTokenizer(), "x" * 200, ct.CLIP_L, truncate=False),
                  "超过上限") is not None)
    check("⑧ ``max_length`` 超过位置嵌入上限 ⇒ 报错 ✓",
          _raises(lambda: ct.sdxl_token_ids(FakeTokenizer(), "a", ct.CLIP_L, max_length=99), "上限") is not None)
    check("⑧ ``max_length`` 太小 ⇒ 报错 ✓",
          _raises(lambda: ct.sdxl_token_ids(FakeTokenizer(), "a", ct.CLIP_L, max_length=2), "太小") is not None)
    class OutOfVocabTokenizer:
        def encode(self, _text: str) -> list[int]:
            return [999999]

    check("⑧ tokenizer 给的 id 越出词表 ⇒ 报错 ✓",
          _raises(lambda: ct.sdxl_token_ids(OutOfVocabTokenizer(), "a", ct.CLIP_L), "词表") is not None)
    check("⑧ tokenizer 不会 ``encode`` ⇒ 报错 ✓",
          _raises(lambda: ct.sdxl_token_ids(object(), "a", ct.CLIP_L), "encode") is not None)


class _ShapeOnly:
    """只带形状的假张量 ✓ —— 真权重**只读头**就能核对键集与切分形状 ✓（不载 1.4 GB ✗）。"""

    def __init__(self, shape: Any) -> None:
        self.shape = [int(v) for v in shape]

    def __getitem__(self, key: Any) -> "_ShapeOnly":
        start, stop, step = key.indices(self.shape[0])
        return _ShapeOnly([len(range(start, stop, step))] + self.shape[1:])

    def transpose(self, a: int, b: int) -> "_ShapeOnly":
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return _ShapeOnly(shape)

    def contiguous(self) -> "_ShapeOnly":
        return self


def _find_weight(pattern: str) -> Any:
    """在本机 ComfyUI 根里找权重 ✓ —— 找不到返回 ``None`` ⇒ 调用方 SKIP ✓（自检不依赖真机 ✓）。"""
    try:
        from app.services.engine.safetensors import read_header  # noqa: PLC0415
        from app.services.local_model_scan import get_default_roots  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    for root in get_default_roots():
        for path in Path(root).glob(f"**/{pattern}"):
            try:
                _, header = read_header(path)
            except Exception:  # noqa: BLE001
                continue
            return path, header
    return None


def case_real_weights(t: Any) -> None:
    """⑨ 本机真权重**只读头**逐键核对 ✓（没有就显式 SKIP ✓ —— SKIP 不是通过 ✗）。"""
    found = _find_weight("sd_xl_base_1.0*.safetensors")
    if found is None:
        skip("本机没找到 `sd_xl_base_1.0.safetensors` ⇒ 真权重键集核对不跑 ✓（显式 SKIP ✓）")
    else:
        path, header = found
        ignored = {"transformer.text_model.embeddings.position_ids", "logit_scale"}
        for cfg in (ct.CLIP_L, ct.CLIP_G):
            scoped = {k[len(cfg.load_prefix):]: _ShapeOnly(v["shape"])
                      for k, v in header.items() if k.startswith(cfg.load_prefix)}
            norm = ct._normalize_clip_state_dict(scoped, cfg)  # noqa: SLF001
            expected = set(ct.clip_text_key_names(cfg))
            check(f"⑨ {cfg.name}：真权重换算后**缺键 0**、多出的只有常量缓冲/``logit_scale`` ✓",
                  not (expected - set(norm)) and not (set(norm) - expected - ignored),
                  (len(scoped), len(norm), sorted(set(norm) - expected - ignored)[:3]))
        scoped_g = {k[len(ct.CLIP_G.load_prefix):]: _ShapeOnly(v["shape"])
                    for k, v in header.items() if k.startswith(ct.CLIP_G.load_prefix)}
        norm_g = ct._normalize_clip_state_dict(scoped_g, ct.CLIP_G)  # noqa: SLF001
        check("⑨ bigG：``in_proj`` 三分后 q/k/v 都是 ``(1280,1280)`` ✓、投影 ``(1280,1280)`` ✓",
              tuple(norm_g["transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"].shape) == (1280, 1280)
              and tuple(norm_g["text_projection.weight"].shape) == (1280, 1280))
        check("⑨ CLIP-L：``position_ids`` 确实在权威里 ✓（所以它进「未用键」是**如实回报** ✓）",
              "conditioner.embedders.0.transformer.text_model.embeddings.position_ids" in header)

    single = _find_weight("clip_l.safetensors")
    if single is None:
        skip("本机没找到官方单发的 `clip_l.safetensors` ⇒ 「少一层前缀」核对不跑 ✓（显式 SKIP ✓）")
    else:
        path, header = single
        expects = {n[len("transformer."):] if n.startswith("transformer.") else n
                   for n in ct.clip_text_key_names(ct.CLIP_L)}
        check("⑨ 官方单发 ``clip_l.safetensors`` 键**正好**是应有 196 个 ✓"
              "（命名是 ``text_model.*`` ✓ ⇒ 装载时补一层 ``transformer.`` ✓）",
              set(header) == expects, (len(header), sorted(set(header) - expects)[:2], Path(path).name))


def main() -> int:
    t = torch_or_skip()
    if t is not None:
        case_config(t)
        case_key_names(t)
        case_forward(t)
        case_causal_pool(t)
        case_open_clip(t)
        case_load(t)
        case_sdxl(t)
        case_token_ids(t)
        case_real_weights(t)
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK** ⇒ 不收编就是 UnicodeEncodeError 崩 ✗。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
