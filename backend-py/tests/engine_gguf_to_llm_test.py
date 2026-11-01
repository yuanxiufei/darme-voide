"""自检：GGUF 权重 → LlmModel 的**装载**（``engine/gguf_to_llm.py`` ✓ 2026-09-25 起 ✓）。

验的是什么 ✗：把 llama.cpp 的 ggml 命名（``blk.N.attn_q.weight`` ✓）映射成 LlmModel 命名
（``blocks.N.attn.q_proj.weight`` ✓）+ 线性层**转置** ✓（GGUF (in,out) → PyTorch (out,in) ✓）。

判据（都是「看着装上了、其实权重接错了」的形状 ✓✗）：
1. ⭐⭐⭐ **往返恒等** ✗✗：LlmModel 的 ``state_dict`` → 转回 GGUF 命名（反向映射 + 反向转置）→
   再正向映射回来 ⇒ 与原始 ``state_dict`` **逐张量相等** ✓ —— 映射或转置错一处，就有一处对不上 ✗✗；
2. ⭐ **tie embeddings** ✓：tie 时 GGUF 的 ``output.weight`` **跳过** ✓（LlmModel 已 tie ✓）；
3. ⭐ **未知名字返回 None** ✓（不抛错 ✓ —— 真「缺不漏」由装配后的**键集对账**兜底 ✓）；
4. ⭐ **流式装配**（2026-10-29 ✓）：峰内存从 3× 权重降到 1× ✓，但 ``load_state_dict(strict=True)``
   那句随之没了 ✗ ⇒ 缺 / 多 / 形状不符三条都得**自己报** ✓（本文件 ②⁵ 专测，三条各一段 ✓）。

⚠️ 本套**不宣称**能装真 Qwen3 GGUF ✗（真文件的张量名可能带 ``.bias``/``.scale`` 等后缀 ✓ 未测 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_gguf_to_llm_test.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="g2l_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import gguf_to_llm as g2l  # noqa: E402
from app.services.engine import llm as llm_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: LlmModel 参数名 → GGUF 张量名（反向 ✓ 独立于实现的正向映射 ✓）
def _reverse(name: str) -> str:
    if name == "embed.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    m = re.match(r"blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)", name)
    if m:
        suffix = {"q_proj": "attn_q", "k_proj": "attn_k",
                  "v_proj": "attn_v", "o_proj": "attn_output"}[m.group(2)]
        return f"blk.{m.group(1)}.{suffix}.{m.group(3)}"
    m = re.match(r"blocks\.(\d+)\.ffn\.(gate|up|down)\.weight", name)
    if m:
        return f"blk.{m.group(1)}.ffn_{m.group(2)}.weight"
    m = re.match(r"blocks\.(\d+)\.norm([12])\.weight", name)
    if m:
        return f"blk.{m.group(1)}.{'attn_norm' if m.group(2) == '1' else 'ffn_norm'}.weight"
    m = re.match(r"blocks\.(\d+)\.attn\.(q_norm|k_norm)\.weight", name)
    if m:
        return f"blk.{m.group(1)}.attn_{m.group(2)[0]}_norm.weight"   # q_norm → attn_q_norm ✓
    raise AssertionError(f"未知 LlmModel 参数名 {name}")


def case_map() -> None:
    check("① 命名映射：token_embd → embed ✓ / output_norm → norm ✓",
          g2l.map_name("token_embd.weight", tie_embeddings=False) == "embed.weight"
          and g2l.map_name("output_norm.weight", tie_embeddings=False) == "norm.weight")
    check("①′ 命名映射：blk.N.attn_q → blocks.N.attn.q_proj ✓（层号透传 ✓）",
          g2l.map_name("blk.7.attn_q.weight", tie_embeddings=False)
          == "blocks.7.attn.q_proj.weight")
    check("①″ 命名映射：output → lm_head ✓（非 tie ✓）",
          g2l.map_name("output.weight", tie_embeddings=False) == "lm_head.weight")
    check("①‴ ⭐ 未知名字 → None ✓（不抛错 ✓）",
          g2l.map_name("blk.0.some_unknown.weight", tie_embeddings=False) is None)
    # ⭐ Qwen3 的 QK-norm（2026-10-29 真机撞出来 ✓）：**开关控制**，不是无脑多映射两条 ✗
    # （qk_norm=False 的模型若也映射 ⇒ 多出 key ⇒ strict 报 unexpected ✗✗）
    check("①⁴ ⭐ QK-norm：`qk_norm=True` 才映射 `attn_q_norm/attn_k_norm` ✓（默认 False ⇒ None ✓）",
          g2l.map_name("blk.3.attn_q_norm.weight", tie_embeddings=True) is None
          and g2l.map_name("blk.3.attn_k_norm.weight", tie_embeddings=True) is None
          and g2l.map_name("blk.3.attn_q_norm.weight", tie_embeddings=True, qk_norm=True)
          == "blocks.3.attn.q_norm.weight"
          and g2l.map_name("blk.3.attn_k_norm.weight", tie_embeddings=True, qk_norm=True)
          == "blocks.3.attn.k_norm.weight")


def _torch_available() -> bool:
    """能不能真建 ``LlmModel`` ✓（``llm._build`` 里 ``import torch`` ✓）。

    ⚠️ 缺 torch 时**必须显式 SKIP 打印** ✓ —— 原来这一套会整片崩在 ``build_llm`` ✗
    ⇒ 连**不需要 torch** 的命名映射 / config 推导（纯逻辑 ✓）也跟着一条都跑不了 ✗✗。
    SKIP ≠ 通过 ✓（``SUMMARY`` 的分母里没有它 ✓，别把「没跑」读成「绿了」✗）。
    """
    try:
        import torch  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        return False
    return True


def case_roundtrip() -> None:
    if not _torch_available():
        print("SKIP  ② 往返恒等：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=2, heads=2, kv_heads=1,
                               head_dim=8, ffn=32, tie_embeddings=False)
    model = llm_mod.build_llm(config)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 反向：LlmModel → GGUF（独立实现 ✓，**按 ggml 的事实写死** ✓）
    # ⚠️ GGUF 存的是一维流，元素序 ``idx = i0 + ne0*i1``（``ne0`` 最内、变化最快 ✓）⇒
    # 对 ``(out, in)`` 的 PyTorch 权重，GGUF 的 ``dims`` 就是 ``(in, out)`` ✓ 而其 flat 流
    # = ``data.flatten()``（C 序 ⇒ 行主序按 (out,in) ⇒ ``i0``=in 先变 ✓ 恰好一致 ✓）。
    # ⚠️ 这里**不能**再用「2D 正序 + .T」表示 ✗✗：那条路对非方阵根本不可逆（矩阵不是方阵时
    # ``reshape(in,out).T`` ≠ 任何一致的摆法 ✗），原实现就是这么把元素整张重排的 ✗（真机 8B 全程乱码 ✗）。
    gguf_weights = {}
    for name, tensor in state.items():
        data = tensor.numpy()
        gguf_weights[_reverse(name)] = (tuple(reversed(data.shape)), data.flatten())

    # 正向：GGUF → LlmModel
    state2 = g2l.gguf_state_dict(gguf_weights, config)

    import torch  # noqa: PLC0415
    ok = set(state2) == set(state) and all(
        tuple(state2[k].shape) == tuple(state[k].shape) and torch.equal(state2[k], state[k])
        for k in state)
    check("② ⭐⭐⭐ 往返恒等 ✓：state_dict → GGUF → 映射回 ⇒ **逐张量相等** ✗✗",
          ok, sorted(set(state) - set(state2)) or sorted(set(state2) - set(state)))


def case_qk_norm() -> None:
    """⭐ Qwen3 的 per-head QK 归一化（2026-10-29 补 ✓）：**结构对不对** + **真有没有接进前向** ✓。

    两条都要 ✓：
    ⓐ 往返恒等里 ``qk_norm=True`` 的参数必须**一个不多一个不少**（``set`` 相等 ✓）——
       这正是真机 8B 装不进的根因（72 个张量无家可归 ✗）；
    ⓑ 把 ``q_norm.weight`` 乘 2 ⇒ logits **必须变** ✓ —— 只加了参数却忘了在 forward 里用，
       形状照样对 ✓、strict 照样过 ✓、**效果全错** ✗✗，只有这条能照出来 ✓。
    """
    if not _torch_available():
        print("SKIP  ③⁗ QK-norm：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    import torch  # noqa: PLC0415

    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=2, heads=4, kv_heads=2,
                               head_dim=8, ffn=32, tie_embeddings=False, qk_norm=True)
    model = llm_mod.build_llm(config)
    weights = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().numpy()
        weights[_reverse(name)] = (tuple(reversed(data.shape)), data.flatten())

    state = g2l.gguf_state_dict(weights, config)
    check("③⁗ ⭐ QK-norm 往返：`q_norm`/`k_norm` 都在映射内，且参数集**严格相等** ✓（多一个就装不进 ✗）",
          set(state) == set(model.state_dict())
          and any(name.endswith("attn.q_norm.weight") for name in state),
          sorted(set(model.state_dict()) - set(state)) or sorted(set(state) - set(model.state_dict())))

    restored = llm_mod.build_llm(config)
    restored.load_state_dict(state, strict=True)

    ids = torch.tensor([[1, 5, 9, 17, 3]])

    def _logits(out):
        """``LlmModel.forward`` 返回 ``(logits, cache)`` ✓ —— 取第 0 项 ✓（不猜结构 ✗，按返回值来 ✓）。"""
        return out[0] if isinstance(out, tuple) else out

    with torch.no_grad():
        before = _logits(restored(ids)).clone()
        restored.blocks[0].attn.q_norm.weight.mul_(2.0)
        after = _logits(restored(ids))
    check("③⁗′ ⭐ 把 `blocks[0].attn.q_norm.weight` 乘 2 ⇒ **logits 必须变** ✓（只存不用 ⇒ 这条红 ✗✗）",
          not torch.allclose(before, after, atol=1e-6),
          float((before - after).abs().max()))


def case_qkv_bias() -> None:
    """⭐ Qwen2 / Qwen2.5 的 q/k/v **偏置**（2026-10-31 补 ✓）：每层三根 ``attn_{q,k,v}.bias`` ✓。

    为什么必须单独立一条 ✗：漏掉这三族**不会报错** ✓ —— ``LlmConfig.qkv_bias`` 默认 ``False``
    ⇒ 模型压根不建 bias 参数 ⇒ 那三根 GGUF 键被当「未知名字」跳过 ✓，键集对账也全绿 ✓✗。
    后果只有「每层少一个常数」⇒ 提示词跟随变弱 ✓✗（静默降质）。
    """
    check("③⁘ `attn_q.bias` 只在 `qkv_bias=True` 时映射 ✓（Qwen3/Llama 没这三根 ⇒ 必须先看张量表 ✗）",
          g2l.map_name("blk.0.attn_q.bias", tie_embeddings=False) is None
          and g2l.map_name("blk.0.attn_q.bias", tie_embeddings=False, qkv_bias=True)
          == "blocks.0.attn.q_proj.bias", None)
    if not _torch_available():
        print("SKIP  ③⁘′ 偏置往返：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    import torch  # noqa: PLC0415

    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=2, heads=4, kv_heads=2,
                               head_dim=8, ffn=32, tie_embeddings=False, qkv_bias=True)
    model = llm_mod.build_llm(config)
    weights = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().numpy()
        weights[_reverse(name)] = (tuple(reversed(data.shape)), data.flatten())

    state = g2l.gguf_state_dict(weights, config)
    check("③⁘ᵃ `detect_features` 按张量表把 `qkv_bias` 翻上来 ✓（装载阶段**真会**这么做，不只映射表支持 ✗）",
          g2l.detect_features(
              llm_mod.LlmConfig(vocab_size=8, hidden=8, depth=1, heads=1, kv_heads=1,
                                 head_dim=8, ffn=16),
              ["blk.0.attn_q.bias", "blk.0.attn_q.weight"]).qkv_bias
          and not g2l.detect_features(
              llm_mod.LlmConfig(vocab_size=8, hidden=8, depth=1, heads=1, kv_heads=1,
                                head_dim=8, ffn=16),
              ["blk.0.attn_q.weight"]).qkv_bias,
          None)

    check("③⁘′ 偏置往返：**三条 bias 一张不少** ✓（少装 ⇒ 每层少一个常数、静默降质 ✗✗）",
          set(state) == set(model.state_dict())
          and sum(name.endswith(".bias") for name in state) == 3 * config.depth,
          sorted(set(model.state_dict()) - set(state)) or sorted(set(state) - set(model.state_dict())))

    restored = llm_mod.build_llm(config)
    restored.load_state_dict(state, strict=True)
    ids = torch.tensor([[1, 5, 9, 17, 3]])

    def _logits(out):
        return out[0] if isinstance(out, tuple) else out

    with torch.no_grad():
        before = _logits(restored(ids)).clone()
        restored.blocks[0].attn.q_proj.bias.add_(1.0)
        after = _logits(restored(ids))
    check("③⁘″ 给 `blocks[0].attn.q_proj.bias` 加 1 ⇒ **logits 必须变** ✓（建了参数却没用 ⇒ 这条红 ✗✗）",
          not torch.allclose(before, after, atol=1e-6),
          float((before - after).abs().max()))


def case_transpose() -> None:
    """⭐ 非方阵的**元素摆法**（2026-10-29 真机撞出来 ✓）：GGUF 一维流的元素序 ``i0 + ne0*i1``。

    ⚠️ 为什么必须单独立一条：往返用例里「反向构造」和「正向映射」只要**错得一样**就仍然恒等 ✗✗
    （实测真发生过 ✓）。所以这里**只给 ggml 事实** ✓：``dims=(4,6)`` + ``flat=0..23``，
    然后逐格断言位置 ✓ —— 形状对不对是次要的 ✗，**哪个值落在哪一格**才是判据 ✓。

    ★ 2026-10-29 事故的正是这条：旧实现 ``flat.reshape(4,6).T`` 与正确摆法
    ``flat.reshape(6,4)`` **形状一模一样（都是 6×4）** ✗ ⇒ 形状断言抓不住 ✗，
    只有「值落在哪一格」能抓住 ✓（下面第二条专门把旧错法的结果算出来做反证 ✓）。
    """
    if not _torch_available():
        print("SKIP  ②′ 摆法：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    import numpy as np  # noqa: PLC0415

    config = llm_mod.LlmConfig(vocab_size=6, hidden=4, depth=1, heads=2, kv_heads=1,
                               head_dim=2, ffn=8, tie_embeddings=False)
    flat = np.arange(24, dtype=np.float32)                        # ggml 顺序的一维流 ✓
    dims = (4, 6)                                                 # (hidden, vocab) ✓
    state = g2l.gguf_state_dict({"token_embd.weight": (dims, flat),
                                 "output.weight": (dims, flat.copy())},
                                config)
    embed = state["embed.weight"].numpy()
    head = state["lm_head.weight"].numpy()
    right = flat.reshape(6, 4)                                    # (vocab, hidden) ✓
    check("②′ ⭐ GGUF `dims=(4,6)` + `flat=0..23` ⇒ `embed.weight` 必须是 `flat.reshape(6,4)` ✓"
          "（逐格：embed[0][0]=0, embed[0][1]=1 —— hidden 维变化最快 ✓）",
          embed.shape == (6, 4) and np.array_equal(embed, right),
          (embed.shape, embed[0][:3].tolist()))
    check("②″ ⭐ `output.weight` 同样按同一个流摆成 `lm_head.weight` 6×4 ✓（非 tie ✓）",
          head.shape == (6, 4) and np.array_equal(head, right), head.shape)
    # 反证：旧错法（正序 reshape 再 .T）形状也对 ✗ ⇒ 必须与正确摆法**不相等**，否则这条用例形同虚设 ✗
    wrong = flat.reshape(4, 6).T
    check("②‴ ⭐ 反证：旧错法 `flat.reshape(4,6).T` 与正确摆法**形状相同但元素不同** ✓"
          "（若两者恰好相等 ⇒ 这条用例没有分辨力 ✗✗）",
          wrong.shape == embed.shape and not np.array_equal(wrong, embed),
          (wrong.shape, wrong[0][:3].tolist()))


def case_tie() -> None:
    if _torch_available():
        config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=2,
                                   head_dim=8, ffn=32, tie_embeddings=True)
        model = llm_mod.build_llm(config)
        check("③ ⭐ tie embeddings ✓（lm_head.weight is embed.weight ✓）",
              model.lm_head.weight is model.embed.weight)
    else:
        print("SKIP  ③ tie embeddings：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
    # ③′ 是纯映射逻辑 ✓ ⇒ 缺 torch 也照跑 ✓（不该陪着一起 skip ✗）
    check("③′ tie 时 GGUF output.weight **跳过** ✓",
          g2l.map_name("output.weight", tie_embeddings=True) is None)


def case_stream_load() -> None:
    """⭐ 流式装配 + 键集对账（2026-10-29 ✓，真机 8B 撞出来的内存事故 ✓）。

    ⚠️ 原实现：**全部**张量反量化进 dict → 整份转 torch → ``load_state_dict`` ⇒ 峰值 = fp32 权重 × 3
    （8B ≈ 98 GB ✗ 在 96 GB 机器上进交换区，实测工作集 86 GB 还没装完 ✗✗）。
    新实现：逐张量 ``copy_`` 进参数 ✓ 峰值 = 1× ✓ —— 但 ``strict=True`` 那句也一起没了 ✗
    ⇒ 「缺 key / 多 key / 形状不符」三条现在**由自己报** ✓，所以必须逐条钉住 ✓（否则从「装不进的模型」
    退化成「静默装半套」✗✗，那比原来的内存问题更坏 ✗）。

    本用例不需要真 GGUF 文件 ✓：``_read_tensor`` 换成替身（直接回吐数组 ✓），只验装配与对账 ✓。
    """
    if not _torch_available():
        print("SKIP  ②⁵ 流式装配：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    import types  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=2, heads=2, kv_heads=1,
                               head_dim=8, ffn=32, tie_embeddings=False)
    model = llm_mod.build_llm(config)

    def _arr(shape: tuple[int, ...], *, start: float = 1.0) -> "np.ndarray":
        """按形状造**互不相同**的值 ✓ —— 全零/同值会让「没装上」与「装上了」看起来一样 ✗。"""
        count = int(np.prod(shape))
        return (np.arange(count, dtype=np.float32) + start).reshape(shape)

    # ``_read_tensor`` 的真实契约是「已按 ``reversed(dims)`` 摆成 numpy 序的数组」✓
    # ⇒ 替身直接回吐 PyTorch 形状的数组即可 ✓（这里不许再手动转置 ✗：那等于把
    #   「读取器摆错」这件事从用例里抹掉 ✗✗）。
    payload: dict[str, "np.ndarray"] = {}
    for name, tensor in model.state_dict().items():
        payload[_reverse(name)] = _arr(tuple(tensor.shape))
    info = types.SimpleNamespace(tensors=dict(payload))   # 值就是数组本身 ✓ 替身读取器原样回吐 ✓

    original_read = g2l._read_tensor
    g2l._read_tensor = lambda path, info, tensor: tensor   # type: ignore[assignment]
    try:
        g2l._load_into_model("<fake>", info, model, config)
        got = model.state_dict()
        ok = all(torch.equal(got[name], torch.from_numpy(_arr(tuple(got[name].shape))))
                 for name in got)
        check("②⁵ ⭐ 流式装配：逐张量 ``copy_`` 后**每个参数都对**（含 embed/lm_head ✓）",
              ok, sorted(name for name in got
                         if not torch.equal(got[name], torch.from_numpy(_arr(tuple(got[name].shape))))))

        # ⓐ 缺一个张量 ⇒ 必须报「缺」（原 strict=True 的 missing key ✓）
        short = dict(payload)
        short.pop("blk.1.ffn_down.weight")
        info_short = types.SimpleNamespace(tensors=short)
        check("②⁵′ 少一个张量 ⇒ **具名拒绝** ✗（缺 key 不许静默过 ✗✗）",
              _message_of(lambda: g2l._load_into_model(
                  "<fake>", info_short, llm_mod.build_llm(config), config)).find("缺") >= 0)

        # ⓑ 映射出模型没有的参数 ⇒ 必须报「没有」（原 strict=True 的 unexpected key ✓）
        original_map = g2l.map_name
        g2l.map_name = lambda name, **kw: ("blocks.0.bogus.weight"
                                           if name == "output.weight" else original_map(name, **kw))
        try:
            message = _message_of(lambda: g2l._load_into_model(
                "<fake>", types.SimpleNamespace(tensors=dict(payload)),
                llm_mod.build_llm(config), config))
        finally:
            g2l.map_name = original_map
        check("②⁵″ 映射出模型没有的参数 ⇒ **具名拒绝** ✗（多 key 不许静默丢 ✗✗）",
              "没有" in message, message)
    finally:
        g2l._read_tensor = original_read   # type: ignore[assignment]

    # ⓒ 形状不符 ⇒ 必须报（这一条是 strict=True 也会报的 ✓，流式版要自己报 ✓）
    bad = dict(payload)
    bad["blk.0.attn_q.weight"] = np.zeros((3, 3), dtype=np.float32)
    g2l._read_tensor = lambda path, info, tensor: tensor   # type: ignore[assignment]
    try:
        message = _message_of(lambda: g2l._load_into_model(
            "<fake>", types.SimpleNamespace(tensors=bad), llm_mod.build_llm(config), config))
    finally:
        g2l._read_tensor = original_read   # type: ignore[assignment]
    check("②⁵‴ 形状不符 ⇒ 报「形状不符」✗（别把转置方向错当成装上了 ✗✗）",
          "形状不符" in message, message)


def _message_of(fn) -> str:
    """跑一遍、**取异常消息** ✓（不是「抛了就过」✗ —— 理由对不上等于没修 ✓）。"""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return ""


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001 —— GgufDequantError / LlmError 都算「按预期拒」✓
        return True
    return False


def _rejects_with(fn, exc_name: str) -> bool:
    """「拒了」不够 ✓ —— 还要**拒得对名** ✓（``NameError`` 也算拒了 ✗ 却把原因丢了 ✗✗）。"""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__ == exc_name
    return False


def case_infer() -> None:
    meta = {
        "general.architecture": "llama",
        "llama.vocab_size": 32, "llama.embedding_length": 16, "llama.block_count": 2,
        "llama.attention.head_count": 2, "llama.attention.head_count_kv": 1,
        "llama.feed_forward_length": 32, "llama.rope.freq_base": 100.0,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.eos_token_id": 31,
    }
    config = g2l.infer_llm_config(meta)
    check("④ infer: 完整元数据 → LlmConfig ✓（GQA kv_heads=1 ✓ head_dim=hidden/heads=8 ✓ eos ✓）",
          config.vocab_size == 32 and config.hidden == 16 and config.depth == 2
          and config.heads == 2 and config.kv_heads == 1 and config.head_dim == 8
          and config.ffn == 32 and config.eos_id == 31 and config.rms_eps == 1e-5,
          config)

    meta3 = {
        "general.architecture": "qwen3",
        "qwen3.vocab_size": 64, "qwen3.embedding_length": 32, "qwen3.block_count": 4,
        "qwen3.attention.head_count": 4, "qwen3.feed_forward_length": 64,
    }
    config3 = g2l.infer_llm_config(meta3)
    check("④′ infer: qwen3 前缀 ✓（kv_heads 缺省=head ✓ head_dim 推导=8 ✓）",
          config3.vocab_size == 64 and config3.hidden == 32
          and config3.kv_heads == 4 and config3.head_dim == 8, config3)

    meta_fb = {"general.architecture": "qwen3", "llama.vocab_size": 32,
               "llama.embedding_length": 16, "llama.block_count": 1,
               "llama.attention.head_count": 2, "llama.feed_forward_length": 32}
    config_fb = g2l.infer_llm_config(meta_fb)
    check("④″ infer: qwen3 缺字段 ⇒ **回退 llama 前缀** ✓", config_fb.vocab_size == 32)

    check("④‴ infer: 缺必需字段 ⇒ 具名拒绝 ✗（不猜默认 ✗）",
          _raises(lambda: g2l.infer_llm_config({"general.architecture": "llama"})))
    check("④⁴ infer: head_dim 推不出（hidden 不被 heads 整除）⇒ 拒 ✗",
          _raises(lambda: g2l.infer_llm_config({
              "general.architecture": "llama", "llama.vocab_size": 32,
              "llama.embedding_length": 15, "llama.block_count": 1,
              "llama.attention.head_count": 2, "llama.feed_forward_length": 32})))

    # ⭐ 2026-10-29 真机撞出来的三条（qwen3:8b 的 ollama blob 实测 ✓）：GGUF **不写**
    # ``{arch}.vocab_size`` ✗，词表大小只能从 ``tokenizer.ggml.tokens`` 数 ✓；而
    # :func:`gguf.inspect` 对数组是**截断保存** ✓（``{"__array__": "string", "len": N, "head": 16}``）✗
    # ⇒ 写 ``len(tokens)`` 会得到 **3**（三个键）而不是 151936 ✗✗（实测真踩过 ✓）。
    _common = {"general.architecture": "qwen3", "qwen3.embedding_length": 4096,
               "qwen3.block_count": 36, "qwen3.attention.head_count": 32,
               "qwen3.attention.head_count_kv": 8, "qwen3.feed_forward_length": 12288}
    meta_tokens = dict(_common, **{"tokenizer.ggml.tokens": {
        "__array__": "string", "len": 151936, "head": ["<unk>", "<s>", "</s>"]}})
    check("④⁵ infer: 没有 vocab_size ⇒ 退回 ``tokenizer.ggml.tokens`` 的 **len**（截断 dict 形态 ✓=151936）",
          g2l.infer_llm_config(meta_tokens).vocab_size == 151936,
          g2l.infer_llm_config(meta_tokens).vocab_size)
    meta_list = dict(_common, **{"tokenizer.ggml.tokens": ["a"] * 64})
    check("④⁶ infer: tokens 是**普通 list**（别的读取器给的原样数组 ✓）⇒ 同样能用 ✓",
          g2l.infer_llm_config(meta_list).vocab_size == 64)
    check("④⁷ ⭐ 词表**完全无从得知** ⇒ 具名拒绝（`GgufDequantError` ✓，不能是 NameError ✗✗）",
          _rejects_with(lambda: g2l.infer_llm_config(_common), "GgufDequantError"))
    meta_klen = dict(meta_tokens, **{"qwen3.attention.key_length": 128})
    check("④⁸ infer: ``attention.key_length`` **优先** ✓（GQA 时 hidden/heads≠head_dim，别倒过来 ✗）",
          g2l.infer_llm_config(meta_klen).head_dim == 128,
          g2l.infer_llm_config(meta_klen).head_dim)


def main() -> int:
    case_map()
    case_roundtrip()
    case_transpose()
    case_stream_load()
    case_qk_norm()
    case_qkv_bias()
    case_tie()
    case_infer()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
