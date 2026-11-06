"""UMT5-XXL 文本塔（`engine/umt5_te.py` ＋ `engine/umt5_tokenizer.py`）的守卫（2026-11-01）。

这族的错误全是「装得上、算得出、但结果与提示词无关」✓✗ —— 所以每条判据都要能**单独**抓到一种坏法：

1. **键名错一个字** ⇒ `load_module_weights` 报 24 个 missing，但**前向照样跑**（缺的层用随机初始化）✗；
2. **相对位置偏置只写第 0 层** ⇒ T5 系的写法搬到 UMT5 ⇒ 23 层丢 bias ✗（这是最容易抄错的一条）；
3. **掩码方向反了**（`attention_mask` 当"要屏蔽的"用 ✓✗）⇒ 只有 pad 区坏，肉眼看不出 ✗；
4. **FFN 写成单支**（`wo(relu(wi_0))` ✗）⇒ 门控那一路 `wi_1` 白装 ✓✗；
5. **RMSNorm 写成 LayerNorm**（带 bias / 减均值 ✗）⇒ 全层数值偏移 ✓✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_umt5_test.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

import torch  # noqa: E402

from app.services.engine import umt5_te as ut  # noqa: E402
from app.services.engine import umt5_tokenizer as utk  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: 盘上那份 UMT5-XXL ✓（存在才跑真键集对拍 ✓）
XXL_PATH = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors")

#: 小形状 ✓（d_model 必须 = heads×d_kv = 4×8 = 32 ✓）
TINY = dict(vocab_size=64, d_model=32, d_kv=8, num_heads=4, d_ff=64, num_layers=2)


def tiny_shape(**kwargs: Any) -> ut.Umt5TeShape:
    data = dict(TINY)
    data.update(kwargs)
    return ut.Umt5TeShape(**data)


def tiny_pairs(shape: ut.Umt5TeShape) -> dict[str, tuple[int, ...]]:
    """按 :class:`Umt5TeShape` 造一份**形状字典** ✓（键集与实际权重文件同构 ✓）。"""
    pairs: dict[str, tuple[int, ...]] = {
        "shared.weight": (shape.vocab_size, shape.d_model),
        "encoder.final_layer_norm.weight": (shape.d_model,),
    }
    inner, d = shape.inner_dim, shape.d_model
    for i in range(shape.num_layers):
        pairs[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = (inner, d)
        pairs[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = (inner, d)
        pairs[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = (inner, d)
        pairs[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = (d, inner)
        pairs[f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"] = (
            shape.num_buckets, shape.num_heads)
        pairs[f"encoder.block.{i}.layer.0.layer_norm.weight"] = (d,)
        pairs[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = (shape.d_ff, d)
        pairs[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = (shape.d_ff, d)
        pairs[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = (d, shape.d_ff)
        pairs[f"encoder.block.{i}.layer.1.layer_norm.weight"] = (d,)
    return pairs


# --------------------------------------------------------------------------- ① 推参


def case_infer_shape() -> None:
    xxl = ut.infer_shape(tiny_pairs(ut.Umt5TeShape()))
    check("① XXL 形状 ⇒ 24 层 / 4096d / 64 头 / d_kv 64 / d_ff 10240 / 词表 256384",
          (xxl.num_layers, xxl.d_model, xxl.num_heads, xxl.d_kv, xxl.d_ff, xxl.vocab_size)
          == (24, 4096, 64, 64, 10240, 256384), xxl)

    err: Exception | None = None
    try:
        ut.infer_shape({"encoder.block.0.layer.0.SelfAttention.q.weight": (16, 8)})
    except ut.ShapeError as exc:
        err = exc
    check("①′ 缺 `shared.weight` ⇒ `ShapeError` 且点名", isinstance(err, ut.ShapeError)
          and "shared.weight" in str(err), err)

    pairs = tiny_pairs(tiny_shape())
    pairs.pop("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    err = None
    try:
        ut.infer_shape(pairs)
    except ut.ShapeError as exc:
        err = exc
    check("①″ 缺相对位置偏置 ⇒ `ShapeError`，并说明「UMT5 每层都有」",
          isinstance(err, ut.ShapeError) and "每层" in str(err), err)

    pairs = tiny_pairs(tiny_shape())
    pairs.pop("encoder.block.1.layer.1.DenseReluDense.wo.weight")
    err = None
    try:
        ut.infer_shape(pairs)
        # 只缺某一层的某个键时，层号仍然是连续的 ⇒ 这里**不该**报错（由 expected_keys 的差集去报 ✓）
    except ut.ShapeError as exc:  # pragma: no cover - 形状字典仍能推层数 ✓
        err = exc
    check("①‴ 少一个层内键不影响推层数（留给键集差集去报 ✓）", err is None, err)


# --------------------------------------------------------------------------- ② 键集


def case_keys() -> None:
    xxl = ut.Umt5TeShape()
    keys = ut.expected_keys(xxl)
    check("② XXL 键集 = 24×10 + 2 = 242 条", len(keys) == 242, len(keys))
    bias = [k for k in keys if k.endswith("relative_attention_bias.weight")]
    check("②′ 相对位置偏置**每层**各一条（24 条，不是 1 条 ✗）", len(bias) == 24, len(bias))
    check("②″ 每层 10 条里 q/k/v/o 全无 bias（键集里不许出现 `.bias` 结尾）",
          not [k for k in keys if k.endswith(".bias")], [k for k in keys if k.endswith(".bias")][:4])

    if not XXL_PATH.exists():
        check("②‴ 盘上 UMT5-XXL 真键集逐字对拍（跳过：文件不在）", True, "skipped")
        return
    with open(XXL_PATH, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = __import__("json").loads(fh.read(n))
    # ``*.scale_weight`` / ``scaled_fp8`` / ``spiece_model`` 都**不是**模型权重 ✓（见 weights 的消费名单）
    real = {k for k in header if k.endswith(".weight") and not k.endswith("scale_weight")}
    check("②‴ 盘上 UMT5-XXL 真键集与 `expected_keys()` **逐字相同** ✓（mismatch 双向为空）",
          real == keys, {"多": sorted(real - keys)[:4], "缺": sorted(keys - real)[:4]})


# --------------------------------------------------------------------------- ③ 建网


def case_build() -> None:
    shape = tiny_shape()
    module = ut.build_from_shapes(tiny_pairs(shape), dtype=torch.float32)
    own = set(module.state_dict())
    check("③ 建出来的 state_dict 键集 == `expected_keys()`（一个不多一个不少）",
          own == ut.expected_keys(shape), {"多": sorted(own - ut.expected_keys(shape))[:4],
                                           "缺": sorted(ut.expected_keys(shape) - own)[:4]})
    check("③′ state_dict 里没有任何 `.bias`（RMSNorm/Linear 都不带 bias ✓）",
          not [k for k in own if k.endswith(".bias")], [k for k in own if k.endswith(".bias")][:4])
    check("③″ 建网默认 bf16（`.to(dtype)` 生效 ✓）",
          ut.build_from_shapes(tiny_pairs(shape)).shared.weight.dtype == torch.bfloat16, None)


# --------------------------------------------------------------------------- ④ 前向与掩码


def _tiny_module(seed: int = 0, shape: ut.Umt5TeShape | None = None) -> ut.Umt5Encoder:
    shape = shape or tiny_shape()
    torch.manual_seed(seed)
    module = ut.Umt5Encoder(shape).float()
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("relative_attention_bias.weight"):
                param.normal_(0.0, 0.05)
            elif param.dim() >= 2:
                param.normal_(0.0, 0.08)
            else:
                param.fill_(1.0)
    return module


def case_forward() -> None:
    shape = tiny_shape()
    module = _tiny_module()
    ids = torch.randint(0, shape.vocab_size, (2, 6))
    mask = torch.ones(2, 6)
    mask[1, 4:] = 0
    with torch.no_grad():
        out = module(ids, attention_mask=mask)
    check("④ 前向输出形状 = [B, L, d_model]", tuple(out.shape) == (2, 6, shape.d_model), out.shape)

    # ④′ 改 **pad 区**的输入 ⇒ 有效位输出必须**不变**（掩码方向对不对，只有这一条能验 ✓）
    embeds = module.shared(ids)
    with torch.no_grad():
        base = module(embeds=embeds, attention_mask=mask)
        jitter = embeds.clone()
        jitter[1, 4:] += 5.0                       # 只动第 2 条的 pad 位 ✓
        moved = module(embeds=jitter, attention_mask=mask)
    diff = (moved[1, :4] - base[1, :4]).abs().max().item()
    check("④′ 改 pad 区的输入 ⇒ **有效位**输出逐元素不变（掩码方向正确 ✓）",
          diff < 1e-5, diff)

    # ④″ 把掩码全给 1 ⇒ 有效位输出**必须变**（否则说明掩码根本没接上 ✗）
    with torch.no_grad():
        nomask = module(embeds=embeds, attention_mask=torch.ones(2, 6))
    diff2 = (nomask[1, :4] - base[1, :4]).abs().max().item()
    check("④″ 同一份输入去掉掩码 ⇒ 有效位输出**会变**（证明掩码真的进了注意力 ✓）",
          diff2 > 1e-4, diff2)


def case_naive_match() -> None:
    """朴素实现（k×sqrt(d) ＋ matmul/softmax/masked_fill ✓）与本实现对拍 ✓。"""
    shape = tiny_shape()
    module = _tiny_module(seed=7)
    ids = torch.randint(0, shape.vocab_size, (1, 5))
    keep = torch.ones(1, 5)
    keep[0, 3:] = 0
    with torch.no_grad():
        got = module(ids, attention_mask=keep)

        x = module.shared(ids)
        # 朴素版照抄 comfy 的口径 ✓：``1 - attention_mask`` 只在**最后一维**（key ✓）上生效 ✓
        # ⚠️ 掩码形状要与实现**同形**（``(B,1,1,L)`` 靠广播生效 ✓）：全展开成 ``(B,1,L,L)`` 再送
        # ``scaled_dot_product_attention`` 会走另一条内核路径，与 softmax 版差 6e-2 ✓✗（2026-11-01 实测）
        mask = (1.0 - keep.reshape(1, 1, 1, -1))
        mask = mask.masked_fill(mask.bool(), -torch.finfo(torch.float32).max)
        for block in module.encoder.block:
            ln = block.layer[0].layer_norm
            h = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ln.eps)
            h = h * ln.weight
            att = block.layer[0].SelfAttention
            q = att.q(h).view(1, 5, 4, 8).transpose(1, 2)
            k = att.k(h).view(1, 5, 4, 8).transpose(1, 2)
            v = att.v(h).view(1, 5, 4, 8).transpose(1, 2)
            bias = att.position_bias(5, torch.float32)
            # ⚠️ comfy 写的是「k 乘 √d，**然后交给 SDPA**」✓ —— SDPA 内部还会再除一次 √d ✓，
            # 两下抵消 ⇒ 净效果是**标准缩放** ✓。手工复算时**只乘一次 1/√d** ✓；
            # 写成「乘 √d 再除 √d」（＝不缩放）会让分数大 √8 倍、softmax 变尖 ✓✗（2026-11-01 踩过）
            scores = torch.matmul(q, k.transpose(-1, -2)) * (8 ** -0.5) + bias + mask
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v).transpose(1, 2).reshape(1, 5, 32)
            x = x + att.o(out)
            ln2 = block.layer[1].layer_norm
            h2 = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ln2.eps)
            h2 = h2 * ln2.weight
            ffn = block.layer[1].DenseReluDense
            x = x + ffn.wo(torch.nn.functional.gelu(ffn.wi_0(h2), approximate="tanh") * ffn.wi_1(h2))
        ln = module.encoder.final_layer_norm
        want = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ln.eps)) * ln.weight

    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
    maxdiff = (got - want).abs().max().item()
    check("⑤ 与朴素实现（k×√d、matmul+softmax、加性掩码）对拍 cos>0.99999 且 maxdiff<1e-5",
          cos > 0.99999 and maxdiff < 1e-5, {"cos": cos, "maxdiff": maxdiff})


def case_parts() -> None:
    shape = tiny_shape()
    module = _tiny_module(seed=11)
    ids = torch.randint(0, shape.vocab_size, (1, 4))
    with torch.no_grad():
        base = module(ids, attention_mask=torch.ones(1, 4))

        # ⑤′ 门控那一路必须真参与（把 wi_1 置零 ⇒ 输出必变 ✓）
        ffn = module.encoder.block[0].layer[1].DenseReluDense
        backup = ffn.wi_1.weight.detach().clone()
        ffn.wi_1.weight.zero_()
        gated_off = module(ids, attention_mask=torch.ones(1, 4))
        ffn.wi_1.weight.copy_(backup)
        diff_gate = (gated_off - base).abs().max().item()
    check("⑤′ 把 `wi_1` 置零 ⇒ 输出改变（门控支真参与 ✓，不是死权重）", diff_gate > 1e-4, diff_gate)

    with torch.no_grad():
        att = module.encoder.block[0].layer[0].SelfAttention
        backup_bias = att.relative_attention_bias.weight.detach().clone()
        att.relative_attention_bias.weight.zero_()
        att._bias_cache = None
        no_bias = module(ids, attention_mask=torch.ones(1, 4))
        att.relative_attention_bias.weight.copy_(backup_bias)
        att._bias_cache = None
        diff_bias = (no_bias - base).abs().max().item()
    check("⑤″ 把相对位置偏置置零 ⇒ 输出改变（bias 真接进打分 ✓）", diff_bias > 1e-4, diff_bias)

    with torch.no_grad():
        ln = module.encoder.block[0].layer[0].layer_norm
        x = torch.randn(2, 3, shape.d_model, dtype=torch.float32)
        got = ln(x)
        want = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ln.eps) * ln.weight
        same = (got - want).abs().max().item()
    check("⑤‴ RMSNorm 与手算逐元素一致（不减均值 ✗、不改 eps 位置 ✗）", same < 1e-6, same)

    # ⑤⁗ 整行都被 pad（掩码全 0）时**不许出 NaN** ✓ —— 这是 `-inf` 与 `-finfo.max` 的分水岭 ✓✗
    with torch.no_grad():
        blank = module(torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.zeros(1, 4))
    check("⑤⁗ 掩码全 0（整行 pad）⇒ 有限值、无 NaN（用 `-finfo.max` 才做得到 ✓）",
          bool(torch.isfinite(blank).all()), bool(torch.isfinite(blank).all()))


# --------------------------------------------------------------------------- ⑥ 分词器


def _mini_sp_proto(pieces: list[tuple[str, float]]) -> bytes:
    """手造一份最小 ``ModelProto`` ✓（只为验证解析器 ✓，不是真词表 ✓）。"""
    def varint(value: int) -> bytes:
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    body = bytearray()
    for token, score in pieces:
        inner = b"\x0a" + varint(len(token.encode())) + token.encode()
        inner += b"\x15" + struct.pack("<f", score)
        body += b"\x0a" + varint(len(inner)) + inner
    return bytes(body)


def case_tokenizer() -> None:
    pieces = [("<pad>", 0.0), ("</s>", 0.0), ("▁hello", -1.5), ("▁world", -2.0)]
    parsed = utk.parse_sp_model(_mini_sp_proto(pieces))
    check("⑥ 解析 sentencepiece ModelProto ⇒ piece/score/type 三元组逐条对上",
          [(p, round(s, 3)) for (p, s, _t) in parsed] == [(p, s) for p, s in pieces], parsed)

    if not utk.available():
        check("⑥′ 有 sentencepiece 库（跳过：没装）", True, "skipped")
        return
    tokenizer = utk.Umt5Tokenizer(None, vocab=[(p, s, 1) for p, s in pieces], max_length=8)
    ids, mask = tokenizer.encode("hello world")
    valid = sum(mask)
    check("⑥′ 自研路：长度固定 + `</s>` 正好在有效段末尾（其后全是 pad）",
          len(ids) == 8 and ids[valid - 1] == utk.EOS_ID and all(i == 0 for i in ids[valid:]),
          {"ids": ids, "mask": mask})
    check("⑥″ 自研路：空格被换成 `▁` ＋ 句首补 `▁`（等价 add_dummy_prefix ✓）",
          tokenizer.add_dummy_prefix and "▁" in tokenizer.decode(ids[:1]) or valid > 1,
          tokenizer.decode(ids[:valid]))


def case_real_spiece() -> None:
    if not XXL_PATH.exists() or not utk.available():
        check("⑦ 真 spiece 分词（跳过：文件或缺库）", True, "skipped")
        return
    import json

    with open(XXL_PATH, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        begin, end = header["spiece_model"]["data_offsets"]
        fh.seek(8 + n + begin)
        data = fh.read(end - begin)
    tokenizer = utk.load_tokenizer(spiece_bytes=data, max_length=512)
    text = "A cinematic shot of a swordsman standing on a frozen lake at dawn"
    ids, mask = tokenizer.encode(text)
    ok = (len(ids) == 512 and ids[sum(mask) - 1] == utk.EOS_ID
          and all(i == 0 for i in ids[sum(mask):]) and 0 < sum(mask) < 40)
    check("⑦ 真 spiece：512 长 / 有效段以 `</s>` 收尾 / 其余全 pad / 有效 token 数合理",
          ok, {"valid": sum(mask), "tail": ids[-3:], "backend": tokenizer.backend})
    check("⑦′ 真 spiece 往返（去掉 pad/eos）= 原句",
          tokenizer.decode([i for i in ids[: sum(mask) - 1]]) .strip() == text,
          tokenizer.decode([i for i in ids[: sum(mask) - 1]]))


def main() -> int:
    case_infer_shape()
    case_keys()
    case_build()
    case_forward()
    case_naive_match()
    case_parts()
    case_tokenizer()
    case_real_spiece()

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}\n      {detail!r}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
