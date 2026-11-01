"""自检：**Qwen-Image 的文本塔**（Qwen2.5-VL-7B 语言部分）形状／键／取层／模板口径（2026-10-31）。

这一族与 Z-Image 的文本塔（`qwen3_te` ✓）**长得几乎一样**（同一份 Qwen 词表 ✓、同一个 SwiGLU ✓），
但四格口径不同 ✓ —— 四格都「装得上、算得出、不报错」✗：

==============================  ============================  ==========================
                                Z-Image（`qwen3_te` ✓）        Qwen-Image（本族 ✓）
==============================  ============================  ==========================
取哪一层                        **倒数第 2 层**（`-2` ✓）      **最后一层**（`-1` ✓）
q/k/v 偏置                      **没有** ✗                   **有**（三根 ✓）
QK-norm                        **有**（`q_norm`/`k_norm`）    **没有** ✗
提示词模板                      `user` 一段 ✓                 `system` + `user` 两段 ✓
==============================  ============================  ==========================

本文件钉的就是这四格 + 「system 段整段切掉」那条（参考实现的 `template_end` 口径 ✓ ——
进主干的**不是**整条模板 ✗，多送一段不会报错，只会让同一句提示词在两边结果不一样 ✓✗）。

⚠️ 不跑真装 ✗（这份 `qwen_2.5_vl_7b_fp8_scaled` 反量化后 ≈14 GiB ✓）：
只读**头部**对账键与结构 ✓，真出图是人为验收那一步 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_qwen25_te_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import torch  # noqa: E402

from app.services.engine import qwen25_te as te  # noqa: E402
from app.services.engine import safetensors  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _error_of(fn) -> str:  # noqa: ANN001
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return ""


# ------------------------------------------------------------------ 小表

#: 词表得**大于** ``IM_START_ID`` ✓ —— 前向用例要用真实的模板 token id ✓（不然 embed 越界 ✓✗）
_TINY = dict(vocab=152100, hidden=32, depth=3, head_dim=8, ffn=48)


def _table(*, qkv_bias: bool = True, qk_norm: bool = False, o_bias: bool = False,
           layers: int | None = None, drop_layer: int | None = None,
           **over) -> dict[str, tuple[int, ...]]:
    spec = dict(_TINY)
    spec.update(over)
    vocab, hidden, depth = spec["vocab"], spec["hidden"], spec["depth"]
    hd, ffn = spec["head_dim"], spec["ffn"]
    heads, kv_heads = hidden // hd, max(1, hidden // hd // 2)
    n = depth if layers is None else layers
    t: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (vocab, hidden),
                                     "model.norm.weight": (hidden,),
                                     "lm_head.weight": (vocab, hidden)}
    for i in range(n):
        if drop_layer is not None and i == drop_layer:
            continue
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = (hidden,)
        t[p + "post_attention_layernorm.weight"] = (hidden,)
        t[p + "self_attn.q_proj.weight"] = (heads * hd, hidden)
        t[p + "self_attn.k_proj.weight"] = (kv_heads * hd, hidden)
        t[p + "self_attn.v_proj.weight"] = (kv_heads * hd, hidden)
        t[p + "self_attn.o_proj.weight"] = (hidden, heads * hd)
        if qkv_bias:
            t[p + "self_attn.q_proj.bias"] = (heads * hd,)
            t[p + "self_attn.k_proj.bias"] = (kv_heads * hd,)
            t[p + "self_attn.v_proj.bias"] = (kv_heads * hd,)
        if o_bias:
            t[p + "self_attn.o_proj.bias"] = (hidden,)
        if qk_norm:
            t[p + "self_attn.q_norm.weight"] = (hd,)
            t[p + "self_attn.k_norm.weight"] = (hd,)
        t[p + "mlp.gate_proj.weight"] = (ffn, hidden)
        t[p + "mlp.up_proj.weight"] = (ffn, hidden)
        t[p + "mlp.down_proj.weight"] = (hidden, ffn)
    #: fp8 的伴随键 ✓（必须被**认领** ✓，否则会被当成多余键误报 ✗）
    t["model.layers.0.self_attn.q_proj.scale_weight"] = ()
    t["model.layers.0.self_attn.q_proj.scale_input"] = ()
    return t


def _shape(**over):
    kw = dict(head_dim=_TINY["head_dim"])
    kw.update(over.pop("kw", {}))
    return te.infer_shape(_table(**over), **kw)


# --------------------------------------------------------------- ① 形状推断


def case_shape() -> None:
    sh = _shape()
    check("① vocab / hidden 读 `model.embed_tokens.weight` ✓、层数按 `model.layers.N` 数 ✓、"
          "ffn 读 `mlp.gate_proj` 的**输出**维 ✓",
          (sh.vocab, sh.hidden, sh.depth, sh.ffn) == (152100, 32, 3, 48),
          (sh.vocab, sh.hidden, sh.depth, sh.ffn))
    check("①′ heads = q_out / head_dim ✓、kv_heads = k_out / head_dim ✓（GQA ✓ 4/2 ✓）",
          (sh.heads, sh.kv_heads, sh.head_dim) == (4, 2, 8), (sh.heads, sh.kv_heads))
    check("①″ **qkv_bias 是族指纹** ✓：三根 `*.bias` 都在 ⇒ True ✓，且 `qk_norm` 必须是 False ✓",
          sh.qkv_bias and not sh.qk_norm, (sh.qkv_bias, sh.qk_norm))

    message = _error_of(lambda: te.infer_shape(_table(qkv_bias=False), head_dim=8))
    check("①‴ 缺 q/k/v 的三根 bias ⇒ 当场报错并点名 Qwen2.5 ✓（那是 Qwen3/Z-Image 的形状 ✓✗）",
          "Qwen2.5" in message and "bias" in message, message or "没报错 ✗")

    message = _error_of(lambda: te.infer_shape(_table(qk_norm=True), head_dim=8))
    check("①⁗ 带 `q_norm`/`k_norm` ⇒ 当场报错并点名 Qwen3（Z-Image 文本塔 ✓）",
          "Qwen3" in message, message or "没报错 ✗")

    message = _error_of(lambda: te.infer_shape(_table(o_bias=True), head_dim=8))
    check("①⁵ `o_proj` 带 bias ⇒ 报错 ✓（Qwen2.5 的 o_proj 无偏置 ✓ —— 有偏置的是别的族 ✗）",
          "o_proj" in message, message or "没报错 ✗")

    message = _error_of(lambda: te.infer_shape(_table(), head_dim=6))
    check("①⁶ head_dim 不整除 q/kv 输出维 ⇒ 报错 ✓（`head_dim` 只能显式给 ✓ —— "
          "q_proj 是方阵、又没有 q_norm ⇒ 形状上读不出来 ✗）",
          "整除" in message, message or "没报错 ✗")

    #: ⚠️ 形状上**真的分不出来**：q_proj 是方阵（3584×3584）⇒「28 头×128」与「14 头×256」
    #: 在张量形状上完全一样 ✓✗。所以 head_dim 只能按发布配置给 ✓ —— 这条钉的是「默认值真的是
    #: 128 而不是随便猜一个能整除的数」✓：小表 hidden=32 配 128 必然报错 ✓（若默认值被悄悄
    #: 改成「能整除就行」⇒ 这条会 FAIL ✓）。
    message = _error_of(lambda: te.infer_shape(_table()))
    check("①⁷ 不给 head_dim ⇒ 用发布配置 `QWEN25_HEAD_DIM=128` ✓（而不是挑一个能整除的数 ✓）"
          "：hidden=32 的小表因此**必须**报错 ✓",
          te.QWEN25_HEAD_DIM == 128 and "整除" in message, message or "没报错 ✗")

    message = _error_of(lambda: te.infer_shape(_table(drop_layer=1), head_dim=8))
    check("①⁸ 层号**不连续**（缺第 1 层、剩 0 与 2）⇒ 报错 ✓（`ModuleList` 按序号排 ✓，"
          "静默跳过 ⇒ 后面所有层都错位 ✓✗）",
          "不连续" in message, message or "没报错 ✗")

    broken = _table()
    broken.pop("model.layers.1.mlp.up_proj.weight")
    message = _error_of(lambda: te.infer_shape(broken, head_dim=8))
    check("①⁹ 某层缺一根键 ⇒ 指明第几层缺什么 ✓（不许静默跳过那一层 ✗）",
          "第 1 层" in message, message or "没报错 ✗")


def case_keys() -> None:
    sh = _shape()
    module = te.build_from_shapes(_table(), head_dim=8)
    built, wanted = set(module.state_dict()), te.expected_keys(sh)
    check("② 建出来的键 = 该族应有的键 ✓（**不含** `lm_head` ✓ —— 参考实现也是 `lm_head=False` ✓）",
          built == wanted, {"缺": sorted(wanted - built)[:6], "多": sorted(built - wanted)[:6]})
    check("②′ 三根 `q_proj/k_proj/v_proj.bias` 真的**建了** ✓（漏了就没处放权重 ✓✗）",
          all(f"model.layers.0.self_attn.{n}_proj.bias" in built for n in "qkv"),
          sorted(k for k in built if k.endswith(".bias"))[:4])
    check("②″ `model.norm.weight` 建了但**不参与**前向输出 ✓（见 ⑤″ ✓）",
          "model.norm.weight" in built, "—")

    extras = [k for k in _table() if k not in wanted]
    check("②‴ 盘上多出来的键都能被**认领** ✓：`visual.*`（视觉塔 ✗ 另一步 ✓）、`lm_head.*`、"
          "fp8 的 `.scale_*` ✓、0 元素哨兵 `scaled_fp8` ✓ —— 认领规则单一真源 "
          "`te.unclaimed_extras()` ✓",
          extras and not te.unclaimed_extras(_table(), sh), extras[:6])


# --------------------------------------------------------------- ③ 模板与切段


def case_template() -> None:
    check("③ 无图模板与参考实现**逐字相同** ✓（`system` 段点名 颜色/形状/大小/质地/数量/文字/空间关系 ✓）",
          te.TE_TEMPLATE == (
              "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
              "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
              "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"), te.TE_TEMPLATE[:40])
    check("③′ 有图模板逐字相同 ✓（`<|vision_start|><|image_pad|><|vision_end|>` 在正文**之前** ✓）",
          te.TE_TEMPLATE_EDIT == (
              "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
              "texture, objects, background), then explain how the user's text instruction should alter "
              "or modify the image. Generate a new image that meets the user's requirements while "
              "maintaining consistency with the original input where appropriate.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
              "<|im_start|>assistant\n"), te.TE_TEMPLATE_EDIT[:40])

    text = "一只黑猫坐在窗台上，阳光斜射"
    out = te.prompt_of(text)
    check("③″ 正文**原样**落进模板 ✓（不裁 ✓ 不改 ✓ 不加标点 ✗）",
          text in out and out.count(text) == 1 and out.startswith("<|im_start|>system"), out[:30])
    check("③‴ 空串补一个**空格** ✓（否则 0 个 token ⇒ 主干拿到空文本却照样出图 ✓✗）",
          te.prompt_of("").endswith("user\n <|im_end|>\n<|im_start|>assistant\n"), repr(te.prompt_of("")[-40:]))
    check("③⁴ 自带 `<|im_start|>` 的串**不再套模板** ✓（参考实现同款旁支 ✓）",
          te.prompt_of("<|im_start|>raw") == "<|im_start|>raw", te.prompt_of("<|im_start|>raw"))

    message = _error_of(lambda: te.prompt_of("x", with_image=True))
    check("③⁵ 只用 Edit 模板、不给图像嵌入 ⇒ **当场报错** ✓（这三格必须是视觉塔输出 ✓，"
          "拿词表嵌入顶替 ⇒ 装得上、算得出、Edit 能力没有 ✓✗）",
          "image_pad" in message and "视觉塔" in message, message or "没报错 ✗")
    message = _error_of(lambda: te.prompt_of("x", image_embeds=True))
    check("③⁶ 反过来给了图像嵌入却用无图模板 ⇒ 也报错 ✓（模板里没有占位符 ✗）",
          "占位符" in message, message or "没报错 ✗")


def case_trim() -> None:
    im = te.IM_START_ID
    ids = [im, 911, 15, 151648, 198, im, te.USER_ID, 198, 7, 8, 9, 151648, 198, im, 520, 198]
    start = te.trim_start(ids)
    check("④ 切段起点 = **第二个** `<|im_start|>` 之后 ✓ 再跳过 `user`+换行 ✓ ⇒ 文案正文的第一个 token ✓",
          start == 8, start)
    check("④′ 切完剩下的是「正文 + 收尾」✓（`<|im_end|>` / `<|im_start|>` / `assistant` 都还在 ✓）",
          ids[start:] == [7, 8, 9, 151648, 198, im, 520, 198], ids[start:])

    two = [im, 911, 151648, 198, im, 520, 198]      # 第二个 im_start 后面不是 user
    check("④″ 第二个 im_start 后面不是 `user`+换行 ⇒ 起点就停在它自己 ✓（不硬凑 +3 ✓）",
          te.trim_start(two) == 4, te.trim_start(two))

    message = _error_of(lambda: te.trim_start([im, 911, 15]))
    check("④‴ 少于两个 `<|im_start|>` ⇒ 报错 ✓（说明给的**不是**套过模板的串 ✗；"
          "参考实现在这种输入上只留最后 1 个 token ✓✗ —— 那是坏掉、不是口径 ✓）",
          "2 个" in message, message or "没报错 ✗")


# --------------------------------------------------------------- ⑤ 前向口径


def case_forward() -> None:
    torch.manual_seed(5)
    module = te.build_from_shapes(_table(), head_dim=8).eval()
    ids = torch.tensor([[te.IM_START_ID, 11, 12, 151648, 198,
                         te.IM_START_ID, te.USER_ID, 198, 21, 22, 23, 151648, 198,
                         te.IM_START_ID, 520, 198]])
    with torch.no_grad():
        full = module(ids, trim=False)
        cut = module(ids, trim=True)
    start = te.trim_start(ids[0].tolist())
    check("⑤ 输出宽度 = hidden ✓（送主干的 `context` 就是它 ✓）",
          tuple(full.shape) == (1, ids.shape[1], 32), tuple(full.shape))
    check("⑤′ 切段 = **先编码整条、再切行** ✓（与 `full[:, start:]` 逐位相同 ✓ —— "
          "先切后编码会让位置整体前移 ✓✗，不报错但语义偏移 ✓）",
          torch.equal(cut, full[:, start:]), tuple(cut.shape))

    expected = module.model.norm(full)
    check("⑤″ **不过** `model.norm` ✓（`layer_norm_hidden_state=False` ✓）："
          "`model.norm` 建了但输出里不带它 ✓（若过了 ⇒ 送进主干的向量被整体缩放 ✓✗，"
          "症状是整图偏灰/偏曝 ✓）",
          not torch.allclose(full, expected, atol=1e-4), float((full - expected).abs().max()))

    wrong = te.build_from_shapes(_table(), head_dim=8, layer_idx=-2).eval()
    wrong.load_state_dict(module.state_dict())
    with torch.no_grad():
        other = wrong(ids, trim=True)
    check("⑤‴ 取 `-2` 与取 `-1` **确实不同** ✓（这就是 Z-Image 与 Qwen-Image 的取层口径差 ✓；"
          "抄错不报错 ✗，差的是整个语义空间 ✓✗）",
          not torch.allclose(cut, other, atol=1e-5), float((cut - other).abs().max()))

    trimmed = torch.zeros_like(module.model.layers[0].self_attn.q_proj.bias)
    module.model.layers[0].self_attn.q_proj.bias.data.copy_(trimmed)
    with torch.no_grad():
        no_bias = module(ids, trim=True)
    check("⑤⁴ 三根 q/k/v bias **真的参与了计算** ✓（清零第 0 层的 q bias ⇒ 输出必须变 ✓）",
          not torch.allclose(cut, no_bias, atol=1e-6), float((cut - no_bias).abs().max()))

    message = _error_of(lambda: module(torch.zeros((2, 4), dtype=torch.long)))
    check("⑤⁵ batch>1 ⇒ 报错 ✓（因果掩码是纯 `triu`，**没有**按条长度 pad 那一路 ✗ ⇒ "
          "两条不同长度会互相看见 ✓✗ —— 静默错，所以必须拦住 ✓）",
          "batch" in message, message or "没报错 ✗")


def case_runtime() -> None:
    sh = _shape()
    rt = te.Qwen25TeRuntime(sh, max_length=512)
    info = rt.to_dict()
    check("⑥ 回执口径：`outputDim` = hidden = 3584 类口径 ✓、`targetLayer` 折算成 0-based ✓、"
          "`qkvBias` 诚实报 True ✓",
          info["outputDim"] == 32 and info["targetLayer"] == 2 and info["qkvBias"] is True
          and info["layerIdx"] == -1, info)
    check("⑥′ 模板随 runtime 走 ✓（默认无图模板 ✓ —— Edit 那套要显式换 ✓）",
          rt.prompt("x").startswith("<|im_start|>system") and rt.template == te.TE_TEMPLATE, "—")


# --------------------------------------------------------------- ⑦ 真实权重头部


_REAL = "qwen_2.5_vl_7b_fp8_scaled.safetensors"


def case_real_header() -> None:
    root = Path(os.environ.get("COMFY_MODELS", r"D:/Comfy-Desktop/ComfyUI-Shared/models"))
    path = root / "text_encoders" / _REAL
    if not path.exists():
        skip(f"⑦ 盘上没有 {path} ⇒ 跳过真键对账")
        return
    try:
        meta = safetensors.inspect(str(path))
    except Exception as err:  # noqa: BLE001
        check("⑦ 头部读不出来 ✗", False, f"{type(err).__name__}: {err}")
        return
    shapes = {name: tuple(entry.shape) for name, entry in meta.tensors.items()}
    try:
        sh = te.infer_shape(shapes)
    except te.ShapeError as err:
        check("⑦ `infer_shape` 推不出这一族 ✗", False, str(err))
        return
    check(f"⑦ 真权重推出来的结构 ✓：{sh.depth} 层 / {sh.hidden} 宽 / {sh.heads}×{sh.head_dim} 头 / "
          f"kv={sh.kv_heads} / ffn={sh.ffn} / vocab={sh.vocab} ✓"
          "（与参考实现 `Qwen25_7BVLI_Config` 的 28/3584/28×128/4/18944/152064 逐格相同 ✓）",
          (sh.depth, sh.hidden, sh.heads, sh.head_dim, sh.kv_heads, sh.ffn, sh.vocab)
          == (28, 3584, 28, 128, 4, 18944, 152064),
          (sh.depth, sh.hidden, sh.heads, sh.head_dim, sh.kv_heads, sh.ffn, sh.vocab))
    check("⑦′ 取层口径 = **最后一层** ✓、且 qkv_bias 为真 ✓",
          sh.layer_idx == -1 and sh.target_layer() == sh.depth - 1 and sh.qkv_bias,
          (sh.layer_idx, sh.target_layer()))

    wanted = te.expected_keys(sh)
    missing = wanted - set(shapes)
    check(f"⑦″ 模型**应有的每一根键**权重里都有 ✓（缺 {len(missing)} 根 ✓）⇒ 这份文本塔装得下 ✓",
          not missing, sorted(missing)[:8])
    extras = set(shapes) - wanted
    unclaimed = te.unclaimed_extras(shapes, sh)
    check(f"⑦‴ 多出来的 {len(extras)} 根都被认领 ✓（`visual.*` 视觉塔 ✓、`lm_head.weight` ✓、"
          "fp8 的 `.scale_input`/`.scale_weight` ✓、哨兵 `scaled_fp8` ✓）⇒ "
          "没有「弄不清是什么」的键 ✓",
          not unclaimed, unclaimed[:8])
    check("⑦⁴ 视觉塔确实在盘上 ✓（714 张量 ⇒ Edit 2511 需要它 ✓，是下一步而不是「没有」✓）",
          sum(1 for k in shapes if k.startswith("visual.")) > 0,
          sum(1 for k in shapes if k.startswith("visual.")))


def main() -> int:
    case_shape()
    case_keys()
    case_template()
    case_trim()
    case_forward()
    case_runtime()
    case_real_header()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
