"""自检：**同一份前向**在 CPU 与 GPU 上跑，且结果一致（``engine/llm.py`` 设备/精度口径 ✓ 2026-10-29 起 ✓）。

验的是什么 ✗：本仓的推理内核**不写两套代码** ✓（:func:`app.services.engine.llm.build_llm` 的
``device``/``dtype`` ✓）—— 所以真正要证的是三件事 ✓：

1. **择优只在一处** ✓：``auto`` 有卡用卡、没卡用 CPU ✓；**显式点名用不了的设备 ⇒ 具名拒绝** ✗✗
   （静默回落 CPU 会让「8B 慢」被当成模型慢 ✓✗）；
2. **同一份权重、两端输出一致** ✓：fp32 下 logits 只差归约顺序（~1e-4 ✓）、greedy 生成的 token **逐个相同** ✓；
   低精度（bf16）下按容差 ✓ + 采样在 fp32 里做 ✓；
3. **设备不是摆设** ✓：装载到卡上的参数 ``.device.type == "cuda"`` ✓、后端 ``describe()`` 的
   ``device``（实测）与 ``deviceRequested``（请求）**分两栏** ✓✗、缓存键含设备 ✓。

⚠️ 没有 CUDA 卡的机器上，GPU 那几条会**显式 SKIP** ✓（**SKIP ≠ 通过** ✗ —— 分母里不算 ✓，结尾单列 ✓）。
⚠️ 每条断言都配了「能失败」的探针 ✓✗：故意把一个权重加 0.05 ⇒ 同一判据必须报出来 ✓
（判据太松 = 假绿 ✗✗）。

参考（**只参考行为、没有复制代码** ✓）：``llama.cpp``（MIT）同一张 ggml 图跑 CPU/CUDA 后端 ✓；
``huggingface/transformers``（Apache-2.0）低精度下 RMSNorm 内部升 fp32 ✓；PyTorch（BSD-3）的
``Module.to`` / ``set_default_dtype`` 语义 ✓。

运行::

    python -X utf8 tests/engine_llm_device_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="llmdev_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
except ModuleNotFoundError as _missing:          # pragma: no cover —— 可选依赖 ✓
    print(f"SKIP  整套：缺 {_missing.name} ⇒ 不判（≠ 通过 ✓）")
    raise SystemExit(0)

from app.services.engine import gguf_to_llm as g2l  # noqa: E402
from app.services.engine import llm as llm_mod  # noqa: E402
from app.services.engine import llm_backend  # noqa: E402

#: 缩小版配置 ✓：GQA（4 头 / 2 组 ✓）、head_dim 偶数 ✓、**不 tie**（好让 ``lm_head`` 也在对账里 ✓）
CONFIG = llm_mod.LlmConfig(vocab_size=64, hidden=32, depth=2, heads=4, kv_heads=2,
                           head_dim=8, ffn=64, tie_embeddings=False)

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPPED: list[tuple[str, str]] = []

#: 需要转置的那几类（GGUF 存 (in,out) ✓，PyTorch 要 (out,in) ✓）
_TRANSPOSED = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
               ".gate.weight", ".up.weight", ".down.weight", ".lm_head.weight")


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str, why: str) -> None:
    _SKIPPED.append((name, why))


def _message_of(call: Any) -> str:
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 这里就是要看消息 ✓
        return f"{type(err).__name__}: {err}"
    return ""


def _gpu_count() -> int:
    return int(llm_mod.device_facts()["cudaCount"])


# ── ① 设备解析 ✓
def case_devices() -> None:
    facts = llm_mod.device_facts()
    auto = llm_mod.resolve_device("auto")
    expected = f"cuda:{torch.cuda.current_device()}" if facts["cuda"] else "cpu"
    check("① ⭐ ``auto`` = 本机事实：有卡 ⇒ cuda:N ✓、没卡 ⇒ cpu ✓（不是写死 ✓）",
          auto == expected, (auto, expected, facts["devices"]))
    check("①′ 显式 ``cpu`` ⇒ 永远 cpu ✓（哪台机器都一样 ✓）", llm_mod.resolve_device("cpu") == "cpu")
    check("①″ ``available_devices()`` 第一项是 cpu ✓（兜底设备永远在 ✓）",
          llm_mod.available_devices()[0] == "cpu", llm_mod.available_devices())

    # ⚠️ 拒绝分支：用**不存在的卡号**造「用不了」✓（有卡 1 张 ⇒ cuda:1 ✗；没卡 ⇒ cuda:0 ✗ ✓ 两种机器都测得到 ✓）
    absent = f"cuda:{_gpu_count()}"
    message = _message_of(lambda: llm_mod.resolve_device(absent))
    check("①‴ ⭐⭐ 显式要一张**不存在**的卡 ⇒ 具名拒绝 ✓（静默回落 cpu ✗✗）",
          "LlmError" in message and absent in message, message)
    check("①⁴ 拒绝时把**本机可用设备**列出来 ✓（不然用户只能猜 ✓）",
          "cpu" in message, message)

    bogus = _message_of(lambda: llm_mod.resolve_device("tpu"))
    check("①⁵ 乱写设备名 ⇒ 具名拒绝 ✓（``tpu`` 不认 ✓）", "LlmError" in bogus and "tpu" in bogus, bogus)


# ── ② 精度口径 ✓
def case_dtype() -> None:
    check("② CPU·auto ⇒ fp32 ✓（CPU 上 bf16 只剩慢路径 ✓）",
          llm_mod.resolve_dtype(None, device="cpu") is torch.float32)
    check("②′ CPU 上显式 bf16 ⇒ 允许 ✓（省内存的唯一正路 ✓）",
          llm_mod.resolve_dtype("bf16", device="cpu") is torch.bfloat16)
    fp16 = _message_of(lambda: llm_mod.resolve_dtype("fp16", device="cpu"))
    check("②″ CPU + fp16 ⇒ 具名拒绝 ✓（一半算子不支持 ✓）", "LlmError" in fp16 and "float16" in fp16, fp16)
    int8 = _message_of(lambda: llm_mod.resolve_dtype("int8", device="cpu"))
    check("②‴ ``int8`` ⇒ 拒绝且**点明那是权重精度** ✓（权重精度归反量化管 ✓ 不是这个参数 ✗）",
          "LlmError" in int8 and "权重" in int8, int8)

    if _gpu_count() == 0:
        skip("②⁴ GPU 上 auto ⇒ bf16（无卡）", "本机没有 CUDA 卡")
        return
    device = f"cuda:{torch.cuda.current_device()}"
    wanted = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    check("②⁴ ⭐ GPU·auto ⇒ bf16 ✓（8B·fp32 = 32 GB ✗ 装不进 24 GB 卡 ✓）",
          llm_mod.resolve_dtype(None, device=device) is wanted)
    check("②⁵ GPU 上显式 float32 ⇒ fp32 ✓（跨设备**逐位判据**靠它 ✓）",
          llm_mod.resolve_dtype("float32", device=device) is torch.float32)


# ── ③ 同一份权重、fp32、跨设备 ✓
def case_forward_fp32() -> None:
    if _gpu_count() == 0:
        skip("③ fp32 跨设备前向/greedy 一致", "本机没有 CUDA 卡")
        return
    device = f"cuda:{torch.cuda.current_device()}"
    cpu_model = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")
    gpu_model = llm_mod.build_llm(CONFIG, device=device, dtype="float32")
    gpu_model.load_state_dict(cpu_model.state_dict())      # ⚠️ 同一份权重 ✓（不是两次随机初始化 ✗）
    ids = torch.tensor([[1, 5, 7, 9, 11, 13]])

    with torch.no_grad():
        logits_cpu = cpu_model(ids)[0]
        logits_gpu = gpu_model(ids.to(device))[0].float().cpu()
    spread = float((logits_cpu - logits_gpu).abs().max())
    check("③ ⭐ 同权重·fp32·整段前向：跨设备 max|Δ| < 2e-3 ✓（差异只来自归约顺序 ✓）",
          spread < 2e-3, spread)
    check("③′ 末尾位置的 argmax 相同 ✓（token 级对齐 ✓）",
          int(logits_cpu[0, -1].argmax()) == int(logits_gpu[0, -1].argmax()))

    generated_cpu = llm_mod.generate_llm(cpu_model, ids, max_new_tokens=8, temperature=0).tolist()
    generated_gpu = llm_mod.generate_llm(gpu_model, ids, max_new_tokens=8, temperature=0).tolist()
    check("③″ ⭐⭐ greedy 生成**逐 token 相同** ✓（这是「两端结果一致」的硬判据 ✓）",
          generated_cpu == generated_gpu, (generated_cpu, generated_gpu))

    with torch.no_grad():
        _, cache_cpu = cpu_model(ids[:, :2], None)
        step_cpu = cpu_model(ids[:, 2:], cache_cpu)[0]
        _, cache_gpu = gpu_model(ids[:, :2].to(device), None)
        step_gpu = gpu_model(ids[:, 2:].to(device), cache_gpu)[0].float().cpu()
    cache_spread = float((step_cpu - step_gpu).abs().max())
    check("③‴ KV cache **增量**那条路也一致 ✓（只测整段 ⇒ 漏掉缓存的设备/精度错 ✗✗）",
          cache_spread < 2e-3, cache_spread)

    # ⚠️ 反验：把卡上模型的首个参数加 0.05 ⇒ 上面同一套判据**必须报** ✓（否则就是假绿 ✗✗）
    with torch.no_grad():
        next(gpu_model.parameters()).add_(0.05)
        corrupted_cpu = cpu_model(ids)[0]
        corrupted_gpu = gpu_model(ids.to(device))[0].float().cpu()
    corrupted_spread = float((corrupted_cpu - corrupted_gpu).abs().max())
    check("③⁴ 反验：权重差 0.05 ⇒ 同一判据**报出来** ✓（判据不是摆设 ✓）",
          corrupted_spread > 2e-3, corrupted_spread)


# ── ④ 低精度 vs fp32（GPU）✓
def case_forward_bf16() -> None:
    if _gpu_count() == 0:
        skip("④ bf16 与 fp32 的一致性（容差）", "本机没有 CUDA 卡")
        return
    device = f"cuda:{torch.cuda.current_device()}"
    fp32_model = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")
    low_model = llm_mod.build_llm(CONFIG, device=device, dtype="bfloat16")
    low_model.load_state_dict(fp32_model.state_dict())
    ids = torch.tensor([[3, 6, 9, 12, 15]])

    with torch.no_grad():
        reference = fp32_model(ids)[0]
        low = low_model(ids.to(device))[0].float().cpu()
    gap = float((reference - low).abs().max())
    scale = float(reference.abs().max()) + 1e-9
    check("④ ⭐ bf16 前向落在 fp32 的容差内 ✓（|Δ| < 2% 量级 ✓ —— bf16 只有 8 位尾数 ✓）",
          gap / scale < 0.02, (gap, scale, gap / scale))
    check("④′ bf16 下 logits **没有 NaN/Inf** ✓（升 fp32 的 norm / 掩码 dtype 那几处漏一个就炸 ✗✗）",
          bool(torch.isfinite(low).all()))
    check("④″ bf16 下末尾 argmax 与 fp32 相同 ✓（近似不改变结论 ✓）",
          int(reference[0, -1].argmax()) == int(low[0, -1].argmax()),
          (int(reference[0, -1].argmax()), int(low[0, -1].argmax())))

    with torch.no_grad():
        next(low_model.parameters()).add_(0.05)
        broken = low_model(ids.to(device))[0].float().cpu()
    broken_gap = float((reference - broken).abs().max()) / scale
    check("④‴ 反验：bf16 权重差 0.05 ⇒ 同一容差判据报出来 ✓", broken_gap >= 0.02, broken_gap)


# ── ⑤ 装载器：把 CPU 反量化的结果**直接装到指定设备** ✓
def _arr(shape: tuple[int, ...], offset: int) -> Any:
    """确定性、互不相同的小数值 ✓（防「两边都错得一样」✓）。"""
    count = int(np.prod(shape))
    return ((np.arange(count, dtype=np.float32) % 17 - 8) * 0.01 + offset).reshape(shape)


def _payload_of(model: Any) -> dict[str, Any]:
    """把模型的每个参数**倒着**变回 GGUF 名字 ✓。

    ⚠️ 值的形状 = ``_read_tensor`` 的契约（**已摆成 numpy 序** ✓）⇒ **不转置** ✓。
    旧版在这里 ``data.T`` 去「模拟 GGUF 的 (in,out)」✗✗ —— 那正好把装载器的摆法错误
    抹平成恒等（真机 8B 全程乱码、这里一路绿 ✗）。摆法对不对由 ``engine_gguf_to_llm_test``
    的 ②′ 逐格判 ✓，这里只管设备与装配 ✓。
    """
    reverse = {"embed.weight": "token_embd.weight", "norm.weight": "output_norm.weight",
               "lm_head.weight": "output.weight"}
    payload: dict[str, Any] = {}
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        shape = tuple(tensor.shape)
        data = _arr(shape, 0.01 * index)
        if name in reverse:
            payload[reverse[name]] = data
            continue
        layer, rest = name.split(".", 2)[1], name.split(".", 2)[2]
        if rest.startswith("attn."):
            field = rest.split(".")[1]
            ggml = {"q_proj": "attn_q", "k_proj": "attn_k",
                    "v_proj": "attn_v", "o_proj": "attn_output"}[field]
            payload[f"blk.{layer}.{ggml}.weight"] = data
        elif rest.startswith("ffn."):
            payload[f"blk.{layer}.ffn_{rest.split('.')[1]}.weight"] = data
        elif rest.startswith("norm1"):
            payload[f"blk.{layer}.attn_norm.weight"] = data
        elif rest.startswith("norm2"):
            payload[f"blk.{layer}.ffn_norm.weight"] = data
        else:                                             # pragma: no cover —— 名字变了要当场炸 ✓
            raise AssertionError(f"测试不知道 {name} 该叫什么 ✓")
    return payload


def case_loader_device() -> None:
    reference = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")
    payload = _payload_of(reference)
    info = types.SimpleNamespace(tensors=dict(payload), metadata={})
    original_inspect, original_read = g2l.gguf_mod.inspect, g2l._read_tensor
    g2l.gguf_mod.inspect = lambda path: info                        # type: ignore[assignment]
    g2l._read_tensor = lambda path, info, tensor: tensor             # type: ignore[assignment]
    # ⚠️ ``materialize=False`` 是**本用例的边界** ✓✗：这条假 ``info`` 的 ``tensors`` 直接就是裸数组 ✓，
    # 而物化计划（``gguf_to_llm._plan_specs`` 读 ``tensor.type_id``）打的是**真** TensorInfo ✓ ——
    # 真文件那条路由 ``engine_materialize_test`` 的 ``_fake_info`` 判 ✓。本用例只判**设备/数值** ✓
    # （装到 cpu/cuda 值不许变 ✓ + 不写死 cpu ✗），掺进物化只会让两件事互相顶 ✗。
    try:
        cpu_model = g2l.load_llm_from_gguf("<fake>", CONFIG, device="cpu", dtype="float32",
                                          materialize=False)
        # ⚠️ 逐张量核**值** ✓：装完的参数必须等于当初造数据时的那一份 ✓（转置在装载里被还原 ⇒ 原样 ✓）
        expected = {name: _arr(tuple(tensor.shape), 0.01 * index)
                    for index, (name, tensor) in enumerate(reference.state_dict().items())}
        loaded = cpu_model.state_dict()
        wrong = [name for name in expected
                 if not torch.equal(loaded[name].detach().cpu(), torch.from_numpy(expected[name]))]
        check("⑤ ⭐ 装载到 CPU：每个参数的值都对 ✓（转置是**往返**的 ✓ 不是单向丢掉 ✗）",
              not wrong, wrong[:3])
        check("⑤′ 转置真的**发生**了 ✓：``k_proj`` 是 (16,32) 非方阵 ✓（不转置会装不上/值错 ✓）",
              tuple(cpu_model.blocks[0].attn.k_proj.weight.shape) == (16, 32),
              tuple(cpu_model.blocks[0].attn.k_proj.weight.shape))
        if _gpu_count() == 0:
            skip("⑤″ 装载到 CUDA（参数真在卡上 + 与 CPU 逐张量相同）", "本机没有 CUDA 卡")
            return
        device = f"cuda:{torch.cuda.current_device()}"
        gpu_model = g2l.load_llm_from_gguf("<fake>", CONFIG, device=device, dtype="float32",
                                           materialize=False)
        check("⑤″ ⭐ 装载到卡上：参数 ``.device.type == cuda`` ✓（流式 copy_ 跨设备 ✓ 没有写死 cpu ✗）",
              all(parameter.device.type == "cuda" for parameter in gpu_model.parameters()))
        same = all(torch.equal(a.detach().cpu(), b.detach().cpu())
                   for a, b in zip(cpu_model.state_dict().values(), gpu_model.state_dict().values()))
        check("⑤‴ ⭐ CPU 装出来的与 GPU 装出来的**逐张量相同** ✓（同一条装载路径 ✓）", same)

        low_model = g2l.load_llm_from_gguf("<fake>", CONFIG, device=device, dtype="bfloat16",
                                           materialize=False)
        check("⑤⁴ 装载时 ``dtype`` 直达 ✓（装完就是 bf16 ✓ —— 不是装完再转 ✗）",
              next(low_model.parameters()).dtype is torch.bfloat16
              and next(low_model.parameters()).device.type == "cuda")
    finally:
        g2l.gguf_mod.inspect = original_inspect                     # type: ignore[assignment]
        g2l._read_tensor = original_read                            # type: ignore[assignment]


# ── ⑥ 后端：请求/实际两栏 ✓ 拒绝口径 ✓ 端到端跨设备一致 ✓
class _StubHub:
    """替身词表 ✓（只为把「设备口径」和「生成」隔开 ✓；词表本身另有专门自检 ✓）。"""

    vocab_size = 64

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1, 2, 3, 4]

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "IDS:" + ",".join(str(int(item)) for item in ids)


def _backend(tmp: Path, **kwargs: Any) -> Any:
    dummy = tmp / "fake.gguf"
    dummy.write_bytes(b"GGUF")                     # 只要求存在 ✓（装载器会被替身接管 ✓）
    backend = llm_backend.LlmBackend(CONFIG, gguf_path=str(dummy), tokenizer_path=str(dummy), **kwargs)
    backend._load_tokenizer = lambda source: _StubHub()   # type: ignore[method-assign]
    return backend


def case_backend_device(tmp: Path) -> None:
    fresh = _backend(tmp)
    report = fresh.describe()
    check("⑥ ⭐ describe 装载前：``device``（实际）是空 ✓ 而 ``deviceRequested`` 是 auto ✓（两栏不许混 ✗✗）",
          report["device"] == "" and report["deviceRequested"] == "auto", (report["device"], report["deviceRequested"]))
    check("⑥′ describe 装载前就把「择优」写清 ✓（``deviceNote`` 不是空话 ✓）",
          "择优" in report["deviceNote"], report["deviceNote"])

    # ⚠️ 拒绝口径：要一张**不存在**的卡 ⇒ load() 报错、且**不许**悄悄落到 cpu 上 ✓✗
    absent = f"cuda:{_gpu_count()}"
    doomed = _backend(tmp, device=absent)
    message = _message_of(doomed.load)
    check("⑥″ ⭐⭐ 后端显式要一张不存在的卡 ⇒ 装载**当场报错** ✓（静默回落 cpu ✗✗）",
          "LlmError" in message and absent in message, message)
    check("⑥‴ 失败后**不许**半装 ✓：``device`` 仍空、``loadError`` 有话说 ✓",
          doomed.describe()["device"] == "" and doomed.describe()["loadError"] != "",
          (doomed.describe()["device"], doomed.describe()["loadError"]))

    original_loader = g2l.load_llm_from_gguf
    base = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")

    def fake_loader(path: Any, config: Any, *, device: Any = "cpu", dtype: Any = None) -> Any:
        model = llm_mod.build_llm(config, device=device, dtype=dtype)
        model.load_state_dict(base.state_dict())     # 同一份权重 ✓
        return model

    g2l.load_llm_from_gguf = fake_loader            # type: ignore[assignment]
    try:
        cpu_backend = _backend(tmp, device="cpu", dtype="float32").load()
        check("⑥⁴ CPU 后端 ``device`` = cpu ✓、``dtype`` = float32 ✓（实际值 ✓ 不是请求值 ✗）",
              cpu_backend.describe()["device"] == "cpu"
              and cpu_backend.describe()["dtype"] == "float32", cpu_backend.describe()["device"])

        if _gpu_count() == 0:
            skip("⑥⁵ 端到端：同一权重在 cpu / cuda 上 greedy 生成逐字相同", "本机没有 CUDA 卡")
            return
        device = f"cuda:{torch.cuda.current_device()}"
        gpu_backend = _backend(tmp, device=device, dtype="float32").load()
        check("⑥⁵ ⭐ 后端把**实际**设备记下来 ✓（``cuda`` ✓ —— 要了卡就用卡 ✓）",
              gpu_backend.describe()["device"] == device, gpu_backend.describe()["device"])
        text_cpu = cpu_backend.generate("你好", temperature=0, max_new_tokens=6)
        text_gpu = gpu_backend.generate("你好", temperature=0, max_new_tokens=6)
        check("⑥⁶ ⭐⭐ 端到端：同一权重、同一提示词，cpu 与 cuda 的 greedy 输出**逐字相同** ✓",
              text_cpu == text_gpu, (text_cpu, text_gpu))
        check("⑥⁷ 生成真的跑出东西 ✓（不是空串 ✓）", text_gpu.startswith("IDS:") and len(text_gpu) > 8, text_gpu)
    finally:
        g2l.load_llm_from_gguf = original_loader    # type: ignore[assignment]


# ── ⑦ 分层放置的**算术**部分 ✓（不碰 torch ⇒ 哪台机器都要判 ✓ 2026-11-08 补 ✓）
def case_layer_plan() -> None:
    params = llm_mod._block_param_count(CONFIG)
    model = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")
    real = sum(int(p.numel()) for p in model.blocks[0].parameters())
    check("⑦ ⭐ 每层参数量是**算出来**的、与真模型逐项一致 ✓（抄配额表会随结构漂移 ✗）",
          params == real, (params, real))

    layer_bytes = params * 2
    devices, report = llm_mod.plan_layer_devices(CONFIG, gpu_bytes=layer_bytes + 8, dtype_bytes=2)
    check("⑦′ 预算够 1 层 ⇒ 卡上放**连续前缀** 1 层、其余在 CPU ✓（不是随机挑层 ✗）",
          devices == ["cuda", "cpu"] and report["gpuLayers"] == 1, (devices, report))

    devices, report = llm_mod.plan_layer_devices(CONFIG, gpu_bytes=layer_bytes // 2, dtype_bytes=2)
    check("⑦″ 一层都放不上 ⇒ 全 CPU 且 **why 写清原因** ✓（静默回落 cpu ✗✗）",
          devices == ["cpu", "cpu"] and bool(report["why"]), (devices, report["why"]))

    devices, report = llm_mod.plan_layer_devices(CONFIG, gpu_bytes=9 * layer_bytes, dtype_bytes=2,
                                                 want_gpu_layers=3)
    check("⑦‴ 想放 3 层但模型只有 2 层 ⇒ shortfall 如实报 ✓（悄悄减配 ✗）",
          report["gpuLayers"] == 2 and report["shortfall"] == 1, report)

    devices, report = llm_mod.plan_layer_devices(CONFIG, gpu_bytes=10 * layer_bytes, dtype_bytes=2,
                                                 reserve_bytes=9 * layer_bytes)
    check("⑦⁴ 预留真的从毛预算里扣掉 ✓（按毛预算算 ⇒ 纸面能装、真装 OOM ✗）",
          report["gpuLayers"] == 1 and report["usableBytes"] == layer_bytes, report)

    message = _message_of(lambda: llm_mod.plan_layer_devices(CONFIG, gpu_bytes=0, dtype_bytes=0))
    check("⑦⁵ dtype_bytes=0 ⇒ 具名拒绝 ✓（会算出除零或谎报能装 ✗）",
          "LlmError" in message and "dtype_bytes" in message, message)


# ── ⑧ 混合放置**真跑**：GPU 前缀 + CPU 尾巴 ✓
def _tokens(out: Any) -> list[Any]:
    return out.tolist() if hasattr(out, "tolist") else list(out)


def case_layer_split_forward() -> None:
    # 长度不符 / 与 device 互斥：**不需要卡**也能判 ✓（放前面 ✓）
    message = _message_of(lambda: llm_mod.build_llm(CONFIG, device="cpu",
                                                   layer_devices=["cpu", "cpu"]))
    check("⑧ ``device`` 与 ``layer_devices`` 同时点名 ⇒ 具名拒绝 ✓（不静默二选一 ✗）",
          "LlmError" in message and "layer_devices" in message, message)
    message = _message_of(lambda: llm_mod.build_llm(CONFIG, layer_devices=["cpu"]))
    check("⑧′ 层表长度 ≠ 层数 ⇒ 具名拒绝 ✓（不补默认值 ✗）",
          "LlmError" in message and "层数" in message, message)

    if _gpu_count() == 0:
        skip("⑧″ 混合放置前向 / KV 归属 / greedy 一致", "本机没有 CUDA 卡")
        return
    device = f"cuda:{torch.cuda.current_device()}"
    torch.manual_seed(7)
    cpu_model = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")
    split_model = llm_mod.build_llm(CONFIG, dtype="float32", layer_devices=[device, "cpu"])
    split_model.load_state_dict(cpu_model.state_dict())     # ⚠️ 同一份权重 ✓

    landed = split_model._layer_devices
    check("⑧″ ⭐ 第 0 层真在卡、第 1 层真在 CPU ✓（都落一处 = 没分层 ✗）",
          str(landed[0]).startswith("cuda") and str(landed[1]) == "cpu", landed)
    check("⑧‴ ``norm`` / ``lm_head`` 跟**末层**同设备 ✓（tie 时拆成两份 = 白涨一份词表 ✗）",
          split_model.norm.weight.device == landed[-1]
          and split_model.lm_head.weight.device == landed[-1],
          (str(split_model.norm.weight.device), str(split_model.lm_head.weight.device)))

    ids = torch.randint(0, CONFIG.vocab_size, (1, 5))
    with torch.inference_mode():
        want, _ = cpu_model(ids)
        got, kv = split_model(ids)
    diff = float((want - got).abs().max())
    check("⑧⁴ 混合放置的 logits 与全 CPU 一致 ✓（fp32 只差归约顺序 ✓）", diff < 1e-4, diff)
    check("⑧⁵ KV cache 留在**各自层**的设备上 ✓（全塞回一处 = 分层白做 ✗）",
          str(kv[0][0].device).startswith("cuda") and str(kv[1][0].device) == "cpu",
          (str(kv[0][0].device), str(kv[1][0].device)))

    with torch.inference_mode():
        a = _tokens(llm_mod.generate_llm(cpu_model, ids, max_new_tokens=4, temperature=0.0))
        b = _tokens(llm_mod.generate_llm(split_model, ids, max_new_tokens=4, temperature=0.0))
    check("⑧⁶ greedy 生成逐个 token 相同 ✓（跨设备判据用 temperature=0 ✓）", a == b, (a, b))

    tied = llm_mod.LlmConfig(vocab_size=64, hidden=32, depth=2, heads=4, kv_heads=2,
                             head_dim=8, ffn=64, tie_embeddings=True)
    tied_model = llm_mod.build_llm(tied, dtype="float32", layer_devices=[device, "cpu"])
    check("⑧⁷ tie_embeddings 下 embed 与 lm_head **仍是同一个 Parameter** ✓（.to 成两份 ✗）",
          tied_model.lm_head.weight is tied_model.embed.weight,
          str(tied_model.embed.weight.device))


# ── ⑨ 后端层的**分层放置**接线 ✓（配置 ⇒ 真装载路径 ✓ 2026-11-08 补 ✓）
def case_backend_gpu_layers(tmp: Path) -> None:
    original_loader = g2l.load_llm_from_gguf
    base = llm_mod.build_llm(CONFIG, device="cpu", dtype="float32")
    calls: list[dict[str, Any]] = []

    def fake_loader(path: Any, config: Any, *, device: Any = "cpu", dtype: Any = None,
                    layer_devices: Any = None) -> Any:
        calls.append({"device": device, "dtype": dtype,
                      "layer_devices": None if layer_devices is None else list(layer_devices)})
        # ⚠️ 与**真**装载器同一分支 ✓：给了层表就**不再**传 device ✓（两个都传会被 build_llm 拒 ✗）
        if layer_devices is not None:
            model = llm_mod.build_llm(config, dtype=dtype, layer_devices=layer_devices)
        else:
            model = llm_mod.build_llm(config, device=device, dtype=dtype)
        model.load_state_dict(base.state_dict())     # 同一份权重 ✓
        return model

    g2l.load_llm_from_gguf = fake_loader            # type: ignore[assignment]
    try:
        off = _backend(tmp, device="cpu", dtype="float32", gpu_layers="off").load()
        check("⑨ ``gpuLayers=off`` ⇒ **不分层** ✓（装载器收到的是 device、没收到层表 ✓ 老路径零变化 ✓）",
              off.describe()["layers"] is None and calls[-1]["layer_devices"] is None
              and calls[-1]["device"] == "cpu", (off.describe()["layers"], calls[-1]))

        zero = _backend(tmp, device="cpu", dtype="float32", gpu_layers=0).load()
        report = zero.describe()["layers"] or {}
        check("⑨′ ``gpuLayers=0`` ⇒ 全 CPU **且回执说是显式要求** ✓（悄悄全 CPU = 看不出配生效 ✗）",
              calls[-1]["layer_devices"] == ["cpu", "cpu"]
              and report.get("gpuLayers") == 0 and report.get("cpuLayers") == 2
              and "显式" in str(report.get("why")), (calls[-1], report))
        check("⑨″ 分层后 ``device`` 报**首层**在哪 ✓（报 embed 所在 = 分层时永远说 cpu ✗✗）",
              zero.describe()["device"] == "cpu" and "分层" in zero.describe()["deviceNote"],
              zero.describe()["deviceNote"])

        bad = _message_of(_backend(tmp, device="cpu", dtype="float32", gpu_layers="atou").load)
        check("⑨‴ ``gpuLayers`` 拼错 ⇒ 具名拒绝 ✓（当成分层与否都算「配了没生效」✗）",
              "LlmError" in bad and "gpuLayers" in bad, bad)

        over = _message_of(_backend(tmp, device="cpu", dtype="float32", gpu_layers=9).load)
        check("⑨⁴ ``gpuLayers`` 超过层数 ⇒ 具名拒绝 ✓（不悄悄截到 2 层 ✗）",
              "LlmError" in over and "层数" in over, over)

        nocard = _message_of(_backend(tmp, device="cpu", dtype="float32", gpu_layers=1).load)
        check("⑨⁵ 点名 N 层上卡但设备是 cpu ⇒ **具名拒绝** ✓（静默退回全 CPU ✗✗）",
              "LlmError" in nocard and "gpuLayers" in nocard and "CUDA" in nocard, nocard)

        if _gpu_count() == 0:
            skip("⑨⁶ 后端分层真跑（前 1 层在卡 + greedy 与全 CPU 逐字相同）", "本机没有 CUDA 卡")
            return
        device = f"cuda:{torch.cuda.current_device()}"
        split = _backend(tmp, device=device, dtype="float32", gpu_layers=1).load()
        layers = split.describe()["layers"] or {}
        check("⑨⁶ ⭐ 配置 ``gpuLayers=1`` ⇒ 装载器真收到层表 ✓、首层真在卡 ✓",
              calls[-1]["layer_devices"] == [device, "cpu"]
              and layers.get("gpuLayers") == 1 and layers.get("cpuLayers") == 1
              and split.describe()["device"] == device,
              (calls[-1], layers, split.describe()["device"]))
        first = next(split._model.blocks[0].parameters())
        check("⑨⁷ 卡上那一层的参数真在 CUDA ✓（表说在卡、实际在 CPU = 分层没落地 ✗✗）",
              first.device.type == "cuda", str(first.device))

        plain = _backend(tmp, device="cpu", dtype="float32").load()
        check("⑨⁸ ⭐⭐ 后端端到端：分层放置与全 CPU 的 greedy 输出**逐字相同** ✓",
              split.generate("你好", temperature=0, max_new_tokens=6)
              == plain.generate("你好", temperature=0, max_new_tokens=6))

        auto = _backend(tmp, device=device, dtype="float32", gpu_layers="auto").load()
        auto_report = auto.describe()["layers"] or {}
        check("⑨⁹ ``gpuLayers=auto`` ⇒ 按此刻余量算出层数、**why 非空** ✓（0 层也要给理由 ✗）",
              0 <= int(auto_report.get("gpuLayers", -1)) <= 2 and bool(auto_report.get("why")),
              auto_report)

        from app.services.text_generation import engine_gpu_layers   # noqa: PLC0415
        check("⑨¹⁰ settings 两种键名都认 ✓（``gpuLayers`` ／ ``gpu_layers`` ✓）",
              engine_gpu_layers({"gpuLayers": 4}) == 4
              and engine_gpu_layers({"gpu_layers": "auto"}) == "auto"
              and engine_gpu_layers({}) is None, engine_gpu_layers({"gpu_layers": "auto"}))
    finally:
        g2l.load_llm_from_gguf = original_loader    # type: ignore[assignment]


def main() -> int:
    case_devices()
    case_dtype()
    case_forward_fp32()
    case_forward_bf16()
    case_loader_device()
    case_layer_plan()
    case_layer_split_forward()
    with tempfile.TemporaryDirectory(prefix="llmdev_") as handle:
        case_backend_device(Path(handle))
        case_backend_gpu_layers(Path(handle))
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for name, why in _SKIPPED:
        print(f"SKIP  {name} —— {why}（SKIP ≠ 通过 ✓）")
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          f"（另有 {len(_SKIPPED)} 条 SKIP，不计入 ✓）")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
