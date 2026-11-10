r"""自研引擎 · **流式装载**（逐张量读 → 反量化 → 写参数 ✓）—— 「小内存跑大模型」的第一步 ✓。

## 为什么单独一个自检（这条在关键路径上 ✓）

旧装载路径用 ``safetensors.torch.load_file`` 把**整份**检查点读进内存 ✗，
19.53 GiB 的 fp8 再经**整份** ``dequantize_state`` 变 bf16 ✓✗ ⇒ 峰值接近 **2 倍模型** ✓✗。
改成流式后必须证明三件事，**缺一条就等于没改** ✗：

1. **结果一样**（逐位相等 ✓）—— 省内存不能以「装错」为代价 ✗；
2. **真的没有整份读**（官方 ``load_file`` 一次都不调 ✓、同一时刻只有**一个**张量的缓冲 ✓）；
3. **失败时模型零污染**（形状不符 / 布局判不出来 / scale 有 0 ✓ 都要在**写第一个字节之前**停 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_weights_streaming_test.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import tracemalloc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import safetensors as st_mod  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _skips() -> list[str]:
    try:
        _torch()
        from safetensors.torch import save_file  # noqa: F401, PLC0415
    except ImportError:
        return ["未装 torch / safetensors ⇒ 流式装载自检跳过（它要造真 safetensors 文件 ✓）"]
    return []


def _tiny() -> Any:
    """两个参数 + 一个 bias 的最小模块 ✓（够验证「多键、含 1 维权重、含 bias」✓）。"""
    torch = _torch()

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 2)

    return Tiny()


def _save(root: Path, name: str, tensors: dict[str, Any]) -> Path:
    from safetensors.torch import save_file  # noqa: PLC0415

    path = root / name
    save_file({key: tensor.detach().clone() for key, tensor in tensors.items()},
              str(path), metadata={})
    return path


def _snapshot(module: Any) -> dict[str, Any]:
    torch = _torch()
    with torch.no_grad():
        return {key: tensor.clone() for key, tensor in module.state_dict().items()}


def _same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    torch = _torch()
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


# ══════════════════════════════════════════════════════════════════════════
# ① 结果必须与「整份读 + load_state_dict」逐位相同
# ══════════════════════════════════════════════════════════════════════════
def case_bitwise_equal(root: Path) -> None:
    torch = _torch()
    torch.manual_seed(11)
    model = _tiny()
    path = _save(root, "plain.safetensors", model.state_dict())

    expected = _tiny()
    with torch.no_grad():
        from safetensors.torch import load_file  # noqa: PLC0415

        expected.load_state_dict(load_file(str(path), device="cpu"))

    fresh = _tiny()
    report = weights_mod.load_module_weights(fresh, path)
    check("① ⭐⭐ 流式装载与「整份读 + `load_state_dict`」**逐位相同** ✓"
          "（省内存不能以装错为代价 ✗）",
          report.complete and _same(_snapshot(fresh), _snapshot(expected)),
          report.to_dict()["error"])

    check("①′ 报告契约 ✓（`to_dict` 的键集合**只增不改** ✓ —— 前端在读这些字段 ✓；"
          "2026-10-29 增 `consumedQuantMeta` / `int8Resident` ✓；2026-10-30 增 `consumedNvfp4` ✓ "
          "与 `consumedScales` ✓（后者数的是 `scale_weight` / `scale_input` / `scaled_fp8` 这一类"
          "**权重侧与激活侧** scale ✓ —— 不吃掉它们就会全落进 `unexpected` ✓✗，"
          "而吃掉了不计数就是**静默** ✓✗）—— 四个都是**只读计数** ✓，且都只统计"
          "「被吃掉的配套张量」✗ 不是权重 ✓；2026-11-02 再增 `int4Resident` ✓"
          "（4bit 打包常驻层的子计数 ✓ —— 它的参数也是 int8 ✓，合计数里看不出「说好 4bit、实际按 int8 乘」✓✗）；"
          "2026-11-07 再增 `skipped` / `int8Skipped` ✓（内存这一级卸掉多少键、其中多少本来是 int8 常驻层 ✓"
          "—— 少了这两格，回执里只看得到 `loadedKeys` 变小 ✓✗，读不出「是策略卸的还是真缺」✓✗）；"
          "2026-11-09 再增 `nvfp4Resident` ✓（其中 **nvfp4 打包**的子计数 ✓ —— nvfp4 与 int8 的目标参数"
          "在报告里同样只体现为「常驻了 N 层」✓，不单列就答不出「那 14.61 GiB 的件按哪条路装的」✓✗；"
          "它与 `dequant.nvfp4`（装载时真还原的层数 ✓）是**两个数** ✓，常驻壳那条路上后者恒 0 ✓）",
          set(report.to_dict()) == {
              "path", "complete", "loadedKeys", "sourceKeys", "targetKeys", "missing",
              "missingCount", "unexpected", "unexpectedCount", "shapeMismatch",
              "shapeMismatchCount", "renamed", "dtypeCasts", "dequant", "error",
              "consumedQuantMeta", "int8Resident", "int4Resident", "nvfp4Resident",
              "consumedNvfp4", "consumedScales", "skipped", "int8Skipped"},
          sorted(report.to_dict()))


# ══════════════════════════════════════════════════════════════════════════
# ② 官方 load_file 一次都不许调（「不再整份读」的唯一硬证据）
# ══════════════════════════════════════════════════════════════════════════
def case_no_official_load_file(root: Path) -> None:
    torch = _torch()
    torch.manual_seed(12)
    model = _tiny()
    path = _save(root, "noload.safetensors", model.state_dict())

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("装载路径不该再调用 safetensors.torch.load_file ✗（那是整份读 ✓✗）")

    import safetensors.torch as st_torch  # noqa: PLC0415

    original = st_torch.load_file
    st_torch.load_file = _forbidden
    try:
        report = weights_mod.load_module_weights(_tiny(), path)
    finally:
        st_torch.load_file = original
    check("② ⭐⭐ 装载**全程不碰** `safetensors.torch.load_file` ✓"
          "（这是「不再整份读」唯一能被断言的证据 ✓ —— 靠 RSS 波动测不出来 ✗）",
          report.complete, report.error[:120])


# ══════════════════════════════════════════════════════════════════════════
# ③ 每个键只读一次，且形状不符时**一个字节都不读**
# ══════════════════════════════════════════════════════════════════════════
class _ReadCounter:
    """包住 ``TensorReader.read`` 记录每次读的名字与字节数 ✓（不改实现 ✓）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __enter__(self) -> "_ReadCounter":
        self._original = st_mod.TensorReader.read

        def _wrapped(reader: Any, name: str) -> bytes:
            payload = self._original(reader, name)
            self.calls.append((name, len(payload)))
            return payload

        st_mod.TensorReader.read = _wrapped
        return self

    def __exit__(self, *_exc: Any) -> bool:
        st_mod.TensorReader.read = self._original
        return False

    @property
    def total_bytes(self) -> int:
        return sum(size for _name, size in self.calls)


def case_reads(root: Path) -> None:
    torch = _torch()
    torch.manual_seed(13)
    model = _tiny()
    path = _save(root, "reads.safetensors", model.state_dict())
    expected_keys = len(model.state_dict())

    with _ReadCounter() as counter:
        report = weights_mod.load_module_weights(_tiny(), path)
    check("③ 每个键**只读一次** ✓（不重读头部、不做第二遍扫描 ✓）",
          report.complete and len(counter.calls) == expected_keys
          and counter.total_bytes == path.stat().st_size - 8 - _header_bytes(path),
          (len(counter.calls), expected_keys))

    bad_tensors = dict(model.state_dict())
    first = sorted(bad_tensors)[0]
    bad_tensors[first] = torch.zeros(3, 5)          # ✗ 形状对不上
    bad_path = _save(root, "badshape.safetensors", bad_tensors)
    with _ReadCounter() as counter:
        bad = weights_mod.load_module_weights(_tiny(), bad_path)
    check("④ ⭐⭐ 形状不符 ⇒ **一个字节都不读** ✓ 就中止（旧实现已经把 20 GiB 拉进内存了 ✗）",
          bad.complete is False and bool(bad.shapeMismatch) and not counter.calls
          and "中止装载" in bad.error,
          (len(counter.calls), bad.error[:100]))


def _header_bytes(path: Path) -> int:
    import struct  # noqa: PLC0415

    with path.open("rb") as handle:
        return struct.unpack("<Q", handle.read(8))[0]


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 峰值不随张量数累加（这才是「小内存」的定义 ✓ 不是「读得快」✓）
# ══════════════════════════════════════════════════════════════════════════
def case_peak_is_single_tensor(root: Path) -> None:
    torch = _torch()
    torch.manual_seed(14)
    count, rows, cols = 8, 2048, 1024
    each = rows * cols * 2                      # bf16 ⇒ 4 MiB
    tensors = {f"w{i}.weight": torch.randn(rows, cols, dtype=torch.bfloat16)
               for i in range(count)}
    path = _save(root, "big.safetensors", tensors)

    class Wide(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, tensor in tensors.items():
                self.register_parameter(name.replace(".", "_"),
                                        torch.nn.Parameter(torch.zeros_like(tensor)))

    #: ⚠️ 这里**不预造模块** ✗：两档各造一份（在 ``_peak`` 里 ✓）—— 预造一份会被 ⑤′ 误当成
    #: 「装好的那份」✓✗（旧版就在这一行的 ``module`` 上核 state_dict ✓ ⇒ 改成两档后恒红 ✓）
    names = {key: key.replace(".", "_") for key in tensors}
    # 用 key_map 把 `w0.weight` 映射成 `w0_weight` ✓（顺带验证改名仍在流式路径上生效 ✓）
    # ⚠️ 必须 `re.escape` ✗：键里的 `.` 是**字面点** ✓，当通配符用会「恰好也能过」✓✗
    key_map = {re.escape(key): target for key, target in names.items()}

    def _peak(read_ahead: int) -> tuple[weights_mod.WeightLoadReport, int, Any]:
        """重新造一份空模块再装一遍 ⇒ 返回（报告, 峰值增长字节, **装进去的那份模块**）✓。

        ⚠️ ``Wide()`` 必须在 ``tracemalloc.start()`` **之前**造 ✓（它自己就要 8 × 4 MiB ✗，
        算进 baseline 之外会把"流式"的账全冲掉 ✓✗）。
        ⚠️ 模块要**随报告一起返回** ✗✗：⑤′ 要核"参数真被写入" ✓，而每次 ``_peak`` 造的都是新模块 ✓
        ⇒ 拿外面那个没装过的 ``module`` 去核 ⇒ 恒红 ✓（第一次改完就是这样 ✓✗）。
        """
        fresh = Wide()
        tracemalloc.start()
        try:
            baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            report = weights_mod.load_module_weights(fresh, path, key_map=key_map,
                                                     read_ahead=read_ahead)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return report, peak - baseline, fresh

    #: ⚠️ 分两档量 ✗✗（旧的单档 8 MiB 断言**一直红** ✓，而它看起来像"流式坏了" ✓✗）：
    #: 装载默认开**读前置**（``weights._READ_AHEAD = 2`` ✓）⇒ 峰值本来就是「当前件 + 2 个预取件」
    #: ＝ 3 × 4 MiB ＝ 12.03 MiB ✓（实测 ✓）—— 那是预取的**设计** ✓，不是"把整份读进来了" ✗。
    #: 单档断言把两件事混成一条 ⇒ 要么把预取关掉（等于不测真默认路径 ✗），要么把预算放宽到
    #: 32 MiB（那样"整份读进来"也过 ✗✗）⇒ 只能拆开 ✓。
    report, growth, _loaded = _peak(0)
    budget = each * 2                            # 关预取：只允许「当前件 + 落地副本」
    check("⑤ ⭐⭐ 关掉预取时峰值 ≈ **单个张量** ✓ 不是整份（8 × 4 MiB = 32 MiB ✗）"
          "—— 「小内存」要靠这条断言证明 ✓ 不能靠「感觉变快了」✓",
          report.complete and growth < budget,
          {"growthMiB": round(growth / 1048576, 2), "budgetMiB": budget // 1048576,
           "totalMiB": count * each // 1048576})

    ahead = int(getattr(weights_mod, "_READ_AHEAD", 0))
    report_ahead, growth_ahead, loaded = _peak(ahead)
    limit = each * (ahead + 1) + each // 2        # 当前件 + ahead 个预取件 + 半件的余量
    check("⑤″ 开预取（默认档）时峰值 ≤ **（1 + read_ahead）件** ✓✗ —— 预取只在**件**级别叠加 ✓，"
          "不是「多读一整份」✗（真整份 = 32 MiB ⇒ 这条会红 ✓）",
          report_ahead.complete and growth_ahead < limit,
          {"growthMiB": round(growth_ahead / 1048576, 2), "readAhead": ahead,
           "limitMiB": limit // 1048576, "totalMiB": count * each // 1048576})
    check("⑤′ 改名在流式路径上仍生效 ✓（`renamed` 有值 + 参数真被写入 ✓）",
          report_ahead.renamed == count and all(
              bool((loaded.state_dict()[target] != 0).any()) for target in names.values()),
          report_ahead.renamed)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 预检：坏 scale / 0 scale 都要在**写第一个字节之前**停 ⇒ 模型零污染
# ══════════════════════════════════════════════════════════════════════════
def case_precheck_no_pollution(root: Path) -> None:
    torch = _torch()
    torch.manual_seed(15)
    model = _tiny()
    base = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    weight = base["a.weight"].to(torch.float32)
    scale = torch.full((weight.shape[0],), 0.01)
    base["a.weight"] = (weight / scale.reshape(-1, 1)).to(torch.float8_e4m3fn)
    base["a.weight_scale"] = scale

    good_path = _save(root, "fp8good.safetensors", base)
    fresh = _tiny()
    report = weights_mod.load_module_weights(fresh, good_path)
    check("⑥ ⭐ fp8 走流式路径**照样装得进** ✓ 且明细里有布局 / 来源 scale / 原 dtype ✓",
          report.complete and report.dequant.get("count") == 1
          and report.dequant["converted"]["a.weight"]["layout"] == "per-row"
          and report.dequant["converted"]["a.weight"]["from"] == "torch.float8_e4m3fn",
          report.dequant)

    # ⑦ scale 形状对不上任何布局（块量化）⇒ 中止 + **只读过 scale**（没读大权重）
    broken = dict(base)
    broken["a.weight_scale"] = torch.ones(3, 3)
    broken_path = _save(root, "fp8broken.safetensors", broken)
    victim = _tiny()
    before = _snapshot(victim)
    with _ReadCounter() as counter:
        bad = weights_mod.load_module_weights(victim, broken_path)
    biggest = max((size for _name, size in counter.calls), default=0)
    check("⑦ ⭐⭐ 布局判不出来 ⇒ **中止装载** ✓ 且目标模块**一个参数都没动** ✓"
          "（逐张量装载一旦写了一半就没救了 ✓ —— 所以必须有预检 ✓）",
          bad.complete is False and "中止装载" in bad.error and _same(_snapshot(victim), before),
          bad.error[:120])
    check("⑦′ ⭐ 而且**只读了 scale** ✓（最大一次读 < 4 KiB ⇒ 没把权重读进来 ✓）",
          biggest < 4096, {"calls": len(counter.calls), "biggest": biggest})

    # ⑧ scale 里有 0（且是 `_inv` 方向）⇒ 除 0 出 inf ⇒ 同样在写入前拒绝
    zeroed = dict(base)
    zeroed.pop("a.weight_scale")
    zeroed["a.weight_scale_inv"] = torch.zeros(weight.shape[0])
    zero_path = _save(root, "fp8zero.safetensors", zeroed)
    victim2 = _tiny()
    before2 = _snapshot(victim2)
    with _ReadCounter() as counter2:
        zero = weights_mod.load_module_weights(victim2, zero_path)
    check("⑧ ⭐ scale 里有 0 ⇒ 预检拒绝 ✓ 且**没写任何参数** ✓（0 是坏 scale ✗，不是「合法的 1」✗）",
          zero.complete is False and "0" in zero.error
          and _same(_snapshot(victim2), before2)
          and max((size for _name, size in counter2.calls), default=0) < 4096,
          zero.error[:120])


# ══════════════════════════════════════════════════════════════════════════
# ⑨ 复刻那条要求：`report.error` 不允许把「读不到文件」说成别的东西
# ══════════════════════════════════════════════════════════════════════════
def case_errors(root: Path) -> None:
    torch = _torch()
    missing = weights_mod.load_module_weights(_tiny(), root / "nope.safetensors")
    check("⑨ 缺文件 ⇒ 一句话说清是缺文件 ✓（不冒 `WinError 2` ✓）",
          missing.complete is False and "不存在" in missing.error, missing.error[:80])

    gguf = root / "fake.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    report = weights_mod.load_module_weights(_tensor_model(), gguf)
    check("⑨′ `.gguf` 交给 `TensorReader` ⇒ 明确说「这是另一套容器格式」✓（不猜 ✓）",
          report.complete is False and "GGUF" in report.error, report.error[:80])


def _tensor_model() -> Any:
    torch = _torch()

    class One(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, 2))

    return One()


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 0 元素张量（哨兵）不许炸
# ══════════════════════════════════════════════════════════════════════════
def case_zero_element(root: Path) -> None:
    """⚠️ **0 元素张量**是「键存在即真」的哨兵 ✓（Edit 2511 的 `__index_timestep_zero__`
    在 int8_convrot 那份里就是 `shape=[0]` ✓、TE 的 `scaled_fp8` 同物种 ✓）。

    2026-11-01 真权重首跑就在这里崩 ✓：`torch.frombuffer(b"")` 抛
    `both buffer length (0) and count (-1) must not be 0` ✓ —— **报错落在 torch 上** ✓，
    读起来像「文件坏了 / 权重是空的」✓✗，而真相是「它本来就该有 0 个字节」✓。
    """
    torch = _torch()

    class Sentinel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Linear(2, 2)
            self.register_buffer("sentinel", torch.zeros(0), persistent=True)

    model = Sentinel()
    path = _save(root, "sentinel.safetensors", model.state_dict())
    fresh = Sentinel()
    report = weights_mod.load_module_weights(fresh, path)
    check("⑤ ⭐ 0 元素张量（哨兵 ✓）能装 ✓（`frombuffer(b\"\")` 会当场抛 ✗ —— 真权重踩过 ✓✗）",
          report.complete and fresh.sentinel.numel() == 0, report.to_dict()["error"][:200])

    #: 反过来：0 元素**不能**被当成「缺键」✓（缺键会让整份判不完整 ✓✗，Edit 2511 就装不起来 ✗）
    check("⑤′ 0 元素张量**不算缺键** ✓（`missing` 里出现它 ⇒ 整份判不完整 ✗）",
          not [k for k in report.missing if "sentinel" in k], report.missing)


def main() -> int:
    skips = _skips()
    if skips:
        for reason in skips:
            check(f"SKIP: {reason}", True, reason)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_bitwise_equal(root)
            case_no_official_load_file(root)
            case_reads(root)
            case_peak_is_single_tensor(root)
            case_precheck_no_pollution(root)
            case_errors(root)
            case_zero_element(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 GBK ⇒ 不加这行，裸跑第一条 print 就 UnicodeEncodeError 崩 ✗
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
