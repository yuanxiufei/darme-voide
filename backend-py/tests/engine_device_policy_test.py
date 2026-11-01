"""设备策略自检 ✓（GPU 优先 + 压力自适应 ✓ 2026-10-31 ✓）。

**为什么单独一份** ✗：这套逻辑是**纯决策** ✓（给定卡况与需求 ⇒ 谁干活 ✓），
不装权重、不跑前向 ✓ ⇒ 可以在**任何**机器上跑全绿 ✓（有卡没卡都行 ✓）。
真前向与换装行为在 `engine_qwen21_test.py` 的 ⑥ / ⑦ 里验 ✓。

判据口径（**别写松** ✗✗）：
- 「卡够」**不是**"能装下"✗ 而是"装下之后**还剩 1 GiB**"✓（刚好装下 ⇒ 真前向必 OOM ✓✗）；
- 「占用率过高」判的是**显存**✓，不是 SM 利用率 ✗（本仓没有 nvml ✓，见 `device_policy` 模块头 ✓）；
- 降级只换**设备/驻留方式** ✓，**不许**顺手动步数与分辨率 ✗✗（那是创作参数 ✓）。
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import device_policy as dp  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402

_FAILS: list[str] = []
_SKIPS: list[str] = []
GIB = dp.GIB


def check(label: str, ok: bool, detail: object = "") -> None:
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _p(total_gib: float, free_gib: float) -> dp.Pressure:
    return dp.Pressure(totalBytes=int(total_gib * GIB), freeBytes=int(free_gib * GIB))


# ---------------------------------------------------------------- ① 纯决策


def case_decide() -> None:
    need = int(13.26 * GIB)

    d = dp.decide(need_bytes=need, pressure=_p(22.49, 20.0))
    check("① 卡上够（余 20 GiB ≥ 13.26＋1 GiB）⇒ 上卡 ✓、不降级 ✓",
          d.device == "cuda" and d.residency == "off" and not d.degraded, d.to_dict())
    check("①′ 决策要带**需求**与**读数** ✓✗（只给结论无法复核 ✓）",
          d.needGiB == 13.26 and (d.pressure or {}).get("freeGiB") == 20.0, d.to_dict())

    d = dp.decide(need_bytes=need, pressure=_p(22.49, 13.9))
    check("①″ 只差 1 GiB 余量 ⇒ 也算**装不下** ✓✗（刚好装下＝真前向必 OOM ✓）",
          d.residency == "split" and d.degraded, d.to_dict())

    d = dp.decide(need_bytes=need, pressure=_p(22.49, 6.0))
    check("①‴ 放不下整件但 ≥ 0.5 GiB ⇒ **GPU/CPU 联合** ✓（不是整份退 CPU ✓）",
          d.device == "cuda" and d.residency == "split" and d.degraded, d.to_dict())

    d = dp.decide(need_bytes=need, pressure=_p(22.49, 0.2))
    check("①⁴ 余量 < 0.5 GiB ⇒ 整件**转移 CPU** ✓（卡上放不下有意义的切片 ✓）",
          d.device == "cpu" and d.degraded, d.to_dict())

    d = dp.decide(pressure=_p(22.49, 1.2))
    check("①⁵ 需求**未知**且已用 ≥92% ⇒ 转移 CPU ✓（不许赌还塞得下 ✓）",
          d.device == "cpu" and d.degraded, d.to_dict())
    d = dp.decide(pressure=_p(22.49, 3.0))
    check("①⁶ 需求未知但已用 ≥80% ⇒ 上卡 + **联合驻留** ✓（保守但仍用卡 ✓）",
          d.device == "cuda" and d.residency == "split", d.to_dict())


# ---------------------------------------------------------------- ② 策略与显式请求


def case_policy() -> None:
    need = int(13.26 * GIB)
    check("② `policy=cpu` ⇒ 一律 CPU ✓ 且**不算降级** ✓（是用户选的 ✓）",
          dp.decide(policy="cpu", need_bytes=need, pressure=_p(22.49, 20.0)).device == "cpu"
          and not dp.decide(policy="cpu", pressure=_p(22.49, 20.0)).degraded)
    check("②′ `policy=gpu` ⇒ 一律上卡 ✓（显存不够也**不转移** ✓，OOM 留给装载层喊 ✓）",
          dp.decide(policy="gpu", need_bytes=need, pressure=_p(22.49, 0.1)).device == "cuda")
    check("②″ 显式 `device=cpu` ⇒ 尊重 ✓（策略不做主 ✓）",
          dp.decide(requested_device="cpu", policy="auto",
                    need_bytes=need, pressure=_p(22.49, 20.0)).device == "cpu")
    check("②‴ 显式 `device=cuda` **不**拦降级 ✓（钉了卡但卡满 ⇒ 还是转移 ✓，否则必 OOM ✓✗）",
          dp.decide(requested_device="cuda", policy="auto",
                    need_bytes=need, pressure=_p(22.49, 0.1)).device == "cpu")
    d = dp.decide(policy="balanced", need_bytes=need, pressure=_p(22.49, 5.0))
    check("②⁴ `balanced` 在装得下时也主动**联合** ✓（给别的任务留余量 ✓）",
          d.residency == "split" and d.degraded, d.to_dict())
    check("②⁵ 无卡 ⇒ CPU ✓ 且**如实说**是降级 ✓",
          dp.decide(gpu_available=False).device == "cpu"
          and dp.decide(gpu_available=False).degraded)
    check("②⁶ 测不到读数 ⇒ 按**默认优先 GPU** ✓（没有依据就不降级 ✓）",
          dp.decide(pressure=None).device == "cuda")
    try:
        dp.normalize_policy("gpu-frist")
    except ValueError:
        check("②⁷ 拼错的策略名 ⇒ **当场抛** ✓✗（不许静默回落 auto ✓）", True)
    else:
        check("②⁷ 拼错的策略名 ⇒ **当场抛** ✓✗（不许静默回落 auto ✓）", False, "没抛")
    check("②⁸ 别名归一 ✓（`gpu-first` / `split` / `default` 都认得 ✓）",
          dp.normalize_policy("gpu-first") == "auto"
          and dp.normalize_policy("split") == "balanced"
          and dp.normalize_policy(None) == "auto")


# ---------------------------------------------------------------- ③ 读数


def case_probe() -> None:
    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info(index=None):  # noqa: ARG004
            return (10 * GIB, 22 * GIB)

        @staticmethod
        def memory_allocated(index=None):  # noqa: ARG004
            return 12 * GIB

        @staticmethod
        def memory_reserved(index=None):  # noqa: ARG004
            return 12 * GIB          # 与 allocated 相等 ⇒ 没有"留着没还"的缓存 ✓

    class _FakeStale:
        """allocator 里压着 17 GiB **没还的缓存**、活张量 0 ✓ ⇒ 真实可用 1+17=18 GiB ✓。"""

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info(index=None):  # noqa: ARG004
            return (1 * GIB, 22 * GIB)

        @staticmethod
        def memory_allocated(index=None):  # noqa: ARG004
            return 0

        @staticmethod
        def memory_reserved(index=None):  # noqa: ARG004
            return 17 * GIB

    class _T1:
        cuda = _NoCuda()

    class _T2:
        cuda = _FakeCuda()

    class _T3:
        cuda = _FakeStale()

    check("③ 没卡 ⇒ 读数 `None` ✓（**不是** 0 ✓✗ —— 0 会被读成「满得一点不剩」✓）",
          dp.probe(_T1()) is None)
    read = dp.probe(_T2())
    check("③′ 有卡 ⇒ 余量/总量/已用都是从 `mem_get_info` 实数 ✓",
          read is not None and read.freeBytes == 10 * GIB and read.totalBytes == 22 * GIB
          and read.usedBytes == 12 * GIB, None if read is None else read.to_dict())
    check("③″ 没接 nvml ⇒ `source` 必须是 `memory` ✓✗（**不许**把显存占用说成 SM 利用率 ✓）",
          read is not None and read.source == "memory")
    check("③‴ `source` 可切 `nvml` ✓（将来接了 nvml 判据一行不用改 ✓）",
          (dp.probe(_T2(), util_percent=93.0) or dp.Pressure(1, 1)).source == "nvml")

    # ③⁴ ⭐⭐ **"留着没还"的缓存块必须算进可用** ✓✗（2026-11-01 端到端抓的真缺陷 ✓）。
    #    `mem_get_info` 的 free 是驱动视角 ✓ —— 我们自己 allocator 压着的缓存块它当成"已占" ✓✗，
    #    可那些块**随时能重用** ✓。不补这 17 GiB 的后果：换装时 probe 报「余 5.44 GiB」✓
    #    （真实可用 21 GiB ✓）⇒ 25 GiB 的塔只上卡 3.5 GiB ✓、17.9 GiB 的 DiT 只上卡 4 GiB ✓
    #    ⇒ 越换越保守 ✓，最后卡被 WDDM 超分（`reserved` 一度 36402 MiB > 卡 23027 MiB ✓✗）。
    stale = dp.probe(_T3())
    check("③⁴ `freeBytes` ＝ 驱动空闲 ＋ **allocator 可回收缓存** ✓✗（只报驱动那个数 ⇒ 换装越换越紧 ✓）",
          stale is not None and stale.freeBytes == 18 * GIB
          and stale.usedBytes == 4 * GIB
          and abs(stale.usedRatio - (1 - 18 / 22)) < 1e-9,
          None if stale is None else stale.to_dict())

    # ③⁵ 切分预算与 probe **同一口径** ✓✗（两处各写一套 ⇒ 一处补了缓存、另一处没补 ✓ 必错 ✓）
    from app.services.engine import residency as res  # noqa: PLC0415

    check("③⁵ `residency.budget_from_cuda` 与 `probe` 同口径 ✓✗（都算上可回收缓存 ✓；"
          "只补其中一处 ⇒ 决策说 18 GiB、切分只肯用 1 GiB ✓）",
          res.budget_from_cuda(_T3(), reserve=0.0) == 18 * GIB,
          res.budget_from_cuda(_T3(), reserve=0.0))


# ---------------------------------------------------------------- ④ 后端接入


def case_backend() -> None:
    real_probe = dp.probe
    try:
        # 「卡上只剩 0.2 GiB」——比任何一件都放不下
        dp.probe = lambda *a, **k: _p(22.49, 0.2)  # type: ignore[assignment]
        #: ⚠️ 用 **`device=None`**（＝auto 语义 ✓）而不是 `"cuda"` ✗：显式钉卡是调用方的决定 ✓，
        #: 装配期**不擅自改**（改了会变成「我说了用卡，它自己跑内存里去了」✓✗）。
        backend = tb.TorchBackend(device=None, policy="auto")
        check("④ 压力爆表 ⇒ 构造期就**不上卡** ✓✗（老路只看「有没有卡」⇒ 卡满也硬上 ⇒ OOM ✓）",
              backend.device == "cpu", backend.device)
        check("④′ 转过去之后**精度跟着变 fp32** ✓✗（在 CPU 上按 bf16 建模块会撞未实现算子 ✓）",
              str(backend._compute_dtype()).endswith("float32"), backend._compute_dtype())

        # 卡空出来了 ⇒ 下一阶段该**自动回切** GPU
        dp.probe = lambda *a, **k: _p(22.49, 20.0)  # type: ignore[assignment]
        decision = backend.decide_for(("dit",))
        check("④″ 卡空了 ⇒ 决策**回到 GPU** ✓✗（只在上卡时才探卡的话，一旦降级就再也回不去 ✓✗）",
              decision.device == "cuda" and not decision.degraded, decision.to_dict())
        backend._apply_decision(decision)
        check("④‴ 回切之后设备与策略实况都对 ✓", backend.device == "cuda"
              and (backend.describe().get("devicePolicy") or {}).get("lastDecision") is not None,
              backend.device)

        # 逐件需求
        need = backend.decide_for(("text",))
        check("④⁴ 需求按**部件**算 ✓（TE 是 14.79 GiB ✓ 不是 DiT 那 13.26 ✓）",
              need.needGiB == 14.79, need.to_dict())
        multi = backend.decide_for(("dit", "text"))
        check("④⁵ 同阶段多件 ⇒ 需求**相加** ✓✗（分时驻留下每阶段一件，但表要能兜住多件 ✓）",
              multi.needGiB == round(13.26 + 14.79, 2), multi.to_dict())
    finally:
        dp.probe = real_probe  # type: ignore[assignment]

    described = tb.TorchBackend(device="cpu", policy="cpu").describe()
    policy = described.get("devicePolicy") or {}
    check("④⁶ `describe()` 报出策略实况 ✓✗（只听 `device` 分不清「钉在 CPU」与「被转移」✓）",
          {"policy", "requestedDevice", "lastDecision", "pressure", "partNeedGiB"} <= set(policy),
          policy)
    import torch  # noqa: PLC0415
    check("④⁷ **本机有没有卡**要看 `gpuAvailable` ✓✗（不是 `cudaAvailable` ✓ —— 后者是「这台后端现在在哪」✓）",
          described.get("gpuAvailable") == bool(torch.cuda.is_available()),
          described.get("gpuAvailable"))

    covered = set().union(*tb.TorchBackend.PHASE_PARTS.values())
    check("④⁸ 每个分时部件都有驻留账 ✓✗（缺一件 ⇒ 决策会把「不知道」当「不要空间」✓）",
          covered <= set(tb.TorchBackend.PART_NEED_GIB), (covered, tb.TorchBackend.PART_NEED_GIB))
    check("④⁹ 驻留账与实测口径一致 ✓（DiT 13.26 / TE 14.79 / VAE 0.63 ⇒ 和 28.68 ✓）",
          abs(sum(tb.TorchBackend.PART_NEED_GIB.values()) - 28.68) < 0.01,
          tb.TorchBackend.PART_NEED_GIB)

    #: ⚠️ **顺序**也要钉 ✓✗：`enter_phase` 必须是「退 → 测 → 定 → 装」✓。
    #: 换成「先装后退」时，测出来的余量是**假的紧** ✓✗（卡上还压着上一件 ✓）⇒ 会把自己越推越往
    #: CPU，而且一旦降级就回不来 ✓✗ —— 这个错**不会报任何异常** ✓✗，只能靠源码顺序钉住 ✓。
    src = inspect.getsource(tb.TorchBackend.enter_phase)
    order = [src.find("release_part"), src.find("decide_for"),
             src.find("_apply_decision"), src.find("load_weights")]
    check("④¹⁰ `enter_phase` 的**退 → 测 → 定 → 装**顺序 ✓✗（先装后退 ⇒ 余量是假的紧 ⇒ 降级回不来 ✓）",
          all(pos >= 0 for pos in order) and order == sorted(order), order)


# ---------------------------------------------------------------- ⑤ 装载键


def case_load_key() -> None:
    from app.services.engine import runtime as rt  # noqa: PLC0415

    check("⑤ `None` / `auto` / `gpu-first` 归一到**同一个键** ✓✗（不归一 ⇒ 白装三遍 ✓）",
          rt._load_key(policy=dp.normalize_policy(None))
          == rt._load_key(policy=dp.normalize_policy("gpu-first")),
          (rt._load_key(policy=dp.normalize_policy(None)),))
    check("⑤′ `cpu` 与 `auto` 是**两个键** ✓（两套设备布局 ✓，复用就错 ✓）",
          rt._load_key(policy="cpu") != rt._load_key(policy="auto"))


def case_sdxl() -> None:
    """⑥ SDXL 出图线 ✓（`runtime._image_device_decision` ✓）—— 三个注入位都验 ✓。

    ⚠️ 这条线**没有分层驻留** ✓✗（四件套一次性 `to(device)` ✓）：`device_policy.decide`
    判出来的 `split` 在这里落不了地 ✓✗ ⇒ 必须**落成 CPU** ✓ 而不是"嘴上联合、实际硬塞" ✓。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415

    check("⑥ 显式 `device=\"cuda\"` ⇒ **不决策** ✓✗（钉死设备是调用方的决定 ✓，不许擅自改 ✓）",
          rt._image_device_decision(dit_path=None, device="cuda") is None)

    got = rt._image_device_decision(dit_path=None, device=None, policy="auto",
                                    need_bytes=int(6.73 * dp.GIB),
                                    pressure=_p(22.49, 21.0), gpu_available=True)
    check("⑥′ 卡上余量够 ⇒ 上卡、不降级 ✓（需求取盘上真字节 ✓ ⇒ 6.73 GiB ＋ 1 GiB 余量 ✓）",
          got is not None and got["device"] == "cuda" and got["degraded"] is False,
          got)

    #: 余 3 GiB、需 6.73 GiB ⇒ `decide` 会给 `cuda+split` ✓；这条线落不了地 ⇒ 必须转 CPU ✓
    got = rt._image_device_decision(dit_path=None, device=None, policy="auto",
                                    need_bytes=int(6.73 * dp.GIB),
                                    pressure=_p(22.49, 3.0), gpu_available=True)
    check("⑥″ 装不下但够联合 ⇒ **这条线转 CPU** ✓✗（没有分层驻留 ⇒ 硬塞必 OOM ✓）",
          got is not None and got["device"] == "cpu" and got["degraded"] is True, got)
    check("⑥‴ 转 CPU 的理由**写清了为什么** ✓✗（不能只说\"降级\" ✓ —— 要能看出是这条线不支持联合 ✓）",
          "分层驻留" in str(got.get("reason") or ""), got.get("reason"))

    got = rt._image_device_decision(dit_path=None, device=None, policy="cpu",
                                    need_bytes=int(6.73 * dp.GIB),
                                    pressure=_p(22.49, 21.0), gpu_available=True)
    check("⑥⁴ `policy=\"cpu\"` ⇒ 全在内存且**不算降级** ✓（这是你选的 ✓，不是被挤下去的 ✓）",
          got is not None and got["device"] == "cpu" and got["degraded"] is False, got)

    got = rt._image_device_decision(dit_path=None, device=None, policy="auto",
                                    need_bytes=0, pressure=None, gpu_available=True)
    check("⑥⁵ 读不到卡况（`pressure=None`）⇒ **保持优先 GPU** ✓✗（不因为\"看不见\"就降级 ✓）",
          got is not None and got["device"] == "cuda", got)

    #: ⚠️ ``split`` 有**两个来源** ✓✗（见 `_chain_device_decision` ✓）：
    #: 「装不下」⇒ 该退 CPU ✓；「`balanced` 装得下但想省余量」⇒ 退 CPU 是**错**的 ✓✗
    #: （为了省那点余量把速度砍一个数量级 ✓）。这条钉住后者 ✓。
    got = rt._image_device_decision(dit_path=None, device=None, policy="balanced",
                                    need_bytes=int(6.73 * dp.GIB),
                                    pressure=_p(100.0, 19.0), gpu_available=True)
    check("⑥⁶ `balanced` 装得下、只是没摊开 ⇒ **照常上卡且不算降级** ✓✗"
          "（0.81 已用率会判联合 ✓，但这条线落不了地 ✓ —— 不能顺手推去 CPU ✓）",
          got is not None and got["device"] == "cuda" and got["degraded"] is False
          and "整份上卡" in str(got.get("reason") or ""), got)


def case_zimage() -> None:
    """⑦ Z-Image 整链线 ✓（`runtime._zimage_device_decision` ✓）—— 按**整链同驻**账判 ✓。

    ⚠️ 这条线最容易犯的错 ✗✗：拿「最重的那一件」（DiT 11.46 ✓）去判 ⇒ 以为卡上剩 12 GiB 就够 ✓✗
    ⇒ DiT 装进去了、装 TE 时才 OOM ✓，报错落在 TE 上、看着像"文本塔坏了" ✓✗（实测踩过 ✓）。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    chain = tb.TorchBackend.CHAIN_NEED_GIB
    check("⑦ 整链账是**三件相加**的实测值 ✓（11.46＋7.49＋0.33 ＝ 19.28 ✓）",
          abs(float(chain["zimage"]) - 19.28) < 0.01, chain)
    check("⑦′ 整链账**大于**分时账里最重的一件 ✓✗（拿 DiT 那 13.26 去判会漏掉 6 GiB ✓）",
          float(chain["zimage"]) > float(tb.TorchBackend.PART_NEED_GIB["dit"]) + 5.0,
          (chain, tb.TorchBackend.PART_NEED_GIB))

    got = rt._zimage_device_decision(device=None, policy="auto",
                                     pressure=_p(22.49, 21.0), gpu_available=True)
    check("⑦″ 卡上空 ⇒ 整链上卡 ✓（余 21 ≥ 19.28＋1 ✓）",
          got is not None and got["device"] == "cuda" and got["degraded"] is False, got)

    got = rt._zimage_device_decision(device=None, policy="auto",
                                     pressure=_p(22.49, 15.0), gpu_available=True)
    check("⑦‴ 余 15 GiB（DiT 装得下、整链装不下）⇒ **整条链转 CPU** ✓✗"
          "（按 DiT 那件判会一路装到 TE 才 OOM ✓）",
          got is not None and got["device"] == "cpu" and got["degraded"] is True, got)
    check("⑦⁴ 转 CPU 的理由写明是**三件同时在卡上** ✓✗（不能只写\"降级\" ✓）",
          "同时在卡上" in str(got.get("reason") or ""), got.get("reason"))

    got = rt._zimage_device_decision(device=None, policy="gpu",
                                     pressure=_p(22.49, 3.0), gpu_available=True)
    check("⑦⁵ `policy=\"gpu\"` ⇒ **不动** ✓（钉住上卡是调用方的决定 ✓，装不下就 OOM ✓✗）",
          got is not None and got["device"] == "cuda" and got["degraded"] is False, got)

    got = rt._zimage_device_decision(device="cpu", policy="auto",
                                     pressure=_p(22.49, 21.0), gpu_available=True)
    check("⑦⁶ 显式 `device=\"cpu\"` ⇒ **不决策** ✓（与 SDXL / `TorchBackend` 同一口径 ✓）",
          got is None, got)


def case_residency_te() -> None:
    """⑧ 联合驻留能不能**切得动文本塔** ✓（`residency.survey` 的穿透规则 ✓）。

    ⚠️ 为什么这条必须存在 ✗✗：`residency` 原先只认**直接子模块**名 ✓，而两条文本塔都是
    HF 式写法 —— ``self.model = nn.Module()`` ✓、真正的层表在 ``model.layers`` ✓
    （`qwen35_te.Qwen35TeBackbone` / `qwen3_te` 同理 ✓）。不穿透的话，`survey` 看到的是
    **一个**名叫 ``model`` 的子模块 ✓ ⇒ 整份 18.5 GiB 被当成「非层参数」✗ ⇒
    「非层参数先上卡」那一步就超预算 ⇒ ``fits=False``、整份退回 CPU ✓✗
    —— 联合驻留**静默变成纯 CPU** ✓✗，正是 `_LAYER_RES` 上面那 50–53 行记的同一个病
    （那次是 ``blocks`` 认不出 ✓，测试模型用 ``layers`` ⇒ 全绿也发现不了 ✓✗）。
    """
    import torch  # noqa: PLC0415 —— 真模块树要靠它造 ✓（本仓 torch 是硬依赖 ✓）
    from torch import nn  # noqa: PLC0415

    from app.services.engine import residency as res  # noqa: PLC0415

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

    class _Backbone(nn.Module):
        """HF 式三层包装 ✓：``model`` 只是外壳、自己名下**没有**参数 ✓。"""

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(4, 4)
            self.model.layers = nn.ModuleList([_Block() for _ in range(6)])
            self.model.norm = nn.LayerNorm(4)

    backbone = _Backbone()
    groups = res.survey(backbone)
    names = [group.name for group in groups]
    check("⑧ 穿透 `self.model = nn.Module()` ⇒ **六层全认出来** ✓✗"
          "（不穿透 ⇒ 整份当「非层参数」⇒ 联合静默退化成纯 CPU）",
          sum(group.layers for group in groups) == 6, names)
    check("⑧′ 嵌入表与最终 norm 仍是**非层**组 ✓（它们每步都用、不该被切到 CPU ✓）",
          "model.embed_tokens" in names and "model.norm" in names
          and not res.is_layer("model.embed_tokens")
          and not res.is_layer("model.norm"), names)
    check("⑧″ `is_layer` 认**带前缀**的层名 ✓（`model.layers.3` ✓），"
          "但**不**把 `model.norm` / `final_modulation.1` 认成层 ✓✗",
          res.is_layer("model.layers.3") and res.is_layer("blocks.7")
          and not res.is_layer("model.norm") and not res.is_layer("final_modulation.1"),
          [res.is_layer(n) for n in ("model.layers.3", "blocks.7", "model.norm",
                                     "final_modulation.1")])

    total = res._bytes_of(backbone)  # noqa: SLF001 —— 自检直接量 ✓
    chosen = res.plan(groups, budget=total // 2)
    check("⑧‴ 预算一半 ⇒ 卡上**连续前缀层** + CPU 其余 ✓（跳号会让每步过界好几次 ✓✗）",
          chosen.layers_on_gpu > 0 and chosen.layers_on_cpu > 0 and chosen.fits,
          chosen.reason)

    #: ⚠️ 穿透有**硬条件** ✗✗：自己名下有直接参数的模块**不许**穿透（它本身就是个真模块 ✓）。
    class _NotAContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 2))
            self.layers = nn.ModuleList([_Block() for _ in range(3)])

    strict = res.survey(_NotAContainer())
    #: ⚠️ 这条的正确口径是「**顶层列表照常展开** ✓，同时**不**因为里面有列表就把整个模块
    #: 再穿一层」✗ —— 第一版写成「一条 `layers.N` 都不许有」✗，那是把**既有正确行为**
    #: 当缺陷了 ✓✗（列表就在根上，本来该展开 ✓）；真正的判据是「层名不多带一层包名」✓。
    check("⑧⁴ 顶层 `ModuleList` 照常展开 ✓，且有直接参数的模块**不**再多穿一层 ✓✗"
          "（多穿＝把它自己的权重混进层里 ⇒ 切错）",
          [group.name for group in strict if group.layers]
          == ["layers.0", "layers.1", "layers.2"]
          and any(group.name == "‹root›" for group in strict),
          [group.name for group in strict])


def case_split_forward() -> None:
    """⑨ 联合驻留**真的能前向** ✓（小同构塔 + 真卡 + 显式小预算 ✓）。

    ⚠️ 为什么这条非跑一次不可 ✗✗：`survey` / `plan` / `report` 全是**账** ✓ ——
    账对不代表跑得起来 ✓✗。`residency` 的 hook 要在每次跨界时把**激活**搬到另一台设备 ✓，
    这条路上的真坑是「某一层没被登记进组、于是它还留在 CPU 而激活已经在卡上」✓、
    或「norm 的 weight 在 CPU、输入在卡上」✓ —— 两者都**不报错**，只是结果错 / 直接炸 ✓✗
    （`_TinyTe` 每层都混一个 `LayerNorm`，正是为了踩这个 ✓）。

    小模型几 KB ⇒ 不占卡 ✓、秒级 ✓；显式给小预算 ⇒ 强制切出「卡上前几层 / CPU 其余」✓
    （不注入预算就得占掉半张卡才走得到这一档 ✓✗）。
    """
    import copy  # noqa: PLC0415
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001 —— 没装 torch 的机器上这条不适用 ✓
        _SKIPS.append("⑨ 联合驻留真前向（本机没装 torch ✓）")
        return
    if not torch.cuda.is_available():
        _SKIPS.append("⑨ 联合驻留真前向（本机没有可用 CUDA ✓）")
        return
    from torch import nn  # noqa: PLC0415

    class _Blk(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.proj = nn.Linear(hidden, hidden)

        def forward(self, x: Any) -> Any:
            return x + self.proj(torch.tanh(self.norm(x)))

    class _TinyTe(nn.Module):
        """与两条文本塔**同构** ✓：``self.model = nn.Module()`` 外壳 ＋ 层表 ＋ 收尾 norm ✓。"""

        def __init__(self, hidden: int = 64, depth: int = 8) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed = nn.Embedding(32, hidden)
            self.model.layers = nn.ModuleList([_Blk(hidden) for _ in range(depth)])
            self.model.norm = nn.LayerNorm(hidden)

        def forward(self, ids: Any) -> Any:
            hidden = self.model.embed(ids)
            for block in self.model.layers:
                hidden = block(hidden)
            return self.model.norm(hidden)

    torch.manual_seed(7)
    model = _TinyTe()
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 32, (2, 5))  # noqa: S311 —— 只要确定性输入 ✓
    with torch.no_grad():
        want = reference(ids)

    backend = tb.TorchBackend(device="cuda")
    #: 预算 ≈ 非层参数（embed + norm，约 9 KB）＋ 三层（每层 16 KB + norm 256 B）✓
    #: ⇒ 卡上放前几层、其余留内存 ✓（`0.06` MiB ＝ 62915 B ✓）
    backend._residency = "split"                       # noqa: SLF001
    backend._residencyBudgetMib = 0.06                 # noqa: SLF001
    report = backend._place_with_residency(model, what="tiny-te")  # noqa: SLF001
    check("⑨ 小预算 ⇒ **真切成**「卡上几层 ＋ CPU 几层」✓✗"
          "（`fits=False` ⇒ 静默退化成纯 CPU：装载成功、比不切还慢 ✓✗）",
          report is not None and report.get("applied") is True
          and report.get("fits") is True
          and int(report.get("layersOnGpu") or 0) > 0
          and int(report.get("layersOnCpu") or 0) > 0, report)

    with torch.no_grad():
        got = model(ids)
    #: ⚠️ `got` 可能在**卡上** ✓（收尾 norm 属「非层参数」⇒ 先上卡 ✓）—— 这不是缺陷 ✗，
    #: 比之前搬回 CPU 即可 ✓（第一版直接在 `cosine_similarity` 里混了 cuda 与 cpu ✓，
    #: 报「Expected all tensors to be on the same device」✓，差点被当成联合驻留的 bug ✓✗）。
    cos = float(torch.nn.functional.cosine_similarity(
        got.detach().flatten().float().cpu(), want.flatten().float(), dim=0))
    check("⑨′ 跨界前向**结果一致** ✓✗（cos<1 ⇒ 有层没被搬、或 norm 的 weight 留在原处 ✓）",
          tuple(got.shape) == tuple(want.shape) and cos > 0.9999, {"cos": round(cos, 6)})
    devices = {str(p.device) for p in model.parameters()}
    check("⑨″ 切完之后参数**真的落在两处** ✓✗（全在卡上 ⇒ 那 0.06 MiB 预算根本没生效 ✓；"
          "报告说切了、参数没动 ⇒ 报告在骗人 ✓✗）",
          any(d.startswith("cpu") for d in devices)
          and any(d.startswith("cuda") for d in devices), sorted(devices))

    # ⑨‴ **释放一件 ⇒ 卡上那件真退** ✓✗（2026-11-01 真权重端到端抓到的真缺陷 ✓）。
    #    `residency.Handle` 攥着 `module` ✓、它的 hook 闭包又攥着 handle 与 node ✓ ⇒ 成环 ✓；
    #    `self._residencyHandle` 是**外部活引用** ✓ ⇒ `gc.collect()` 收不掉 ✓✗。
    #    症状：`release_part("text")` 报告 `released=True` ✓、函数返回正常 ✓，
    #    可卡上那 14.8 GiB **一字节没退** ✓✗（`torch.cuda.empty_cache()` 只还**空闲**块 ✓）
    #    ⇒ 紧接着装 DiT 时预算≈0 ⇒ 整份 17.94 GiB 落到内存 ⇒ 宿主 71 GiB / 卡 22.5 GiB 卡死 ✓。
    import gc as _gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    backend._textEncoder = model                               # noqa: SLF001
    alive = weakref.ref(model)
    before = int(torch.cuda.memory_allocated())
    del model
    backend.release_part("text")
    _gc.collect()
    torch.cuda.empty_cache()
    after = int(torch.cuda.memory_allocated())
    check("⑨‴ `release_part` 之后那件**真的被回收** ✓✗：把手不 `detach()` ⇒ hook 环还在 ⇒ "
          "模块活着、卡上权重退 0 字节 ✓（报告照样说 released=True ✓ 这就是静默的那种错 ✓）",
          alive() is None and after < before,
          {"alive": alive() is not None, "beforeMiB": round(before / 2**20, 3),
           "afterMiB": round(after / 2**20, 3)})


def case_hand_copies() -> None:
    """⑩ 「哪里还在手抄」✓：同一件事只许有**一份**实现 ✓✗（2026-10-31 ✓）。

    ⚠️ 为什么值得单开一段 ✗✗：本仓最贵的 bug 不是写错的逻辑 ✗，而是**同一份判断抄了几遍** ✓✗
    —— 改的时候只改到一处 ✓、另几处照旧 ✓，而它们在"最省事的那一档输入"上**全对** ✓✗
    ⇒ 测试全绿、线上错 ✓（这轮就抓到一个：判「在不在卡上」有三种写法，
    ``== "cuda"`` 那种在 ``cuda:0`` 上答 False ✓✗）。

    ⚠️ 这两条判据盯的是**字面量**✓✗（不是文风 ✓）：万一将来有人把 `is_cuda_device` 绕开、
    直接再写一次字符串比较，这里当场红 ✓。
    """
    from app.services.engine import device_policy as pol  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415
        device_arg = torch.device("cuda", 0)
    except Exception:  # noqa: BLE001 —— 没装 torch 的机器上跳过这一格 ✓
        device_arg = None
        _SKIPS.append("⑩ `is_cuda_device` 认 `torch.device` 那一格（本机没装 torch ✓）")

    check("⑩ 判「在不在卡上」**只认一种写法** ✓✗：`cuda` / `cuda:0` / `CUDA:1` / "
          "`torch.device(\"cuda\",0)` 全 True ✓；`None` / `cpu` / `cudafoo` 全 False ✓"
          "（`startswith(\"cuda\")` 会把 `cudafoo` 判成卡 ✓✗）",
          pol.is_cuda_device("cuda") and pol.is_cuda_device("cuda:0")
          and pol.is_cuda_device("CUDA:1") and pol.is_cuda_device("  cuda:3 ")
          and (device_arg is None or pol.is_cuda_device(device_arg))
          and not pol.is_cuda_device(None) and not pol.is_cuda_device("cpu")
          and not pol.is_cuda_device("cudafoo"),
          [pol.is_cuda_device(x) for x in ("cuda", "cuda:0", "CUDA:1", None, "cpu", "cudafoo")])

    check("⑩′ `is_cpu_device` 与 `is_cuda_device` **不是取反** ✓✗：`None` / `mps` 两边都 False"
          "（拿 `not is_cuda_device` 顶替 ⇒ 会把「不知道」当 CPU，正好是最省事的那一档 ✓✗）",
          pol.is_cpu_device("cpu") and pol.is_cpu_device("cpu:0") and pol.is_cpu_device("  CPU ")
          and not pol.is_cpu_device(None) and not pol.is_cpu_device("mps")
          and not pol.is_cpu_device("cuda") and not pol.is_cpu_device("cuda:0"),
          [pol.is_cpu_device(x) for x in ("cpu", "cpu:0", "  CPU ", None, "mps", "cuda", "cuda:0")])

    #: ⚠️ 判 CPU 的那一处也收进来的理由 ✗✗：H3 的 int8 分档写着
    #: ``str(self._device).split(":")[0] != "cpu"`` ✓ —— 它**是**同一件事的第四种写法 ✓✗
    #: （改设备口径时漏掉它 ⇒ int8 走错分支：全份塞卡上 OOM、或反过来留在内存 ✓✗）。
    #: 上一版曾以"语义是『不是 CPU』、含 mps"为由放行 ✓，这次改成由 `is_cpu_device` 表达 ✓ ——
    #: 语义不变（``not is_cpu_device("mps")`` 仍为 True ✓），但判据只剩一份 ✓。
    from app.services.engine import sdxl_backend as sdxl  # noqa: PLC0415

    #: ⚠️ 只抓**把设备当字符串比**的那三种形态 ✓✗；**放行** ``device.type == "cuda"`` ✓ ——
    #: 那是 ``torch.device`` 对象自己的字段 ✓，是全仓最正宗的写法 ✓（第一版判据拿
    #: ``'== "cuda"' in line`` 去扫，把这两处也抓了 ⇒ 判据本身报假错 ✓✗：
    #: 「不许手抄」的对象是**字符串字面量比较**，不是 torch 的类型判断 ✓）。
    bad_patterns = (
        re.compile(r'_device\s*==\s*"cuda"'),
        re.compile(r'startswith\(\s*"cuda"\s*\)'),
        re.compile(r'split\(\s*":"\s*\)\[0\]\s*!=\s*"cuda"'),
        re.compile(r'_device\s*==\s*"cpu"'),
        re.compile(r'split\(\s*":"\s*\)\[0\]\s*!=\s*"cpu"'),
    )
    offenders: dict[str, list[str]] = {}
    for name, module in (("torch_backend", tb), ("sdxl_backend", sdxl)):
        hits = [line.strip() for line in inspect.getsource(module).splitlines()
                if not line.strip().startswith("#")
                and any(pattern.search(line) for pattern in bad_patterns)]
        if hits:
            offenders[name] = hits
    #: ⚠️ **判据自检** ✗✗：⑩′ 是"源码里不许出现 X"型断言 ✓ —— 这种断言最容易变成
    #: 「正则写歪了 ⇒ 永远绿 ⇒ 摆设」✓✗（本仓有先例：`if args.X:` 恒真、口径写着却一字不生效 ✓）。
    #: 所以这里拿**当年真实出现过的三行**喂给它 ✓，必须在红；再喂一行**合法**的必须在绿 ✓。
    should_hit = (
        '"cudaAvailable": self._device == "cuda",',
        'if not str(self._device).startswith("cuda"):',
        'if str(self._device).split(":")[0] != "cuda":',
        'elif self._int8Offload and str(self._device).split(":")[0] != "cpu":',
        'return self._device == "cpu"',
    )
    should_miss = (
        'return torch.bfloat16 if device.type == "cuda" else torch.float32',
        'if device.type == "cuda" and device.index is None:',
        'if not device_policy.is_cpu_device(self._device):',
        'if tensor.device.type == "cpu":',
    )
    check("⑩″′ ⑩′ 的正则**真抓得住**那五种手抄（含判 CPU 的两行 ✓✗），且不误伤 "
          "`device.type == \"cuda\"` / 已收口的 `is_cpu_device(...)` ✓"
          "（不验这一条 ⇒ ⑩′ 可能只是个永绿的摆设 ✓✗）",
          all(any(p.search(s) for p in bad_patterns) for s in should_hit)
          and not any(p.search(s) for p in bad_patterns for s in should_miss),
          {"该抓没抓": [s for s in should_hit
                        if not any(p.search(s) for p in bad_patterns)],
           "不该抓却抓": [s for s in should_miss
                          if any(p.search(s) for p in bad_patterns)]})

    check("⑩′ 两个后端里**不许**再手抄「是不是卡」✗✗（只许调 `device_policy.is_cuda_device` ✓）："
          "`torch_backend` 曾写 `startswith` 与 `split(\":\")[0]`、`describe().cudaAvailable` 写 "
          "`== \"cuda\"` ⇒ 后者在 `cuda:0` 上答 False（后端在卡上、能力表说不在）✓✗",
          not offenders, offenders)

    #: ⭐ `declare_chain().peakGiB` 必须**由常量算** ✓✗（原来写死 `14.79` ✓）：
    #: 写死的话，某件权重变大、`PART_NEED_GIB` 跟着改了，这一格还报旧值 ⇒ 读者按它估余量估错 ✓✗。
    peak = tb.TorchBackend(device="cpu").declare_chain(
        dit_path="a", text_weights="b", vae_path="c")  # type: ignore[arg-type]
    check("⑩″ `declare_chain().peakGiB` **由 `PART_NEED_GIB` 算** ✓✗（写死 14.79 ⇒ 常量一改就撒谎）",
          float(peak.get("peakGiB") or 0) == round(max(tb.TorchBackend.PART_NEED_GIB.values()), 2),
          {"报的": peak.get("peakGiB"), "常量": tb.TorchBackend.PART_NEED_GIB})


def main() -> int:
    print(f"== 设备策略自检（{dp.POLICIES} ✓）==")
    case_decide()
    case_policy()
    case_probe()
    case_backend()
    case_load_key()
    case_sdxl()
    case_zimage()
    case_residency_te()
    case_split_forward()
    case_hand_copies()
    for note in _SKIPS:
        print(f"SKIP  {note}")
    print(f"\nSUMMARY: {len(_FAILS)} failed")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
