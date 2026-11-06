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

#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 第一条打印就崩，判据一条都跑不到 ✗
#: （``run_all`` 已重配 stdout，只有单独跑才露头 ⇒ 每个 ``__main__`` 测试都自带这一句 ✓）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def case_residency_kind() -> None:
    """⑪ 余量档的**件类型归一** ✓✗（2026-11-01 加 ✓）—— 这一层漏了，改动白做。

    ## 抓的是什么 ✗

    `_place_with_residency(what=...)` 的 ``what`` 是**报告名** ✓，H3 那两件写的是
    ``"h3"`` / ``"h3-te"`` ✓（grep `_place_int8_module` 可证 ✓）。余量表只认
    ``dit``/``te``/``vae`` ✓ ⇒ 少了归一这一步，**最需要省余量的那件**（25.44 GiB 的 H3 文本塔 ✓）
    查表查不到、退回 25% ✓✗ —— 而报告里 ``reserve`` 那一格还会显示 0.25，看起来"生效了" ✓✗。
    """
    from app.services.engine import residency as r  # noqa: PLC0415

    got = {name: tb._residency_kind(name)
           for name in ("dit", "h3", "te", "h3-te", "h3-audio-vae", "vae", "text", "text-te")}
    check("⑪ `what` ⇒ 件类型：DiT 两侧（dit / h3）都归 dit ✓",
          got["dit"] == "dit" and got["h3"] == "dit", got)
    check("⑪ `what` ⇒ 件类型：文本塔两侧（te / h3-te）都归 te ✓✗",
          got["te"] == "te" and got["h3-te"] == "te", got)
    check("⑪ `what` ⇒ 件类型：含 vae 的归 vae（h3-audio-vae ✓）", got["h3-audio-vae"] == "vae", got)
    check("⑪ `what` ⇒ 件类型：认不出**返回 None** ✓（不许瞎猜一个类型 ✗）",
          tb._residency_kind("unknown-part") is None
          and tb._residency_kind(None) is None
          and tb._residency_kind("") is None,
          [tb._residency_kind(n) for n in ("unknown-part", None, "")])
    #: ⭐ 顺序判据 ✗✗：``"h3-te"`` 同时含 ``h3`` ✓ —— 归一里 te 必须**先判** ✓，
    #: 否则文本塔被判成 DiT、余量从 5% 变 25% ✓✗（这条断言就是那次踩坑的锚 ✓）。
    check("⑪ 归一顺序：`h3-te` 不许被 `h3` 抢走（先判 te 再判 dit）",
          r.reserve_for(tb._residency_kind("h3-te")) < r.reserve_for(tb._residency_kind("h3")),
          {"h3-te": r.reserve_for(tb._residency_kind("h3-te")),
           "h3": r.reserve_for(tb._residency_kind("h3"))})
    check("⑪ 认不出件类型 ⇒ 余量退回 25%（保守侧）",
          r.reserve_for(tb._residency_kind("unknown-part")) == r.DEFAULT_RESERVE,
          r.reserve_for(tb._residency_kind("unknown-part")))


def case_reserve_report() -> None:
    """⑪′ **报告里要看得见**「按哪一档余量算的」与「GPU 承担了多少」✓✗（2026-11-01 加 ✓）。

    ## 为什么这两格值得占一条测试 ✗

    用户的问题原话是「为什么绝大部分计算由 CPU 承担」✓ —— 而改动前**报告里根本没有能回答它的数** ✓✗：
    ``budgetMib`` / ``layersOnGpu`` 都只说明"切成了什么样" ✓，不说明"按什么口径切的" ✓
    （同一个 15.53 GiB 预算，可能是 25% 余量算出来的、也可能是显式给的 ✓✗，读者分不出 ✓），
    也不说明"GPU 到底吃下几成" ✓（``layersOnGpu`` 是**层数**不是字节 ✓ —— 各层大小不等时它和
    字节占比不是一回事 ✓✗）。

    ⇒ 判据三格：``reserve`` 报对 ✓、``reserveKind`` 报对 ✓、``gpuShare`` 分母算对 ✓。
    这里刻意用**默认预算**（现算、远大于小塔 ⇒ 整份上卡 ✓）⇒ ``gpuShare`` 必须正好 1.0 ✓✗
    （写成 ``cpu/(gpu+cpu)`` 或漏了分母为 0 的分支，这里立刻现形 ✓）。
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        _SKIPS.append("⑪′ 报告里的余量/GPU 占比（本机没装 torch ✓）")
        return
    if not torch.cuda.is_available():
        _SKIPS.append("⑪′ 报告里的余量/GPU 占比（本机没有可用 CUDA ✓）")
        return
    from torch import nn  # noqa: PLC0415

    class _Tower(nn.Module):
        def __init__(self, hidden: int = 32, depth: int = 3) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, hidden)
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
            self.norm = nn.LayerNorm(hidden)

    torch.manual_seed(11)
    backend = tb.TorchBackend(device="cuda")
    backend._residency = "split"            # noqa: SLF001 —— 走联合那条路才会算余量 ✓
    backend._residencyBudgetMib = None      # noqa: SLF001 —— 刻意让预算**现算** ✓
    report = backend._place_with_residency(_Tower(), what="h3-te")  # noqa: SLF001
    check("⑪′ 报告写明**按哪一档余量**算的（`what=h3-te` ⇒ 5% ✓，不是默认 25%）",
          report is not None and report.get("reserve") == 0.05
          and report.get("reserveKind") == "te", report)
    check("⑪′ 报告写明 **GPU 承担的字节占比**（默认预算 ⇒ 整份上卡 ⇒ 正好 1.0 ✓✗）",
          report is not None and report.get("gpuShare") == 1.0
          and int(report.get("layersOnCpu") or 0) == 0, report)
    #: ⚠️ 反向：DiT 那件**不许**跟着降到 5% ✓✗（它的激活真能到十几 GiB ✓）
    backend2 = tb.TorchBackend(device="cuda")
    backend2._residency = "split"           # noqa: SLF001
    backend2._residencyBudgetMib = None     # noqa: SLF001
    report2 = backend2._place_with_residency(_Tower(), what="dit")  # noqa: SLF001
    check("⑪′ 反向：`what=dit` 仍是 25% 余量 ✓（小件不许把 DiT 的尺子借走）",
          report2 is not None and report2.get("reserve") == 0.25
          and report2.get("reserveKind") == "dit", report2)


def case_residency_total() -> None:
    """⑫ 跨件汇总成**一个数** ✓✗（`runtime.residency_total` ✓）。

    ## 为什么这条值得占一格 ✗

    创作者的原问题就是「显存那么大，为什么绝大部分计算由 CPU 承担」✓ —— 而报告里此前
    **没有一处能一眼回答** ✓✗：每件各报各的，而且**深度不统一**（DiT 在 `weights.residency`、
    TE/VAE 在 `components[x].residency`、`phased` 的件根本没有 ✓、`components["dit"]`
    常常只是路径字符串 ✓✗）⇒ 要人自己加五六个小数 ✓，加错也看不出来 ✓✗。

    ⇒ 判据四件事：**深度**（两处都要找到 ✓）、**去重**（同一份账挂两处不许翻倍 ✓✗）、
    **空账不编 0**（`None` ≠ `0` ✓✗）、**分母为 0 不给 `gpuShare`**（编 1.0 会被读成"全在卡上"✓✗）。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415

    check("⑫ 一件驻留账都没有（全是路径字符串 / `phased: true`）⇒ **`None`** ✓✗"
          "（编成 `0` 会把「这次没有联合驻留的账」读成「卡上什么都没」✓）",
          rt.residency_total({"components": {"dit": "img.safetensors",
                                             "textEncoder": {"phased": True, "loaded": False}}}) is None,
          "")

    #: 刻意照**实测那份**摆（2512 bf16：DiT 卡上 15.9 / 内存 22.15，VAE 0.24 在卡上 ✓）
    dit = {"gpuGiB": 15.9, "cpuGiB": 22.15}
    report = {"weights": {"path": "dit.safetensors", "residency": dit},
              "components": {"dit": "dit.safetensors",
                             "vae": {"kind": "qwen_image", "residency": {"gpuGiB": 0.24, "cpuGiB": 0.0}},
                             "textEncoder": {"phased": True}}}
    total = rt.residency_total(report)
    check("⑫′ **两处深度都找到** ✓（`weights.residency` ＋ `components[x].residency` ⇒ 16.14 / 22.15）",
          total is not None and total["gpuGiB"] == 16.14 and total["cpuGiB"] == 22.15
          and total["parts"] == 2, total)
    check("⑫″ **按字节**给占比 ✓（16.14 / 38.29 ⇒ 0.4215），且写明「不是速度占比」✓✗",
          total is not None and total["gpuShare"] == 0.4215
          and total.get("judgedOn") == "bytes" and "速度" in str(total.get("note")), total)

    same = {"gpuGiB": 15.9, "cpuGiB": 22.15}
    twice = rt.residency_total({"weights": {"residency": same},
                                "components": {"dit": {"residency": same}}})
    check("⑫‴ 同一份账挂在两处 ⇒ **只算一次** ✓✗（按 `id` 去重；不去重会凭空翻倍成 31.8 / 44.3）",
          twice is not None and twice["gpuGiB"] == 15.9 and twice["parts"] == 1, twice)

    empty = rt.residency_total({"components": {"x": {"residency": {"gpuGiB": 0.0, "cpuGiB": 0.0}}}})
    check("⑫⁴ 分母为 0 ⇒ **`gpuShare` 给 `None`** ✓✗（编 1.0 会被读成「整份在卡上」—— "
          "而那恰好是 `fits=False` 那一档的反面 ✓）",
          empty is not None and empty["gpuShare"] is None and empty["gpuGiB"] == 0.0, empty)

    #: ⚠️ 真跑才露头的那一格 ✗✗（2026-11-01 ✓）：Edit 2511 走 int8 常驻后 `residency="off"`
    #: ⇒ **一次 split 都没发生** ⇒ 报告里没有任何逐件驻留账 ⇒ 只看字节汇总会得 `None` ✓✗，
    #: 而 `None` 恰好落在最该被夸的那一次（整份上卡 ✓）上 ⇒ 必须退回**决策**读 ✓。
    off_report = {"devicePolicy": {"lastDecision": {"device": "cuda", "residency": "off"}}}
    cpu_report = {"devicePolicy": {"lastDecision": {"device": "cpu", "residency": "off"}}}
    off_total = rt.residency_total(off_report)
    cpu_total = rt.residency_total(cpu_report)
    check("⑫⁶ 一次 `split` 都没有 ⇒ 退回**决策**：`gpuShare=1.0`、`gpuGiB=None`（字节数**就是不知道** ✓✗"
          "—— 编一个数比空着更坏 ✓）、`judgedOn=\"residency\"` ✓",
          off_total is not None and off_total["gpuShare"] == 1.0
          and off_total["gpuGiB"] is None and off_total["judgedOn"] == "residency", off_total)
    check("⑫⁷ `device=cpu` ⇒ `gpuShare=0.0` ✓（**降级** ✓，note 里不许写成已优化 ✓✗）",
          cpu_total is not None and cpu_total["gpuShare"] == 0.0
          and "降级" in str(cpu_total.get("note")), cpu_total)
    check("⑫⁸ 连决策都没有 ⇒ 仍是 `None` ✓✗（`None` 只留给「这次什么都不知道」✓）",
          rt.residency_total({"components": {"dit": "x.safetensors"}}) is None, "")

    #: ⚠️ 接线 ✗✗：算得再对，不挂到**四条线共用的那一个人口** ⇒ 回执里永远没有这一格 ✓✗
    source = inspect.getsource(rt.EngineRuntime._commit_load)
    check("⑫⁵ 汇总挂在 `_commit_load`（四条装配线的**唯一**人口 ✓）—— 各线自己加必漏一条 ✓✗",
          "residency_total(report)" in source and 'report["residencyTotal"]' in source, "")


def case_no_park_in_auto() -> None:
    """⑬ `auto` **不选** `park` ✓✗（2026-11-01 裁定 ✓）。

    ## 判据不是"我这么想"，是两条可算的数 ✓

    * ``park`` 每步单向搬 ``plan.layer_bytes`` ＝ **全部层的权重** ✓（2512 那族 ≈38 GiB ✓、双向 76 ✓）；
    * ``split`` 每步搬 **一条跨界激活** ✓（`Plan.transfer_bytes("split")` 明写**返回 None** ✓
      —— 它不知道激活多大，且**不猜** ✓）。

    ⇒ 同样的模型同样的卡，自动选 park 只会更慢一个量级 ✓✗。它留在手工档 ✓，
    并且 `split` 必须继续标 ``degraded=True`` ✓（"卡装不下但还得用卡"的兜底 ✓，不是优化 ✓✗）。
    """
    from app.services.engine import device_policy as dp  # noqa: PLC0415
    from app.services.engine import residency as res  # noqa: PLC0415

    check("⑬ `AUTO_RESIDENCIES` 里**没有** `park` ✓✗，但 `park` 仍在 `residency.MODES` 里 ✓"
          "（手工档要留 ✓ —— 收成「删掉 park」是把工具也一起删了 ✓✗）",
          "park" not in dp.AUTO_RESIDENCIES and "park" in res.MODES
          and set(dp.AUTO_RESIDENCIES) <= set(res.MODES), (dp.AUTO_RESIDENCIES, res.MODES))

    total = int(22.49 * dp.GIB)
    seen: set[str] = set()
    bad: list[str] = []
    for free_gib, need_gib in ((20.0, 13.26), (6.0, 38.05), (0.2, 38.05),
                               (20.0, 0.0), (0.5, 0.0), (0.9, 13.26)):
        for name in ("auto", "balanced"):
            decided = dp.decide(policy=name, need_bytes=int(need_gib * dp.GIB),
                                pressure=dp.Pressure(totalBytes=total,
                                                     freeBytes=int(free_gib * dp.GIB)))
            seen.add(decided.residency)
            if decided.residency not in dp.AUTO_RESIDENCIES:
                bad.append(f"{name}@free{free_gib}/need{need_gib}⇒{decided.residency}")
    check("⑬′ 六种压力 × 两档策略 ⇒ 只出现 `off`/`split` ✓✗（`park` 一次都不许冒头 ✓）",
          not bad and seen <= set(dp.AUTO_RESIDENCIES), {"seen": sorted(seen), "bad": bad})

    split = dp.decide(policy="auto", need_bytes=int(38.05 * dp.GIB),
                      pressure=dp.Pressure(totalBytes=total, freeBytes=int(6.0 * dp.GIB)))
    check("⑬″ 联合是**兜底不是优化** ✓✗：`split` 必须 `degraded=True`（写成 `False` ⇒ "
          "报告会读成「已经安排好了」✓✗，而它其实每步都在跨界搬 ✓）",
          split.residency == "split" and split.degraded, split.to_dict())


def case_need_by_file() -> None:
    """⑭ 需求数**跟着文件走** ✓✗（2026-11-02 ✓）：同一族的 int8 / int4 / bf16 三档分别算 ✓。

    ## 抓的那个真缺陷

    Krea2 那条线的驻留账 ``{dit 12.57, text 7.5, vae 0.24}`` 是**int8 档**量的 ✓
    （``krea2_turbo_int8_convrot.safetensors`` ✓、224 层 ``Int8Linear`` 常驻 ✓）。同族还有
    **int4 档**（5.98 GiB ✓）—— 它**没有** int4 常驻壳 ⇒ 装起来要反量化成 bf16 ＝
    **23.89 GiB** ✓✗。拿 12.57 判「装得下」⇒ `auto` 判 `off` ⇒ 整份上卡 ⇒ 采样第 1 步 OOM ✓，
    而报错落在 ``alloc`` 上 ✓（读起来像"显存不够"，其实**需求数用错了档** ✗）。

    ## 反向的那一半

    表值也**不能**被精算覆盖成小值 ✗：Qwen-Image 2512 的 ``38.05`` 就是 bf16 那份的真驻留 ✓
    ⇒ 把 ``residency_total`` 的"看谁小用谁"搬到这里 ⇒ int8 那份的 19.09 会被 38.05 顶掉 ✓✗
    （反过来也会：拿 19.09 判 2512 ⇒ 装到 20 GiB 才 OOM ✓）。所以判据是**档位对不对** ✓，
    不是**谁大谁小** ✓✗。
    """
    from app.services.engine import safetensors as st  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    class _Entry:
        """假头部里的一格 ✓ —— ``nbytes`` 必须**真算**（精算是逐张量乘的 ✓，返回 0 就测不出差 ✓）。"""

        def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
            self.dtype, self.shape = dtype, shape

        @property
        def element_count(self) -> int:
            total = 1
            for dim in self.shape:
                total *= dim
            return total

        @property
        def nbytes(self) -> int:
            return self.element_count * tb._DTYPE_BYTES.get(self.dtype.upper(), 1)  # noqa: SLF001

    class _Info:
        def __init__(self, tensors: dict[str, Any]) -> None:
            self.tensors, self.ok, self.problems = tensors, True, []
            self.total_tensor_bytes = sum(item.nbytes for item in tensors.values())
            self.file_bytes = self.total_tensor_bytes + 2048
            self.metadata: dict[str, Any] = {}

    rows = (1_000_000, 1000)            #: 1e9 参 ⇒ int8 档占 0.93 GiB、反量化后 1.86 GiB ✓
    int8_info = _Info({"b.0.w.weight": _Entry("I8", rows),
                       "b.0.w.weight_scale": _Entry("F32", (1_000_000, 1))})
    plain_info = _Info({"b.0.w.weight": _Entry("BF16", rows)})
    int4_info = _Info({"b.0.w.weight": _Entry("I8", (1_000_000, 500)),   #: 打包列＝逻辑列÷2 ✓
                       "b.0.w.weight_scale": _Entry("F32", (1_000_000, 1))})
    int4_specs = {"b.0.w": {"format": "convrot_w4a4", "convrot_groupsize": 256}}

    real_inspect, real_specs = st.inspect, tb.loader.quant_specs_of

    def _fake(info: Any, specs: dict[str, Any]) -> None:
        st.inspect = lambda path: info                                    # noqa: ARG005
        tb.loader.quant_specs_of = lambda path, info=None: dict(specs)    # noqa: ARG005

    try:
        _fake(plain_info, {})
        gib_match, why_match = tb.part_need_gib(
            "x.safetensors", fallback=7.5, adapter="int8", assume="plain")
        check("⑭ 文件档 == 表值的出处档（bf16 文件 ＋ `assume=\"plain\"`）⇒ **用表值** ✓✗"
              "（表值是**实测驻留**、含骨架自建参数 ✓ ⇒ 比只算数据区更准 ✓）",
              abs(gib_match - 7.5) < 1e-9 and why_match.get("source") == "table", why_match)

        _fake(int8_info, {})
        gib_shell, why_shell = tb.part_need_gib(
            "x.safetensors", fallback=7.5, adapter="krea2", assume="plain")
        gib_noshell, why_noshell = tb.part_need_gib(
            "x.safetensors", fallback=7.5, adapter=None, assume="plain")
        check("⑭′ 档不匹配 ⇒ **按文件精算** ✓✗：int8 层**有常驻壳**（`adapter=\"krea2\"` ✓）"
              "⇒ 数据区 0.93 GiB ✓；**没有壳**（`adapter=None` ✓）⇒ 反量化成 bf16 ＝ "
              "**1.86 GiB**（1 B/参 → 2 B/参 ✓，按字面换算、不是拍的倍率 ✓）",
              abs(gib_shell - 0.93) < 0.02 and abs(gib_noshell - 1.86) < 0.03
              and why_shell.get("source") == "file"
              and why_noshell.get("source") == "file", (why_shell, why_noshell))

        _fake(int4_info, int4_specs)
        #: ⭐ 4bit 档的**两个方向都要钉住**（2026-11-02 补壳后 ✓）：
        #:   有壳（`adapter="krea2"` ✓）⇒ **打包字节** 0.47 GiB ✓；
        #:   无壳（`adapter=None` ✓）⇒ 反量化成 bf16 ＝ 1.86 GiB ✓（逻辑参 × 2 B ✓）。
        #: ⚠️ 无壳那一支**不能**拿打包形状乘（那样只有一半 ✓✗）—— 就是 `Krea2 int4` 真身
        #: 23.89 GiB 与「表里 12.57」之间那 2 倍差距的同一条算错 ✓。
        gib_int4, why_int4 = tb.part_need_gib(
            "x.safetensors", fallback=12.57, adapter="krea2", assume="int8")
        gib_int4_bare, why_int4_bare = tb.part_need_gib(
            "x.safetensors", fallback=12.57, adapter=None, assume="int8")
        check("⑭″ **int4 档**：有 4bit 常驻壳 ⇒ **打包字节 0.47 GiB** ✓；没有壳 ⇒ 反量化 bf16 "
              "＝ **1.86 GiB** ✓（逻辑参 × 2 B ✓，按字面换算 ✓）—— 两条都必须是 `source=file` ✓",
              why_int4.get("mode") == "int4" and abs(gib_int4 - 0.47) < 0.02
              and abs(gib_int4_bare - 1.86) < 0.03
              and why_int4.get("int4ShellLayers") == 1
              and why_int4_bare.get("int4ShellLayers") == 0,
              (why_int4, why_int4_bare))

        plan_raw = tb.int8_scale_plan(int4_info, int4_stems=quant_int4_stems(int4_specs))
        plan_bad = tb.int8_scale_plan(int4_info)
        check("⑭‴ 4bit 打包层**不进** int8 常驻计划 ✓✗（它存成 `I8` ＋ 带 `weight_scale`，"
              "与 per-row int8 **逐字同形** ✓）—— 传了 `int4_stems` ⇒ 计划空 ✓；不传 ⇒ "
              "**会被当成 int8 换壳**（形状全对、数值全错 ✓✗），这条留作反证 ✓",
              plan_raw == {} and plan_bad != {}, {"排除": plan_raw, "不排除": plan_bad})

        def _boom(path: Any) -> Any:
            raise OSError("模拟读不出头部")

        st.inspect = _boom
        gib_miss, why_miss = tb.part_need_gib(
            "no_such_file.safetensors", fallback=12.57, adapter="krea2", assume="int8")
        check("⑭⁴ 读不到头部 ⇒ **退回表值** ✓（不猜、不炸 ✓），且写明「为什么是表值」✓",
              abs(gib_miss - 12.57) < 1e-9 and why_miss.get("source") == "table"
              and "读不出头部" in str(why_miss.get("why") or ""), why_miss)
    finally:
        st.inspect, tb.loader.quant_specs_of = real_inspect, real_specs

    #: ⚠️ 接线 ✗✗：算得再对，装配线不调它 ⇒ 白改 ✓（五条线逐条查，一条都不许漏 ✓✗）。
    from app.services.engine import runtime as rt  # noqa: PLC0415
    sources = {
        "h3": inspect.getsource(rt.EngineRuntime._ensure_loaded_internal),  # noqa: SLF001
        "zimage": inspect.getsource(rt.EngineRuntime._ensure_zimage_loaded),
        "qwen21": inspect.getsource(rt.EngineRuntime._ensure_qwen21_loaded),
        "krea2": inspect.getsource(rt.EngineRuntime._ensure_krea2_loaded),
        "qwen_image": inspect.getsource(rt.EngineRuntime._ensure_qwen_image_loaded),
        "sdxl": inspect.getsource(rt._image_device_decision),
    }
    missing = [name for name, text in sources.items() if "part_need_gib(" not in text]
    check("⑭⁵ 六条装配线（h3 / zimage / qwen21 / krea2 / qwen_image / sdxl）**都**按文件取需求 ✓✗"
          "（漏一条 ⇒ 那条换量化档时仍按旧表判 ✓，症状是 OOM 或白白切一半到内存 ✓✗）",
          not missing, {"缺": missing})


def quant_int4_stems(specs: dict[str, Any]) -> frozenset[str]:
    """测试用：把假 specs 翻成「4bit 层集合」✓（判据只走 `quant.int4_stems_of` 一处 ✓）。"""
    from app.services.engine import quant  # noqa: PLC0415

    return quant.int4_stems_of(specs)


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
    case_residency_kind()
    case_reserve_report()
    case_residency_total()
    case_need_by_file()
    case_no_park_in_auto()
    case_hand_copies()
    for note in _SKIPS:
        print(f"SKIP  {note}")
    print(f"\nSUMMARY: {len(_FAILS)} failed")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
