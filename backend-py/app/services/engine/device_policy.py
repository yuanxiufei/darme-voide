"""设备策略：**默认 GPU 优先** ✓，显存吃紧时自动**转移 / 联合 / 降级** ✓（2026-10-31 创作者口径）。

## 这一层补的是什么洞 ✗

此前设备只有三档「手工开关」✓：全上卡 / 全在内存 / `residency` 手工切成 `split|park` ✓。
三个都是**静态**的 ✗ —— 谁在哪台设备上跑，取决于**构造时传的参数** ✓，与**当时卡上有多满**无关 ✓✗。
后果：卡被别的进程占着（或本进程另一头还压着激活）时，装配照样硬上 ✓ ⇒ `CUDA out of memory` ✓✗；
而反过来，卡明明空着却因为上一次开了 `--cpu` 就整份在内存里爬 ✓✗。

这里把「**谁跑在哪**」变成**每次进阶段前现算** ✓：

| 判据 | 结论 | 代价 |
|---|---|---|
| 剩余显存 ≥ 该件 + 1 GiB 余量 | 上卡 ✓（老路） | 0 |
| 剩余放不下整件，但仍 ≥ 512 MiB | 上卡 ＋ `residency="split"` ✓（GPU/CPU **联合**） | 少数层在 CPU 上算 |
| 剩余 < 512 MiB | 整件**转移到 CPU** ✓（降级，但不失败） | 慢，但跑得完 |
| 显存占用 ≥ 92%（连该件要多少都不知道时） | 转移到 CPU ✓ | 同上 |

## 为什么不拿「GPU 利用率」当主判据 ✗

`nvidia-smi` 的 **SM 利用率**才是"算得忙不忙" ✓，但它要 nvml ✓ 而 torch 只给显存
（`torch.cuda.mem_get_info` ✓）；本仓不引新依赖 ✓ ⇒ 主判据是**显存占用率** ✓，
`utilPercent` 留成**可注入**的字段 ✓（将来接了 nvml 直接填 ✓，判据代码一行不用改 ✓）。
⇒ 报告里必须写 `source` ✓：`"memory"` 就是「按显存推的」✗不许读成「实测利用率」✗✓。

## 降级的边界（**不动用户给的质量参数** ✗✗）

「降级」在本模块里只有两种含义 ✓：**换设备**（cuda → cpu）与**换驻留方式**（off → split）✓。
**不降步数、不降分辨率、不降精度以外的任何创作参数** ✗✗ —— 那些由调用方决定，
本模块擅自调小会让"图变糊了但没人知道为什么"✓✗。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: 对外可配的策略名 ✓（**单一真源** ✓ —— `TorchBackend` / `runtime` / 报告都读它）
POLICIES: tuple[str, ...] = ("auto", "gpu", "cpu", "balanced")
#: 默认 ✓：GPU 优先、压力大才动 ✓（与老行为一致：有卡就用卡 ✓）
DEFAULT_POLICY = "auto"

GIB = 1 << 30
#: 给激活/工作区留的**硬余量** ✓：「刚好装得下」在真前向上几乎必然 OOM ✗
HEADROOM_BYTES = 1 * GIB
#: 少于这个数就不值得往卡上放 ✗（连几个 embedding 都塞不下 ⇒ 别假装联合 ✓）
MIN_SPLIT_FREE_BYTES = 512 * (1 << 20)
#: 已用率超过它 ⇒ 视为「压力高」✓（够用时仍上卡，但报告要标出来 ✓）
HIGH_USED_RATIO = 0.80
#: 已用率超过它 ⇒ **需求未知**时直接转移 ✓（这时候赌"还塞得下"是最坏的选择 ✗）
HARD_USED_RATIO = 0.92

def is_cuda_device(device: Any) -> bool:
    """``device`` 指向的是不是**卡** ✓（``"cuda"`` / ``"cuda:0"`` / ``torch.device("cuda:1")`` ✓）。

    ⚠️⚠️ 这是**全仓唯一**的「它在不在卡上」判断 ✗✗（2026-10-31 收口 ✓）。此前**三种写法并存** ✓✗：

    * ``str(device).split(":")[0] == "cuda"`` ✓（`torch_backend` 的联合驻留那处，已改 ✓）
    * ``str(device).startswith("cuda")`` ✓（`sdxl_backend._is_cuda_device`、`torch_backend:798`）
    * ``device == "cuda"`` ✗✗ —— **这个在 ``cuda:0`` 上答 False** ✓✗，而它出现在
      ``TorchBackend.describe()`` 的 ``cudaAvailable`` 上 ✓：多卡机上后端明明在 ``cuda:1``
      （或调用方显式写了 ``device="cuda:0"`` ✓），能力表却报「不在卡上」✓✗ ——
      而 ``engine_readiness.device_facts`` 正是读它做判断的口 ✓✗。

    ⚠️ 三种写法在**默认 ``"cuda"`` 这一种输入上全对** ✓ ⇒ 单卡 + 不写卡号的本机上永远试不出来 ✓✗
    （本仓的老毛病：判据只在最省事的那一档成立 ✓）。收成一个函数后，新写法只能有一种 ✓。

    ⚠️ 不 import torch ✗：``str(torch.device("cuda", 1))`` 就是 ``"cuda:1"`` ✓ ⇒ 纯字符串够用 ✓
    （本模块的契约是「无 torch 也能判」✓，`selftest()` 也靠它 ✓）。
    """
    if device is None:
        return False
    #: ⚠️ 小写化再判 ✓：``"CUDA:0"`` 也是合法写法 ✓（`torch.device` 自己会小写化 ✓，
    #: 但调用方传裸字符串时不会 ✓ ⇒ 这里兜住 ✓）。
    return str(device).strip().lower().split(":", 1)[0] == "cuda"


def is_cpu_device(device: Any) -> bool:
    """``device`` 指向的是不是**内存** ✓（``"cpu"`` / ``"cpu:0"`` / ``torch.device("cpu")`` ✓）。

    ⚠️ 与 :func:`is_cuda_device` **不是取反关系** ✗✗，所以单列一个函数、而不是让调用方写 ``not``：
    ``None``／``"mps"``／``"xpu"`` 两边都是 False ✓。合成 ``not is_cuda_device(...)`` 会把
    「不知道是什么设备」判成 CPU ✓✗，那正好是本仓最怕的一档（静默选了最省事的答案 ✓✗）。

    ⚠️ 收口理由同上（2026-10-31）：`torch_backend` 的 H3 int8 分档写过
    ``str(self._device).split(":")[0] != "cpu"`` ✓ —— 与判卡那三处是**同一件事的第四种写法** ✓✗，
    改设备口径时漏掉它 ⇒ int8 走错分支（全份塞卡上 ⇒ OOM，或反过来留在内存 ✓✗）。
    """
    if device is None:
        return False
    return str(device).strip().lower().split(":", 1)[0] == "cpu"


#: 策略别名 ✓（命令行/前端容易写错大小写或换说法 ❗️）
_ALIASES: dict[str, str] = {
    "auto": "auto", "gpu-first": "auto", "gpufirst": "auto", "default": "auto",
    "gpu": "gpu", "gpu-only": "gpu", "cuda": "gpu",
    "cpu": "cpu", "cpu-only": "cpu",
    "balanced": "balanced", "balance": "balanced", "joint": "balanced", "split": "balanced",
}


def normalize_policy(value: Any) -> str:
    """``None`` ⇒ 默认 ✓；不认识 ⇒ **抛** ✓（不静默回落到 auto ✗ —— 拼错一个字母就换策略，查不出来 ✗✗）。"""
    if value is None or str(value).strip() == "":
        return DEFAULT_POLICY
    key = str(value).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"不认识的设备策略 {value!r} ✗（可用 {' / '.join(POLICIES)} ✓）")


@dataclass(frozen=True)
class Pressure:
    """**当时的**卡况 ✓（不是"能力"，是**读数** ✓ —— 每次要现测 ✓）。

    ⚠️ ``freeBytes`` 的口径是「**本次分配真能拿到多少**」 ✓✗ —— 它＝驱动报的空闲 ＋ 本进程
    allocator 里**可回收/可重用**的缓存块 ✓（见 `probe()` 的注释：那部分在 `mem_get_info`
    眼里是"已占" ✓，实际却随时能复用 ✓）。所以它**不小于** ``mem_get_info()[0]`` ✓✗，
    两者不再等价 —— 想要裸驱动读数，用 ``totalBytes - usedBytes`` 之外的场景请直接调 torch ✓。
    """

    totalBytes: int
    freeBytes: int
    #: 本进程占的字节数 ✓（`memory_allocated` ✓；取不到 ⇒ None ✓ 不猜 0 ✗）
    allocatedBytes: int | None = None
    #: SM 利用率 ✓（0–100 ✓；无 nvml ⇒ None ✓ —— 见模块头「为什么不拿它当主判据」✓）
    utilPercent: float | None = None
    #: 这个数**从哪来** ✓：``"memory"``（显存推的 ✓）/ ``"nvml"``（真利用率 ✓）
    source: str = "memory"

    @property
    def usedBytes(self) -> int:
        return max(0, int(self.totalBytes) - int(self.freeBytes))

    @property
    def usedRatio(self) -> float:
        total = int(self.totalBytes)
        return 0.0 if total <= 0 else self.usedBytes / total

    @property
    def freeGiB(self) -> float:
        return round(self.freeBytes / GIB, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "freeGiB": self.freeGiB,
            "totalGiB": round(int(self.totalBytes) / GIB, 2),
            "usedGiB": round(self.usedBytes / GIB, 2),
            "usedRatio": round(self.usedRatio, 3),
            "utilPercent": self.utilPercent,
            "source": self.source,
        }


def probe(torch_mod: Any = None, *, index: int | None = None,
          util_percent: float | None = None) -> Pressure | None:
    """读一次卡况 ✓；**没卡 / 取不到 ⇒ None** ✓（不猜 ✗ —— 见 `residency.budget_from_cuda` 同口径 ✓）。"""
    if torch_mod is None:
        try:  # noqa: SIM105 —— 可选依赖 ✓：没装 torch 也要能报"测不到"而不是崩 ✓
            import torch as torch_mod  # noqa: PLC0415
        except Exception:
            return None
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return None
    try:
        free, total = cuda.mem_get_info(index)
    except Exception:  # pragma: no cover - 老驱动 / 不支持的设备
        return None
    allocated = None
    reclaimable = 0
    try:
        allocated = int(cuda.memory_allocated(index))
        #: ⭐⭐ **allocator 里"空闲但还没还给驱动"的那部分，必须算进可用** ✓✗（2026-11-01 实测 ✓）。
        #: 为什么必须这么算 ✗✗：`mem_get_info` 的 free 是**驱动视角** ✓ —— 我们自己 allocator
        #: 留着的缓存块在它眼里就是"已被占用" ✓✗；可那些块**随时能重用** ✓（真分配时会直接复用 ✓），
        #: 只有 `empty_cache()` 之后才会还给驱动 ✓、而驱动归还**有延迟**（实测同一毫秒连清两次 ✓：
        #: 第一次放 0 字节 ✓、0.07 秒后再清一次才放掉 16058 MiB ✓）。
        #: 后果是硬的：换装时 probe 报「余 5.41 GiB」✓（真实可用 21 GiB ✓）⇒ 25 GiB 的塔只上卡
        #: 3.5 GiB ✓、17.9 GiB 的 DiT 只上卡 4 GiB ✓ ⇒ 越换越保守 ✓、最后卡还被 WDDM 超分 ✓✗
        #: （`reserved` 一度 36402 MiB > 卡 23027 MiB ✓）。
        reclaimable = max(0, int(cuda.memory_reserved(index)) - allocated)
    except Exception:  # pragma: no cover - 同上
        allocated = None
        reclaimable = 0
    return Pressure(totalBytes=int(total), freeBytes=int(free) + reclaimable,
                    allocatedBytes=allocated, utilPercent=util_percent,
                    source="memory" if util_percent is None else "nvml")


@dataclass(frozen=True)
class Decision:
    """**这次**让谁干活 ✓ —— 每个字段都要能进报告 ✓（静默改设备 = 用户以为在上卡 ✓✗）。"""

    device: str
    residency: str
    degraded: bool
    reason: str
    policy: str
    #: 这次估算要多少 GiB ✓（``None`` ⇒ 调用方没给、按占用率粗判 ✓）
    needGiB: float | None = None
    pressure: dict[str, Any] | None = None
    #: 属于哪个阶段/部件 ✓（``None`` ⇒ 装配期的一次性决策 ✓）
    phase: str | None = None
    part: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device": self.device, "residency": self.residency,
            "degraded": self.degraded, "reason": self.reason, "policy": self.policy,
        }
        if self.needGiB is not None:
            out["needGiB"] = self.needGiB
        if self.pressure is not None:
            out["pressure"] = dict(self.pressure)
        if self.phase is not None:
            out["phase"] = self.phase
        if self.part is not None:
            out["part"] = self.part
        return out


def decide(*, policy: Any = DEFAULT_POLICY, requested_device: str | None = None,
           need_bytes: int = 0, pressure: Pressure | None = None,
           gpu_available: bool = True, phase: str | None = None,
           part: str | None = None) -> Decision:
    """算出**这次**该去哪 ✓（纯函数 ✓ —— 不读盘、不碰 torch ✓，可反复调 ✓）。

    ⚠️ ``need_bytes=0`` 是「**不知道**要多少」✗ 而不是「不要空间」✓：
    这时只能按**已用率**粗判 ✓（结论会更保守 ✓，报告里的 ``needGiB`` 也缺席 ✓ 看得见 ✓）。
    """
    name = normalize_policy(policy)
    need_gib = None if need_bytes <= 0 else round(int(need_bytes) / GIB, 2)
    metrics = None if pressure is None else pressure.to_dict()

    def _make(device: str, residency: str, degraded: bool, reason: str) -> Decision:
        return Decision(device=device, residency=residency, degraded=degraded,
                        reason=reason, policy=name, needGiB=need_gib,
                        pressure=metrics, phase=phase, part=part)

    # ① 显式要 CPU（或策略就是 cpu ✓）⇒ 尊重，不改 ✗
    requested = str(requested_device or "").strip().lower()
    if name == "cpu":
        return _make("cpu", "off", False, "策略 `cpu` ⇒ 全在内存 ✓（不是降级，是你选的 ✓）")
    if requested.startswith("cpu"):
        return _make("cpu", "off", False, f"显式请求 `{requested}` ✓ ⇒ 不上卡 ✓")

    # ② 没有可用 GPU
    if not gpu_available:
        if name == "gpu":
            return _make("cuda", "off", False,
                         "策略 `gpu` ⇒ **照上卡** ✓（⚠️ 本机实测没有可用 CUDA ✗ ⇒ 装载会当场失败 ✓，"
                         "这是策略要求的结果 ✓ 不是本模块能兜的 ✗）")
        return _make("cpu", "off", True, "本机没有可用 CUDA ✓ ⇒ 整条链在 CPU 上跑 ✓（降级 ✓）")

    # ③ 强制上卡
    if name == "gpu":
        return _make("cuda", "off", False, "策略 `gpu` ⇒ 只上卡 ✓（显存不够时**直接 OOM** ✓ 不转移 ✗）")

    # ④ 测不到卡况 ⇒ 按老路（优先 GPU ✓），但如实说没测到 ✓
    if pressure is None:
        return _make("cuda", "off", False, "测不到显存读数 ✓ ⇒ 按默认**优先 GPU** ✓（没有依据降级 ✗）")

    free = int(pressure.freeBytes)
    used_ratio = pressure.usedRatio
    pressured = used_ratio >= HIGH_USED_RATIO

    # ⑤ 需求**未知** ⇒ 只能按占用率粗判
    if need_bytes <= 0:
        if used_ratio >= HARD_USED_RATIO:
            return _make("cpu", "off", True,
                         f"显存已用 **{used_ratio:.0%}** ≥ {HARD_USED_RATIO:.0%} ✗ 且这次不知道要装多大 "
                         "⇒ 转移 CPU ✓（赌它还塞得下是最坏的选择 ✗）")
        if pressured and name in ("auto", "balanced"):
            return _make("cuda", "split", True,
                         f"显存已用 **{used_ratio:.0%}** ≥ {HIGH_USED_RATIO:.0%} ⇒ 上卡但改**联合驻留** ✓"
                         f"（卡上 {pressure.freeGiB} GiB 留给一部分层 ✓，其余在 CPU ✓）")
        return _make("cuda", "off", False,
                     f"显存已用 **{used_ratio:.0%}** ✓、余 {pressure.freeGiB} GiB ✓ ⇒ 正常上卡 ✓")

    # ⑥ 需求已知：够 / 差一点 / 差太多
    if free >= need_bytes + HEADROOM_BYTES:
        if pressured and name == "balanced":
            return _make("cuda", "split", True,
                         f"装得下（余 {pressure.freeGiB} GiB ≥ {need_gib} + 1 GiB ✓），但已用 "
                         f"**{used_ratio:.0%}** ≥ {HIGH_USED_RATIO:.0%} ⇒ 按 `balanced` 主动**联合** ✓"
                         "（给别的任务留余量 ✓）")
        note = (f"装得下 ✓：余 {pressure.freeGiB} GiB ≥ 需 {need_gib} GiB ＋ 1 GiB 余量 ✓ ⇒ 上卡 ✓"
                + (f"（⚠️ 显存已用 {used_ratio:.0%}，压力偏高但够用 ✓）" if pressured else ""))
        return _make("cuda", "off", False, note)

    if free >= MIN_SPLIT_FREE_BYTES:
        return _make("cuda", "split", True,
                     f"余 {pressure.freeGiB} GiB < 需 {need_gib} GiB ＋ 1 GiB ✗ 但 ≥ "
                     f"{MIN_SPLIT_FREE_BYTES / GIB:.2f} GiB ✓ ⇒ **GPU/CPU 联合** ✓"
                     "（卡上放能放的层 ✓，剩下的在 CPU 上算 ✓；比整份退 CPU 快 ✓）")
    return _make("cpu", "off", True,
                 f"余 {pressure.freeGiB} GiB < {MIN_SPLIT_FREE_BYTES / GIB:.2f} GiB ✗ ⇒ 卡上放不下任何"
                 f"有意义的切片 ⇒ 整件**转移到 CPU** ✓（需要 {need_gib} GiB ✓）")


def selftest() -> list[str]:
    """本模块自检 ✓（纯逻辑，无 torch ✓）⇒ 返回问题清单 ✓（空 ⇒ 全绿 ✓）。"""
    problems: list[str] = []

    def _fake(total_gib: float, free_gib: float) -> Pressure:
        return Pressure(totalBytes=int(total_gib * GIB), freeBytes=int(free_gib * GIB))

    need = int(13.26 * GIB)
    # ① 够用 ⇒ 上卡、不降级
    d = decide(need_bytes=need, pressure=_fake(22.49, 20.0))
    if d.device != "cuda" or d.residency != "off" or d.degraded:
        problems.append(f"够用时应上卡不降级，实际 {d.to_dict()}")
    # ② 差一点 ⇒ 联合
    d = decide(need_bytes=need, pressure=_fake(22.49, 6.0))
    if d.device != "cuda" or d.residency != "split" or not d.degraded:
        problems.append(f"差一点时应 GPU/CPU 联合，实际 {d.to_dict()}")
    # ③ 快满 ⇒ 转 CPU
    d = decide(need_bytes=need, pressure=_fake(22.49, 0.2))
    if d.device != "cpu" or not d.degraded:
        problems.append(f"余量过小应转 CPU，实际 {d.to_dict()}")
    # ④ 无卡 ⇒ CPU（除非策略强制 gpu）
    d = decide(gpu_available=False, pressure=None)
    if d.device != "cpu" or not d.degraded:
        problems.append(f"无卡应转 CPU，实际 {d.to_dict()}")
    if decide(policy="gpu", gpu_available=False, pressure=None).device != "cuda":
        problems.append("策略 gpu 在无卡时应**照上卡**（失败留给装载层说）")
    # ⑤ 需求未知 + 高占用 ⇒ CPU
    d = decide(pressure=_fake(22.49, 1.0))
    if d.device != "cpu":
        problems.append(f"需求未知且占用 ≥92% 应转 CPU，实际 {d.to_dict()}")
    # ⑥ 测不到卡况 ⇒ 按优先 GPU
    if decide(pressure=None).device != "cuda":
        problems.append("测不到卡况时应保持默认优先 GPU")
    # ⑦ 显式 cpu 请求 ⇒ 不动
    if decide(requested_device="cpu", pressure=_fake(22.49, 20.0)).device != "cpu":
        problems.append("显式请求 cpu 时应尊重")
    # ⑧ 拼错的策略名必须抛
    try:
        normalize_policy("gpu-frist")
    except ValueError:
        pass
    else:
        problems.append("拼错的策略名应抛 ValueError（不许静默回落 auto）")
    return problems


if __name__ == "__main__":  # pragma: no cover - 手工入口
    bugs = selftest()
    print("\n".join(bugs) if bugs else "device_policy selftest: 全部通过 ✓")
