"""机器档（``profile``）—— G1：「8 GB 卡全按最低配跑」这句话的**唯一入口** ✓。

## 为什么要有这一层

同一个「低配」现在要动**三处**旋钮才成立 ✓✗：

* ``device_policy`` 的 ``policy``（``auto`` / ``gpu`` / ``cpu`` / ``balanced`` ✓）—— 决定「谁来算」✓；
* ``host_cache`` 的**配额比例**（:data:`host_cache.HOST_USE_RATIO`＝0.75 ✓）—— 决定「多少内存能拿来缓存权重」✓；
* ``vaeTiling`` 的 ``temporalChunk``（:func:`torch_backend.normalize_vae_tiling` ✓）—— 决定「解码怎么分时」✓。

**没有一处能回答「8 GB 卡按最低配跑」** ✗：用户改 ``policy=cpu`` 之后，配额还是按 96 GB 的机器给的 ✓✗、
解码还是一次性铺开 ✓✗（CPU 解码整段会顶满内存 ✓✗）。

⇒ 本模块把「机器档」做成**一组旋钮的唯一展开点** ✓：``profile`` 说一句，三处一起动 ✓。

⚠️ **``foresight`` 槽数与 ``reserveFor`` / ``gpuLayers`` / ``readAhead`` 一样，故意不在档位表里** ✓✗ ——
理由逐条写在 :data:`NOT_WIRED` ✓（最关键的一条：引擎侧默认槽数本来就是 **0＝关** ✓，
低配档没有可压的东西 ✓；而唯一能打开它的是显式环境变量 ✓，那条指令必须赢过档位 ✓✗）。

## 三条纪律（都是踩过的坑 ✓✗）

1. **档位表只引用，不抄数** ✓：默认值一律 ``host_cache.HOST_USE_RATIO`` / ``device_policy.DEFAULT_POLICY`` ✓
   —— 本文件里**不出现** ``0.75`` 这种"等于默认值"的字面量 ✗
   （抄一遍就是第二本账 ✓✗：那边改了、这边还按旧值跑，症状是"改了没生效"）。
2. **档位假设 ≠ 真读数** ✓✗：:attr:`Profile.gpu_gib` / :attr:`Profile.host_gib` 只用于
   :func:`budgets` 里写「这一档是按多少预算算的」✓ 和 :func:`hint` 的**提示** ✓ ——
   **绝不**覆盖 ``device_policy.probe()`` / ``scheduler.host_memory()`` 的实测值 ✗✗
   （真要模拟小机器走 ``DARME_FAKE_GPU_TOTAL_GIB`` / ``DARME_FAKE_HOST_GIB`` ✓ —— 那是 G4 的夹具 ✓，
   它替换的是**读数本身** ✓，与本模块是两件事 ✗）。
3. **档位不覆盖的不许装成旋钮** ✓✗：``reserveFor`` / ``gpuLayers`` / ``readAhead`` / ``foresight`` 四项
   **故意**不在档位表里 ✓（其中 ``gpuLayers`` 已按 §4.1 ✓、``readAhead`` 已按 §4.3 ✓ 接好线，但接线点是
   ``/load`` 的显式旋钮与**盘速** ✓，不是机器档 ✗），登记在 :data:`NOT_WIRED` 并**随回执带出** ✓
   —— 否则用户以为改了档位就改了余量/层数/预取深度 ✓✗。

## 与 ``policy`` 的分工（§5.2 明文）

``profile`` ＝ **机器档**（这台机器大概什么样 ✓）；``policy`` ＝ **本件驻留方式**（这一件放哪 ✓）。
两者不是两套词 ✓。冲突时**以 ``policy`` 为准** ✓ 并把被忽略的项写进 ``ignored`` ✓
（``profileIgnored=true`` ✓）—— 因为 ``policy`` 是调用方对这一件说的更具体的话 ✓。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import device_policy as policy_mod
from . import host_cache as host_cache_mod

__all__ = [
    "HostMemory",
    "NOT_WIRED",
    "PROFILES",
    "Profile",
    "budgets",
    "describe",
    "hint",
    "normalize_profile",
    "resolve",
]

GIB = 1024 ** 3


@dataclass(frozen=True)
class Profile:
    """一档机器档 ✓（纯数据 ✓：不读盘、不 import torch、不看读数 ✓）。"""

    key: str
    title: str
    #: 这一档**假设**的卡多大 ✓（``None`` ＝ 这一档不看卡 ✓，例如 ``cpu``）
    gpu_gib: float | None
    #: 这一档**假设**的内存多大 ✓
    host_gib: float | None
    #: 覆盖 ``device_policy`` 的策略 ✓（``None`` ＝ 不覆盖 ✓ —— ``auto`` 档专用 ✗）
    policy: str | None
    #: ``host_cache`` 配额比例 ✓（``None`` ＝ 不覆盖 ✓）
    host_ratio: float | None
    #: 解码时间分块（潜帧 ✓）—— 注入 ``vaeTiling.temporalChunk`` ✓（``None`` ＝ 不覆盖 ✓）
    temporal_chunk: int | None
    why: str


#: ⚠️ 本模块**不覆盖**的旋钮 ✓✗（每条都写清"为什么不是漏了" ✓ —— 回执里 ``notWired`` 原样带出 ✓）。
NOT_WIRED: tuple[tuple[str, str], ...] = (
    ("reserveFor",
     "余量按件分档（dit 25% / te 5% / vae 5%）已按真机实测定过；本仓**没有 8 GB 卡上的余量实测** ⇒ "
     "档位不敢乘一刀（乘 0.8 就是编数 ✗）。要动先补 L1/L2 实测表（§6）"),
    ("gpuLayers",
     "对外整数旋钮**已接**（§4.1 ✓，`/load` 的 `gpuLayers` ⇒ `residency.place()` 的层前缀算法 ✓），"
     "但它与「机器档」是两个层次 ✗：要几层取决于**这一份权重**与这张卡此刻的空闲 ✓，档位给的 "
     "`policy` 已经让 `place()` 自己按预算切 ✓ ⇒ 档位不覆盖它 ✗"),
    ("readAhead",
     "预取深度**已接**（§4.3 ✓，2026-11-07：`prefetch.resolve_depth` 按**盘速**分档 ✓，NVMe 2 / SATA 2 / "
     "机械 1，`DARME_READ_AHEAD` 显式赢过它 ✓），但**档位不覆盖它** ✗✗：预取深度是**盘**的属性（与卡的 "
     "大小无关 ✓），而单件读前置有 256 MiB 上限 ✓ ⇒ 2 槽只多占十几 MiB，「低配档压到 1」换不来内存、"
     "只会丢掉「读」与「反量化」的叠合 ✓✗"),
    ("foresight",
     "**引擎侧默认槽数已经是 0（关 ✓，`torch_backend.FORESIGHT_SLOTS_DEFAULT`）** ⇒ 低配档无需再压 ✓；"
     "而唯一能打开它的是显式环境变量 `DARME_FORESIGHT_SLOTS` ✓ —— 那条指令必须赢过档位 ✗✗，"
     "否则用户在 8 GB 机上设的 0 会被「档位说 1」静默顶掉（多占的那一层显存正是他要避开的 ✓）。"
     "⇒ 这一格**故意**不进档位表 ✓"),
)


#: 五档 ✓（§5.2 的取值域，一字不改 ✗）。
#:
#: ⚠️ ``l1`` / ``l2`` 的 ``gpu_gib`` / ``host_gib`` 是**假设**（8+32 / 6+16）✓ —— 只用于
#: :func:`budgets` 的 "按多少预算算" 与 :func:`hint` 的错配提示 ✓，**不参与任何判定** ✗✗。
PROFILES: dict[str, Profile] = {
    "auto": Profile(
        key="auto", title="不预设", gpu_gib=None, host_gib=None,
        policy=None, host_ratio=None, temporal_chunk=None,
        why="一律走现有默认（`policy=auto`、配额按 `HOST_USE_RATIO`、解码不分时 ✓）。"
            "这一档不存在「低配」这回事 ✗ —— 真空机器请显式选 l1/l2/cpu ✓",
    ),
    "gpu": Profile(
        key="gpu", title="富机器（≥22 GiB 卡）", gpu_gib=22.0, host_gib=64.0,
        policy=None, host_ratio=host_cache_mod.HOST_USE_RATIO, temporal_chunk=None,
        why="卡够大 ⇒ 不动任何「省」旋钮 ✓：配额就是默认 0.75 ✓、解码不分时 ✓"
            "（`temporalChunk` 留着不改 = 族默认 ✓，视频解码快 ✓）",
    ),
    "l1": Profile(
        key="l1", title="低配 L1（8 GiB 卡 / 32 GB 内存）", gpu_gib=8.0, host_gib=32.0,
        #: ⚠️ 仍是 ``auto``（**不是** ``cpu`` ✓）：8 GiB 卡上「联合驻留」正是要救的那条路 ✓✗ ——
        #: 钉成 ``cpu`` 就等于放弃用卡 ✓✗（`auto` 的取值域只有 ``off``/``split`` ✓，不会选 ``park`` ✓）。
        policy="auto", host_ratio=0.60, temporal_chunk=1,
        why="配额降到 60%（32 GB 要留给激活与帧缓冲 ✓ —— 720p 采样实测 3~6 GiB、CPU 前向还要翻倍 ✓）、"
            "视频解码按 1 潜帧分时（不换字节 ✓，只慢 ✓）。**不开** `tile`/`attnChunk` ✗ —— "
            "那两个一开就换字节（实测相对差 5.8e-03 ✓），低配不是换画质的理由 ✓✗",
    ),
    "l2": Profile(
        key="l2", title="低配 L2（6 GiB 卡 / 16 GB 内存）", gpu_gib=6.0, host_gib=16.0,
        policy="auto", host_ratio=0.50, temporal_chunk=1,
        why="比 L1 更紧：配额 50%（16 GB 上 60% 会把内存吃满 ✓✗ —— CPU 溢出段与激活都抢这块 ✓）、"
            "解码同样按 1 潜帧分时 ✓。卡只有 6 GiB ⇒ `auto` 大概率判 `split` ✓（`degraded` 如实标 ✓）",
    ),
    "cpu": Profile(
        key="cpu", title="无卡（纯 CPU）", gpu_gib=None, host_gib=32.0,
        #: 钉死 ``cpu`` ✓：机器根本没卡 ⇒ 让 `auto` 去"试探"只是白等一次 OOM ✓✗。
        policy="cpu", host_ratio=host_cache_mod.HOST_USE_RATIO, temporal_chunk=1,
        why="全 CPU ⇒ 显存那套旋钮都不生效 ✓，真正管用的是**内存配额** ✓（按总量 75% 缓存权重 ✓）"
            "与**解码分时** ✓（CPU 解码整段会顶满内存 ✓✗）",
    ),
}

@dataclass(frozen=True)
class HostMemory:
    """宿主内存读数 ✓（**带名字**的二元组 —— 见下面的 ⚠️）。

    ⚠️ 为什么要这个类型 ✗✗：:func:`scheduler.host_memory` 返回的是 ``(可用, 总量)`` ✓，
    而 :func:`budgets` 要的是 ``(总量, 可用)`` ✓✗ —— 两个 ``tuple[int, int]`` **形状全同** ✓，
    顺序反了**不会报错** ✓，只会把「可用」当「总量」算配额（症状＝配额虚高、看着像档位没生效 ✓✗）。
    ⇒ 装配层一律传本类型 ✓；元组仍然收 ✓，但按 ``(总量, 可用)`` 解释 ✓（写在签名里）。
    """

    totalBytes: int
    availBytes: int


#: 别名 ✓（人话输入 ⇒ 同一档 ✓；**不认的当场报错** ✗ —— 静默回落 ``auto`` 会让"我选了 L1"变成假话 ✓✗）
_ALIASES: dict[str, str] = {
    "default": "auto", "none": "auto", "": "auto",
    "big": "gpu", "high": "gpu", "22g": "gpu", "24g": "gpu",
    "8g": "l1", "8gb": "l1", "low": "l1", "l1-8g": "l1",
    "6g": "l2", "6gb": "l2", "l2-6g": "l2",
    "nocuda": "cpu", "nogpu": "cpu", "host": "cpu",
}


def normalize_profile(value: Any) -> str:
    """``None`` / 别名 / 大小写 → 五档之一 ✓（不认的 **抛** ``ValueError`` ✗，不静默回落 ✓）。"""
    if value is None:
        return "auto"
    key = str(value).strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    if key not in PROFILES:
        raise ValueError(
            f"未知机器档 {value!r} ✗；可用：{sorted(PROFILES)} ✓"
            f"（别名：{sorted(k for k in _ALIASES if k)}）")
    return key


def describe() -> list[dict[str, Any]]:
    """全部档位（给人看 ✓）：CLI / 前端「机器档」下拉直接用 ✓。"""
    return [
        {
            "key": prof.key, "title": prof.title,
            "gpuGiB": prof.gpu_gib, "hostGiB": prof.host_gib,
            "knobs": {
                "policy": prof.policy,
                "hostRatio": prof.host_ratio,
                "temporalChunk": prof.temporal_chunk,
            },
            #: 相对默认**真会改**的那几格 ✓（前端拿它决定要不要标「已改」✓；``auto``/``gpu`` 两档为空 ✓）
            "changes": _wants(prof),
            "why": prof.why,
        }
        for prof in PROFILES.values()
    ]


def _wants(prof: Profile) -> dict[str, Any]:
    """档位**真想改**的项 ✓（等于默认值的一律不算 ✓ —— 见 :func:`resolve` 的 ⚠️）。

    ⚠️ 默认值全部**引用**（``DEFAULT_POLICY`` / ``HOST_USE_RATIO`` / ``DEFAULT_SLOTS`` ✓）⇒
    那边改了、这里跟着变 ✓，永远不会出现「档位在压一个已经不是默认值的数」✓✗。
    """
    want: dict[str, Any] = {}
    if prof.policy is not None and prof.policy != policy_mod.DEFAULT_POLICY:
        want["policy"] = prof.policy
    if prof.host_ratio is not None and prof.host_ratio != host_cache_mod.HOST_USE_RATIO:
        want["hostRatio"] = prof.host_ratio
    if prof.temporal_chunk is not None:
        want["vaeTiling"] = {"temporalChunk": prof.temporal_chunk}
    return want


def resolve(profile: Any = "auto", *, policy: Any = None,
            vae_tiling: dict[str, Any] | None = None) -> dict[str, Any]:
    """档位 → **这一趟真该用的旋钮** ✓（纯函数 ✓：不看读数、不装载 ✓）。

    冲突裁决（§5.2 ✓）：调用方显式给的 ``policy`` **赢** ✓ ⇒ 档位那条被忽略并记进 ``ignored`` ✓
    （``profileIgnored=true`` ✓）。``vaeTiling.temporalChunk`` 同理 ✓。

    返回的 ``applied`` 只放**真的要改**的项 ✓✗：``auto`` 档 ⇒ ``applied={}`` ＋ ``source="default"``
    ✓ —— 不往装载键里塞一串「等于默认值」的假配置 ✗（那会让换回 ``auto`` 也白装一遍 ✓✗）。
    ⚠️ 同理 **``gpu`` 档的 ``applied`` 也是空的** ✓✗：它那两格（0.75 / 2）就是默认值 ✓ ⇒
    报「改了配额」会让用户以为档位有区别 ✓✗（真区别是"不压"而不是"压到 0.75" ✓）。
    """
    key = normalize_profile(profile)
    prof = PROFILES[key]
    want = _wants(prof)
    applied = dict(want)
    ignored: list[str] = []

    if "policy" in want and policy is not None:
        applied.pop("policy")
        ignored.append("policy")
    if "vaeTiling" in want:
        #: ⚠️ 只看**这一格**有没有被显式给 ✓✗：用户写了 ``{"tile": 512}`` 但没提
        #: ``temporalChunk`` ⇒ 档位**补上**它是合理的 ✓（tile 换字节、分时不换 ✓，两件事 ✗）。
        given = vae_tiling if isinstance(vae_tiling, dict) else {}
        if given.get("temporalChunk") is not None or given.get("temporal_chunk") is not None:
            applied.pop("vaeTiling")
            ignored.append("vaeTiling.temporalChunk")

    return {
        "profile": key,
        "title": prof.title,
        "source": "profile" if applied else "default",
        "applied": applied,
        "ignored": ignored,
        "profileIgnored": bool(ignored),
        "notWired": [{"knob": name, "why": why} for name, why in NOT_WIRED],
        "assumed": {"gpuGiB": prof.gpu_gib, "hostGiB": prof.host_gib},
        "why": prof.why,
    }


def hint(profile: Any = "auto", *, gpu_gib: float | None = None,
         host_gib: float | None = None) -> str | None:
    """档位与**实测**错配时说一句人话 ✓（只提示、不拦 ✗）。

    ``None`` ＝ 对得上 ✓（或者档位本来不预设 ✓）。
    """
    key = normalize_profile(profile)
    prof = PROFILES[key]
    notes: list[str] = []
    if prof.gpu_gib is not None and gpu_gib is not None:
        if gpu_gib + 0.5 < prof.gpu_gib:
            notes.append(f"这一档假设卡有 {prof.gpu_gib:g} GiB，而实测只有 {gpu_gib:.2f} GiB ✓")
        elif gpu_gib > prof.gpu_gib * 2:
            #: ⚠️ 阈值取 **2 倍**（不是 3 ✓）：8 GiB 档跑到 22.49 GiB 的机上就是 2.8 倍 ✓ ——
            #: 那正是"选错了档"，得说一句 ✓✗（3 倍会让本机 22.49 恰好滑过去 ✓✗）。
            notes.append(f"实测卡有 {gpu_gib:.2f} GiB、比这一档假设的 {prof.gpu_gib:g} GiB 大得多 ✓"
                         f"（低配档会白慢：内存配额被压小、解码被强行分时 ✗）")
    #: ⚠️ 内存按 **10% 相对容差**（不是 1 GiB 绝对 ✓）：32 GB 的机器报 30.0 GiB 可用是常态 ✓，
    #: 差 2 GiB 就喊一次"选错档"会把提示变成噪音 ✓✗（而 16 vs 30 这种真错配照样抓得住 ✓）。
    if prof.host_gib is not None and host_gib is not None and host_gib < prof.host_gib * 0.9:
        notes.append(f"这一档假设内存 {prof.host_gib:g} GiB，而实测只有 {host_gib:.2f} GiB ✓")
    return "；".join(notes) or None


def budgets(profile: Any = "auto", *, pressure: Any = None,
            host: HostMemory | tuple[int, int] | None = None) -> dict[str, Any]:
    """**一处回答这台机器现在的预算** ✓（``/engine/runtime`` 的 ``budgets`` ✓，§5.2）。

    ``source`` 如实说明**哪几栏是真读数** ✓（``cuda+host`` / ``cuda`` / ``host`` / ``none`` ✓）——
    测不到就是 ``none`` 且数值 ``None`` ✓，**不写 0** ✗（0 会被读成"一点都没有" ✓✗）。

    ⚠️ ``host`` 传元组时按 ``(总量, 可用)`` 解释 ✓✗ —— 而 :func:`scheduler.host_memory` 给的是
    ``(可用, 总量)`` ✓ ⇒ 调用方要么传 :class:`HostMemory` ✓，要么自己换个顺序 ✓。
    """
    key = normalize_profile(profile)
    prof = PROFILES[key]
    gpu_total = float(getattr(pressure, "totalBytes", 0) or 0) / GIB if pressure is not None else 0.0
    gpu_free = float(getattr(pressure, "freeBytes", 0) or 0) / GIB if pressure is not None else 0.0
    if isinstance(host, HostMemory):
        host_total = float(host.totalBytes) / GIB
        host_avail = float(host.availBytes) / GIB
    elif host:
        host_total = float(host[0]) / GIB
        host_avail = float(host[1]) / GIB
    else:
        host_total = host_avail = 0.0
    ratio = prof.host_ratio if prof.host_ratio is not None else host_cache_mod.HOST_USE_RATIO
    have_gpu, have_host = gpu_total > 0, host_total > 0
    return {
        "profile": key,
        #: ⚠️ 这一栏必须说清"哪来的" ✓✗：低配模拟（``DARME_FAKE_*``）开着时读数本身就是假的 ✓，
        #: 而档位假设永远只是假设 ✓ —— 混作一谈就会把"我以为有 8 GiB"当成实测 ✓✗。
        "source": ("cuda+host" if have_gpu and have_host
                   else "cuda" if have_gpu
                   else "host" if have_host
                   else "none"),
        "gpuGiB": round(gpu_total, 2) if have_gpu else None,
        "gpuFreeGiB": round(gpu_free, 2) if have_gpu else None,
        "hostGiB": round(host_total, 2) if have_host else None,
        "hostFreeGiB": round(host_avail, 2) if have_host else None,
        "hostRatio": ratio,
        #: 这一档下 ``host_cache`` 能拿多少（＝可用内存 × 比例 ✓）
        "hostQuotaGiB": round(host_avail * ratio, 2) if have_host else None,
        "assumed": {"gpuGiB": prof.gpu_gib, "hostGiB": prof.host_gib},
        "hint": hint(key, gpu_gib=gpu_total if have_gpu else None,
                     host_gib=host_total if have_host else None),
    }
