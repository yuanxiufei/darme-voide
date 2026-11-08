"""任务级可行性预检 ✓ —— **不装载**就答「这台机器跑不跑得动这条链」（G2，2026-11-07 ✓）。

## 为什么要单开一层 ✗

仓里已经有两个"装不装得下"的答案，但都不答用户最先问的那句 ✓：

* :func:`inventory.readiness` 答「**件齐不齐**」✓（盘上有没有、头部对不对 ✓）；
* :func:`loader.plan_stage`（``GET /load-plan``）答「**怎么排班**」✓，可它算的是
  ``component.gib`` ＝ **文件字节** ✓✗ —— 那是"要读进来多少" ✗，不是"装完占多少" ✓。

两者之间空着的问题就是本模块：**权重驻留 vs 本机预算** ✓。
历史已经踩过两次（都是"预检说装得下、真装 OOM、报错落在后装的件上"✗）：

* Krea2 那张 ``12.57 GiB`` 的表是 **int8 档**量的 ✓ —— 换成 int4 档照抄它 ⇒ 反量化成 bf16
  23.9 GiB ⇒ 整份上卡 ⇒ 采样第 1 步 OOM ✓✗；
* Qwen-Image 2511 按 bf16 那份的 ``38.05`` 判 ✓ —— 换成 int8 文件（真驻留 19.09 ✓）却仍按 38.05
  判 ⇒ 白白切一半到内存里逐层爬 ✓✗（反过来，把表压到 19.09 又会让 bf16 那份装到 20 GiB 才 OOM ✗）。

⇒ 所以本模块**一个数字都不自己写** ✓✗：逐件需求一律走
:func:`torch_backend.stage_part_need` ✓（装配线的**同一个出口** ✓，见 :data:`STAGE_PART_SPECS`），
表值一律走 :attr:`TorchBackend.STAGE_NEED_TABLES` ✓（**引用**那几本账 ✓，见 ``engine_feasibility_test`` ③）。

## 只报「权重驻留」，激活如实标 unknown ✓（2026-11-07 方案 A ✓）

本仓**没有**逐族「激活峰值」实测表 ✓ —— 采样工作区随分辨率、分块解码、注意力实现而变 ✓✗。
⇒ 回执里 ``activation.basis`` 一律 ``"unknown"`` ✓，**不编数** ✓✗；代价是 ``verdict`` 只能
保守下判（见 :data:`DOWNGRADE`：算出来是 ``perfect`` 也只报 ``good`` ✓）。

⚠️ 这不是"漏了一半"✗、而是**划清边界** ✓：``need.gpuGiB`` 是**权重**要多少 ✓，
真判决要拿它 + 采样工作区一起看 ✓ —— 回执里 ``activation.whatItMeans`` 逐字写这句 ✓。

## 「峰值」怎么算 ✓（不是简单求和 ✗）

同一条链的三件**不一定同时**在卡上 ✓：``qwen_image21`` 装配期只装 DiT（13.26 ✓），
text（14.79 ✓）与 VAE 按阶段换装 ✓ —— 三件求和 28.68 GiB 会得出「22.49 的卡装不下」✗✗，
而那正是本机每天都在跑的那条线 ✓。反过来 ``music3`` 三件**确实**同驻（实测 13.34 ✓），
按 max 算会少 8 GiB ✓✗。

⇒ 逐档登记 :data:`STAGE_PEAK` ✓（``combine="sum"`` 同驻 / ``"max"`` 换装 ✓），每行 ``why`` 写依据 ✓，
并由 ``engine_feasibility_test`` ⑦ 拿 ``runtime.py`` 的装配事实**机械对账** ✓（不许手写两套 ✗）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import device_policy, inventory as inventory_mod, scheduler
from .torch_backend import STAGE_PART_SPECS, stage_part_need

GIB = float(device_policy.GIB)


@dataclass(frozen=True)
class PeakSpec:
    """一档的「**峰值**驻留由哪几件决定」✓（见模块头「峰值怎么算」✓）。

    * ``parts``：参与这条链的件 ✓（``dit`` / ``text`` / ``vae``）——
      ⚠️ 是"这条链要用到的件"✓，不一定"同时在卡上"✗（看 ``combine`` ✓）；
    * ``combine``：``"sum"`` ＝ 这些件**同时**驻留在卡上 ✓✗（整链同驻 / 双件配对）；
      ``"max"`` ＝ 按阶段换装 ⇒ 峰值是其中**最大**的那一件 ✓；
    * ``mode``：给人读的档型（``chain`` / ``pair`` / ``phased`` / ``single`` ✓）；
    * ``why``：**依据** ✓ —— 写清"哪条装配线怎么装的"，别写"经验值" ✗。
    """

    parts: tuple[str, ...]
    combine: str
    mode: str
    why: str


#: ⭐ 逐档「峰值由哪几件决定」✓ —— **每一行都有出处** ✓（``why`` 指向装配线行为 ✓）。
#:
#: ⚠️ 新增一族**必须**在这里加一行 ✓✗（``engine_feasibility_test`` ② 拿 ``STAGE_PART_SPECS``
#: 的键集对账 ✓：漏一行就红 ✓）；否则 ``check()`` 会退回「只算 DiT 一件」（报得偏小 ⇒ 假绿 ✗）。
STAGE_PEAK: dict[str, PeakSpec] = {
    "h3": PeakSpec(
        ("dit", "text", "vae"), "max", "phased",
        "``declare_chain(phase_parts={encode:text, sample:dit, decode:vae})`` ✓ —— 三件分时，"
        "真 VAE 常驻 CPU、decode 阶段用 ``_relocate_keep_parts`` 把 DiT 退掉再搬上来 ✓ ⇒ 峰值=max ✓"),
    "zimage": PeakSpec(
        ("dit", "text", "vae"), "sum", "chain",
        "整链同驻 ✓（``CHAIN_NEED_PARTS_GIB['zimage']`` 11.46+7.49+0.33≈19.28 与 "
        "``CHAIN_NEED_GIB['zimage']`` 逐格对得上 ✓）⇒ 峰值=求和 ✓"),
    "qwen_image21": PeakSpec(
        ("dit", "text", "vae"), "max", "phased",
        "装配线**只装 DiT** ＋ ``declare_chain`` 登记 ✓，``pipeline`` 在 encode/sample/decode 三边界"
        "``enter_phase`` 互斥换装 ✓（三件同驻 28.68 GiB 装不进 22.49 的卡 ✓✗）⇒ 峰值=max ✓"),
    "krea2": PeakSpec(
        ("dit", "text", "vae"), "max", "phased",
        "``decide_for(('dit',))`` ＋ text/VAE 标 ``phased=True`` ✓（卡上常驻 DiT 12.57 ✓，"
        "文本塔与 VAE 按需换装 ✓）⇒ 峰值=max ✓"),
    "seedvr2": PeakSpec(
        ("dit", "vae"), "sum", "pair",
        "``decide_for(('dit','vae'))`` ✓ —— 这条线**没有文本塔** ✓，DiT 与 VAE 同驻 ✓ ⇒ 峰值=求和 ✓"),
    "music3": PeakSpec(
        ("dit", "text", "vae"), "sum", "chain",
        "``decide_for(('dit','text','vae'))`` ✓ —— 实测三件同驻 13.34 GiB ✓（20 秒歌全链）⇒ 峰值=求和 ✓"),
    "wan22": PeakSpec(
        ("dit", "text", "vae"), "max", "phased",
        "``decide_for(('dit',))`` ✓；text/VAE 分时 ✓；**双专家不并存** ✓（按 σ 阈值换装 ✓）"
        "⇒ 峰值=max（DiT 只按**一件**算 ✓，不能加上 ``dit_path2`` ✗）"),
    "qwen_image": PeakSpec(
        ("dit", "text", "vae"), "max", "phased",
        "``decide_for(('dit',))`` ＋ ``declare_chain`` ✓ —— DiT 常驻、text/VAE 换装 ✓ ⇒ 峰值=max ✓"),
    "image": PeakSpec(
        ("dit",), "sum", "single",
        "SDXL 单 checkpoint ✓（``assembly_for_stage`` 把 checkpoint 塞进 ``dit_path`` ✓）"
        "⇒ 峰值=这一件 ✓"),
}

#: 四档结论 ✓（从宽到紧 ✓）。**语义**都是「权重」这一半 ✓，见模块头 ✓。
VERDICTS: tuple[str, ...] = ("perfect", "good", "marginal", "too_tight")

#: ⭐ 因为激活是 ``unknown`` ✓ ⇒ ``perfect`` **降一档**成 ``good`` ✓（2026-11-07 方案 A ✓）。
#: 为什么只降 ``perfect`` ✗：``good`` 自己就写着"还行但不富余" ✓，再往下压会让"空卡跑小模型"
#: 也报 ``marginal``（那是在喊狼来了 ✗✗）；而 ``perfect`` 是**唯一**一个会被读者当成
#: "闭眼能跑"的词 ✓ —— 在激活未知时那个承诺不成立 ✓✗。``too_tight`` 不再往下压（已经是最坏 ✓）。
DOWNGRADE: dict[str, str] = {
    "perfect": "good",
    "good": "marginal",
    "marginal": "marginal",
    "too_tight": "too_tight",
}

PART_TITLES: dict[str, str] = {
    "dit": "主模型（DiT）",
    "text": "文本塔（TE）",
    "vae": "VAE",
}

#: 卡上余量 ÷ 权重峰值 ⇒ 结论 ✓（**权重**这一半的阈值 ✓，见模块头 ✓）。
PERFECT_RATIO = 1.60
GOOD_RATIO = 1.25

#: 「够几条跑次才敢报秒数」的闸门 ✓（§5.1 ✓）。
#:
#: ⚠️ 为什么不是 1 ✗：单次跑次会被首次装载、磁盘冷缓存、别的进程抢卡拉长两三倍 ✓✗；
#: 一条样本就把那个数报成「这条链要多久」 ⇒ 用户按它排版期，第一次真跑就差一个小时 ✓✗。
#: 3 条是"能看出量级、又不至于一条都没有就闭嘴"的最小值 ✓（比 :data:`tuning.MIN_SAMPLES` 松 ✓：
#: 那个是**改默认参数**的闸门 ✗，这里只是**报个数** ✓）。
ESTIMATE_MIN_SAMPLES = 3


def _vae_path(assembly: dict[str, Any]) -> str | None:
    """装配里的 VAE 落在哪个槽 ✓ —— **三格都看** ✓（不抄 ``_SLOT_BY_STAGE_VAE`` ✗：
    那个映射将来加一档，这里跟着错位 ✓✗）。"""
    for key in ("image_vae_weights", "video_vae_weights", "audio_vae_weights"):
        value = assembly.get(key)
        if value:
            return str(value)
    return None


def _file_gib(path: str | None) -> float:
    """文件字节 ⇒ GiB ✓（读不到 ⇒ 0 ✓，缺件另由 ``missing`` 报 ✓）。"""
    try:
        return os.stat(str(path)).st_size / GIB
    except (OSError, TypeError, ValueError):
        return 0.0


def weight_need(stage: str, *, root: Path | None = None,
                assembly: dict[str, Any] | None = None) -> dict[str, Any]:
    """某档**逐件**要占多少 GiB ＋ **峰值** ✓（纯计算 ＋ 读文件头 ✓；**不装载** ✓✗）。

    返回 ``{"gpuGiB", "sumGiB", "parts", "peakParts", "mode", "modeWhy", "missing", "note"}`` ✓。
    ``missing`` ＝ 该档声明要用、却**没挑到路径**的件 ✓（``tokenizer`` 不算 ✗：它不是权重 ✓）。
    """
    stage = str(stage or "h3")
    asm = dict(assembly) if assembly is not None else inventory_mod.assembly_for_stage(stage, root=root)

    paths: dict[str, str | None] = {
        "dit": str(asm["dit_path"]) if asm.get("dit_path") else None,
        "text": str(asm["text_weights"]) if asm.get("text_weights") else None,
        "vae": _vae_path(asm),
    }
    declared = list(STAGE_PART_SPECS.get(stage, {}).keys())
    if not declared:
        #: 档不在 ``STAGE_PART_SPECS`` 里 ✓ ⇒ 只按"真有路径的件"算 ✓，并把这件事写进 note ✓
        #:（不默默按三件算 ✗ —— 那会让一个不认识的档凭空多出 20 GiB 需求 ✓✗）。
        declared = [part for part in ("dit", "text", "vae") if paths.get(part)]

    parts: dict[str, Any] = {}
    missing: list[str] = []
    for part in declared:
        path = paths.get(part)
        if path is None:
            missing.append(part)
        #: ⚠️ ``image``（SDXL）**没有逐件表** ✓ ⇒ 显式把它自己的文件字节当 fallback ✓
        #:（``stage_part_need`` 的 ``fallback=None`` 会去查表、查不到就抛 ✓ —— 那是**故意的** ✓）。
        fallback = _file_gib(path) if stage == "image" else None
        try:
            gib, why = stage_part_need(stage, part, path, fallback=fallback)
        except ValueError as err:
            #: 没有这本账 ✓ ⇒ **如实**给 0 ＋ 原因 ✓（编一个数会让 verdict 假绿 ✗✗）。
            gib, why = 0.0, {"source": "none", "mode": "no-table",
                             "why": f"这条线没有该件的驻留账 ⇒ 不估（{err}）"}
        parts[part] = {
            "title": PART_TITLES.get(part, part),
            "gib": round(float(gib), 2),
            "path": path,
            "source": why,
        }

    spec = STAGE_PEAK.get(stage)
    if spec is None:
        #: 没登记的档 ⇒ **只按 DiT 一件**算 ✓（保守**下限** ✓：报多了会假红、报少了会假绿 ✗，
        #: 单件是"至少这么多"的那个数 ✓），并把"没登记"写在 note 里 ✓（不装作算过 ✗）。
        spec = PeakSpec(("dit",), "sum", "single",
                        "⚠️ 该档没有登记峰值口径（STAGE_PEAK 缺行 ✓）⇒ 只按 DiT 一件算（下限 ✓）")
    values = [parts[part]["gib"] for part in spec.parts if part in parts]
    peak = sum(values) if spec.combine == "sum" else max(values, default=0.0)
    total = sum(item["gib"] for item in parts.values())

    return {
        "stage": stage,
        "gpuGiB": round(float(peak), 2),
        "sumGiB": round(float(total), 2),
        "parts": parts,
        "peakParts": list(spec.parts),
        "combine": spec.combine,
        "mode": spec.mode,
        "modeWhy": spec.why,
        "missing": missing,
        "note": str(asm.get("note") or ""),
        "assembly": {key: str(value) for key, value in asm.items()
                     if key not in ("note",) and isinstance(value, str)},
    }


def _ratio_verdict(ratio: float) -> str:
    if ratio >= PERFECT_RATIO:
        return "perfect"
    if ratio >= GOOD_RATIO:
        return "good"
    return "marginal"


def check(stage: str = "h3", *, root: Path | None = None, policy: str = "auto",
          capacity_gib: float | None = None,
          pressure: device_policy.Pressure | None = None,
          host: tuple[int, int] | None = None,
          assembly: dict[str, Any] | None = None) -> dict[str, Any]:
    """**这台机器能不能跑这条链** ✓（不装载 ✓、不 import torch 也能跑 ✓ —— 读数走 ``probe()`` ✓）。

    * ``capacity_gib``：假装卡只有这么大 ✓（"如果我买 16G 卡" ✓）——
      ⚠️ 这时**丢开**真读数 ✓、按**空卡**算 ✓，并在 ``have.assumed`` 标 True ✓✗；
    * ``pressure`` / ``host``：注入读数 ✓（测试与"先看方案"用 ✓）；
    * ``policy``：与 ``POST /load`` 同名的设备策略 ✓（``auto`` / ``gpu`` / ``cpu`` / ``balanced`` ✓）。

    回执结构 ✓见 :func:`describe_shape`（前端/CLI 只认这几个键 ✓）。
    """
    stage = str(stage or "h3")
    need = weight_need(stage, root=root, assembly=assembly)

    assumed = False
    if capacity_gib is not None and float(capacity_gib) > 0:
        total_bytes = int(float(capacity_gib) * GIB)
        pressure = device_policy.Pressure(
            totalBytes=total_bytes, freeBytes=total_bytes, allocatedBytes=0,
            utilPercent=0.0, source="assumed")
        assumed = True
    elif pressure is None:
        pressure = device_policy.probe()

    host_pair = host if host is not None else scheduler.host_memory()
    host_free = None if host_pair is None else float(host_pair[0]) / GIB
    host_total = None if host_pair is None else float(host_pair[1]) / GIB

    peak = float(need["gpuGiB"])
    decision = device_policy.decide(
        policy=policy, need_bytes=int(peak * GIB), pressure=pressure,
        gpu_available=pressure is not None)

    verdict, verdict_why = _judge(need=need, decision=decision, peak=peak,
                                  pressure=pressure, host_free=host_free)
    raw = verdict
    downgraded = False
    downgrade_why = ""
    if verdict == "perfect":
        verdict = DOWNGRADE["perfect"]
        downgraded = True
        downgrade_why = ("算出来是 ``perfect`` ✓，但**激活峰值没有表**（``basis=unknown`` ✓）"
                         "⇒ 按方案 A 降一档报 ``good`` ✓✗")

    advice: list[str] = []
    if need["missing"]:
        names = "、".join(PART_TITLES.get(part, part) for part in need["missing"])
        advice.append(f"缺件：{names} ⇒ 这条链现在**跑不了**（先补齐，或换一个档 ✓）")
    if decision.residency == "split":
        advice.append("卡上装不下整份（需 " + f"{peak:.2f}" + " GiB）⇒ 会走 **GPU/CPU 联合驻留** ✓；"
                      "⚠️ 那是兜底（``degraded`` ✓）不是优化 ✗ —— 想全上卡请换更小的量化档，或先腾卡 ✓")
    if decision.device == "cpu":
        advice.append("整条链在**内存**里跑 ✓：能出片，但比在卡上慢一个量级 ✓（不是失败 ✓）")
    if peak > 0 and pressure is not None and peak > float(pressure.totalBytes) / GIB:
        advice.append(f"需求 {peak:.2f} GiB **已超过卡总量** "
                      f"{float(pressure.totalBytes) / GIB:.2f} GiB ⇒ 只能联合或整份退内存 ✓")
    if verdict == "good":
        advice.append("余量够但不宽 ✓：采样工作区（本报告**没算**）也吃显存 ⇒ 想稳就留 ≥25% 空闲 ✓")

    return {
        "stage": stage,
        "policy": decision.policy,
        "verdict": verdict,
        "verdictRaw": raw,
        "downgraded": downgraded,
        "downgradeWhy": downgrade_why,
        "verdictWhy": verdict_why,
        "need": need,
        "have": {
            "gpuTotalGiB": round(float(pressure.totalBytes) / GIB, 2) if pressure is not None else None,
            "gpuFreeGiB": round(float(pressure.freeBytes) / GIB, 2) if pressure is not None else None,
            "gpuUsedRatio": round(float(pressure.usedRatio), 4) if pressure is not None else None,
            "hostFreeGiB": None if host_free is None else round(host_free, 2),
            "hostTotalGiB": None if host_total is None else round(host_total, 2),
            "assumed": assumed,
            "faked": bool(getattr(pressure, "faked", False)) if pressure is not None else False,
        },
        "decision": decision.to_dict(),
        "activation": {
            "basis": "unknown",
            "gib": None,
            "whatItMeans": ("本仓没有逐族「激活峰值」实测表 ⇒ **不给数** ✓。"
                            "``need.gpuGiB`` 只含**权重**；真跑还要加采样工作区"
                            "（随分辨率 / 分块解码 / 注意力实现而变 ✓）⇒ 结论请按「权重之外还有余量」读 ✓"),
        },
        "advice": advice,
    }


def estimate(*, stage: str = "h3", samples: list[float] | tuple[float, ...] | None = None,
             matched: int | None = None, kind: str = "",
             min_samples: int = ESTIMATE_MIN_SAMPLES) -> dict[str, Any]:
    """这条链**大概要多久** ✓ ⇒ ``{basis, runSeconds, loadSeconds, …}`` ✓（**只报有据的数** ✗ 不编 ✓）。

    ``samples`` ＝ 历史跑次的 ``elapsed_ms``（**同一 stage** ✓，由调用方从跑次台账取 ✓ ——
    引擎内核不读业务库 ✓，见 :mod:`app.services.engine` 的独立性约束 ✓）。本函数**纯计算** ✓。

    三处刻意报「不知道」的地方 ✓✗：

    * **样本不够** ⇒ ``basis="unknown"`` 且 ``runSeconds=None`` ✓。⚠️ 绝不给 ``0`` ✗✗：
      ``0 秒`` 会被读成"立刻完成"，比"不知道"错得更远 ✓；
    * **``loadSeconds`` 恒 ``None``** ✓：本仓只落库**整任务**耗时 ✗，没有单独的装载耗时 ✓ ——
      拿整任务耗时冒充装载秒数，会让"装载 40 秒"变成"装载 300 秒" ✓✗；
    * **``runSeconds`` 是整任务** ✓（含装载 / 采样 / 解码 / 落盘 ✓），**不是纯采样** ✓✗ ——
      这一格最容易被按"推理速度"读 ✓。

    返回的 ``matched`` 与 ``samples`` **必须分开** ✓（沿用台账口径 ✓）：``matched`` 是台账里
    命中多少条、``samples`` 是这次真拿到几条 ✓ —— 合成一个会把"抽样了"读成"就这么多" ✓✗。
    """
    values: list[float] = []
    for raw in samples or ():
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        #: ⚠️ 非正数（0 / 负数 / 坏值）一律丢 ✓：记 0 ms 的跑次多半是"没跑就失败了" ✓，
        #: 把它算进中位数会把整条链的估计拉向 0 ✓✗。
        if number > 0 and number == number and number != float("inf"):
            values.append(number)
    values.sort()
    count = len(values)
    hits = int(matched) if matched is not None else count

    base: dict[str, Any] = {
        "stage": str(stage),
        "kind": str(kind or ""),
        "samples": count,
        "matched": hits,
        "minSamples": int(min_samples),
        #: 口径写在回执里 ✓：读者不必去翻文档才知道这个秒数含不含装载 ✓
        "unit": "秒 · **整任务**耗时（装载＋采样＋解码＋落盘 ✓），不是纯采样 ✓",
        "loadSeconds": None,
        "loadWhy": ("本仓跑次台账只记**整任务**耗时 ✓、没有单独的装载耗时 ✗ ⇒ "
                    "不拆一个数出来 ✓（拿整任务耗时冒充装载秒数会把结论说反 ✓✗）"),
    }

    if count < max(1, int(min_samples)):
        base["basis"] = "unknown"
        base["runSeconds"] = None
        base["note"] = ("还没有足够的同族成功跑次 ⇒ **不给秒数** ✓（``0 秒`` 会被读成「立刻完成」✗✗）"
                        if count == 0 else
                        f"只有 {count} 条样本（闸门 {int(min_samples)} 条 ✓）⇒ **不给秒数** ✓："
                        "一条样本会被首次装载 / 冷缓存 / 别的进程抢卡拉长两三倍 ✓✗")
        base["whatToDo"] = ("先真跑几次（每次都是同一条链 ✓），再回来看这一格 ✓；"
                            "或直接看 ``verdict`` / ``advice``——那两个不需要跑次 ✓")
        return base

    def _at(fraction: float) -> float:
        """最近秩百分位 ✓（样本少时插值会造出一个**没出现过**的数 ✗）。"""
        index = int(round(fraction * (count - 1)))
        return values[max(0, min(count - 1, index))]

    base["basis"] = "measured"
    base["runSeconds"] = {
        "p50": round(_at(0.50) / 1000.0, 2),
        "p95": round(_at(0.95) / 1000.0, 2),
        "min": round(values[0] / 1000.0, 2),
        "max": round(values[-1] / 1000.0, 2),
        "mean": round(sum(values) / count / 1000.0, 2),
    }
    base["note"] = f"按 {count} 条同族跑次报 ✓（``p50`` 用来排期、``p95`` 用来估最坏 ✓）"
    if hits > count:
        base["note"] += (f"；⚠️ 台账命中 {hits} 条、只取了 {count} 条 ⇒ "
                         "这是**抽样** ✓ 不是「就这么多」✗（区间会更宽 ✓）")
    return base


def _judge(*, need: dict[str, Any], decision: device_policy.Decision, peak: float,
           pressure: device_policy.Pressure | None,
           host_free: float | None) -> tuple[str, str]:
    """按读数下判 ✓ ⇒ ``(verdict, 人话依据)`` ✓（阈值见 :data:`PERFECT_RATIO` / :data:`GOOD_RATIO` ✓）。"""
    if need["missing"]:
        return "too_tight", f"缺件 {len(need['missing'])} 件 ⇒ 装不起来（不是显存问题 ✓）"
    if peak <= 0:
        return "too_tight", "连权重要多少都算不出来（没有可用的账 ✓）⇒ 不能下「能跑」的结论 ✓✗"

    if decision.device == "cpu":
        if host_free is None:
            return "marginal", ("量不到内存读数 ✓ ⇒ 只能按「整份退 CPU」的经验判"
                                "（⚠️ 这不是通过 ✗）")
        if host_free < peak + 2.0:
            return "too_tight", (f"整份要进内存（{peak:.2f} GiB ✓）而可用物理内存只有 "
                                 f"{host_free:.2f} GiB ✓ ⇒ 会进交换区 ✓✗")
        return "marginal", (f"整份在内存里跑（{peak:.2f} GiB ✓，可用 {host_free:.2f} GiB ✓）"
                            "⇒ 能跑但慢 ✓")

    if pressure is None:
        return "marginal", "测不到卡况 ✓ ⇒ 按默认优先 GPU 判（⚠️ 没有依据 ✓）"

    free = float(pressure.freeBytes) / GIB
    if decision.residency in ("split", "park"):
        #: ⚠️ 联合驻留**也吃内存** ✓✗：卡上放不下的那部分要落到物理内存里 ✓ ——
        #: 只看卡就报 ``marginal`` 会在「8 GiB 卡 ＋ 32 GB 内存」这种低配机上假绿 ✗
        #:（split 要 ``peak - free`` 那么多内存，比"整份进内存"少了、但不是零 ✓）。
        spill = max(peak - free, 0.0)
        if host_free is not None and host_free < spill + 1.0:
            return "too_tight", (f"卡上只放得下 {free:.2f} GiB、剩下 {spill:.2f} GiB 要进内存 ✓，"
                                 f"而可用内存只有 {host_free:.2f} GiB ✓ ⇒ 联合驻留也装不下 ✓✗")
        host_note = "" if host_free is None else f"，内存余 {host_free:.2f} GiB 够放这 {spill:.2f} GiB ✓"
        return "marginal", (f"卡上余 {free:.2f} GiB < 需 {peak:.2f} GiB ＋ 1 GiB 余量 ✓ ⇒ "
                            f"只能联合驻留 ✓（``degraded`` ✓{host_note}）")
    ratio = free / peak if peak > 0 else 0.0
    verdict = _ratio_verdict(ratio)
    return verdict, (f"卡上余 {free:.2f} GiB ÷ 权重峰值 {peak:.2f} GiB = {ratio:.2f} 倍 ✓"
                     f"（``perfect`` ≥{PERFECT_RATIO} ✓、``good`` ≥{GOOD_RATIO} ✓）⇒ {verdict} ✓")


def describe_shape() -> dict[str, Any]:
    """回执的**键清单** ✓（前端/CLI 只认这几个 ✓；改成这里的形状就必须同步改测试 ✓）。"""
    return {
        "top": ["stage", "policy", "verdict", "verdictRaw", "downgraded", "downgradeWhy",
                "verdictWhy", "need", "have", "decision", "activation", "advice"],
        "need": ["stage", "gpuGiB", "sumGiB", "parts", "peakParts", "combine", "mode",
                 "modeWhy", "missing", "note", "assembly"],
        "have": ["gpuTotalGiB", "gpuFreeGiB", "gpuUsedRatio", "hostFreeGiB", "hostTotalGiB",
                 "assumed", "faked"],
        "activation": ["basis", "gib", "whatItMeans"],
        "verdicts": list(VERDICTS),
    }


def selftest() -> list[str]:
    """本模块自检 ✓（纯逻辑 ＋ 临时文件 ✓，不碰 torch ✓）⇒ 返回问题清单 ✓（空 ⇒ 全绿 ✓）。"""
    problems: list[str] = []

    def _fake(gib: float) -> device_policy.Pressure:
        return device_policy.Pressure(totalBytes=int(22.49 * GIB), freeBytes=int(gib * GIB))

    # ① 表里的档必须都在 STAGE_PEAK 里（漏一行 ⇒ check() 会退回"只算 DiT"假绿）
    for stage in STAGE_PART_SPECS:
        if stage not in STAGE_PEAK:
            problems.append(f"档 {stage!r} 在 STAGE_PART_SPECS 里、却没有峰值口径（STAGE_PEAK 缺行）")

    #: ⚠️ 路径故意指向**不存在**的文件 ✓ —— 走 ``part_need_gib`` 的「读不到头部 ⇒ 按表」那条路 ✓
    #:（这正是没装模型时的场景 ✓，而不是"缺件" ✗：缺件另有 ``missing`` 那条判据 ✓）。
    fake_assembly = {"stage": "krea2",
                     "dit_path": "no_such_dir/krea2_int8.safetensors",
                     "text_weights": "no_such_dir/qwen3vl4b.safetensors",
                     "image_vae_weights": "no_such_dir/vae.safetensors"}

    # ② 大把余量 ⇒ 只许报 good（perfect 必降档）
    report = check("krea2", pressure=_fake(22.0), host=(90 * GIB, 96 * GIB),
                   assembly=fake_assembly)
    if report["verdict"] != "good" or not report["downgraded"]:
        problems.append(f"余量充足时应报 good 且标降档，实际 {report['verdict']} / "
                        f"downgraded={report['downgraded']}")

    # ③ 卡上空、但内存够 ⇒ 整份退 CPU ⇒ marginal（不是失败）
    report = check("krea2", pressure=_fake(0.2), host=(90 * GIB, 96 * GIB),
                   assembly=fake_assembly)
    if report["verdict"] != "marginal" or report["decision"]["device"] != "cpu":
        problems.append(f"卡上空、内存够 ⇒ 应整份退 CPU 并报 marginal，实际 "
                        f"{report['verdict']} / device={report['decision']['device']}")

    # ④ 卡上空、内存也不够 ⇒ too_tight
    report = check("krea2", pressure=_fake(0.2), host=(1 * GIB, 16 * GIB),
                   assembly=fake_assembly)
    if report["verdict"] != "too_tight":
        problems.append(f"卡与内存都放不下时应报 too_tight，实际 {report['verdict']}")

    # ⑤ 激活不许编数
    if report["activation"]["basis"] != "unknown" or report["activation"]["gib"] is not None:
        problems.append("activation 必须 basis=unknown 且不给数（方案 A）")

    # ⑥ 同驻 / 换装两种口径必须真的算出不同的数（写错了 combine 就一样了）
    need_chain = weight_need("zimage", assembly=fake_assembly)
    if need_chain["combine"] != "sum" or need_chain["gpuGiB"] <= need_chain["parts"]["dit"]["gib"]:
        problems.append("zimage 是整链同驻 ⇒ 峰值必须大于单件 DiT")
    need_phased = weight_need("qwen_image21", assembly=fake_assembly)
    if need_phased["combine"] != "max":
        problems.append("qwen_image21 是分时换装 ⇒ 峰值口径应为 max")
    return problems


if __name__ == "__main__":  # pragma: no cover - 手工自检入口
    import sys

    found = selftest()
    print("\n".join(found) if found else "feasibility: OK")
    sys.exit(1 if found else 0)
