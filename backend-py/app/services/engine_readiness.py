r"""**自研引擎就绪体检**（服务层 ✓ 2026-09-21）—— 把四块串成**一次调用** ✓ 并给**可行动的下一步** ✓。

## 为什么从脚本搬到这里 ✗

第一版它是 `app/scripts/h3_readiness.py` ✓✗ —— 但 `app/scripts/` **不是包**（没有 `__init__.py` ✓）
⇒ **路由引用不到它** ✗（`app.scripts.h3_readiness` 导不进来 ✓）⇒ 结果就是「只有人手敲命令才能用」✗✗
（本仓那条判据：**没人调用的能力不算功能** ✓）。现在逻辑在这一层 ✓：

* 路由 :mod:`app.routers.preflight` 的 ``GET /production/engine-readiness`` 用它 ✓（前端可显示 ✓）；
* CLI `app/scripts/h3_readiness.py` 只是**薄壳** ✓（渲染与参数解析 ✓ 业务不重复 ✗）。

## 四块（顺序就是实际顺序 ✓）

1. **环境** ✓：必需依赖（缺 ⇒ 给 `pip install` ✓）/ 可选依赖 / `ffmpeg` ✓；
2. **权重就绪** ✓：:func:`inventory.readiness` + :func:`loader.plan_stage`（缺哪些 ✓ 峰值/fits ✓）；
3. **真权重预检** ✓（**只在给了路径时** ✓）：:func:`h3_keys.audit_h3_checkpoint`
   → :func:`h3_keys.infer_h3_trunk_config` →（可选）:func:`tokenizer_hub.load` ✓；
4. **结论** ✓：``ready`` + ``blockers`` + ``unchecked`` + ``nextSteps``（按成本排序 ✓）。

⚠️⚠️ 两条口径（都会误导，所以钉死 ✓）：

* **没给权重 ⇒ 那一项进 ``unchecked``** ✓（「**没查**」✗ **不是「通过」** ✓ —— 无从体检 ≠ 绿灯 ✓）；
* 「还要下多少」按**清单的 ``expectedGiB``** 求和 ✓ ✗ —— 用 `bytes` 求和会**恒为 0** ✓✗
  （缺文件时 `bytes` 是 0 ✓ —— 第一版就这么打出「还差约 0.00 GiB」✓✗，明明一个都没下 ✓）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import h3_keys
from .engine import inventory as inv
from .engine import loader as loader_mod
from .engine import media as media_mod

__all__ = ["collect", "precheck_weights", "quant_plan", "summary", "next_steps", "planned_gib"]


def device_facts() -> dict[str, Any]:
    """本机**实测**设备 ✓（设备名 / 显存总量 ✓）—— ⚠️ **只报实测** ✗：缺 torch 就报 ``None`` ✓，
    **绝不**写死"本机无 NVIDIA 卡"这类断言 ✓✗（本仓在 A5000 工作站上也跑 ✓：写死的断言会把
    「现在就能真跑」说成「得换机器」✗）。
    """
    from .engine import torch_backend as tb  # noqa: PLC0415 —— 局部引 ✓：它模块级不碰 torch ✓

    available, _ = tb.torch_available()
    if not available:
        return {"device": None, "cudaAvailable": False, "deviceName": None, "totalGiB": None}
    described = tb.TorchBackend().describe()          # 轻 ✓：只探设备 ✓ 不装权重 ✓
    facts: dict[str, Any] = {"device": described.get("device"),
                             "cudaAvailable": described.get("cudaAvailable"),
                             "deviceName": None, "totalGiB": None}
    if facts["cudaAvailable"]:
        try:
            import torch  # noqa: PLC0415 —— 只有真在 CUDA 上才需要它 ✓

            facts["deviceName"] = torch.cuda.get_device_name(0)
            facts["totalGiB"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        except Exception:  # noqa: BLE001 —— 问不到名字不影响"有没有卡"这个结论 ✓
            pass
    return facts


def environment_report() -> dict[str, Any]:
    """① 环境 ✓（**不导入**重库 ✓ ⇒ 秒级 ✓）。"""
    from .engine import torch_backend as tb  # noqa: PLC0415 —— 局部引 ✓：它模块级不碰 torch ✓

    available, reason = tb.torch_available()
    status = tb.dependency_status()
    return {
        "torchAvailable": available,
        "torchReason": reason,
        "missingRequired": status["missing"],
        "optionalMissing": status.get("optionalMissing", []),
        "installRequired": status["install"],
        "installOptional": status.get("installOptional", []),
        "ffmpeg": media_mod.have_ffmpeg(),
        "ffmpegVersion": media_mod.ffmpeg_version(),
        # ⭐ 设备**实测** ✓（2026-09-26 加 ✓）：没有这一项，报告就只能靠"猜本机有没有卡" ✓✗
        "device": device_facts(),
    }


def planned_gib(ready: dict[str, Any], *, only_missing: bool = False) -> float:
    """按**清单的 `expectedGiB`** 求和 ✓（``only_missing`` ⇒ 只加没下的 ✓ = 实际要下载多少 ✓）。

    ⚠️ **别用 `bytes`** ✗：缺文件时它是 0 ✓✗（见模块头那条口径 ✓）。
    """
    total = 0.0
    for item in ready.get("components") or []:
        if not item.get("required"):
            continue
        if only_missing and item.get("present"):
            continue
        total += float(item.get("expectedGiB") or 0.0)
    return total


def quant_plan(load_plan: dict[str, Any]) -> dict[str, Any]:
    """**低精度权重能不能自动还原** ✓（按组件 ✓ 结论取自 `loader.plan_stage` 的 `dequantPlan` ✓）。

    ⚠️ 三种状态**分开说** ✗（本仓那条口径：**没查 ≠ 通过** ✓）：

    * ``supported`` ✓：该组件里有低精度权重、且布局判得出来 ⇒ 附**布局计数** ✓；
    * ``unsupported`` ✗：有低精度权重但**判不出来**（块量化 / 分组量化 / 缺 scale ✓）⇒ 装载时会被**中止** ✓
      ⇒ 逐条点名 ✓（⚠️ GGUF **豁免** ✗：它的反量化由运行时做 ✓ 见 `dequantPlan["note"]` ✓）；
    * ``notChecked`` ✓：组件**没下载** ⇒ `dequantPlan` 是空的 `{}` ✓ ⇒ 记这里 ✓（不是「通过」✗）。
    """
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    not_checked: list[str] = []
    for component in load_plan.get("components") or []:
        key = str(component.get("key") or "")
        detail = component.get("dequantPlan") or {}
        if not detail:
            if not component.get("present"):
                not_checked.append(key)
            continue
        if detail.get("note"):
            continue                                    # GGUF：反量化归运行时 ✓ 不算「装不上」✗
        if detail.get("supported"):
            supported.append({"key": key, "weights": detail.get("weights"),
                              "layouts": detail.get("layouts")})
        elif detail.get("weights"):
            unsupported.append({"key": key, "weights": detail.get("weights"),
                                "unresolved": detail.get("unresolved"),
                                "unpaired": detail.get("unpaired"),
                                "grouped": detail.get("grouped")})
    return {"supported": supported, "unsupported": unsupported, "notChecked": not_checked}


def precheck_weights(weights: str | None, tokenizer: str | None) -> dict[str, Any] | None:
    """③ 真权重预检 ✓（没给路径 ⇒ ``None`` ✓ —— **不是**"通过" ✗）。"""
    if not weights:
        return None
    from .engine import gguf as gguf_mod  # noqa: PLC0415
    from .engine import safetensors as st  # noqa: PLC0415

    target = Path(weights)
    report: dict[str, Any] = {"path": str(target)}
    if not target.exists():
        report["error"] = "文件不存在 ✗"
        return report
    is_gguf = target.suffix.lower() == ".gguf"
    info = gguf_mod.inspect(target) if is_gguf else st.inspect(target)
    report["format"] = "gguf" if is_gguf else "safetensors"
    report["structureOk"] = bool(getattr(info, "ok", False))
    report["structureProblems"] = list(getattr(info, "problems", []) or [])
    report["tensorCount"] = int(getattr(info, "tensor_count", 0) or 0)
    shapes = {name: tensor.shape for name, tensor in getattr(info, "tensors", {}).items()}
    if not shapes:
        report["note"] = "读不到张量表 ⇒ 预检到此为止 ✓（结构问题见上 ✓）"
        return report
    report["audit"] = h3_keys.audit_h3_checkpoint(shapes).to_dict()
    inferred = h3_keys.infer_h3_trunk_config(shapes)
    report["trunkConfig"] = inferred.config
    report["configSources"] = inferred.sources
    report["configOk"] = inferred.ok
    report["configProblems"] = inferred.problems
    if tokenizer:
        try:
            from .engine import tokenizer_hub  # noqa: PLC0415

            hub = tokenizer_hub.load(tokenizer_hub.HubConfig(path=str(tokenizer)))
            report["tokenizer"] = hub.describe()
            report["tokenizerFingerprint"] = hub.fingerprint()
        except Exception as err:  # noqa: BLE001 —— 词表问题也是**结论** ✓ 不抛 ✗
            report["tokenizerError"] = f"{type(err).__name__}: {err}"
    return report


def next_steps(report: dict[str, Any]) -> list[str]:
    """「下一步做什么」✓（**按成本排序** ✓ 便宜的先 ✓ —— 与生产体检同一口径 ✓）。"""
    steps: list[str] = []
    for command in report["environment"].get("installRequired") or []:
        steps.append(f"装必需依赖：{command} ✓")
    for command in report["environment"].get("installOptional") or []:
        steps.append(f"（可选）装参考实现：{command} ✓")
    if not report["environment"]["ffmpeg"]:
        steps.append("装 `ffmpeg` 并确保在 PATH 里 ✓")
    missing = report["weightsReadiness"].get("missingRequired") or []
    if missing:
        models_dir = report["weightsReadiness"].get("modelsDir")
        steps.append(f"下权重到 `{models_dir}` ✓：缺 {missing} ✓"
                     f"（还要下约 {planned_gib(report['weightsReadiness'], only_missing=True):.2f} GiB ✓；"
                     f"全量约 {planned_gib(report['weightsReadiness']):.2f} GiB ✓）")
    check = report.get("weightsCheck")
    if check is None and not missing:
        steps.append("权重已在盘上 ⇒ 加 `--weights <DiT 路径> [--tokenizer <词表目录>]` "
                     "跑**真权重预检** ✓")
    for item in (report.get("quantPlan") or {}).get("unsupported") or []:
        steps.append(f"`{item['key']}` 的低精度权重**要随附 json 的 group size** 才能反量化 ✓"
                     f"（判不出：{item['unresolved'] or item['grouped'] or item['unpaired']} ✓ "
                     f"⇒ 本仓不猜 ✗；先换成**非分组**量化的重导出 ✓ 或把 group size 给出来 ✓）")
    if isinstance(check, dict) and (check.get("audit") or {}).get("ok"):
        # ⚠️ 下一步说什么，**要看本机实测设备** ✗ —— 不许再写死"本机无 NVIDIA 卡" ✓✗
        #    （本仓在 A5000 工作站上也跑 ✓：2026-09-26 实测 `device=cuda` ✓ 22.5 GiB ✓ ⇒
        #     写死的断言会把"能真跑"说成"得换机器" ✓✗ —— 判据一律**实测** ✓）
        device = (report["environment"].get("device") or {})
        where = str(device.get("device") or "?")
        if where == "cuda":
            steps.append(
                f"⭐ 权重核对通过 ✓ ⇒ 本机就是 CUDA（{device.get('deviceName') or '?'} "
                f"{device.get('totalGiB') or '?'} GiB ✓）⇒ **现在就能真跑** ✓："
                f"`POST /api/v1/engine/load` ✓ → `POST /api/v1/engine/generate` ✓"
                f"（或 `TorchBackend.load_weights(path=…)` → `pipeline.run_sync(...)` ✓）；"
                f"⚠️ 单靠一次 simple 装载装不下 39.55 GiB 全量 ✗ ⇒ 大权重走 hybrid 计划 ✓")
        else:
            steps.append("⭐ 权重核对通过 ✓ ⇒ 下一步：`TorchBackend.load_weights(path=…)` "
                         "→ `pipeline.run_sync(...)` ✓"
                         f"（本机实测 device={where} ✓ —— CPU 上能跑但很慢 ✗）")
    return steps


def collect(*, weights: str | None = None, tokenizer: str | None = None,
            stage: str = "h3", capacity_gib: float = inv.DEFAULT_CAPACITY_GIB) -> dict[str, Any]:
    """把四块串成**一次调用** ✓（纯函数 ✓ 不打印 ✓ —— 路由与 CLI 都调它 ✓）。"""
    env = environment_report()
    ready = inv.readiness(stage, capacity_gib=capacity_gib)
    plan = loader_mod.plan_stage(stage, capacity_gib=capacity_gib)
    report: dict[str, Any] = {
        "stage": stage,
        "environment": env,
        "weightsReadiness": ready,
        "loadPlan": plan,
        "weightsCheck": precheck_weights(weights, tokenizer),
    }
    blockers: list[str] = []
    if env["missingRequired"]:
        blockers.append(f"缺必需依赖：{', '.join(env['missingRequired'])} ✓ "
                        f"⇒ {env['installRequired'][0] if env['installRequired'] else ''}")
    if not env["ffmpeg"]:
        blockers.append("缺 `ffmpeg` ✗ ⇒ 出不了 mp4 ✓（装它 ✓）")
    if not ready["ready"]:
        blockers.append(f"权重没齐 ✓：缺 {ready['missingRequired']} ✓"
                        f"（还要下约 {planned_gib(ready, only_missing=True):.2f} GiB ✓）")
    check = report["weightsCheck"]
    if isinstance(check, dict):
        if check.get("error"):
            blockers.append(f"给了权重但读不到：{check['error']} ✓")
        elif (check.get("audit") or {}).get("ok") is False:
            blockers.append("权重键名/形状核对**没通过** ✓ ⇒ 装不进去 ✓（细节见 `weightsCheck.audit` ✓）")
        if check.get("tokenizerError"):
            blockers.append(f"词表装不上：{check['tokenizerError']} ✓")
    # ⭐ 低精度权重**能不能自动还原** ✓：在盘上、但布局判不出来 ⇒ 装载会被**中止** ✗ ⇒ 也是阻塞 ✓
    #    ⚠️ 组件**没下载** ⇒ 记 `notChecked` ✓（**没查 ≠ 通过** ✗ —— 不阻塞、但也不假装绿 ✓）
    report["quantPlan"] = quant_plan(plan)
    for item in report["quantPlan"]["unsupported"]:
        blockers.append(
            f"`{item['key']}` 的低精度权重**判不出反量化布局** ✗ ⇒ 装载会中止 ✓"
            f"（判不出：{item['unresolved'] or item['grouped'] or item['unpaired']} ✓）"
            f"⇒ 需要随附 json 里的 group size ✓ 本仓不猜 ✗")
    report["ready"] = bool(not blockers)
    report["blockers"] = blockers
    report["unchecked"] = ([] if check is not None else
                           ["**没做**真权重预检 ✓（没给权重路径 ✓ ⇒ 这一项是**没查** ✓ "
                            "不是「通过」 ✗）"])
    report["nextSteps"] = next_steps(report)
    return report


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """**路由用**的精简版 ✓ —— ⚠️ 别把整份计划塞进响应 ✗（几十个组件 + 逐步排班 ✓ 太大 ✗）。"""
    ready = report["weightsReadiness"]
    residency = report["loadPlan"]["residency"]
    return {
        "stage": report["stage"],
        "ready": report["ready"],
        "blockers": report["blockers"],
        "unchecked": report["unchecked"],
        "nextSteps": report["nextSteps"],
        "environment": {
            "torchAvailable": report["environment"]["torchAvailable"],
            "torchReason": report["environment"]["torchReason"],
            "missingRequired": report["environment"]["missingRequired"],
            "optionalMissing": report["environment"]["optionalMissing"],
            "ffmpeg": report["environment"]["ffmpeg"],
        },
        "weights": {
            "modelsDir": ready.get("modelsDir"),
            "ready": ready.get("ready"),
            "missingRequired": ready.get("missingRequired"),
            "missingOptional": ready.get("optionalMissing"),
            "suspect": ready.get("suspectRequired"),
            "plannedGiB": round(planned_gib(ready), 2),
            "downloadedGiB": round(float(ready.get("requiredWeightsGiB") or 0.0), 2),
            "remainingGiB": round(planned_gib(ready, only_missing=True), 2),
        },
        "vram": {
            "capacityGiB": residency.get("capacityGiB"),
            "peakResidentGiB": residency.get("peakResidentGiB"),
            "fits": residency.get("fits"),
            "strategy": residency.get("strategy"),
            "note": residency.get("note"),
        },
        #: ⭐ 低精度（fp8/int8）权重**能不能自动还原** ✓ —— 三档分开报 ✗（支持 / 不支持 / **没查** ✓）
        "quant": {
            "supported": report["quantPlan"]["supported"],
            "unsupported": report["quantPlan"]["unsupported"],
            "notChecked": report["quantPlan"]["notChecked"],
        },
        #: ⚠️ 真权重预检**要文件路径** ✗ ⇒ 不走 HTTP ✗（那等于开放任意路径读取 ✓✗）
        #: ⇒ 说明清楚「要更深的检查请用 CLI」✓（可行动 ✓ 而不是留个空白 ✓）。
        "hint": "本端点只看**盘上**就绪情况 ✓；要给**具体权重文件**做键名/形状核对 ⇒ "
                "跑 `python app/scripts/h3_readiness.py --weights <路径> [--tokenizer <词表目录>]` ✓",
    }
