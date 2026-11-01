r"""**引擎的独立门面（SDK 层）** —— 不经过 HTTP、不经过路由、不经过项目数据。

## 为什么要有这一层

`engine/` 里已经有真本事（装载 / 采样 / 解码 / 量化还原 / 权重核对），但**怎么用**只写在调用方里：
路由 `app.routers.engine` 知道要 `loader.plan_stage` → `runtime.engine_runtime`，体检脚本知道
要 `inventory.readiness` + `engine_readiness.collect`。也就是**能力齐了、门面没有**，后果三条：

1. **换消费者就重写一遍**：想在进程内跑一次体检，得把路由里的调用顺序再抄一次 —— 抄错的版本
   一定会漂（判据两处写就必然不一致）；
2. **没法独立验收**：引擎的结论只能靠「起整个后端 + 调 HTTP」拿到，于是「引擎自己行不行」
   永远混着 Web 层的问题；
3. **能力边界说不清**：没有一处能回答「现在能做什么、不能做什么、哪些**没查**」。

⇒ 本模块给出稳定契约（对应 WBS-1.2 / 1.4 / 2.1）：版本号、错误码、能力声明、计划、就绪、
权重核对、会话式生命周期。

## 三条硬约束（「独立性」就落在这三条上）

1. **不碰 Web 层**：顶层不 import `fastapi` / `pydantic` / `starlette` / `app.routers` ——
   由 `tests/engine_independence_test.py` 用 AST 静态断言（靠人记会漂）；
2. **不碰项目数据**：不读剧本 / 分镜 / 体检报告，只认权重清单与权重文件（`ffmpeg`、pip 建议
   那些属项目侧编排，留在 `app.services.engine_readiness`）；
3. **不 import torch 也能回答**：能力 / 计划 / 权重核对 / 就绪全部只靠文件头。

## 与 WBS 的对应

| WBS | 落点 |
|---|---|
| 1.2 领域模型 / 错误码 | `EngineError` + `CODES` |
| 1.4 SDK（不依赖 HTTP） | `Engine` / `plan` / `readiness` |
| 2.1 适配器契约（版本 / 能力 / 生命周期 / 错误语义） | `API_VERSION` + `capabilities` + `Engine.load/submit/release` |
| 2.6 权重清单（来源 / 哈希 / 格式 / 版本 / 许可） | `manifest_of` |
| G2 前置「不兼容版本在任务启动前报可读错误」 | `plan` / `readiness` 的 `blockers` |
| G2 前置「损坏权重不能被误判为有效模型」 | `inspect_weights` ⇒ `E_WEIGHT_CORRUPT` |
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

__all__ = ["API_VERSION", "CODES", "Engine", "EngineError", "SCHEMA_VERSION",
           "capabilities", "compat", "inspect_weights", "manifest_of", "plan",
           "readiness", "translate"]

#: 门面 API 版本（语义化：MAJOR＝不兼容变化 / MINOR＝新增可选能力 / PATCH＝兼容修复）。
API_VERSION = "1.0.0"

#: 报告结构版本（与 API_VERSION 分开：同一份 API 可以只改报告字段）。
#: ⚠️ 2 = `status().cache` 新增（E3.4 ✓，2026-10-29 ✓）—— 顶层**加了一个键**就是破坏性变更 ✓✗：
#: 老调用方按「字段集合相等」校验会红 ✓（这正是 `engine_router_contract_test` ⑨ 的用法 ✓）。
#: 3 = `status().lora` 新增（E2.5 ✓，2026-10-30 ✓）—— 同上 ✓。
#: 4 = `capabilities().devicePolicy` 新增（2026-10-31 ✓）—— 同上 ✓✗。这一栏补的是
#:     「**凭什么**把这条链放在这台设备上」✓：此前只有各条的 `device`（结论 ✓）⇒ 一次
#:     显存降级会被读成「这台机器不支持 GPU」✓✗（读的人只能猜 ✓）。
SCHEMA_VERSION = 4

#: 错误码（WBS-1.2：机器可判，不是只有一句中文）。
CODES: dict[str, str] = {
    "E_ENGINE_UNAVAILABLE": "引擎后端不可用（缺依赖 / 缺 torch / 缺 CUDA，见 detail.reason）",
    "E_ENGINE_BUSY": "引擎忙（已有任务在跑，见 hint）",
    "E_BAD_CONFIG": "配置/参数非法（阶段名、容量、路径形状）",
    "E_UNSUPPORTED_QUANT": "权重带本仓未实现的量化族（mxfp4 等；nvfp4 自 2026-10-30 起已支持）",
    "E_WEIGHT_MISSING": "权重文件不存在（清单落点与探测根都没找到）",
    "E_WEIGHT_CORRUPT": "权重文件结构坏（截断 / 头部非法 / dtype 不认识）",
    "E_VRAM_EXHAUSTED": "显存不够（权重或激活装不下，见 hint 里的三条路）",
    "E_TASK_FAILED": "任务失败（引擎内部异常，见 detail.error）",
}


class EngineError(Exception):
    """引擎的统一错误：机器可判的 `code` + 给人看的 `hint`。

    ⚠️ **不替换**原有异常（`TorchBackendUnavailable` / `EngineUnavailable` / `EngineBusy` /
    `SafetensorsError` 都还在）：门面只在**边界**上翻译，库内部继续用自己的类型 —— 否则四处
    `except` 全要改，那不是独立化，是制造回归。
    """

    def __init__(self, message: str, *, code: str = "E_ENGINE_UNAVAILABLE",
                 hint: str = "", detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code if code in CODES else "E_ENGINE_UNAVAILABLE"
        self.hint = hint
        self.detail = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "hint": self.hint, "detail": dict(self.detail)}


def _friendly(err: BaseException, *, fallback: str = "E_ENGINE_UNAVAILABLE") -> EngineError:
    """把库内异常**按类型**翻成错误码（不看消息文本 —— 文案一改映射就失效）。"""
    from . import safetensors as st  # noqa: PLC0415
    from . import torch_backend as tb  # noqa: PLC0415

    name = type(err).__name__
    text = str(err) or name
    if isinstance(err, EngineError):
        return err
    #: 自带 ``code`` ＋ ``hint`` ＋ ``detail`` 的异常原样翻（`registry.RegistryError` /
    #: `CompatError` 就是这套，E2.1/E2.2）—— ⚠️ 必须在下面那串 isinstance 之前 ✗✗：
    #: ``RegistryError`` 是 ``ValueError`` 的子类 ⇒ 落到最后会被翻成 ``E_BAD_CONFIG``，
    #: 「这张卡装不下」就变成「参数非法」✗（`engine_registry_test` 钉这条）。
    carried = getattr(err, "code", None)
    if isinstance(carried, str) and carried in CODES:
        return EngineError(text, code=carried, hint=str(getattr(err, "hint", "") or ""),
                           detail=dict(getattr(err, "detail", {}) or {}))
    if name == "EngineBusy":
        return EngineError(text, code="E_ENGINE_BUSY", hint="等当前任务结束，或先 cancel 它")
    # ⚠️ 显存不够**必须单独成码** ✗✗ —— 否则落到下面的 ``fallback``（＝ ``E_TASK_FAILED``
    #    「引擎内部异常」✗）：那是本仓最恨的误导 ✓✗，用户会照着「内部异常」去查实现，
    #    而真相是「这张卡上这一档装不下」✓、改一处配置就能跑 ✓（同一个口径：
    #    :func:`pipeline._fail` 里也是把 ``OutOfMemoryError`` 单独点名 ✓，两处都覆盖
    #    —— 装载/plan 阶段走这里 ✓、采样阶段走那里 ✓，一条路都不许漏 ✓）。
    #    判据同样用**类名**、不解析英文报错 ✓（torch 大版本升级会改文案 ✓✗）。
    if "OutOfMemory" in name:
        from . import registry as registry_mod  # noqa: PLC0415

        return EngineError(text, code="E_VRAM_EXHAUSTED", hint=registry_mod.VRAM_HINT)
    if isinstance(err, tb.TorchBackendUnavailable) or name == "EngineUnavailable":
        return EngineError(text, code="E_ENGINE_UNAVAILABLE",
                           hint="先跑 python -m app.services.engine doctor 看缺什么")
    if isinstance(err, st.SafetensorsError):
        missing = "不存在" in text or "No such file" in text
        return EngineError(text, code="E_WEIGHT_MISSING" if missing else "E_WEIGHT_CORRUPT",
                           hint="核对路径，或重新下载该权重")
    if isinstance(err, (ValueError, TypeError, KeyError)):
        return EngineError(text, code="E_BAD_CONFIG", hint="核对参数名与取值")
    return EngineError(text, code=fallback)


def translate(err: BaseException, *, fallback: str = "E_ENGINE_UNAVAILABLE") -> EngineError:
    """**公开**的「异常 ⇒ 错误码」入口 —— Web 层唯一该用的那个 ✓。

    ⚠️ 为什么必须公开 ✗：router 直接调 `runtime` 时拿到的是 ``runtime.EngineBusy`` ✓，它**不是**
    门面异常 ✓；若在 router 里手写 ``"E_ENGINE_BUSY"`` ⇒ 错误码就有了第二个真源 ✓✗（E1.5 的
    判据正是盯这个 ✓）。转发一层不增加逻辑，只是把「翻译规则」留在门面里 ✓。
    """
    return _friendly(err, fallback=fallback)


def capabilities() -> dict[str, Any]:
    """引擎**现在能做什么**：零副作用、不 import torch、不读权重（毫秒级）。

    ⚠️ `unverified` 单独一栏：模型能装、能跑通一次，和「有性能基线 / 有回归」是两件事 ——
    混成一个 `supported` 就是许空头（WBS §12 硬/软阈值口径）。
    """
    from . import inventory as inv  # noqa: PLC0415
    from . import quant  # noqa: PLC0415
    from . import torch_backend as tb  # noqa: PLC0415
    from . import device_policy as policy_mod  # noqa: PLC0415

    available, reason = tb.torch_available()
    status = tb.dependency_status()
    catalog = inv.load_catalog()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "apiVersion": API_VERSION,
        "engine": "darme-engine (in-process)",
        "python": platform.python_version(),
        "stages": sorted(inv.STAGE_FILTERS),
        #: 已验证：都只依赖文件头 / 纯 Python（无需 torch）
        "operations": ["capabilities", "plan", "readiness", "inspect_weights", "manifest", "status"],
        #: 有实现但**没有性能与长稳验收**（WBS §12：骨架期不产出 GPU 数字）
        "unverified": ["generate（进程内推理缺性能基线与 60 分钟稳定性验收）",
                       "training（本仓未实现训练闭环）"],
        "backend": {
            "available": available,
            "reason": reason,
            "missingRequired": status["missing"],
            "optionalMissing": status.get("optionalMissing", []),
            "installRequired": status["install"],
        },
        "quant": {
            "supportedLayouts": ["per-tensor", "per-row", "per-col", "elementwise"],
            "scaleSuffixes": list(quant.SCALE_SUFFIXES),
            #: ⭐ 「做得了的族」与「做不了的族」**分两栏** ✓✗（2026-10-30）：混成一栏时报告没法
            #: 说出「nvfp4 能还原」✓，读的人只会看到「未实现 = mxfp4, nvfp4」⇒ 去换权重 ✓✗。
            "supportedFormats": sorted(quant.SUPPORTED_FORMATS),
            "unsupportedFamilies": sorted(quant.UNSUPPORTED_FAMILIES),
            "unsupportedFormats": sorted(quant.UNSUPPORTED_FORMATS),
        },
        #: ⭐ **设备策略**（2026-10-31 加 ✓）：这一栏答的是「这台引擎把什么跑在哪台设备上、
        #: **凭什么判**」✓✗。只报各条的 `device` 时，读的人只能看到**结论** ✓（“这次在 CPU” ✓），
        #: 看不出「是不是机器不支持」/「能不能改回来」⇒ 会把一次降级读成能力缺失 ✓✗。
        #: ⚠️ 这里**只报静态事实**（常量 / 判定规则 / 实测账）✗✗ —— :func:`capabilities` 的契约是
        #: 「零副作用、不 import torch、毫秒级」✓，读卡况是另一回事 ✓✗：**当时的**卡况走
        #: ``GET /engine/runtime`` 的 ``devicePolicy.pressure`` ✓（那里真读 ✓）。两个面各答一半 ✓。
        "devicePolicy": {
            "policies": list(policy_mod.POLICIES),
            "default": policy_mod.DEFAULT_POLICY,
            #: 余量/联合下限**从常量算** ✓✗（不手抄数字：改常量时报告自动跟着变 ✓，
            #: 手抄的话两处必然漂移 ✓✗ —— 本仓文档就是这么烂过一遍的 ✓）。
            "headroomGiB": round(policy_mod.HEADROOM_BYTES / policy_mod.GIB, 2),
            "minSplitFreeGiB": round(policy_mod.MIN_SPLIT_FREE_BYTES / policy_mod.GIB, 3),
            "highUsedRatio": policy_mod.HIGH_USED_RATIO,
            "hardUsedRatio": policy_mod.HARD_USED_RATIO,
            #: ⚠️ 主判据是**显存占用率**、不是 SM 利用率 ✓✗（见 `device_policy` 模块头 ✓）：
            #: 这一格写出来，读的人才不会把 `usedRatio` 当成 "GPU 忙不忙" ✓✗。
            "judgedOn": "memory",
            "rules": [
                f"剩余显存 ≥ 该件需求 + {policy_mod.HEADROOM_BYTES / policy_mod.GIB:g} GiB ⇒ 上卡",
                f"剩余放不下整件、但仍 ≥ {policy_mod.MIN_SPLIT_FREE_BYTES / policy_mod.GIB:g} GiB"
                f" ⇒ 上卡 + 联合驻留（residency=split ✓）",
                f"剩余 < {policy_mod.MIN_SPLIT_FREE_BYTES / policy_mod.GIB:g} GiB ⇒ 整件转移到 CPU"
                f"（慢，但跑得完 ✓）",
                f"需求未知且显存占用 ≥ {policy_mod.HARD_USED_RATIO * 100:g}% ⇒ 转移到 CPU",
            ],
            #: 两本账必须分开报 ✓✗：`part` 是**分时**账（2.1 三段各占一段 ✓）、
            #: `chain` 是**整链同驻**账（Z-Image 三件同时占满 ✓）。混在一起报，
            #: 读者会按"最重的单件"去估 ⇒ 差 4.5 GiB 就 OOM ✓✗（Z-Image 实测踩过 ✓）。
            "needGiB": {
                "part": dict(tb.TorchBackend.PART_NEED_GIB),
                "chain": dict(tb.TorchBackend.CHAIN_NEED_GIB),
                #: ⚠️ 第三本 ✓✗：2512 的**分时账**（与 `part` 同为分时，但**族不同** ✓）。
                #: 不能并进 `part` ✗：两族的键名一样（`dit`/`text`/`vae` ✓）⇒ 并进去必然互相覆盖 ✓✗，
                #: 2512 的 38.05 会被 2.1 的 13.26 顶掉 ✓ ⇒ 读者按 13.26 估余量 ⇒ 装到一半 OOM ✓✗。
                "qwenImagePart": dict(tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB),
            },
        },
        "catalog": {
            "path": str(inv.catalog_path()),
            "models": len(catalog.get("models") or []),
            "error": catalog.get("error"),
            "modelsDir": str(inv.models_dir() or ""),
        },
        #: 能力边界（WBS §1.1）：写清楚「不是什么都保证」
        "limits": [
            "只做单机单卡；多卡仅预留接口，无性能承诺",
            "显存配额是预算控制，不是 GPU 硬隔离（外部分配接口外无法强制）",
            "量化按**族**判，判不出就拒绝（不猜）：per-row/per-col/per-tensor/elementwise 由**形状**自证；"
            "nvfp4（U8 打包 + 128×4 块 scale + 二级 scale + AWQ）由**等式**判；"
            "4bit 打包（`w4a4`/`int4`，每字节两个有符号 nibble）由**文件自述**判 —— 打包后的形状与 "
            "per-row int8 逐字相同，按形状猜必错；"
            "mxfp4（E8M0 共享指数）与带 `qzeros`/`g_idx` 的 int4 **分组量化**仍未实现，一律拒绝",
        ],
        #: ⭐ **环境既知缺口**（E0.1 判据要求的字段）：与 `unverified`（**能力**未验收）不同 ——
        #: 这里说的是「**这台机器上**当前不满足的前提」✓。空数组 = 没有既知缺口（**不等于**已验收 ✗）。
        "knownGaps": _known_gaps(status),
    }


def _known_gaps(status: dict[str, Any]) -> list[dict[str, Any]]:
    """把「环境与运行时的既知缺口」写成结构化条目（**不猜、不省略**）。

    ⚠️ 为什么要单独存在 ✗：`backend.available=False` 只说「后端不可用」✓，但「为什么」有两种
    完全相反的原因 —— ① 包没装（装一下就好 ✓）；② **torch 是 CPU 轮子**（装了也上不了卡 ✗）。
    混在一句话里，读者会把 ② 当成 ① 处理 ✓✗（2026-10-29 实测本机正是 ②）。
    """
    from . import torch_backend as tb  # noqa: PLC0415

    gaps: list[dict[str, Any]] = []
    facts = tb.torch_facts()
    if facts.get("installed") and facts.get("cpuWheel"):
        gaps.append({
            "id": "torch-cpu-wheel",
            "severity": "blocking-for-performance",
            "summary": "torch 是 CPU 轮子（编译时无 CUDA）⇒ 依赖齐了也只能 CPU 跑",
            "facts": {"version": facts.get("version"), "cudaVersion": facts.get("cudaVersion")},
            "fix": "按驱动版本重装 CUDA 轮子（索引源按 nvidia-smi 的 CUDA Version 选，本仓不写死版本）",
        })
    elif not facts.get("installed"):
        gaps.append({"id": "torch-missing", "severity": "blocking",
                     "summary": "torch 未安装或导入失败",
                     "facts": {"error": facts.get("error")},
                     "fix": "装 torch（有卡时选 CUDA 轮子）"})
    if status["missing"]:
        gaps.append({"id": "deps-missing", "severity": "blocking",
                     "summary": f"缺必需依赖：{', '.join(status['missing'])}",
                     "facts": {"missing": status["missing"]},
                     "fix": (status["install"] or [""])[0]})
    if facts.get("installed") and not facts.get("cudaAvailable") and not facts.get("cpuWheel"):
        #: 轮子里有 CUDA、但运行时没看到卡 ⇒ 驱动/设备映射问题（与「轮子不对」是两回事 ✓）
        gaps.append({"id": "cuda-not-available", "severity": "blocking-for-gpu",
                     "summary": "torch 带 CUDA 但 `cuda.is_available()` 为 False",
                     "facts": {"cudaVersion": facts.get("cudaVersion")},
                     "fix": "查驱动版本与设备可见性（`nvidia-smi`）"})
    return gaps


def _check_stage(stage: str) -> None:
    from . import inventory as inv  # noqa: PLC0415

    if stage not in inv.STAGE_FILTERS:
        raise EngineError(f"未知阶段 {stage!r}", code="E_BAD_CONFIG",
                          hint=f"可用阶段：{', '.join(sorted(inv.STAGE_FILTERS))}")


# ══════════════════════════════════════════════════════════════════════════
# 计划 / 就绪（装载前就能给结论，不需要 torch）
# ══════════════════════════════════════════════════════════════════════════
def plan(stage: str = "h3", *, root: str | Path | None = None,
         capacity_gib: float | None = None) -> dict[str, Any]:
    """某阶段的**装载计划**：逐组件 + 显存排班。缺文件是**结论**不是异常。

    与 :func:`readiness` 的分工：本函数回答「**怎么装**」（键名/形状/量化/是否分块读），
    :func:`readiness` 回答「**能不能装**」（齐不齐 + 阻断项）。
    """
    from . import loader as loader_mod  # noqa: PLC0415

    _check_stage(stage)
    kwargs: dict[str, Any] = {"root": Path(root) if root else None}
    if capacity_gib is not None:
        kwargs["capacity_gib"] = float(capacity_gib)
    try:
        data = loader_mod.plan_stage(stage, **kwargs)
    except Exception as err:  # noqa: BLE001 —— 边界上统一翻译
        raise _friendly(err) from err
    return {"schemaVersion": SCHEMA_VERSION, "apiVersion": API_VERSION, **data}


def readiness(stage: str = "h3", *, root: str | Path | None = None,
              capacity_gib: float | None = None) -> dict[str, Any]:
    """**引擎侧**就绪：权重齐不齐 / 结构坏没坏 / 量化能不能还原 / 显存估。

    ⚠️ **不含**项目侧编排（`ffmpeg`、pip 建议、分镜有无）—— 那些要读项目数据，归
    `app.services.engine_readiness`。引擎要能**独立**回答自己的问题。

    `blockers` 非空即不可装；`unchecked` 是「**没查**」，不是通过（没查 ≠ 通过）。
    """
    from . import inventory as inv  # noqa: PLC0415
    from . import loader as loader_mod  # noqa: PLC0415

    _check_stage(stage)
    kwargs: dict[str, Any] = {"root": Path(root) if root else None}
    if capacity_gib is not None:
        kwargs["capacity_gib"] = float(capacity_gib)
    try:
        ready = inv.readiness(stage, **kwargs)
        #: 齐不齐归 `inventory`，量化族归 `loader`（`dequantPlan` 只在计划的组件里）—— 两处都只读文件头
        plans = {item["key"]: item for item in loader_mod.plan_stage(stage, **kwargs)["components"]}
    except Exception as err:  # noqa: BLE001
        raise _friendly(err) from err

    blockers: list[str] = []
    unchecked: list[str] = []
    unsupported: list[dict[str, Any]] = []
    problems: list[str] = []
    for component in ready["components"]:
        key = str(component["key"])
        if not component["present"]:
            #: 文件不在 ⇒ 量化族**无法判断** ⇒ 进 unchecked，不能算通过
            unchecked.append(f"`{key}` 的量化布局（文件不在盘上）")
            if component["required"]:
                blockers.append(f"缺必需权重 `{key}` ⇒ 装载不了")
            continue
        problems.extend(str(text) for text in (component.get("problems") or []))
        detail = (plans.get(key) or {}).get("dequantPlan") or {}
        family = detail.get("family") or {}
        if family.get("name"):
            unsupported.append({"key": key, "family": family["name"],
                                "reason": family.get("reason")})
            blockers.append(f"`{key}` 属**未实现的量化族**（`{family['name']}`）⇒ 装载会中止")
        elif detail.get("unresolvedCount"):
            blockers.append(f"`{key}` 有 {detail['unresolvedCount']} 个低精度权重"
                            f"**判不出反量化布局** ⇒ 装载会中止")
    for key in ready["brokenRequired"]:
        blockers.append(f"权重 `{key}` **结构坏**（截断 / 头部非法）⇒ 装载会中止")
    for key in ready.get("suspectRequired") or []:
        #: 大小与清单不符但结构完好：**不阻断**（社区量化版/重新导出会让体积合法地变）
        unchecked.append(f"`{key}` 的体积是否只是「重新导出」（大小与清单不符，未人工确认）")
    if ready.get("catalogError"):
        blockers.append(f"权重清单读不了：{ready['catalogError']}")

    return {
        "schemaVersion": SCHEMA_VERSION,
        "apiVersion": API_VERSION,
        "stage": stage,
        "ready": not blockers,
        "blockers": blockers,
        "unchecked": unchecked,
        "unsupportedQuant": unsupported,
        "problems": problems,
        "weights": {
            "modelsDir": ready["modelsDir"],
            "requiredWeightsGiB": ready["requiredWeightsGiB"],
            "resolvedElsewhere": ready.get("resolvedElsewhere") or {},
            "note": ready.get("resolvedElsewhereNote", ""),
        },
        "vram": ready.get("vram") or {},
        "tiers": ready.get("tiers") or {},
        "components": ready["components"],
    }


# ══════════════════════════════════════════════════════════════════════════
# 权重核对（WBS-2.6 / G2 前置「损坏权重不能被误判为有效模型」）
# ══════════════════════════════════════════════════════════════════════════
def inspect_weights(path: str | Path) -> dict[str, Any]:
    """**单个权重文件**核对：结构 / 张量统计 / 量化族 / 自动还原计划（不装载、不占显存）。

    这是「**装之前**能说的全部」：坏文件在这里就该被挡住（WBS G2 前置），而不是等装载到一半才炸
    —— 半装状态最难查。
    """
    from . import loader as loader_mod  # noqa: PLC0415
    from . import quant as quant_mod  # noqa: PLC0415
    from . import quant_nvfp4 as nvfp4_mod  # noqa: PLC0415
    from . import safetensors as st  # noqa: PLC0415

    target = Path(path)
    #: ⚠️ 「**不在盘上**」与「**在盘上但坏了**」必须分开 ✗（`st.inspect` 对两者都给 `ok=False` ✓✗）：
    #: 前者要下载、后者要重下或换源，处置完全不同 ✓。判断放在 `inspect` **之前**才行 ✗。
    if not target.is_file():
        raise EngineError(f"权重文件不在盘上：{target}",
                          code="E_WEIGHT_MISSING",
                          hint="确认路径，或按 `configs/models.json` 的来源下载",
                          detail={"path": str(target)})
    try:
        info = st.inspect(target)
    except Exception as err:  # noqa: BLE001
        raise _friendly(err) from err
    if not info.ok:
        raise EngineError(f"{target.name} 结构坏：{'；'.join(info.problems) or '未知'}",
                          code="E_WEIGHT_CORRUPT",
                          detail={"path": str(target), "problems": list(info.problems)})

    formats = loader_mod.quant_formats_of(target, info)
    #: ⚠️ convrot 必须**逐层**判 ✗✗（2026-10-30 实测两处，见 `loader.quant_specs_of`）：
    #: ① H3 全量版同文件里 64 与 256 并存 ⇒ 「整文件必须同组」会把正品整个拒掉 ✓；
    #: ② Krea2 int4 那族一个 `*.comfy_quant` 都没有、说明全在 metadata `_quantization_metadata`
    #:    里 ⇒ 只看张量名等于没看见旋转 ⇒ 少一步反旋转却报「可自动还原」✓✗（假绿 ✓）。
    layer_specs = loader_mod.quant_specs_of(target, info)
    convrot, convrot_gap = quant_mod.convrot_group_of(layer_specs)
    #: ⚠️ **4bit 打包**（`w4a4`/`int4` ✓）也必须在这里认出来 ✗✗：打包后的形状与普通 per-row
    #: int8 **逐字相同**（`(6144,3072)` + `(6144,)` ✓）⇒ 形状判据全都验不出来 ✓✗，
    #: 唯一证词是文件自己写的 `w4a4` ✓（2026-10-30 实测：一度报成「可自动还原」✓）。
    int4 = quant_mod.int4_stems_of(layer_specs)
    #: ⚠️ 走 `loader.dequant_plan_of` 而**不是**自己拼 `plan_dequant` ✗（2026-10-30）：那才是
    #: 「nvfp4 摘层 + 合并」的唯一口径 ✓ —— 自己拼一份 ⇒ 同一个文件在 `片场 inspect` 里说
    #: 「判不出布局 ✗」、在 `plan_stage` 里说「能还原 ✓」，两处**互相打脸** ✓✗。
    dequant, dequant_problems = loader_mod.dequant_plan_of(info, formats, convrot=convrot,
                                                          int4=int4)
    family: dict[str, Any] | None = None
    for spec in formats:
        hit = quant_mod.format_family(spec)
        if hit:
            family = {"name": hit[0], "reason": hit[1], "spec": spec}
            break
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "apiVersion": API_VERSION,
        "path": str(target),
        "ok": True,
        "fileBytes": info.file_bytes,
        "gib": round(info.file_bytes / 1024 ** 3, 2),
        "headerBytes": info.header_bytes,
        "tensorCount": info.tensor_count,
        "totalTensorBytes": info.total_tensor_bytes,
        "dtypeCounts": info.dtype_counts,
        "metadata": info.metadata,
        "problems": list(info.problems),
        "quant": {
            "supported": bool(dequant.get("supported")),
            "layouts": dequant.get("layouts") or {},
            "weights": dequant.get("weights", 0),
            "unresolved": dequant.get("unresolved") or [],
            "grouped": dequant.get("grouped") or [],
            "unpaired": dequant.get("unpaired") or [],
            #: ⭐ **没人要的 scale** ✓（2026-10-30）：反向那一半 —— 融合 QKV 被拆成分片
            #: `weight_scale` 时正是这个形态 ✓。旧判据只从**权重**侧找 scale ⇒ 三份分片全看不见 ✓✗。
            #: 只报**不拦** ✗：真会算错的形态（融合权重自己缺 scale ✓）已经落在 `unpaired` 里 ✓
            #: （理由写在 `quant.orphan_scales` 的 docstring 里 ✓）。
            "orphanScales": dequant.get("orphanScales") or [],
            #: ⚠️ **自称 4bit 打包却用不了** ✓（层名对不上 / 不是单字节存储 ✓）—— 这类文件
            #: 必须**整份拒** ✓：解包不发生 ⇒ 按 int8 乘 ⇒ 数值全错但形状全对 ✓✗（假绿 ✓）
            "int4Problems": dequant.get("int4Problems") or [],
            "int4Packed": dequant.get("int4Packed") or [],
            "int4PackedCount": dequant.get("int4PackedCount", 0),
            "convrot": convrot,
            #: nvfp4 的逐层体检 ✓（不是这一族时为 ``None`` ✓）—— 报告里要**看得出**它是
            #: 靠 `quant_nvfp4` 的布局等式判的 ✓，不是「形状看着像就放行」✓✗
            "nvfp4": dequant.get(nvfp4_mod.FORMAT) or None,
        },
        "formats": formats,
        "family": family,
    }
    report["problems"].extend(dequant_problems)
    if convrot_gap:
        #: 自相矛盾（多 group / 读不成整数）⇒ 拒绝反解 ✓ —— 与装载路径同一口径 ✗
        report["quant"]["supported"] = False
        report["problems"].append(f"convrot 说明自相矛盾 ✗：{convrot_gap}")
    if family:
        #: 结构没问题、但这一族本仓没实现 ⇒ **明确拒绝**（不是"装载时再看"）
        report["ok"] = False
        raise EngineError(f"{target.name} 属未实现的量化族 `{family['name']}`",
                          code="E_UNSUPPORTED_QUANT",
                          hint="换成本仓支持的导出（per-row int8 无旋转 / bf16）",
                          detail={"path": str(target), "inspect": report})
    return report


def manifest_of(path: str | Path, entry: dict[str, Any] | None = None,
                *, full_hash: bool = False) -> dict[str, Any]:
    """**权重清单条目**（WBS-2.6：来源 / 哈希 / 格式 / 版本 / 许可）。

    哈希分两档，别混（19.53 GiB 全量哈希要几十秒，体检不该付这个代价）：

    * 默认 ``hashScope="header+size"``：对「头部字节 + 文件总字节数」算 sha256 ——
      **防头部/结构被换**，毫秒级；**不防数据区篡改**（字段名照实写）；
    * ``full_hash=True`` ⇒ ``hashScope="full"`` + 真 ``sha256``（慢，用于发行登记）。

    许可**不猜**：清单里没有 ``license`` ⇒ 输出 ``None`` + ``licenseNote``
    （写「未声明」比编一个「Apache-2.0」强）。
    """
    from . import inventory as inv  # noqa: PLC0415
    from . import safetensors as st  # noqa: PLC0415

    target = Path(path)
    try:
        info = st.inspect(target)
    except Exception as err:  # noqa: BLE001
        raise _friendly(err) from err
    if entry is None:
        for item in (inv.load_catalog().get("models") or []):
            if str(item.get("filename") or "") == target.name:
                entry = item
                break
    entry = entry or {}
    record: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "apiVersion": API_VERSION,
        "path": str(target),
        "name": target.name,
        "bytes": info.file_bytes,
        "gib": round(info.file_bytes / 1024 ** 3, 2),
        "tensorCount": info.tensor_count,
        "dtypeCounts": info.dtype_counts,
        "hashScope": "header+size",
        "headSha256": _sha256_of(_header_plus_size(target)),
        "sha256": None,
        "source": {"url": entry.get("url"), "repo": entry.get("repo"),
                   "catalogKey": entry.get("key")},
        "kind": entry.get("kind"),
        "version": (info.metadata or {}).get("version"),
        "format": (info.metadata or {}).get("format"),
        "license": entry.get("license"),
        "licenseNote": "" if entry.get("license") else
                       "权重清单未声明许可 ⇒ 需人工登记（本仓不猜）",
    }
    if full_hash:
        record["hashScope"] = "full"
        record["sha256"] = _sha256_of_file(target)
    return record


def _header_plus_size(path: Path) -> bytes:
    """头部原始字节 + 文件总字节数（不读数据区 ⇒ 19 GiB 也是毫秒级）。"""
    import struct  # noqa: PLC0415

    with path.open("rb") as handle:
        raw_length = handle.read(8)
        length = struct.unpack("<Q", raw_length)[0]
        payload = handle.read(int(length))
    return raw_length + payload + str(path.stat().st_size).encode("ascii")


def _sha256_of(data: bytes) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# 会话式门面（WBS-2.1 生命周期：load / infer / release）
# ══════════════════════════════════════════════════════════════════════════
def compat(stage: str = "h3", *, root: str | Path | None = None,
           capacity_gib: float | None = None,
           activation_bytes: int = 0) -> dict[str, Any]:
    """**装载前**四类兼容性报告（E2.2 ✓）—— 依赖 / 权重 / 量化族 / 显存。

    与 :func:`readiness` 的分工 ✗：``readiness`` 回答「**这一套**权重齐不齐、估多大」✓；
    本函数回答「**这台机器上这个阶段装得了吗**」✓ —— 多了依赖与版本下限（``readiness``
    完全不管依赖 ✓）、多了**按类分开的码**（缺依赖 vs 装不下 vs 量化族不支持 ✓）。
    ``unchecked`` 一栏照 old 口径带出去：**没查 ≠ 通过** ✓✗。
    """
    from . import registry as registry_mod  # noqa: PLC0415

    try:
        return {"schemaVersion": SCHEMA_VERSION, "apiVersion": API_VERSION,
                **registry_mod.check_compat(
                    stage, root=Path(root) if root else None,
                    capacity_gib=capacity_gib, activation_bytes=activation_bytes)}
    except Exception as err:  # noqa: BLE001
        raise _friendly(err) from err


def status() -> dict[str, Any]:
    """运行时的**事实**（已装什么、跑没跑、显存占用）—— 只读、不触发装载。"""
    from . import runtime as runtime_mod  # noqa: PLC0415

    try:
        return {"schemaVersion": SCHEMA_VERSION, "apiVersion": API_VERSION,
                **runtime_mod.engine_runtime.status()}
    except Exception as err:  # noqa: BLE001
        raise _friendly(err) from err


class Engine:
    """**一次会话**：`capabilities` / `plan` / `readiness` / `load` / `submit` / `release`。

    用法（进程内，不需要 HTTP）::

        from app.services.engine.facade import Engine

        with Engine("h3") as engine:
            print(engine.readiness()["ready"])
            task = engine.submit(request, assembly={"seed": 42})
            print(engine.wait(task["taskId"]))

    ⚠️ `request` 的**结构**由 `app.services.engine.pipeline` 定义 —— 本门面**原样透传**，
    不解释、不改写（解释它就得跟着脚本格式漂，而脚本格式是本仓最容易变的东西）。
    """

    def __init__(self, stage: str = "h3", *, root: str | Path | None = None,
                 capacity_gib: float | None = None) -> None:
        _check_stage(stage)
        self.stage = stage
        self.root = Path(root) if root else None
        self.capacity_gib = capacity_gib

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"<Engine stage={self.stage!r} api={API_VERSION}>"

    def capabilities(self) -> dict[str, Any]:
        return capabilities()

    def plan(self) -> dict[str, Any]:
        return plan(self.stage, root=self.root, capacity_gib=self.capacity_gib)

    def readiness(self) -> dict[str, Any]:
        return readiness(self.stage, root=self.root, capacity_gib=self.capacity_gib)

    def inspect(self, path: str | Path) -> dict[str, Any]:
        return inspect_weights(path)

    def manifest(self, path: str | Path, *, full_hash: bool = False) -> dict[str, Any]:
        return manifest_of(path, full_hash=full_hash)

    def status(self) -> dict[str, Any]:
        return status()

    def compat(self, *, activation_bytes: int = 0) -> dict[str, Any]:
        return compat(self.stage, root=self.root, capacity_gib=self.capacity_gib,
                      activation_bytes=activation_bytes)

    def load(self, *, check: bool = True, require: tuple[str, ...] = ("deps", "weights"),
             **kwargs: Any) -> dict[str, Any]:
        """把该阶段权重装进显存（**幂等**：同配置重复调用不重复占显存）。

        ``check=True``（默认）⇒ **先过 E2.2 的兼容性闸门** ✓：装不了的话在搬张量**之前**
        就给可读错误 ＋ 机器可判的码 ✓✗。

        ``require`` 选**哪几类算硬阻断** ✓，默认只有 ``deps`` / ``weights`` —— 这两类与「装哪几件」
        无关，结论是确定的 ✓。另两类**依赖装配**，闸门这时还不知道装配 ✗✗：
        * ``quant``：当时实测本机 h3 的 `te_nvfp4`（清单标必需、**那时** nvfp4 本仓未实现 ✓）并不挡
          现有出片链路 —— 说明那条链路不吃它 ⇒ 一律硬拦会把**能跑的**挡在门外 ✓✗
          （nvfp4 自 2026-10-30 起已能还原 ✓，但这条「quant 依赖装配」的判断与具体族无关，仍然成立 ✓）；
        * ``vram``：估算按「权重全精度常驻」算，而 ``int8_offload`` / 分辨率这时还没定 ⇒ 同样会误拦 ✓✗
          （真爆由装载期 OOM 带**同一份** `registry.VRAM_HINT` 兜底 ✓）。
        要严格把关就传 ``require=("deps","weights","quant","vram")`` ✓，或直接看 :func:`compat` ✓。
        """
        from . import registry as registry_mod  # noqa: PLC0415
        from . import runtime as runtime_mod  # noqa: PLC0415

        try:
            if check:
                registry_mod.ensure_compat(
                    self.stage, root=self.root, capacity_gib=self.capacity_gib,
                    require=tuple(require) if require else ())
            return runtime_mod.engine_runtime.ensure_loaded(stage=self.stage, **kwargs)
        except Exception as err:  # noqa: BLE001
            raise _friendly(err) from err

    def submit(self, request: Any, *, assembly: dict[str, Any] | None = None,
               **kwargs: Any) -> dict[str, Any]:
        from . import runtime as runtime_mod  # noqa: PLC0415

        try:
            return runtime_mod.engine_runtime.submit(request, assembly=assembly, **kwargs)
        except Exception as err:  # noqa: BLE001
            raise _friendly(err) from err

    def wait(self, task_id: str, *, timeout: float | None = None) -> dict[str, Any] | None:
        from . import runtime as runtime_mod  # noqa: PLC0415

        return runtime_mod.engine_runtime.wait(task_id, timeout=timeout)

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        from . import runtime as runtime_mod  # noqa: PLC0415

        return runtime_mod.engine_runtime.cancel(task_id)

    def release(self) -> dict[str, Any]:
        """卸载并回收（**可观测**：返回里带回收后的显存事实，不只看对象删除）。"""
        from . import runtime as runtime_mod  # noqa: PLC0415

        try:
            return runtime_mod.engine_runtime.unload()
        except Exception as err:  # noqa: BLE001
            raise _friendly(err) from err
