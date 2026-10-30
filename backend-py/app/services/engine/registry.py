"""**stage 注册表 ＋ 装载前兼容性闸门**（E2.1 / E2.2 ✓ 2026-10-30）。

装载只有两类失败对用户有意义：①「这台机器/这份权重根本装不了」、②「装得进但跑出来不对」。
本模块管 ①，而且要在**动手装载之前**说清楚 —— 等张量搬到卡上才发现，代价是几十 GB 搬运 ＋
一次 OOM ＋ 一段没人看得懂的 traceback。

四类判据（:data:`COMPAT_KINDS`，各带机器可判错误码）：

| 类 | 问什么 | 码 |
|---|---|---|
| ``deps`` | 依赖齐不齐、版本够不够下限 | ``E_ENGINE_UNAVAILABLE`` |
| ``weights`` | 必需权重在不在盘上、结构坏没坏 | ``E_WEIGHT_MISSING`` / ``E_WEIGHT_CORRUPT`` |
| ``quant`` | 量化族本仓实现了没有（convrot / nvfp4 已实现 ✓；mxfp4 未实现 ✗） | ``E_UNSUPPORTED_QUANT`` |
| ``vram`` | 权重 ＋ 激活装得下这张卡吗 | ``E_VRAM_EXHAUSTED`` |

三条硬规矩：

1. **「没查」不等于「通过」** —— 文件不在盘上就没法判量化族 ⇒ 记 ``unchecked``，不许写成"量化 OK"。
2. **体检自己炸了也算拒绝** —— 读不了头部 / 清单坏了 ⇒ 出 blocker；把"我不知道"读成"可以装"是本仓最恨的错。
3. **组件名册不在这里抄一遍** —— 哪些组件必需归 ``configs/models.json`` 的 ``required``、按
   ``category`` 筛选归 :data:`inventory.STAGE_FILTERS` ⇒ 本模块只声明「阶段 → 分类 / 能力 / 依赖 /
   版本下限」，组件由 :func:`inventory.readiness` 算出来。手抄一份 key 名册＝第二个真源，模型一换名就静默过期。

边界：不含项目侧编排（ffmpeg / 分镜 / 提示词，归 ``app.services.engine_readiness``）；**不 import torch**
（依赖探测走 ``find_spec``、版本走 ``importlib.metadata``）⇒ 无卡机器上也能给结论；**不装载**（只读文件头）。
"""
from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "COMPAT_KINDS",
    "CompatError",
    "DEFAULT_CODE",
    "RegistryError",
    "STAGES",
    "StageSpec",
    "VRAM_HINT",
    "check_compat",
    "describe",
    "ensure_compat",
    "registration_problems",
    "stage_spec",
    "stages",
    "unclaimed_categories",
    "version_of",
]

#: 四类判据 —— 顺序即报告顺序，也决定 :func:`ensure_compat` 先报哪一条。
COMPAT_KINDS: tuple[str, ...] = ("deps", "weights", "quant", "vram")

#: 四类 ⇒ 默认错误码（同类里更具体的码由判据自己给，例如缺件 vs 结构坏）。
#: ⚠️ 这些码必须都在 `facade.CODES` 里，否则 `_friendly` 会翻成 ``E_ENGINE_UNAVAILABLE`` ——
#: 用户照着"引擎不可用"去查依赖，而真相是"这张卡装不下"（`engine_registry_test` 钉这条）。
DEFAULT_CODE: dict[str, str] = {
    "deps": "E_ENGINE_UNAVAILABLE",
    "weights": "E_WEIGHT_MISSING",
    "quant": "E_UNSUPPORTED_QUANT",
    "vram": "E_VRAM_EXHAUSTED",
}

#: 显存不够时可行动的三条路 —— **单一真源**：`facade._friendly` 的 OOM 分支也读它
#: （同一个事实两处各写一遍，迟早给出两个说法；`engine_queue_backpressure_test` 钉的
#: 「hint 必须点名 `int8Offload`」盯的也是这一段）。
VRAM_HINT = (
    "按代价从低到高试：① 开 int8Offload（装配键 int8_offload），"
    "int8 常驻权重改钉 CPU 锁页内存、前向逐层搬上卡，显存立省 17.94 GiB"
    "（dit_fl2va_int8 的实测值，别的量化档按它自己的体积算）；"
    "② 降分辨率 / 帧数（megapixels、ratio、seconds）—— 激活按它们长；"
    "③ 换更大的卡。已经开着 int8Offload 还是爆，就只剩 ②③"
    "（offload 省的是权重那份，剩下的是激活，它跟分辨率走）"
)


class RegistryError(ValueError):
    """注册表层的拒绝：``code`` 给机器判、``hint`` 给人看。

    继承 ``ValueError`` 是有意的：门面 :func:`facade._friendly` 对 ``ValueError`` 一律翻成
    ``E_BAD_CONFIG`` ⇒ 即使调用方漏了 ``code``，「未知阶段」也不会退化成「引擎不可用」。
    """

    code = "E_BAD_CONFIG"

    def __init__(self, message: str, *, code: str = "E_BAD_CONFIG",
                 hint: str = "", detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint
        self.detail = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "hint": self.hint, "detail": dict(self.detail)}


class CompatError(RegistryError):
    """装载前已知装不了：带四类里的那一个码 ＋ 全部阻断项。

    ⚠️ ``code`` 只取第一条，全部阻断项在 ``detail["blockers"]`` —— 只报一条会让用户
    "修一个、再撞一个"，三条一起看见才能一次改完。
    """


@dataclass(frozen=True)
class StageSpec:
    """一个阶段的注册信息（不含组件名册 —— 那份归 ``models.json``，见模块头规矩 3）。

    ``deps`` 写**发行包名**（pip 名，例如 ``PIL`` 的包名是 ``pillow``）：版本下限靠
    ``importlib.metadata`` 读包元数据，不 import 它。
    """

    name: str
    #: 与 :data:`inventory.STAGE_FILTERS` 同名的清单 ``category``（由 registration_problems 双向核）
    category: str
    #: 这个阶段真能做的事（必须对得上本仓实现 —— 能力页不许许空头）
    capabilities: tuple[str, ...]
    #: 必需依赖（发行包名）
    deps: tuple[str, ...] = ("torch",)
    #: 版本下限 ``((包名, 下限), …)``；空 = 只要求装得上
    minVersions: tuple[tuple[str, str], ...] = ()
    #: 给读者的一句话（边界、已知缺口）
    note: str = ""


#: 能力名必须对得上实现：``h3`` → pipeline ＋ h3_edit 的续拍/重拍计划；``image`` → sdxl* ＋ upscale；
#: ``audio`` → audio_mix；``text`` → llm_backend ＋ gguf_to_llm。
STAGES: dict[str, StageSpec] = {
    "h3": StageSpec(
        name="h3", category="video",
        capabilities=("video.generate", "video.generate.audio", "video.retake", "video.continue"),
        deps=("torch", "numpy", "safetensors"),
        #: 2.0 之前没有本仓注意力全走的 `scaled_dot_product_attention` ⇒ 低于 2.0 明确拒，
        #: 不去赌"也许能跑"。
        minVersions=(("torch", "2.0.0"),),
        note="双流（画面 ＋ 音频）；续拍/重拍走 h3_edit 的计划层；int8_offload 是 720×1280 唯一装得下的档",
    ),
    "image": StageSpec(
        name="image", category="image",
        capabilities=("image.generate", "image.upscale"),
        deps=("torch", "numpy", "safetensors", "pillow"),
        minVersions=(("torch", "2.0.0"),),
        note="SDXL 骨架 ＋ 超清放大器（>2× 要调用方显式给 maxTile，不猜安全块）",
    ),
    "audio": StageSpec(
        name="audio", category="audio",
        capabilities=("audio.mix",),
        deps=("numpy",),
        note=("⚠️ 已知缺口：清单里这类组件的 category 写的是 `tts` ⇒ 本阶段权重筛选一件都匹配不到 ⇒ "
              "readiness 会空就绪（`ready=True` 是「没有问题」、不是「有东西」，见 unclaimed_categories）；"
              "TTS 本身不在引擎内（走 provider），引擎这层只有混音 audio_mix"),
    ),
    "text": StageSpec(
        name="text", category="text",
        capabilities=("text.generate",),
        deps=("torch", "numpy"),
        minVersions=(("torch", "2.0.0"),),
        note="GGUF 权重 → 自研前向；⚠️ GGUF-only 模型没有 tokenizer ⇒ 能装载＋前向，不一定能出文本",
    ),
}


def stages() -> list[str]:
    """已注册的阶段名（排序 ⇒ 报告稳定）。"""
    return sorted(STAGES)


def stage_spec(stage: str) -> StageSpec:
    """按名字取注册信息；**未注册 ⇒ :class:`RegistryError`（``E_BAD_CONFIG``）**。

    ⚠️ 不回落成默认阶段：``stage="video"`` 之类写错时静默按 ``h3`` 装，用户以为在跑另一个阶段。
    """
    name = str(stage or "").strip()
    if not name:
        raise RegistryError("阶段名是空的（不知道要装哪一类）", detail={"stage": ""})
    spec = STAGES.get(name)
    if spec is None:
        raise RegistryError(
            f"未注册的阶段 {name!r} ⇒ 拒绝装载（不回落成任何默认阶段）",
            hint=f"可用阶段：{', '.join(stages())}",
            detail={"stage": name, "registered": stages()})
    return spec


# ══════════════════════════════════════════════════════════════════════════
# 依赖 / 版本（四类里的第 1 类）
# ══════════════════════════════════════════════════════════════════════════
def version_of(package: str) -> str | None:
    """读**已装包**的版本，没装 ⇒ ``None``。

    走 ``importlib.metadata``（读 ``*.dist-info``）而不是 ``import``：① 判下限不需要把 torch
    拖进内存；② 包装坏了（import 就炸）时仍读得到版本 —— 那种情况下"装坏了"才是真相，
    由 ``dependency_status``（``find_spec`` 也认得出）与真装载那次 import 暴露。
    """
    name = str(package or "").strip()
    if not name:
        return None
    try:
        return str(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 —— 元数据坏掉也是"读不出来"，不抛
        return None


def _version_tuple(raw: str) -> tuple[int, ...] | None:
    """``"2.14.1+cu130"`` ⇒ ``(2, 14, 1)``；一个数字都没有 ⇒ ``None``（判不了就别说"够"）。"""
    numbers = re.findall(r"\d+", str(raw or ""))
    if not numbers:
        return None
    return tuple(int(part) for part in numbers[:4])


def _meets_floor(version: str | None, floor: str) -> bool | None:
    """``None`` = **判不了**（调用方不许把它当通过）。"""
    left, right = _version_tuple(str(version or "")), _version_tuple(floor)
    if left is None or right is None:
        return None
    width = max(len(left), len(right))
    return (left + (0,) * (width - len(left))) >= (right + (0,) * (width - len(right)))


def _deps_facts() -> dict[str, Any]:
    """依赖现状 —— 转 :func:`torch_backend.dependency_status`（``find_spec``，不 import）。"""
    from . import torch_backend as tb  # noqa: PLC0415 —— 注册表不在 import 期拖后端

    status = tb.dependency_status()
    return {
        "missing": list(status["missing"]),
        "optionalMissing": list(status.get("optionalMissing") or []),
        "install": list(status.get("install") or []),
        "items": status["items"],
    }


# ══════════════════════════════════════════════════════════════════════════
# 四类判据 ⇒ 报告
# ══════════════════════════════════════════════════════════════════════════
def _blocker(kind: str, code: str, message: str, *, hint: str = "",
             facts: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"kind": kind, "code": code, "message": message,
            "hint": hint, "facts": dict(facts or {})}


def _readiness(stage: str, root: Path | None, capacity_gib: float | None) -> Mapping[str, Any]:
    from . import inventory as inv  # noqa: PLC0415

    kwargs: dict[str, Any] = {"root": root}
    if capacity_gib is not None:
        kwargs["capacity_gib"] = float(capacity_gib)
    return inv.readiness(stage, **kwargs)


def _plans(stage: str, root: Path | None, capacity_gib: float | None) -> Mapping[str, Any]:
    """组件计划 —— 量化族判据的真源（``loader.plan_stage`` 的 ``dequantPlan.family``）。"""
    from . import loader as loader_mod  # noqa: PLC0415

    kwargs: dict[str, Any] = {"root": root}
    if capacity_gib is not None:
        kwargs["capacity_gib"] = float(capacity_gib)
    return loader_mod.plan_stage(stage, **kwargs)


def _scan(kind: str, stage: str, fn: Any, root: Path | None,
          capacity_gib: float | None) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]]]:
    """跑一次体检 —— **炸了就是 blocker**（规矩 2：不许把"我不知道"读成"可以装"）。"""
    try:
        return fn(stage, root, capacity_gib), []
    except Exception as err:  # noqa: BLE001 —— 体检失败本身就是结论
        #: ⚠️ 码要分清「文件坏了」（safetensors 层报的 ⇒ 用户该重下）与「体检自己崩了」
        #: （⇒ 那是引擎/环境的锅，报成"缺件"会把用户送去检查磁盘 ✗）。
        module = type(err).__module__.lower()
        code = "E_WEIGHT_CORRUPT" if "safetensors" in module or "struct" in module \
            else "E_ENGINE_UNAVAILABLE" if kind == "weights" else DEFAULT_CODE[kind]
        return None, [_blocker(
            kind, code,
            f"兼容性体检自己失败了（{type(err).__name__}: {err}）⇒ 不能判成「可以装」",
            hint="先看这份权重/清单能不能读：读不了就不是参数问题，是文件或安装坏了",
            facts={"error": f"{type(err).__name__}: {err}"})]


def check_compat(stage: str, *, root: Path | None = None,
                 capacity_gib: float | None = None,
                 activation_bytes: int = 0,
                 deps: Mapping[str, Any] | None = None,
                 readiness: Mapping[str, Any] | None = None,
                 plans: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """**装载前**四类兼容性 → 报告（不抛错：缺件/坏件都是结论的一种）。

    ``deps`` / ``readiness`` / ``plans`` 是**注入点**（自检按类各造一条故障，不碰真盘），
    真机调用不传 ⇒ 全走真实现。注入点不是"开关"：传 ``{}`` 表示"这一类没查"，仍按
    「没查 ≠ 通过」记 ``unchecked``。

    返回 ``{"stage", "ok", "blockers", "warnings", "unchecked", "checks", "spec"}``；
    ``ok`` 只看 ``blockers``，``unchecked`` 不影响 ``ok`` 但会原样带出去 —— 报告不替读者
    把"没查"说成"过了"。
    """
    spec = stage_spec(stage)                                  # 未注册 ⇒ RegistryError
    blockers: list[dict[str, Any]] = []
    warnings: list[str] = []
    unchecked: list[str] = []
    checks: dict[str, dict[str, Any]] = {kind: {"ok": True, "blockers": 0, "facts": {}}
                                        for kind in COMPAT_KINDS}

    # ── ① 依赖 ＋ 版本下限 ───────────────────────────────────────────────
    dep_facts: Mapping[str, Any] = dict(deps) if deps is not None else _deps_facts()
    missing = [str(item) for item in (dep_facts.get("missing") or [])]
    wanted = {str(item) for item in spec.deps}
    required_missing = [name for name in missing if name in wanted]
    if required_missing:
        install = str((dep_facts.get("install") or [""])[0] or "")
        blockers.append(_blocker(
            "deps", DEFAULT_CODE["deps"],
            f"缺必需依赖：{', '.join(required_missing)} ⇒ 这一阶段装不了",
            hint=f"先装：{install or 'pip install ' + ' '.join(required_missing)}",
            facts={"missing": required_missing, "required": list(spec.deps),
                   "optionalMissing": list(dep_facts.get("optionalMissing") or [])}))
    versions: dict[str, str | None] = {}
    for package, floor in spec.minVersions:
        installed = version_of(package)
        versions[package] = installed
        verdict = _meets_floor(installed, floor)
        if installed is None:
            # 没装 ⇒ 上面那条"缺依赖"已经在说，这里不重复报（重复报会让阻断项数虚高）
            if package not in required_missing:
                unchecked.append(f"`{package}` 版本（包没装 ⇒ 判不了下限 {floor}）")
            continue
        if verdict is None:
            unchecked.append(f"`{package}` 版本下限（读到 {installed!r}，判不出数字 ⇒ 不当作通过）")
            continue
        if not verdict:
            blockers.append(_blocker(
                "deps", DEFAULT_CODE["deps"],
                f"`{package}` 版本过低：装的是 {installed}，本阶段要求 ≥ {floor}",
                hint=f"升级：pip install -U \"{package}>={floor}\"",
                facts={"package": package, "installed": installed, "floor": floor}))
    checks["deps"]["facts"] = {"missing": missing, "versions": versions,
                               "optionalMissing": list(dep_facts.get("optionalMissing") or [])}

    # ── ② 权重在不在盘上 / 结构坏没坏 ────────────────────────────────────
    from . import inventory as inv  # noqa: PLC0415

    #: 「必需」只有一处真源：清单的 ``required`` ✓（`loader.plan_component` 的组件计划**不带**它 ✗
    #: ⇒ 少了这张映射，量化判据会把**可选件**当成硬阻断 ✓✗ —— 当时实测 h3 的 `te_nvfp4`
    #: 是 nvfp4 未实现族但**非必需** ✓，按必需判就会把本来能跑的 h3 拦下来 ✗（该件 2026-10-30
    #: 起已能还原 ✓，这条口径仍成立 —— 换成任何一个做不了的族都会踩同一脚 ✓）。
    required_by_key = {str(item.get("key")): bool(item.get("required"))
                       for item in inv.load_catalog()["models"]}
    if readiness is not None:
        ready: Mapping[str, Any] = readiness
    else:
        scanned, failed = _scan("weights", stage, _readiness, root, capacity_gib)
        ready = scanned or {}
        blockers.extend(failed)
    weights_facts = {key: ready.get(key) for key in
                     ("stage", "category", "ready", "totalGiB", "weightsGiB",
                      "vram", "counts", "problems")}
    weights_facts["components"] = [{"key": comp.get("key"), "kind": comp.get("kind"),
                                    "path": comp.get("path"), "bytes": comp.get("bytes"),
                                    "required": required_by_key.get(str(comp.get("key")), False)}
                                   for comp in (ready.get("components") or [])]
    if ready.get("catalogError"):
        blockers.append(_blocker(
            "weights", "E_BAD_CONFIG",
            f"模型清单读不了（{ready['catalogError']}）⇒ 不知道要装哪些组件",
            hint="先修 configs/models.json", facts={"catalogError": ready["catalogError"]}))
    for key in ready.get("missingRequired") or []:
        blockers.append(_blocker(
            "weights", "E_WEIGHT_MISSING", f"必需权重不在盘上：`{key}`",
            hint=f"python -X utf8 scripts/engine_readiness.py --weights（按清单下载 {key}）",
            facts={"key": key}))
    for key in ready.get("brokenRequired") or []:
        blockers.append(_blocker(
            "weights", "E_WEIGHT_CORRUPT", f"必需权重读不了/结构不符：`{key}`",
            hint="这份文件下载不全或不是 safetensors：重新下载，别改参数硬试",
            facts={"key": key}))
    for key in ready.get("suspectRequired") or []:
        warnings.append(f"`{key}` 字节数与清单登记的差得多（能读 ⇒ 放行，但它可疑）")
    if ready.get("ready") is False and not (ready.get("missingRequired") or ready.get("brokenRequired")):
        warnings.append("readiness 报未就绪，但没有点名缺件/坏件（体检不足 ⇒ 不要当成已就绪）")
    if not ready:
        unchecked.append("权重盘点（这一类没查 ⇒ 不知道要装哪些组件）")
    checks["weights"]["facts"] = weights_facts

    # ── ③ 量化族 ────────────────────────────────────────────────────────
    if plans is not None:
        plan: Mapping[str, Any] = plans
    else:
        scanned_plan, failed_plan = _scan("quant", stage, _plans, root, capacity_gib)
        plan = scanned_plan or {}
        blockers.extend(failed_plan)
    families: dict[str, Any] = {}
    unresolved = 0
    for comp in plan.get("components") or []:
        dequant = comp.get("dequantPlan") or {}
        key = str(comp.get("key"))
        #: ⚠️ `supported=False` **不等于"不支持"** ✗✗：`quant.plan_dequant` 对**全精度**文件也回
        #: `supported=False`（`weights=0` ⇒ 没有要反量化的东西 ⇒ 没有什么可支持 ✓）。只盯 supported
        #: 会把 `sdxl_base`（quantScheme=none ✓ 实测）报成"未实现的族"✗ —— 这是本模块自己
        #: 造成过的假阳性 ✓✗。真相判据三条：**有低精度张量**（`weights>0`）＋（家族明确 ✗ 或
        #: 反解标了 supported=False ✓ 或 有 unresolved ✓）。`weights>0` 是入口闸 ✓。
        low_precision = int(dequant.get("weights") or 0) > 0
        if not low_precision:
            continue
        family = dequant.get("family")
        #: nvfp4 的族名不在 `family` 里 ✓（那是「做不了的族」栏 ✓）—— 它在自己的子档 ✓
        nvfp4 = dequant.get("nvfp4") or {}
        name = str((family.get("name") if isinstance(family, dict) else family)
                   or nvfp4.get("format") or "") or "unknown"
        counts = {"lowPrecision": int(dequant.get("weights") or 0),
                  "unresolvedCount": int(dequant.get("unresolvedCount") or 0)}
        families[key] = {"family": name, "supported": bool(dequant.get("supported", True)),
                         **counts}
        if family or dequant.get("supported") is False or counts["unresolvedCount"]:
            unresolved += 1
            message = (f"`{key}` 带低精度权重，但本仓反解不了：族 {name}、"
                       f"未解析 {counts['unresolvedCount']} / 共 {counts['lowPrecision']} 个"
                       + ("（本仓未实现的族）" if family else "（反量化计划判为不支持）"))
            if required_by_key.get(key, False):
                blockers.append(_blocker(
                    "quant", "E_UNSUPPORTED_QUANT", message,
                    hint="换成本仓实现过的量化（convrot / int8 / nvfp4）或换全精度权重；不要指望它跑",
                    facts={"key": key, "family": name, "layouts": dequant.get("layouts"), **counts}))
            else:
                #: 可选件反解不了 ⇒ **不阻断**（不选它就行 ✓），但必须看得见 ✓ ——
                #: 静默放过的话，配置里真选中它时才炸，而那时报的是底层反量化异常 ✗。
                warnings.append(f"{message}（可选件 ⇒ 不阻断，但装配里别选中它）")
    if not plan:
        unchecked.append("量化族（这一类没查 ⇒ 本仓没实现过的族要等真装载才暴露）")
    elif not families:
        unchecked.append("量化族（这份阶段没有低精度组件 ⇒ 无需判）")
    checks["quant"]["facts"] = {"families": families, "unsupportedComponents": unresolved}

    # ── ④ 显存装不装得下 ────────────────────────────────────────────────
    vram = dict(ready.get("vram") or {})
    weights_bytes = int(sum(int(comp.get("bytes") or 0)
                            for comp in (ready.get("components") or [])
                            if required_by_key.get(str(comp.get("key")), False)))
    if weights_bytes:
        #: ``capacity_gib=None`` ⇒ 用清单默认容量（**要标出来** ✓：默认值不是这张卡的实测 ✓✗，
        #: 读者得知道"装不下"是按哪个数算的 ✓）。
        capacity = float(capacity_gib) if capacity_gib is not None else float(inv.DEFAULT_CAPACITY_GIB)
        vram = inv.estimate_vram(weights_bytes, activation_bytes=activation_bytes,
                                 capacity_gib=capacity)
        vram["activationBytes"] = int(activation_bytes)
        vram["capacitySource"] = "given" if capacity_gib is not None else "default"
    elif not vram:
        unchecked.append("显存预算（盘上还没有必需权重 ⇒ 算不出要占多少）")
    if vram and not vram.get("fits", True):
        blockers.append(_blocker(
            "vram", DEFAULT_CODE["vram"],
            f"装不下：需要约 {vram.get('requiredGiB')} GiB，这张卡可用 {vram.get('capacityGiB')} GiB",
            hint=VRAM_HINT, facts=vram))
    checks["vram"]["facts"] = vram

    for kind in COMPAT_KINDS:
        mine = [item for item in blockers if item["kind"] == kind]
        checks[kind]["ok"] = not mine
        checks[kind]["blockers"] = len(mine)
    return {
        "stage": spec.name,
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "unchecked": unchecked,
        "checks": checks,
        "spec": {"category": spec.category, "capabilities": list(spec.capabilities),
                 "deps": list(spec.deps),
                 "minVersions": [list(pair) for pair in spec.minVersions]},
    }


def ensure_compat(stage: str, *, require: tuple[str, ...] | None = None,
                  **kwargs: Any) -> dict[str, Any]:
    """:func:`check_compat` 的**闸门**版：有阻断 ⇒ :class:`CompatError`（不返回 False）。

    ``require`` 选**哪几类算硬阻断**（默认全四类）。⚠️ 为什么需要它 ✗✗：``vram`` 那类是按
    「权重**全精度常驻**」估的 ⇒ 装配里的 ``int8_offload``（权重钉 CPU 锁页内存）与分辨率
    都还没定，闸门在装载时**不知道**这些 ⇒ 一律硬拦会把"开 offload 就能跑"的那些配置
    也挡在门外 ✗。所以装载闸门只硬拦 ``deps`` / ``weights`` / ``quant``（这三类与装配无关 ✓），
    显存那一类仍原样返回（调用方要看就看 ✓），真爆由 OOM 那条带**同一份** :data:`VRAM_HINT` 兜底 ✓。
    """
    report = check_compat(stage, **kwargs)
    kinds = set(COMPAT_KINDS if require is None else require)
    hard = [blocker for blocker in report["blockers"] if blocker["kind"] in kinds]
    report["hardBlockers"] = hard
    if not hard:
        return report
    first = hard[0]
    more = len(hard) - 1
    tail = f"；另有 {more} 条阻断在 detail.blockers" if more else ""
    raise CompatError(
        f"阶段 `{report['stage']}` 装载前就不兼容：{first['message']}{tail}",
        code=str(first["code"]), hint=str(first.get("hint") or ""),
        detail={"stage": report["stage"], "blockers": hard,
                "soft": [blocker for blocker in report["blockers"] if blocker not in hard],
                "unchecked": report["unchecked"]})


# ══════════════════════════════════════════════════════════════════════════
# 注册表自洽（E2.1 的「相等」判据 ＋ 看得见的缺口）
# ══════════════════════════════════════════════════════════════════════════
def registration_problems() -> list[str]:
    """注册表与清单规则的**双向核对** ⇒ 问题清单（空 = 自洽）。

    核三件事：① 集合**相等**（清单有、注册表没有 ⇒ 那个阶段根本没被兼容性判据覆盖；
    注册表有、清单没有 ⇒ 那个阶段永远匹配不到组件、`ready` 空就绪）；② 同名阶段的
    ``category`` 必须一致（不一致 ⇒ 体检的组件集与装载的组件集不是同一批）；
    ③ 能力/依赖不许空（空的阶段声明＝没有声明）。
    """
    from . import inventory as inv  # noqa: PLC0415

    problems: list[str] = []
    filters = {str(name) for name in inv.STAGE_FILTERS}
    registered = set(STAGES)
    for name in sorted(filters - registered):
        problems.append(f"清单阶段 `{name}` 未注册 ⇒ 它没有兼容性判据（装起来全靠运气）")
    for name in sorted(registered - filters):
        problems.append(f"注册表阶段 `{name}` 清单里没有 ⇒ 它匹配到 0 个组件（空就绪，别当成就绪）")
    for name in sorted(registered & filters):
        spec = STAGES[name]
        rule = inv.STAGE_FILTERS[name] or {}
        category = str(rule.get("category") or "")
        if category and category != spec.category:
            problems.append(f"阶段 `{name}` 的分类对不上：注册表 {spec.category!r} vs 清单 {category!r}")
        if not spec.capabilities:
            problems.append(f"阶段 `{name}` 没声明能力（能力页会漏掉它）")
        if not spec.deps:
            problems.append(f"阶段 `{name}` 没声明依赖（缺依赖时无从报起）")
    return problems


def unclaimed_categories() -> dict[str, int]:
    """清单里有、但**没有任何阶段**认领的 ``category`` ⇒ ``{分类: 组件数}``。

    ⚠️ 这不是"必须有 0 条"的判据，而是**看得见的缺口**：某个分类没人认领 ⇒ 那批权重永远不会被
    任何阶段的 ``readiness`` 数到，而 ``ready=True`` 也照旧（空就绪）。当前实测是 ``{"tts": n}``
    —— TTS 走 provider、引擎层不管，属**已知且有意**，但它必须能被看见。
    """
    from . import inventory as inv  # noqa: PLC0415

    claimed = {spec.category for spec in STAGES.values()}
    counts: dict[str, int] = {}
    for item in inv.load_catalog()["models"]:
        category = str(item.get("category") or "").strip()
        if category and category not in claimed:
            counts[category] = counts.get(category, 0) + 1
    return counts


def describe() -> dict[str, Any]:
    """:func:`stages` 的**可读版**（给 ``Capabilities`` / 自检 / 文档用）。"""
    from . import inventory as inv  # noqa: PLC0415

    return {
        "stages": {
            name: {
                "category": spec.category,
                "capabilities": list(spec.capabilities),
                "deps": list(spec.deps),
                "minVersions": [list(pair) for pair in spec.minVersions],
                "note": spec.note,
            }
            for name, spec in sorted(STAGES.items())
        },
        "registered": stages(),
        "filterStages": sorted(str(name) for name in inv.STAGE_FILTERS),
        "problems": registration_problems(),
        "unclaimedCategories": unclaimed_categories(),
        "compatKinds": list(COMPAT_KINDS),
        "vramHint": VRAM_HINT,
    }
