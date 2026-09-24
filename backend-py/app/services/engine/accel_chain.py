"""**加速链**：把「一串加速节点」当**配置** ✓ 而不是代码 ✗（2026-09-24 补 ✓）。

## 解决什么
生成链上常要挂几个加速插件（LoRA 4 步 / TE-Speed / SAGE / Spectrum ✓）。写死在代码里 ✗ ⇒
换插件就得改代码 ✗；而**写错的代价很隐性** ✗：节点没注册 ⇒ 提交之后只拿到一句看不懂的错 ✓✗。
⇒ 拆成两半：**配置**（:class:`ChainEntry` ✓）+ **提交前校验**（:func:`validate_chain` ✓
对照节点注册表逐项查 ⇒ ``ok`` / ``warn`` / ``error`` / ``unchecked`` ✓；判 ``error`` 的项
**不许提交** ✗ —— 见 :attr:`ChainStatus.blocks_submit` ✓）。

## 接线硬规则（判 ``error`` ✓，不是「先提交看看」✗）
``kind`` 决定这个节点接管哪条流 ✓：
* ``lora``：必须**同时有** ``model`` 与 ``clip`` 输入 ✓（如 ``LoraLoader`` ✓）；
* ``model_only``：至少有一个 ``model`` 输入 ✓。
接错 ⇒ 进不了生成链 ✓✗ ⇒ 当场判错 ✓，理由点名是「缺哪条流」✓。

## 不猜（本模块的边界 ✓）
* **不联网** ✗：节点注册表由调用方从 ``/object_info`` 取好传进来 ✓
  （见 :mod:`app.local_services.h3.comfyui_client` ✓）；
* ⚠️ **没给注册表** ⇒ 判 ``unchecked`` ✓ 并写明「**没查**」✗ —— **不是** ``ok`` ✓
  （本仓纪律：没查 ≠ 通过 ✗）；
* ⚠️ **输出槽位**只能在 :func:`build_chain_nodes` 里从注册表 ``output`` 里数出来 ✓；
  没注册表就**报错** ✗（猜槽位 = 把线接错 ✗✗）；
* ⚠️ 参数**语义**不核 ✗（只核「名字在不在 ✓ / 必填缺不缺 ✓」）⇒ 值填错只有真提交才知道 ✓。

## 口径来源
默认链的**成员与插件节点名**取自参考实现的内置默认链 ✓（2026-09-24 逆向 ✓）；
⚠️ 参数值**不照填** ✗ —— 照插件自己 ``INPUT_TYPES`` 的定义给 ✓（见 :data:`DEFAULT_CHAIN` ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["CHAIN_KINDS", "ChainBuild", "ChainEntry", "ChainError", "ChainIssue", "ChainStatus",
           "DEFAULT_CHAIN", "KIND_REQUIRED_INPUTS", "build_chain_nodes", "parse_chain",
           "validate_chain"]

#: 允许的 ``kind`` ✓（决定这个节点接管哪条流 ✓）。
CHAIN_KINDS: tuple[str, ...] = ("lora", "model_only")

#: ``kind`` → 该节点**必须**具备的输入名 ✓（接错就是进不了生成链 ✓✗）。
KIND_REQUIRED_INPUTS: dict[str, tuple[str, ...]] = {"lora": ("model", "clip"),
                                                   "model_only": ("model",)}


class ChainError(ValueError):
    """加速链配置**读不进来 / 接不起来** ✓ ⇒ 当场报 ✗（别把半条链提交上去 ✗）。"""


@dataclass(frozen=True)
class ChainEntry:
    """加速链上的一环 ✓（配置形态 ✓，与 ComfyUI 的节点实例无关 ✓）。"""

    id: str
    class_type: str
    kind: str = "model_only"
    label: str = ""
    third_party: bool = False
    default_on: bool = True
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "class_type": self.class_type,
                "kind": self.kind, "third_party": self.third_party,
                "default_on": self.default_on, "inputs": dict(self.inputs)}

    @classmethod
    def from_dict(cls, raw: Any) -> ChainEntry:
        if not isinstance(raw, Mapping):
            raise ChainError(f"加速链的每一项必须是对象 ✗（收到 {type(raw).__name__} ✓）")
        entry_id = str(raw.get("id") or "").strip()
        class_type = str(raw.get("class_type") or "").strip()
        if not entry_id:
            raise ChainError(f"加速链每一项都要有 ``id`` ✗（前端勾选/开关按它对齐 ✓）：{dict(raw)!r}")
        if not class_type:
            raise ChainError(f"加速项 {entry_id!r} 缺 ``class_type`` ✗ ⇒ 没它不知道该起哪个节点 ✓")
        kind = str(raw.get("kind") or "model_only").strip()
        if kind not in CHAIN_KINDS:
            raise ChainError(
                f"加速项 {entry_id!r} 的 kind={kind!r} 不认识 ✗ ⇒ 只能是 "
                + " / ".join(CHAIN_KINDS) + " ✓（它决定这个节点接管哪条流 ✓）")
        for flag in ("third_party", "default_on"):
            if flag in raw and not isinstance(raw[flag], bool):
                raise ChainError(
                    f"加速项 {entry_id!r} 的 {flag}={raw[flag]!r} 不是布尔值 ✗"
                    f" ⇒ 「有值」不等于「为真」✓（本项目在别处踩过这个坑 ✓✗）")
        inputs = raw.get("inputs") or {}
        if not isinstance(inputs, Mapping):
            raise ChainError(f"加速项 {entry_id!r} 的 inputs 必须是对象 ✗"
                             f"（收到 {type(inputs).__name__} ✓）")
        return cls(id=entry_id, class_type=class_type, kind=kind,
                   label=str(raw.get("label") or ""), third_party=bool(raw.get("third_party")),
                   default_on=bool(raw.get("default_on", True)),
                   inputs={str(key): value for key, value in inputs.items()})


def parse_chain(raw: Any) -> list[ChainEntry]:
    """``{\"chain\": [...]}`` 或**裸数组** ✓ → ``[ChainEntry, ...]`` ✓（**id 不许重复** ✗）。"""
    if isinstance(raw, Mapping):
        items = raw.get("chain")
        if items is None:
            raise ChainError("加速链配置里没有 ``chain`` ✗（要么给 ``{\"chain\": [...]}`` ✓、"
                             "要么直接给数组 ✓）")
    else:
        items = raw
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ChainError(f"加速链必须是数组 ✗（收到 {type(items).__name__} ✓）")
    entries = [ChainEntry.from_dict(item) for item in items]
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise ChainError(f"加速链里 id={entry.id!r} 出现两次 ✗ ⇒ 前端勾选会串行 ✗"
                             f"（换个 id ✓）")
        seen.add(entry.id)
    return entries


def _default_chain() -> list[ChainEntry]:
    #: 内置默认链 ✓（成员与插件节点名：口径来自参考实现 ✓；⚠️ 参数值**不照填** ✗ ——
    #: 「4 步 turbo LoRA」那个文件名是参考实现给的默认值 ✓，本机没有就自己换 ✓）。
    return [
        ChainEntry(id="lora", class_type="LoraLoader", kind="lora", label="LoRA 4步",
                   third_party=False, default_on=True,
                   inputs={"lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                           "strength_model": 1.0, "strength_clip": 1.0}),
        ChainEntry(id="tespeed", class_type="TESpeedMiniMaxH3", kind="model_only",
                   label="TE-Speed", third_party=True, default_on=True),
        ChainEntry(id="sage", class_type="PathchSageAttentionKJ", kind="model_only",
                   label="SAGE", third_party=True, default_on=True),
        ChainEntry(id="spectrum", class_type="SpectrumApplyMiniMaxH3", kind="model_only",
                   label="Spectrum（频谱加速）", third_party=True, default_on=True),
    ]


#: 内置默认链 ✓（⚠️ ``third_party=True`` 的那几个是**第三方插件** ✓ ⇒ 界面要标注 ✓，
#: 而且要「改了就自己勾选剔除」的自由 ✓ —— 见 :attr:`ChainEntry.default_on` ✓）。
DEFAULT_CHAIN: tuple[ChainEntry, ...] = tuple(_default_chain())


@dataclass(frozen=True)
class ChainIssue:
    """一条问题 ✓（``level`` 只有 ``error`` / ``warn`` ✓ —— 「没查」不在这里 ✗，它是状态 ✓）。"""

    level: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message}


@dataclass(frozen=True)
class ChainStatus:
    """一环的校验结论 ✓（``level``：``ok`` / ``warn`` / ``error`` / ``unchecked`` ✓）。"""

    id: str
    class_type: str
    kind: str
    label: str = ""
    third_party: bool = False
    default_on: bool = True
    level: str = "unchecked"
    issues: tuple[ChainIssue, ...] = ()

    @property
    def blocks_submit(self) -> bool:
        """⚠️ 判 ``error`` 的项**不许提交** ✗；``warn`` 可以提交（参数会被忽略 / 缺必填会报 ✓）；
        ``unchecked`` = **没查** ✗ ⇒ 由调用方决定要不要拦（⚠️ 别当通过 ✗）。"""
        return self.level == "error"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "classType": self.class_type, "label": self.label, "kind": self.kind,
                "thirdParty": self.third_party, "defaultOn": self.default_on, "level": self.level,
                "issues": [issue.to_dict() for issue in self.issues],
                "blocksSubmit": self.blocks_submit}


def _declared_inputs(info: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """注册表里某一类的 ``(required, optional, 参数表)`` ✓（形状不对就当空 ✓ —— 宁判 warn ✗）。"""
    block = info.get("input") if isinstance(info, Mapping) else None
    table: dict[str, Any] = {}
    required: list[str] = []
    optional: list[str] = []
    if isinstance(block, Mapping):
        for key, names in (("required", required), ("optional", optional)):
            part = block.get(key)
            if isinstance(part, Mapping):
                for name, spec in part.items():
                    names.append(str(name))
                    table[str(name)] = spec
    return required, optional, table


def _has_default(spec: Any) -> bool:
    """这个输入有没有**默认值** ✓（``["FLOAT", {"default": 1.0}]`` ⇒ 有 ✓）。

    ⚠️ 判据很重要 ✗：ComfyUI 里「required **且带默认值**」的参数**不传也行**（节点自己填 ✓）
    ⇒ 把它算成「缺必填」会让**每条真链都多出一堆噪音 warn** ✗✗（2026-09-24 接真注册表时踩到 ✓）。
    """
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) > 1:
        options = spec[1]
        return isinstance(options, Mapping) and "default" in options
    return False


def _input_type(spec: Any) -> str | None:
    """``INPUT_TYPES`` 里一个输入声明的**类型名** ✓（``["MODEL", {...}]`` ⇒ ``MODEL`` ✓）。"""
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and spec:
        head = spec[0]
        return str(head).upper() if isinstance(head, str) else None
    if isinstance(spec, str):
        return spec.upper()
    return None


def validate_chain(chain: Sequence[ChainEntry],
                   node_types: Mapping[str, Any] | None = None) -> list[ChainStatus]:
    """对照**节点注册表**逐项校验 ✓ ⇒ 每环一条 :class:`ChainStatus` ✓。

    ``node_types`` 就是 ``GET /object_info`` 的返回 ✓（可只给用到的几类 ✓）；
    ⚠️ **给 ``None`` ⇒ 全部判 ``unchecked``（没查 ✗）** ✓ —— 不许当成通过 ✗。
    """
    statuses: list[ChainStatus] = []
    for entry in chain:
        issues: list[ChainIssue] = []
        if node_types is None:
            statuses.append(ChainStatus(id=entry.id, class_type=entry.class_type, kind=entry.kind,
                                        label=entry.label, third_party=entry.third_party,
                                        default_on=entry.default_on, level="unchecked"))
            continue
        info = node_types.get(entry.class_type)
        if not isinstance(info, Mapping):
            statuses.append(ChainStatus(
                id=entry.id, class_type=entry.class_type, kind=entry.kind, label=entry.label,
                third_party=entry.third_party, default_on=entry.default_on, level="error",
                issues=(ChainIssue("error", f"ComfyUI 未注册节点「{entry.class_type}」✗ ⇒ "
                                            f"确认插件已安装并**完全重启** ComfyUI ✓"
                                            f"（刷新网页不算 ✗）"),)))
            continue
        required, optional, table = _declared_inputs(info)
        declared = set(required) | set(optional)
        wired = set(KIND_REQUIRED_INPUTS[entry.kind])
        # ⚠️⚠️ 这两个「接管流」的输入（``model`` / ``clip`` ✓）**由链供给** ✓ ⇒ 它们不在
        #    ``entry.inputs`` 里是**正常**的 ✗ —— 若算成「缺必填」，**每一环都会报 warn** ⇒
        #    真正的问题（节点没注册 / 接错流 ✓）被淹没 ✓✗（本套自检 ⑤‴ 钉住这点 ✓）。
        self_wired = sorted(name for name in wired if name in entry.inputs)
        if self_wired:
            issues.append(ChainIssue(
                "error", "不许自己接流（会**绕开前一级** ✓✗）：" + ", ".join(self_wired)
                         + " —— 这两条流由**链**接管 ✓"))
        unknown = sorted(name for name in entry.inputs
                         if name not in declared and name not in wired)
        if unknown:
            issues.append(ChainIssue("warn", "参数不存在（会被忽略）：" + ", ".join(unknown)))
        missing = sorted(name for name in required
                         if name not in entry.inputs and name not in wired
                         and not _has_default(table.get(name)))
        if missing:
            issues.append(ChainIssue("warn", "缺少必填参数（该参数没有默认值时提交会被拒）："
                                             + ", ".join(missing)))
        for stream in KIND_REQUIRED_INPUTS[entry.kind]:
            if stream not in declared:
                issues.append(ChainIssue(
                    "error", f"kind={entry.kind} 要求这个节点有 ``{stream}`` 输入 ✗，而"
                             f"「{entry.class_type}」没有 ⇒ 它会**进不了生成链** ✓✗"
                             f"（换节点 / 改 kind ✓）"))
                continue
            got = _input_type(table.get(stream))
            if got is not None and got != stream.upper():
                issues.append(ChainIssue(
                    "error", f"``{stream}`` 输入的类型是 {got} ✗，不是 {stream.upper()} ✓"
                             f" ⇒ 接上去会拿错流 ✓✗"))
        level = "error" if any(issue.level == "error" for issue in issues) else (
            "warn" if issues else "ok")
        statuses.append(ChainStatus(id=entry.id, class_type=entry.class_type, kind=entry.kind,
                                    label=entry.label, third_party=entry.third_party,
                                    default_on=entry.default_on, level=level,
                                    issues=tuple(issues)))
    return statuses


@dataclass(frozen=True)
class ChainBuild:
    """接线结果 ✓（**:dict:`nodes` 可直接并进 ``/prompt`` 的 API 图 ✓**）。"""

    nodes: dict[str, dict[str, Any]]
    model: list[Any]
    clip: list[Any]
    enabled: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]      # (id, 原因 ✓)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "model": self.model, "clip": self.clip,
                "enabled": list(self.enabled),
                "skipped": [{"id": key, "reason": reason} for key, reason in self.skipped]}


def _output_slot(info: Mapping[str, Any], wanted: str) -> int:
    outputs = info.get("output")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        names = [str(item).upper() for item in outputs]
        if wanted.upper() in names:
            return names.index(wanted.upper())
    raise ChainError(
        f"节点「{info.get('name') or '?'}」的输出里找不到 {wanted} 槽 ✗ ⇒ 接不了线 ✓"
        f"（⚠️ 本仓**不猜**槽位 ✗ —— 猜错就是把线接到别的输出上 ✓✗）")


def build_chain_nodes(chain: Sequence[ChainEntry], *, node_types: Mapping[str, Any],
                      model: Sequence[Any], clip: Sequence[Any],
                      statuses: Sequence[ChainStatus] | None = None,
                      enabled: Mapping[str, bool] | None = None,
                      prefix: str = "accel") -> ChainBuild:
    """把链**串起来** ✓ ⇒ 可直接并进 API 图的节点字典 + 最终 ``model`` / ``clip`` 指针 ✓。

    * ``model`` / ``clip``：上游给的 ``[节点id, 输出槽]`` ✓（如 UNETLoader / CLIPLoader 的输出 ✓）；
    * 跳过规则 ✓：判 ``error`` 的项**自动跳过** ✗（并在 ``skipped`` 里给理由 ✓）；
      ``enabled`` 里显式 ``False`` 的项跳过 ✓（没给就看 ``default_on`` ✓）；
    * ⚠️ **必须给注册表** ✗ —— 输出槽位靠它数出来 ✓（没给 ⇒ 报错 ✓ 不猜 ✗）。
    """
    if node_types is None:
        raise ChainError("接线必须有节点注册表 ✗ ⇒ 本仓**不猜**输出槽位 ✓"
                         "（见 :func:`validate_chain` 的 ``node_types`` ✓）")
    levels = {status.id: status.level for status in (statuses or [])}
    nodes: dict[str, dict[str, Any]] = {}
    model_ref: list[Any] = [str(model[0]), int(model[1])]
    clip_ref: list[Any] = [str(clip[0]), int(clip[1])]
    enabled_ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    for entry in chain:
        level = levels.get(entry.id)
        if level == "error":
            skipped.append((entry.id, "校验判 error ✗ ⇒ 不许提交 ✓"))
            continue
        want = enabled.get(entry.id) if enabled is not None else None
        if want is False or (want is None and not entry.default_on):
            skipped.append((entry.id, "未勾选 ✓"))
            continue
        info = node_types.get(entry.class_type)
        if not isinstance(info, Mapping):
            skipped.append((entry.id, f"节点「{entry.class_type}」不在注册表里 ✗"))
            continue
        if "model" in entry.inputs or "clip" in entry.inputs:
            raise ChainError(
                f"加速项 {entry.id!r} 的 inputs 里自己写了 model/clip ✗ ⇒ 那两条流由"
                f"**链**接管 ✓（自己接会绕开前一级 ✓✗）")
        inputs: dict[str, Any] = dict(entry.inputs)
        inputs["model"] = list(model_ref)
        node_id = f"{prefix}_{entry.id}"
        model_ref = [node_id, _output_slot(info, "MODEL")]
        if entry.kind == "lora":
            inputs["clip"] = list(clip_ref)
            clip_ref = [node_id, _output_slot(info, "CLIP")]
        nodes[node_id] = {"class_type": entry.class_type, "inputs": inputs}
        enabled_ids.append(entry.id)
    return ChainBuild(nodes=nodes, model=model_ref, clip=clip_ref,
                      enabled=tuple(enabled_ids), skipped=tuple(skipped))
