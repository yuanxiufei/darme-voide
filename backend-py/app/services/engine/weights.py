"""**权重装载**（torch 侧 ✓）—— 把 safetensors 里的张量装进 ``nn.Module``，并**逐条报出差异** ✓。

## 为什么不直接用 ``load_state_dict`` 了事（这是本模块存在的全部理由 ✓）

``module.load_state_dict(state, strict=False)`` 有**两个**危险行为 ✗：

1. **键对不上时静默放过** ✗ —— 名字差一个字母 ✓ 就是"少装了一层却照常跑" ✓，
   产物**看起来有画面** ✗ 但其实是残缺模型 ✓（本仓最怕这种"像是成功"的失败 ✗）；
2. 它会**改名/改语义地兼容**（``_orig_mod.`` / ``module.`` 前缀 ✓ 等），
   让人误以为"装上了" ✓ —— 我们宁可**先报清楚**再决定 ✓。

⇒ 本模块先**自己算差集** ✓（missing / unexpected / 形状不符 ✓），再调用 ``load_state_dict`` ✓，
最后把结果**如实**回给调用方 ✓：``complete=False`` 就代表"**不是**完整的模型" ✗。

## 与其它层的关系

* :mod:`app.services.engine.safetensors` —— 纯 Python **读头部**（体检用 ✓ 不需要 torch ✓）；
* :mod:`app.services.engine.loader` —— 该不该装 / 装得下吗（加载计划 ✓）；
* 本模块 —— **真的把张量交给模块** ✓。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["WeightLoadReport", "diff_state_dict", "load_module_weights", "save_module_weights"]


@dataclass
class WeightLoadReport:
    """装载结果 ✓（``complete`` 看 **missing 与 shapeMismatch 都为空** ✓ —— unexpected 不影响完整性 ✓）。"""

    path: str = ""
    loadedKeys: int = 0
    sourceKeys: int = 0
    targetKeys: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shapeMismatch: list[dict[str, Any]] = field(default_factory=list)
    renamed: int = 0
    dtypeCasts: dict[str, int] = field(default_factory=dict)
    error: str = ""

    @property
    def complete(self) -> bool:
        """**完整**装载 ⇔ 没有缺键、没有形状不符 ✓（多出来的键只报告不阻断 ✓）。"""
        return not self.error and not self.missing and not self.shapeMismatch

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "complete": self.complete, "loadedKeys": self.loadedKeys,
            "sourceKeys": self.sourceKeys, "targetKeys": self.targetKeys,
            "missing": self.missing[:12], "missingCount": len(self.missing),
            "unexpected": self.unexpected[:12], "unexpectedCount": len(self.unexpected),
            "shapeMismatch": self.shapeMismatch[:6], "shapeMismatchCount": len(self.shapeMismatch),
            "renamed": self.renamed, "dtypeCasts": self.dtypeCasts, "error": self.error,
        }


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def diff_state_dict(module: Any, state: dict[str, Any]) -> WeightLoadReport:
    """只算差集 ✓（不装载 ✓）—— 便于**先看**再决定 ✓。"""
    report = WeightLoadReport(targetKeys=len(module.state_dict()), sourceKeys=len(state))
    own = module.state_dict()
    for key, tensor in state.items():
        if key not in own:
            report.unexpected.append(key)
            continue
        if tuple(own[key].shape) != tuple(tensor.shape):
            report.shapeMismatch.append({
                "key": key, "expected": list(own[key].shape), "got": list(tensor.shape)})
            continue
        report.loadedKeys += 1
    report.missing = [key for key in own if key not in state]
    return report


def _apply_key_map(state: dict[str, Any], key_map: dict[str, str]) -> tuple[dict[str, Any], int]:
    """按**显式给出的**正则映射改名 ✓（多条规则按顺序 ✓；没匹配上的原样保留 ✓）。

    ⚠️ 不做"猜前缀"式的自动改名 ✗ —— 那正是上面说的第 2 个危险行为 ✓。
    """
    renamed = 0
    out: dict[str, Any] = {}
    compiled = [(re.compile(pattern), target) for pattern, target in key_map.items()]
    for key, tensor in state.items():
        new_key = key
        for pattern, target in compiled:
            if pattern.search(key):
                new_key = pattern.sub(target, key)
                break
        if new_key != key:
            renamed += 1
        out[new_key] = tensor
    return out, renamed


def load_module_weights(module: Any, path: str | Path, *, dtype: Any = None,
                        device: str | None = None,
                        key_map: dict[str, str] | None = None) -> WeightLoadReport:
    """把 ``path`` 的权重装进 ``module`` ✓，并**如实**报告差异 ✓（不静默 ✗）。"""
    target = Path(path)
    report = WeightLoadReport(path=str(target), targetKeys=len(module.state_dict()))
    if not target.exists():
        report.error = f"权重文件不存在：{target} ✗（先按 `loader.plan_stage` 的报告去装 ✓）"
        return report

    try:
        from safetensors.torch import load_file  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - 依赖缺失时给出可行动信息 ✓
        report.error = (f"官方 safetensors 未安装（{err} ✓）⇒ 先装它 ✓；"
                        f"注意纯 Python 读取器**只**能体检、不能装载 ✗")
        return report

    try:
        state = load_file(str(target), device=str(device or "cpu"))
    except Exception as err:  # noqa: BLE001 - 加载期异常类型不稳定，统一转成报告 ✓
        report.error = f"读取权重失败：{type(err).__name__}: {err}"
        return report

    report.sourceKeys = len(state)
    if key_map:
        state, report.renamed = _apply_key_map(state, key_map)

    casts: dict[str, int] = {}
    if dtype is not None:
        converted = {}
        for key, tensor in state.items():
            if tensor.dtype != dtype:
                casts[str(tensor.dtype).replace("torch.", "")] = \
                    casts.get(str(tensor.dtype).replace("torch.", ""), 0) + 1
                tensor = tensor.to(dtype)
            converted[key] = tensor
        state = converted
    report.dtypeCasts = casts

    summary = diff_state_dict(module, state)
    report.missing = summary.missing
    report.unexpected = summary.unexpected
    report.shapeMismatch = summary.shapeMismatch
    report.loadedKeys = summary.loadedKeys
    if report.shapeMismatch:
        # ⚠️ 形状不符 ⇒ **不装** ✓（装了也没用 ✗ 而且 `load_state_dict` 会直接抛错 ✓）
        report.error = (f"{len(report.shapeMismatch)} 个张量形状不符 ✗ ⇒ 模型结构与权重不匹配，"
                        f"已**中止装载**（不是「部分装上」✗）：{report.shapeMismatch[:2]}")
        return report

    # 形状都对了才真装 ✓；`assign=False` 表示沿用模块自己的参数对象（保留 device ✓）
    result = module.load_state_dict(state, strict=False)
    still_missing = [key for key in result.missing_keys if key not in report.missing]
    if still_missing:  # pragma: no cover - 理论上上面已算过 ✓
        report.missing.extend(still_missing)
    return report


def save_module_weights(module: Any, path: str | Path, *,
                        metadata: dict[str, str] | None = None) -> str:
    """把模块权重写成 safetensors ✓（自检里造「真权重」用 ✓；也可用于导出 ✓）。"""
    from safetensors.torch import save_file  # noqa: PLC0415

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {key: tensor.detach().to("cpu").contiguous()
             for key, tensor in module.state_dict().items()}
    save_file(state, str(target), metadata=dict(metadata or {}))
    return str(target)
