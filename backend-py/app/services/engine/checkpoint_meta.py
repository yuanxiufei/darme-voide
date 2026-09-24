"""**权重里的内嵌元数据契约** ✓ —— 严格读、坏就拒 ✗（2026-09-24 补 ✓，口径来自逆向 ✓）。

## 为什么值得单独一层
有的权重把「自己是什么」**写在文件里** ✓：safetensors 的 ``metadata`` 里塞一段 JSON ✓
（口径：latent 放大器的检查点带 ``format`` / ``strict_latent_only`` / ``base_config`` /
``config`` / ``step`` ✓）。这类信息比**文件名**可靠得多 ✓ —— 文件名会被人改 ✗、
而元数据是产物自己写的 ✓；但也**只有严格读**才有意义 ✗（松读 ⇒ 拿旧格式当新格式用 ⇒ 静默错 ✓✗）。

## 严格到什么程度（照参考实现的力度 ✓）
* 缺 ``metadata`` ⇒ ``None`` ✓（⚠️ **不是通过** ✗ —— 要不要因此拒由调用方定 ✓）；
* ``metadata`` 不是字符串 / JSON 坏了 ⇒ **分开报** ✓（别一律说「格式不对」✗ —— 排查方向不同 ✓）；
* ``format`` 必须**等于**期望值 ✓，且报错时**把两个值都印出来** ✓；
* 布尔开关（如 ``strict_latent_only``）**必须真是 ``True``** ✓（有键 ≠ 为真 ✓✗）；
* 整数字段（如 ``step``）必须是整数且 ``≥0`` ✓（``"3"`` 不算 ✓ —— 同上，有值 ≠ 是对的类型 ✗）。

## 不猜
* 只**读**契约 ✓，不解释 ``base_config`` / ``config`` 里的字段语义 ✗（那是各自的模型配置 ✓）；
* 额外键**原样保留** ✓（前向兼容：产物多写字段不该把读取方弄崩 ✓）。
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = ["CONTRACT_KEY", "ContractError", "read_contract"]

#: safetensors 里放内嵌契约的那个键 ✓（参考实现口径 ✓）。
CONTRACT_KEY = "metadata"


class ContractError(ValueError):
    """内嵌契约**坏了 / 对不上** ✓ ⇒ 当场报 ✗（松读会让「拿错格式」变成静默错 ✓✗）。"""


def read_contract(metadata: Mapping[str, Any] | None, *, expect_format: str | None = None,
                  required: Sequence[str] = (), truthy: Sequence[str] = (),
                  integers: Sequence[str] = ()) -> dict[str, Any] | None:
    """从 safetensors 的 ``metadata`` 读出内嵌契约 ✓ ⇒ 解析后的 dict ✓。

    * **没有** ``metadata`` / 里面没有契约 ⇒ ``None`` ✓（⚠️ 别当通过 ✗）；
    * ``expect_format`` 给了 ⇒ ``format`` 必须等于它 ✓；
    * ``required`` 里的键必须都在 ✓；``truthy`` 里的键必须**真是 ``True``** ✓；
      ``integers`` 里的键必须是整数且 ``≥0`` ✓（缺了不算错 ✗ —— 用 ``required`` 去要求 ✓）。
    """
    if not isinstance(metadata, Mapping) or CONTRACT_KEY not in metadata:
        return None
    raw = metadata.get(CONTRACT_KEY)
    if not isinstance(raw, str):
        raise ContractError(
            f"safetensors 的 ``{CONTRACT_KEY}`` 必须是**字符串**（里面装 JSON ✓），"
            f"收到 {type(raw).__name__} ✗ —— 这个字段的形状不对，说明产物不是按契约写的 ✓")
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ContractError(f"safetensors 的 ``{CONTRACT_KEY}`` 不是合法 JSON ✗（{err} ✓）"
                            f" —— 文件可能被改过或写入时截断了 ✓") from err
    if not isinstance(contract, Mapping):
        raise ContractError(f"内嵌契约必须是 JSON **对象** ✓，收到 {type(contract).__name__} ✗")
    data = dict(contract)
    if expect_format is not None:
        got = data.get("format")
        if got != expect_format:
            raise ContractError(
                f"内嵌契约的 ``format`` 对不上 ✗：期望 {expect_format!r} ✓、实得 {got!r} ✗"
                f" ⇒ 宁可拒 ✓ —— 拿新格式当旧格式读会**静默错** ✓✗")
    missing = [key for key in required if key not in data]
    if missing:
        raise ContractError(f"内嵌契约缺字段 ✗：{', '.join(missing)} ✓（契约里只写了 "
                            f"{', '.join(sorted(data)) or '（空）'} ✓）")
    for key in truthy:
        if data.get(key) is not True:
            raise ContractError(
                f"内嵌契约的 ``{key}`` 必须**真是 True** ✗，实得 {data.get(key)!r} ✓"
                f" —— 有键 ≠ 为真 ✓✗（松读会把这个开关当已声明 ✓✗）")
    for key in integers:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(
                f"内嵌契约的 ``{key}`` 必须是非负整数 ✗，实得 {value!r}（{type(value).__name__} ✓）"
                f" —— 字符串数字（\"3\" ✓）也不收 ✗：有值 ≠ 是对的类型 ✓✗")
    return data
