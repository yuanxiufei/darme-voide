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

## 训练产物契约（E4.2 ✓ 2026-10-30）

LoRA / 微调产物**必须自述三件事** ✓，缺任一 ⇒ **读取时拒绝** ✓，**不补默认值** ✗：

| 字段 | 语义 | 判定 |
|---|---|---|
| ``base_model`` | 基座模型（哪个权重训出来的） | 非空字符串 |
| ``train_config`` | 训练配置（学习率/秩/步数那一套） | **非空对象** ✓（字符串 JSON 不收 ✗ —— 那等于没解析 ✓✗） |
| ``version`` | 产物版本 | 非空字符串或非负整数（``True`` 不算 ✓✗） |

⚠️ 三个字段**都不是**「有键就算」✗：``""`` / ``{}`` / ``None`` 一律按缺处理 ✓（与 ``truthy`` /
``integers`` 同一条纪律：有键 ≠ 有值 ≠ 是对的类型 ✓✗）。

⚠️ **自称口径**：只有 ``format`` 以 ``training`` 开头（如 ``training-v1``）的契约才算训练产物 ✓
—— 放大器的 ``h3-latent-upscaler-v1`` 也带 ``base_config``/``config``，但它是**另一套**契约 ✓，
拿这三条去要求它会把能用的权重拒在门外 ✓✗（症状是「装不上」而不是「报错」✓✗）。
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = ["CONTRACT_KEY", "TRAINING_FIELDS", "TRAINING_FORMAT_PREFIX", "ContractError",
           "is_training_contract", "read_contract", "read_training_contract",
           "training_field_problems"]

#: safetensors 里放内嵌契约的那个键 ✓（参考实现口径 ✓）。
CONTRACT_KEY = "metadata"

#: 训练产物自述格式的**前缀** ✓（``training-v1`` ✓）—— 判「这是不是训练产物」只认这一条 ✓。
TRAINING_FORMAT_PREFIX = "training"

#: 训练产物**必须**自述的三个字段 ✓（顺序＝报错顺序 ✓）。
TRAINING_FIELDS: tuple[str, ...] = ("base_model", "train_config", "version")


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


def _contract_of(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """宽松读出契约（不查 ``format`` ✓）；没有 ⇒ ``None`` ✓。"""
    if not isinstance(metadata, Mapping) or CONTRACT_KEY not in metadata:
        return None
    return read_contract(metadata)


def is_training_contract(metadata: Mapping[str, Any] | None) -> bool:
    """这个文件是不是**自述**为训练产物 ✓（``format`` 以 ``training`` 开头 ✓）。

    ⚠️ 判据只认自述格式 ✗，不认「有没有 ``base_model`` 键」✗ —— 后者会让**全缺**的产物
    判成「不是训练产物 ⇒ 不用查」✓✗，正好漏掉 E4.2 要拦的那种文件 ✓。
    """
    data = _contract_of(metadata)
    if data is None:
        return False
    return str(data.get("format") or "").startswith(TRAINING_FORMAT_PREFIX)


def training_field_problems(contract: Mapping[str, Any]) -> list[str]:
    """逐字段判定训练契约（**只报不改** ✓）⇒ 问题清单（空＝齐 ✓）。

    单独成函数是为了**体检**能用它：报告要说出「缺哪几个」（一次列全 ✓），而加载路径要**拒绝** ✓
    —— 同一套判据两处复用 ✓，各写一份必然漂 ✓✗。
    """
    problems: list[str] = []
    base = contract.get("base_model")
    if not isinstance(base, str) or not base.strip():
        problems.append(
            f"``base_model`` 必须是非空字符串（说明基座是哪个权重）✗，实得 {base!r}"
            f"（{type(base).__name__}）")
    config = contract.get("train_config")
    if not isinstance(config, Mapping) or not config:
        problems.append(
            f"``train_config`` 必须是非空对象（学习率/秩/步数那一套）✗，实得 {config!r}"
            f"（{type(config).__name__}）—— 字符串 JSON 不收：那等于**没解析**，读的人会以为字段齐了 ✓✗")
    version = contract.get("version")
    if isinstance(version, bool) or not isinstance(version, (str, int)):
        problems.append(
            f"``version`` 必须是非空字符串或非负整数 ✗，实得 {version!r}"
            f"（{type(version).__name__}）—— ``True`` 不算版本号 ✓✗")
    elif isinstance(version, str) and not version.strip():
        problems.append("``version`` 是空字符串 ✗ —— 有键 ≠ 有值 ✓✗")
    elif isinstance(version, int) and version < 0:
        problems.append(f"``version`` 不能是负数 ✗，实得 {version}")
    return problems


def read_training_contract(metadata: Mapping[str, Any] | None, *,
                           expect_format: str | None = None) -> dict[str, Any]:
    """读训练产物契约 ✓ ⇒ 三字段齐才返回 ✓；否则 :class:`ContractError` ✗（**不补默认值** ✓）。

    * ``expect_format`` 给了 ⇒ 严格相等 ✓（调用方知道自己是第几版契约）；
      没给 ⇒ 只要求 ``format`` 以 ``training`` 开头 ✓（族判，不是版判 ✓）。
    * **没有**内嵌契约 ⇒ 也**拒绝** ✗（与 :func:`read_contract` 的「没有 ⇒ ``None``」不同 ✓）：
      对训练产物来说「没自述」和「自述不全」一样不可用 —— 补默认值等于替产物编历史 ✓✗。
    """
    data = _contract_of(metadata)
    if data is None:
        raise ContractError(
            f"没有内嵌契约 ✗ —— 训练产物必须自述 {', '.join(TRAINING_FIELDS)} ✓"
            f"（缺了就拒绝，本仓**不补默认值** ✓✗）")
    if expect_format is not None:
        if data.get("format") != expect_format:
            raise ContractError(
                f"内嵌契约的 ``format`` 对不上 ✗：期望 {expect_format!r} ✓、实得 "
                f"{data.get('format')!r} ✗ ⇒ 宁可拒 ✓")
    elif not str(data.get("format") or "").startswith(TRAINING_FORMAT_PREFIX):
        raise ContractError(
            f"这不是训练产物的自述格式 ✗：``format`` 实得 {data.get('format')!r} ✓，"
            f"必须以 {TRAINING_FORMAT_PREFIX!r} 开头 ✓ —— 放大器那类契约另有字段，别拿这三条去要求它 ✓✗")
    problems = training_field_problems(data)
    if problems:
        missing = [key for key in TRAINING_FIELDS if key not in data]
        raise ContractError(
            "训练产物元数据不完整 ✗：" + "；".join(problems)
            + (f"（**整段缺**：{', '.join(missing)} ✓）" if missing else "")
            + " ⇒ 拒绝读取 ✓ —— 不补默认值 ✓✗")
    return data
