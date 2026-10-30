"""模型参数的**类型转换** —— 移植 ``api/types.go::FormatParams``。

为什么需要它：Modelfile / ``/api/create`` / ``/api/generate`` 里传进来的参数
一律是**字符串**（``"num_ctx": "4096"``），而真正发给推理引擎的必须是
正确类型的 JSON（``4096`` 而不是 ``"4096"``）。Go 侧靠反射读 ``Options``
的 struct tag 决定每个参数该转成什么类型，Python 没有反射可用，所以这里
把那张表**显式抄下来**（值与 ``api.Options`` / ``api.Runner`` 逐字段一致）。

⚠️ 参数表要跟上游同步维护：新增参数不登记会直接报 ``unknown parameter``，
这是故意的 —— 静默吞掉一个拼错的参数比报错更难查。
"""

from __future__ import annotations

import re
from typing import Any

#: ``json tag -> 类型``。``int*`` / ``bool*`` 指 Go 侧的指针类型（可缺省、可显式给值）。
#: 来源：``api/types.go`` 的 ``Options``（13 个预测参数）+ 内嵌的 ``Runner``（7 个加载参数）。
OPTION_KINDS: dict[str, str] = {
    # Options —— 预测阶段
    "num_keep": "int",
    "seed": "int",
    "num_predict": "int",
    "top_k": "int",
    "top_p": "float",
    "min_p": "float",
    "typical_p": "float",  # 已废弃：新请求会被拒，但旧模型参数仍认
    "repeat_last_n": "int",
    "temperature": "float",
    "repeat_penalty": "float",
    "presence_penalty": "float",
    "frequency_penalty": "float",
    "stop": "strlist",
    # Runner —— 加载阶段（必须在装载模型时确定）
    "num_ctx": "int",
    "num_batch": "int",
    "num_gpu": "int",
    "main_gpu": "int*",
    "use_mmap": "bool*",
    "num_thread": "int",
    "draft_num_predict": "int",
}

#: 已废弃、新请求应当拒绝的参数（与 Go 的 ``Deprecated`` 说明对应）
DEPRECATED_PARAMS: frozenset[str] = frozenset({"typical_p"})

_INT_RE = re.compile(r"^[+-]?\d+$")
_TRUE_LITERALS = frozenset({"1", "t", "T", "true", "TRUE", "True"})
_FALSE_LITERALS = frozenset({"0", "f", "F", "false", "FALSE", "False"})


class ParamError(ValueError):
    """参数名未知或取值无法转换。文案与 Go 侧保持一致，便于对照日志。"""


def _parse_int(value: str) -> int:
    # Go 的 strconv.ParseInt 不接受前后空白，Python 的 int() 接受 ⇒ 这里显式收紧
    if not _INT_RE.match(value):
        raise ParamError(f"invalid int value {value!r}")
    return int(value, 10)


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ParamError(f"invalid float value {value!r}") from exc


def _parse_bool(value: str) -> bool:
    if value in _TRUE_LITERALS:
        return True
    if value in _FALSE_LITERALS:
        return False
    raise ParamError(f"invalid bool value {value!r}")


def format_params(params: dict[str, list[str]] | None) -> dict[str, Any]:
    """把 ``{参数名: [字符串值, …]}`` 转成正确类型的 dict。

    :raises ParamError: 参数名未登记，或首个取值无法按该参数的类型解析
    """
    if not params:
        return {}

    out: dict[str, Any] = {}
    for key, values in params.items():
        kind = OPTION_KINDS.get(key)
        if kind is None:
            raise ParamError(f"unknown parameter '{key}'")
        if not values:
            raise ParamError(f"parameter '{key}' has no value")

        first = values[0]
        if kind == "int":
            out[key] = _parse_int(first)
        elif kind == "int*":
            out[key] = _parse_int(first)
        elif kind == "float":
            out[key] = _parse_float(first)
        elif kind == "bool*":
            out[key] = _parse_bool(first)
        elif kind == "strlist":
            out[key] = list(values)
        else:  # pragma: no cover - OPTION_KINDS 自身写错才会到这里
            raise ParamError(f"unknown type {kind} for {key}")

    return out


def reject_deprecated(params: dict[str, Any] | None) -> list[str]:
    """返回请求里出现的废弃参数名（调用方决定是拒绝还是仅告警）。"""
    if not params:
        return []
    return [key for key in params if key in DEPRECATED_PARAMS]


__all__ = [
    "DEPRECATED_PARAMS",
    "OPTION_KINDS",
    "ParamError",
    "format_params",
    "reject_deprecated",
]
