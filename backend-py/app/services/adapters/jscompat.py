"""适配器层共用的 **JS 语义垫片**。

适配器是从 TS 逐字移植过来的纯函数（只构建请求 / 解析响应，自己不发起 HTTP），
里面大量用到 JS 特有的取值与判空语义。集中放在这里，避免每处手写一遍而写歪。
"""
from __future__ import annotations

import base64
import binascii
import json
import random
import re
from typing import Any

__all__ = [
    "add_query_param",
    "as_dict",
    "dig",
    "gcd",
    "is_pure_hex",
    "js_base64_to_hex",
    "js_json_stringify",
    "js_length",
    "js_num_str",
    "js_parse_int",
    "random_seed",
    "split_xy",
]


def as_dict(value: Any) -> dict[str, Any]:
    """把「可能是 None / 字符串 / 数字」的响应归一成字典。

    JS 里对非对象点号取值只会得到 ``undefined``（``"abc".task_id`` → undefined），
    而 Python 的 ``{"a":1}.get`` 遇到字符串会 ``AttributeError``。适配器都要先过这一层。
    """
    return value if isinstance(value, dict) else {}


def js_length(value: Any) -> int | None:
    """镜像 JS 读 ``.length``：**只有字符串与数组有**，其余是 ``undefined``（这里 None）。

    ⚠️ 别用 ``len()`` 直接替：``len({})`` 是 0，而 JS 的 ``({}).length`` 是 undefined
    —— 在 ``x.length === 0`` 这类判断里结论正好相反。
    """
    if isinstance(value, (list, str)):
        return len(value)
    return None


def dig(obj: Any, *path: Any) -> Any:
    """镜像 JS 的可选链 ``a?.b?.[0]?.c`` —— 任一层落空即返回 None。

    适配器的响应解析全是这种链式取值，手写 ``a["b"][0]["c"]`` 会在缺字段时抛
    ``KeyError``/``IndexError``，而 JS 只会安静地得到 ``undefined`` 再走 ``||`` 兜底。
    """
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(key, int):
            if not isinstance(cur, list) or key < 0 or key >= len(cur):
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
    return cur


def split_xy(size: Any) -> tuple[Any, Any]:
    """镜像 ``const [w, h] = size.split('x')`` 的解构。

    超出部分丢弃；不足则对应位置是 ``undefined``（这里给 None）——
    ``"1920"`` 得到 ``("1920", None)``，原 TS 的 ``if (w && h)`` 会在 h 上判假。
    """
    if size is None:
        return None, None
    parts = str(size).split("x")
    return (parts[0] if len(parts) > 0 else None), (parts[1] if len(parts) > 1 else None)


_PARSE_INT = re.compile(r"^[+-]?\d+")


def js_parse_int(raw: Any) -> int | None:
    """镜像 JS ``parseInt``：取前导数字、向零截断；解析不出来得 ``NaN``（这里 None）。

    ⚠️ 与 ``float()``/``int()`` 都不同：``parseInt('1920abc')`` 是 1920，
    ``parseInt('1e3')`` 是 1，``parseInt('abc')`` 是 NaN。
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw == raw and raw not in (float("inf"), float("-inf")) else None
    match = _PARSE_INT.match(str(raw).lstrip())
    return int(match.group(0)) if match else None


def js_num_str(value: Any) -> str:
    """镜像 JS 的「数字 → 字符串」。

    ⚠️ 关键差异：Python ``f"{16.0}"`` 得 ``"16.0"``，而 JS ``String(16)`` 得 ``"16"``。
    画幅比例（``${w / gcd}:${h / gcd}``）全靠这个，差一点就成了 ``"16.0:9.0"``。
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def js_json_stringify(value: Any) -> str:
    """镜像 JS ``JSON.stringify``：**紧凑分隔符**（无空格）+ **不转义非 ASCII**。

    Ali 两个适配器的报错信息里把响应体截 200 字符塞进消息，用 Python 默认的
    ``json.dumps`` 会多出空格、中文变 ``\\uXXXX``，报错文本与 Node 不一致。
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def is_pure_hex(text: str) -> bool:
    """镜像 ``/^[0-9a-fA-F]+$/.test(raw.slice(0, 100))``（空串不算）。"""
    return bool(text) and re.fullmatch(r"[0-9a-fA-F]+", text) is not None


def js_base64_to_hex(raw: str) -> str:
    """镜像 ``Buffer.from(raw, 'base64').toString('hex')``。

    ⚠️ Node 的 base64 解码是**宽容**的：忽略非法字符、容忍缺失的 ``=`` 填充；
    而 Python 的 ``b64decode`` 遇到填充不对会直接抛错。这里先按 Buffer 的规矩
    清洗 + 补齐填充，行为就对上了。
    """
    text = re.sub(r"[^A-Za-z0-9+/=]", "", str(raw)).rstrip("=")
    text += "=" * (-len(text) % 4)
    try:
        return base64.b64decode(text, validate=False).hex()
    except (binascii.Error, ValueError):
        return ""


def gcd(a: float, b: float) -> float:
    """镜像 JS 的递归 ``gcd``（``b === 0 ? a : gcd(b, a % b)``）。"""
    return a if b == 0 else gcd(b, a % b)


def random_seed() -> int:
    """镜像 ``Math.floor(Math.random() * 2147483647)``（[0, 2147483646]）。

    独立成函数是为了**可测**：Ali 两个适配器用它填 ``seed``，测试里替换掉就能拿到
    确定性请求体。
    """
    return random.randrange(2147483647)


def add_query_param(url: str, name: str, value: str) -> str:
    """镜像 ``url.searchParams.set(name, value)`` 之后再 ``url.toString()``。

    用 ``urlencode`` 而不是手拼，因为 JS 的 ``URLSearchParams`` 是按
    ``application/x-www-form-urlencoded`` 序列化的（空格 → ``+``）——
    Gemini 系适配器把 apiKey 放在 query 里，编码规则必须一致。
    """
    from urllib.parse import urlencode

    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({name: value})}"
