"""成本单价目录 —— 移植 ``services/cost-catalog.ts``（整文件）。

定位：让「多集长剧 + QC 自动重拍」的烧钱情况**可量化**。单价是按公开定价的**估算值**，
真实账单以厂商结算为准。

匹配规则：``provider`` → ``[模型前缀 → 单价]``，与 ``model`` 做**包含匹配**、取**首个命中**；
也可在 ``ai_service_configs.settings.pricing`` 里覆盖（对象 ``{unit, price}`` 或纯数字=单价）。
"""

from __future__ import annotations

from typing import Any

from ..core.response import js_number, js_round

#: 无 pricing 覆盖时，各任务类型默认按什么单位计数
DEFAULT_UNITS: dict[str, str] = {
    "image": "image",
    "video": "second",
    "audio": "char",
    "text": "k-token",
}

#: 单价目录：provider → [(模型前缀, {unit, price})]。**顺序有意义**（首个命中即返回）
COST_CATALOG: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "minimax": [
        ("hailuo", {"unit": "second", "price": 0.3}),
        ("h3", {"unit": "second", "price": 0.3}),
        ("video", {"unit": "second", "price": 0.3}),
        ("image", {"unit": "image", "price": 0.1}),
        ("speech", {"unit": "char", "price": 0.003}),
        ("abab", {"unit": "k-token", "price": 0.01}),
        ("mini-max", {"unit": "k-token", "price": 0.01}),
    ],
    "volcengine": [
        ("seedream-4", {"unit": "image", "price": 0.3}),
        ("seedream-3", {"unit": "image", "price": 0.15}),
        ("seedream", {"unit": "image", "price": 0.15}),
        ("seedance-1-0-pro", {"unit": "second", "price": 0.5}),
        ("seedance-1-0", {"unit": "second", "price": 0.35}),
        ("seedance", {"unit": "second", "price": 0.35}),
        ("doubao", {"unit": "k-token", "price": 0.003}),
    ],
    "vidu": [("vidu", {"unit": "second", "price": 0.3})],
    "ali": [
        ("wanx", {"unit": "image", "price": 0.2}),
        ("wan", {"unit": "second", "price": 0.3}),
        ("qwen", {"unit": "k-token", "price": 0.002}),
        ("cosyvoice", {"unit": "char", "price": 0.002}),
    ],
    "openai": [
        ("gpt-4o-mini", {"unit": "k-token", "price": 0.006}),
        ("gpt-4o", {"unit": "k-token", "price": 0.03}),
        ("tts", {"unit": "char", "price": 0.002}),
    ],
    "openrouter": [
        ("openai", {"unit": "k-token", "price": 0.02}),
        ("anthropic", {"unit": "k-token", "price": 0.03}),
        ("deepseek", {"unit": "k-token", "price": 0.003}),
    ],
    "deepseek": [("deepseek", {"unit": "k-token", "price": 0.002})],
    "chatfire": [
        ("claude", {"unit": "k-token", "price": 0.03}),
        ("gpt", {"unit": "k-token", "price": 0.02}),
    ],
    "ollama": [],
    "local": [],
}


def _round_money(value: float) -> float:
    """``Math.round(x * 10000) / 10000`` —— 必须用 js_round（Python 的 round 是银行家舍入）。"""
    return js_round(value * 10000) / 10000


def parse_pricing_override(
    settings: dict[str, Any] | None, default_unit: str
) -> dict[str, Any] | None:
    """解析 ``settings.pricing`` 覆盖：纯数字视为「单价 + 默认单位」。"""
    if not isinstance(settings, dict):
        return None
    pricing = settings.get("pricing")
    if pricing is None:
        return None
    if isinstance(pricing, (int, float)) and not isinstance(pricing, bool):
        return {"unit": default_unit, "price": float(pricing)}
    if isinstance(pricing, dict):
        price = js_number(pricing.get("price"))
        if price is not None and price >= 0:
            return {"unit": pricing.get("unit") or default_unit, "price": price}
    return None


def estimate_cost(
    service_type: str,
    provider: str,
    model: str,
    units: float | None,
    settings: dict[str, Any] | None = None,
) -> float | None:
    """估算一次调用的成本（元）；查不到单价返回 None。

    ``units`` = 图片张数 / 视频秒数 / 音频字符数 / 千 token 数。
    """
    units_num = js_number(units)
    if not model or units_num is None or units_num <= 0:
        return None

    provider_key = (provider or "").lower()
    # `DEFAULT_UNITS[serviceType] || 'request'`：未识别类型退化成 'request'
    default_unit = DEFAULT_UNITS.get(service_type) or "request"

    override = parse_pricing_override(settings, default_unit)
    if override:
        return _round_money(override["price"] * units_num)

    entries = COST_CATALOG.get(provider_key) or []
    needle = str(model).lower()
    for key, rule in entries:
        if key in needle:
            return _round_money(rule["price"] * units_num)
    return None
