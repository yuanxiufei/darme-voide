"""**人类可读的数字/时间格式化** —— 上游 ``format`` 包（``bytes.go``/``format.go``/``time.go``）的对等实现。

⚠️ 两个「看起来一样、实际不同」的地方，是本模块存在的理由：

* ``HumanBytes`` 用**十进制** KB/MB/GB（1 KB = 1000 B），而 ``HumanBytes2`` 用**二进制**
  KiB/MiB/GiB（1 KiB = 1024 B）。混用会让「下载了 2.4 GB」和实际字节数对不上；
* ``HumanBytes`` 的精度是**分档**的：``>= 10`` 不带小数（``15 GB``）、有小数且 < 10 带一位
  （``1.5 GB``）、整数 < 10 不带小数（``2 GB``）。写死 ``%.1f`` 会输出 ``15.0 GB``。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

__all__ = [
    "Billion",
    "GibiByte",
    "GigaByte",
    "HumanBytes",
    "HumanBytes2",
    "HumanNumber",
    "HumanTime",
    "HumanTimeLower",
    "KibiByte",
    "KiloByte",
    "MebiByte",
    "MegaByte",
    "Million",
    "TeraByte",
    "Thousand",
    "humanDuration",
]

Thousand = 1000
Million = Thousand * 1000
Billion = Million * 1000

KiloByte = 1000
MegaByte = KiloByte * 1000
GigaByte = MegaByte * 1000
TeraByte = GigaByte * 1000

KibiByte = 1024
MebiByte = KibiByte * 1024
GibiByte = MebiByte * 1024


def HumanBytes(value: int) -> str:
    """十进制单位（``1 KB = 1000 B``），与模型体积的展示口径一致。"""
    if value >= TeraByte:
        amount, unit = value / TeraByte, "TB"
    elif value >= GigaByte:
        amount, unit = value / GigaByte, "GB"
    elif value >= MegaByte:
        amount, unit = value / MegaByte, "MB"
    elif value >= KiloByte:
        amount, unit = value / KiloByte, "KB"
    else:
        return f"{value} B"

    if amount >= 10:
        return f"{int(amount)} {unit}"
    if amount != math.trunc(amount):
        return f"{amount:.1f} {unit}"
    return f"{int(amount)} {unit}"


def HumanBytes2(value: int) -> str:
    """二进制单位（``1 KiB = 1024 B``），用于显存/内存这类按 2 的幂计的场合。"""
    if value >= GibiByte:
        return f"{value / GibiByte:.1f} GiB"
    if value >= MebiByte:
        return f"{value / MebiByte:.1f} MiB"
    if value >= KibiByte:
        return f"{value / KibiByte:.1f} KiB"
    return f"{value} B"


def HumanNumber(value: int) -> str:
    """大数缩写：``1200`` → ``1K``、``1500000`` → ``1.50M``、``2500000000`` → ``2.5B``。"""
    if value >= Billion:
        number = value / Billion
        return f"{number:.0f}B" if number == math.floor(number) else f"{number:.1f}B"
    if value >= Million:
        number = value / Million
        return f"{number:.0f}M" if number == math.floor(number) else f"{number:.2f}M"
    if value >= Thousand:
        return f"{value / Thousand:.0f}K"
    return str(value)


def humanDuration(delta: timedelta | float) -> str:
    """把时长说成话（``"About a minute"`` / ``"4 hours"``），同上游 ``humanDuration``。"""
    seconds = int(delta.total_seconds()) if isinstance(delta, timedelta) else int(delta)

    if seconds < 1:
        return "Less than a second"
    if seconds == 1:
        return "1 second"
    if seconds < 60:
        return f"{seconds} seconds"

    minutes = seconds // 60
    if minutes == 1:
        return "About a minute"
    if minutes < 60:
        return f"{minutes} minutes"

    # 上游是 math.Round（四舍五入、远离零），不是 Python 的银行家舍入
    hours = int(math.floor(seconds / 3600 + 0.5))
    if hours == 1:
        return "About an hour"
    if hours < 48:
        return f"{hours} hours"
    if hours < 24 * 7 * 2:
        return f"{hours // 24} days"
    if hours < 24 * 30 * 2:
        return f"{hours // 24 // 7} weeks"
    if hours < 24 * 365 * 2:
        return f"{hours // 24 // 30} months"
    return f"{hours // 24 // 365} years"


def HumanTime(when: datetime | None, zero_value: str) -> str:
    """相对时间描述（``"3 hours ago"``）。``when`` 为 ``None`` 时返回 ``zero_value``。"""
    if when is None:
        return zero_value
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - when
    # 上游是整数除法 int(delta.Hours())/24/365 < -20，用浮点会在边界上差一天
    if int(delta.total_seconds() // 3600) // 24 // 365 < -20:
        return "Forever"
    if delta.total_seconds() < 0:
        return humanDuration(-delta.total_seconds()) + " from now"
    return humanDuration(delta) + " ago"


def HumanTimeLower(when: datetime | None, zero_value: str) -> str:
    return HumanTime(when, zero_value).lower()
