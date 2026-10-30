"""**拉取/推送进度** —— 上游 ``progress`` 包的可计算部分 + ``api.ProgressResponse`` 聚合。

上游 ``progress`` 包是**终端 UI**：``Progress`` 结构体开一个 100ms 的 goroutine，用
``\\033[?2026l``（同步输出模式）、``\\033[?25h``（恢复光标）这套 ANSI 控制序列往 stderr
刷多行进度条。本仓是 FastAPI 后端 + 前端，**没有 stderr 终端**，所以那部分按「不适用」处理：
不搬 ``Progress``/``spinner`` 的 ANSI 渲染，只搬三样真正有语义的东西 ——

* :class:`Bar` —— 进度、速率（滑动 10 桶，1 秒节流）、ETA 的计算。速率算法逐行对等上游
  ``Bar.rate()``：**stopped 后改用全程平均**，未停止时用桶窗口；照抄是因为两条曲线在
  「快下载完但卡住」时差异极大，前端展示跳变会被当成故障。
* :func:`render_bar` —— 纯文本渲染（无 ANSI、无终端宽度探测），供日志/CLI 用。
* :class:`ProgressTracker` —— 把 ``/api/pull``/``/api/push`` 的流式事件按 digest 聚合成
  总进度，这是本仓真正要用的形态（前端轮询 ``GET /api/pull/progress`` 之类）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Iterable

from .format import HumanBytes

__all__ = ["Bar", "DigestProgress", "ProgressTracker", "render_bar"]

#: 上游 ``Bar.maxBuckets = 10``、桶更新节流 1 秒。
MAX_BUCKETS = 10
BUCKET_INTERVAL = 1.0


def _format_duration(seconds: float) -> str:
    """上游 ``progress.formatDuration`` —— 只显示两级单位，超 100 小时折叠成 ``99h+``。"""
    if seconds >= 100 * 3600:
        return "99h+"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60)}m"
    rounded = int(seconds + 0.5)
    if rounded < 60:
        return f"{rounded}s"
    return f"{rounded // 60}m{rounded % 60:02d}s"


class Bar:
    """对应上游 ``progress.Bar``（去掉终端宽度与 ANSI，保留计算语义）。

    ``clock`` 可注入，便于测试确定性地推进时间。
    """

    __slots__ = ("message", "max_value", "initial_value", "current_value",
                 "started", "stopped", "_buckets", "_clock")

    def __init__(self, message: str, max_value: int, initial_value: int = 0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.message = message
        self.max_value = max_value
        self.initial_value = initial_value
        self.current_value = initial_value
        self._clock = clock
        self.started = clock()
        self.stopped: float | None = clock() if initial_value >= max_value else None
        self._buckets: list[tuple[float, int]] = []

    # -- 更新 --
    def set(self, value: int) -> None:
        """更新进度。到顶即 ``stop``；桶按 1 秒节流、最多留 10 个。"""
        if value >= self.max_value:
            value = self.max_value
        self.current_value = value
        if self.current_value >= self.max_value and self.stopped is None:
            self.stopped = self._clock()

        now = self._clock()
        if not self._buckets or now - self._buckets[-1][0] > BUCKET_INTERVAL:
            self._buckets.append((now, value))
            if len(self._buckets) > MAX_BUCKETS:
                self._buckets.pop(0)

    def add(self, delta: int) -> None:
        self.set(self.current_value + delta)

    def stop(self) -> None:
        if self.stopped is None:
            self.stopped = self._clock()

    # -- 读数 --
    @property
    def done(self) -> bool:
        return self.stopped is not None

    def percent(self) -> float:
        if self.max_value > 0:
            return self.current_value / self.max_value * 100
        return 0.0

    def rate(self) -> float:
        """字节/秒。逐行对等上游 ``Bar.rate()``（含 ``Round(time.Second)`` 的取整）。"""
        numerator = 0.0
        denominator = 0.0

        if self.stopped is not None:
            numerator = float(self.current_value - self.initial_value)
            denominator = round(self.stopped - self.started)
        elif not self._buckets:
            pass
        elif len(self._buckets) == 1:
            when, value = self._buckets[0]
            numerator = float(value - self.initial_value)
            denominator = round(when - self.started)
        else:
            first, last = self._buckets[0], self._buckets[-1]
            numerator = float(last[1] - first[1])
            denominator = round(last[0] - first[0])

        return numerator / denominator if denominator else 0.0

    def eta(self) -> float | None:
        """剩余秒数；速率未知时返回 ``None``（**不要**返回 0 —— 那会被画成「已完成」）。"""
        rate = self.rate()
        if rate <= 0:
            return None
        if self.stopped is not None:
            return 0.0
        return float(self.max_value - self.current_value) / rate


def render_bar(bar: Bar, width: int = 80) -> str:
    """纯文本进度条（无 ANSI）。上游那 13/8/10 字符的对齐也照抄了，方便对照日志。"""
    prefix = ""
    if bar.message:
        prefix = bar.message.strip() + " "
    prefix += f"{bar.percent():3.0f}%"

    if not bar.done:
        suffix = f"{HumanBytes(bar.current_value).rjust(6)}/{HumanBytes(bar.max_value).rjust(6)}"
    else:
        suffix = f"{HumanBytes(bar.max_value).rjust(6)}{' ' * 7}"

    rate = bar.rate()
    if not bar.done and rate > 0:
        suffix += "  " + HumanBytes(int(rate)).rjust(6) + "/s"
    else:
        suffix += " " * 10

    remaining = bar.eta()
    if not bar.done and remaining is not None:
        suffix += "  " + _format_duration(remaining).rjust(6)
    else:
        suffix += " " * 8

    span = width - len(prefix) - len(suffix) - 5
    filled = int(span * bar.percent() / 100)
    middle = " ▕" + "█" * max(filled, 0) + " " * max(span - filled, 0) + "▏ "
    return prefix + middle + suffix


# --------------------------------------------------------------------------- #
# 流式进度事件（/api/pull、/api/push）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DigestProgress:
    """单个 blob 的进度（对应 ``api.ProgressResponse`` 里按 ``digest`` 分组的那条）。"""

    digest: str
    total: int = 0
    completed: int = 0

    @property
    def fraction(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0


@dataclass(slots=True)
class ProgressTracker:
    """把 ``/api/pull`` 的流式事件聚合成总进度。

    ⚠️ 上游这里有几个**真实的坑**，聚合时必须照它们的语义处理：

    * 同一个 digest 会**多条重复事件**（先 ``total`` 后多次 ``completed``），只能 upsert；
    * ``status`` 为 ``"success"``/``"verifying sha256 digest"`` 等阶段事件**不带** ``digest``，
      不能被当成「另一个 blob」；
    * ``total`` 缺失（0）的条目**不参与**分母，否则总分母会被 ``0/0`` 搅成负数或虚高。
    """

    blobs: dict[str, DigestProgress] = dc_field(default_factory=dict)
    status: str = ""

    def update(self, event: Any) -> "ProgressTracker":
        """喂一条事件（dict 或带属性的对象）。"""
        digest = _field(event, "digest")
        status = _field(event, "status")
        if status:
            self.status = status
        if not digest:
            return self

        entry = self.blobs.get(digest)
        if entry is None:
            entry = DigestProgress(digest=digest)
            self.blobs[digest] = entry

        total = _field(event, "total")
        completed = _field(event, "completed")
        if total is not None and total > 0:
            entry.total = total
        if completed is not None and completed >= 0:
            entry.completed = completed
        # 有的服务端只给 total 不给 completed，用 status 里的 "pulling manifest" 兜不住；
        # 这里不编造进度，宁可让前端显示「未知」。
        return self

    def update_all(self, events: Iterable[Any]) -> "ProgressTracker":
        for event in events:
            self.update(event)
        return self

    @property
    def total(self) -> int:
        return sum(item.total for item in self.blobs.values() if item.total > 0)

    @property
    def completed(self) -> int:
        return sum(item.completed for item in self.blobs.values() if item.total > 0)

    def percent(self) -> float:
        total = self.total
        return self.completed / total * 100 if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "percent": round(self.percent(), 1),
            "digests": [
                {"digest": item.digest, "total": item.total, "completed": item.completed}
                for item in sorted(self.blobs.values(), key=lambda entry: entry.digest)
            ],
        }


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
