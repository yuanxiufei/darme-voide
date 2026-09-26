"""**训练日志缓冲 + 进度事实** ✓ —— 给前端轮询用的那一小块 ✓。

## 为什么要有它 ✗

参考实现（``reference/lora/LoRAMaster`` ✓）是 NiceGUI 单体 ✓ ⇒ 它直接
``ui.log.push(line)`` ✓ 就把日志显示掉了 ✓ —— 用不到"缓冲 + 增量取"这一层 ✓。
本仓是 **HTTP 后端** ✓ ⇒ 前端要靠轮询取日志 ✓✗，于是必须解决三件事 ✓：

1. **日志不能无限涨** ✓：一次训练几小时、几万到几十万行 ✓✗ ⇒ 只留最近
   :data:`LOG_LIMIT` 行 ✓（**全量在磁盘上** ✓，见 runtime 的 ``logPath`` ✓）。
2. **进度条不许刷屏** ✓：两个上游工具都用 ``tqdm`` ✓（见 ``runner`` 模块头第 4 条 ✓）
   ⇒ 它每秒十几次原地刷新 ✓✗ ⇒ 必须**替换**上一条进度条 ✓，而不是追加 ✓。
   这靠 :attr:`~app.services.lora_train.runner.OutputLine.overwrite` 这个标记 ✓。
3. **增量取** ✓：每次轮询都回整份日志 ✗ ⇒ 带上一个**单调序号** ✓，
   前端只取序号之后的那些行 ✓。

## 进度事实**只认实测格式** ✓（不许凭印象加规则 ✗）

* ``epoch X/Y`` —— musubi-tuner（``musubi_tuner/*_train_network.py`` ✓）与
  sd-scripts（``train_network.py`` 第 1574 行 ``accelerator.print(f"\\nepoch {epoch+1}/{num_train_epochs}\\n")`` ✓）
  **两边都是这个格式** ✓ ⇒ 可认 ✓。
* ``tqdm`` 的百分比/步数 —— sd-scripts ``train_network.py`` 第 1529 行
  ``tqdm(range(...), desc="steps")`` ✓；它的默认 ``bar_format`` 里
  ``l_bar = "{desc}: {percentage:3.0f}%"`` ✓ 紧跟一根 ``|`` ✓ ⇒ 形如
  ``steps:  42%|████▏     | 42/100 [00:10<00:13,  4.04it/s]`` ✓ ⇒ 可认 ✓。
* ⚠️ 认不出就**不编** ✗：某个字段没出现过 ⇒ 回报里**根本没有这个键** ✓
  （本仓判据：``False`` 是"没有证据" ✓，不是"它不存在" ✗ —— 与 ``paths.environment_report`` 同一口径 ✓）。
"""
from __future__ import annotations

import re
import threading
from typing import Callable

from .runner import OutputLine

#: 内存里留多少行 ✓（磁盘上有全量 ✓）
LOG_LIMIT = 800

#: ``epoch 3/200`` ✓（两端都实测过 ✓，见模块头 ✓）
_EPOCH_RE = re.compile(r"^epoch\s+(\d+)\s*/\s*(\d+)\s*$")
#: tqdm 行 ✓ —— 认 ``\d+%`` 与同一行里的 ``n/total`` ✓（见模块头 ✓）
_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%")
_TQDM_COUNT_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def parse_facts(text: str) -> dict[str, int]:
    """从一行输出里认出**实测格式**的进度 ✓ ⇒ 字段子集 ✓（认不出 ⇒ 空字典 ✓，不猜 ✗）。"""
    found: dict[str, int] = {}
    match = _EPOCH_RE.match(text.strip())
    if match:
        epoch, total = int(match.group(1)), int(match.group(2))
        found["epoch"] = epoch
        found["epochs"] = total
        if total > 0:
            found["percent"] = int(epoch * 100 / total)
        return found
    percent_match = _TQDM_PERCENT_RE.search(text)
    count_match = _TQDM_COUNT_RE.search(text)
    if percent_match is None and count_match is None:
        return found
    if percent_match is not None:
        value = int(percent_match.group(1))
        if 0 <= value <= 100:
            found["percent"] = value
    if count_match is not None:
        found["step"] = int(count_match.group(1))
        found["totalSteps"] = int(count_match.group(2))
    return found


class LogBuffer:
    """**有界**日志缓冲 ✓（行数有上限 ✓，进度条原地替换 ✓，读取可增量 ✓）。

    ⚠️ 线程安全 ✓：写的是运行器的读行线程 ✓，读的是 HTTP 线程 ✓ —— 两拨人 ✓✗，
    所以 :meth:`append` / :meth:`snapshot` 共用一个锁 ✓。
    """

    def __init__(self, *, limit: int = LOG_LIMIT,
                 persist: Callable[[str], None] | None = None) -> None:
        self._limit = max(1, int(limit))
        self._persist = persist
        self._lock = threading.RLock()
        self._sequence = 0
        self._lines: list[str] = []
        #: 最近一条**进度条**行 ✓（还没落进 ``_lines`` ✓ —— 下一条普通行或 flush 时才落 ✓）
        self._progress = ""
        self._facts: dict[str, int] = {}

    # ---- 写 ---------------------------------------------------------------
    def append(self, line: OutputLine) -> None:
        """收一行 ✓ —— ``overwrite`` 为真就**替换**上一条进度条 ✓（见模块头第 2 条 ✓）。

        ⚠️ **序号只数真正进 ``_lines`` 的行** ✗✗：进度条的每次刷新**不占号** ✓ ——
        否则「序号 ↔ 下标」的对应关系就断了 ✓✗（``snapshot(since=…)`` 靠它算起点 ✓）。
        进度条的**最新值**由 :attr:`progress` 单独回 ✓，不靠序号 ✓。
        """
        with self._lock:
            self._facts.update(parse_facts(line.text))
            if line.overwrite:
                self._progress = line.text
                return
            if self._progress:
                self._push(self._progress)
                self._progress = ""
            self._push(line.text)

    def flush_progress(self) -> None:
        """把"还挂着的那条进度条"落进正文 ✓（一步跑完时调 ✓，否则最后一条会丢 ✓✗）。"""
        with self._lock:
            if self._progress:
                self._push(self._progress)
                self._progress = ""

    def _push(self, text: str) -> None:
        """⚠️ **必须持锁**调用 ✓。"""
        self._lines.append(text)
        self._sequence += 1
        if len(self._lines) > self._limit:
            del self._lines[: len(self._lines) - self._limit]
        if self._persist is not None:
            try:
                self._persist(text)
            except OSError as err:  # 落盘失败不该把训练带崩 ✓，但**要让人知道** ✓
                print(f"[lora-train] 日志落盘失败：{err}")

    # ---- 读 ---------------------------------------------------------------
    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def facts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._facts)

    def snapshot(self, *, since: int = 0, limit: int | None = None) -> dict[str, object]:
        """增量取一份 ✓ ⇒ ``{"sequence", "lines", "progress", "facts", "truncated"}`` ✓。

        * ``since=0`` ⇒ 取尾部 ``limit`` 行 ✓（首次加载 ✓）；
        * ``since>0`` ⇒ 取序号 **大于** ``since`` 的行 ✓（轮询 ✓）；
          ⚠️ 请求的起点已经被挤掉了 ⇒ ``truncated=True`` ✓ **明说"你漏了一段"** ✓
          （静默少给几行会让前端显示错位 ✓✗）。
        """
        with self._lock:
            if since <= 0:
                wanted = self._lines if limit is None else self._lines[-max(1, int(limit)):]
                return {
                    "sequence": self._sequence,
                    "lines": list(wanted),
                    "progress": self._progress,
                    "facts": dict(self._facts),
                    "truncated": False,
                }
            available = len(self._lines)
            # 序号从 1 开始、**只数进过正文的行** ✓ ⇒ 第 i 行的序号 = oldest + i ✓
            oldest = self._sequence - available + 1
            # 要给的 = 序号 **大于** since 的那些 ⇒ 起点序号 = since + 1 ✓
            first_index = max(0, since + 1 - oldest)
            truncated = since + 1 < oldest
            rows = self._lines[first_index:]
            if limit is not None:
                rows = rows[-max(1, int(limit)):]
            return {
                "sequence": self._sequence,
                "lines": list(rows),
                "progress": self._progress,
                "facts": dict(self._facts),
                "truncated": truncated,
            }

    def text(self, *, limit: int | None = None) -> str:
        """整段文本 ✓（调试/下载用 ✓）。"""
        with self._lock:
            rows = self._lines if limit is None else self._lines[-max(1, int(limit)):]
            rows = [*rows, self._progress] if self._progress else list(rows)
        return "\n".join(rows)


