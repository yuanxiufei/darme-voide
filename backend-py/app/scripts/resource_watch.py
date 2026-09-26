#!/usr/bin/env python3
r"""**负载观测** ✓ —— 逐秒记下「谁在干活」✗：**GPU 用了几成 / 整机 CPU 用了几核** ✓。

为什么需要（用户原话 ✓ 2026-09-26）
  「这几次都是 CPU 占比高、GPU 几乎没有用到，造成 CPU 负载过高」✓✗ —— 而在此之前，本仓**没有
  任何**能拿出这句话**数字**的东西 ✗：任务管理器看的是瞬时曲线 ✓，`torch` 自述的是它自己的配置 ✓，
  两边都对不上「这一次到底谁忙」✓✗。口径要求也是现成的（用户定的 ✓）：**两侧都参与、谁也别闲着 ✓，
  但都不许被打满过载 ✗✗** ⇒ 判据必须有**峰值**，不是"某一瞬间看着还行" ✓。

它做什么
  每 ``--interval`` 秒采一行 ✓：
    * ``GPU util%`` ✓ + 显存 ``used/total GiB`` ✓ —— 来自 ``nvidia-smi`` ✓（零第三方依赖 ✓，
      驱动自带的命令行工具 ✓；路径优先 ``$NVIDIA_SMI`` ✓ → ``PATH`` ✓ → Windows 标准位置 ✓）；
    * ``整机 CPU%`` ✓ 与**核等效** ✓ —— Windows 走 ``kernel32!GetSystemTimes`` ✓（ctypes ✓），
      Linux 走 ``/proc/stat`` ✓；⚠️ 两者都没有 ⇒ **明说探不到** ✗（**不编一个 0 糊过去** ✗）；
    * ``--run`` 时另报**子进程**的墙钟与 CPU 秒数 ✓（``resource.getrusage(RUSAGE_CHILDREN)`` ✓）
      ⇒ 「CPU 是它在烧 ✓ 还是 GPU 在算 ✓」一眼能分 ✓。
  结束时给**总账**：峰值 / 均值 + 过载判定 ✓。

⚠️ ``--run`` 会把**本仓口径的 CPU 线程预算**灌给子进程 ✓（``app.core.cpu_budget`` ✓ 同一个口径 ✓，
  不在这里另写一套 ✗）—— 这就是「跑重负载前先估算负载、两侧都留裕量」里那段可执行的 ✓。

用法::

    python backend-py/app/scripts/resource_watch.py --seconds 30
    python backend-py/app/scripts/resource_watch.py --run "python backend-py/tests/run_all.py"
    python backend-py/app/scripts/resource_watch.py --seconds 15 --json

退出码：0 = 采样完成且**两侧都没见过载** ✓；1 = 采到过载（某侧 ≥ ``--overload`` % ✓）或压根没 GPU ✓。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # ⚠️ `resource` 是 **Unix-only** ✗ —— Windows 上 `import` 就 `ModuleNotFoundError` ✓（实测）
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None  # type: ignore[assignment]

BACKEND_PY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_PY))

from app.core import cpu_budget  # noqa: E402

#: 判「过载」的阈值 ✓（两侧同一把尺子 ✓ —— 用户口径是「都不许打满」✓，没说要按侧给不同值 ✓）
OVERLOAD_PERCENT = 90.0

#: 「本仓口径」来源提示 ✓
BUDGET_ENV = cpu_budget.ENV_OVERRIDE


# ── GPU ─────────────────────────────────────────────────────────────────────
def nvidia_smi() -> str | None:
    """``nvidia-smi`` 在哪 ✓ —— 优先显式 ``$NVIDIA_SMI`` ✓，再 ``PATH`` ✓，再 Windows 标准位置 ✓。

    ⚠️ 最后那条是**父进程早于驱动**（``PATH`` 还没刷到）时的兜底 ✓：``SystemRoot\\System32`` 是
    **Windows 自身**的标准位置 ✓ —— **不是本机专有盘符** ✓（写 ``D:\\…`` 那种才是禁区 ✗）。
    """
    explicit = os.environ.get("NVIDIA_SMI")
    if explicit:
        return explicit if Path(explicit).is_file() else None  # 显式给了却没这个文件 ⇒ 当没有 ✓
    found = shutil.which("nvidia-smi")
    if found:
        return found
    root = os.environ.get("SystemRoot")
    if root:
        candidate = Path(root) / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def gpu_sample(binary: str) -> dict[str, Any] | None:
    """采一帧 GPU ✓（探不到 ⇒ ``None`` ✓ —— 不编数字 ✗）。"""
    try:
        out = subprocess.run(
            [binary, "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.splitlines()[0] if out else ""
    parts = [piece.strip() for piece in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        util, mem_util, used_mib, total_mib = (float(parts[0]), float(parts[1]),
                                               float(parts[2]), float(parts[3]))
    except ValueError:
        return None
    return {"utilPercent": util, "memUtilPercent": mem_util,
            "memUsedGiB": used_mib / 1024.0, "memTotalGiB": total_mib / 1024.0}


def gpu_owners(binary: str) -> list[str]:
    """谁占着显存 ✓（结论用得上 ✓：例如「显存被别的服务占着 ✗、而它自己没在算 ✓」）。

    ⚠️ 必须**滤掉** ``used_memory = N/A`` 的那些 ✗ —— 实测（2026-09-26✓）：``--query-compute-apps``
    在 Windows 上会把**一屏图形上下文**（explorer / Chrome / 终端…✅）也算进来 ✓，全标 ``N/A`` ✓；
    照单全收 ⇒ 报出「31 个进程占着显存」这种**看着像事实的假事实** ✓✗。
    判据就是那个内存字段**必须是数字** ✓（N/A ⇒ 不是计算进程 ⇒ 不计 ✓）。
    """
    try:
        out = subprocess.run(
            [binary, "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[str] = []
    for line in out.splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        rows.append(f"{parts[0]}（{Path(parts[1]).name} {int(parts[2])} MiB）")
    return rows


# ── 整机 CPU ────────────────────────────────────────────────────────────────
class WindowsCpu:
    """Windows：``GetSystemTimes`` ✓（ctypes ✓ 标准库 ✓ —— 不引 psutil ✗）。"""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        self._get = kernel32.GetSystemTimes
        self._counter = ctypes.c_ulonglong

    def raw(self) -> tuple[int, int]:
        """返回 ``(busy, total)`` ✓（单位 100ns ✓）。⚠️ kernel 时间**已含** idle ✗ ⇒ 要减掉 ✓。"""
        idle, kernel, user = self._counter(), self._counter(), self._counter()
        ok = self._get(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            raise OSError(f"GetSystemTimes 失败 ✗（错误码 {ctypes.get_last_error()} ✓）")
        return (kernel.value + user.value) - idle.value, kernel.value + user.value


class ProcStatCpu:
    """Linux：``/proc/stat`` 第一行 ✓。"""

    def raw(self) -> tuple[int, int]:
        with open("/proc/stat", encoding="ascii") as handle:
            fields = handle.readline().split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values) - idle, sum(values)


def cpu_reader() -> Any | None:
    """拿一个能读整机 CPU 计数器的对象 ✓；**这个平台没有** ⇒ ``None`` ✓（调用方必须明说 ✗）。"""
    try:
        if sys.platform == "win32":
            return WindowsCpu()
        if sys.platform.startswith("linux"):
            return ProcStatCpu()
    except (OSError, AttributeError):  # pragma: no cover - 平台/权限差异
        return None
    return None


# ── 采样 ────────────────────────────────────────────────────────────────────
def sample_once(reader: Any, binary: str | None) -> dict[str, Any]:
    """一帧事实 ✓（读不到的那一项**就是 ``None``** ✓，由渲染层说清 ✗）。"""
    row: dict[str, Any] = {"at": time.monotonic()}
    if reader is not None:
        row["cpuRaw"] = reader.raw()
    row["gpu"] = gpu_sample(binary) if binary else None
    return row


def cpu_percent(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    """两次读数的差值 ⇒ 整机 CPU 占用 ✓（拿不到读数 ⇒ ``None`` ✓）。"""
    if "cpuRaw" not in previous or "cpuRaw" not in current:
        return None
    (busy0, total0), (busy1, total1) = previous["cpuRaw"], current["cpuRaw"]
    delta = total1 - total0
    if delta <= 0:
        return None
    return max(0.0, min(100.0, (busy1 - busy0) / delta * 100.0))


def _windows_handle_cpu_seconds(handle: int) -> float | None:
    """Windows：``GetProcessTimes`` 读该**句柄**的累计 CPU ✓（100ns ⇒ 秒 ✓）。

    ⚠️ 句柄必须按 ``c_void_p`` 传 ✗ —— 当普通 ``int`` 传会被截成 32 位 ✓✗（64 位句柄直接失效 ✓）。
    ⚠️ 用 ``Popen`` 持有的句柄而不是 pid ✓：子进程**已经退出**时 pid 可能已被回收 ✓✗，
    而 Popen 手里那个句柄在 ``wait()`` 之前一直有效 ✓ ⇒ 退出的进程照样读得到 ✓。
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    counter = ctypes.c_ulonglong
    creation, exited, kernel, user = counter(), counter(), counter(), counter()
    ok = kernel32.GetProcessTimes(ctypes.c_void_p(handle), ctypes.byref(creation),
                                  ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user))
    if not ok:
        return None
    return (kernel.value + user.value) / 1e7


def child_cpu_seconds(child: Any) -> float | None:
    """子进程累计 CPU 秒 ✓；**这个平台探不到** ⇒ ``None`` ✓（明说 ✗，不编 0 ✗）。"""
    if _resource is not None:
        usage = _resource.getrusage(_resource.RUSAGE_CHILDREN)
        return float(usage.ru_utime + usage.ru_stime)
    handle = getattr(child, "_handle", None) if sys.platform == "win32" else None
    return _windows_handle_cpu_seconds(int(handle)) if handle is not None else None


def render_row(elapsed: float, percent: float | None, gpu: dict[str, Any] | None,
               cores: int) -> str:
    """人读一行 ✓ —— ⚠️ 探不到的项要**写成「探不到」** ✗，不许留个空格让人以为是 0 ✓✗。"""
    if percent is None:
        cpu_text = "CPU 探不到 ✗"
    else:
        cpu_text = f"CPU {percent:5.1f}%（{percent / 100.0 * cores:5.2f}/{cores} 核）"
    if gpu is None:
        gpu_text = "GPU 探不到 ✗"
    else:
        gpu_text = (f"GPU {gpu['utilPercent']:5.1f}%（显存 {gpu['memUsedGiB']:5.2f}/"
                    f"{gpu['memTotalGiB']:.2f} GiB）")
    return f"[+{elapsed:5.1f}s] {gpu_text}  {cpu_text}"


def main() -> int:
    parser = argparse.ArgumentParser(description="负载观测：GPU 用了几成 / 整机 CPU 用了几核 ✓")
    parser.add_argument("--seconds", type=float, default=30.0, help="采样时长（秒 ✓ 默认 30 ✓）")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔（秒 ✓ 默认 1 ✓）")
    parser.add_argument("--run", help="要**顺带**跑的命令 ✓（会带上本仓的 CPU 线程预算 ✓）")
    parser.add_argument("--overload", type=float, default=OVERLOAD_PERCENT,
                        help=f"判过载的阈值百分比（默认 {OVERLOAD_PERCENT:g} ✓ 两侧同尺 ✓）")
    parser.add_argument("--json", action="store_true", help="把全部样本打成 JSON ✓（给自动化用 ✓）")
    args = parser.parse_args()

    binary = nvidia_smi()
    reader = cpu_reader()
    budget = cpu_budget.describe()
    cores = int(budget["logicalCores"])

    if not args.json:
        print("── 观测环境 ───────────────────────────────────────")
        print(f"GPU 探针：{binary or '**探不到 nvidia-smi** ✗（设 $NVIDIA_SMI=<路径> ✓ 或确认驱动已装 ✓）'}")
        print(f"CPU 读法：{'探不到 ✗（本平台没有可用的整机计数器 ✓）' if reader is None else '系统计数器 ✓'}"
              f"　逻辑核 {cores} ✓")
        print(f"CPU 线程预算：{budget['threads']}/{cores} ✓（{budget['source']} ✓；"
              f"{BUDGET_ENV}=N 可显式覆盖 ✓）")
        if binary:
            owners = gpu_owners(binary)
            print(f"GPU 计算进程：{len(owners)} 个 ✓（图形上下文已滤掉 ✓ 它们的内存字段是 N/A ✗）"
                  + ("" if not owners else "　" + "｜".join(owners[:6])
                     + ("…" if len(owners) > 6 else "")))
        print("── 逐秒 ───────────────────────────────────────────")

    child: subprocess.Popen[bytes] | None = None
    child_env = {**os.environ, **cpu_budget.env_for_child()}
    if args.run:
        child = subprocess.Popen(args.run, shell=True, env=child_env)  # noqa: S602 - 本仓自用工具 ✓

    samples: list[dict[str, Any]] = []
    cpu_series: list[float] = []
    gpu_series: list[float] = []
    overloaded: list[str] = []
    started = time.monotonic()
    previous = sample_once(reader, binary)

    while True:
        elapsed = time.monotonic() - started
        if elapsed >= args.seconds:
            break
        time.sleep(max(0.05, args.interval))
        current = sample_once(reader, binary)
        percent = cpu_percent(previous, current)
        gpu = current["gpu"]
        previous = current
        row = {"elapsed": round(time.monotonic() - started, 2), "cpuPercent": percent,
               "gpu": gpu}
        samples.append(row)
        if percent is not None:
            cpu_series.append(percent)
            if percent >= args.overload:
                overloaded.append(f"CPU 峰值 {percent:.1f}% ≥ {args.overload:g}% ✗")
        if gpu is not None:
            gpu_series.append(float(gpu["utilPercent"]))
            if float(gpu["utilPercent"]) >= args.overload:
                overloaded.append(f"GPU 峰值 {gpu['utilPercent']:.1f}% ≥ {args.overload:g}% ✗")
        if not args.json:
            print(render_row(row["elapsed"], percent, gpu, cores))

    verdict: dict[str, Any] = {}
    if args.run and child is not None:
        code = child.wait()
        wall = time.monotonic() - started
        cpu_seconds = child_cpu_seconds(child)
        verdict = {"childExitCode": code, "childWallSeconds": round(wall, 2),
                   "childCpuSeconds": None if cpu_seconds is None else round(cpu_seconds, 2),
                   "childCpuCores": (round(cpu_seconds / wall, 2)
                                     if cpu_seconds is not None and wall > 0 else None)}

    summary = {
        "gpuSamples": len(gpu_series),
        "gpuPeakPercent": max(gpu_series) if gpu_series else None,
        "gpuMeanPercent": round(sum(gpu_series) / len(gpu_series), 1) if gpu_series else None,
        "cpuSamples": len(cpu_series),
        "cpuPeakPercent": max(cpu_series) if cpu_series else None,
        "cpuMeanPercent": round(sum(cpu_series) / len(cpu_series), 1) if cpu_series else None,
        "logicalCores": cores,
        "budgetThreads": budget["threads"],
        "overloaded": sorted(set(overloaded)),
        **verdict,
    }

    if args.json:
        print(json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False, indent=2))
    else:
        print("── 总账 ───────────────────────────────────────────")
        if summary["gpuSamples"]:
            print(f"GPU：峰值 {summary['gpuPeakPercent']:.1f}% ✓　均值 {summary['gpuMeanPercent']:.1f}% ✓"
                  f"　（{summary['gpuSamples']} 帧 ✓）")
        else:
            print("GPU：**一帧都没采到** ✗ ⇒ 上面那句「GPU 没用到」在本次**无法证实** ✗")
        if summary["cpuSamples"]:
            print(f"CPU：峰值 {summary['cpuPeakPercent']:.1f}% ✓　均值 {summary['cpuMeanPercent']:.1f}% ✓"
                  f"　（≈ {summary['cpuMeanPercent'] / 100 * cores:.2f}/{cores} 核 ✓，"
                  f"预算 {summary['budgetThreads']} ✓）")
        else:
            print("CPU：**一帧都没采到** ✗（本平台没有整机计数器 ✓）")
        if verdict:
            cpu_text = ("探不到 ✗（这个平台既没有 RUSAGE_CHILDREN ✓、也拿不到句柄 ✓）"
                        if verdict["childCpuSeconds"] is None
                        else f"{verdict['childCpuSeconds']}s ✓（≈ {verdict['childCpuCores']} 核等效 ✓）")
            print(f"子进程：退出码 {verdict['childExitCode']} ✓　墙钟 {verdict['childWallSeconds']}s ✓　"
                  f"CPU {cpu_text}")
        if summary["overloaded"]:
            print("⚠️ 过载：" + "；".join(summary["overloaded"]))
        else:
            print("✓ 两侧都**没到**阈值 ⇒ 符合「都参与、都不打满」✓")

    if summary["overloaded"]:
        return 1
    if binary is None or not summary["gpuSamples"]:
        return 1  # 采不到 GPU ⇒ **不许当成"GPU 很闲"** ✗（那是两种完全不同的结论 ✓）
    return 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 GBK ⇒ 不 reconfigure 就是第一条 print 崩 ✗（`check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
