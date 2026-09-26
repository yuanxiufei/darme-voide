"""**子进程运行器** —— 训练/预缓存真正是怎么被跑起来的 ✓（LoRAMaster 移植 ✓）。

出处（参考实现 ✓）：``reference/lora/LoRAMaster/wan_lora_train.py`` 等 5 个脚本里的
``subprocess.Popen(..., stdout=PIPE, stderr=STDOUT, text=True, env=env, encoding='utf-8',
errors='ignore')`` ✓、``terminate_process_tree()`` ✓、``stop_caching()`` / ``stop_train()`` ✓。

本仓与参考实现**刻意不同**的地方 ✓（四条都写清理由 ✓）：
1. **不依赖 ``psutil``** ✓：参考实现用 ``psutil.Process(pid).children(recursive=True)`` 递归杀子进程 ✗
   ⇒ 本仓改用操作系统自带的整树杀 ✓（Windows ``taskkill /T`` ✓ / POSIX ``killpg`` ✓）
   —— 本仓纪律是**不引新第三方包** ✓，而这两条都不需要额外依赖 ✓。
2. **取消分两段** ✓：先「请它自己退」✓，超时再「强制杀」✓ ——
   参考实现只调 ``terminate()`` ✗（Windows 上 ``terminate()`` = 直接 ``TerminateProcess`` ✓ 也是硬杀 ✓，
   而它的**孙进程**（``accelerate launch`` 起的那个 python ✓）能不能一起走全靠 psutil 那一次递归 ✓✗）。
3. **CPU 线程预算接线** ✓：本仓重负载前要钳住并行度 ✓（口径只有一处：
   :mod:`app.core.cpu_budget` ✓）⇒ 子进程 env 里带上 ``OMP/MKL/OPENBLAS…`` ✓，
   免得"训练进程按逻辑核数开满、把整机拉满"✓✗（参考实现这一处**没做** ✗）。
4. **进度条不刷屏** ✓：两个上游工具都用 ``tqdm`` ✓（sd-scripts ``train_network.py`` 第 1529 行
   ``tqdm(..., desc="steps")`` ✓；musubi-tuner 同样 ✓）—— 它靠裸 ``\\r`` 原地刷新 ✓。
   参考实现按 ``text=True`` 读行 ✗ ⇒ 每一次刷新都变成一条**新日志行** ✗（一小时几万条 ✓✗），
   而且 ``errors='ignore'`` 会把坏字节**静默丢掉** ✗。
   本仓改成**按字节读 + 自己按 ``\\r`` / ``\\n`` 切** ✓ ⇒ 每行带一个 :attr:`OutputLine.overwrite` 标记 ✓
   （``True`` = 它是进度条的一次刷新 ✓ ⇒ 上层应当**替换**上一条进度条 ✓，不是追加 ✓），
   解码用 ``errors='replace'`` ✓（坏字节变成 U+FFFD ✓ 看得见 ✓，而不是无声消失 ✓）。
"""
from __future__ import annotations

import codecs
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence

from app.core import cpu_budget

from .errors import LoraTrainConfigError, LoraTrainRunError
from .paths import ResolvedTool, build_env

#: Windows：给子进程单独开一个进程组 ✓（Ctrl+C 不会串到本进程 ✓；``taskkill /T`` 也按它找树 ✓）
_WINDOWS_NEW_PROCESS_GROUP = 0x00000200

#: 一次读多少字节 ✓（小了 syscall 多 ✓，大了首行延迟高 ✓）
_READ_CHUNK = 8192


@dataclass(frozen=True)
class OutputLine:
    """子进程 stdout 的一行 ✓。"""

    text: str
    #: ``True`` = 这一行是**回车原地刷新**（进度条 ✓）⇒ 日志里应**替换**上一条进度条 ✓
    overwrite: bool = False


#: 收到一行就调一次 ✓（不许抛 ✗ —— 抛了会把读行循环带崩 ✓）
LineSink = Callable[[OutputLine], None]


@dataclass
class CommandResult:
    """一次子进程执行的**事实** ✓（不是"我跑过了"的自述 ✓）。"""

    argv: list[str]
    #: 进程退出码 ✓（``-1`` 表示压根没起来 ✓ —— 那种情况其实会抛 :class:`LoraTrainRunError` ✗）
    returncode: int
    #: 是否因**取消**而结束 ✓（取消是正常路径 ✓，不是失败 ✗）
    cancelled: bool
    #: 实际耗时（秒 ✓）
    seconds: float
    #: 读到多少行输出 ✓（「一行都没有」是个重要事实 ✓）
    lines: int
    #: 取消时用的招 ✓（没取消则空串 ✓）
    kill: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.cancelled

    def as_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "cancelled": self.cancelled,
            "seconds": round(self.seconds, 3),
            "lines": self.lines,
            "kill": self.kill,
        }


# ---------------------------------------------------------------------------
# 输出切行
# ---------------------------------------------------------------------------


def iter_output(stream: BinaryIO) -> Iterator[OutputLine]:
    """把字节流按 ``\\r`` / ``\\n`` 切成 :class:`OutputLine` ✓（见模块头第 4 条 ✓）。

    ⚠️ 四个细节都是**实测会踩的** ✓：

    * **``\\r`` 单独一个字节时，必须再往后看一眼**才能定性 ✗✗（``\\r\\n`` = 一个换行 ✓，
      裸 ``\\r`` = 进度条原地刷新 ✓）⇒ 见到 ``\\r`` 先**攥着**这一行（``pending``），
      等下一个字符到了再放 ✓。**这一步漏了会很难发现**：Windows 上 Python 的文本流会把
      ``print()`` 的 ``\\n`` 翻成 ``\\r\\n`` ✓ ⇒ 每条普通日志都会被误判成"进度条刷新" ✓✗
      ⇒ 上层的替换语义**会把整场训练的日志吞成最后一行** ✓✗✗。
    * ``\\r`` 落在**上一个 chunk 的末尾** ✓ ⇒ ``pending`` 是跨 chunk 的状态 ✓（不能只看当前 chunk ✗）；
    * UTF-8 多字节字符可能**被 chunk 切断** ✓ ⇒ 用 ``IncrementalDecoder`` ✓（不是逐 chunk ``decode`` ✗）；
    * 结尾没有换行的那半行**也要交出来** ✓（报错信息常常就挂在最后一行 ✓✗）。

    ⚠️ 一个已知的**有意取舍** ✓：Windows 写出的空行是裸 ``\\r\\n`` ✓ ⇒
    本函数**不**为它产出一条空行 ✗（否则每条空行都多一行日志 ✓✗）。
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    #: 以 ``\r`` 结尾、**还没定性**的那一行 ✓（等下一个字符到了才知道它是 CRLF 还是进度条 ✓）
    pending: str | None = None
    while True:
        chunk = stream.read1(_READ_CHUNK) if hasattr(stream, "read1") else stream.read(_READ_CHUNK)
        if not chunk:
            break
        buffer += decoder.decode(chunk)
        while True:
            index_n = buffer.find("\n")
            index_r = buffer.find("\r")
            if index_n < 0 and index_r < 0:
                break
            if index_r < 0 or (index_n >= 0 and index_n < index_r):
                # ``\n`` 直接结尾 ⇒ 普通行 ✓
                line, buffer = buffer[:index_n], buffer[index_n + 1:]
                if pending is not None and not line:
                    # 攥着的那一行**后面紧跟** ``\n`` ⇒ 它是 ``\r\n``：其实是个普通行 ✓，
                    # 而这里的空串只是那个 ``\n`` 的另一半 ⇒ 不额外产行 ✓
                    if pending:
                        yield OutputLine(pending, overwrite=False)
                    pending = None
                    continue
                if pending is not None:
                    yield OutputLine(pending, overwrite=True)
                    pending = None
                yield OutputLine(line, overwrite=False)
                continue
            # ``\r`` 结尾 ⇒ 先攥着 ✓（它到底是 CRLF 的一半、还是进度条刷新，看下一个字符 ✓）
            line, buffer = buffer[:index_r], buffer[index_r + 1:]
            if not line and pending is None:
                # 空的 ``\r`` ⇒ 攒着别产空行 ✓（上面的 ``\n`` 分支会把它消掉 ✓）
                pending = ""
                continue
            if pending is not None:
                # 上一个 ``\r`` 后面跟的是内容而不是 ``\n`` ⇒ 它是进度条的一次刷新 ✓
                yield OutputLine(pending, overwrite=True)
            pending = line
    buffer += decoder.decode(b"", final=True)
    if pending:
        # 流到这儿 ``pending`` 后面再没有字符 ⇒ 它就是**进度条停止刷新时的那一条** ✓
        yield OutputLine(pending, overwrite=True)
    if buffer:
        yield OutputLine(buffer, overwrite=False)


# ---------------------------------------------------------------------------
# 终止
# ---------------------------------------------------------------------------


def _terminate_tree(proc: subprocess.Popen, *, force: bool = False) -> str:
    """**整棵树**发终止 ✓ ⇒ 说清用的是哪一招 ✓（写进日志便于核对 ✓）。"""
    pid = proc.pid
    if os.name == "nt":
        args = ["taskkill", "/T", "/PID", str(pid)]
        if force:
            args.insert(1, "/F")
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if done.returncode == 0:
                return "taskkill " + ("/F /T" if force else "/T")
        except (OSError, subprocess.SubprocessError):
            pass
        # ``taskkill`` 不在/失败 ⇒ 退回只打父进程 ✓（孙进程可能漏 ✗ —— 如实写进返回串 ✓）
        try:
            (proc.kill if force else proc.terminate)()
        except OSError:
            pass
        return "taskkill 不可用 ⇒ 只终止了父进程（孙进程可能仍在 ✓✗）"
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
        return "killpg " + ("SIGKILL" if force else "SIGTERM")
    except OSError:
        pass
    try:
        (proc.kill if force else proc.terminate)()
    except OSError:
        pass
    return "单进程信号（进程组已不在 ✓）"


def child_env(tool: ResolvedTool, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """训练子进程的环境变量 ✓ —— 在 :func:`paths.build_env` 之上接**线程预算** ✓。

    ⚠️ 顺序不能反 ✗：线程预算只有一处口径 ✓（:mod:`app.core.cpu_budget` ✓），
    ``VOIDE_CPU_THREADS`` 显式优先 ✓、越界当场报错 ✓ ——
    这里把它的 :class:`CpuBudgetError` 翻成 :class:`LoraTrainConfigError` ✓，
    免得"环境变量写错了"以 500 的样子冒出去 ✓（本仓 400/409/500 的分界见 ``errors`` ✓）。
    """
    env = build_env(tool, dict(extra) if extra else None)
    try:
        facts = cpu_budget.describe()
    except cpu_budget.CpuBudgetError as err:
        raise LoraTrainConfigError(f"CPU 线程预算不可用 ✗：{err}") from err
    env.update(cpu_budget.env_for_child())
    env[cpu_budget.ENV_OVERRIDE] = str(facts["threads"])
    return env


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------


class CommandRunner:
    """**串行**子进程运行器 ✓ —— 同一时刻最多一个子进程 ✓（一张卡跑两份只会互相抢 ✗）。

    ⚠️ :meth:`run` **阻塞**实现 ✓（它自己把 stdout 读到 EOF ✓）。
    HTTP 层请把它丢进线程里 ✓（:meth:`run_in_thread` ✓）。
    """

    def __init__(self, on_line: LineSink | None = None,
                 on_error: Callable[[BaseException], None] | None = None) -> None:
        self._on_line: LineSink = on_line if on_line is not None else (lambda _line: None)
        self._on_error: Callable[[BaseException], None] = (
            on_error if on_error is not None
            else (lambda err: print(f"[lora-train] 子进程执行异常：{err!r}"))
        )
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._cancelled = False
        #: 最近一次终止用的招 ✓（事实，供报告用 ✓）
        self.last_kill = ""

    # ---- 状态 -------------------------------------------------------------
    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def pid(self) -> int | None:
        with self._lock:
            return None if self._proc is None else self._proc.pid

    # ---- 取消 -------------------------------------------------------------
    def cancel(self, *, grace: float = 20.0) -> bool:
        """请进程树退出 ✓ ⇒ 是否真的打断了一个**活着的**进程 ✓。

        ``grace`` 秒内没退 ⇒ 强制杀 ✓（先软后硬 ✓，见模块头第 2 条 ✓）。
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return False
            self._cancelled = True
        self.last_kill = _terminate_tree(proc, force=False)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.last_kill = _terminate_tree(proc, force=True)
        return True

    # ---- 执行 -------------------------------------------------------------
    def run(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None,
            cwd: str | os.PathLike[str] | None = None) -> CommandResult:
        """跑一条命令 ✓，逐行交给 ``on_line`` ✓，跑完返回 :class:`CommandResult` ✓。

        ⚠️ 返回码非 0 **不抛** ✗：调用方（runtime ✓）要按"第几步失败、返回码多少"记进任务里 ✓，
        抛异常会把这几条事实丢掉 ✓。**命令起不来**（找不到解释器 ✓）才抛 ✓。
        """
        args = [str(item) for item in argv]
        if not args:
            raise LoraTrainConfigError("命令是空的 ✗ —— 没法执行 ✓")

        popen_kwargs: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "env": dict(env) if env is not None else None,
            "cwd": str(cwd) if cwd is not None else None,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = _WINDOWS_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        started = time.monotonic()
        with self._lock:
            self._cancelled = False
            self.last_kill = ""
            try:
                self._proc = subprocess.Popen(args, **popen_kwargs)  # type: ignore[arg-type]
            except OSError as err:
                self._proc = None
                raise LoraTrainRunError(
                    f"命令起不来 ✗：{args[0]}\n原由：{err}\n"
                    "⇒ 检查这个解释器/脚本在不在 ✓"
                    "（训练侧可用环境变量 LORA_TRAIN_PYTHON 指定解释器 ✓）"
                ) from err
            proc = self._proc

        lines = 0
        returncode = -1
        try:
            stream = proc.stdout
            if stream is not None:
                for line in iter_output(stream):
                    lines += 1
                    self._emit(line)
            returncode = proc.wait()
        finally:
            with self._lock:
                if self._proc is proc:
                    try:
                        if proc.stdout is not None:
                            proc.stdout.close()
                    except OSError:
                        pass
                    self._proc = None

        with self._lock:
            cancelled = self._cancelled
        return CommandResult(
            argv=args,
            returncode=int(returncode),
            cancelled=cancelled,
            seconds=time.monotonic() - started,
            lines=lines,
            kill=self.last_kill,
        )

    def _emit(self, line: OutputLine) -> None:
        """把一行交给上层 ✓；上层抛异常 ⇒ **不当它是子进程的错** ✓，如实往上报 ✓（不静默吞 ✗）。"""
        try:
            self._on_line(line)
        except Exception as err:  # noqa: BLE001 - 日志回调不许把训练带崩 ✓
            print(f"[lora-train] 输出回调自己抛了：{err!r}（这一行：{line.text[:120]!r}）")

    def run_in_thread(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None,
                      cwd: str | os.PathLike[str] | None = None) -> threading.Thread:
        """把 :meth:`run` 放进一个线程 ✓（HTTP 层要的非阻塞 ✓）⇒ 那个线程 ✓。

        ⚠️ 线程里的异常**默认没人看见** ✗ ⇒ 一律交给 ``on_error`` ✓（**不静默吞** ✗）。
        """
        def _target() -> None:
            try:
                self.run(argv, env=env, cwd=cwd)
            except BaseException as err:  # noqa: BLE001 - 线程里的异常必须有人接 ✓
                self._on_error(err)

        thread = threading.Thread(target=_target, name="lora-train-runner", daemon=True)
        thread.start()
        return thread


@dataclass
class StepRecord:
    """任务里**一步**的账 ✓（预缓存 1/2、预缓存 2/2、训练 ✓）。"""

    name: str
    argv: list[str]
    state: str = "pending"
    returncode: int | None = None
    seconds: float = 0.0
    lines: int = 0
    kill: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3),
            "lines": self.lines,
            "kill": self.kill,
            "argv": list(self.argv),
        }
