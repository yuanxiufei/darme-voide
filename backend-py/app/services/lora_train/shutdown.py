"""**训练完成后自动关机** ✓（LoRAMaster ``auto_shutdown.py`` 移植 ✓）。

出处（参考实现 ✓）：``reference/lora/LoRAMaster/auto_shutdown.py`` 的 ``is_admin()`` ✓ 与
``shutdown(delay_seconds=300, restart=False)`` ✓；调用点见 ``wan_lora_train.py`` 第 458~467 行 ✓
——**只在返回码为 0 且 ``auto_shutdown`` 开着时**才调 ✓，GUI 上写着「训练完成5分钟后，自动关机」✓
⇒ 那个 5 分钟就是 ``delay_seconds=300`` 这个默认值 ✓。

## 本仓与参考实现的**刻意不同** ✓

1. **不用 ``os.system``** ✗ ⇒ 用 :func:`subprocess.run` 传 **argv 列表** ✓
   （``os.system`` 走 shell ✓✗；这里虽然参数都是自己拼的整数、没有注入面 ✓，
   但"走 shell"这件事本身不该有 ✓）。
2. **多一个"撤单"** ✓：:func:`cancel` 调 ``shutdown /a`` / ``shutdown -c`` ✓ ——
   参考实现一旦排上就没法撤 ✗（等 5 分钟里想改主意只能自己敲命令 ✓✗）。
   ⚠️ 它是一个**安全阀** ✓，不是抄漏的东西 ✗。
3. **返回事实而不是打日志** ✓：:func:`schedule` 回来的是
   ``{"scheduled", "administrator", "supported", "command", "delaySeconds"}`` ✓ ——
   调用方（runtime ✓）把它记进任务里 ✓，前端才看得见"到底排上了没有"✓
   （参考实现只往 logger 里写一行 ✗，界面上看不出成败 ✓✗）。

⚠️ **默认不开关机** ✗✗：本仓是常驻服务 ✓，关机是**破坏性**动作 ✓ ⇒
只有用户显式把 ``auto_shutdown`` 设成真才会走到这里 ✓（它在本仓的默认值是 ``False`` ✓）。
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess

#: 参考实现的默认延迟 ✓（GUI 文案"训练完成5分钟后，自动关机" ✓）
DEFAULT_DELAY_SECONDS = 300


def is_admin() -> bool:
    """当前进程是否有管理员 / root 权限 ✓（和参考实现同一判据 ✓）。"""
    system = platform.system()
    if system == "Windows":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 拿不到就如实当"没有" ✓（不猜 ✗）
            return False
    try:
        return os.geteuid() == 0
    except AttributeError:  # pragma: no cover - 非 POSIX
        return False


def _argv(*, delay_seconds: int, restart: bool, cancel: bool) -> list[str] | None:
    """拼关机/重启/撤单的命令行 ✓；系统不支持 ⇒ ``None`` ✓。"""
    system = platform.system()
    if system == "Windows":
        if cancel:
            return ["shutdown", "/a"]
        return ["shutdown", "/r" if restart else "/s", "/t", str(max(0, int(delay_seconds)))]
    if system in ("Linux", "Darwin"):
        if cancel:
            return ["shutdown", "-c"]
        minutes = max(1, int(delay_seconds) // 60) if delay_seconds > 0 else 0
        flag = "-r" if restart else "-h"
        # ⚠️ POSIX 的 ``shutdown`` 只认**分钟** ✓ ⇒ ``delay_seconds`` 会被向上取整到分钟 ✓
        return ["shutdown", flag, f"+{minutes}"] if minutes > 0 else ["shutdown", flag, "now"]
    return None


def schedule(*, delay_seconds: int = DEFAULT_DELAY_SECONDS,
             restart: bool = False) -> dict[str, object]:
    """排一次延迟关机 / 重启 ✓ ⇒ **事实** ✓（排上了没有、为什么没排上 ✓）。

    ⚠️ 没有管理员权限 ⇒ ``scheduled=False`` 且 ``reason`` 说清 ✓ ——
    **不抛异常** ✗：训练已经跑完了 ✓，为了"关不了机"把整个任务判失败是错的分界 ✓✗
    （参考实现也只是记一行日志 ✓，见模块头第 3 条 ✓）。
    """
    facts: dict[str, object] = {
        "requested": True,
        "restart": bool(restart),
        "delaySeconds": max(0, int(delay_seconds)),
        "delaySecondsNote": (
            "Windows 按秒 ✓；Linux/macOS 的 shutdown 只认分钟 ⇒ 会向上取整到分钟 ✓"
        ),
        "administrator": is_admin(),
        "supported": True,
        "scheduled": False,
        "command": None,
        "reason": "",
    }
    argv = _argv(delay_seconds=delay_seconds, restart=restart, cancel=False)
    if argv is None:
        facts["supported"] = False
        facts["reason"] = f"当前系统（{platform.system()}）不支持关机命令 ✗"
        return facts
    if not facts["administrator"]:
        facts["reason"] = "当前进程没有管理员 / root 权限 ⇒ 关不了机 ✗（参考实现同样如此 ✓）"
        return facts
    facts["command"] = argv
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as err:
        facts["reason"] = f"关机命令起不来 ✗：{err}"
        return facts
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()
        facts["reason"] = f"关机命令返回 {done.returncode} ✗：{detail}"
        return facts
    facts["scheduled"] = True
    return facts


def cancel() -> dict[str, object]:
    """撤掉已排的关机 ✓（``shutdown /a`` / ``shutdown -c`` ✓）⇒ 事实 ✓。"""
    facts: dict[str, object] = {
        "supported": True,
        "cancelled": False,
        "command": None,
        "reason": "",
    }
    argv = _argv(delay_seconds=0, restart=False, cancel=True)
    if argv is None:
        facts["supported"] = False
        facts["reason"] = f"当前系统（{platform.system()}）不支持撤单 ✗"
        return facts
    facts["command"] = argv
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as err:
        facts["reason"] = f"撤单命令起不来 ✗：{err}"
        return facts
    # ⚠️ 没有待撤的关机时，``shutdown /a`` 会返回非 0 ✓ ⇒ 如实报"没撤成" ✓（不谎报成功 ✗）
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()
        facts["reason"] = f"撤单命令返回 {done.returncode} ✗（可能本来就没有待执行的关机 ✓）：{detail}"
        return facts
    facts["cancelled"] = True
    return facts
