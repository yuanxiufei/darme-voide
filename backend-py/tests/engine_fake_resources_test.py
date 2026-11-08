"""低配模拟自检（G4，2026-11-07）✓✗。

本机是 A5000／22.49 GiB／96 GiB 内存 ✓，而验收档是 **8 GB 卡 + 32 GB 内存** ✓✗。
这份夹具的全部意义是：**不用插真卡**就能让「决策 / 切分预算 / 内存余量」三处一起
按小机器的读数走 ✓。

判据盯三件事：

1. **口径**：只改总量、**不改占用** ✓✗（改 free 会让 `usedRatio` 恒 0 ⇒ 第 ⑤ 支判据永不触发 ✗）；
2. **三处都接上**：`probe` / `budget_of_cuda` / `host_memory` ✓ —— 只接一处会让决策说
   8 GiB、预算按 22.49 GiB 装 ✓✗；
3. **反向验证**：不给变量时结论必须**不一样** ✓✗（否则判据是空转 ✗）。

⚠️ 模拟值必须能被看见（`Pressure.faked` / 回执 `pressure.faked` ✓）：假数据冒充真读数
正是本仓最恨的那种"假绿" ✓✗。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import device_policy as dp  # noqa: E402
from app.services.engine import residency  # noqa: E402
from app.services.engine import scheduler  # noqa: E402

_FAILS: list[str] = []
_TOTAL = 0
GIB = dp.GIB


def check(label: str, ok: bool, detail: object = "") -> None:
    global _TOTAL
    _TOTAL += 1
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


class _FakeCuda:
    def __init__(self, total: int, free: int, *,
                 allocated: int = 0, reserved: int = 0) -> None:
        self._total = int(total)
        self._free = int(free)
        self._allocated = int(allocated)
        self._reserved = int(reserved)

    def is_available(self) -> bool:
        return True

    def mem_get_info(self, index: int | None = None) -> tuple[int, int]:
        return self._free, self._total

    def memory_allocated(self, index: int | None = None) -> int:
        return self._allocated

    def memory_reserved(self, index: int | None = None) -> int:
        return self._reserved


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


def _card(*, total_gib: float = 22.49, free_gib: float = 20.0,
          allocated_gib: float = 18.0, reserved_gib: float = 20.0) -> _FakeTorch:
    return _FakeTorch(_FakeCuda(int(total_gib * GIB), int(free_gib * GIB),
                                allocated=int(allocated_gib * GIB),
                                reserved=int(reserved_gib * GIB)))


class _Env:
    """临时设/删环境变量 ✓（用完必须复原 ✓ —— 泄漏会污染同进程里后面每一套判据 ✓✗）。"""

    def __init__(self, **values: str) -> None:
        self._values = values
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for name, value in self._values.items():
            self._saved[name] = os.environ.get(name)
            os.environ[name] = value

    def __exit__(self, *exc: Any) -> None:
        for name, old in self._saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def case_parse() -> None:
    """① 变量解析：没设/空串 ⇒ None；数字 ⇒ 字节；写错 ⇒ **当场抛** ✓✗。"""
    name = dp.FAKE_GPU_TOTAL_ENV
    with _Env(**{name: ""}):
        check("①a 空串 ⇒ 不模拟（None）", dp.fake_total_bytes(name) is None)
    with _Env(**{name: "  "}):
        check("①b 全空格 ⇒ 不模拟", dp.fake_total_bytes(name) is None)
    with _Env(**{name: "8"}):
        check("①c `8` ⇒ 8 GiB", dp.fake_total_bytes(name) == 8 * GIB)
    with _Env(**{name: "7.5"}):
        check("①d `7.5` ⇒ 7.5 GiB（允许小数）", dp.fake_total_bytes(name) == int(7.5 * GIB))

    raised: list[str] = []
    for bad in ("abc", "0", "-2", "8GB"):
        with _Env(**{name: bad}):
            try:
                dp.fake_total_bytes(name)
            except ValueError as exc:
                raised.append(str(exc))
    check("①e 写错/非正数 ⇒ 抛（`abc`/`0`/`-2`/`8GB` 四种）",
          len(raised) == 4, raised)
    check("①f 报错消息点名变量与合法写法 ✓",
          all(name in msg and "GiB" in msg for msg in raised), raised)


def case_fake_memory() -> None:
    """② 纯函数口径：**占用不动、总量变小** ✓✗。"""
    total, free = dp.fake_memory(8 * GIB, 22 * GIB, 20 * GIB)
    check("②a 总量换成 8 GiB", total == 8 * GIB, total)
    check("②b 可用 = 8 − 已用 2 = 6 GiB（占用原样保留）", free == 6 * GIB, free)

    total, free = dp.fake_memory(8 * GIB, 22 * GIB, 22 * GIB)
    check("②c 一点没占用 ⇒ 可用 = 全部", free == 8 * GIB, free)

    total, free = dp.fake_memory(2 * GIB, 22 * GIB, 20 * GIB)
    check("②d 已用超过假总量 ⇒ 可用夹到 0（不许负数）", free == 0, free)


def case_probe() -> None:
    """③ `probe()`：接上模拟、如实标 `faked`、`source` 口径不变 ✓。"""
    torch_mod = _card()
    real = dp.probe(torch_mod)
    check("③a 无模拟 ⇒ total 就是真卡 22.49 GiB",
          real is not None and real.totalBytes == int(22.49 * GIB), real)
    check("③b 无模拟 ⇒ `faked=False`", real is not None and real.faked is False)
    # 口径提醒：可用 = 驱动空闲 20 + allocator 缓存块 (reserved 20 − allocated 18) = 22 GiB ✓
    check("③c 无模拟 ⇒ 可用 = 驱动空闲 + 缓存块 = 22 GiB（回归保护）",
          real is not None and real.freeBytes == 22 * GIB,
          None if real is None else real.to_dict())

    with _Env(**{dp.FAKE_GPU_TOTAL_ENV: "8"}):
        fake = dp.probe(torch_mod)
    check("③d 模拟 8 GiB ⇒ totalBytes = 8 GiB",
          fake is not None and fake.totalBytes == 8 * GIB, fake)
    check("③e 模拟后**已用字节一字不改**（不是把卡洗白成空卡）",
          fake is not None and real is not None and fake.usedBytes == real.usedBytes,
          None if fake is None or real is None else (fake.usedBytes, real.usedBytes))
    # 驱动已用 2.49 GiB ⇒ 模拟总量 8 ⇒ 驱动可用 5.51 ⇒ 再加缓存块 2 = 7.51 GiB ✓
    check("③f 模拟后可用 = 8 − 驱动已用 2.49 + 缓存 2 = 7.51 GiB",
          fake is not None and abs(fake.freeBytes - int(7.51 * GIB)) <= GIB // 50,
          None if fake is None else fake.to_dict())
    check("③g 模拟值必须被标出来（`faked=True`）", fake is not None and fake.faked is True)
    check("③h 回执字典里也看得见 `faked`",
          fake is not None and fake.to_dict().get("faked") is True,
          None if fake is None else fake.to_dict())
    check("③i `source` 口径不被模拟污染（仍答利用率从哪来）",
          fake is not None and fake.source == "memory",
          None if fake is None else fake.to_dict())

    # allocator 里"没还给驱动"的块仍要算进可用（2026-11-01 那条口径不许被模拟弄丢）
    with _Env(**{dp.FAKE_GPU_TOTAL_ENV: "8"}):
        cached = dp.probe(_card(free_gib=4.0, allocated_gib=18.0, reserved_gib=20.0))
    check("③j 模拟不影响「缓存块算可用」（8 − 已用 18.49 ⇒ 夹到 0，不是负数）",
          cached is not None and cached.freeBytes >= 0, None if cached is None else cached.to_dict())

    # 老调用方构造 Pressure 时不传 faked 也必须能跑 ✓
    old = dp.Pressure(totalBytes=GIB, freeBytes=GIB)
    check("③k 老构造（不传 `faked`）⇒ 默认 False", old.faked is False)


def case_budget() -> None:
    """④ `residency.budget_of_cuda`：预算必须跟着模拟一起变小 ✓✗。"""
    torch_mod = _card(free_gib=20.0, allocated_gib=18.0, reserved_gib=20.0)
    real = residency.budget_of_cuda(torch_mod, reserve=0.0, kind="dit")
    with _Env(**{dp.FAKE_GPU_TOTAL_ENV: "8"}):
        fake = residency.budget_of_cuda(torch_mod, reserve=0.0, kind="dit")
    check("④a 两侧都算得出预算", real is not None and fake is not None)
    if real is not None and fake is not None:
        check("④b 真卡预算 = 可用 22 GiB（`reserve=0` 不砍）",
              real.effective == 22 * GIB, real.effective)
        check("④c 模拟 8 GiB ⇒ 预算掉到 7.51 GiB（驱动已用 2.49 + 缓存 2）",
              abs(fake.effective - int(7.51 * GIB)) <= GIB // 50, fake.effective)
        check("④d 预算真的变小了（不是同一份读数）",
              fake.effective < real.effective, (fake.effective, real.effective))


def case_host() -> None:
    """⑤ `scheduler.host_memory`：内存侧同一口径（否则只模拟了半台机器 ✗）。"""
    real = scheduler.host_memory()
    with _Env(**{dp.FAKE_HOST_TOTAL_ENV: "32"}):
        fake = scheduler.host_memory()
    with _Env(**{dp.FAKE_HOST_TOTAL_ENV: "16"}):
        fake16 = scheduler.host_memory()
    check("⑤a 真机读数可用（否则本套判据没法判）", real is not None, real)
    check("⑤b 模拟 32 GiB ⇒ 总量 = 32 GiB",
          fake is not None and fake[1] == 32 * GIB, fake)
    check("⑤c 模拟 16 GiB ⇒ 总量 = 16 GiB",
          fake16 is not None and fake16[1] == 16 * GIB, fake16)
    check("⑤d 可用不为负（已用超过假总量时夹到 0）",
          fake16 is not None and fake16[0] >= 0, fake16)
    if real is not None and fake is not None:
        check("⑤e 真机总量 ≠ 32 GiB（证明模拟真的生效了，不是碰巧）",
              real[1] != 32 * GIB, real)
        check("⑤f 模拟后「可放多少」跟着变小 ⇒ `host_room_bytes` 联动",
              (scheduler.host_room_bytes(fake) or 0) < (scheduler.host_room_bytes(real) or 0),
              (scheduler.host_room_bytes(fake), scheduler.host_room_bytes(real)))


def case_decision() -> None:
    """⑥ 端到端：模拟出来的小卡必须**改结论** ✓✗（这才是这份夹具存在的理由 ✓）。"""
    torch_mod = _card(total_gib=22.49, free_gib=20.0, allocated_gib=18.0, reserved_gib=20.0)
    need = int(12.57 * GIB)

    real_p = dp.probe(torch_mod)
    real_d = dp.decide(need_bytes=need, pressure=real_p)
    check("⑥a 真卡（余 20 GiB）跑 12.57 GiB ⇒ 直接上卡、不降级",
          real_d.device == "cuda" and real_d.residency == "off" and not real_d.degraded,
          real_d.to_dict())

    with _Env(**{dp.FAKE_GPU_TOTAL_ENV: "8"}):
        small_p = dp.probe(torch_mod)
    small_d = dp.decide(need_bytes=need, pressure=small_p)
    check("⑥b 模拟 8 GiB ⇒ 改**联合驻留**且如实 `degraded=True`",
          small_d.device == "cuda" and small_d.residency == "split" and small_d.degraded,
          small_d.to_dict())
    check("⑥c 回执里带着 `pressure.faked=True`（模拟不许冒充真读数）",
          (small_d.pressure or {}).get("faked") is True, small_d.pressure)

    # 这一档要"卡上一点都放不下"⇒ 必须用**没有 allocator 缓存**的卡 ✓
    # （有缓存时 reserved−allocated 那部分会算成可用 ✓，8 GiB 卡于是仍挤得出 2 GiB 切片 ✓）
    tight = _card(total_gib=22.49, free_gib=20.0, allocated_gib=4.0, reserved_gib=4.0)
    with _Env(**{dp.FAKE_GPU_TOTAL_ENV: "2"}):
        tiny_p = dp.probe(tight)
    tiny_d = dp.decide(need_bytes=need, pressure=tiny_p)
    check("⑥d 模拟 2 GiB 且已占满（无缓存块）⇒ 整件转 CPU",
          tiny_p is not None and tiny_p.freeBytes == 0
          and tiny_d.device == "cpu" and tiny_d.degraded, tiny_d.to_dict())

    check("⑥e 反向验证：不给变量时结论确实不同（判据不是空转）",
          real_d.to_dict()["residency"] != small_d.to_dict()["residency"],
          (real_d.to_dict()["residency"], small_d.to_dict()["residency"]))


def case_bad_env_reaches_probe() -> None:
    """⑦ 写错的变量必须一路抛到读数口（不是被 try 吞掉变成"没模拟" ✗）。"""
    raised = ""
    with _Env(**{dp.FAKE_GPU_TOTAL_ENV: "8GB"}):
        try:
            dp.probe(_card())
        except ValueError as exc:
            raised = str(exc)
    check("⑦ 写错的模拟变量在 `probe` 上当场抛、且点名变量",
          dp.FAKE_GPU_TOTAL_ENV in raised, raised)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("== 低配模拟自检（G4）==")
    case_parse()
    case_fake_memory()
    case_probe()
    case_budget()
    case_host()
    case_decision()
    case_bad_env_reaches_probe()
    passed = _TOTAL - len(_FAILS)
    print(f"\nSUMMARY: {passed}/{_TOTAL} passed")
    return 0 if passed == _TOTAL and _TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
