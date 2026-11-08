"""读前置自检（`weights.load_module_weights` 的 `read_ahead` ✓，`prefetch` 的**唯一**接线点）。

判据只有一条主线：**预取只能让装载变快，不许让装载变样** ✓✗ ——
`read_ahead=0`（老路 ✓）与 `read_ahead=2`（预取 ✓）装出来的权重必须**逐张量相同** ✓；
另外三条是「别接成空壳」：真命中过 ✓、大件不许被预取 ✓、出函数不留后台线程 ✓。
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.services.engine import prefetch as pf_mod  # noqa: E402
from app.services.engine import weights as w_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class Tiny(torch.nn.Module):
    """六个 Linear ✓：够让「第 i 件在写卡、第 i+1 件在读」这件事真的发生 ✓。"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 96) for _ in range(6)])

    def forward(self, x):  # pragma: no cover - 只用权重
        for layer in self.layers:
            x = layer(x)
        return x


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _prefetch_threads() -> list[str]:
    return [item.name for item in threading.enumerate() if "-prefetch" in item.name]


def case_same_bytes(path: str) -> None:
    """① 两条路的权重逐张量相同 ✓✗（接错偏移＝静默装出另一个模型 ✗✗）。"""
    torch.manual_seed(20261101)
    source = Tiny()
    w_mod.save_module_weights(source, path)
    expect = _snapshot(source)

    plain = Tiny()
    report = w_mod.load_module_weights(plain, path, read_ahead=0)
    plain_stats = w_mod.last_prefetch()
    check("① 老路（read_ahead=0）装得上 ✓", report.error == "", report.error)
    check("①′ 老路**一次都没预取** ✓（0 就该真的不开线程 ✓）",
          plain_stats.get("hits", 0) == 0 and plain_stats.get("submitted", 0) == 0, plain_stats)

    fast = Tiny()
    report = w_mod.load_module_weights(fast, path, read_ahead=2)
    fast_stats = w_mod.last_prefetch()
    check("①″ 预取路（read_ahead=2）装得上 ✓", report.error == "", report.error)

    same = all(torch.equal(expect[key], value) for key, value in _snapshot(fast).items())
    plain_same = all(torch.equal(expect[key], value) for key, value in _snapshot(plain).items())
    check("② ⭐ 预取装出来的权重与**原模块**逐张量相同 ✓✗", same and plain_same,
          sorted(set(expect) ^ set(fast.state_dict())))

    check("③ ⭐ 真命中过 ✓✗（hits=0 ⇒ 这层就是空壳 ✗✗）",
          fast_stats.get("hits", 0) > 0 and fast_stats.get("failures", 0) == 0, fast_stats)

    check("④ 预取器**出函数就没线程了** ✓✗（不等干净 ⇒ 换装退完后还在读旧文件 ✗✗）",
          _prefetch_threads() == [], _prefetch_threads())


def case_big_tensor_skipped(path: str) -> None:
    """⑤ 超过字节线的大件**不许**预取 ✓（预取它＝峰值翻倍 ✗），但仍要装得上 ✓。"""
    saved = w_mod._READ_AHEAD_MAX_BYTES
    w_mod._READ_AHEAD_MAX_BYTES = 1  # 任何张量都超线 ⇒ 全部落回就地读 ✓
    try:
        module = Tiny()
        report = w_mod.load_module_weights(module, path, read_ahead=2)
        stats = w_mod.last_prefetch()
    finally:
        w_mod._READ_AHEAD_MAX_BYTES = saved
    check("⑤ 全被筛掉时仍装得上 ✓（预取是加速器，不是通路 ✓✗）", report.error == "", report.error)
    check("⑤′ 全被筛掉 ⇒ hits 必须是 0 ✓（否则说明字节线没生效 ✗）",
          stats.get("hits", 0) == 0, stats)


def case_broken_file(tmp: str) -> None:
    """⑥ 读坏了要**说出来** ✓（绝不许留半装还不报 ✓✗）。"""
    broken = str(Path(tmp) / "broken.safetensors")
    torch.manual_seed(7)
    w_mod.save_module_weights(Tiny(), broken)
    data = Path(broken).read_bytes()
    Path(broken).write_bytes(data[:-4096])  # 砍掉尾部：最后几件读不全 ✓
    for read_ahead in (0, 2):
        module = Tiny()
        told = False
        try:
            report = w_mod.load_module_weights(module, broken, read_ahead=read_ahead)
            told = bool(report.error) and "读取" in report.error
        except Exception as err:  # noqa: BLE001 - 开文件阶段就拒也算「说出来」✓
            told = type(err).__name__ in {"SafetensorsError", "OSError"} or "读取" in str(err)
        check(f"⑥ 截断文件（read_ahead={read_ahead}）必须当场报 ✓✗", told)


def _env(**values: str | None):
    """临时设环境变量 ✓（夹具／坏值都要**还回去** ✓✗，否则会污染同进程后续用例 ✓）。"""
    class _Scope:
        def __enter__(self):
            self._saved = {key: os.environ.get(key) for key in values}
            for key, value in values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return self

        def __exit__(self, *exc: object) -> None:
            for key, value in self._saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return None

    return _Scope()


def case_depth_plan(path: str) -> None:
    """⑦ §4.3：预取深度**按盘速分档** ✓✗（写死 2 ⇒ 机械盘上并行读只会更慢 ✗✗）。"""
    table = pf_mod.DISK_CLASSES
    check("⑦ 盘档表四档齐 ✓", sorted(table) == ["hdd", "nvme", "sata", "unknown"], sorted(table))
    check("⑦′ 每档默认都在 1..上限内 ✓",
          all(1 <= default <= hi for default, hi, _why in table.values()),
          {key: value[:2] for key, value in table.items()})
    check("⑦″ 机械盘必须 1 槽 ✓✗（两条并行读＝磁头来回跑 ✗）", table["hdd"][0] == 1, table["hdd"])
    check("⑦‴ 上限守住计划原文「NVMe 2–4 / SATA 1–2」✓",
          table["nvme"][1] == 4 and table["sata"][1] == 2, {k: v[1] for k, v in table.items()})
    check("⑦⁗ 测不出盘速 ⇒ 沿用旧值 2 ✓（老行为零变化 ✓✗）", table["unknown"][0] == 2, table["unknown"])

    for mbps, want_disk, want_value in ((120.0, "hdd", 1), (500.0, "sata", 2), (2500.0, "nvme", 2)):
        with _env(DARME_FAKE_DISK_MBPS=str(mbps), DARME_READ_AHEAD=None):
            plan = pf_mod.resolve_depth(path)
        check(f"⑧ 夹具 {mbps:g} MiB/s ⇒ {want_disk} 档、深度 {want_value} ✓",
              (plan["disk"], plan["value"], plan["source"]) == (want_disk, want_value, "disk"), plan)

    with _env(DARME_FAKE_DISK_MBPS="2500", DARME_READ_AHEAD=None):
        big = pf_mod.resolve_depth(path, explicit=6)
        zero = pf_mod.resolve_depth(path, explicit=0)
    check("⑧′ 显式值超盘档上限 ⇒ 截断且如实报 capped ✓✗（静默改成别的数 ✗）",
          (big["value"], big["requested"], big["capped"]) == (4, 6, True), big)
    check("⑧″ 显式 0＝明确关预取 ✓（不是「没提」✗）",
          (zero["value"], zero["capped"], zero["source"]) == (0, False, "caller"), zero)
    with _env(DARME_FAKE_DISK_MBPS="120", DARME_READ_AHEAD=None):
        slow = pf_mod.resolve_depth(path, explicit=3)
    check("⑧‴ 机械盘上要 3 槽 ⇒ 截到 1 ✓✗（并行读在机械盘上有害 ✓）",
          (slow["value"], slow["capped"]) == (1, True), slow)

    with _env(DARME_FAKE_DISK_MBPS="2500", DARME_READ_AHEAD="3"):
        env_win = pf_mod.resolve_depth(path)
    with _env(DARME_FAKE_DISK_MBPS="2500", DARME_READ_AHEAD="0"):
        env_off = pf_mod.resolve_depth(path)
    with _env(DARME_FAKE_DISK_MBPS="2500", DARME_READ_AHEAD="auto"):
        env_auto = pf_mod.resolve_depth(path)
    check("⑨ DARME_READ_AHEAD 赢过盘档默认 ✓（显式指令必须赢 ✓✗）",
          (env_win["value"], env_win["source"]) == (3, "env"), env_win)
    check("⑨′ DARME_READ_AHEAD=0 真关掉 ✓", env_off["value"] == 0, env_off)
    check("⑨″ DARME_READ_AHEAD=auto ⇒ 交回盘档 ✓", env_auto["value"] == 2, env_auto)

    bad = {"abc": "不是整数", -1: "负数", True: "布尔", 3.5: "整数", 9: "上限"}
    for value, why in bad.items():
        told = ""
        try:
            pf_mod.normalize_depth(value)
        except ValueError as err:
            told = str(err)
        check(f"⑩ 坏值 {value!r} 当场抛且说清「{why}」✓✗（静默回落＝旋钮假生效 ✗）",
              why in told, told)
    with _env(DARME_FAKE_DISK_MBPS=None, DARME_READ_AHEAD="abc"):
        told = ""
        try:
            pf_mod.resolve_depth(path)
        except ValueError as err:
            told = str(err)
    check("⑩′ 坏环境变量同样当场抛 ✓（不许「以为设了实际没设」✓✗）", "不是整数" in told, told)

    small = pf_mod.probe_disk(path)
    check("⑪ 小文件不测速 ✓（量出来的与盘速无关 ⇒ 不编 ✗）",
          small["disk"] == "unknown" and small["how"] == "none", small)
    with _env(DARME_FAKE_DISK_MBPS=None, DARME_READ_AHEAD=None):
        fallback = pf_mod.resolve_depth(path)
    check("⑪′ 量不出盘速 ⇒ 深度 2 ✓（＝本文件原来的写死值 ✓，老路零变化 ✓✗）",
          fallback["value"] == 2 and fallback["disk"] == "unknown", fallback)

    default = inspect.signature(w_mod.load_module_weights).parameters["read_ahead"].default
    check("⑫ ⭐ 装载默认**不再写死** ✓✗（默认＝None＝自适应 ✓，写死 2 ⇒ §4.3 等于没接 ✗）",
          default is None, default)

    torch.manual_seed(11)
    source = Tiny()
    w_mod.save_module_weights(source, path)
    expect = _snapshot(source)
    auto = Tiny()
    report = w_mod.load_module_weights(auto, path)  # 不传 ⇒ 自适应 ✓
    stats = w_mod.last_prefetch()
    plan = w_mod.last_read_ahead()
    check("⑬ 不传 read_ahead 也装得上 ✓", report.error == "", report.error)
    check("⑬′ ⭐ 观测面读得出「用了多深、凭什么」✓✗（读不出＝没法判它有没有生效 ✗）",
          stats.get("readAhead") == plan.get("value") == 2 and plan.get("disk") == "unknown"
          and plan.get("how") == "none", {"stats": stats, "plan": plan})
    check("⑬″ 自适应路装出来的权重逐张量相同 ✓✗",
          all(torch.equal(expect[key], value) for key, value in _snapshot(auto).items()), "")

    off = Tiny()
    w_mod.load_module_weights(off, path, read_ahead=0)
    off_stats = w_mod.last_prefetch()
    check("⑭ 关掉预取时观测面也要如实 ✓✗（留着上一趟的统计 ⇒ 以为它还在跑 ✗）",
          off_stats.get("readAhead") == 0 and off_stats.get("disabled") is True
          and off_stats.get("hits") is None, off_stats)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tiny.safetensors")
        for case in (lambda: case_same_bytes(path), lambda: case_big_tensor_skipped(path),
                     lambda: case_broken_file(tmp), lambda: case_depth_plan(path)):
            try:
                case()
            except Exception:  # noqa: BLE001 - 用例自己炸了也要计入，不要静默跳过 ✓
                check("用例自身抛异常 ✗", False, traceback.format_exc()[-600:])
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
