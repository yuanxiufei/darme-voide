"""读前置自检（`weights.load_module_weights` 的 `read_ahead` ✓，`prefetch` 的**唯一**接线点）。

判据只有一条主线：**预取只能让装载变快，不许让装载变样** ✓✗ ——
`read_ahead=0`（老路 ✓）与 `read_ahead=2`（预取 ✓）装出来的权重必须**逐张量相同** ✓；
另外三条是「别接成空壳」：真命中过 ✓、大件不许被预取 ✓、出函数不留后台线程 ✓。
"""
from __future__ import annotations

import sys
import tempfile
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

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


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tiny.safetensors")
        for case in (lambda: case_same_bytes(path), lambda: case_big_tensor_skipped(path),
                     lambda: case_broken_file(tmp)):
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
