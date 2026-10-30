"""自检：**LoRA 内容账本** ✓（E2.5 ✓ —— 内容哈希 / 引用计数 / 释放事实 ✓）。

这一套钉的是三件**会真花钱**的事 ✓（不是"函数能跑"✗）：

1. **内容身份** ✓：同一份 LoRA 换个文件名／路径，摘要**必须**相同 ✓✗
   —— 按路径认身份（E2.5 之前正是这样 ✓）⇒ 换个名字就是**第二份张量** ✓✗，
   显存与 RSS 双翻倍、而且**没有任何地方看得出来** ✓✗。
2. **同一 LoRA 加载两次只占一份** ✓：`refTotal == 2` 但 `entryCount == 1`、
   `tensorBytes` **只算一次** ✓，且**读盘只发生一次** ✓（第二次是复用 ✓）。
   ⚠️ 判据必须同时看"读了几次"与"占多少" ✗✗：只看引用计数会被"读两次、丢一份"骗过去 ✓✗。
3. **释放后真的少了** ✓：最后一个引用还掉 ⇒ 条目消失、`tensorBytes` 归 0、
   `freedBytes` 记上这次丢了多少 ✓ —— 台账说"已释放"而字节数没变，就是**假释放** ✓✗。

⚠️ 这里**不碰 torch** ✓（零依赖模块 ✓，假张量只实现 `numel` / `element_size` ✓）——
   真机路径（`torch_backend._apply_loras` 借/还）由 `engine_lora_test` 那套覆盖 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_lora_cache_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import lora_cache as cache_mod  # noqa: E402
from app.services.engine.lora_cache import LoraCache, LoraCacheError  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class _Tensor:
    """假张量 ✓（只回答"多少元素、每元素几字节" ✓ —— 够算字节数 ✓，不拖 torch ✓）。"""

    def __init__(self, count: int, *, broken: bool = False) -> None:
        self._count = count
        self._broken = broken

    def numel(self) -> int:
        if self._broken:
            raise TypeError("假张量：算不出来 ✓（专门用来验「None 而不是编 0」✓）")
        return self._count

    def element_size(self) -> int:
        return 2


def _tensors(count: int) -> dict[str, Any]:
    return {"a": _Tensor(count), "b": _Tensor(count)}


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class _Reader:
    """记次数 ✓（"读盘几次"是**第一类判据** ✓✗ —— 只记引用计数会被"读两遍丢一份"骗过 ✓）。"""

    def __init__(self, count: int = 1024) -> None:
        self.count = count
        self.calls: list[str] = []

    def __call__(self, path: str) -> dict[str, Any]:
        self.calls.append(path)
        return _tensors(self.count)


def case_identity(tmp: Path) -> None:
    cache = LoraCache()
    same_a = _write(tmp / "a.safetensors", b"same-lora-bytes" * 64)
    same_b = _write(tmp / "copy-of-a.safetensors", b"same-lora-bytes" * 64)
    other = _write(tmp / "c.safetensors", b"another-lora" * 88)

    digest_a = cache.digest_of(same_a)
    check("内容身份: 同内容不同文件名 ⇒ **同一个摘要** ✓（这才是「一份 LoRA」的定义 ✓）",
          digest_a == cache.digest_of(same_b), (digest_a[:12], cache.digest_of(same_b)[:12]))
    check("内容身份: 内容不同 ⇒ 摘要不同 ✓", digest_a != cache.digest_of(other),
          (digest_a[:12], cache.digest_of(other)[:12]))
    check("内容身份: 摘要长度 = sha256 十六进制 64 ✓", len(digest_a) == 64, len(digest_a))

    # 快路 ✓：同一路径第二次不重算（⚠️ 快路只是**省时间** ✓，不是身份 ✗ —— 文件被原地换掉
    # 且大小与 mtime 都不变时它认不出来 ✓✗，这条限制写在 `_byStat` 的注释里 ✓）
    calls: list[str] = []
    original = cache_mod.digest_file

    def counting(path: str | Path) -> str:
        calls.append(str(path))
        return original(path)

    cache_mod.digest_file = counting              # type: ignore[assignment]
    try:
        cache.digest_of(same_a)
        cache.digest_of(same_a)
    finally:
        cache_mod.digest_file = original          # type: ignore[assignment]
    check("内容身份: 同一路径第二次走 (路径,大小,mtime) 快路 ⇒ 不重算 sha256 ✓",
          len(calls) == 0, calls)

    missing = tmp / "nope.safetensors"
    try:
        cache.digest_of(missing)
        raised = ""
    except LoraCacheError as err:
        raised = str(err)
    check("内容身份: 文件不在 ⇒ **具名错误** ✓（不是静默 0 字节摘要 ✗✗）",
          "不在" in raised and "nope" in raised, raised)
    cache.reset()


def case_dedup(tmp: Path) -> None:
    cache = LoraCache()
    reader = _Reader(count=4096)
    first = _write(tmp / "x.safetensors", b"payload-x" * 100)
    alias = _write(tmp / "renamed-x.safetensors", b"payload-x" * 100)

    lease1 = cache.acquire(first, reader=reader, reason="第一次挂 ✓")
    after_first = cache.facts()
    check("借一份: 第一次 ⇒ 读盘 1 次、1 个条目、引用 1 ✓",
          len(reader.calls) == 1 and after_first["entryCount"] == 1
          and after_first["refTotal"] == 1, after_first)

    lease2 = cache.acquire(first, reader=reader, reason="同一路径再挂一次 ✓")
    after_second = cache.facts()
    check("同一 LoRA 加载两次: **读盘仍是 1 次** ✓（第二次复用，不重读 ✓）",
          len(reader.calls) == 1, reader.calls)
    check("同一 LoRA 加载两次: 引用 2、条目 1、字节数**只算一份** ✓",
          after_second["refTotal"] == 2 and after_second["entryCount"] == 1
          and after_second["tensorBytes"] == 4096 * 4, after_second["tensorBytes"])
    check("同一 LoRA 加载两次: `hits` 记 1（同路径复用 ✓）",
          after_second["hits"] == 1 and after_second["misses"] == 1,
          (after_second["hits"], after_second["misses"]))

    lease3 = cache.acquire(alias, reader=reader, reason="换个文件名再挂 ✓")
    after_alias = cache.facts()
    check("换路径同内容: **仍然只有 1 个条目** ✓、字节数不变 ✓（换名字不是第二份 ✓✗）",
          after_alias["entryCount"] == 1 and after_alias["tensorBytes"] == 4096 * 4,
          after_alias["tensorBytes"])
    check("换路径同内容: `deduped` 记 1 且 `aliases` 留痕 ✓（「只占一份」的凭据 ✓）",
          after_alias["deduped"] == 1 and after_alias["entries"][0]["aliases"] == [str(alias)],
          after_alias["entries"][0]["aliases"])
    check("换路径同内容: 读盘**依然只有 1 次** ✓（命中就不该再读 ✓✗）",
          len(reader.calls) == 1, reader.calls)
    check("同一内容的三次借出指向**同一份张量对象** ✓",
          lease1.tensors is lease2.tensors is lease3.tensors, "不是同一对象 ⇒ 至少两份内存 ✗")
    cache.reset()


def case_release(tmp: Path) -> None:
    cache = LoraCache()
    reader = _Reader(count=2048)
    path = _write(tmp / "r.safetensors", b"release-me" * 50)
    lease = cache.acquire(path, reader=reader)
    second = cache.acquire(path, reader=reader)

    first_drop = cache.release(lease, reason="先还一个 ✓")
    mid = cache.facts()
    check("还一份: 还有引用 ⇒ **不释放** ✓（`released: false` ✓ + 字节数不动 ✓）",
          first_drop["released"] is False and first_drop["refCount"] == 1
          and mid["tensorBytes"] == 2048 * 4 and mid["entryCount"] == 1, (first_drop, mid["tensorBytes"]))

    final_drop = cache.release(second, reason="最后一个 ✓")
    done = cache.facts()
    check("还最后一份: **真释放** ✓（`released: true` ✓ + 字节数 = 这次丢的 ✓）",
          final_drop["released"] is True and final_drop["tensorBytes"] == 2048 * 4,
          final_drop)
    check("还最后一份: `status().lora` 的字节数归 **0** ✓（「释放后显存下降」的账本侧证据 ✓）",
          done["tensorBytes"] == 0 and done["entryCount"] == 0 and done["refTotal"] == 0, done["tensorBytes"])
    check("还最后一份: `freed` / `freedBytes` 记上这次真丢了多少 ✓",
          done["freed"] == 1 and done["freedBytes"] == 2048 * 4, (done["freed"], done["freedBytes"]))

    try:
        cache.release(lease)
        over = ""
    except LoraCacheError as err:
        over = str(err)
    check("还多了 ⇒ **当场报错** ✓✗（不许静默把别人的那份还掉 ✗✗）",
          "不存在" in over or "已经" in over, over)

    stranger = cache_mod.LoraLease(digest="0" * 64, path="x", tensors={})
    try:
        cache.release(stranger)
        stranger_err = ""
    except LoraCacheError as err:
        stranger_err = str(err)
    check("还一个从没借过的摘要 ⇒ **具名错误** ✓", "不存在" in stranger_err, stranger_err)
    cache.reset()


def case_release_all(tmp: Path) -> None:
    cache = LoraCache()
    reader = _Reader(count=1000)
    a = _write(tmp / "ra.safetensors", b"ra" * 40)
    b = _write(tmp / "rb.safetensors", b"rb" * 40)
    lease_a = cache.acquire(a, reader=reader)
    cache.acquire(a, reader=reader)          # 还有第二个引用**没还** ✓（模拟"有人忘了放" ✓）
    cache.acquire(b, reader=reader)

    freed = cache.release_all(reason="模型卸载 ✓")
    check("整份清掉: 2 个条目都清 ✓、字节数合计上报 ✓",
          freed["entries"] == 2 and freed["fileBytes"] > 0 and freed["tensorBytes"] == 1000 * 4 * 2,
          freed)
    check("整份清掉: **有没还的引用必须说出来** ✓✗（这里 2 个条目都还有租约没还 ⇒ 内存不会当场少 ✓）",
          freed["outstanding"] == 2 and "未归还" in freed["note"], freed["note"])
    check("整份清掉: 账本归零 ✓", cache.facts()["entryCount"] == 0 and cache.facts()["tensorBytes"] == 0,
          cache.facts()["entryCount"])

    # ⚠️ 反面对照 ✓✗：**先按规矩还清再清账** ⇒ `outstanding` 必须是 0 ✓
    #    （这一条才区分得开「正常卸载」与「有人忘了还」✓ —— 否则每份永远 ≥1 ✓✗，字段就成了噪声 ✓）
    tidy = LoraCache()
    lease = tidy.acquire(_write(tmp / "tidy.safetensors", b"tidy" * 30), reader=reader)
    tidy.release(lease, reason="按规矩还 ✓")
    tidy_freed = tidy.release_all(reason="模型卸载 ✓")
    check("整份清掉: 租约都还清之后 ⇒ `outstanding` **0** ✓（真泄漏才 > 0 ✓✗）",
          tidy_freed["outstanding"] == 0 and "未归还" not in tidy_freed["note"],
          (tidy_freed["outstanding"], tidy_freed["note"]))

    try:
        cache.release(lease_a)
        stale_err = ""
    except LoraCacheError as err:
        stale_err = str(err)
    check("整份清掉之后还旧租约 ⇒ 报错 ✓（`torch_backend` 那边会把它当「已经清过」吞掉 ✓，"
          "但账本层面**不许静默** ✓✗）", "不存在" in stale_err, stale_err)
    cache.reset()


def case_unknown_bytes(tmp: Path) -> None:
    cache = LoraCache()

    def broken_reader(path: str) -> dict[str, Any]:
        return {"w": _Tensor(10, broken=True)}

    path = _write(tmp / "broken.safetensors", b"broken" * 30)
    cache.acquire(path, reader=broken_reader)
    facts = cache.facts()
    check("算不出的字节数 ⇒ **None** ✓✗（不许按文件大小顶上 ✗：文件字节恒大于张量字节 ✓）",
          facts["entries"][0]["tensorBytes"] is None, facts["entries"][0]["tensorBytes"])
    check("算不出的字节数 ⇒ `tensorBytesUnknown` 报 1 且 note 说明「总数偏小」✓✗",
          facts["tensorBytesUnknown"] == 1 and "偏小" in facts["note"], facts["note"])
    cache.reset()


def case_reentrant(tmp: Path) -> None:
    """并发去重 ✓：读盘期间另一条路径已把同一份读进来 ⇒ **丢掉自己这份** ✓✗。"""
    cache = LoraCache()
    path = _write(tmp / "race.safetensors", b"race" * 64)
    alias = _write(tmp / "race-alias.safetensors", b"race" * 64)
    calls: list[str] = []
    winner: list[Any] = []

    def racing_reader(target: str) -> dict[str, Any]:
        calls.append(target)
        if len(calls) == 1:
            # 读盘途中"另一个人"抢先 ✓ —— 它走的是**另一条路径**（内容相同 ✓）
            winner.append(cache.acquire(alias, reader=_ReaderSimple()))
        return _tensors(512)

    lease = cache.acquire(path, reader=racing_reader)
    facts = cache.facts()
    check("并发同内容: 后读的那份被丢掉 ✓（条目仍 1 ✓、字节数只有一份 ✓）",
          facts["entryCount"] == 1 and facts["tensorBytes"] == 512 * 4, facts["tensorBytes"])
    check("并发同内容: `deduped` 记 1、引用 2 ✓（两边都借到了，但只有一份张量 ✓）",
          facts["deduped"] == 1 and facts["refTotal"] == 2,
          (facts["deduped"], facts["refTotal"]))
    check("并发同内容: 这次借出的是**内存里那份**（不是自己刚读的那份 ✗）",
          winner and lease.tensors is winner[0].tensors,
          "不是同一个对象 ⇒ 自己那份没被丢掉 ✗（泄漏一份内存 ✓✗）")
    cache.reset()


class _ReaderSimple:
    def __call__(self, path: str) -> dict[str, Any]:
        return _tensors(512)


def case_events(tmp: Path) -> None:
    cache = LoraCache()
    reader = _Reader(count=8)
    path = _write(tmp / "e.safetensors", b"evt" * 30)
    for _ in range(cache_mod._EVENT_KEEP + 4):
        lease = cache.acquire(path, reader=reader)
        cache.release(lease)
    recent = cache.facts()["recent"]
    check(f"事件环形缓冲: 只留最近 {cache_mod._EVENT_KEEP} 条 ✓（长跑不涨 ✓✗）",
          len(recent) == cache_mod._EVENT_KEEP, len(recent))
    check("事件环形缓冲: 每条都带 at / kind / digest / path / reason ✓",
          all({"at", "kind", "digest", "path", "reason"} <= set(event) for event in recent),
          recent[-1] if recent else None)
    cache.reset()


def case_self_describe() -> None:
    cache = LoraCache()
    facts = cache.facts()
    check("自述: `scope` 明写「不含主干装配复用门」✓（与 `status().cache` 是两件事 ✓✗）",
          "主干装配复用门" in facts["scope"] and "torch allocator" in facts["scope"], facts["scope"])
    check("自述: `note` 明写「进程重启即归零」✓（别把 0 读成「从来没复用」✓✗）",
          "重启" in facts["note"] and "归零" in facts["note"], facts["note"])
    check("自述: 空账本也**照样给七个以上字段** ✓（前端不必判空 ✓）",
          {"entries", "entryCount", "refTotal", "tensorBytes", "tensorBytesUnknown",
           "hits", "misses", "deduped", "freed", "freedBytes", "since", "recent",
           "scope", "note"} <= set(facts), sorted(facts))
    check("单例: `shared()` 两次是**同一个** ✓（挂在后端实例上 ⇒ 换装配就丢一次 ✓✗）",
          cache_mod.shared() is cache_mod.shared(), id(cache_mod.shared()))


def _call_sites(source_path: Path, func_name: str) -> list[tuple[int, str]]:
    """把某个函数的**所有**调用点抽成 ``(行号, 点号名)`` ✓（``cache.acquire`` ✓）。

    ⚠️ 为什么用 AST 而不是 import 后看行为 ✗：`_apply_loras` 要 torch 才能跑 ✓，
    而这套的纪律是**零 torch** ✓ ⇒ 只能判"接线还在不在" ✓ —— 判据很窄 ✓，
    但它拦的正是最容易被改回去的那种退化：**绕过账本、直接读盘** ✓✗。
    ⚠️ **行号要留着** ✗：`ast.walk` 是广度优先、**不是源码顺序** ✓✗ ⇒ 想判"先还租约再清账本"
    只能比 `lineno` ✓（拿 walk 出来的下标比大小会随机绿/红 ✓✗）。
    """
    import ast  # noqa: PLC0415

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != func_name:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            parts: list[str] = []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
            if parts:
                sites.append((int(inner.lineno), ".".join(reversed(parts))))
    return sites


def _call_names(source_path: Path, func_name: str) -> list[str]:
    return [name for _, name in _call_sites(source_path, func_name)]


def _first_line(sites: list[tuple[int, str]], name: str) -> int | None:
    lines = [line for line, call in sites if call == name]
    return min(lines) if lines else None


def case_wiring() -> None:
    torch_backend = BACKEND_PY / "app/services/engine/torch_backend.py"
    runtime = BACKEND_PY / "app/services/engine/runtime.py"
    apply_calls = _call_names(torch_backend, "_apply_loras")
    release_calls = _call_names(torch_backend, "release_loras")
    unload_sites = _call_sites(runtime, "unload")
    status_calls = _call_names(runtime, "status")

    check("接线: `_apply_loras` **必须**经账本借张量 ✓（`cache.acquire` ✓✗ —— 直接 "
          "`read_tensors(path)` 就退回「同 LoRA 两份内存」✓✗）",
          any(name in apply_calls for name in ("cache.acquire", "shared.acquire")), sorted(apply_calls))
    check("接线: `_apply_loras` 把借来的 `reader` 交给 `apply_specs` ✓（读盘口只有一个 ✓）",
          "lora_mod.apply_specs" in apply_calls, sorted(apply_calls))
    check("接线: `_apply_loras` **开工前**先还上一批 ✓（换装配 ⇒ 旧的 LoRA 该放 ✓✗）",
          "self.release_loras" in apply_calls, sorted(apply_calls))
    check("接线: `_apply_loras` 失败路径**两条**都还 ✓（挂炸了 / 没装全 ✓ ⇒ 不许吊着 ✓✗）",
          sum(1 for name in apply_calls if name == "self._release_leases") >= 2,
          [name for name in apply_calls if "_release_leases" in name])
    check("接线: `release_loras` 走同一条 `_release_leases` ✓（成对还账只有一处实现 ✓）",
          "self._release_leases" in release_calls, sorted(set(release_calls)))
    lease_line = _first_line(unload_sites, "backend.release_loras")
    all_line = _first_line(unload_sites, "release_all")
    check("接线: `unload` **先**让后端还租约 ✓（`backend.release_loras` 在前 ✓）"
          "**再**清账本 ✓（`release_all` ✓）—— 顺序反了 `outstanding` 就永远是假泄漏 ✓✗",
          lease_line is not None and all_line is not None and lease_line < all_line,
          {"backend.release_loras": lease_line, "release_all": all_line})
    check("接线: `status()` 里读的是 `lora_cache.shared().facts()` ✓（账本只有一处真源 ✓）",
          "lora_cache.shared" in status_calls and "facts" in status_calls, sorted(set(status_calls)))


def case_zero_deps() -> None:
    check("零依赖: 只 import `lora_cache` **不拖 torch** ✓（无卡 / 无 torch 也能读账本 ✓✗）",
          "torch" not in sys.modules, [name for name in ("torch", "numpy") if name in sys.modules])


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="engine_lora_cache_"))
    case_zero_deps()
    case_identity(tmp)
    case_dedup(tmp)
    case_release(tmp)
    case_release_all(tmp)
    case_unknown_bytes(tmp)
    case_reentrant(tmp)
    case_events(tmp)
    case_self_describe()
    case_wiring()

    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
