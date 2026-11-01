"""无缓冲重叠读自检 ✓（移植自 `reference/expert-stream` ✓，2026-11-01 ✓）。

判据只认一件事：**读回来的字节与普通 `read` 逐字节相同** ✓✗（性能不进判据 ✗ —— 测速会飘 ✗）。

⚠️ ⑧ 那条是**真事故**的回归钉 ✓✗：第一版把原件里的「每槽手动复位事件」漏掉了 ✗，
症状不是报错 ✗ 而是 **`GetOverlappedResult` 对还没完成的那笔返回 TRUE ＋ 0 字节** ✓✗
（单笔读全对 ✓、三笔一批必挂 ✓）⇒ 若把它"修"成容忍短读，装权重时就会**静默装出半件模型** ✗✗。
"""
from __future__ import annotations

import json
import os
import random
import struct
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import win_io as wio  # noqa: E402

_FAILS: list[str] = []
_SKIPS: list[str] = []

PAYLOAD = bytes(range(256)) * 4097 + b"tail-9bytes"


def check(label: str, ok: bool, detail: object = "") -> None:
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _write(tmp: str) -> str:
    path = os.path.join(tmp, "payload.bin")
    with open(path, "wb") as fh:
        fh.write(PAYLOAD)
    return path


# ---------------------------------------------------------------- ① 字节一致


def case_vectors(path: str) -> None:
    """任意字节偏移 ✓（safetensors 的张量偏移就是任意字节 ✓✗）。"""
    vectors = [(0, 1), (1, 4096), (4095, 3), (4096, 1), (4097, 8191),
               (len(PAYLOAD) - 9, 9), (0, 0), (12345, 99999)]
    for force_plain in (True, False):
        with wio.UnbufferedReader(path, slots=3, slot_bytes=wio.SECTOR_BYTES * 3,
                                  force_plain=force_plain) as reader:
            mode = reader.describe()["mode"]
            bad = [f"{off}+{ln}" for off, ln in vectors
                   if reader.read(off, ln) != PAYLOAD[off:off + ln]]
            check(f"① [{mode}] 未对齐偏移/跨扇区长度逐字节相同 ✓（{len(vectors)} 组）",
                  not bad, bad)


def case_mode(path: str) -> None:
    with wio.UnbufferedReader(path, force_plain=False) as reader:
        info = reader.describe()
        if wio.windows_available():
            check("②′ Windows 上真走**无缓冲重叠**路径 ✓（不是静默降级 ✓✗）",
                  info["mode"] == "unbuffered" and not info["error"], info)
        else:
            _SKIPS.append("②′ 非 Windows：无缓冲路径不可用，只验降级口径")
            check("②′ 非 Windows ⇒ 降级为 plain 且**写明原因** ✓✗（读得慢还查不出为什么 ✗）",
                  info["mode"] == "plain" and bool(info["error"]), info)
        check("②″ describe 报清槽数/槽容量/扇区 ✓（排版一致才有可比性 ✓）",
              info["slots"] == wio.DEFAULT_SLOTS
              and info["slotBytes"] == wio.DEFAULT_SLOT_BYTES
              and info["sectorBytes"] == wio.SECTOR_BYTES
              and info["fileSize"] == len(PAYLOAD), info)
    with wio.UnbufferedReader(path, force_plain=True) as reader:
        check("②‴ force_plain 能强制走退化路径 ✓（A/B 测速的口 ✓，不然没法对照 ✓✗）",
              reader.describe()["mode"] == "plain", reader.describe())


# ---------------------------------------------------------------- ③ 批量与切块


def case_batch_order(path: str) -> None:
    spans = [(7, 100), (0, 3), (4000, 5000), (4095, 2)]
    with wio.UnbufferedReader(path, slots=2, slot_bytes=wio.SECTOR_BYTES * 2,
                              force_plain=False) as reader:
        got = reader.read_many(spans)
    bad = [span for span, data in zip(spans, got)
           if data != PAYLOAD[span[0]:span[0] + span[1]]]
    check("③ 批量读**按入参顺序**回填 ✓✗（乱序提交是允许的 ✓，回填错位＝静默错数据 ✗）",
          len(got) == len(spans) and not bad, bad)


def case_chunking(path: str) -> None:
    """⚠️ 一段超过槽容量必须切块 ✓：单笔无缓冲读的缓冲得落在一个槽里 ✓✗。"""
    slot_bytes = wio.SECTOR_BYTES * 2
    limit = slot_bytes - wio.SECTOR_BYTES
    length = limit * 4 + 17
    with wio.UnbufferedReader(path, slots=2, slot_bytes=slot_bytes,
                              force_plain=False) as reader:
        if reader.describe()["mode"] != "unbuffered":
            _SKIPS.append("③′ 非 Windows：切块逻辑只在无缓冲路径上，跳过")
            return
        got = reader.read(0, length)
    check("③′ 超槽请求自动切块并**按同一趟顺序**拼回 ✓✗（两趟各算一遍 ⇒ 错位 ✗）",
          got == PAYLOAD[0:length], {"want": length, "got": len(got)})


def case_bounds(path: str) -> None:
    with wio.UnbufferedReader(path, force_plain=False) as reader:
        for off, ln in ((0, len(PAYLOAD) + 1), (len(PAYLOAD) - 1, 2), (10 ** 9, 1)):
            raised = False
            try:
                reader.read(off, ln)
            except ValueError:
                raised = True
            check(f"④ 越界读当场拒 ✓（offset={off} length={ln}）—— 读短的后果是装出半件模型 ✗✗",
                  raised)
        for bad in ((-1, 1), (0, -1)):
            raised = False
            try:
                wio.Span(*bad)
            except ValueError:
                raised = True
            check(f"④′ 负偏移/负长度构造期就拒 ✓（{bad}）", raised)


def case_closed(path: str) -> None:
    reader = wio.UnbufferedReader(path, force_plain=True)
    reader.read(0, 4)
    reader.close()
    raised = False
    try:
        reader.read(0, 4)
    except RuntimeError:
        raised = True
    check("⑤ 关闭后再读**当场抛** ✓✗（静默返回空 ⇒ 上层拿到空张量还以为装上了 ✗✗）", raised)
    reader.close()  # 幂等
    check("⑤′ close 幂等 ✓（退出路径重复关不许炸 ✓）", True)


def case_stats(path: str) -> None:
    with wio.UnbufferedReader(path, slots=2, force_plain=False) as reader:
        reader.read_many([(0, 10), (10, 10)])
        stats = reader.describe()["stats"]
    check("⑥ stats 记批数/笔数/字节 ✓（装载可观测，才答得出为何慢 ✓）",
          stats["batches"] == 1 and stats["bytes"] == 20 and stats["reads"] >= 2, stats)


# ---------------------------------------------------------------- ⑦ 对齐不变式


def case_alignment(path: str) -> None:
    """无缓冲读的硬约束（偏移/长度/地址都按扇区对齐 ✓）必须被 `slop` 吃掉 ✓✗。"""
    offsets = list(range(0, 40000, 4093))
    with wio.UnbufferedReader(path, slots=4, slot_bytes=wio.SECTOR_BYTES * 4,
                              force_plain=False) as reader:
        if reader.describe()["mode"] != "unbuffered":
            _SKIPS.append("⑦ 非 Windows：对齐不变式只在无缓冲路径上有意义")
            return
        bad = []
        for off in offsets:
            if reader.read(off, 5) != PAYLOAD[off:off + 5]:
                bad.append(off)
    check(f"⑦ 任意偏移各读 5 字节全对 ✓（{len(offsets)} 个起点，试出 slop 有没有按真偏移切 ✗）",
          not bad, bad)


# ---------------------------------------------------------------- ⑧ 并发事件回归


def case_concurrent_events(path: str) -> None:
    """真事故回归 ✓✗：**一次提交多笔**时每笔必须有自己的手动复位事件 ✓。"""
    spans = [(0, 8192), (8192, 8192), (16384, 8192)]
    with wio.UnbufferedReader(path, slots=3, slot_bytes=wio.DEFAULT_SLOT_BYTES,
                              force_plain=False) as reader:
        if reader.describe()["mode"] != "unbuffered":
            _SKIPS.append("⑧ 非 Windows：并发完成语义只在无缓冲路径上存在")
            return
        got = reader.read_many(spans)
    bad = [span for span, data in zip(spans, got)
           if data != PAYLOAD[span[0]:span[0] + span[1]]]
    check("⑧ 三笔同时在飞仍逐字节正确 ✓✗（无事件 ⇒ 完成信号串台 ⇒ 返回 0 字节 ✗✗）",
          not bad and all(len(d) == span[1] for span, d in zip(spans, got)),
          [(span, len(d)) for span, d in zip(spans, got)])


# ---------------------------------------------------------------- ⑨ 接进 TensorReader


def _write_safetensors(path: str) -> dict[str, bytes]:
    """手写一份最小 safetensors ✓（不引 torch ✓：这条判据要在没装 torch 的机器上也绿 ✓）。"""
    payloads = {"a": PAYLOAD[:4096], "b": PAYLOAD[4000:40000], "c": PAYLOAD[:1]}
    header: dict[str, Any] = {}
    offset = 0
    for name, blob in payloads.items():
        header[name] = {"dtype": "U8", "shape": [len(blob)],
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for blob in payloads.values():
            fh.write(blob)
    return payloads


def case_concurrent_correctness(path: str) -> None:
    """⑩ ⭐ 多线程同时读 ✓✗（**真缺陷**回归：槽按批内下标用 ⇒ 两家共用一个 OVERLAPPED ＋
    一个事件 ＋ 一块槽区 ⇒ 短读/读到 0 字节 ✓，而且**单线程永远测不出来** ✗✗）。"""
    size = os.path.getsize(path)
    rng = random.Random(20261101)
    plan = []
    for _ in range(24):
        offset = rng.randrange(0, size - 65536)
        length = rng.randrange(1, 65536)
        plan.append((offset, min(length, size - offset)))

    with wio.UnbufferedReader(path, slots=4, slot_bytes=1 << 20, force_plain=True) as plain:
        expect = [(span, plain.read(*span)) for span in plan]

    with wio.UnbufferedReader(path, slots=4, slot_bytes=1 << 20) as reader:
        bad: list[str] = []

        def worker(worker_id: int) -> None:
            for span, want in expect[worker_id::4]:
                try:
                    got = reader.read(*span)
                except Exception as err:  # noqa: BLE001 - 线程里抛也要收上来 ✓
                    bad.append(f"线程{worker_id} {span} 抛 {type(err).__name__}: {err}")
                    continue
                if got != want:
                    bad.append(f"线程{worker_id} {span} 字节不符（得 {len(got)} 要 {len(want)}）")

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    check("⑩ ⭐ 四个线程同时读，每段字节仍与普通读相同 ✓✗（槽共用的症状是**静默短读** ✗✗）",
          not bad, bad[:3])


def case_tensor_reader(tmp: str) -> None:
    """``safetensors.TensorReader`` 接上无缓冲读口后，字节必须不变 ✓✗。"""
    from app.services.engine import safetensors as st

    path = os.path.join(tmp, "mini.safetensors")
    payloads = _write_safetensors(path)
    names = list(payloads)

    with st.TensorReader(path, unbuffered=False) as plain:
        plain_mode, plain_note = plain.ioMode, plain.ioNote
        plain_bytes = {name: plain.read(name) for name in names}
    with st.TensorReader(path, unbuffered=True) as fast:
        fast_mode, fast_note = fast.ioMode, fast.ioNote
        fast_bytes = {name: fast.read(name) for name in names}
        order = fast.read_many([names[2], names[0], names[1]])

    check("⑨ 同一份文件两条读路逐字节相同 ✓✗（接错偏移就能让权重静默错位 ✗✗）",
          plain_bytes == fast_bytes == payloads, {k: len(v) for k, v in fast_bytes.items()})
    check("⑨′ 强制 plain ⇒ 报 plain 且无降级原因 ✓", plain_mode == "plain" and not plain_note,
          (plain_mode, plain_note))
    if wio.windows_available():
        check("⑨″ Windows 上默认真的走无缓冲 ✓（否则这层就是白接 ✗）",
              fast_mode == "unbuffered" and not fast_note, (fast_mode, fast_note))
    else:
        _SKIPS.append("⑨″ 非 Windows：TensorReader 会退回 plain，只验降级口径")
        check("⑨″ 非 Windows ⇒ 退回 plain 并写明原因 ✓", fast_mode == "plain" and bool(fast_note),
              (fast_mode, fast_note))
    check("⑨‴ read_many 按入参顺序回填 ✓✗（乱序入参错位＝静默错权重 ✗）",
          order == [payloads[names[2]], payloads[names[0]], payloads[names[1]]],
          [len(x) for x in order])
    raised = False
    try:
        fast.read(names[0])  # with 块已退出 ⇒ 已 close
    except st.SafetensorsError:
        raised = True
    check("⑨⁴ 关闭后再读当场抛 ✓（静默重开会把「关干净没」弄丢 ✗）", raised)


def main() -> int:
    print(f"== 无缓冲重叠读自检（windows_available={wio.windows_available()} ✓）==")
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp)
        case_vectors(path)
        case_mode(path)
        case_batch_order(path)
        case_chunking(path)
        case_bounds(path)
        case_closed(path)
        case_stats(path)
        case_alignment(path)
        case_concurrent_events(path)
        case_concurrent_correctness(path)
        case_tensor_reader(tmp)
    for note in _SKIPS:
        print(f"SKIP  {note}")
    print(f"\nSUMMARY: {len(_FAILS)} failed")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
