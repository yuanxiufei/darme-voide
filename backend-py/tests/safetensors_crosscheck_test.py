"""S7 自检：**纯 Python safetensors 读取器 vs 官方库**（独立交叉验证 ✓ 2026-09-17）。

为什么值得单独一套：`engine/safetensors.py` 是**自己写的**（第 68 步 ✓）—— 它撑起了权重体检
（截断检测 ✓、量化配套检查 ✓、加载计划 ✓）。自测通过只能说明"自圆其说" ✗，
所以这里请**参照实现**（官方 `safetensors` 库 ✓）来比对 ✓：

* 张量名集合、形状、精度**逐项**要对得上 ✓；
* `read_tensor_bytes` 的字节要**逐字节**等于官方 `get_slice` 的原始数据 ✓；
* **截断文件**：官方库也拒绝 ✓ ⇒ 证明我们的"截断检测"不是凭空判 ✓。

⚠️ 官方库没装就**显式 SKIP** ✓（本套件不许因为没有依赖就默默变绿 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/safetensors_crosscheck_test.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import safetensors as mine  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _official() -> Any:
    try:
        import numpy as np  # noqa: PLC0415
        from safetensors.numpy import load_file, save_file  # noqa: PLC0415
    except ImportError:
        return None
    return np, save_file, load_file


def case_crosscheck(root: Path) -> None:
    bundle = _official()
    if bundle is None:
        skip("官方 `safetensors` / `numpy` 未安装 ⇒ 交叉验证跳过（本读取器仍由 engine_inventory_test 覆盖 ✓）")
        return
    np, save_file, load_file = bundle

    tensors = {
        "blocks.0.attn.wq.weight": np.arange(24, dtype=np.float32).reshape(4, 6),
        "blocks.1.attn.wq.weight": np.arange(24, dtype=np.float32).reshape(4, 6) + 100.0,
        "embed.weight": np.arange(16, dtype=np.float16).reshape(4, 4),
        "quant.weight": np.zeros((8,), dtype=np.uint8),
    }
    path = root / "official.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt", "made-by": "crosscheck"})

    info = mine.inspect(path)
    check("① 官方库写的文件，我们的读取器判「结构完好」（不误报 ✗）", info.ok, info.problems)

    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        keys = sorted(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
        dtypes = {key: handle.get_slice(key).get_dtype() for key in keys}

    check("② 张量名集合逐项一致 ✓", sorted(info.tensors) == keys, (sorted(info.tensors), keys))
    check("③ 形状逐项一致 ✓",
          {key: info.tensors[key].shape for key in keys} == shapes,
          {key: info.tensors[key].shape for key in keys})
    check("④ 精度串一致（官方也用 `F32`/`F16`/`U8` 这套 ✓）",
          {key: info.tensors[key].dtype for key in keys} == dtypes,
          {key: info.tensors[key].dtype for key in keys})
    check("⑤ 元数据一致（`__metadata__` 解析对 ✓）",
          info.metadata.get("made-by") == "crosscheck", info.metadata)

    raw_mine = mine.read_tensor_bytes(path, "blocks.1.attn.wq.weight")
    raw_reference = tensors["blocks.1.attn.wq.weight"].tobytes()
    check("⑥ ⭐ `read_tensor_bytes` 与官方字节**逐字节相同**（我们的偏移算法是对的 ✓）",
          raw_mine == raw_reference, (len(raw_mine), len(raw_reference)))
    # ⚠️ 第一版这里写的是 `total_tensor_bytes == sum(每个张量的 nbytes)` ✗ —— 那是**套套逻辑** ✗
    #    （前者本来就是后者累加出来的 ✓，永远为真 ⇒ 这种断言只增加"通过数"、不增加信息 ✗）。
    #    改成与**数据区实际大小**比 ✓：相等 ⇒ 没有空洞/没有重叠 ✓（这是独立事实 ✓）。
    data_region = info.file_bytes - 8 - info.header_bytes
    check("⑦ 各张量字节之和 == 数据区大小（无空洞、无重叠 ✓ —— 与声明值比较，非自证 ✓）",
          info.total_tensor_bytes == data_region, (info.total_tensor_bytes, data_region))

    # 官方写的文件，用官方读回来必须与我们的预览一致（顺序/数值都对 ✓）
    loaded = load_file(str(path))
    check("⑧ 数值层面也对得上（取前几个元素比 ✓）",
          loaded["embed.weight"].reshape(-1)[:4].tolist()
          == list(np.frombuffer(mine.read_tensor_bytes(path, "embed.weight"), dtype=np.float16)[:4]),
          "")

    # ── 截断：官方拒绝 ⇒ 说明我们的"截断检测"不是凭空判 ✓ ────────────────
    broken = root / "broken.safetensors"
    raw = path.read_bytes()
    broken.write_bytes(raw[:-32])
    check("⑨ ⭐ 截断文件：**我们的读取器拒绝**并说清原因 ✓", not mine.inspect(broken).ok,
          mine.inspect(broken).problems)
    official_rejected = False
    try:
        load_file(str(broken))
    except Exception:  # noqa: BLE001 —— 官方抛的异常类型不稳定，只关心"它拒绝了" ✓
        official_rejected = True
    check("⑩ ⭐ **官方库同样拒绝** ⇒ 交叉证明「截断检测」成立 ✓（不是我们一家之言 ✓）",
          official_rejected, "官方居然读成功了 ✗")

    # ── 我们自造的文件（不经官方库 ✓）也要能被官方读回来 ────────────────
    hand = root / "handmade.safetensors"
    payload = json.dumps({
        "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    }).encode("utf-8")
    hand.write_bytes(struct.pack("<Q", len(payload)) + payload
                     + np.arange(4, dtype=np.float32).tobytes())
    check("⑪ 我们**手写**的表头能被官方库读懂（格式理解无误 ✓）",
          load_file(str(hand))["w"].tolist() == [[0.0, 1.0], [2.0, 3.0]],
          load_file(str(hand))["w"].tolist())


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="st_cross_"))
    case_crosscheck(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    total = len(_RESULTS)
    print()
    print(f"SUMMARY: {total - len(failures)}/{total} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
