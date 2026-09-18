"""S7 自检：引擎的**权重体检层**（纯 Python 读 safetensors + 组件就绪报告；2026-09-17）。

为什么值得单独一套：真实模型 **19.53 GiB/个** ✗ —— 跑到一半才发现「文件被截断 / 装错了组件」
是最贵的失败 ✗。本层把校验提前，而且**完全不需要 torch、不需要 GPU、不下载任何权重** ✓：

* safetensors 容器格式极简 ⇒ 我们**自己读写**它（测试里的「合法文件 / 截断文件 / 头部矛盾」
  都用我们自己合成的小文件造出来 ✓ —— 不靠下载真权重才能测 ✗）；
* 就绪报告跑的是**真实清单** `configs/models.json` ✓，但权重根目录指向临时目录 ⇒
  「缺什么 / 装齐了会怎样」都能真验一遍 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_inventory_test.py
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
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="engineinv_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import inventory as inv  # noqa: E402
from app.services.engine import safetensors as st  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ══════════════════════════════════════════════════════════════════════════
# 合成 safetensors（**我们自己的写入器**，只为造测试数据 ✓）
# ══════════════════════════════════════════════════════════════════════════
def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]], *,
                      metadata: dict[str, str] | None = None,
                      truncate_bytes: int = 0,
                      trailing_garbage: int = 0,
                      break_offset: bool = False) -> None:
    """写出一个（默认合法的）safetensors ✓；三个参数用来故意造坏文件 ✓。"""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    blobs: list[bytes] = []
    for name, (dtype, shape) in tensors.items():
        size = 1
        for dim in shape:
            size *= int(dim)
        size *= st.DTYPE_ITEMSIZE[dtype]
        end = offset + size
        if break_offset:  # 头部自述比真实大 ⇒ 触发「超出数据区 / 自相矛盾」✓
            end = offset + size + 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        blobs.append(bytes(size))
        offset = end
    payload = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)
        if trailing_garbage:
            handle.write(b"\x00" * trailing_garbage)
    if truncate_bytes:
        raw = path.read_bytes()
        path.write_bytes(raw[:-truncate_bytes])


# ══════════════════════════════════════════════════════════════════════════
# ① safetensors 读取器
# ══════════════════════════════════════════════════════════════════════════
def case_safetensors(root: Path) -> None:
    good = root / "good.safetensors"
    write_safetensors(good, {
        "weight": ("F16", [4, 8]),
        "bias": ("F32", [8]),
        "quant": ("F8_E4M3", [16]),
    }, metadata={"format": "pt", "note": "自检合成 ✓"})

    info = st.inspect(good)
    check("① 合法文件体检通过（无问题 ✓）", info.ok, info.problems)
    # ⚠️ 一律用 .get()：体检**失败**时 tensors 可能是空的 ✓ —— 直连下标会让自检自己崩 ✗
    #    （初版就崩在 `info.tensors["weight"]` 上：KeyError 而不是 FAIL ✗）
    weight, bias, quant = (info.tensors.get(name) for name in ("weight", "bias", "quant"))
    check("② 张量清单/形状/精度都对（3 个张量；F16 4×8、F32 8、F8_E4M3 16 ✓）",
          info.tensor_count == 3 and weight is not None and bias is not None and quant is not None
          and weight.shape == [4, 8] and weight.dtype == "F16"
          and info.dtype_counts == {"F16": 1, "F32": 1, "F8_E4M3": 1},
          (info.dtype_counts, list(info.tensors)))
    check("③ 字节数 = Σ 形状×dtype（4×8×2 + 8×4 + 16×1 = 112 ✓）",
          weight is not None and bias is not None and quant is not None
          and weight.nbytes == 64 and bias.nbytes == 32 and quant.nbytes == 16
          and info.total_tensor_bytes == 112,
          info.total_tensor_bytes)
    check("④ 元数据可读（__metadata__ ✓）", info.metadata.get("note") == "自检合成 ✓", info.metadata)
    check("⑤ 最大张量排序可用（看结构时最有用 ✓）",
          info.biggest(1)[0].name == "weight", [t.name for t in info.biggest()])

    # 只读一个张量的原始字节 ✓（不整文件进内存 ✓）
    raw = st.read_tensor_bytes(good, "bias")
    check("⑥ 按需只读某张量的字节（bias = 32 字节 ✓）", len(raw) == 32, len(raw))
    check("⑦ 超过 max_bytes 时**拒绝**（防手滑把 19 GiB 拉进内存 ✗）",
          "超过 max_bytes" in str(_raises(lambda: st.read_tensor_bytes(good, "bias", max_bytes=8))))

    # 截断：结构上就能发现 ✓（真机上这是最贵的失败 ✗）
    cut = root / "truncated.safetensors"
    write_safetensors(cut, {"weight": ("F32", [16])}, truncate_bytes=24)
    info_cut = st.inspect(cut)
    check("⑧ **截断文件**当场报出来（不是等到加载时才炸 ✗）",
          not info_cut.ok and any("截断" in p or "超出数据区" in p for p in info_cut.problems),
          info_cut.problems)

    # 头部自述与形状矛盾 ✓
    liar = root / "liar.safetensors"
    write_safetensors(liar, {"weight": ("F32", [4])}, break_offset=True)
    check("⑨ 头部自相矛盾（shape×dtype ≠ 区间长度）也能抓出来 ✓",
          not st.inspect(liar).ok
          and any("自相矛盾" in p or "超出数据区" in p for p in st.inspect(liar).problems),
          st.inspect(liar).problems)

    # 尾部有没人声明的字节 ⇒ 可能写入中断 ✓
    tail = root / "tail.safetensors"
    write_safetensors(tail, {"weight": ("F32", [4])}, trailing_garbage=64)
    check("⑩ 数据区尾部有无人声明的字节 ⇒ 提示可能写入中断 ✓",
          not st.inspect(tail).ok and any("无人声明" in p for p in st.inspect(tail).problems),
          st.inspect(tail).problems)

    # 未知 dtype ✓
    weird = root / "weird.safetensors"
    payload = json.dumps({"w": {"dtype": "F4_TINY", "shape": [1], "data_offsets": [0, 1]}}).encode()
    weird.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\x00")
    check("⑪ 不认识的 dtype ⇒ 明确报错（不猜 ✓）",
          not st.inspect(weird).ok and any("dtype" in p for p in st.inspect(weird).problems),
          st.inspect(weird).problems)

    # GGUF 与不存在 ✓
    gguf = root / "x.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 32)
    check("⑫ GGUF 明确说「读取器不实现」（不假装读过 ✓）",
          any("GGUF" in p for p in st.inspect(gguf).problems), st.inspect(gguf).problems)
    check("⑬ 不存在的文件 ⇒ 有结论而不是抛异常 ✓",
          not st.inspect(root / "nope.safetensors").ok)
    check("⑭ 只有 3 字节的文件 ⇒ 判「太小，不是 safetensors」✓",
          any("太小" in p for p in st.inspect(_tiny(root)).problems))


def _tiny(root: Path) -> Path:
    path = root / "tiny.safetensors"
    path.write_bytes(b"\x01\x02\x03")
    return path


def _raises(fn) -> Exception:  # noqa: ANN001
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return err
    return Exception("（没有抛错 ✗）")


# ══════════════════════════════════════════════════════════════════════════
# ② 就绪报告（**真实清单** + 临时权重根目录 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_readiness(root: Path) -> None:
    catalog = inv.load_catalog()
    check("⑮ 读得到**真实清单**（configs/models.json ✓）",
          catalog.get("error") is None and len(catalog["models"]) >= 10,
          (catalog.get("error"), len(catalog["models"])))
    video = [item for item in catalog["models"] if item.get("category") == "video"]
    required = [item for item in video if item.get("required")]
    check("⑯ 清单里 H3 视频条目结构符合预期（含 required + kind + filename ✓）",
          bool(required) and all(item.get("kind") and item.get("filename") for item in required),
          [(item.get("key"), item.get("kind")) for item in required])

    # 空目录 ⇒ 必需组件全缺 ✓（这正是本机现状 ✓）
    empty = inv.readiness("h3", root=root / "empty")
    check("⑰ 空权重目录 ⇒ ready=False，且**逐条列出缺哪些**（不是只说「失败」✓）",
          empty["ready"] is False and len(empty["missingRequired"]) == len(required)
          and empty["missingRequired"], (empty["missingRequired"], empty["ready"]))
    check("⑰' 缺件时 `brokenRequired` 为空（缺 ≠ 坏 ✓ 两者分开报 ✓）",
          empty["brokenRequired"] == [], empty["brokenRequired"])
    check("⑰'' 可选件也列出来但**不影响**就绪判定 ✓",
          isinstance(empty["optionalMissing"], list)
          and len(empty["components"]) == len(video)
          and empty["ready"] is False, len(video))

    # 按**清单里的真实相对路径**铺好合成权重 ⇒ 应当判定就绪 ✓✓
    filled = root / "filled"
    for item in video:
        if not item.get("required"):
            continue
        where = inv.component_path(item, filled)
        assert where is not None
        if where.suffix.lower() == ".gguf":
            where.parent.mkdir(parents=True, exist_ok=True)
            where.write_bytes(b"GGUF" + b"\x00" * 64)
        else:
            write_safetensors(where, {"weight": ("F16", [8, 8]), "bias": ("F32", [8])},
                              metadata={"source": "selftest"})
    ready = inv.readiness("h3", root=filled)
    check("⑱ 按清单真实路径铺齐必需件 ⇒ ready=True（路径解析逻辑真的对 ✓）",
          ready["ready"] is True and ready["missingRequired"] == []
          and ready["brokenRequired"] == [],
          (ready["missingRequired"], ready["brokenRequired"]))
    check("⑲ 就绪时仍会提示「大小与清单不符」（合成的 128 字节 ≠ 19.53 GiB ✓）",
          bool(ready["suspectRequired"]) and any("大小与清单不符" in p
                                                for item in ready["components"] for p in item["problems"]),
          ready["suspectRequired"])
    check("⑲' 大小可疑**不阻断**就绪（结构完好才是硬判据 ✓ —— 重新导出的量化版体积会合法地变 ✓）",
          ready["ready"] is True and ready["suspectRequired"], ready["ready"])

    # 把一个必需件做成**截断** ⇒ 必须判 broken ✓
    broken_root = root / "broken"
    first = next(item for item in video if item.get("required"))
    for item in video:
        if not item.get("required"):
            continue
        where = inv.component_path(item, broken_root)
        assert where is not None
        if item["key"] == first["key"]:
            write_safetensors(where, {"weight": ("F16", [16, 16])}, truncate_bytes=32)
        elif where.suffix.lower() == ".gguf":
            where.parent.mkdir(parents=True, exist_ok=True)
            where.write_bytes(b"GGUF" + b"\x00" * 64)
        else:
            write_safetensors(where, {"weight": ("F16", [4, 4])})
    bad = inv.readiness("h3", root=broken_root)
    check("⑳ 有一个必需件**截断** ⇒ ready=False 且进 `brokenRequired`（缺 ≠ 坏 ✓）",
          bad["ready"] is False and bad["brokenRequired"] == [first["key"]],
          bad["brokenRequired"])
    check("⑳' 截断组件的问题里带「截断/超出数据区」字样（能据此行动 ✓）",
          any("截断" in p or "超出数据区" in p
              for item in bad["components"] if item["key"] == first["key"] for p in item["problems"]),
          [item["problems"] for item in bad["components"] if item["key"] == first["key"]])

    # 显存估算 ✓（纯函数 ✓）
    est = inv.estimate_vram(19 * 1024 ** 3, capacity_gib=24)
    check("㉑ 显存估算：权重 19 GiB + 安全系数 1.25 ⇒ 约 23.75 GiB，24GB 卡**勉强放得下**",
          est["fits"] is True and 23.0 <= est["estimatedGiB"] <= 24.0, est)
    check("㉒ 显存估算：权重 30 GiB ⇒ fits=False 且余量为负（如实说不够 ✓）",
          inv.estimate_vram(30 * 1024 ** 3)["fits"] is False
          and inv.estimate_vram(30 * 1024 ** 3)["headroomGiB"] < 0, "")
    check("㉓ 估算里带**免责声明**（激活/offload 决定峰值 ⇒ 不承诺 ✓）",
          "disclaimer" in inv.estimate_vram(1024 ** 3))

    # 真实清单里 H3 必需件的**总体量**（本机要下载多少 —— 用户最需要知道的数字 ✓）
    total_gib = sum(float(item.get("size_gib") or 0) for item in required)
    check("㉔ 清单给出的 H3 必需件总体量可算（这是「要下载多少」的答案 ✓）",
          total_gib > 0, round(total_gib, 2))


# ══════════════════════════════════════════════════════════════════════════
# ③ 路由：这份能力要**真能被调用**（只躺在库里不算功能 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_api(root: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    report = client.get("/api/v1/engine/readiness", params={"stage": "h3"})
    data = report.json().get("data") or {}
    check("㉕ GET /engine/readiness 真能调用（200 ✓）且给出逐条缺件清单",
          report.status_code == 200 and data.get("ready") is False
          and bool(data.get("missingRequired")) and data.get("components"),
          (report.status_code, (data.get("missingRequired") or [])[:4]))
    check("㉖ 就绪报告带显存估算与**免责声明**（不承诺 ✓）",
          isinstance(data.get("vram"), dict) and bool((data.get("vram") or {}).get("disclaimer")),
          data.get("vram"))
    check("㉗ 未知阶段 ⇒ 400（不静默回默认 ✗）",
          client.get("/api/v1/engine/readiness", params={"stage": "nope"}).status_code == 400)

    good = root / "api_good.safetensors"
    write_safetensors(good, {"weight": ("F16", [4, 4])}, metadata={"src": "api"})
    detail = client.post("/api/v1/engine/inspect", json={"path": str(good)})
    payload = detail.json().get("data") or {}
    check("㉘ POST /engine/inspect 体检任意文件（只读头部 ✓）",
          detail.status_code == 200 and payload.get("ok") is True
          and payload.get("tensorCount") == 1 and payload.get("biggest"),
          (detail.status_code, payload.get("problems")))

    by_key = client.post("/api/v1/engine/inspect", json={"key": "dit_fl2va_int8"})
    key_payload = by_key.json().get("data") or {}
    check("㉙ 按**清单 key** 体检（前端不用自己拼 kind/filename ✓）；没装也照实说",
          by_key.status_code == 200 and key_payload.get("ok") is False
          and any("不存在" in p or "未安装" in p for p in key_payload.get("problems") or []),
          key_payload.get("problems"))

    check("㉚ 既没 path 也没 key ⇒ 400（不猜 ✓）",
          client.post("/api/v1/engine/inspect", json={}).status_code == 400)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engineinv_files_"))
    case_safetensors(root)
    case_readiness(root)
    case_api(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
