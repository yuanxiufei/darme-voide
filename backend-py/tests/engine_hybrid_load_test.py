"""自检：**混合加载的加载层**（``fl2va`` 基底 + ``ref2va`` 的 adaLN 覆盖层 ✓ 2026-09-24 补 ✓）。

计划层 :mod:`hybrid_merge` 早就立了判据 ✓，但**没有任何调用方** ✗✗（「能力接不出去不算功能」✓）；
本套验的是接上之后那半边：**真搬字节** ✓ + **原子落盘** ✓ + **缓存复用两道门** ✓ + **余量/异常** ✓。

判据里三条最容易「看着对、其实错」的 ✓✗：
* ⭐⭐ **合并不改键集**：产物键集必须 == **基底**键集 ✓（少一个键 = 装出来少一层 ✓：装得上、跑得动、画面不对 ✓；
  计划层只看「覆盖层里哪些是 adaLN」✗ —— 它**不保证**产出完整 ✓✗）；
* ⭐⭐ **继承基底的 ``__metadata__``** ✗✗（``load_weights`` 就是靠它读结构的 ✓；丢了它 ⇒ 生产物「装不上」，
  而真因在合并这一步 ✓✗）—— 只比对**键集**不算验过 ✗，这里用**读出结构串**来钉 ✓；
* ⭐ **坏缓存 / 源变了两道门**：产物自证（截断 ⇒ 拒 ✓）+ sidecar 源指纹（源数不等也算变 ✓✗）。

⚠️ 本套**零依赖** ✓（不碰 torch ✗）：夹具是**自己写的小 safetensors** ✓ —— 所以验的是
**管道与容器**，**不宣称**数值/画质 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_hybrid_load_test.py
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import cache_guard as cg  # noqa: E402
from app.services.engine import hybrid_load as hl  # noqa: E402
from app.services.engine import safetensors as st  # noqa: E402
from app.services.engine.hybrid_merge import HybridMergeError  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def write_st(path: Path, tensors: Mapping[str, tuple[list[int], bytes]],
             metadata: Mapping[str, str] | None = None) -> Path:
    """写一个**合法**的 safetensors ✓（头部 + 数据区连续铺满 ✓ —— 与规范一致 ✓）。"""
    table: dict[str, Any] = {}
    offset = 0
    for name, (shape, blob) in tensors.items():
        table[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    header = json.dumps({**table, "__metadata__": dict(metadata or {})},
                        separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        for _shape, blob in tensors.values():
            handle.write(blob)
    return path


#: 基底：四层键（一个 adaLN ✓、一个 final adaLN ✓、两个不是 ✗）+ 内嵌结构元数据 ✓
#: ⚠️ ``final_adaln`` 两边都要有 ✓✗ —— 只在覆盖层里出现 ⇒ 计划层会判「不是同族」拒掉 ✓
#:    （那条判据本身是对的 ✓：覆盖层出现基底没有的键 = 混了别的版本 ✓）
BASE_TENSORS: dict[str, tuple[list[int], bytes]] = {
    "blocks.0.adaln.weight": ([2], f32(1.0, 1.0)),
    "blocks.0.final_adaln.weight": ([2], f32(6.0, 6.0)),
    "blocks.0.mlp.weight": ([2], f32(2.0, 2.0)),
    "blocks.0.norm.weight": ([2], f32(3.0, 3.0)),
}
OVERLAY_TENSORS: dict[str, tuple[list[int], bytes]] = {
    "blocks.0.adaln.weight": ([2], f32(9.0, 9.0)),          # ⇒ 该被盖 ✓
    "blocks.0.mlp.weight": ([2], f32(8.0, 8.0)),            # ⇒ 非 adaLN ⇒ 跳过 ✓
    "blocks.0.final_adaln.weight": ([2], f32(7.0, 7.0)),    # ⇒ final ⇒ 默认跳过 ✓
}


def fixtures(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    base = write_st(root / "fl2va.safetensors", BASE_TENSORS,
                    metadata={"config": json.dumps({"hidden": 2}, separators=(",", ":"))})
    overlay = write_st(root / "ref2va.safetensors", OVERLAY_TENSORS)
    return base, overlay


def values_of(path: Path, name: str) -> tuple[float, ...]:
    raw = st.read_tensor_bytes(path, name)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


# ══════════════════════════════════════════════════════════════════════════
# ① 计划层（接上之后的第一道门 ✓）：三种「拒」都在动手之前 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_plan(root: Path) -> None:
    base, overlay = fixtures(root)
    plan = hl.plan_hybrid_load(base, overlay)
    check("① 计划：只取 adaLN ✓、非 adaLN 跳过**并给理由** ✗（不静默忽略 ✓）",
          plan["take"] == ["blocks.0.adaln.weight"]
          and any(item["name"] == "blocks.0.mlp.weight" and item["reason"] for item in plan["skip"]),
          plan)
    check("①′ 计划：``final`` 那档默认**不带** ✓（开关开了才带 ✓，且默认那条在 skip 里**给了理由** ✓）",
          all("final_adaln" not in name for name in plan["take"])
          and any(item["name"] == "blocks.0.final_adaln.weight" and item["reason"]
                  for item in plan["skip"])
          and "blocks.0.final_adaln.weight" in
          hl.plan_hybrid_load(base, overlay, include_final_adaln=True)["take"], plan)
    check("①″ ⭐ 键集不变 ⇒ **产物大小 == 基底大小** ✗✗（数据区就是基底那一份 ✓）",
          plan["mergedBytes"] == plan["baseBytes"] and plan["bytesFromOverlay"] == 8,
          (plan["mergedBytes"], plan["bytesFromOverlay"]))
    check("①‴ 计划里带上**要继承的元数据键** ✓（装载靠它读结构 ✓✗）",
          plan["inheritMetadataKeys"] == ["config"], plan["inheritMetadataKeys"])

    foreign = write_st(root / "other.safetensors", {"blocks.9.adaln.weight": ([2], f32(0.0, 0.0))})
    try:
        hl.plan_hybrid_load(base, foreign)
        check("①⁴ 基底里没有这个键 ⇒ **拒** ✗（两边不是同族 ✓ —— 别把别的模型混进来 ✓）", False, "没拒")
    except HybridMergeError as err:
        check("①⁴ 基底里没有这个键 ⇒ **拒** ✗（两边不是同族 ✓ —— 别把别的模型混进来 ✓）",
              "不是同族" in str(err), str(err)[:120])

    wrong_shape = write_st(root / "shape.safetensors", {"blocks.0.adaln.weight": ([4], f32(0, 0, 0, 0))})
    try:
        hl.plan_hybrid_load(base, wrong_shape)
        check("①⁵ 形状不一致 ⇒ **拒** ✗ 且**两个形状都印出来** ✓", False, "没拒")
    except HybridMergeError as err:
        check("①⁵ 形状不一致 ⇒ **拒** ✗ 且**两个形状都印出来** ✓",
              "(2,)" in str(err) and "(4,)" in str(err), str(err)[:140])

    broken = root / "broken.safetensors"
    broken.write_bytes(b"\x00" * 4)
    try:
        hl.plan_hybrid_load(base, broken)
        check("①⁶ 坏源（不是 safetensors）⇒ 拒 ✗ 且异常类型是 ``HybridLoadError`` ✓"
              "（别漏出别人的异常类型 ✓✗）", False, "没拒")
    except hl.HybridLoadError as err:
        check("①⁶ 坏源（不是 safetensors）⇒ 拒 ✗ 且异常类型是 ``HybridLoadError`` ✓"
              "（别漏出别人的异常类型 ✓✗）", "不拿坏文件去合并" in str(err), str(err)[:120])

    truncated = root / "truncated.safetensors"
    truncated.write_bytes(base.read_bytes()[:-4])          # 数据区少 4 字节 ✓
    try:
        hl.plan_hybrid_load(base, truncated)
        check("①⁷ 被截断的源 ⇒ 拒 ✗（头说 8 字节、文件只有 4 ✓✗）", False, "没拒")
    except hl.HybridLoadError as err:
        check("①⁷ 被截断的源 ⇒ 拒 ✗（头说 8 字节、文件只有 4 ✓✗）", bool(err), str(err)[:120])


# ══════════════════════════════════════════════════════════════════════════
# ② 真合并（搬字节 + 原子落盘 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_merge(root: Path) -> None:
    base, overlay = fixtures(root)
    cache = root / "merged.safetensors"
    report = hl.run_hybrid_load(base, overlay, cache)
    check("② 合并 ⇒ 产物是真 safetensors ✓（自家读取器判 ok ✓ 无 problems ✓）",
          cache.exists() and st.inspect(cache).ok, st.inspect(cache).problems[:3])
    check("②′ ⭐⭐ **键集 == 基底键集** ✗✗（产物键集被计划层「只取 adaLN」误导的话会少键 ✓："
          "装得上、跑得动、画面不对 ✓）",
          set(st.inspect(cache).tensors) == set(BASE_TENSORS), sorted(st.inspect(cache).tensors))
    check("②″ ⭐ adaLN 的值来自**覆盖层** ✓（逐值比 ✓ 不是看名字对 ✓）",
          values_of(cache, "blocks.0.adaln.weight") == (9.0, 9.0),
          values_of(cache, "blocks.0.adaln.weight"))
    check("②‴ ⭐ 非 adaLN 与 ``final`` 的值都来自**基底** ✓（被覆盖层同名的那个**不许**串进来 ✓✗）",
          values_of(cache, "blocks.0.mlp.weight") == (2.0, 2.0)
          and values_of(cache, "blocks.0.final_adaln.weight") == (6.0, 6.0),
          (values_of(cache, "blocks.0.mlp.weight"), values_of(cache, "blocks.0.final_adaln.weight")))
    merged_meta = st.read_header(cache)[0]
    check("②⁴ ⭐⭐ 产物**继承了基底的内嵌元数据** ✗✗（``load_weights`` 靠它读结构 ✓；"
          "丢了它 ⇒ 装不上而真因在合并这步 ✓✗）",
          json.loads(merged_meta.get("config") or "{}").get("hidden") == 2, merged_meta)
    hybrid_meta = json.loads(merged_meta.get("h3_hybrid") or "{}")
    check("②⁵ 产物里**自带混合理由** ✓（两边名字 + 源指纹 + 取/跳数 ✓ —— 出问题时先看它 ✓）",
          hybrid_meta.get("base") == "fl2va.safetensors"
          and hybrid_meta.get("overlay") == "ref2va.safetensors"
          and len(hybrid_meta.get("baseFingerprint") or []) == 3, hybrid_meta)
    check("②⁶ 报告的字节数与盘上一致 ✓（不是估算 ✓）",
          report.bytes == cache.stat().st_size and report.bytes > 0, (report.bytes, cache.stat().st_size))
    check("②⁷ ⭐ 原子落盘 ⇒ **不留 ``.tmp``** ✗（别人看不到半成品 ✓）",
          not list(root.glob("*.tmp")) and report.reused is False, [p.name for p in root.glob("*.tmp")])
    check("②⁸ 报告的 ``notes`` 里**明说本层不验数值** ✗（别让人以为头自洽 == 数值对 ✓）",
          any("不验张量内容" in note for note in report.notes), report.notes)

    # 数据区中间被改：头仍然自洽 ⇒ 本层**仍判有效** ✗（这是**已知边界** ✓，写出来而不是假装 ✓）
    with cache.open("r+b") as handle:
        handle.seek(cache.stat().st_size - 1)
        handle.write(b"\xff")
    reused_after_tamper = hl.run_hybrid_load(base, overlay, cache)
    check("②⁹ 边界（**如实写出来** ✓）：只改数据区字节 ⇒ 头自洽 + 源没变 ⇒ **仍判有效** ✗"
          "（要验数值必须拿真权重比对 ✓ —— 本层不声称能发现 ✓✗）",
          reused_after_tamper.reused is True, reused_after_tamper.reason)


# ══════════════════════════════════════════════════════════════════════════
# ③ 缓存复用两道门（坏缓存 / 源变了 ⇒ 一律重合并 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_cache(root: Path) -> None:
    base, overlay = fixtures(root)
    cache = root / "cache2.safetensors"
    first = hl.run_hybrid_load(base, overlay, cache)
    second = hl.run_hybrid_load(base, overlay, cache)
    check("③ 第二次 ⇒ **复用** ✓ 且理由说清两道门（产物自证 + 源指纹 ✓）",
          first.reused is False and second.reused is True
          and "自证" in second.reason and "指纹" in second.reason, second.reason)

    forced = hl.run_hybrid_load(base, overlay, cache, force=True)
    check("③′ ``force=True`` ⇒ **即使缓存有效也重合并** ✓（门禁统一支持 force ✓）",
          forced.reused is False and "force" in forced.reason, forced.reason)

    # 源变了（内容与头哈希都变 ✓）
    write_st(overlay, {**OVERLAY_TENSORS, "blocks.0.adaln.weight": ([2], f32(5.0, 5.0))})
    changed = hl.run_hybrid_load(base, overlay, cache)
    check("③″ ⭐ 覆盖层内容变了 ⇒ 失效并重合并 ✓ 且新值**真进了产物** ✓（不是只看状态 ✓）",
          changed.reused is False and values_of(cache, "blocks.0.adaln.weight") == (5.0, 5.0),
          (changed.reused, values_of(cache, "blocks.0.adaln.weight")))

    cg.cache_sidecar_path(cache).unlink()
    no_sidecar = hl.run_hybrid_load(base, overlay, cache)
    check("③‴ sidecar 被删 ⇒ **失效** ✓（没有指纹 = 缓存失效 ✓ —— 宁可慢一次 ✓ 不拿可疑缓存跑 ✓）",
          no_sidecar.reused is False and "sidecar" in no_sidecar.reason, no_sidecar.reason)

    sidecar = json.loads(cg.cache_sidecar_path(cache).read_text(encoding="utf-8"))
    sidecar.pop("overlay")                       # 源数不等 ✓
    cg.cache_sidecar_path(cache).write_text(json.dumps(sidecar), encoding="utf-8")
    fewer = hl.run_hybrid_load(base, overlay, cache)
    check("③⁴ ⭐ **源数不等** ⇒ 失效 ✓✗（少一个源说明输入集合变了 ⇒ 产物不再对应 ✓）",
          fewer.reused is False, fewer.reason)

    # ⭐ 产物被截断 ⇒ 头部自证失败（数据区少了 ⇒ 偏移吃不满 ✓）
    hl.run_hybrid_load(base, overlay, cache)
    data = cache.read_bytes()
    cache.write_bytes(data[:-8])
    cut = hl.run_hybrid_load(base, overlay, cache)
    check("③⁵ ⭐⭐ 产物被**截断** ⇒ 头部自证失败 ⇒ 重合并 ✓✗（坏缓存被当有效缓存反复加载是"
          "上游踩过的坑 ✓）",
          cut.reused is False and "不自洽" in cut.reason, cut.reason)


# ══════════════════════════════════════════════════════════════════════════
# ④ 余量与 sidecar 失败（都不许静默 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_disk_and_sidecar(root: Path) -> None:
    base, overlay = fixtures(root)
    cache = root / "disk.safetensors"
    original = shutil.disk_usage
    usage = collections.namedtuple("usage", ["total", "used", "free"])
    shutil.disk_usage = lambda _path: usage(1000, 900, 100)  # type: ignore[assignment]
    try:
        hl.run_hybrid_load(base, overlay, cache)
        rejected = False
        detail: Any = "居然写了"
    except hl.HybridLoadError as err:
        rejected = True
        detail = str(err)
    finally:
        shutil.disk_usage = original
    check("④ ⭐ 余量不足 ⇒ **拒** ✗ 且**一个字节都不写** ✗（文案里两个数字都要有 ✓：需要 / 可用 ✓）",
          rejected and "GiB" in str(detail) and "一个字节都没写" in str(detail)
          and not cache.exists() and not list(root.glob("*.tmp")), detail)

    ok_cache = root / "sidecar.safetensors"
    sidecar_path = cg.cache_sidecar_path(ok_cache)
    sidecar_path.mkdir()                       # 让 sidecar 写不进去 ✓
    report = hl.run_hybrid_load(base, overlay, ok_cache)
    check("④′ ⭐ sidecar 写失败 ⇒ **不阻塞** ✓（退化成「下次重合并」= 慢 ✓ 不是错 ✓✗）"
          "但报告里**必须看得见** ✗",
          report.reused is False and report.sidecar_written is False
          and bool(report.sidecar_error) and ok_cache.exists(), report.sidecar_error)
    next_run = hl.run_hybrid_load(base, overlay, ok_cache)
    check("④″ 没写成 sidecar ⇒ **下次会重新合并** ✓（如实反映，不是假装缓存好了 ✓）",
          next_run.reused is False, next_run.reason)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 端点：业务面真拿得到 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_api(root: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    base, overlay = fixtures(root / "api")
    plan = client.post("/api/v1/engine/hybrid-merge",
                       json={"base": str(base), "overlay": str(overlay)})
    check("⑤ POST /engine/hybrid-merge（默认**只计划** ✓ 不写盘 ✗）",
          plan.status_code == 200 and (plan.json().get("data") or {}).get("take") == ["blocks.0.adaln.weight"],
          (plan.status_code, plan.json().get("data")))

    cache = root / "api" / "out.safetensors"
    applied = client.post("/api/v1/engine/hybrid-merge",
                          json={"base": str(base), "overlay": str(overlay), "cachePath": str(cache),
                                "apply": True})
    data = applied.json().get("data") or {}
    check("⑤′ ``apply=true`` ⇒ **真合并** ✓ 产物在盘上 + 报告带 ``bytes`` ✓",
          applied.status_code == 200 and data.get("reused") is False and cache.exists()
          and data.get("bytes") == cache.stat().st_size, (applied.status_code, data.get("bytes")))
    again = client.post("/api/v1/engine/hybrid-merge",
                        json={"base": str(base), "overlay": str(overlay), "cachePath": str(cache),
                              "apply": True})
    check("⑤″ 再调一次 ⇒ **复用** ✓（业务面也能吃到缓存 ✓）",
          again.status_code == 200 and (again.json().get("data") or {}).get("reused") is True,
          again.json().get("data"))

    bad = client.post("/api/v1/engine/hybrid-merge",
                      json={"base": str(base), "overlay": str(root / "nope.safetensors"), "apply": True})
    check("⑤‴ 源读不到 ⇒ **400 带理由** ✗（不是 500 不是静默 ✓✗）",
          bad.status_code == 400 and "不拿坏文件去合并" in str(bad.json().get("message")),
          (bad.status_code, bad.json().get("message")))
    check("⑤⁴ 少参数 ⇒ **400** ✓（不是 500 ✗）",
          client.post("/api/v1/engine/hybrid-merge", json={}).status_code == 400)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="hybrid_load_"))
    case_plan(root)
    case_merge(root / "merge")
    case_cache(root / "cache")
    case_disk_and_sidecar(root / "disk")
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
