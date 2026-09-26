"""S24 自检：引擎**GGUF 读取器**（零依赖 ✓ 2026-09-20）。

此前 ``loader`` / ``inventory`` 对 ``.gguf`` 只能报「在/不在 + 大小」✗（``gguf-unknown`` ✗）
—— 本套验证**新读取器**真的能读 ✓：合成合法 GGUF（v3 与 v1 ✓）→ 读回核对
（张量表 / 量化方案 / 对齐 ✓）+ 一组「故意坏的」反向证明（截断 / 块不整除 /
未知类型 / 坏 magic / 重叠 ✓）。

结构事实来源：``reference/ollama/fs/gguf/``（``gguf.go`` 头格式 / ``tensor.go`` 块布局 /
``file_type.go`` 方案名 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_gguf_test.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import gguf as gg  # noqa: E402
from app.services.engine import inventory as inv  # noqa: E402
from app.services.engine import loader as ld  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ══════════════════════════════════════════════════════════════════════════
# 合成器（自检自用 ✓）：把「KV + 张量表 + 数据区」按 ollama 的线格式写出来 ✓
# ══════════════════════════════════════════════════════════════════════════
def _kv_bytes(key: str, value_type: int, payload: bytes, version: int,
              prefix: str = "<") -> bytes:
    key_bytes = key.encode() + (b"\x00" if version == 1 else b"")
    return (struct.pack(prefix + "Q", len(key_bytes)) + key_bytes
            + struct.pack(prefix + "I", value_type) + payload)


def _string_payload(text: str, version: int, prefix: str = "<") -> bytes:
    blob = text.encode() + (b"\x00" if version == 1 else b"")
    return struct.pack(prefix + "Q", len(blob)) + blob


def build_gguf(kvs: list[tuple[str, int, bytes]], tensors: list[dict[str, Any]],
               *, version: int = 3, alignment: int = 32, big: bool = False,
               truncate_data: int = 0) -> bytes:
    """合成 GGUF ✓：``kvs`` 是 ``(键, 值类型, 值字节)`` ✓；``tensors`` 每项含
    ``name / type / shape / offset`` ✓（未给 offset 时**按顺序自动分配** ✓；
    数据区自动补零到覆盖最大 end ✓）。"""
    prefix = ">" if big else "<"
    out = bytearray()
    out += (b"FUGG" if big else b"GGUF")
    out += struct.pack(prefix + "I", version)
    out += struct.pack(prefix + "QQ", len(tensors), len(kvs))
    for key, value_type, payload in kvs:
        out += _kv_bytes(key, value_type, payload, version, prefix)
    # 数据区相对偏移：先算每个张量的字节数，再顺序分配 ✓（不自批 offset 就不重叠 ✓）
    cursor = 0
    assigned: list[int] = []
    for tensor in tensors:
        if "offset" in tensor:
            offset = int(tensor["offset"])
        else:
            offset = cursor
        nbytes = gg.tensor_nbytes(int(tensor["type"]), list(tensor["shape"]))
        assigned.append(offset)
        cursor = offset + max(nbytes, 0)
    for tensor, offset in zip(tensors, assigned):
        name_bytes = tensor["name"].encode() + (b"\x00" if version == 1 else b"")
        out += struct.pack(prefix + "Q", len(name_bytes)) + name_bytes
        shape = tensor["shape"]
        out += struct.pack(prefix + "I", len(shape))
        for dim in shape:
            out += struct.pack(prefix + "Q", int(dim))
        out += struct.pack(prefix + "I", int(tensor["type"]))
        out += struct.pack(prefix + "Q", offset)
    # 数据区起点对齐 ✓
    if len(out) % alignment:
        out += b"\x00" * (alignment - len(out) % alignment)
    data_len = cursor
    if truncate_data:
        data_len = max(0, data_len - truncate_data)
    out += b"\x00" * data_len
    return bytes(out)


def _u32_payload(value: int, prefix: str = "<") -> bytes:
    return struct.pack(prefix + "I", value)


def _good_kvs(prefix: str = "<") -> list[tuple[str, int, bytes]]:
    return [
        ("general.architecture", 8, _string_payload("minimax_h3", 3, prefix)),
        ("general.file_type", 4, _u32_payload(15, prefix)),          # 15 = Q4_K_M ✓
        ("general.alignment", 4, _u32_payload(32, prefix)),
    ]


def _good_tensors() -> list[dict[str, Any]]:
    return [
        {"name": "blocks.0.attn.qkv.weight", "type": 12, "shape": [256, 512]},   # Q4_K ✓
        {"name": "blocks.1.attn.qkv.weight", "type": 12, "shape": [256, 512]},
        {"name": "blocks.2.attn.qkv.weight", "type": 12, "shape": [256, 512]},
        {"name": "blocks.0.mlp.fc.weight", "type": 14, "shape": [256, 256]},     # Q6_K ✓
        {"name": "final.weight", "type": 0, "shape": [64]},                      # F32 ✓
    ]


# ══════════════════════════════════════════════════════════════════════════
# ① 块布局常量表（与 ollama 公式逐一对齐 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_type_table() -> None:
    check("① 块字节数与 ollama 公式一致（Q4_K=144 / Q6_K=210 / Q8_0=34 / NVFP4=36 ✓）",
          gg.TENSOR_TYPES[12][2] == 144 and gg.TENSOR_TYPES[14][2] == 210
          and gg.TENSOR_TYPES[8][2] == 34 and gg.TENSOR_TYPES[40][2] == 36,
          {k: v for k, v in gg.TENSOR_TYPES.items() if k in (8, 12, 14, 40)})
    check("② tensor_nbytes：256×512 的 Q4_K = (131072/256)×144 = 73728 字节 ✓",
          gg.tensor_nbytes(12, [256, 512]) == (256 * 512 // 256) * 144,
          gg.tensor_nbytes(12, [256, 512]))
    check("③ 第 0 维不整除块大小 ⇒ -1（装载器会拒载 ✗）",
          gg.tensor_nbytes(12, [100, 512]) == -1, gg.tensor_nbytes(12, [100, 512]))
    check("④ 方案名表：file_type 15 = Q4_K_M ✓（ollama file_type.go ✓）",
          gg.FILE_TYPES[15] == "Q4_K_M" and gg.FILE_TYPES[0] == "F32", gg.FILE_TYPES.get(15))


# ══════════════════════════════════════════════════════════════════════════
# ② 读回核对（合成 → inspect ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_roundtrip(root: Path) -> None:
    path = root / "good.gguf"
    path.write_bytes(build_gguf(_good_kvs(), _good_tensors()))
    info = gg.inspect(path)
    check("⑤ 合法 v3 ⇒ ok ✓（张量 5 个 / 版本 3 / 元数据读回 ✓）",
          info.ok and info.tensor_count == 5 and info.version == 3
          and info.metadata.get("general.architecture") == "minimax_h3",
          (info.ok, info.tensor_count, info.version, info.metadata, info.problems))
    check("⑥ 量化方案来自 general.file_type ⇒ Q4_K_M ✓（不再是 gguf-unknown ✓）",
          info.quant_scheme == "Q4_K_M", info.quant_scheme)
    check("⑦ dtype 计数按类型名（Q4_K=3 / Q6_K=1 / F32=1 ✓）",
          info.dtype_counts == {"Q4_K": 3, "Q6_K": 1, "F32": 1}, info.dtype_counts)
    check("⑧ 张量字节数逐个核对（Q4_K 256×512=73728 / F32 64=256 ✓）",
          info.tensors["blocks.0.attn.qkv.weight"].nbytes == 73728
          and info.tensors["final.weight"].nbytes == 256,
          {name: t.nbytes for name, t in info.tensors.items()})
    check("⑨ 数据区起点按 32 对齐 ✓",
          info.data_start % 32 == 0 and info.data_start >= 24, info.data_start)

    # 大端镜像（FUGG ✓）
    big_path = root / "big.gguf"
    big_path.write_bytes(build_gguf(_good_kvs(">"), _good_tensors(), big=True))
    big_info = gg.inspect(big_path)
    check("⑩ 大端 FUGG 也能读（ollama 同款支持 ✓）",
          big_info.ok and big_info.tensor_count == 5 and big_info.byte_order == "big",
          (big_info.byte_order, big_info.tensor_count, big_info.problems))


# ══════════════════════════════════════════════════════════════════════════
# ③ v1 字符串（null 终止 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_v1(root: Path) -> None:
    kvs = [("general.architecture", 8, _string_payload("wan2.1", 1))]
    tensors = [{"name": "blocks.0.weight", "type": 8, "shape": [32]}]
    path = root / "v1.gguf"
    path.write_bytes(build_gguf(kvs, tensors, version=1))
    info = gg.inspect(path)
    check("⑪ v1 字符串带 null 终止（长度含终止符 ✓）⇒ 读回剥掉终止符 ✓",
          info.ok and info.metadata.get("general.architecture") == "wan2.1",
          (info.metadata, info.problems))

    # v1 但漏了 null 终止 ⇒ 明确报错 ✓
    bad = bytearray(path.read_bytes())
    # 找到字符串负载区（magic4+ver4+tc8+kvc8 = 24 之后就是第一个 key）✓
    key_len = struct.unpack_from("<Q", bad, 24)[0]
    bad[24 + 8 + key_len - 1] = 0x41   # 终止符位置改成 'A' ✗
    bad_path = root / "v1_bad.gguf"
    bad_path.write_bytes(bytes(bad))
    bad_info = gg.inspect(bad_path)
    check("⑫ v1 缺 null 终止符 ⇒ 明确报错（不猜 ✓）",
          not bad_info.ok and any("null" in p for p in bad_info.problems), bad_info.problems)


# ══════════════════════════════════════════════════════════════════════════
# ④ 反向证明（故意坏的 ✓ —— 检测不到才是 bug ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_broken(root: Path) -> None:
    # 截断：数据区砍掉一半 ✗
    cut = root / "cut.gguf"
    cut.write_bytes(build_gguf(_good_kvs(), _good_tensors(), truncate_data=100000))
    info = gg.inspect(cut)
    check("⑬ 数据区截断 ⇒ 报「文件被截断」且 verified=False ✓",
          not info.ok and any("截断" in p for p in info.problems), info.problems)

    # 第 0 维不整除块大小 ✗（Q4_K 块 256，给 100 ✓）
    misaligned = root / "mis.gguf"
    tensors = [{"name": "blocks.0.weight", "type": 12, "shape": [100, 512]}]
    misaligned.write_bytes(build_gguf(_good_kvs(), tensors))
    minfo = gg.inspect(misaligned)
    check("⑭ 块不整除 ⇒ 报「不整除块大小」（装载前发现 ✗ 不留到显存里炸 ✓）",
          not minfo.ok and any("不整除" in p for p in minfo.problems), minfo.problems)

    # 未知类型 id（34 = TQ1_0，ollama 也没有块布局 ✗）
    unknown = root / "unknown.gguf"
    tensors = [{"name": "blocks.0.weight", "type": 34, "shape": [256]}]
    unknown.write_bytes(build_gguf(_good_kvs(), tensors))
    uinfo = gg.inspect(unknown)
    check("⑮ 未知块布局 ⇒ 明确报「算不出字节数」（不假装验过 ✓）",
          not uinfo.ok and any("块布局" in p for p in uinfo.problems), uinfo.problems)

    # 坏 magic ✗
    bad_magic = root / "bad.gguf"
    bad_magic.write_bytes(b"NOPE" + b"\x00" * 64)
    binfo = gg.inspect(bad_magic)
    check("⑯ 坏 magic ⇒ 明确说「不是 GGUF 文件」✓",
          not binfo.ok and any("不是 GGUF" in p for p in binfo.problems), binfo.problems)

    # 重叠 ✗：两个张量 offset 都从 0 开始 ⇒ spans 相交 ✓
    overlap = root / "overlap.gguf"
    tensors = [
        {"name": "a.weight", "type": 0, "shape": [128], "offset": 0},
        {"name": "b.weight", "type": 0, "shape": [128], "offset": 0},
    ]
    overlap.write_bytes(build_gguf(_good_kvs(), tensors))
    oinfo = gg.inspect(overlap)
    check("⑰ 两个张量数据区重叠 ⇒ 报「拼错」✓",
          not oinfo.ok and any("重叠" in p for p in oinfo.problems), oinfo.problems)

    # 缺 file_type ⇒ 从张量类型分布推 ✓
    no_ft = root / "noft.gguf"
    no_ft.write_bytes(build_gguf([("general.architecture", 8, _string_payload("x", 3))],
                                 _good_tensors()))
    ninfo = gg.inspect(no_ft)
    check("⑱ 缺 general.file_type ⇒ 从主导量化类型推（Q4_K ✓）",
          ninfo.ok and ninfo.quant_scheme == "Q4_K", (ninfo.quant_scheme, ninfo.problems))


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 接线层：inventory / loader 不再是「gguf-unknown」✗
# ══════════════════════════════════════════════════════════════════════════
def case_wiring(root: Path) -> None:
    model_root = root / "models" / "diffusion_models"
    model_root.mkdir(parents=True, exist_ok=True)
    good = model_root / "h3.gguf"
    good.write_bytes(build_gguf(_good_kvs(), _good_tensors()))
    entry = {"key": "dit_gguf", "name": "H3 GGUF", "kind": "diffusion_models",
             "filename": "h3.gguf", "file_path": "diffusion_models/h3.gguf",
             "size_gib": good.stat().st_size / 2 ** 30, "required": True}

    status = inv.component_status(entry, root=root / "models")
    check("⑲ inventory：verified=True + tensorCount=5 + quantScheme=Q4_K_M ✓",
          status["verified"] and status["tensorCount"] == 5
          and status.get("quantScheme") == "Q4_K_M",
          (status.get("verified"), status.get("tensorCount"), status.get("quantScheme"),
           status.get("problems")))

    plan = ld.plan_component(entry, root=root / "models")
    check("⑳ loader：quantScheme=Q4_K_M（不再是 gguf-unknown ✓）且无「未实现」警告 ✓",
          plan.quantScheme == "Q4_K_M" and plan.verified
          and not any("未实现" in w for w in plan.warnings),
          (plan.quantScheme, plan.verified, plan.warnings, plan.problems))
    check("㉑ loader：层块 blocks.N 连续（头 blocks、3 层 ✓）",
          plan.blockHead == "blocks" and plan.blockCount == 3, (plan.blockHead, plan.blockCount))

    # 垃圾 GGUF（旧测试的写法：GGUF + 全零 ✓）⇒ 现在**真的去读** ⇒ 结构坏 ⇒ 阻断 ✓
    garbage = model_root / "garbage.gguf"
    garbage.write_bytes(b"GGUF" + b"\x00" * 128)
    gstatus = inv.component_status({**entry, "filename": "garbage.gguf",
                                    "file_path": "diffusion_models/garbage.gguf"},
                                   root=root / "models")
    check("㉒ 垃圾 GGUF ⇒ verified=False 且有明确结论（不假装读过 ✓）",
          not gstatus["verified"] and gstatus["problems"], gstatus["problems"])
    gplan = ld.plan_component({**entry, "filename": "garbage.gguf",
                               "file_path": "diffusion_models/garbage.gguf"},
                              root=root / "models")
    check("㉓ 垃圾 GGUF 的加载计划 ⇒ problems 非空（阻断 ✓）",
          bool(gplan.problems) and not gplan.verified, (gplan.problems, gplan.verified))


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_type_table()
        case_roundtrip(root)
        case_v1(root)
        case_broken(root)
        case_wiring(root)
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"\n      ↳ {detail}"))
    # ⚠️ 汇总行**必须是 `SUMMARY: n/m passed`** ✓ —— `run_all.py` 按这个前缀收敛项数 ✓✗
    #    （写成「n/m 项通过」会在总表里显示成**空摘要** ✓✗，项数汇总就少了这一套 ✓）。
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
