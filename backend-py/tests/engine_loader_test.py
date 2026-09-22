"""S7 自检：引擎**加载计划**（零依赖 ✓ 2026-09-17）。

这套专盯「**权重装齐之后会不会白跑**」✗ —— 真机上最贵的失败是"加载到一半才发现"：
量化权重**缺 scale**（反量化时才炸 ✗）、层号**有洞**（文件被截断/拼错 ✗）、
以及「能不能同时待在显存里」✗（H3 主权重 19.53 GiB + 文本编码器 ≠ 24 GB 卡 ✗）。

做法：**自己合成权重文件**（含"故意坏的"✓）⇒ 不需要下载任何真实模型 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_loader_test.py
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

from app.services.engine import loader as ld  # noqa: E402
from app.services.engine import safetensors as st  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """合成权重文件（自检自用 ✓ —— 与 inventory 那套同形 ✓）。"""
    header: dict[str, Any] = {}
    offset = 0
    blobs: list[bytes] = []
    for name, (dtype, shape) in tensors.items():
        size = 1
        for dim in shape:
            size *= int(dim)
        size *= st.DTYPE_ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        blobs.append(bytes(size))
        offset += size
    payload = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)


def component(gib: float, role: str, key: str, *, present: bool = True,
              load_mode: str = "sequential") -> ld.LoadPlan:
    """造一个（不落盘的）组件计划 ✓ —— 专门用来验**排班数学** ✓。"""
    return ld.LoadPlan(key=key, name=key, role=role, present=present,
                       bytes=int(gib * 1024 ** 3), loadMode=load_mode,
                       verified=True, tensorCount=10)


# ══════════════════════════════════════════════════════════════════════════
# ① 基础：dtype 分类与分组
# ══════════════════════════════════════════════════════════════════════════
def case_basics() -> None:
    check("① dtype 分类认得 fp8 / int8 / 浮点 / 未知",
          ld.classify_dtype("F8_E4M3") == "fp8" and ld.classify_dtype("I8") == "int8"
          and ld.classify_dtype("BF16") == "float" and ld.classify_dtype("X4") == "unknown",
          [ld.classify_dtype(x) for x in ("F8_E4M3", "I8", "BF16", "X4")])
    groups = ld.prefix_groups(["blocks.0.attn.wq.weight", "blocks.0.attn.wk.weight",
                               "blocks.1.attn.wq.weight", "embed.weight"], depth=2)
    # depth=2 就是「两段」⇒ `blocks.0` ✓（我第一版把期望写成三段 `blocks.0.attn` ✗，是**自检写错**
    # 而不是实现错 ✓ —— depth=2 正好够 `_blocks_of` 用 rpartition 拆出「头 + 层号」✓）
    check("② 前缀分组**保留数字段**（能看出层号，才能查洞 ✓）",
          groups == {"blocks.0": 2, "blocks.1": 1, "embed.weight": 1}, groups)


# ══════════════════════════════════════════════════════════════════════════
# ② plan_component：合成权重文件
# ══════════════════════════════════════════════════════════════════════════
def _quantized(root: Path, *, blocks: int = 3, drop_scale_for: int | None = None,
               block_indices: list[int] | None = None) -> Path:
    """造一份"像量化 DiT"的权重 ✓（可注入两种坏法：缺 scale / 层号有洞 ✓）。"""
    tensors: dict[str, tuple[str, list[int]]] = {"embed.weight": ("BF16", [8, 8])}
    for index in (block_indices if block_indices is not None else list(range(blocks))):
        tensors[f"blocks.{index}.attn.wq.weight"] = ("F8_E4M3", [8, 8])
        if drop_scale_for != index:
            tensors[f"blocks.{index}.attn.wq.scale"] = ("F32", [8, 8])
    # ⚠️ 落点必须与 entry 的 `file_path` **一致**（`component_path` 会拼成
    #    `<root>/diffusion_models/dit.safetensors` ✓）—— 初版写到 `<root>/dit.safetensors`，
    #    于是"文件其实存在"却判成缺失 ✗，自检自己就红 ✓。
    path = root / "diffusion_models" / "dit.safetensors"
    write_safetensors(path, tensors)
    return path


def case_component(root: Path) -> None:
    good = _quantized(root / "good")
    entry = {"key": "dit", "name": "DiT（合成）", "kind": "diffusion_models",
             "filename": "dit.safetensors", "file_path": "diffusion_models/dit.safetensors"}
    plan = ld.plan_component(entry, root=root / "good")
    check("③ 量化方案被认出（fp8 + 配套 scale ⇒ 需反量化 ✓）",
          plan.quantScheme == "fp8" and "fp8" in plan.computeDtype, plan.computeDtype)
    check("④ 层块数与头名被认出来（`blocks` + 3 层 ✓，且**不写死** blocks 这个名字 ✓）",
          plan.blockHead == "blocks" and plan.blockCount == 3, (plan.blockHead, plan.blockCount))
    check("⑤ 好文件：无 problems ✓ + verified ✓",
          plan.problems == [] and plan.verified is True, plan.problems)
    check("⑥ 读盘耗时是**估算**（按假定吞吐算，字段名就写明 ✓）",
          plan.readSecondsEstimate >= 0, plan.readSecondsEstimate)

    # ⚠️ 坏法一：缺 scale ⇒ 必须在**加载前**发现
    _quantized(root / "bad_scale", drop_scale_for=1)
    bad_scale = ld.plan_component(entry, root=root / "bad_scale")
    check("⑦ ⭐ 量化权重**缺 scale** ⇒ 加载前就报（否则反量化时才炸 ✗）",
          any("scale" in item for item in bad_scale.problems), bad_scale.problems)

    # ⚠️ 坏法二：层号有洞 ⇒ 典型是文件被截断/拼错
    _quantized(root / "gap", block_indices=[0, 1, 3])
    bad_gap = ld.plan_component(entry, root=root / "gap")
    check("⑧ ⭐ 层号**不连续**（缺第 2 层）⇒ 报「被截断或拼错」（下载没下完的典型 ✓）",
          any("不连续" in item and "缺 [2]" in item for item in bad_gap.problems),
          bad_gap.problems)

    # 纯 bf16 权重**不要求** scale（不误报 ✓）
    naive = root / "naive"
    write_safetensors(naive / "bf16.safetensors", {"blocks.0.attn.wq.weight": ("BF16", [4, 4]),
                                                   "blocks.1.attn.wq.weight": ("BF16", [4, 4])})
    naive_plan = ld.plan_component({**entry, "file_path": "bf16.safetensors"}, root=naive)
    check("⑨ 整份 bf16 的权重**不会**被要求有 scale（不误报 ✓）",
          naive_plan.quantScheme == "none" and naive_plan.problems == [],
          (naive_plan.quantScheme, naive_plan.problems))
    check("⑨′ ⭐ 纯 bf16 ⇒ `dequantPlan.weights == 0` ✓（反量化计划**不虚报** ✓）",
          naive_plan.dequantPlan["weights"] == 0 and naive_plan.dequantPlan["supported"] is False,
          naive_plan.dequantPlan)

    # ⭐⭐ 2026-09-22：**反量化能不能自动做**要在**体检阶段**就有结论 ✓（只靠形状 ✓ 不需要 torch ✓）
    check("⑨″ ⭐⭐ fp8 + 配套 scale ⇒ 体检报告直接给出**自动反量化可行** ✓ 且标出布局 ✓"
          "（`loader` 与 `quant` 用**同一套判据** ✗ —— 不然会出现「体检说能装、装载时被拒」✓✗）",
          plan.dequantPlan["weights"] == 3 and plan.dequantPlan["paired"] == 3
          and plan.dequantPlan["layouts"] == {"elementwise": 3}
          and plan.dequantPlan["supported"] is True
          and any("自动反量化" in item for item in plan.warnings),
          (plan.dequantPlan, plan.warnings))

    # ⚠️ 块量化的 scale 形状（[2,2] 对不上权重 [8,8] ✓）⇒ **体检就要拦下** ✓（装的时候会中止 ✓）
    blocked = root / "blocked"
    # ⚠️ 落点必须拼成 `<root>/diffusion_models/dit.safetensors` ✗（与 entry 的 `file_path` 一致 ✓）
    #    —— 第一版直接写 `<root>/blocked/dit.safetensors` ⇒ 被当成「文件缺失」⇒ 早退 ⇒
    #    `dequantPlan` 还是空的 `{}` ⇒ `KeyError` ✓✗（**同一个坑这仓已经注释过一次了** ✓ 我又踩 ✓）。
    write_safetensors(blocked / "diffusion_models" / "dit.safetensors",
                      {"blocks.0.attn.wq.weight": ("F8_E4M3", [8, 8]),
                       "blocks.0.attn.wq.scale": ("F32", [2, 2])})
    blocked_plan = ld.plan_component(entry, root=blocked)
    check("⑨‴ ⚠️ 块量化（scale 形状对不上 ✓）⇒ 体检就报 problem 并点名「块量化」✓、"
          "`supported=False` ✓（不等到装载才炸 ✓ —— group size 在随附 json 里 ✓ 本仓不猜 ✗）",
          blocked_plan.dequantPlan["unresolvedCount"] == 1
          and blocked_plan.dequantPlan["supported"] is False
          and any("块量化" in item for item in blocked_plan.problems),
          (blocked_plan.dequantPlan, blocked_plan.problems))

    # GGUF：2026-09-20 起有**真读取器** ✓（详见 engine_gguf_test.py ✓）
    # —— 垃圾 GGUF（GGUF + 全零 ✗）现在**真的去读** ⇒ 结构坏 ⇒ 阻断 ✓（不再是 `gguf-unknown` ✗）
    gguf = root / "gguf"
    (gguf).mkdir(parents=True, exist_ok=True)
    (gguf / "w.gguf").write_bytes(b"GGUF" + b"\x00" * 128)
    gguf_plan = ld.plan_component({**entry, "filename": "w.gguf", "file_path": "w.gguf"}, root=gguf)
    check("⑩ 垃圾 GGUF ⇒ 真读取后结构坏 ⇒ problems 非空 + verified=False（不假装读过 ✓）",
          bool(gguf_plan.problems) and gguf_plan.verified is False,
          (gguf_plan.quantScheme, gguf_plan.problems, gguf_plan.verified))

    # 缺文件：是**结论**不是异常 ✓
    missing = ld.plan_component(entry, root=root / "nope")
    check("⑪ 文件缺失 ⇒ loadMode=`missing`（有结论，不抛错 ✓）",
          missing.loadMode == "missing" and missing.present is False, missing.loadMode)


def case_mode_thresholds() -> None:
    """体积阈值是**纯函数** ⇒ 直接测（不去造 18 GiB 的假文件 ✗）。"""
    gib = 1024 ** 3
    check("⑫ 远小于显存 ⇒ full（整份驻留 ✓）", ld._mode_for(int(4 * gib), 24) == "full")  # noqa: SLF001
    check("⑬ 接近显存（19.53 GiB / 24GB）⇒ sequential（逐层进出 ✓，**不是** oversize ✗）",
          ld._mode_for(int(19.53 * gib), 24) == "sequential", ld._mode_for(int(19.53 * gib), 24))
    check("⑭ 超过显存 ⇒ streaming（必须流式，如实说 ✗）",
          ld._mode_for(int(30 * gib), 24) == "streaming")


# ══════════════════════════════════════════════════════════════════════════
# ③ 显存排班（本轮最有价值的一段 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_residency() -> None:
    parts = [component(8.0, "text_encoders", "te"), component(19.53, "diffusion_models", "dit"),
             component(0.5, "vae", "vae_video")]
    plan = ld.residency_plan(parts, capacity_gib=24)
    check("⑮ 排班顺序按**生命周期**：文本编码器 → DiT → VAE ✓",
          [step["role"] for step in plan["steps"]] == ["text_encoders", "diffusion_models", "vae"],
          [step["role"] for step in plan["steps"]])
    check("⑯ ⭐ 峰值 = **最大那个**（19.53 GiB）而**不是**三者之和 28.03 ✗ —— 用完即退 ✓",
          abs(plan["peakResidentGiB"] - 19.53) < 0.01 and plan["fits"] is True,
          (plan["peakResidentGiB"], plan["fits"]))
    check("⑰ 峰值超过 75% 显存 ⇒ strategy=sequential-residency（提醒要排班 ✓）",
          plan["strategy"] == "sequential-residency", plan["strategy"])
    check("⑱ 每一步都标「本步可释放」（把策略说给调用方，而不是让它猜 ✓）",
          all("可释放" in step["release"] for step in plan["steps"]))

    single = ld.residency_plan([component(30.0, "diffusion_models", "huge")], capacity_gib=24)
    check("⑲ 单件就超显存 ⇒ fits=False（不因为「只有一件」就说没问题 ✗）",
          single["fits"] is False and single["peakResidentGiB"] == 30.0, single["fits"])

    # ⚠️ 这是本轮抓到并修掉的那条：**组件缺失**时绝不能报 fits=True
    absent = ld.residency_plan([component(0.0, "diffusion_models", "absent", present=False)],
                               capacity_gib=24)
    check("⑳ ⭐⭐ 必需组件**缺失** ⇒ fits=False（初版按体积算 ⇒ 全缺时回 true，会让前端读成"
          "「能跑」✗ —— 最坏的一种误导 ✓）",
          absent["fits"] is False and absent["missing"] == ["absent"],
          (absent["fits"], absent["missing"]))


# ══════════════════════════════════════════════════════════════════════════
# ④ plan_stage：接真实清单
# ══════════════════════════════════════════════════════════════════════════
def case_stage(root: Path) -> None:
    empty = ld.plan_stage("h3", root=root / "empty")
    check("㉑ 空目录 + **真实清单** ⇒ ready=False 且逐条列出缺件 ✓（本机现状 ✓）",
          empty["ready"] is False and len(empty["components"]) >= 4
          and empty["residency"]["missing"], empty["components"][0]["key"])
    check("㉒ 缺件时 fits 也是 False ✓（不会让前端误以为「只差下载」就能跑 ✓）",
          empty["residency"]["fits"] is False, empty["residency"])

    filled = root / "filled"
    for item in ld.inv.load_catalog()["models"]:
        if item.get("category") != "video" or not item.get("required"):
            continue
        where = ld.inv.component_path(item, filled)
        assert where is not None
        if where.suffix.lower() == ".gguf":
            where.parent.mkdir(parents=True, exist_ok=True)
            where.write_bytes(b"GGUF" + b"\x00" * 64)
        else:
            write_safetensors(where, {"blocks.0.w.weight": ("BF16", [4, 4]),
                                      "blocks.1.w.weight": ("BF16", [4, 4])})
    stage = ld.plan_stage("h3", root=filled)
    check("㉓ 铺齐必需件（合成的小文件）⇒ ready=True ✓（路径解析真的对 ✓）",
          stage["ready"] is True, [c["problems"] for c in stage["components"] if c["problems"]])
    check("㉔ 小文件时峰值为 0 GiB、fits=True（阈值逻辑两个方向都验到 ✓）",
          stage["residency"]["fits"] is True and stage["residency"]["peakResidentGiB"] < 0.1,
          stage["residency"]["peakResidentGiB"])


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 路由：加载计划要**真能被调用** ✓
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/engine/load-plan", params={"stage": "h3"})
    data = response.json().get("data") or {}
    check("㉕ GET /engine/load-plan 真能调用（200 ✓）并给出逐组件计划",
          response.status_code == 200 and len(data.get("components") or []) >= 4
          and "residency" in data, (response.status_code, len(data.get("components") or [])))
    check("㉖ 本机现状（权重全缺）⇒ ready=False、fits=False、缺件逐条列出 ✓",
          data.get("ready") is False and (data.get("residency") or {}).get("fits") is False
          and bool((data.get("residency") or {}).get("missing")),
          (data.get("ready"), (data.get("residency") or {}).get("missing")))
    check("㉗ 计划里带**排班说明**（告诉我们这顺序是怎么推出来的 ✓，不是黑箱 ✓）",
          "不依赖架构细节" in str((data.get("residency") or {}).get("note")),
          (data.get("residency") or {}).get("note"))
    check("㉘ 未知阶段 ⇒ 400 ✓",
          client.get("/api/v1/engine/load-plan", params={"stage": "nope"}).status_code == 400)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_loader_"))
    case_basics()
    case_component(root)
    case_mode_thresholds()
    case_residency()
    case_stage(root)
    case_api()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
