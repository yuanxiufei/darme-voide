r"""**上机前自检（CLI 薄壳）** ✓ —— 业务在 :mod:`app.services.engine_readiness` ✓。

⚠️ 这个文件**只做两件事** ✗：解析命令行 ✓ 与人读渲染 ✓。逻辑**不在这儿** ✓ ——
它在服务层 ✓，因为（a）路由要用它（`GET /api/v1/production/engine-readiness` ✓）；
（b）`app/scripts/` **不是包**（没有 `__init__.py` ✓）⇒ 放这儿路由就**引用不到** ✓✗
（第一版就犯了这个错 ✓ —— 「只有人手敲命令才能用」= 没人调用的能力 ✓）。

用法::

    ./.venv/Scripts/python.exe app/scripts/h3_readiness.py
    ./.venv/Scripts/python.exe app/scripts/h3_readiness.py --weights <DiT.safetensors>
    ./.venv/Scripts/python.exe app/scripts/h3_readiness.py --weights <DiT.gguf> --tokenizer <词表目录>
    ./.venv/Scripts/python.exe app/scripts/h3_readiness.py --json      # 给自动化用 ✓
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services import engine_readiness  # noqa: E402
from app.services.engine import inventory as inv  # noqa: E402


def render(report: dict[str, Any]) -> None:
    """人读版 ✓（**字段名从返回值读** ✗ —— 第一版按想象写 `plan['vram']['peakGiB']` ✓✗，
    连撞两次 `KeyError` ✓ ⇒ 真字段是 `residency.peakResidentGiB` ✓）。"""
    env = report["environment"]
    print("── ① 环境 ─────────────────────────────────────────")
    print(f"torch：{'可用 ✓' if env['torchAvailable'] else '不可用 ✗ —— ' + env['torchReason']}")
    print(f"必需依赖缺失：{env['missingRequired'] or '无 ✓'}　"
          f"可选依赖缺失：{env['optionalMissing'] or '无 ✓'}")
    print(f"ffmpeg：{env['ffmpegVersion'] or '没装 ✗'}")

    ready = report["weightsReadiness"]
    residency = report["loadPlan"]["residency"]
    print("\n── ② 权重就绪 ─────────────────────────────────────")
    # ⚠️ 三个口径**分开说** ✗（只报一个都会误导 ✓✗）：
    print(f"ready={ready['ready']}　清单全量≈{engine_readiness.planned_gib(ready):.2f} GiB　"
          f"已在盘≈{ready['requiredWeightsGiB']:.2f} GiB　"
          f"还要下≈{engine_readiness.planned_gib(ready, only_missing=True):.2f} GiB　"
          f"缺失 {ready['missingRequired'] or '无 ✓'}")
    print(f"加载计划：fits={residency['fits']}　策略={residency['strategy']}　"
          f"峰值≈{residency['peakResidentGiB']:.2f} GiB / {residency['capacityGiB']:.0f} GiB　"
          f"缺 {residency['missing'] or '无 ✓'}")

    check = report["weightsCheck"]
    print("\n── ③ 真权重预检 ───────────────────────────────────")
    if check is None:
        print("没做 ✓（没给 `--weights` ✓ —— 这一项是**没查** ✗ 不是通过 ✓）")
    elif check.get("error"):
        print(f"读不到：{check['error']} ✗")
    else:
        audit = check.get("audit", {})
        print(f"格式={check['format']}　结构={check['structureOk']}　张量={check['tensorCount']}　"
              f"键名核对={'通过 ✓' if audit.get('ok') else '不通过 ✗'}")
        if audit:
            print(f"变体={audit.get('variant')}　层数={audit.get('depth')}　"
                  f"refiner={audit.get('refinerLayers')}　"
                  f"PDD 头库 video/audio={audit.get('headBanksVideo')}/{audit.get('headBanksAudio')}")
        if check.get("configOk"):
            config = check["trunkConfig"]
            print(f"推导出的结构 ✓：hidden={config.get('hidden')} layers={config.get('layers')} "
                  f"heads={config.get('heads')} modalities={config.get('modalities')} "
                  f"banks={config.get('head_banks')}")
        if check.get("tokenizer"):
            tokenizer = check["tokenizer"]
            print(f"词表：backend={tokenizer['backend']}　name={tokenizer['name']}　"
                  f"vocab={tokenizer['vocabSize']} ✓")
        for problem in check.get("structureProblems") or []:
            print(f"  ⚠️ {problem}")

    print("\n── 结论 ───────────────────────────────────────────")
    print("ready = " + ("True ✓" if report["ready"] else "False ✗"))
    for item in report["blockers"]:
        print(f"  ✗ {item}")
    for item in report["unchecked"]:
        print(f"  ⚠️ {item}")
    for index, step in enumerate(report["nextSteps"], start=1):
        print(f"  {index}. {step}")


def main() -> int:
    parser = argparse.ArgumentParser(description="H3 上机前自检（CLI 薄壳 ✓）")
    parser.add_argument("--weights", help="DiT 权重的**具体文件**（.safetensors / .gguf ✓）")
    parser.add_argument("--tokenizer", help="词表目录或 `tokenizer.json`（可选 ✓）")
    parser.add_argument("--stage", default="h3", help="阶段（默认 h3 ✓）")
    parser.add_argument("--capacity-gib", type=float, default=inv.DEFAULT_CAPACITY_GIB,
                        help=f"显存容量（默认 {inv.DEFAULT_CAPACITY_GIB} GiB = A5000 ✓）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（给自动化用 ✓）")
    args = parser.parse_args()

    report = engine_readiness.collect(weights=args.weights, tokenizer=args.tokenizer,
                                      stage=args.stage, capacity_gib=args.capacity_gib)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        render(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
