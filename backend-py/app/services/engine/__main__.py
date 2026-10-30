r"""引擎的**独立命令行**（WBS-1.5：CLI 与 SDK 同一套能力，不经过 HTTP）。

不依赖后端进程、不依赖前端、不 import torch 也能回答（能力 / 计划 / 就绪 / 核对全部只读文件头）。

用法（在 `backend-py/` 下）::

    python -m app.services.engine doctor          # 能做什么、缺什么、哪些没验证
    python -m app.services.engine plan            # 逐组件装载计划 + 显存排班
    python -m app.services.engine readiness       # 能不能装（阻断项 / 未查项分开）
    python -m app.services.engine inspect <权重>  # 单文件核对（结构 / 量化族 / 还原计划）
    python -m app.services.engine manifest <权重> [--full-hash] [--out m.json]
    python -m app.services.engine status          # 运行时事实（已装什么、显存）

退出码（**机器可判**，WBS-1.2）：0＝成功且结论为「通过」；1＝结论为「不通过」；
2＝用法错；3＝内部错误。`--json` 一律输出机器可读结构。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import facade

EXIT_OK = 0
EXIT_NOT_READY = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3


def _emit(payload: dict[str, Any], *, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif text:
        print(text)


def _cmd_doctor(args: argparse.Namespace) -> int:
    data = facade.capabilities()
    backend = data["backend"]
    lines = [
        f"engine   : {data['engine']}  api={data['apiVersion']}  schema={data['schemaVersion']}",
        f"python   : {data['python']}",
        f"stages   : {', '.join(data['stages'])}",
        f"backend  : {'可用' if backend['available'] else '不可用 —— ' + backend['reason']}",
        f"必需缺失 : {backend['missingRequired'] or '无'}",
        f"可选缺失 : {backend['optionalMissing'] or '无'}",
        f"清单     : {data['catalog']['models']} 条  {data['catalog']['path']}",
        f"量化族   : 支持布局 {', '.join(data['quant']['supportedLayouts'])}；"
        f"已实现的族 {', '.join(data['quant'].get('supportedFormats') or []) or '无'}；"
        f"未实现 {', '.join(data['quant']['unsupportedFamilies'] + data['quant']['unsupportedFormats'])}",
        f"已验证   : {', '.join(data['operations'])}",
        f"未验收   : {', '.join(data['unverified'])}",
        "边界     : " + "；".join(data["limits"]),
    ]
    #: ⭐ 环境既知缺口（E0.1）：**空数组不等于已验收**，只是「没有已知前提缺失」
    gaps = data.get("knownGaps") or []
    lines.append("既知缺口 : " + ("无 ✓（≠ 已验收）" if not gaps
                                  else f"{len(gaps)} 条"))
    for gap in gaps:
        lines.append(f"  - [{gap.get('severity')}] {gap.get('id')}：{gap.get('summary')}"
                     f"　⇒ {gap.get('fix')}")
    _emit(data, as_json=args.json, text="\n".join(lines))
    return EXIT_OK


def _cmd_plan(args: argparse.Namespace) -> int:
    data = facade.plan(args.stage, root=args.root, capacity_gib=args.capacity)
    if args.json:
        _emit(data, as_json=True)
    else:
        print(f"阶段 {data['stage']}  就绪={data['ready']}  权重目录 {data['modelsDir']}")
        for item in data["components"]:
            mark = "在" if item.get("present") else "缺"
            print(f"  [{mark}] {item['key']:<22} {float(item.get('gib') or 0):7.2f} GiB "
                  f"{item.get('loadMode') or ''}　{item.get('quantScheme') or ''}")
            for problem in item.get("problems") or []:
                print(f"        ! {problem}")
        residency = data.get("residency") or {}
        if residency:
            print(f"  显存排班: 峰值 {residency.get('peakResidentGiB')} GiB / 容量 "
                  f"{residency.get('capacityGiB')} GiB ⇒ "
                  f"{'装得下' if residency.get('fits') else '装不下'}")
            print(f"  {residency.get('strategy') or ''}")
    return EXIT_OK if data.get("ready") else EXIT_NOT_READY


def _cmd_readiness(args: argparse.Namespace) -> int:
    data = facade.readiness(args.stage, root=args.root, capacity_gib=args.capacity)
    if args.json:
        _emit(data, as_json=True)
    else:
        print(f"阶段 {data['stage']}  可装={'是' if data['ready'] else '否'}")
        for text in data["blockers"]:
            print(f"  阻断  {text}")
        for text in data["unchecked"]:
            print(f"  未查  {text}（不等于通过）")
        weights = data["weights"]
        print(f"  权重  {weights['requiredWeightsGiB']} GiB  目录 {weights['modelsDir']}")
        vram = data.get("vram") or {}
        if vram:
            print(f"  显存  估 {vram.get('estimatedGiB')} GiB（含安全系数 {vram.get('safety')}）"
                  f" / 容量 {vram.get('capacityGiB')} GiB ⇒ "
                  f"{'fits' if vram.get('fits') else '溢出'}"
                  f"　余量 {vram.get('headroomGiB')} GiB")
    return EXIT_OK if data["ready"] else EXIT_NOT_READY


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        data = facade.inspect_weights(args.path)
    except facade.EngineError as err:
        #: 「未实现的量化族」是**结论**不是意外 ⇒ 报告照打，只是把 ok 翻成 False
        report = (err.detail or {}).get("inspect")
        if err.code != "E_UNSUPPORTED_QUANT" or not report:
            raise
        report = dict(report, ok=False, verdict=err.message)
        if args.json:
            _emit(report, as_json=True)
        else:
            print(f"{report['path']}")
            print(f"  {report['gib']} GiB  张量 {report['tensorCount']}  ⇒ {err.message}")
            for spec in report.get("formats") or []:
                print(f"  文件自述: { {k: v for k, v in spec.items() if k != 'tensor'} }")
            print(f"  ⇒ {err.hint}")
        return EXIT_NOT_READY
    if args.json:
        _emit(data, as_json=True)
    else:
        print(f"{data['path']}")
        print(f"  {data['gib']} GiB  张量 {data['tensorCount']}  头 {data['headerBytes']} 字节")
        print(f"  dtype: " + ", ".join(f"{k}×{v}" for k, v in
                                       sorted(data["dtypeCounts"].items(), key=lambda iv: -iv[1])))
        quant = data["quant"]
        print(f"  量化: 权重 {quant['weights']} 个  可自动还原={quant['supported']}")
        if quant["layouts"]:
            print("  布局: " + ", ".join(f"{k}×{v}" for k, v in
                                        sorted(quant["layouts"].items(), key=lambda iv: -iv[1])))
        for spec in data["formats"]:
            shown = {k: v for k, v in spec.items() if k != "tensor"}
            print(f"  自述: {shown}")
        for problem in data["problems"]:
            print(f"  ! {problem}")
    return EXIT_OK


def _cmd_manifest(args: argparse.Namespace) -> int:
    records = [facade.manifest_of(Path(item), full_hash=args.full_hash) for item in args.paths]
    payload: dict[str, Any] = {"schemaVersion": facade.SCHEMA_VERSION,
                               "apiVersion": facade.API_VERSION,
                               "hashScope": "full" if args.full_hash else "header+size",
                               "models": records}
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写 {target}（{len(records)} 条）")
        return EXIT_OK
    _emit(payload, as_json=True)
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    data = facade.status()
    if args.json:
        _emit(data, as_json=True)
    else:
        loaded = data.get("loaded") or {}
        print(f"阶段 {data.get('stage')}  运行中={data.get('running')}")
        print("  已装: " + (", ".join(k for k, v in loaded.items() if v) or "无"))
        if data.get("vram"):
            print(f"  显存: {data['vram']}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.services.engine",
                                     description="darme 引擎独立命令行（不依赖 HTTP）")
    parser.add_argument("--json", action="store_true", default=False,
                        help="输出机器可读 JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    def _child(name: str, help_text: str) -> argparse.ArgumentParser:
        """每个子命令**也认** `--json`（`engine --json doctor` 与 `engine doctor --json` 都对）。

        ⚠️ `default=SUPPRESS` 是关键 ✗：不设它，子命令的 `False` 会**覆盖**父命令已解析的 `True`
        ⇒ `engine --json doctor` 静默变回人读输出（脚本按 JSON 解析直接炸 ✓✗）。
        """
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                           help="输出机器可读 JSON")
        return child

    doctor = _child("doctor", "能力声明 + 环境（缺什么、哪些没验证）")
    doctor.set_defaults(func=_cmd_doctor)

    for name, func, help_text in (("plan", _cmd_plan, "装载计划"),
                                  ("readiness", _cmd_readiness, "能不能装")):
        item = _child(name, help_text)
        item.add_argument("--stage", default="h3", help="阶段（默认 h3）")
        item.add_argument("--root", default=None, help="钉死权重根目录（默认按清单+探测）")
        item.add_argument("--capacity", type=float, default=None, help="显存容量 GiB")
        item.set_defaults(func=func)

    inspect = _child("inspect", "单文件核对")
    inspect.add_argument("path")
    inspect.set_defaults(func=_cmd_inspect)

    manifest = _child("manifest", "权重清单条目（来源/哈希/格式/许可）")
    manifest.add_argument("paths", nargs="+")
    manifest.add_argument("--full-hash", action="store_true", help="全量 sha256（慢）")
    manifest.add_argument("--out", default=None, help="写到文件而不是 stdout")
    manifest.set_defaults(func=_cmd_manifest)

    status = _child("status", "运行时事实")
    status.set_defaults(func=_cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except facade.EngineError as err:
        payload = err.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[{err.code}] {err.message}", file=sys.stderr)
            if err.hint:
                print(f"  ⇒ {err.hint}", file=sys.stderr)
        #: 用法/参数错 ⇒ 2；其余（坏权重、未实现族、引擎不可用）都是**结论** ⇒ 1
        return EXIT_USAGE if err.code == "E_BAD_CONFIG" else EXIT_NOT_READY
    except Exception as err:  # noqa: BLE001 —— 未预期异常一律 3（可定位，不静默）
        print(f"[E_INTERNAL] {type(err).__name__}: {err}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
