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

from . import facade, hooks

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
    #: ⭐ 接线状态（进程态）**只挂在这里**，不进 `capabilities()` —— 那是静态能力表
    #: （`engine_env_docs_test` 拿它跟 `docs/env/*.json` 逐字段对，两次跑必须逐字相同 ✓）。
    wiring = hooks.status()
    data = dict(data, wiring=wiring)
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
    #: ⚠️ 未登记**不是故障**：纯引擎进程（CLI / 单测 / 画布只做装配）本来就没有服务层那一步 ✓。
    #: 但也不能一句"正常"带过 ✓✗ —— 少登记一个源的症状是「建议为空 / 证据核不过 / 拒写」，
    #: 外部读起来跟「本来就没有数据」一模一样 ✓，所以这里必须点明是**降级**而不是没数据 ✓。
    if wiring["allProvided"]:
        #: ⚠️ 别拿 `wiring['sources']` 去查 `registered` ✓✗ —— 反验时抓到过：两处各说各话
        #: 时这里抛 `KeyError`，**诊断命令自己炸了**。诊断口的原则是「状态再烂也只报不炸」✓。
        tags = ", ".join(f"{wiring['registered'][n]}@{n}" for n in wiring["registered"]) or "无"
        lines.append(f"接线     : {len(wiring['sources'])} 个源全部登记 ✓（{tags}）")
    else:
        lines.append(f"接线     : 未登记 {', '.join(wiring['missing'])}"
                     + (f"；已登记 {', '.join(wiring['registered'])}" if wiring["registered"] else "")
                     + "　⇒ 用到它们的口会答「没接入」（**不是**没有数据 ✓✗）")
        lines.append("           纯引擎进程没有服务层登记属常态 ✓；Web 通路由 app/main.py 登记 ✓")
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


def _cmd_feasibility(args: argparse.Namespace) -> int:
    """§5.1／§5.3：**这台机器跑这一档要多少、跑不跑得动** ✓ ⇒ 退出码机器可判 ✓。

    退出码与 :func:`_cmd_readiness` **同口径** ✓✗（0＝能跑 / 1＝跑不了 / 2＝参数错）：

    * ``perfect`` / ``good`` ⇒ **0** ✓（权重装得下 ✓）；
    * ``marginal`` ⇒ **0**，但**一定要看 ``degraded``** ✓（能跑、要联合驻留或被挤到 CPU ⇒ 慢 ✓）；
    * ``too_tight`` ⇒ **1** ✓（权重本身放不下 ⇒ 装不起来，与"慢"是两件事 ✓）。

    ⚠️ 不给 ``--strict`` 时 ``marginal`` 判 0 是**有意的** ✗：它跑得动 ✓，判 1 会让 CI 把
    "这台机器能出图、只是慢"当成"跑不了" ✓✗。要拿"必须宽裕"当门槛 ⇒ ``--strict`` ✓。
    """
    samples: list[float] | None = None
    matched = 0
    if args.measure:
        try:
            samples = [float(part) * 1000.0 for part in str(args.measure).replace("，", ",").split(",")
                       if part.strip()]
        except ValueError as err:
            raise facade.EngineError(f"--measure 只认秒数（逗号分隔，如 `95,102,88`），收到 "
                                     f"{args.measure!r}", code="E_BAD_CONFIG") from err
        if not samples:
            raise facade.EngineError("--measure 给了空清单 ⇒ 要秒数，如 `--measure 95,102`",
                                    code="E_BAD_CONFIG")
        matched = len(samples)
    data = facade.feasibility(args.stage, root=args.root, capacity_gib=args.capacity,
                              policy=args.policy, size=args.size,
                              samples=samples, matched=matched)
    verdict = str(data.get("verdict") or "")
    decision = data.get("decision") or {}
    ok = verdict != "too_tight" if not args.strict else verdict in ("perfect", "good")
    if args.json:
        _emit(data, as_json=True)
        return EXIT_OK if ok else EXIT_NOT_READY
    print(f"阶段 {data['stage']}  ⇒ {verdict}（policy={data.get('policy')}）")
    if data.get("verdictWhy"):
        print(f"        {data['verdictWhy']}")
    if data.get("downgraded"):
        print(f"  ! 降档：{data.get('downgradeWhy')}")
    need, have = data.get("need") or {}, data.get("have") or {}
    print(f"  要  卡上峰值 {need.get('gpuGiB')} GiB（三件合计 {need.get('sumGiB')} GiB，"
          f"装法 {need.get('mode')}）")
    print(f"  有  卡 {have.get('gpuTotalGiB')} GiB（空闲 {have.get('gpuFreeGiB')}）"
          f" / 内存 {have.get('hostFreeGiB')}/{have.get('hostTotalGiB')} GiB")
    if decision:
        print(f"  判  device={decision.get('device')}　residency={decision.get('residency')}"
              f"　degraded={decision.get('degraded')}")
        if decision.get("reason"):
            print(f"        {decision['reason']}")
    size = data.get("size") or {}
    if size:
        detail = size.get("problem") or size.get("note") or ""
        print(f"  尺寸 {size.get('asked')} ⇒ {size.get('snapped') or '不吸附'}"
              f"（步长 {size.get('step')}，{size.get('megapixels')} MP）"
              + (f"　! {detail}" if detail else ""))
    estimate = data.get("estimate") or {}
    run = estimate.get("runSeconds") or {}
    sample_note = f"　样本 {estimate.get('samples')}（闸门 {estimate.get('minSamples')}）"
    if run:
        print(f"  估时 整任务 p50 {run.get('p50')} 秒 / p95 {run.get('p95')} 秒{sample_note}")
    else:
        print(f"  估时 整任务 未知（basis={estimate.get('basis')}）{sample_note}")
    if estimate.get("note"):
        print(f"        {estimate['note']}")
    for text in data.get("advice") or []:
        print(f"  建议 {text}")
    return EXIT_OK if ok else EXIT_NOT_READY


def _cmd_status(args: argparse.Namespace) -> int:
    #: ⭐ 机器档（G1）✓：``--profile`` 只影响 ``budgets`` 那一栏 ✓ —— **不改装载** ✗（要改档得重装 ✓）。
    #: ⚠️ 拼错**当场报用法错**（退出 2 ✓）✗✗：不归一就往下传 ⇒ `EngineRuntime.status` 抛 `ValueError` ⇒
    #: 落到 ``main`` 最后那条兜底 ⇒ 退出 3 且写「E_INTERNAL」✓✗ —— 用户只是把 `l1` 写成了 `low1` ✓。
    key = args.profile
    if key is not None:
        from . import profile as machine_profile  # noqa: PLC0415

        try:
            key = machine_profile.normalize_profile(key)
        except ValueError as err:
            raise facade.EngineError(str(err), code="E_BAD_CONFIG",
                                     hint=f"可用：{sorted(machine_profile.PROFILES)}") from err
    data = facade.status(profile=key)
    if args.json:
        _emit(data, as_json=True)
    else:
        loaded = data.get("loaded") or {}
        print(f"阶段 {data.get('stage')}  运行中={data.get('running')}")
        print("  已装: " + (", ".join(k for k, v in loaded.items() if v) or "无"))
        if data.get("vram"):
            print(f"  显存: {data['vram']}")
        #: 档位与预算两行分开打 ✓✗：``assumed`` 是**假设**、``gpuGiB`` 是**实测** ✓
        #: （混一行读就会把「我以为有 8 GiB」当成真读数 ✓✗）。
        prof = data.get("profile") or {}
        budget = data.get("budgets") or {}
        if prof:
            print(f"  机器档: {prof.get('profile')}（来源 {prof.get('source')}）"
                  f"  配额 {budget.get('hostRatio')}")
        if budget:
            def _gib(value: Any) -> str:
                return "未知" if value is None else f"{value} GiB"

            print(f"  卡: {_gib(budget.get('gpuGiB'))}（可用 {_gib(budget.get('gpuFreeGiB'))}）"
                  f"  内存: {_gib(budget.get('hostGiB'))}（可用 {_gib(budget.get('hostFreeGiB'))}）"
                  f"  配额 {_gib(budget.get('hostQuotaGiB'))}"
                  f"　读数 source={budget.get('source')}")
            if budget.get("hint"):
                print(f"  ! {budget['hint']}")
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

    #: ⭐ §5.1／§5.3：**任务级可行性**（能不能跑 + 要多少 + 估时）✓
    #: ⚠️ 名字**不是** ``plan`` ✗✗：`plan` 早就被"装载计划"占了 ✓（退出码语义＝件齐不齐 ✓）——
    #: 把它的退出码改成 verdict 会让既有脚本把"缺件"读成"卡不够" ✓✗。本命令的名字与
    #: ``GET /engine/feasibility``、``facade.feasibility`` **同一个词** ✓（三处同源 ✓）。
    feasible = _child("feasibility", "能不能跑（verdict + 要多少 + 估时 + 尺寸吸附）")
    feasible.add_argument("--stage", default="h3", help="阶段（默认 h3）")
    feasible.add_argument("--root", default=None, help="钉死权重根目录（默认按清单+探测）")
    feasible.add_argument("--capacity", type=float, default=None,
                          help="假装卡只有这么大（GiB；不给就用真读数）")
    feasible.add_argument("--policy", default="auto", help="auto/gpu/cpu/balanced（默认 auto）")
    feasible.add_argument("--size", default=None, help="目标尺寸，如 1024x1024（吸附口径委托画布）")
    feasible.add_argument("--measure", default=None,
                          help="实测整任务秒数（逗号分隔，如 `95,102,88`）⇒ 估时用实测而不是表")
    #: ⚠️ 有意**不**提供 ``--from-ledger`` ✗✗：跑次台账在业务库 ✓，而本模块是**引擎内核**
    #: （`feasibility.py` 纯函数 ✓、零 IO ✓）。要从台账取样本 ⇒ 走**服务层**
    #: `app.services.engine_readiness.task_feasibility(use_ledger=True)` ✓ 或
    #: `GET /engine/feasibility?fromLedger=true` ✓ —— 那两处才许碰库 ✓。
    feasible.add_argument("--strict", action="store_true",
                          help="只认 perfect/good（marginal 也判 1；用在「必须宽裕」的闸门上）")
    feasible.set_defaults(func=_cmd_feasibility)

    inspect = _child("inspect", "单文件核对")
    inspect.add_argument("path")
    inspect.set_defaults(func=_cmd_inspect)

    manifest = _child("manifest", "权重清单条目（来源/哈希/格式/许可）")
    manifest.add_argument("paths", nargs="+")
    manifest.add_argument("--full-hash", action="store_true", help="全量 sha256（慢）")
    manifest.add_argument("--out", default=None, help="写到文件而不是 stdout")
    manifest.set_defaults(func=_cmd_manifest)

    status = _child("status", "运行时事实（含机器档与预算）")
    #: ⚠️ ``--profile`` 只换**预算那几行** ✓，不装载 ✗ —— 与 ``GET /engine/runtime?profile=`` 同一份事实 ✓
    status.add_argument("--profile", default=None,
                        help="机器档 auto/gpu/l1/l2/cpu（只影响 budgets，不装载）")
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
