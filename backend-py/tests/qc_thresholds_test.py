"""技术质检阈值的**覆盖层**守卫（2026-11-06）—— `knowledge-loop.md` §9 最后一条空着的接线点。

为什么要有覆盖层
================
`TECH_QC_THRESHOLDS` 的九个数字决定「黑场多久算缺陷、响度差多少算不合格、帧率低于多少算废」。
它们此前**只有常量**：真实数据证明口径偏了（比如把角色动的微慢动作判成冻帧批量误杀），
除了改源码没有任何办法 —— 而改源码既不能回滚、也留不下「谁按什么凭据改的」。

现在它们走 `engine/tuning.py` 的 `qc` 伪档：白名单 + 区间 + 写前备份 + **证据核验**
（必须指向台账 run 或盘上真实存在的路径）+ history。质检阈值**最需要**证据：
调低 `blackminduration` 就是让烂片过审、调高就是批量误杀。

这套守卫盯五类失效（全都不报错）：
① 覆盖生效了但常量被写坏 ⇒ TS 镜像守卫变成假绿；
② 覆盖文件里塞了 typo 键/越界值 ⇒ 「以为调了其实没调」；
③ 覆盖值没送到 ffmpeg 参数（读了没用 = 白读）；
④ `apply_to_request` 把 `qc.*` 当引擎参数换进请求（伪档越界）；
⑤ 默认值与 `TECH_QC_THRESHOLDS` 漂开（同一套判据两处写法，早晚两套口径）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/qc_thresholds_test.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"
sys.path.insert(0, str(BP))
os.environ.setdefault("PROXY_TO_NODE", "0")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

TMP = Path(tempfile.mkdtemp(prefix="qc-thr-"))
#: 覆盖文件与台账库都必须落在夹具目录：这套会真写覆盖文件（含 `_backups/`）
os.environ["ENGINE_TUNING_JSON"] = (TMP / "tuning.json").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "ledger.sqlite").as_posix()
#: ⚠️ 数据根也必须挪走：``probe_video_technical_spec`` 先过 ``get_absolute_path()``，
#: 它按 **storageRoot** 做防穿越检查 ⇒ 夹具视频若放在系统临时目录会被当场判
#: ``Path traversal blocked`` 早退（于是 ffmpeg 一次都不跑、判据看起来"没接线"✗）。
os.environ["DATA_ROOT"] = TMP.as_posix()

from app.core.config import get_storage_root  # noqa: E402
from app.services import technical_qc  # noqa: E402
from app.services.engine import tuning  # noqa: E402

SRC = (BP / "app" / "services" / "technical_qc.py").read_text(encoding="utf-8")
CONST = technical_qc.TECH_QC_THRESHOLDS
_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write_overrides(overrides: dict[str, dict]) -> None:
    data = {"version": 1, "updatedAt": "", "overrides": overrides, "history": []}
    tuning.path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _entry(value: object, basis: str = "测试", evidence: list[str] | None = None) -> dict:
    return {"value": value, "basis": basis, "evidence": evidence or [], "by": "test",
            "updatedAt": "2026-11-06T00:00:00"}


def main() -> int:
    tuning.path().unlink(missing_ok=True)

    # ==================================================== ① 无文件 ⇒ 一分不变
    values, sources = technical_qc.qc_thresholds()
    check("默认: 没有覆盖文件 ⇒ 逐值与常量相同（能默认开着的前提）",
          values == CONST and set(sources.values()) == {"default"}, sources)

    check("默认: 返回的是副本，改它不动常量（镜像守卫才不会是假绿 ✗）",
          values is not CONST, id(values))

    # ============================================ ② 合法覆盖 ⇒ 生效且只动那一格
    _write_overrides({"qc.blackminduration": _entry(0.5)})
    values, sources = technical_qc.qc_thresholds()
    check("覆盖: qc.blackminduration=0.5 ⇒ 生效、来源如实标 override ✓",
          values["blackMinDuration"] == 0.5
          and sources["blackMinDuration"] == "override:qc.blackminduration", sources)
    check("覆盖: 其余八格仍是 default（只动点名的这一格，不整份替换 ✗）",
          sum(1 for source in sources.values() if source != "default") == 1
          and values["fpsMin"] == CONST["fpsMin"], values)
    check("覆盖: TECH_QC_THRESHOLDS 常量仍逐值等于 TS 镜像值（route_parity 逐值比对）",
          CONST["blackMinDuration"] == 0.25, CONST["blackMinDuration"])

    # ================================== ③ 覆盖值真送到 ffmpeg 参数（读了有用）
    calls: list[tuple[str, list[str]]] = []

    def fake_run(binary: str, args: list[str]) -> tuple[str, str]:
        calls.append((binary, args))
        if binary == "ffprobe":
            return json.dumps({"streams": [{"avg_frame_rate": "24/1", "codec_type": "video"}],
                               "format": {"duration": "5.0"}}), ""
        #: 只给黑场那一次调用造一段真实的 ffmpeg 输出（判据要顺着它走到**人看的文案**）
        if any("blackdetect" in arg for arg in args):
            return "", "black_start:0 black_end:1.2 black_duration:1.2"
        return "", ""

    sample = Path(get_storage_root()) / "qc-thr-test.mp4"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x00" * 16)
    original_run = technical_qc._run
    technical_qc._run = fake_run
    try:
        #: 传**相对** storageRoot 的路径（``get_absolute_path`` 的防穿越边界就在那一层）
        probed = technical_qc.probe_video_technical_spec("qc-thr-test.mp4", expected_duration=5.0)
    finally:
        technical_qc._run = original_run
    check("接线: 夹具视频真被探测（没被防穿越检查早退 ⇒ 下面的参数判据才有意义）",
          "error" not in probed and calls != [], probed.get("error"))

    filters = [arg for binary, args in calls if binary == "ffmpeg"
               for arg in args if "blackdetect" in arg or "freezedetect" in arg]
    check("接线: 覆盖值进了 ffmpeg 的 blackdetect 参数（不是读了没用 ✗）",
          any("blackdetect=d=0.5" in item for item in filters), filters)
    check("接线: 没被覆盖的冻帧仍用常量（d=1，不是被 0.5 一起带偏 ✗）",
          any("freezedetect" in item and "d=1" in item for item in filters), filters)
    #: 回执纪律：人看到的必须是**生效值**（issues 里那句「标准 >X s」是唯一可见处 ✗）
    check("回执: 黑场缺陷文案里写的是覆盖后的 0.5s（不是常量 0.25s ✗）",
          any("标准 >0.5s" in item["message"] for item in probed["issues"]),
          [item["message"] for item in probed["issues"]])

    # ================================================ ④ 坏值/typo ⇒ 退回默认 + 报出
    _write_overrides({"qc.blackminduration": _entry(0.01)})
    values, sources = technical_qc.qc_thresholds()
    check("坏值: 越界（0.01 < 0.05）⇒ 退回默认，不半信半疑地用",
          values["blackMinDuration"] == 0.25 and sources["blackMinDuration"] == "default", values)
    check("坏值: problems() 必须报出来（静默退回＝人以为调了其实没调 ✗）",
          any("qc.blackminduration" in item for item in tuning.problems()), tuning.problems())

    _write_overrides({"qc.blackminduration": _entry("abc")})
    values, _ = technical_qc.qc_thresholds()
    check("坏值: 非数字（\"abc\"）⇒ 退回默认",
          values["blackMinDuration"] == 0.25, values["blackMinDuration"])

    _write_overrides({"qc.fpsminimum": _entry(10)})  # 少一个 n 的 typo
    values, sources = technical_qc.qc_thresholds()
    check("typo: 未登记键（qc.fpsminimum）⇒ 不生效，且 problems 点名（防「改了没反应」）",
          values["fpsMin"] == CONST["fpsMin"]
          and any("未登记的字段" in item for item in tuning.problems()), tuning.problems())

    # =================================================== ⑤ 写入口：证据是硬的
    tuning.path().unlink(missing_ok=True)
    no_evidence = tuning.set_field("qc", "blackminduration", 0.5, basis="凭感觉")
    check("写入口: 缺 evidence ⇒ 拒（没有凭据的调优就是拍脑袋）",
          no_evidence["ok"] is False and "evidence" in no_evidence["why"], no_evidence)

    ghost = tuning.set_field("qc", "blackminduration", 0.5, basis="按统计",
                             evidence=[(TMP / "no-such-file.txt").as_posix()])
    check("写入口: evidence 指向不存在的路径 ⇒ 拒（「看起来像凭据」不算凭据）",
          ghost["ok"] is False and "证据核不过" in ghost["why"], ghost)

    bad_range = tuning.set_field("qc", "blackminduration", 9.0, basis="按统计",
                                 evidence=[sample.as_posix()])
    check("写入口: 越界值直接拒写（区间不是装饰）",
          bad_range["ok"] is False and "非法" in bad_range["why"], bad_range)

    typo_write = tuning.set_field("qc", "blackMinDuration", 0.5, basis="按统计",
                                  evidence=[sample.as_posix()])
    check("写入口: 字段名大小写归一（qc.blackMinDuration 与 qc.blackminduration 是同一格 ✓）",
          typo_write["ok"] is True, typo_write)

    spaced = tuning.set_field("qc", "freezemin duration", 1.0, basis="占位检查",
                              evidence=[sample.as_posix()])
    check("写入口: 带内部空白的键归一后仍不存在 ⇒ 拒（别让空白混出第二个「同名」字段）",
          spaced["ok"] is False and "白名单" in spaced["why"], spaced)

    real = tuning.set_field("qc", "fpsmin", 24, basis="成片按 24fps 出，15 的下限恒不触发",
                            evidence=[sample.as_posix()])
    backups = sorted((tuning.path().parent / "_backups").glob("tuning.*.json"))
    check("写入口: 有真凭据 ⇒ 写入成功 + 写前备份 + history 留痕 ✓",
          real["ok"] is True and backup_dirs_ok(backups)
          and any(item.get("key") == "qc.fpsmin"
                  for item in tuning.load().get("history", [])),
          {"write": real, "backups": [p.name for p in backups]})

    values, sources = technical_qc.qc_thresholds()
    check("写入口: 写进去立刻对该读生效（不是等重启 ✗）",
          values["fpsMin"] == 24 and sources["fpsMin"] == "override:qc.fpsmin", sources)

    report = tuning.report()
    row = next((item for item in report["fields"]
                if item["stage"] == "qc" and item["field"] == "fpsmin"), None)
    check("可见面: report() 把 qc 行与依据一并列出（前端能回答「这个数是谁改的」✓）",
          row is not None and row["value"] == 24 and row["source"] == "override:qc.fpsmin"
          and "24fps" in row["basis"], row)

    cleared = tuning.clear_field("qc", "fpsmin")
    values, sources = technical_qc.qc_thresholds()
    check("回滚: clear_field 后回默认（可撤销才敢让人去调）",
          cleared["ok"] is True and values["fpsMin"] == CONST["fpsMin"]
          and sources["fpsMin"] == "default", sources)

    # ======================================== ⑥ 伪档不许被当成引擎参数换进请求
    request = SimpleNamespace(steps=30)
    updated, notes = tuning.apply_to_request("qc", request)
    check("伪档: apply_to_request(qc) 原样返回 + 零 notes（它不是 GenerationRequest 的档 ✗）",
          updated is request and notes == [], notes)

    _write_overrides({"qc.steps": _entry(4)})
    check("伪档: 就算文件里塞 qc.steps 也不生效（未登记）且 problems 报",
          tuning.resolve("qc", "steps", 30)[1] == "default"
          and any("qc.steps" in item for item in tuning.problems()), tuning.problems())
    tuning.path().unlink(missing_ok=True)

    # ================================================== ⑦ 单一真源 / 源码判据
    qc_specs = {name: spec for (stage, name), spec in tuning.TUNABLES.items() if stage == "qc"}
    check("真源: 登记键集合与 TECH_QC_THRESHOLDS 完全一致（多一个少一个都是两套口径 ✗）",
          {name.lower() for name in CONST} == set(qc_specs),
          {"const": sorted(CONST), "table": sorted(qc_specs)})
    const_lower = {name.lower(): value for name, value in CONST.items()}
    drift = {key: {"table": spec.default, "const": const_lower.get(key)}
             for key, spec in qc_specs.items() if spec.default != const_lower.get(key)}
    check("真源: 默认值逐值相等（TS 镜像常量与覆盖表不许漂开）", drift == {}, drift)

    tree = ast.parse(SRC)
    direct = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(isinstance(tgt, ast.Name) and tgt.id == "thresholds" for tgt in node.targets)
              and isinstance(node.value, ast.Name) and node.value.id == "TECH_QC_THRESHOLDS"]
    check("源码: 探测函数不再直接读常量（AST：`thresholds = TECH_QC_THRESHOLDS` 不许存在 ✗）",
          direct == [], [ast.dump(node)[:60] for node in direct])
    check("源码: 常量仍是字面 dict（route_parity 靠它逐值比对，别改成运行时拼装 ✗）",
          isinstance(next(n for n in tree.body
                          if isinstance(n, ast.AnnAssign)
                          and getattr(n.target, "id", "") == "TECH_QC_THRESHOLDS").value,
                     ast.Dict), None)

    # =============================================================== 结果
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(f"{'PASS' if passed_ok else 'FAIL'}  {name}"
              + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


def backup_dirs_ok(backups: list[Path]) -> bool:
    """写前备份：第二次写之前文件已存在 ⇒ 至少留了一份。"""
    return bool(backups)


if __name__ == "__main__":
    sys.exit(main())
