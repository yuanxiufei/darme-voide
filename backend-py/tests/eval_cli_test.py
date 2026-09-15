"""S7 自检：基准资产搬迁 + 评测 CLI（``evaluation/cli.ts`` 64 行）。

两件事一起锁：

1. **4 个 case JSON 已搬到 ``<项目根>/benchmarks``**（``catalog.benchmarks_dir()`` 的新默认值）
   —— 这是「删 ``backend/``」的前置条件，删完不能再去读 ``backend/benchmarks``；
2. **CLI 的四个分支**：用法错误 / 未知 kind / evaluate / optimize（含 ``--iterations`` 解析），
   全部用桩驱动，**不碰真 LLM、不写库**。

运行::

    ./.venv/Scripts/python.exe tests/eval_cli_test.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="cli_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ Windows 控制台默认 GBK：检查名里带 emoji 时**打印阶段**会 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.config import PROJECT_ROOT  # noqa: E402
from app.services.agent_prompts import get_default_instructions  # noqa: E402
from app.agent.evaluation import catalog as cat  # noqa: E402
from app.agent.evaluation import cli  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def run_cli(*argv: str) -> tuple[int, str, str]:
    """跑 CLI，捕获 stdout/stderr（返回 退出码/标准输出/标准错误）。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def main() -> int:  # noqa: C901
    # ================= 基准目录搬迁 =================
    os.environ.pop("BENCHMARKS_DIR", None)
    check("基准: 默认目录已指向 `<项目根>/benchmarks`（**不再依赖 backend/**）",
          cat.benchmarks_dir() == PROJECT_ROOT / "benchmarks"
          and not str(cat.benchmarks_dir()).replace("\\", "/").endswith("backend/benchmarks"),
          cat.benchmarks_dir())
    check("基准: 该目录真实存在且与 backend/ 无关", cat.benchmarks_dir().is_dir())

    files = sorted(p.name for p in cat.benchmarks_dir().iterdir() if p.suffix == ".json")
    check("基准: 4 个 case JSON 就位", files == [
        "extractor.json", "script-rewriter.json", "storyboard-breaker.json",
        "voice-assigner.json"], files)
    # ⚠️ 只在 Node 侧还在时比对（**删 `backend/` 之后这条自动跳过**，不能再红）
    legacy_dir = PROJECT_ROOT / "backend" / "benchmarks"
    if legacy_dir.is_dir():
        check("基准: 与 Node 侧原文件**逐字节一致**（删 backend/ 不丢资产）",
              all((PROJECT_ROOT / "benchmarks" / name).read_bytes()
                  == (legacy_dir / name).read_bytes() for name in files))
    else:
        print("[skip] backend/ 已删除：跳过「与 Node 侧逐字节一致」比对")

    cases = cat.list_benchmark_cases()
    # ⚠️ 这四条以**真实数据**为准（我先前按文件名猜的 id/kind 是错的）
    check("基准: 扫描出 4 条，kind→agentType 映射正确",
          sorted((c["id"], c["kind"], c["agentType"]) for c in cases) == sorted([
              ("ext-case-001", "extractor", "extractor"),
              ("script-rewriter-001", "script_rewriter", "script_rewriter"),
              ("sb-case-001", "storyboard", "storyboard_breaker"),
              ("voice-assigner-001", "voice_assigner", "voice_assigner"),
          ]), cases)
    loaded = cat.load_case_by_id("ext-case-001")
    check("基准: 按 id 能加载**完整** case（含 statement/rubric）",
          isinstance(loaded, dict) and "statement" in loaded and "rubric" in loaded
          and loaded["kind"] == "extractor", list((loaded or {}).keys()))

    with tempfile.TemporaryDirectory() as empty_dir:
        os.environ["BENCHMARKS_DIR"] = empty_dir
        check("基准: `BENCHMARKS_DIR` 覆盖仍然优先（测试与迁移期都要用）",
              cat.benchmarks_dir() == Path(empty_dir) and cat.list_benchmark_cases() == [])
        os.environ.pop("BENCHMARKS_DIR", None)

    # ================= CLI：用法与分支 =================
    code, out, _err = run_cli()
    check("CLI: 无参数 -> 打印用法 + 退出码 1",
          code == 1 and out.startswith("用法：") and "evaluate benchmarks/" in out, out[:40])
    code, out, _err = run_cli("nonsense", "benchmarks/extractor.json")
    check("CLI: 未知子命令 -> 用法 + 1", code == 1 and out.startswith("用法："))
    code, out, _err = run_cli("evaluate")
    check("CLI: 缺 case 路径 -> 用法 + 1", code == 1 and out.startswith("用法："))

    with tempfile.TemporaryDirectory() as tmp:
        bad_case = Path(tmp) / "bad.json"
        bad_case.write_text(json.dumps({"id": "x", "kind": "nope"}), encoding="utf-8")
        code, _out, err = run_cli("evaluate", str(bad_case))
        check("CLI: 未知 kind -> stderr 提示 + 1（**在调评测前**拦掉）",
              code == 1 and err.strip() == "未知 case kind: nope", err)

    check("CLI: case 路径 **cwd 优先**（存在的相对路径原样命中）",
          cli.resolve_case_path("tests/eval_cli_test.py")
          == Path.cwd() / "tests" / "eval_cli_test.py")
    check("CLI: cwd 里没有时**回退项目根** ⇒ 从 `backend-py/` 也能用 `benchmarks/x.json`",
          cli.resolve_case_path("benchmarks/extractor.json")
          == PROJECT_ROOT / "benchmarks" / "extractor.json"
          and cli.resolve_case_path("benchmarks/extractor.json").exists())

    # ================= CLI：evaluate（桩） =================
    seen: dict[str, object] = {}
    report = {"total": 82, "dimensions": [
        {"name": "角色完整性", "score": 40, "max": 50, "detail": "命中 3/3"},
        {"name": "提示词长度", "score": 42, "max": 50, "detail": "均 ≥20 字"},
    ]}

    async def fake_evaluate(_conn, case_def, instructions):
        seen["caseId"] = case_def["id"]
        seen["instructions"] = instructions
        return report

    cli.evaluate_case = fake_evaluate  # type: ignore[assignment]
    code, out, err = run_cli("evaluate", "benchmarks/extractor.json")
    check("CLI evaluate: 打印「评测 <agent>（Reference 提示词）case=<id>」",
          "评测 extractor（Reference 提示词）case=ext-case-001" in out, out[:80])
    check("CLI evaluate: 报告打印格式（总分 x/100 + 每维度一行）",
          "总分 82/100" in out and "  角色完整性: 40/50  命中 3/3" in out
          and "  提示词长度: 42/50  均 ≥20 字" in out, out)
    check("CLI evaluate: 用的是 Reference 默认提示词（不是 DB 里的 agent_configs）",
          seen.get("instructions") == get_default_instructions("extractor"))
    check("CLI evaluate: 成功退出码 0 且无 stderr", code == 0 and err == "", (code, err))

    # ================= CLI：optimize（桩） =================
    seen.clear()

    async def fake_optimize(_conn, agent_type, case_def, options):
        seen["agentType"] = agent_type
        seen["caseId"] = case_def["id"]
        seen["options"] = options
        return {"agentType": agent_type, "iterations": options["iterations"], "entries": []}

    cli.optimize_agent_prompt = fake_optimize  # type: ignore[assignment]
    code, _out, _err = run_cli("optimize", "benchmarks/extractor.json")
    check("CLI optimize: 默认 3 次迭代 + 传 agentType/case",
          code == 0 and seen.get("options") == {"iterations": 3}
          and seen.get("agentType") == "extractor", seen)
    run_cli("optimize", "benchmarks/extractor.json", "--iterations", "5")
    check("CLI optimize: `--iterations 5` 生效", seen.get("options") == {"iterations": 5}, seen)
    code, _out, err = run_cli("optimize", "benchmarks/extractor.json", "--iterations", "abc")
    check("CLI optimize: `--iterations abc` **回退默认 3 并告警**（不制造 NaN 静默 0 次）",
          seen.get("options") == {"iterations": 3} and "取值非法" in err and code == 0,
          (seen.get("options"), err))
    run_cli("optimize", "benchmarks/extractor.json", "--iterations")
    check("CLI optimize: `--iterations` 后面没值 -> 默认 3",
          seen.get("options") == {"iterations": 3}, seen)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("provider 未配置")

    cli.optimize_agent_prompt = boom  # type: ignore[assignment]
    code, _out, err = run_cli("optimize", "benchmarks/extractor.json")
    check("CLI: 运行异常 -> stderr `评测失败：…` + 退出码 1（对应 TS 顶层 catch）",
          code == 1 and err.startswith("评测失败：") and "provider 未配置" in err, err)

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
