"""S? 自检：Skill 注入台账（``services/skill_usage.py``）＋ 超预算裁剪接线（缺口 10）。

判据集中在四件事：

1. **记的是事实**：注入了哪些、各多少字符、哪些被预算裁掉、跑次最后成败 —— 这些都是可观测的；
   「模型有没有照它做」**没有观测通路**，所以这里断言 ``okRate`` 的措辞与语义，
   不许被悄悄改名成「命中率」。
2. **同 run 幂等**：``load_agent_skills`` 在同一跑次被调用两次（``build_agent_config`` 与
   ``default_generate``）时，不能把一次注入记成两笔样本。
3. **样本不足不说话**：``< MIN_SAMPLES`` 的 skill 不产生惩罚、也不进 ``ok_rates``。
4. **默认行为零变化**：预算装得下时注入顺序与内容一个字节都不变；只有**装不下**才动排序，
   且降级只针对「样本够 + 成功率明显低」的条目。

运行::

    ./.venv/Scripts/python.exe tests/skill_usage_test.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="skusage_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(__file__).resolve().parents[1]
_WORK = Path(tempfile.mkdtemp(prefix="skusage_work_"))
os.environ["SKILLS_DIR"] = str(_WORK / "skills")
os.environ["SKILL_USAGE_DB"] = str(_WORK / "usage.sqlite")
os.environ.pop("AGENT_SKILL_BUDGET", None)

from app.agent import skills as sk  # noqa: E402
from app.services import skill_parser as sp  # noqa: E402
from app.services import skill_usage as su  # noqa: E402
from app.routers import skills as skr  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_SKILL = """---
name: {name}
description: {name} 的说明
agents: [storyboard_breaker]
priority: {priority}
---

# {name} 正文

{body}
"""


def _make_skill(skill_id: str, priority: int, filler: int) -> str:
    directory = _WORK / "skills" / skill_id
    directory.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"第 {i} 条：镜头切分规则。" for i in range(filler))
    (directory / "SKILL.md").write_text(
        _SKILL.format(name=skill_id, priority=priority, body=body), encoding="utf-8")
    return skill_id


def _reset_db() -> None:
    """清库。Windows 上刚关闭的连接可能还有 -journal 残留 ⇒ 逐个静默，不因清理失败判红。"""
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            Path(os.environ["SKILL_USAGE_DB"] + suffix).unlink(missing_ok=True)
        except OSError:
            pass


def _seed(skill_id: str, outcomes: list[str], *, agent_type: str = "storyboard_breaker",
          prefix: str = "seed") -> None:
    for index, outcome in enumerate(outcomes):
        su.record([{"id": skill_id, "chars": 100}], agent_type=agent_type,
                  run_id=f"{prefix}-{skill_id}-{index}", outcome=outcome)


def _called_names(path: Path, func: str) -> set[str]:
    """AST 取某函数体内被调用的名字（不做运行期推断，专抓"接线忘了写"）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            names: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    target = inner.func
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        names.add(target.attr)
            return names
    return set()


def case_db_and_record() -> None:
    check("路径: SKILL_USAGE_DB 覆盖生效",
          su.db_path() == Path(os.environ["SKILL_USAGE_DB"]), su.db_path().as_posix())
    _reset_db()
    check("记录: 空输入 -> recorded 0（不建无意义的行）",
          su.record([], agent_type="storyboard_breaker")["recorded"] == 0)
    result = su.record(
        [{"id": "alpha", "chars": 120}, {"id": "beta", "chars": 80, "dropped": True},
         "gamma"],
        agent_type="storyboard_breaker", run_id="run-1", outcome=None)
    check("记录: 三条都写入（含纯字符串写法）", result == {"ok": True, "recorded": 3}, result)

    items = {i["skillId"]: i for i in su.summary()}
    check("汇总: dropped 单列（beta 记 1 次被裁、injected 0）",
          items["beta"]["dropped"] == 1 and items["beta"]["injected"] == 0, items.get("beta"))
    check("汇总: 未结算 -> samples 0、okRate None（**不是 0**）",
          items["alpha"]["samples"] == 0 and items["alpha"]["okRate"] is None, items.get("alpha"))
    check("汇总: 未达样本门槛 -> judged False（还没有结论）",
          items["alpha"]["judged"] is False)


def case_idempotent() -> None:
    _reset_db()
    for _ in range(3):
        su.record([{"id": "alpha", "chars": 10}], agent_type="storyboard_breaker",
                  run_id="same-run", outcome="ok")
    item = {i["skillId"]: i for i in su.summary()}["alpha"]
    check("幂等: 同一 run 重复记 3 次 -> 只有 1 条样本",
          item["samples"] == 1 and item["ok"] == 1, item)
    su.record([{"id": "alpha", "chars": 10}], agent_type="storyboard_breaker",
              run_id="same-run", outcome="failed")
    item = {i["skillId"]: i for i in su.summary()}["alpha"]
    check("幂等: 同 run 换结论 -> 覆盖（不是并列两条）",
          item["samples"] == 1 and item["failed"] == 1, item)


def case_settle() -> None:
    _reset_db()
    su.record([{"id": "alpha", "chars": 10}, {"id": "beta", "chars": 10}],
              agent_type="storyboard_breaker", run_id="run-s")
    check("结算: 回填 2 行", su.settle("run-s", "ok") == 2, su.settle("run-s", "ok"))
    item = {i["skillId"]: i for i in su.summary()}["alpha"]
    check("结算: 结论写进去了", item["ok"] == 1 and item["okRate"] == 1.0, item)
    check("结算: 未知跑次 -> 0（不是错误）", su.settle("nope", "ok") == 0)
    check("结算: 认不出的结果字面 -> 0（不瞎归类）", su.settle("run-s", "什么") == 0)
    check("结算: success/done 等别名归一到 ok",
          su._norm_outcome("SUCCESS") == "ok" and su._norm_outcome("error") == "failed")


def case_threshold_and_penalty() -> None:
    _reset_db()
    _seed("alpha", ["failed"] * (su.MIN_SAMPLES - 1))
    check("门槛: 差一次样本 -> 不进 ok_rates（没有基线不说话）",
          "alpha" not in su.ok_rates(), su.ok_rates())
    check("门槛: 差一次样本 -> 惩罚 0", su.penalty("storyboard_breaker", "alpha") == (0, ""))

    _seed("alpha", ["failed"], prefix="nth")
    rates = su.ok_rates()
    check("门槛: 达到 MIN_SAMPLES -> 有结论且 okRate=0",
          rates.get("alpha") == 0.0, rates)
    cut, why = su.penalty("storyboard_breaker", "alpha")
    check("惩罚: 成功率明显低 -> 降级并给出理由",
          cut == su.PRIORITY_PENALTY and "成功率" in why, (cut, why))

    _seed("beta", ["ok"] * su.MIN_SAMPLES)
    check("惩罚: 成功率高 -> 不降级", su.penalty("storyboard_breaker", "beta") == (0, ""))
    check("惩罚: 没见过的 skill -> 不降级（不猜）",
          su.penalty("storyboard_breaker", "从未出现") == (0, ""))


def case_load_integration() -> None:
    _make_skill("alpha", priority=1, filler=120)
    _make_skill("beta", priority=2, filler=8)
    alpha_len = len(sp.render_skill(sk.load_skill("alpha")))
    beta_len = len(sp.render_skill(sk.load_skill("beta")))
    check("造景: alpha 正文显著长于 beta（预算才切得开）", alpha_len > beta_len + 100,
          (alpha_len, beta_len))

    _reset_db()
    os.environ["AGENT_SKILL_BUDGET"] = "60000"
    sizes: dict[str, int] = {}
    dropped: list[str] = []
    text = sk.load_agent_skills("storyboard_breaker", None, sizes=sizes, dropped=dropped)
    check("预算够: 两个都注入、顺序按 priority（alpha 在前）",
          isinstance(text, str) and text.index("# alpha 正文") < text.index("# beta 正文"))
    check("预算够: 出参 sizes 与 dropped 如实",
          set(sizes) == {"alpha", "beta"} and not dropped, (sizes, dropped))
    check("预算够: sizes 与 render 长度同口径",
          sizes["alpha"] == alpha_len, (sizes.get("alpha"), alpha_len))

    # 预算刚好装下 alpha、装不下 alpha+beta
    os.environ["AGENT_SKILL_BUDGET"] = str(alpha_len + 5)
    sizes2: dict[str, int] = {}
    dropped2: list[str] = []
    text2 = sk.load_agent_skills("storyboard_breaker", None, sizes=sizes2, dropped=dropped2)
    check("超预算+无统计: 仍按 priority -> alpha 先装下、beta 被裁",
          isinstance(text2, str) and "# alpha 正文" in text2
          and dropped2 == ["beta"] and "（超预算）" in text2, (dropped2, (text2 or "")[:60]))

    _seed("alpha", ["failed"] * su.MIN_SAMPLES)
    sizes3: dict[str, int] = {}
    dropped3: list[str] = []
    text3 = sk.load_agent_skills("storyboard_breaker", None, sizes=sizes3, dropped=dropped3)
    check("超预算+历史成功率低: 让位给 beta（alpha 被裁）",
          isinstance(text3, str) and "# beta 正文" in text3
          and "# alpha 正文" not in text3 and dropped3 == ["alpha"], (dropped3, (text3 or "")[:60]))
    check("超预算 + 有统计: 通知文案仍是「超预算」（理由没变，只是谁被裁变了）",
          "（超预算）" in text3 and f"超出 {alpha_len + 5} 字符预算" in text3,
          (text3 or "")[:80])

    # 预算不够时排序会变，但**装得下时必须逐字不变** —— 用同一个库但把预算放开对拍
    os.environ["AGENT_SKILL_BUDGET"] = "60000"
    check("默认行为: 预算放开后与「有统计数据但装得下」那次逐字一致",
          sk.load_agent_skills("storyboard_breaker") == text, "顺序或内容被统计影响了")

    # 统计库坏了不该让 agent 起不来
    os.environ["SKILL_USAGE_DB"] = str(_WORK / "broken.sqlite")
    Path(os.environ["SKILL_USAGE_DB"]).mkdir(parents=True, exist_ok=True)
    check("健壮性: 统计库打不开 -> ok_rates 退化成空（不抛）", su.ok_rates() == {})
    os.environ["AGENT_SKILL_BUDGET"] = str(alpha_len + 5)
    sizes4: dict[str, int] = {}
    dropped4: list[str] = []
    try:
        broken = sk.load_agent_skills("storyboard_breaker", None,
                                      sizes=sizes4, dropped=dropped4)
        survived = isinstance(broken, str) and broke_note(dropped4)
    except Exception as exc:  # noqa: BLE001
        survived = False
        check("健壮性: 统计库坏时 load_agent_skills 抛了异常", False, str(exc))
    check("健壮性: 统计库坏时 load_agent_skills 照常按 priority 工作", survived, dropped4)
    os.environ["SKILL_USAGE_DB"] = str(_WORK / "usage.sqlite")
    os.environ.pop("AGENT_SKILL_BUDGET", None)


def broke_note(dropped: list[str]) -> bool:
    """统计库坏掉时的期望：退化成纯 priority ⇒ 被裁的仍是 beta。"""
    return dropped == ["beta"]


def case_router() -> None:
    paths = [getattr(route, "path", "") for route in skr.router.routes]
    usage = paths.index("/api/v1/skills/usage")
    capture = paths.index("/api/v1/skills/{skill_id:path}")
    check("端点: /skills/usage 存在于 /{skill_id:path} **之前**（否则被当 skill id）",
          usage < capture, paths)
    check("端点: /skills/usage 先于 /meta 之外的一切通配", usage < capture)


def case_wiring() -> None:
    runtime = _ROOT / "app" / "agent" / "runtime.py"
    tree = ast.parse(runtime.read_text(encoding="utf-8"))
    uses = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "load_agent_skills":
            keywords = {kw.arg for kw in node.keywords}
            if {"sizes", "dropped"} <= keywords:
                uses = True
    check("接线: runtime 的 load_agent_skills 带 sizes/dropped 出参", uses)
    names = _called_names(runtime, "run_agent_with_instructions")
    check("接线: run_agent 里记录 skill 用量", "_record_skills" in names, sorted(names)[:8])
    inner = _called_names(runtime, "_record_skills")
    check("接线: _record_skills 真调 record_skill_usage", "record_skill_usage" in inner, inner)
    rec = _called_names(runtime, "record_skill_usage")
    check("接线: record_skill_usage 落到 skill_usage.record",
          "record" in rec and "log_task_warn" in rec, rec)

    skills_src = (_ROOT / "app" / "agent" / "skills.py").read_text(encoding="utf-8")
    check("接线: 排序只在超预算分支调用 _trim_order",
          "total > budget else list(bindings)" in skills_src, "找不到超预算闸")
    check("接线: 预算够时不做任何统计查询（性能与行为都要稳）",
          skills_src.count("_trim_order(") == 2, skills_src.count("_trim_order("))


def main() -> int:
    case_db_and_record()
    case_idempotent()
    case_settle()
    case_threshold_and_penalty()
    case_load_integration()
    case_router()
    case_wiring()

    failed = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            failed += 1
            print(f"FAIL  {name}  -> {detail}")
    print(f"\nSUMMARY: {len(_RESULTS) - failed}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # Windows 中文控制台是 GBK ⇒ 不重配就崩在第一条含 ⇒/✓ 的判据行上（口径见 check_cli_encoding.py）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
