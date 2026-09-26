#!/usr/bin/env python3
"""守卫自检：把「人工负向实证」固化成可重跑用例。

> 本文件是 ``scripts/test-guards.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。
> 用例、判据、文案保持不变；新增守卫用法请照此补用例。

为什么需要
  ``check_memory.py`` 的 ①~⑥ 每一项都是**手工造故障 + 手工还原**验证的（首版 ⑥ 正是
  这样抓出了假警：扫全文撞上日志摘要里的 ``models.vue``）。但手测证据是一次性的，而
  「守卫已失效」这件事**不会自己暴露**：校验逻辑被改坏、正则被放宽、白名单被删，
  基线都会**照样是绿的**（因为真实仓库本来就合规）⇒ **不报错 ≠ 还能报错**。

原理
  把真实 ``.codebuddy/memory/`` 拷进临时目录 → 用 ``MEMORY_DIR`` 把守卫指向副本 →
  造故障 → 断言「退出码 1 + 命中预期文案」。真仓库全程只读，跑完删临时目录。
  仓库根相对路径（``docs/api-contract.md`` 等）仍按真实 ROOT 解析 ⇒ 用例贴近实战。

用例（``check_memory.py``：基线 1 + 负向 7 ｜ ``check_skill_refs.py``：基线 1 + 负向 3 ｜
``check_cli_encoding.py``：基线 1 + 负向 3）
  基线  副本未改动          ⇒ 0 致命（防「用例自身把基线弄坏」）
  ① 缺 TOPICS.md ｜② MEMORY.md 超 8k ｜③ 锚点越界 ｜④ 末节未登记锚点 ｜④b 中间**步骤小节**未登记锚点
  ⑤ 磁盘日志未登记 ｜⑥ 落点表路径不存在 ｜⑥b 落点表 §小节指针落空
  ⑦ 引用真断链 ｜⑧ ``docs/`` 引用断链（守住 2026-09-12 才补上的 ``docs/`` 前缀）
  ⑨ 示意引用不误报（``e.g.`` 紧邻的路径必须被跳过，否则真信号会被噪声淹没）
  ⓪ 含 locale 外字符 + 有 print + 无 ``reconfigure`` ⇒ 报 ✗ ｜⓪b 纯 ASCII ⇒ **不报**（防假警 ✓）
  ⓪c 老写法 ``sys.stdout = io.TextIOWrapper(...)`` **不算数** ✗（丢原 wrapper ⇒ 必须报 ✗）

刻意不做
  - **不校验「没报致命」之外的输出**：只认退出码与命中文案，避免把措辞变动变成回归。

用法::

    python backend-py/app/scripts/test_guards.py      # 退出码 1 = 有用例失败（含「夹具失配」）
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # depth-adjusted-to-app
MEM = ROOT / ".codebuddy" / "memory"
# ⚠️ 兄弟守卫按**自身目录**推导 —— 别再拼 `ROOT / "backend-py" / "scripts"`：
#    2026-09-15 三目录并入 `app/` 后那个路径不存在，本自检会以「找不到文件」整体失败 ✗（实测踩过）。
SCRIPTS = Path(__file__).resolve().parent
GUARD = SCRIPTS / "check_memory.py"
REFS_GUARD = SCRIPTS / "check_skill_refs.py"
#: 2026-09-25 新增的**编码守卫**（拦「GBK 控制台裸跑就崩」✗，见 `check_cli_encoding.py` ✓）：
#: ⚠️ 它与其他两套的**指路方式不同** ✗ —— 那两套靠环境变量指夹具 ✓，它靠 `--root` 参数 ✓
#: （所以 `_spawn` 才要支持透传 `*args` ✓）。
ENC_GUARD = SCRIPTS / "check_cli_encoding.py"
SKILLS = ROOT / "backend-py" / "app" / "skills"  # 2026-09-15 起技能库并入 app/

SECTION_RE = re.compile(r"^(#{1,4}\s|-\s*【)")
LOG_MARKER_RE = re.compile(r"\*\*`(\d{4}-\d{2}-\d{2}\.md)`\*\*")
ANCHOR_RE = re.compile(r"@\d+")

_state: dict[str, Path | None] = {"base": None, "sandbox": None, "skills_base": None}


def _lines_of(text: str) -> list[str]:
    """与 ``check_memory.py`` 同口径：末尾换行不额外算一行。"""
    arr = re.split(r"\r?\n", text)
    if arr and arr[-1] == "":
        arr.pop()
    return arr


def fresh() -> Path:
    """每个用例都从真实记忆层的干净副本开始，互不污染。"""
    if _state["base"] is None:
        _state["base"] = Path(tempfile.mkdtemp(prefix="mem-guard-"))
    sandbox = _state["base"] / "memory"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    for name in sorted(os.listdir(MEM)):
        source = MEM / name
        if source.is_file():
            shutil.copy2(source, sandbox / name)
    _state["sandbox"] = sandbox
    return sandbox


def fresh_skills() -> Path:
    """skills 树的干净副本（拷**整棵**：小夹具会漏掉真实引用形态）。"""
    if _state["skills_base"] is None:
        _state["skills_base"] = Path(tempfile.mkdtemp(prefix="skill-guard-"))
    box = _state["skills_base"] / "skills"
    if box.exists():
        shutil.rmtree(box)
    shutil.copytree(SKILLS, box)
    return box


def _at(name: str) -> Path:
    return _state["sandbox"] / name  # type: ignore[operator]


def _read(name: str) -> str:
    with _at(name).open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write(name: str, text: str) -> None:
    _at(name).write_text(text, encoding="utf-8", newline="")


def _patch(name: str, old: str, new: str) -> None:
    """精确替换；找不到原文说明夹具已过时 ⇒ 响亮失败，不静默跳过。"""
    text = _read(name)
    if old not in text:
        raise AssertionError(f"夹具失配：{name} 内找不到 {old!r}")
    _write(name, text.replace(old, new, 1))


def _blocks(text: str) -> list[dict]:
    """INDEX.md 的日志块 [at, end)，使 ③④ 能在正确的日志块内下手（锚点数字会跨日志重复）。"""
    lines = text.split("\n")
    headers: list[dict] = []
    for index, line in enumerate(lines):
        match = LOG_MARKER_RE.search(line)
        if match:
            headers.append({"log": match.group(1), "at": index})
    for position, header in enumerate(headers):
        header["end"] = headers[position + 1]["at"] if position + 1 < len(headers) else len(lines)
        header["lines"] = lines
    return headers


def _case_missing_topics() -> None:
    os.remove(_at("TOPICS.md"))


def _case_over_budget() -> None:
    _write("MEMORY.md", _read("MEMORY.md") + "x" * 3000)


def _case_anchor_out_of_range() -> None:
    block = _blocks(_read("INDEX.md"))[0]
    lines = list(block["lines"])
    for index in range(block["at"], block["end"]):
        match = ANCHOR_RE.search(lines[index])
        if match:
            lines[index] = lines[index].replace(match.group(0), "@9999", 1)
            _write("INDEX.md", "\n".join(lines))
            return
    raise AssertionError("夹具失配：INDEX.md 首个日志块内没有锚点")


def _case_last_section_unregistered() -> None:
    block = max(_blocks(_read("INDEX.md")), key=lambda item: item["log"])
    lines_of_log = _lines_of(_read(block["log"]))
    last = 0
    for index, line in enumerate(lines_of_log):
        if SECTION_RE.match(line):
            last = index + 1
    if not last:
        raise AssertionError(f"夹具失配：{block['log']} 内没有小节行")
    lines = list(block["lines"])
    pattern = re.compile(rf"@{last}\b")
    for index in range(block["at"], block["end"]):
        if pattern.search(lines[index]):
            lines[index] = pattern.sub("", lines[index])
            _write("INDEX.md", "\n".join(lines))
            return
    raise AssertionError(f"夹具失配：INDEX.md 的 {block['log']} 块内找不到 @{last}")


#: ④b 的**合成夹具**日志名（找不到真实步骤日志时用；见 ``_synthetic_step_log``）。
SYNTH_STEP_LOG = "2099-01-01.md"


def _synthetic_step_log() -> tuple[dict, list[int]]:
    """自己造一篇带 3 个步骤小节的日志，并把它登记进副本的 ``INDEX.md``。

    ⚠️ 2026-09-26 实测踩到：记忆层清理成「只保留最新一天」后，仓里再没有 ``## S7 第 N 步`` 形态的
    日志 ⇒ ④b 直接「夹具失配」整体红 ✗。**用例意图是「抹掉中间某步的锚点」，不该依赖仓库此刻恰好
    存在哪种日志** ✗ —— 故找不到就自己造一篇（判据、文案一字不改 ✓）。
    """
    lines = [
        "# 2099-01-01 守卫夹具（合成）",
        "",
        "## S7 第 1 步：合成第一步",
        "",
        "正文。",
        "",
        "## S7 第 2 步：合成第二步",
        "",
        "正文。",
        "",
        "## S7 第 3 步：合成第三步",
        "",
        "正文。",
    ]
    steps = [index + 1 for index, line in enumerate(lines) if line.startswith("## S7 第")]
    _write(SYNTH_STEP_LOG, "\n".join(lines) + "\n")

    # 块必须落在「## 日志清单」小节内（守卫只在该小节内解析 @N）⇒ 插到下一个 `## ` 之前。
    index_lines = _read("INDEX.md").split("\n")
    list_at = next(i for i, line in enumerate(index_lines) if line.startswith("## 日志清单"))
    next_at = next(i for i in range(list_at + 1, len(index_lines))
                   if index_lines[i].startswith("## "))
    entry = [
        "",
        f"**`{SYNTH_STEP_LOG}`**（合成夹具：④b 步骤小节锚点用例）",
        "三个步骤小节，锚点 " + " / ".join(f"@{num}" for num in steps) + "。",
    ]
    index_lines[next_at:next_at] = entry
    _write("INDEX.md", "\n".join(index_lines))

    for block in _blocks("\n".join(index_lines)):
        if block["log"] == SYNTH_STEP_LOG:
            return block, steps
    raise AssertionError("夹具失配：合成日志没登记进 INDEX.md")


def _case_step_section_unregistered() -> None:
    """抹掉**倒数第二个步骤小节**的锚点（末节已被 ④ 覆盖，这里专测「中间小节被挤掉」）。

    ⚠️ 2026-09-15 真实事故：一条 `re.sub` 把索引里的第 40/41 步与「待办」条目一起改写掉，
    而末节（第 42 步）锚点还在 ⇒ ④ 全绿、**三条跳读入口静默消失** ✗。本用例锁住 ④b。
    """
    # ⚠️ 夹具选择**不能固定取最新那篇**：有的日记只有 1 个步骤小节（当天刚开一篇、只写了 1 步 ✓）
    #    ⇒ 硬要求 ≥2 会让自检**自己失配**（2026-09-16 实测：「夹具失配：2026-09-16.md 内步骤小节
    #    不足 2 个」✗）。改为「按日期**从新到旧**取第一篇含 ≥2 个步骤小节的日志」✓ ——
    #    用例意图不变（抹掉中间某步的锚点），且对「日记粒度」不敏感 ✓。
    # ⚠️ 2026-09-26 再补一层：真实日志**一篇都没有**时（记忆层只留最新一天 ✗）自己造 ✓。
    chosen = None  # (INDEX 里的日志块, 该日志的步骤小节行号列表)
    for candidate in sorted(_blocks(_read("INDEX.md")), key=lambda item: item["log"], reverse=True):
        log_lines = _lines_of(_read(candidate["log"]))
        found = [index + 1 for index, line in enumerate(log_lines)
                 if re.match(r"^##\s*S7 第\s*\d+\s*步", line)]
        if len(found) >= 2:
            chosen = (candidate, found)
            break
    if chosen is None:
        chosen = _synthetic_step_log()
    block, steps = chosen
    target = steps[-2]
    lines = list(block["lines"])
    pattern = re.compile(rf"@{target}\b")
    for index in range(block["at"], block["end"]):
        if pattern.search(lines[index]):
            lines[index] = pattern.sub("", lines[index])
            _write("INDEX.md", "\n".join(lines))
            return
    raise AssertionError(f"夹具失配：INDEX.md 的 {block['log']} 块内找不到 @{target}")


def _case_disk_log_unregistered() -> None:
    _write("2099-12-31.md", "# 探针\n")


def _case_ref_path_missing() -> None:
    _patch("INDEX.md", "`docs/api-contract.md`", "`docs/api-contract-missing.md`")


def _case_ref_section_missing() -> None:
    _patch("INDEX.md", "`MEMORY.md §Skill`", "`MEMORY.md §SkillZZZ`")


#: （名称, 期望退出码, 命中文案, 变异）
MEM_CASES: list[tuple[str, int, str, object]] = [
    ("基线（副本未改动）", 0, "", None),
    ("① 缺三层文件（删 TOPICS.md）", 1, "缺少三层文件", _case_missing_topics),
    ("② MEMORY.md 超 8k 预算", 1, "预算 8000", _case_over_budget),
    ("③ 锚点越界（首个日志块改用 @9999）", 1, "锚点越界", _case_anchor_out_of_range),
    ("④ 日志末节未登记锚点（抹掉最新日志末节锚点）", 1, "日志末节未登记锚点",
     _case_last_section_unregistered),
    ("④b 中间步骤小节未登记锚点（抹掉倒数第二个步骤锚点）", 1, "步骤小节未登记锚点",
     _case_step_section_unregistered),
    ("⑤ 磁盘日志未登记（造 2099-12-31.md）", 1, "日志未登记进 INDEX.md",
     _case_disk_log_unregistered),
    ("⑥ 落点表路径不存在", 1, "落点表引用的路径不存在", _case_ref_path_missing),
    ("⑥b 落点表 §小节指针落空", 1, "落点表的小节指针落空", _case_ref_section_missing),
]


def _append(box: Path, relative: str, text: str) -> None:
    with (box / relative).open("a", encoding="utf-8", newline="") as handle:
        handle.write(text)


#: 引用守卫用例：夹具是 skills 树副本，通过 ``SKILL_REFS_DIR`` 指过去；
#: 探针一律**追加到文件末尾**，故不影响该文件既有引用的结论。
REF_CASES: list[tuple[str, int, str, object]] = [
    ("基线（skills 副本未改动）", 0, "", None),
    ("⑦ 真断链：引用不存在的 references 文件", 1, "断链",
     lambda box: _append(box, "prompt-style-library/SKILL.md",
                         "\n探针：`references/__no_such_file__.md`\n")),
    ("⑧ docs 引用断链（验证 docs/ 前缀确实纳入校验）", 1, "断链",
     lambda box: _append(box, "prompt-style-library/SKILL.md",
                         "\n探针：`docs/__no_such_doc__.md`\n")),
    ("⑨ 示意引用不误报（e.g. 紧邻的路径应被跳过）", 0, "",
     lambda box: _append(box, "prompt-style-library/SKILL.md",
                         "\n示意引用（e.g. `references/__no_such_file__.md` 只是举例）\n")),
]


def fresh_enc() -> Path:
    """编码守卫的夹具：一个**假的 backend-py** ✓（只需它扫的两个目录 ✓）。

    ⚠️ 与另两套夹具的取向**相反** ✗：那两套拷**真实资产**（结论依赖真实内容 ✓），
    这一套**必须造空壳** ✓ —— 它扫的是**源码文本**，拷全仓既慢、又会让断言随仓库自身
    改动漂 ✓✗（真实仓库一旦全合规，「还能不能报错」就再也测不出来了 ✓）。
    """
    box = Path(tempfile.mkdtemp(prefix="enc-guard-"))
    (box / "tests").mkdir()
    (box / "app" / "scripts").mkdir(parents=True)
    return box


def _write_enc(box: Path, relative: str, text: str) -> None:
    (box / relative).write_text(text, encoding="utf-8")


#: 编码守卫用例（`check_cli_encoding.py`：基线 1 + 负向 3）。
#: ⚠️ 夹具文件**不必语法完整** ✗（守卫只读源码文本 ✓ 不 import ✗）⇒ 摆得越短越能看清判据 ✓。
#: 三条负向分别守住口径的三条：① locale 外字符（⓪）② 真的 print（⓪b 反向）③ reconfigure（⓪c）✓。
ENC_CASES: list[tuple[str, int, str, object]] = [
    ("基线（假入口已 reconfigure）", 0, "",
     lambda box: _write_enc(box, "tests/fake_test.py",
                            'import sys\n'
                            'sys.stdout.reconfigure(encoding="utf-8", errors="replace")\n'
                            'print("✓")\n')),
    ("⓪ 裸跑会崩：含 locale 外字符 + 有 print + **无** reconfigure", 1, "致命",
     lambda box: _write_enc(box, "tests/fake_test.py", 'print("✓ 判据")\n')),
    ("⓪b 纯 ASCII ⇒ **不报**（口径①不成立 ✓ —— 守住「别变成见 print 就红」✗）", 0, "",
     lambda box: _write_enc(box, "tests/fake_test.py", 'print("plain ascii")\n')),
    ("⓪c 老写法 `sys.stdout = io.TextIOWrapper(...)` **不算数** ✗（丢原 wrapper ✓ ⇒ 必须报 ✗）",
     1, "致命",
     lambda box: _write_enc(box, "app/scripts/fake.py",
                            'import io, sys\n'
                            'sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")\n'
                            'print("✓")\n')),
]


def _spawn(guard: Path, extra_env: dict[str, str], *args: str):
    """``*args`` 供**按目录扫描**的守卫用 ✓（编码守卫靠 ``--root`` 指夹具 ✓，
    记忆 / 引用守卫则是靠环境变量指夹具 ✓ —— 两种指路方式都收在这儿 ✓）。"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **extra_env}
    return subprocess.run([sys.executable, str(guard), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def main() -> int:
    passed: list[str] = []
    failed: list[str] = []
    try:
        for name, expect, needle, mutate in MEM_CASES:
            try:
                sandbox = fresh()
                if mutate:
                    mutate()  # type: ignore[operator]
                result = _spawn(GUARD, {"MEMORY_DIR": str(sandbox)})
                out = (result.stdout or "") + (result.stderr or "")
                if expect == 0:
                    # 两套守卫的零值文案不同：记忆层「致命 0 处」｜引用层「致命断链 0 处」
                    if result.returncode == 0 and re.search(r"致命[^\n]*0 处", out):
                        passed.append(f"{name} → exit 0 / 致命 0")
                    else:
                        failed.append(f"{name} → 期望 exit 0/致命 0，实得 exit {result.returncode}：{out.strip()}")
                elif result.returncode == 1 and needle in out:
                    passed.append(f"{name} → exit 1 / 命中「{needle}」")
                else:
                    failed.append(
                        f"{name} → 期望 exit 1 且含「{needle}」，实得 exit {result.returncode}：{out.strip()}"
                    )
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{name} → {exc}")

        for name, expect, needle, mutate in REF_CASES:
            try:
                box = fresh_skills()
                if mutate:
                    mutate(box)  # type: ignore[operator]
                result = _spawn(REFS_GUARD, {"SKILL_REFS_DIR": str(box)})
                out = (result.stdout or "") + (result.stderr or "")
                if expect == 0:
                    if result.returncode == 0 and re.search(r"致命[^\n]*0 处", out):
                        passed.append(f"{name} → exit 0 / 致命 0")
                    else:
                        failed.append(f"{name} → 期望 exit 0/致命 0，实得 exit {result.returncode}：{out.strip()}")
                elif result.returncode == 1 and needle in out:
                    passed.append(f"{name} → exit 1 / 命中「{needle}」")
                else:
                    failed.append(
                        f"{name} → 期望 exit 1 且含「{needle}」，实得 exit {result.returncode}：{out.strip()}"
                    )
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{name} → {exc}")
        for name, expect, needle, mutate in ENC_CASES:
            box = fresh_enc()
            try:
                if mutate:
                    mutate(box)  # type: ignore[operator]
                result = _spawn(ENC_GUARD, {}, "--root", str(box))
                out = (result.stdout or "") + (result.stderr or "")
                if expect == 0:
                    if result.returncode == 0 and re.search(r"致命[^\n]*0 处", out):
                        passed.append(f"{name} → exit 0 / 致命 0")
                    else:
                        failed.append(f"{name} → 期望 exit 0/致命 0，实得 exit {result.returncode}：{out.strip()}")
                elif result.returncode == 1 and needle in out:
                    passed.append(f"{name} → exit 1 / 命中「{needle}」")
                else:
                    failed.append(
                        f"{name} → 期望 exit 1 且含「{needle}」，实得 exit {result.returncode}：{out.strip()}"
                    )
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{name} → {exc}")
            finally:
                shutil.rmtree(box, ignore_errors=True)  # 夹具是空壳 ✓ 每例一删 ✓
    finally:
        for key in ("base", "skills_base"):
            target = _state[key]
            if isinstance(target, Path) and target.exists():
                shutil.rmtree(target, ignore_errors=True)

    total = len(MEM_CASES) + len(REF_CASES) + len(ENC_CASES)
    for item in passed:
        print(f"✓ {item}")
    for item in failed:
        print(f"✗ {item}")
    print(
        f"守卫自检：{len(passed)}/{total} 通过 ｜ check_memory.py 基线+①~⑥（含 ④b，{len(MEM_CASES)} 例）｜ "
        f"check_skill_refs.py 基线+⑦~⑨（{len(REF_CASES)} 例）｜ "
        f"check_cli_encoding.py 基线+⓪~⓪c（{len(ENC_CASES)} 例）"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
