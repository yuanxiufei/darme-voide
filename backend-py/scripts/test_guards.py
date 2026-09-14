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

用例（``check_memory.py``：基线 1 + 负向 7 ｜ ``check_skill_refs.py``：基线 1 + 负向 3）
  基线  副本未改动          ⇒ 0 致命（防「用例自身把基线弄坏」）
  ① 缺 TOPICS.md ｜② MEMORY.md 超 8k ｜③ 锚点越界 ｜④ 末节未登记锚点
  ⑤ 磁盘日志未登记 ｜⑥ 落点表路径不存在 ｜⑥b 落点表 §小节指针落空
  ⑦ 引用真断链 ｜⑧ ``docs/`` 引用断链（守住 2026-09-12 才补上的 ``docs/`` 前缀）
  ⑨ 示意引用不误报（``e.g.`` 紧邻的路径必须被跳过，否则真信号会被噪声淹没）

刻意不做
  - **不校验「没报致命」之外的输出**：只认退出码与命中文案，避免把措辞变动变成回归。

用法::

    python backend-py/scripts/test_guards.py      # 退出码 1 = 有用例失败（含「夹具失配」）
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEM = ROOT / ".codebuddy" / "memory"
SCRIPTS = ROOT / "backend-py" / "scripts"
GUARD = SCRIPTS / "check_memory.py"
REFS_GUARD = SCRIPTS / "check_skill_refs.py"
SKILLS = ROOT / "skills"

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


def _spawn(guard: Path, extra_env: dict[str, str]):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **extra_env}
    return subprocess.run([sys.executable, str(guard)], capture_output=True, text=True,
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
    finally:
        for key in ("base", "skills_base"):
            target = _state[key]
            if isinstance(target, Path) and target.exists():
                shutil.rmtree(target, ignore_errors=True)

    total = len(MEM_CASES) + len(REF_CASES)
    for item in passed:
        print(f"✓ {item}")
    for item in failed:
        print(f"✗ {item}")
    print(
        f"守卫自检：{len(passed)}/{total} 通过 ｜ check_memory.py 基线+①~⑥（{len(MEM_CASES)} 例）｜ "
        f"check_skill_refs.py 基线+⑦~⑨（{len(REF_CASES)} 例）"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
