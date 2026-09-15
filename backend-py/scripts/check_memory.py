#!/usr/bin/env python3
"""记忆层自检：MEMORY.md 体积预算 + INDEX.md 锚点有效性。

> 本文件是 ``scripts/check-memory.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。
> 判据、文案、退出码保持不变 —— 改动它是「改守卫」，请连 ``test_guards.py`` 一起改。

为什么需要
  ``.codebuddy/memory/`` 的两条不变量全靠人工维护，已反复漂移（第十五轮治理截断、
  第十四/十七轮修索引），故补机器校验：

    ① MEMORY.md 受注入长度限制（实测 9.4k 即被截断，且**断在半句** —— 尾部
       ``## 协作与提交`` 最先丢）⇒ **8k 字符是硬预算**，不是建议。
    ② INDEX.md 的 ``@行号`` 是日志跳读入口，日志**只追加** ⇒ 锚点必须永远落在同一
       小节的首行；一旦有人重排/改写日志，锚点会静默指错位置。

检查项
  致命  ① 三层文件（MEMORY.md / TOPICS.md / INDEX.md）存在
        ② MEMORY.md 字符数 ≤ BUDGET（默认 8000）
        ③ INDEX.md 每个 ``@N``：所属日志存在、N 在范围内、第 N 行是「小节首行」
        ④ 每篇日志的**最后一节**都有登记锚点（防「写了日志忘登记索引」→ 新内容不可达）
        ④b 每个**步骤小节**（``## S7 第 N 步…``）都有登记锚点
           ⚠️ ④ 只看末节 ⇒ **中间小节被就地覆盖**时全绿（2026-09-15 真实发生：索引里
           第 40/41 步与「待办」条目被一条正则改写掉，守卫照样 exit 0 ✗）
        ⑤ 磁盘上每篇日志都已在 INDEX.md 登记（防「新的一天建了日志忘登记」→ 整篇不可跳读）
        ⑥ 「已出栈的落点」表里的路径引用必须存在，且 ``文件 §小节`` 的**小节名**要在该
           文件内真的出现（只扫本小节，日志摘要里的历史文件名是叙述、不算引用）
  提示  ⑦ MEMORY.md 逼近预算（余量 < 400）

约定
  - 「小节首行」= ``#``~``####`` 标题行，或 ``- 【…】`` 条目行。
  - ``@58/66`` 视为**两个**锚点；``@51~64`` 是范围，**只校验 51**。
  - 锚点只在 ``## 日志清单`` 小节内解析，尾部「写入规范」等段落里的 ``@N`` 不算锚点。

⚠️ **字符口径**：Node 的 ``String.length`` 数的是 **UTF-16 码元**，Python ``len()`` 数的是
   **码点** —— 而预算 8000 正是按前者校准的。故此处一律用 ``_js_len()``，否则带非 BMP
   字符（如 🔄）时两边会差出几百字符，警告线静默偏移。

用法::

    python backend-py/scripts/check_memory.py                  # 退出码 1 = 存在致命项
    MEMORY_DIR=<目录> python backend-py/scripts/check_memory.py   # 仅供 test_guards.py 造故障
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

#: ``backend-py/scripts/check_memory.py`` -> 仓库根
ROOT = Path(__file__).resolve().parents[2]
MEM = Path(os.environ["MEMORY_DIR"]).resolve() if os.environ.get("MEMORY_DIR") else (
    ROOT / ".codebuddy" / "memory"
)
BUDGET = 8000
WARN_AT = BUDGET - 400
SECTION_RE = re.compile(r"^(#{1,4}\s|-\s*【)")

#: ``@N`` 或 ``@N/M/...``（斜杠链视为多个锚点；``~`` 是范围，只取起点）
ANCHOR_RE = re.compile(r"@(\d+(?:\s*/\s*\d+)*)")
DIGITS_RE = re.compile(r"\d+")
#: 日志块标记 ``**`YYYY-MM-DD.md`**（…）``
LOG_MARKER_RE = re.compile(r"\*\*`(\d{4}-\d{2}-\d{2}\.md)`\*\*")
LOG_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
H2_RE = re.compile(r"^##\s")
#: 「步骤小节」：本项目日志里 `## S7 第 N 步：…` 这类小节**每节都该是跳读入口**（④b 用）
STEP_SECTION_RE = re.compile(r"^##\s*S7 第\s*\d+\s*步")
REF_TABLE_RE = re.compile(r"^##\s*已出栈的落点")
LOG_LIST_RE = re.compile(r"^##\s*日志清单")
TICK_RE = re.compile(r"`([^`]+)`")
BARE_MEM_FILE_RE = re.compile(r"^(MEMORY|TOPICS|INDEX)\.md$|^\d{4}-\d{2}-\d{2}\.md$")
SEC_SKIP_RE = re.compile(r"""[<>*…"'()|]""")


def _js_len(text: str) -> int:
    """按 Node 的 ``String.length`` 口径数字符（UTF-16 码元）。"""
    return len(text.encode("utf-16-le")) // 2


def _lines_of(text: str) -> list[str]:
    """行数口径与 PowerShell ``Get-Content`` 一致：末尾换行不额外算一行。"""
    arr = re.split(r"\r?\n", text)
    if arr and arr[-1] == "":
        arr.pop()
    return arr


def _read_text(path: Path) -> str:
    """读原文 —— **必须 ``newline=""``**：Python 默认做换行转换（``\\r\\n`` → ``\\n``），
    而 Node 的 ``readFileSync(p,'utf8')`` **不做** ⇒ 少了行数那么多字符（本项目实测差 53，
    恰好是 MEMORY.md 的行数，直接把「逼近预算」的警告吃掉了）。
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _read_lines(path: Path) -> list[str]:
    return _lines_of(_read_text(path))


def main() -> int:
    fatal: list[str] = []
    info: list[str] = []

    # ① 三层文件必须齐（读法链：MEMORY → TOPICS → INDEX）
    for name in ("MEMORY.md", "TOPICS.md", "INDEX.md"):
        if not (MEM / name).exists():
            fatal.append(f"缺少三层文件  .codebuddy/memory/{name}")

    mem_path = MEM / "MEMORY.md"
    mem_len = 0
    if mem_path.exists():
        mem_len = _js_len(_read_text(mem_path))
        # ② 预算为致命：一旦超限，注入会从尾部截断，`## 协作与提交` 最先丢失
        if mem_len > BUDGET:
            fatal.append(
                f"MEMORY.md {mem_len} 字符 > 预算 {BUDGET} ⇒ 注入必被截断"
                f"（尾部「协作与提交」最先丢），请把细节下移 TOPICS.md"
            )
        elif mem_len > WARN_AT:
            info.append(
                f"MEMORY.md {mem_len}/{BUDGET}，余量仅 {BUDGET - mem_len} ⇒ 下轮进内容前先下移 TOPICS.md"
            )

    # ③ INDEX.md 锚点：按日志分块，块首 `**`YYYY-MM-DD.md`**（…）` 决定后续 @N 归属
    idx_path = MEM / "INDEX.md"
    logs = 0
    anchors = 0
    #: 日志名 -> (行列表, 该日志已登记的锚点集合)
    by_log: dict[str, tuple[list[str], set[int]]] = {}
    if idx_path.exists():
        cur: str | None = None
        cur_lines: list[str] = []
        for line in _read_lines(idx_path):
            marker = LOG_MARKER_RE.search(line)
            if marker:
                name = marker.group(1)
                if not (MEM / name).exists():
                    fatal.append(f"INDEX.md 引用的日志不存在  {name}")
                    cur = None
                    continue
                cur = name
                cur_lines = _read_lines(MEM / name)
                logs += 1
                if cur not in by_log:
                    by_log[cur] = (cur_lines, set())
                continue
            # 锚点只在「## 日志清单」小节内解析（尾部「写入规范」等段落里的 @N 不算锚点）
            if H2_RE.match(line) and not LOG_LIST_RE.match(line):
                cur = None
                continue
            if cur is None:
                continue
            for match in ANCHOR_RE.finditer(line):
                for digits in DIGITS_RE.findall(match.group(1)):
                    num = int(digits)
                    anchors += 1
                    by_log[cur][1].add(num)
                    if num < 1 or num > len(cur_lines):
                        fatal.append(f"锚点越界  {cur} @{num}（该日志仅 {len(cur_lines)} 行）")
                    elif not SECTION_RE.match(cur_lines[num - 1]):
                        fatal.append(
                            f"锚点未落在小节首行  {cur} @{num}  → {cur_lines[num - 1][:40]}"
                        )

    # ④ 每篇日志的末节必须有锚点：日志只追加 ⇒ 忘登记 = 刚写的结论在索引里完全不可达
    for name, (lines, nums) in by_log.items():
        last = 0
        for index, line in enumerate(lines):
            if SECTION_RE.match(line):
                last = index + 1
        if last and last not in nums:
            fatal.append(
                f"日志末节未登记锚点  {name} @{last}（共 {len(lines)} 行，末节从第 {last} 行起）"
            )

    # ④b 每个「步骤小节」也要有锚点：④ 只兜末节 ⇒ 索引条目被**就地覆盖/挤掉**时无感
    #     （2026-09-15 实测：一条 `re.sub` 把第 40/41 步与「待办」条目一起改掉，而末节
    #     第 42 步的锚点还在 ⇒ 守卫全绿，丢了三条跳读入口）
    for name, (lines, nums) in by_log.items():
        for index, line in enumerate(lines):
            if STEP_SECTION_RE.match(line) and (index + 1) not in nums:
                fatal.append(
                    f"步骤小节未登记锚点  {name} @{index + 1}  → {line[:44]}"
                )

    # ⑤ 磁盘上的每日日志必须都已登记进 INDEX.md（新的一天最容易漏；漏了则整篇不可跳读）
    disk_logs = 0
    if idx_path.exists():
        for entry in sorted(os.listdir(MEM)):
            if not LOG_NAME_RE.match(entry):
                continue
            disk_logs += 1
            if entry not in by_log:
                fatal.append(f"日志未登记进 INDEX.md  {entry}（整篇小节都无法按 @行号 跳读）")

    # ⑥ 「已出栈的落点」表引用的路径必须存在（表本身承诺「优先看这些，别翻日志」）
    #    ⚠️ 只扫该小节：日志清单里的摘要文字会提到历史文件名（如 `models.vue`），那些是
    #    **叙述**不是引用，扫全文必误报（Node 版首版即被 `models.vue` 撞出致命 1 处）。
    refs = 0
    sec_refs = 0
    if idx_path.exists():
        in_ref_table = False
        for line in _read_lines(idx_path):
            if H2_RE.match(line):
                in_ref_table = bool(REF_TABLE_RE.match(line))
            if not in_ref_table:
                continue
            for match in TICK_RE.finditer(line):
                # `TOPICS.md §小节` → 拆成「文件」+「小节名」，**两段都要落地**
                file_part, *sec_rest = match.group(1).split("§")
                sec = "§".join(sec_rest).strip()
                raw = file_part.strip()
                if not raw or SEC_SKIP_RE.search(raw) or raw.startswith("http"):
                    continue
                from_root = "/" in raw
                # 裸文件名只认三层记忆文件与日期日志，其余（`models.vue` 之类）是叙述
                if not from_root and not BARE_MEM_FILE_RE.match(raw):
                    continue
                target = (ROOT / raw) if from_root else (MEM / raw)
                refs += 1
                if not target.exists():
                    fatal.append(
                        f"落点表引用的路径不存在  {raw}（相对{'仓库根' if from_root else 'memory 目录'}）"
                    )
                elif sec:
                    sec_refs += 1
                    if sec not in _read_text(target):
                        fatal.append(f"落点表的小节指针落空  {raw} §{sec}（该文件内查不到此小节名）")

    for item in fatal:
        print(f"✗ {item}")
    for item in info:
        print(f"! {item}")
    print(
        f"扫描 .codebuddy/memory ｜ 日志 {logs} 篇已登记 / {disk_logs} 篇在盘 ｜ "
        f"MEMORY.md {mem_len}/{BUDGET} 字符 ｜ 校验锚点 {anchors} 处、落点表路径 {refs} 处"
        f"（其中 §小节指针 {sec_refs} 处）｜ 致命 {len(fatal)} 处"
    )
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
