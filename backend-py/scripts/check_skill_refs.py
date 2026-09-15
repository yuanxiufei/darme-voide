#!/usr/bin/env python3
"""skills 库引用完整性检查 —— 防止「改名 / 挪库」造成的静默断链。

> 本文件是 ``scripts/check-skill-refs.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。
> 判据、文案、退出码保持不变；改动它是「改守卫」，请连 ``test_guards.py`` 一起改。

为什么需要它：skill 库按 ``backend-py/skills/<库名>/<skill名>/`` 组织（**2026-09-15 起并入
后端**，原为仓库根 ``skills/``），库层级与 skill 名都可能变动。
而正文里的路径引用**改坏了不会报任何错** —— 加载器只读 ``SKILL.md``，没人会去点那些路径，
断链只会在读者真的走到那一行时表现为「指向空处」。本脚本把引用全部拉出来逐个验存在性。

用法（仓库根目录）::

    python backend-py/scripts/check_skill_refs.py [--verbose]

``--verbose`` 额外列出被跳过的候选（上游路径 / 基准不明），便于审计本脚本自身的盲区。
环境变量 ``SKILL_REFS_DIR`` 可把扫描根指向别处（仅供 ``test_guards.py`` 在副本上做负向实证）。
退出码：0 = 无致命断链；1 = 存在致命断链。

── 解析基准（实测只有这几种，混用会被误判）────────────────────────
  ① ``references/x.md`` / ``scripts/x.mjs`` / ``agents/x.yaml`` —— 相对 **skill 根目录**。
     注意：写在 ``references/foo.md`` 内部的这种引用同样指 skill 根，而不是 references/ 自身。
  ② ``../other-skill/references/x.md`` —— 相对 **当前文档目录**（真·相对路径）。
  ③ ``backend/…`` / ``frontend/…`` / ``skills/…`` / ``docs/…`` / ``backend-py/…`` —— repo 根相对。

── 分级 ────────────────────────────────────────────────────────────────
  · 致命：``references/…``、跨 skill ``../…``、repo 根相对路径指向不存在 ⇒ 退出码 1。
  · 非致命：``scripts/…`` 指向不存在。外部技能库只随行 SKILL.md + references/，
    上游 scripts/ 普遍未 vendored，故只列出、不判失败 —— 避免守卫长期红灯而被忽略。
  · 跳过：上游/外来宿主路径（``.ci/``、``spec/``、``.opencode-v2/``、``.claude/`` 等）、
    含通配符的模式、**示意引用**（同一行紧邻的 ``e.g.`` / ``such as`` / ``例如``）、
    以及基准不明确的候选（不在任何 skill 内 / 非资产目录开头，如 ``export/…``）。
    各类跳过分别计数，``--verbose`` 可逐条审计 —— 「跳过」不是静默丢弃。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
#: 技能库根：**2026-09-15 起并入后端**（`backend-py/skills/`，原为仓库根 `skills/`）。
#: ⚠️ 必须与后端代码一致：``app/services/skills.py`` 的 ``SKILLS_DIR`` 与
#: ``app/services/agents/skills.py`` 的 ``skills_dir()``（三处同步改）。
#: 可被 ``SKILL_REFS_DIR`` 覆盖：供 ``test_guards.py`` 在**临时副本**上做负向实证
#: （真仓库全程只读）。只覆盖 skills 树，``docs/`` 等 repo 根相对引用仍按真实 REPO_ROOT 解析。
SKILLS_DIR = (Path(os.environ["SKILL_REFS_DIR"]).resolve()
              if os.environ.get("SKILL_REFS_DIR") else REPO_ROOT / "backend-py" / "skills")
VERBOSE = "--verbose" in sys.argv

FILE_EXT = "md|json|ya?ml|txt|py|js|ts|mjs|cjs|sh|jsonl"
#: ⚠️ ``re.ASCII``：JS 的 ``\w`` 只认 ``[A-Za-z0-9_]``，而 Python 默认认 Unicode 词字符
#: ⇒ 不加这个标志，含中文的 token 判定会与 Node 版分叉。
PATH_TOKEN_RE = re.compile(rf"^(?:\.\./)*[\w.-]+(?:/[\w.*-]+)*\.(?:{FILE_EXT})$", re.ASCII)
#: 上游 Hub 仓库 / 外来宿主平台的专有前缀：本就不属于本项目
UPSTREAM_PREFIXES = (".ci/", "spec/", "hub-skill-market", ".opencode-v2/", ".agents/", ".claude/")
#: 明确相对 repo 根的路径前缀。
#: ``docs/`` 于 2026-09-12 补入（skill 正文常引用 ``docs/*.md`` 作为细节落点）。
#: ``backend-py/`` 于 2026-09-15 补入：Python 后端已是仓库一等公民（``backend/`` 即将下线），
#: 少了它，指向 ``backend-py/…`` 的引用会落进「基准不明」被静默跳过 —— 与当初漏 ``docs/`` 同病。
#: ⚠️ **`skills/` 已从本表移除**：技能库当天并入 ``backend-py/skills/`` ⇒ 现在写 ``skills/…``
#: 就是**失效引用**，由下面的 ``_LEGACY_SKILLS_PREFIX`` 判**致命**（不给它「基准不明」的静默出口）。
REPO_ROOT_PREFIXES = ("backend/", "frontend/", "docs/", "backend-py/")
#: 旧位置前缀：挪库后最危险的是「看着像路径、其实已失效」的写法 ⇒ 一律致命并给出改法。
_LEGACY_SKILLS_PREFIX = "skills/"
#: 基准为 skill 根的资产目录（文档不在任何 skill 内时基准不明，跳过）
SKILL_ROOT_ASSET_DIRS = ("references", "scripts", "agents")
#: 这些目录的缺失只提示、不判失败
NON_FATAL_ASSET_DIRS = ("scripts",)
#: 「示意引用」标记：紧邻 token 之前的措辞表明该路径只是**举例**，不是本 skill 的依赖。
#: 必须**锚定在 token 紧前方**（标记与路径之间只允许「非字母数字、非汉字」的字符）；
#: 放宽成「同一行出现过 e.g.」会误伤真依赖（Node 版实测过这个假阴性）。
ILLUSTRATIVE_RE = re.compile(
    r"(?:e\.g\.|eg\.|for example|such as|例如|比如|譬如)[^A-Za-z0-9\u4e00-\u9fa5]*$", re.IGNORECASE
)

TICK_TOKEN_RE = re.compile(r"`([^`\n]+)`")
MD_LINK_RE = re.compile(r"\]\(([^)\n]+)\)")


def walk(directory: Path) -> list[Path]:
    out: list[Path] = []
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        path = directory / entry.name
        if entry.is_dir():
            out.extend(walk(path))
        else:
            out.append(path)
    return out


def collect_tokens(text: str) -> dict[str, int]:
    """路径候选 → **首次**出现偏移：反引号 token + markdown 链接目标。"""
    found: dict[str, int] = {}
    for pattern in (TICK_TOKEN_RE, MD_LINK_RE):
        for match in pattern.finditer(text):
            token = match.group(1).strip()
            if token not in found:
                found[token] = match.start(1)
    return found


def skill_root_of(doc_dir: Path) -> Path | None:
    """文档所属 skill 根（最近的存在 ``SKILL.md`` 的祖先）；不在任何 skill 内 → None。"""
    current = doc_dir
    while str(current).startswith(str(SKILLS_DIR)):
        if (current / "SKILL.md").exists():
            return current
        if current == SKILLS_DIR:
            break
        current = current.parent
    return None


def make_line_locator(text: str):
    """偏移 → 行号（二分；先建行首表，避免逐 token 重扫全文）。"""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)

    def locate(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    return locate


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> int:
    docs = [f for f in walk(SKILLS_DIR) if f.suffix == ".md"]
    fatal: list[tuple[str, str]] = []
    notes: list[tuple[str, str]] = []
    skipped: list[str] = []
    checked = upstream = illustrative = ambiguous = 0

    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        locate = make_line_locator(text)
        doc_dir = doc.parent
        skill_root = skill_root_of(doc_dir)

        for token, offset in collect_tokens(text).items():
            if "/" not in token or "*" in token or not PATH_TOKEN_RE.match(token):
                continue
            clean = re.sub(r"^\./", "", token)
            pos = f"{_rel(doc)}:{locate(offset)}"

            if any(clean.startswith(prefix) for prefix in UPSTREAM_PREFIXES):
                upstream += 1
                skipped.append(f"[上游/外来宿主路径] {pos}  →  {clean}")
                continue

            # 旧位置（技能库已并入 backend-py/skills/）⇒ 致命：这类引用「看着对、其实指空」，
            # 正是本守卫存在的理由；给提示比让它落进「基准不明」被跳过有用得多。
            if clean.startswith(_LEGACY_SKILLS_PREFIX):
                checked += 1
                fatal.append((pos, f"{clean}（skills/ 已并入 backend-py/skills/，请改前缀）"))
                continue

            # 同一行内、token 之前若出现「举例」措辞 → 该路径泛指而非依赖，跳过
            line_start = text.rfind("\n", 0, offset) + 1
            if ILLUSTRATIVE_RE.search(text[line_start:offset]):
                illustrative += 1
                skipped.append(f"[示意引用：举例而非依赖] {pos}  →  {clean}")
                continue

            first_seg = re.sub(r"^(?:\.\./)+", "", clean).split("/")[0]
            if any(clean.startswith(prefix) for prefix in REPO_ROOT_PREFIXES):
                target = REPO_ROOT / clean
            elif clean.startswith("../"):
                target = (doc_dir / clean).resolve()
            elif first_seg in SKILL_ROOT_ASSET_DIRS:
                if skill_root is None:
                    ambiguous += 1
                    skipped.append(f"[基准不明：不在任何 skill 内] {pos}  →  {clean}")
                    continue
                target = (skill_root / clean).resolve()
            else:
                ambiguous += 1
                skipped.append(f"[基准不明：非资产目录开头] {pos}  →  {clean}")
                continue

            checked += 1
            if target.exists():
                continue
            if first_seg in NON_FATAL_ASSET_DIRS:
                notes.append((pos, token))
            else:
                fatal.append((pos, token))

    for pos, token in sorted(fatal, key=lambda item: item[0]):
        print(f"断链  {pos}  →  {token}")
    if fatal:
        print("")
    for pos, token in sorted(notes, key=lambda item: item[0]):
        print(f"缺脚本（非致命）  {pos}  →  {token}")
    if notes:
        print("")

    if VERBOSE:
        for item in sorted(skipped):
            print(item)
        if skipped:
            print("")

    print(f"扫描 backend-py/skills/**/*.md 共 {len(docs)} 个文件")
    print(
        f"待校验 {checked} 处 ｜ 致命断链 {len(fatal)} 处 ｜ 非致命缺脚本 {len(notes)} 处 ｜ "
        f"跳过：上游/外来宿主 {upstream} 处 / 示意引用 {illustrative} 处 / 基准不明 {ambiguous} 处"
    )
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
