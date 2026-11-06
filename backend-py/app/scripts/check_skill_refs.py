#!/usr/bin/env python3
"""skills 库引用完整性检查 —— 防止「改名 / 挪库」造成的静默断链。

> 本文件是 ``scripts/check-skill-refs.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。
> 判据、文案、退出码保持不变；改动它是「改守卫」，请连 ``test_guards.py`` 一起改。

为什么需要它：skill 库按 ``backend-py/app/skills/<库名>/<skill名>/`` 组织（**2026-09-15 起并入
后端**，原为仓库根 ``skills/``），库层级与 skill 名都可能变动。
而正文里的路径引用**改坏了不会报任何错** —— 加载器只读 ``SKILL.md``，没人会去点那些路径，
断链只会在读者真的走到那一行时表现为「指向空处」。本脚本把引用全部拉出来逐个验存在性。

用法（仓库根目录）::

    python backend-py/app/scripts/check_skill_refs.py [--verbose]

``--verbose`` 额外列出被跳过的候选（上游路径 / 基准不明），便于审计本脚本自身的盲区。
环境变量 ``SKILL_REFS_DIR`` 可把扫描根指向别处（仅供 ``test_guards.py`` 在副本上做负向实证）。
退出码：0 = 无致命断链；1 = 存在致命断链。

── 解析基准（实测只有这几种，混用会被误判）────────────────────────
  ① ``references/x.md`` / ``scripts/x.mjs`` / ``agents/x.yaml`` —— 相对 **skill 根目录**。
     注意：写在 ``references/foo.md`` 内部的这种引用同样指 skill 根，而不是 references/ 自身。
  ② ``../other-skill/references/x.md`` —— 相对 **当前文档目录**（真·相对路径）。
  ③ ``backend/…`` / ``frontend/…`` / ``skills/…`` / ``docs/…`` / ``backend-py/…`` —— repo 根相对。
  ④ ``<另一个 skill 的目录名>/references/x.md`` —— 相对本库 **``skills/`` 根**。
     2026-11-04 补：这类「细节见某某 skill」的写法很常见，早先全被当「基准不明」跳过，
     于是一条指错的跨 skill 链接可以一直绿着。同名 skill 有多处时不猜（仍归基准不明）。
     2026-11-05 补：首段也认 frontmatter 的 ``name`` —— 本仓 5 个 core skill 的目录名跟 agent
     type 走（``storyboard_breaker``），而 ``name`` 是上游 kebab（``storyboard-breaker``）；
     文档两种写法都会出现，只收目录名等于让一半引用重新退回「基准不明」。**id 仍是目录名**。
  ⑤ ``<库目录>/<skill>/references/x.md``（``wuxia-action/combat-choreography/…``）—— 同按
     **``skills/`` 根**解析。库目录＝``skills/`` 下一层不含 ``SKILL.md`` 的容器；写成不存在的
     库名仍归「基准不明」（不猜，也不乱判红）。

── 分级 ────────────────────────────────────────────────────────────────
  · 致命：``references/…``、跨 skill 的 ``../…`` 与 ``<skill>/…``、repo 根相对路径指向不存在 ⇒ 退出码 1。
  · 非致命：``scripts/…`` 指向不存在。外部技能库只随行 SKILL.md + references/，
    上游 scripts/ 普遍未 vendored，故只列出、不判失败 —— 避免守卫长期红灯而被忽略。
  · 跳过：上游/外来宿主路径（``.ci/``、``spec/``、``.opencode-v2/``、``.claude/`` 等）、
    **示意引用**（同一行紧邻的 ``e.g.`` / ``such as`` / ``例如``）、
    以及基准不明确的候选（不在任何 skill 内 / 非资产目录开头 / 同名 skill 有歧义）。
    各类跳过分别计数，``--verbose`` 可逐条审计 —— 「跳过」不是静默丢弃。
  · 通配符模式（``references/*.md``）：**不再是跳过项**。基准同上四类，命中 ≥1 个真文件
    才算过，一个都没命中 ⇒ 致命（``模式未命中任何文件``）。2026-11-04 前这类串在 token
    收集阶段就被丢掉 —— 连跳过计数都不进，等于「指着空目录写 ``*.md``」可以永远绿。
    仍按上游前缀的（``.ci/prompts/desc-*.md``）照旧跳过，不判红。
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # depth-adjusted-to-app
#: 技能库根：**2026-09-15 起并入后端**（`backend-py/app/skills/`，原为仓库根 `skills/`）。
#: ⚠️ 应用侧的**唯一权威**是 ``app/core/config.py::skills_dir()``（``app/agent/skills.py`` 转发它；
#: services 侧那份转发者已于 2026-10-30 随模块一并删除）。本守卫**刻意保留自己的独立实现**（脚本不
#: ``import app.*``，见 ``app/scripts/README.md`` 的依赖约定）—— 所以这里**不再是「三处必须同步」**，
#: 而是「两套：应用侧单一权威 + 守卫侧独立」；改路径时两边都要改，但改漏一边会有测试/守卫显形
#: （守卫自检 + ``test_guards.py`` 的基线用例）。
#: 可被 ``SKILL_REFS_DIR`` 覆盖：供 ``test_guards.py`` 在**临时副本**上做负向实证
#: （真仓库全程只读）。只覆盖 skills 树，``docs/`` 等 repo 根相对引用仍按真实 REPO_ROOT 解析。
SKILLS_DIR = (Path(os.environ["SKILL_REFS_DIR"]).resolve()
              if os.environ.get("SKILL_REFS_DIR") else REPO_ROOT / "backend-py" / "app" / "skills")
VERBOSE = "--verbose" in sys.argv

FILE_EXT = "md|json|ya?ml|txt|py|js|ts|mjs|cjs|sh|jsonl"
#: ⚠️ ``re.ASCII``：JS 的 ``\w`` 只认 ``[A-Za-z0-9_]``，而 Python 默认认 Unicode 词字符
#: ⇒ 不加这个标志，含中文的 token 判定会与 Node 版分叉。
PATH_TOKEN_RE = re.compile(rf"^(?:\.\./)*[\w.-]+(?:/[\w.*-]+)*\.(?:{FILE_EXT})$", re.ASCII)

def _is_pattern(text: str) -> bool:
    """含通配符 ⇒ 这一条不是「某个文件」，而是「一类文件」，判据改成 glob 命中数。"""
    return "*" in text or "?" in text or "[" in text

#: 上游 Hub 仓库 / 外来宿主平台的专有前缀：本就不属于本项目
UPSTREAM_PREFIXES = (".ci/", "spec/", "hub-skill-market", ".opencode-v2/", ".agents/", ".claude/")
#: 明确相对 repo 根的路径前缀。
#: ``docs/`` 于 2026-09-12 补入（skill 正文常引用 ``docs/*.md`` 作为细节落点）。
#: ``backend-py/`` 于 2026-09-15 补入：Python 后端已是仓库一等公民，少了它，指向
#: ``backend-py/…`` 的引用会落进「基准不明」被静默跳过 —— 与当初漏 ``docs/`` 同病。
#: ⚠️ **`skills/` 与 `backend/` 已从本表移除**（都并入/让位给 ``backend-py/``）⇒ 现在写它们
#: 就是**失效引用**，由下面的 ``_LEGACY_HINTS`` 判**致命**（不给「基准不明」的静默出口）。
REPO_ROOT_PREFIXES = ("frontend/", "docs/", "backend-py/")
#: 旧前缀 → 提示语：迁移/下线后最危险的是「看着像路径、其实已失效」的写法 ⇒ 一律致命并给改法。
#: （``backend/`` 于 2026-09-15 加入：Python 后端已全量覆盖，TS 旧后端待删 ⇒ 引用它等于指空。）
_LEGACY_HINTS = {
    "skills/": "skills/ 已并入 backend-py/app/skills/",
    "backend/": "backend/ 是 TS 旧后端（Python 版在 backend-py/）",
}
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


def frontmatter_name(skill_md: Path) -> str | None:
    """``SKILL.md`` frontmatter 里的 ``name:``；没有 frontmatter / 没有该字段 ⇒ ``None``。

    只读前 8 KB 且只认**首段 frontmatter**（正文里随便一行 ``name:`` 不算）。
    """
    text = skill_md.read_text(encoding="utf-8", errors="replace")[:8192]
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    head = text if end < 0 else text[:end]
    for line in head.splitlines():
        if line.startswith("name:"):
            value = line[len("name:"):].strip().strip("\"'").strip()
            return value or None
    return None


def skill_dir_index() -> dict[str, list[Path]]:
    """skill 名 → 含 ``SKILL.md`` 的目录，**同名可能多处**（如 ``storyboard``）。

    用途：解析**跨 skill 引用**。skill 正文里写 ``film-shot/references/shot-language.md``
    是常见写法（“细节见另一个 skill 那份”），它的基准是本库 ``skills/`` 根、不是 repo 根。
    早先这类 token 一律落进「基准不明」被跳过 ⇒ 链接断了也没人知道（2026-11-04 实测：
    5 条跳过里 2 条其实完全可判，其中一条指向的正是另一个 skill 的真文件）。

    **两个名字都收**：本仓 5 个 core skill 的目录名跟着 **agent type** 走（``storyboard_breaker``），
    而 frontmatter 的 ``name`` 是上游 kebab（``storyboard-breaker``）—— 后者也是模型/文档会写在
    引用里的名字。只认目录名 ⇒ 这类引用全部落「基准不明」被静默跳过，而它们**完全可判**。
    """
    index: dict[str, list[Path]] = {}
    for skill_md in walk(SKILLS_DIR):
        if skill_md.name != "SKILL.md":
            continue
        parent = skill_md.parent
        index.setdefault(parent.name, []).append(parent)
        declared = frontmatter_name(skill_md)
        if declared and declared != parent.name:
            index.setdefault(declared, []).append(parent)
    return index


def first_segment(clean: str) -> str:
    """引用串的首段（跳过前导 ``../``）。「首段是不是资产目录 / 是不是某个 skill 名」全问它。"""
    return re.sub(r"^(?:\.\./)+", "", clean).split("/")[0]


def library_dirs() -> set[str]:
    """``skills/`` 下一层**不含 SKILL.md 的容器目录**（外部技能库，如 ``wuxia-action``）。

    用来解析**库限定形式**的引用：``wuxia-action/combat-choreography/references/x.md``
    这种写法在库内文档与 README 里都很自然 —— 首段是库目录、整串从 ``skills/`` 根起算。
    早先它不在任何基准里 ⇒ 落「基准不明」被跳过（2026-11-05 实测：10 条跳过里
    只有它一条真能判，且文件确实存在）。只认**真实存在的容器目录**，库名写错照样不猜。
    """
    names: set[str] = set()
    if not SKILLS_DIR.is_dir():
        return names
    for entry in os.scandir(SKILLS_DIR):
        if entry.is_dir() and not (Path(entry.path) / "SKILL.md").exists():
            names.add(entry.name)
    return names


ORPHAN_DIRS = ("references", "scripts")

# 反白名单：只跳过「显然不是知识资产」的文件，其余**一律**要求在正文里被指路。
#
# 这里原本是一张正向白名单（``.md/.py/.json/.yaml/.yml/.ts/.sh/.txt``），2026-11-06 发现
# ``multi-shot`` 与 ``n-storyboard`` 的 ``scripts/index.js`` **根本不在网内** —— 白名单
# 是有 `.ts`、有 `.sh`、偏偏没 `.js`。两份当时恰好被人指路（``entry: "scripts/index.js"``）
# 才没露出症状，但形状本身是错的：正向白名单意味「明天新加的任何类型从落地那刻起就默认
# 不受看管，且永远不报」。故改为反白名单 —— 默认**在**网内，只有图片/音视频/压缩包/字体/
# 二进制/占位符这类「没有可读内容、也无从指路」的才跳过。
NON_ASSET_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico", ".tif", ".tiff",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".pkl", ".npy", ".npz",
    ".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".onnx",
)
# 占位与凭据类：``.gitkeep`` 是空文件、``LICENSE``/``NOTICE`` 是许可正文，
# 要求正文去「指路」它们只会逼出凑数句子。
NON_ASSET_NAMES = (".gitkeep", ".gitignore", ".DS_Store", "LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.md")


def _is_non_asset(asset: Path) -> bool:
    """该文件是否「没有可读内容、也无从指路」—— 只有这类才允许不被正文提到。"""
    if asset.name in NON_ASSET_NAMES:
        return True
    return asset.suffix.lower() in NON_ASSET_SUFFIXES


def _mentions(text: str, name: str, folder: str, suffix: str) -> bool:
    """**本 skill 内**的三种指路形态：点名文件名 / 通配符 / 多级目录。"""
    # 词边界匹配：`A.md` 不该被 `CTA.md` 里的同名尾巴蒙混过去
    if re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", text):
        return True
    # 通配符整体指路（`references/*.md`）也是明确指路，不是「没说」
    if re.search(rf"(?<![\w.-]){re.escape(folder)}\*{re.escape(suffix)}", text):
        return True
    # 多级目录（`references/styles/`）：一整个目录按需读，不该被迫逐个点名
    return folder.count("/") >= 2 and folder in text


def _skill_aliases(skill_root: Path) -> list[str]:
    """一个 skill 可被跨 skill 引用的两种写法：**目录名**与 frontmatter 的 ``name``。

    两者**不保证一致**（实测 ``extractor`` 的 ``name`` 是 ``character-scene-extractor``、
    ``voice_assigner`` 是 ``voice-assigner``）。只认目录名会把用别名写的合法指路判成孤儿。
    """
    names = [skill_root.name]
    try:
        head = (skill_root / "SKILL.md").read_text(encoding="utf-8", errors="replace")[:800]
    except OSError:
        return names
    m = re.search(r"^name:\s*(\S+)", head, re.MULTILINE)
    if m:
        alias = m.group(1).strip("'\"")
        if alias and alias not in names:
            names.append(alias)
    return names


def _mentions_qualified(text: str, aliases: list[str], rel_posix: str, folder: str) -> bool:
    """**跨 skill** 指路：必须带 ``<skill>/`` 前缀，裸名不算（裸名只说得出文件名，说不出是谁的）。"""
    tail = rel_posix.rsplit("/", 1)[-1]
    gap = r"[^\s`\"')<>\[\]]*"
    for name in aliases:
        if re.search(rf"{re.escape(name)}/{gap}{re.escape(tail)}", text):
            return True
        if re.search(rf"{re.escape(name)}/{gap}{re.escape(folder)}", text):
            return True
    return False


def orphan_assets() -> list[tuple[str, str]]:
    """``references/`` 与 ``scripts/`` 下**从未被本 skill 任何 md 提到**的资产。

    这是断链的**反面**，但后果同级：断链是「指向空白」，孤儿是「空白无人指向」—— 两种都让
    知识进不了模型上下文，只不过程序化检查一直只盯前者。2026-11-06 实测 121 份资产里 12 份
    孤儿，全是真的能力白废：``ui-motion`` 正文写着 "The 8 style files" 却没说它们在
    ``references/styles/``、``common-adaptations.md`` 无人指路；``subtitle-correction`` 的
    Step 1 让用户手抄术语，而 ``references/terminology.md`` 早已备好对照表；
    ``skill-creator`` 有 ``NAMING.cn.md`` 中文版却只指英文版。补齐指路后归零。

    指路有两种合法形态、**都算**：点名文件名（``terminology.md``），或指到**多级**目录
    （``references/styles/``，一整个目录按需读，不该被迫逐个点名）。单级 ``references/``
    不算 —— 那只是「有这么个目录」，不构成「去读这一份」。
    只扫 ``references/`` 与 ``scripts/`` **子树**（``references/styles/A.md`` 的直接父目录是
    ``styles``，只看直接父目录会整片漏掉）；``NOTICE.md`` / ``agents/`` / ``gui.json`` /
    ``meta.yaml`` 这类平台与元数据文件本来就不该被正文引用，不在子树内、自然不判。

    资产**类型**走反白名单（`NON_ASSET_SUFFIXES`／`NON_ASSET_NAMES`）：默认**在**网内，只有
    图片 / 音视频 / 压缩包 / 字体 / 二进制 / 占位符跳过。原先那张正向白名单会让「新加的类型」
    从落地那刻起就默认不受看管（`.js` 就这样漏掉 2 份），且**永远不报**。

    「谁指的路」取**整棵树**的 md，不是只取本 skill 的：跨 skill 引用（``a/SKILL.md`` 里写
    ``b/references/x.md``）同样是有效指路，只认本 skill 会把这类正常结构全判成孤儿。
    """
    found: list[tuple[str, str]] = []
    if not SKILLS_DIR.is_dir():
        return found
    chunks: list[str] = []
    for md in sorted(SKILLS_DIR.rglob("*.md")):
        try:
            chunks.append(md.read_text(encoding="utf-8"))
        except OSError:
            continue
    text = "\n".join(chunks)
    for skill_md in sorted(SKILLS_DIR.rglob("SKILL.md")):
        skill_root = skill_md.parent
        own_text = "\n".join(
            md.read_text(encoding="utf-8", errors="replace")
            for md in sorted(skill_root.rglob("*.md"))
        )
        aliases = _skill_aliases(skill_root)
        for asset in sorted(skill_root.rglob("*")):
            if not asset.is_file() or _is_non_asset(asset):
                continue
            rel = asset.relative_to(skill_root)
            if not ({"references", "scripts"} & set(rel.parts[:-1])):
                continue
            folder = rel.as_posix().rsplit("/", 1)[0] + "/"
            if _mentions(own_text, asset.name, folder, asset.suffix):
                continue
            # 跨 skill 指路**必须带路径限定**才认。2026-11-06 反验暴露：判据原来看全树文本，
            # 于是「别的 skill 里有个同名文件」会替这一份担保 —— 抹掉
            # ``multi-shot`` 对 ``scripts/index.js`` 的点名后，``n-storyboard`` 那句
            # ``entry: "scripts/index.js"`` 仍在全树里，孤儿数照旧 0、一声不响。
            # 裸名只在**本 skill 内**算指路；出了这个 skill 就必须写成
            # ``<skill>/references/x.md`` 这种带路径的形式，否则无从判断指的是谁的。
            if _mentions_qualified(text, aliases, rel.as_posix(), folder):
                continue
            found.append((_rel(skill_root), rel.as_posix()))
    return found


def resolve_base(
    clean: str,
    doc_dir: Path,
    skill_root: Path | None,
    skill_dirs: dict[str, list[Path]],
    library_names: set[str] | None = None,
) -> tuple[Path, str] | None:
    """把引用串拆成 ``(基准目录, 其后片段)``；判不了基准 → ``None``（调用方归「基准不明」）。

    抽成函数是为了让**单一路径**与**通配符模式**走同一套基准判断 —— 两份各写一遍的话，
    迟早有一份漏掉新加的基准（跨 skill 引用），而漏掉的表现是「悄悄退回不做检查」。
    """
    head = first_segment(clean)
    if any(clean.startswith(prefix) for prefix in REPO_ROOT_PREFIXES):
        return REPO_ROOT, clean
    if clean.startswith("../"):
        return doc_dir, clean
    if head in SKILL_ROOT_ASSET_DIRS:
        if skill_root is None:
            return None
        return skill_root, clean
    if len(skill_dirs.get(head) or ()) == 1:
        return skill_dirs[head][0], clean[len(head) + 1:]
    if head in (library_names if library_names is not None else library_dirs()):
        return SKILLS_DIR, clean
    return None


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
    skill_dirs = skill_dir_index()
    library_names = library_dirs()
    orphans = orphan_assets()

    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        locate = make_line_locator(text)
        doc_dir = doc.parent
        skill_root = skill_root_of(doc_dir)

        for token, offset in collect_tokens(text).items():
            if "/" not in token or not PATH_TOKEN_RE.match(token):
                continue
            clean = re.sub(r"^\./", "", token)
            pos = f"{_rel(doc)}:{locate(offset)}"

            if any(clean.startswith(prefix) for prefix in UPSTREAM_PREFIXES):
                upstream += 1
                skipped.append(f"[上游/外来宿主路径] {pos}  →  {clean}")
                continue

            # 旧位置（技能库已并入 backend-py/app/skills/、TS 旧后端 backend/ 将下线）⇒ 致命：
            # 这类引用「看着对、其实指空」，正是本守卫存在的理由；给提示比落进「基准不明」有用得多。
            legacy = next((prefix for prefix in _LEGACY_HINTS if clean.startswith(prefix)), None)
            if legacy:
                checked += 1
                fatal.append((pos, f"{clean}（{_LEGACY_HINTS[legacy]}，请改前缀）"))
                continue

            # 同一行内、token 之前若出现「举例」措辞 → 该路径泛指而非依赖，跳过
            line_start = text.rfind("\n", 0, offset) + 1
            if ILLUSTRATIVE_RE.search(text[line_start:offset]):
                illustrative += 1
                skipped.append(f"[示意引用：举例而非依赖] {pos}  →  {clean}")
                continue

            base = resolve_base(clean, doc_dir, skill_root, skill_dirs, library_names)
            if base is None:
                ambiguous += 1
                skipped.append(f"[基准不明] {pos}  →  {clean}")
                continue
            base_dir, rest = base
            first_seg = first_segment(clean)

            if _is_pattern(rest):
                # 通配符模式（`references/*.md`）：早先这一整类在收集阶段就被 `continue` 掉，
                # 连「跳过」计数都不进 —— 等于「指着不存在的目录写 *.md」可以永远绿。
                # 现在按同一套基准去 glob：命中 ≥1 个真文件才算数。同目录别名不再猜。
                checked += 1
                hits = [p for p in glob.glob(os.path.join(str(base_dir), rest), recursive=True)
                        if os.path.isfile(p)]
                if hits:
                    continue
                if first_seg in NON_FATAL_ASSET_DIRS:
                    notes.append((pos, token))
                else:
                    fatal.append((pos, f"{token}（模式未命中任何文件）"))
                continue

            target = (base_dir / rest).resolve()
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
    for pos, token in sorted(orphans):
        print(f"资产无人指路  {pos}  →  {token}")
    if orphans:
        print("")

    if VERBOSE:
        for item in sorted(skipped):
            print(item)
        if skipped:
            print("")

    print(f"扫描 {_rel(SKILLS_DIR)}/**/*.md 共 {len(docs)} 个文件")
    print(
        f"待校验 {checked} 处 ｜ 致命断链 {len(fatal)} 处 ｜ 非致命缺脚本 {len(notes)} 处 ｜ "
        f"资产无人指路 {len(orphans)} 份 ｜ "
        f"跳过：上游/外来宿主 {upstream} 处 / 示意引用 {illustrative} 处 / 基准不明 {ambiguous} 处"
    )
    return 1 if (fatal or orphans) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
