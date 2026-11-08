"""**画布修复内核** —— 提案 / 应用 / 回滚（源：`assistant-runtime/repair-tools.mjs` ✓）。

为什么这块值得搬：本仓 ``runtime_doctor`` 只回答「服务连没连上」（L0 只读 ✓），
用户拿到 ``blockers`` 之后**没法自助修** ✗ —— 修不了就只能回滚到作者。
源项目那套护栏是**踩出来的**（每处替换唯一命中、应用前全量备份、方案过期检测），
逐条重写比重新设计一遍省事，也少踩一遍坑 ✓。

⚠️⚠️ **三个必须保真的点**（改了就等于拆护栏）：

1. **每处替换必须唯一命中** ✓✗ —— ``search`` 在文件里出现 0 次报「找不到」、
   ≥2 次报「出现多次，请给更长的上下文」。允许"出现多次就替换第一处"会**误伤同名的另一段代码** ✗。
2. **应用前先全量校验 + 全量备份** ✓ —— 任一文件「已被创建 / 已不存在 / 已被其他操作改过」
   就整单拒绝（方案过期），别写一半 ✗。
3. **换行符原样读写** ⚠️⚠️ —— Python 的 ``open(..., newline=None)`` 默认会把 ``\\n`` 转成
   ``os.linesep``：Windows 下等于把整份 CRLF 文件改写成 LF ✓✗，于是
   ``oldContent == 盘上内容`` 的校验从此**永远为假** ⇒ 方案永远"过期"、
   回滚备份与线上文件字节不一致。**读写两边都必须 ``newline=""``** ✓。

与源的三处**刻意差异**（不是漏搬）：

* 源把可改范围硬编码成 ``infinite-canvas/web/``；本仓前端在 ``frontend/app/``
  ⇒ 改成 ``allowed_prefixes`` 参数（默认只给前端 ✓，最小权能）。
* 源 ``applyProposalFiles`` **没有写中途失败回滚**（它靠"写前已全量校验"兜底）。
  本仓补上：第 3 个文件写失败时，把前 2 个**就地回滚** ✓ —— 校验过 ≠ 写不失败（磁盘满/占用）。
* 源用 ``Buffer.byteLength`` 量字节；本仓用 ``len(s.encode())`` ✓（同一口径）。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

#: 单文件可读上限（源 ``MAX_READ_BYTES = 260_000``）
MAX_READ_BYTES = 260_000
#: 单文件可写上限（源 ``MAX_WRITE_BYTES = 500_000``）
MAX_WRITE_BYTES = 500_000
#: 列目录最多回多少条（源 ``MAX_LIST_ENTRIES = 220``）
MAX_LIST_ENTRIES = 220
#: 搜索最多扫多少文件（源 ``MAX_SEARCH_FILES = 1500``）
MAX_SEARCH_FILES = 1500
#: 搜索最多回多少条（源 ``MAX_SEARCH_RESULTS = 60``）
MAX_SEARCH_RESULTS = 60
#: 一次最多提议几个文件（源硬编码 8）
MAX_PROPOSAL_FILES = 8
#: 目录树递归深度上限（源 0-4）
MAX_LIST_DEPTH = 4

#: **绝不进入**的目录段 ✓ —— 命中任意一段就跳过（不是"只准这几段"）
#: 2026-11-07：随 ``data/`` 分层补 ``indexes`` / ``corpus`` —— 前者是 2 GB 派生 sqlite
#: （Search 的 rglob 会白扫一遍），后者是随仓语料（Agent 要读走 ``search_script_library``，
#: 不该从工作区直接改源文件）。旧结构下这两类都在 ``data/`` 里、被那一段顺手挡住了。
DENY_SEGMENTS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    "data", "models", "indexes", "corpus", "backups", ".repair-backups", ".assistant-backups",
    ".update-backup", "output", "outputs", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".nuxt", ".output",
})

#: **绝不修改**的文件名（大小写不敏感）✓ —— 依赖清单/锁/密钥：改这些会毁掉可复现性
DENY_FILES: frozenset[str] = frozenset({
    ".env", "package.json", "package-lock.json", "pnpm-lock.yaml",
    "yarn.lock", "bun.lock", "bun.lockb", "poetry.lock", "pipfile.lock",
    "pyproject.toml", "requirements.txt", "uv.lock", "dockerfile",
})

#: 认作文本的扩展名 ✓（不在表里的一律拒绝，避免把二进制当文本改写）
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".css", ".scss",
    ".less", ".html", ".htm", ".json", ".jsonc", ".md", ".mdx", ".txt",
    ".yml", ".yaml", ".toml", ".ini", ".py", ".sql", ".graphql", ".svg",
    ".sh", ".ps1", ".bat",
})

#: 默认只允许改前端源码 ✓（与源的 ``infinite-canvas/web/`` 同口径，换成本仓路径）
DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = ("frontend/app/",)


class RepairError(Exception):
    """护栏拒绝 —— **文案直接给用户看** ✓（别包成 500：这不是服务出错，是方案不合格）。"""


def repair_error_message(err: BaseException) -> str:
    """取用户可读文案（``RepairError`` 原样，其余兜底）。"""
    text = str(err).strip()
    return text or "修复失败"


# ---------------------------------------------------------------- 路径与文件


def _normalize_relative(value: Any) -> str:
    """把用户/AI 给的路径规整成 ``a/b/c`` 相对形式 ✓（统一分隔符、去掉 ``./``）。"""
    if not isinstance(value, str) or not value.strip():
        raise RepairError("缺少 path")
    rel = value.strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.strip("/")
    if not rel:
        raise RepairError("path 不能为空")
    return rel


def resolve_inside(root: Path, relative: str) -> Path:
    """解析到 ``root`` 之内 ✓ —— **路径逃逸在这里拦死**（``..`` / 绝对路径 / 盘符）。"""
    root_resolved = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive:
        raise RepairError(f"只接受相对路径：{relative}")
    target = (root_resolved / candidate).resolve()
    # ⚠️ 用「相对路径是否以 .. 开头」判，而不是 startswith 字符串比较：
    # `…\x\ab` 与 `…\x\a` 的字符串前缀会假命中 ✓✗
    try:
        target.relative_to(root_resolved)
    except ValueError as err:
        raise RepairError(f"路径越界：{relative}") from err
    return target


def _assert_allowed_prefix(relative: str, allowed_prefixes: Sequence[str]) -> None:
    """只允许改白名单前缀下的文件 ✓（源：只准 ``infinite-canvas/web/``）。"""
    prefixes = tuple(p.replace("\\", "/").strip("/") + "/" for p in allowed_prefixes if p)
    if not prefixes:
        return
    if not any(relative.startswith(prefix) for prefix in prefixes):
        shown = "、".join(prefixes)
        raise RepairError(f"本版只自动应用这些目录下的源码：{shown}；其余目录请只给方案，不自动落地")


def _assert_not_denied(file: Path, relative: str) -> None:
    """拒绝段/拒绝文件名 —— 校验的是**相对路径的每一段** ✓（不是绝对路径，免误伤盘符外的同名目录）。"""
    parts = [part.lower() for part in Path(relative).parts]
    for part in parts[:-1]:
        if part in DENY_SEGMENTS:
            raise RepairError(f"禁止读写目录 {part}（{relative}）")
    name = parts[-1] if parts else ""
    if name in DENY_FILES:
        raise RepairError(f"禁止修改 {name}：依赖清单/锁/密钥类文件会影响构建与安全")
    if name.endswith((".lock", ".lockb")):
        raise RepairError(f"禁止修改锁文件 {name}")
    # 构建配置与入口 HTML：源用正则拦，这里按同名规则展开 ✓
    if name.startswith(("tsconfig", "vite.config", "nuxt.config", "postcss.config",
                        "tailwind.config", "eslint.config")) and name.endswith((".json", ".js", ".ts", ".mjs", ".cjs")):
        raise RepairError(f"禁止修改构建配置 {name}：它会改变构建产物与执行安全")
    if name == "index.html":
        raise RepairError("禁止修改入口 HTML：它会改变首屏加载与安全边界")


def _assert_text_file(file: Path, relative: str) -> None:
    """只接受文本文件 ✓ —— 扩展名白名单 + 头 8KB 无 NUL（二进制判定）。"""
    if file.suffix.lower() not in TEXT_EXTENSIONS:
        raise RepairError(f"不是可编辑的文本源码：{relative}")
    if file.exists():
        with open(file, "rb") as handle:
            head = handle.read(8192)
        if b"\x00" in head:
            raise RepairError(f"看起来是二进制文件：{relative}")


def _read_text_file(file: Path, relative: str) -> str:
    """读文本，**保留原换行** ⚠️（``newline=""`` 是硬要求，见模块头第 3 条）。"""
    size = file.stat().st_size
    if size > MAX_READ_BYTES:
        raise RepairError(f"{relative} 太大（{size} 字节 > {MAX_READ_BYTES}），不自动修改")
    try:
        with open(file, "r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except UnicodeDecodeError as err:
        raise RepairError(f"{relative} 不是 UTF-8 文本，拒绝自动修改") from err


def _write_text_file(file: Path, content: str) -> None:
    """写文本，**保留给定内容原样** ⚠️（``newline=""`` 同上）。"""
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def _clip(value: Any, size: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= size else text[:size]


# ---------------------------------------------------------------- 只读工具


def list_workspace(root: Path, relative: str = ".", depth: int = 2) -> dict[str, Any]:
    """列工作区目录树 ✓（不进 ``DENY_SEGMENTS``、只回文本文件、条数封顶）。"""
    safe_depth = max(0, min(int(depth or 0), MAX_LIST_DEPTH))
    base_rel = "" if relative in (".", "", "./") else _normalize_relative(relative)
    base = Path(root).resolve() if not base_rel else resolve_inside(root, base_rel)
    entries: list[dict[str, Any]] = []
    full = False

    def walk(directory: Path, rel_prefix: str, level: int) -> None:
        nonlocal full
        try:
            children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return
        for child in children:
            if full:
                return
            name = child.name
            child_rel = f"{rel_prefix}{name}" if not rel_prefix else f"{rel_prefix}/{name}"
            if child.is_dir():
                if name.lower() in DENY_SEGMENTS:
                    continue
                entries.append({"path": child_rel + "/", "type": "dir"})
                if level < safe_depth:
                    walk(child, child_rel, level + 1)
            elif child.suffix.lower() in TEXT_EXTENSIONS:
                entries.append({"path": child_rel, "type": "file", "size": child.stat().st_size})
            if len(entries) >= MAX_LIST_ENTRIES:
                full = True

    walk(base, base_rel, 0)
    return {"root": base_rel or ".", "entries": entries[:MAX_LIST_ENTRIES], "truncated": full}


def read_source(root: Path, relative: str, *, start_line: int = 1, max_lines: int = 400) -> dict[str, Any]:
    """读一段源码 ✓（带行号；一次最多 ``max_lines`` 行，避免把整个大文件塞进上下文）。"""
    rel = _normalize_relative(relative)
    file = resolve_inside(root, rel)
    _assert_not_denied(file, rel)
    _assert_text_file(file, rel)
    if not file.exists():
        raise RepairError(f"文件不存在：{rel}")
    content = _read_text_file(file, rel)
    lines = content.splitlines()
    begin = max(1, int(start_line or 1))
    span = max(1, min(int(max_lines or 400), 2000))
    window = lines[begin - 1: begin - 1 + span]
    return {
        "path": rel,
        "totalLines": len(lines),
        "startLine": begin,
        "lines": [{"n": begin + offset, "text": text} for offset, text in enumerate(window)],
    }


def search_source(root: Path, query: str, *, relative: str = ".", max_results: int = MAX_SEARCH_RESULTS) -> dict[str, Any]:
    """全文搜索 ✓ —— 纯字符串包含（不做正则：AI 给的正则经常漏转义 ✗）。"""
    needle = str(query or "")
    if not needle.strip():
        raise RepairError("query 不能为空")
    base_rel = "" if relative in (".", "", "./") else _normalize_relative(relative)
    base = Path(root).resolve() if not base_rel else resolve_inside(root, base_rel)

    results: list[dict[str, Any]] = []
    scanned = 0
    limit = max(1, min(int(max_results or MAX_SEARCH_RESULTS), MAX_SEARCH_RESULTS))

    for file in base.rglob("*"):
        if len(results) >= limit or scanned >= MAX_SEARCH_FILES:
            break
        if not file.is_file() or file.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part.lower() in DENY_SEGMENTS for part in file.relative_to(Path(root).resolve()).parts[:-1]):
            continue
        try:
            if file.stat().st_size > MAX_READ_BYTES:
                continue
            scanned += 1
            with open(file, "r", encoding="utf-8", newline="") as handle:
                for number, text in enumerate(handle, start=1):
                    if needle in text:
                        results.append({
                            "path": file.relative_to(Path(root).resolve()).as_posix(),
                            "line": number,
                            "text": _clip(text.rstrip("\r\n").strip(), 240),
                        })
                        if len(results) >= limit:
                            break
        except (OSError, UnicodeDecodeError):
            continue
    return {"query": needle, "count": len(results), "scanned": scanned, "results": results}


# ---------------------------------------------------------------- 提案


def make_proposal(
    root: Path,
    summary: Any,
    edits: Iterable[dict[str, Any]],
    *,
    allowed_prefixes: Sequence[str] = DEFAULT_ALLOWED_PREFIXES,
    max_files: int = MAX_PROPOSAL_FILES,
) -> dict[str, Any]:
    """把一组「找一段、换一段」编译成**待确认提案** ✓（不落盘，纯校验）。

    每条 edit：``{path, search, replace}`` 或新建 ``{path, create: true, replace}``。
    """
    prepared: list[dict[str, Any]] = []
    for edit in edits:
        rel = _normalize_relative(edit.get("path"))
        file = resolve_inside(root, rel)
        _assert_allowed_prefix(rel, allowed_prefixes)
        _assert_not_denied(file, rel)
        _assert_text_file(file, rel)

        if edit.get("create") is True:
            if file.exists():
                raise RepairError(f"文件已存在，不能按新建处理：{rel}")
            content = "" if edit.get("replace") is None else str(edit.get("replace"))
            if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
                raise RepairError(f"{rel} 新建内容过大")
            prepared.append({
                "path": rel, "isCreate": True, "oldContent": "", "newContent": content,
                "search": "", "replace": content, "oldPreview": "", "newPreview": _clip(content, 500),
            })
            continue

        if not file.exists():
            raise RepairError(f"文件不存在：{rel}")
        old_content = _read_text_file(file, rel)
        search = "" if edit.get("search") is None else str(edit.get("search"))
        if not search:
            raise RepairError(f"修改 {rel} 需要 search 原文")
        first = old_content.find(search)
        if first < 0:
            raise RepairError(f"{rel} 中找不到 search 原文")
        # ⚠️ 唯一命中：第二次还能找到就拒 —— 见模块头第 1 条
        if old_content.find(search, first + len(search)) >= 0:
            raise RepairError(f"{rel} 中 search 原文出现多次，请给更长的上下文")
        replace = "" if edit.get("replace") is None else str(edit.get("replace"))
        new_content = old_content[:first] + replace + old_content[first + len(search):]
        if len(new_content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise RepairError(f"{rel} 修改后文件过大")
        if new_content == old_content:
            raise RepairError(f"{rel} 没有产生变化")
        prepared.append({
            "path": rel, "isCreate": False, "oldContent": old_content, "newContent": new_content,
            "search": search, "replace": replace,
            "oldPreview": _clip(search, 500), "newPreview": _clip(replace, 500),
        })

    if not prepared:
        raise RepairError("方案里没有任何修改")
    if len(prepared) > max_files:
        raise RepairError(f"一次最多提议 {max_files} 个文件修改")

    return {
        "id": uuid.uuid4().hex,
        "summary": _clip(summary, 500) or "画布修复",
        "edits": prepared,
        "status": "pending",
    }


# ---------------------------------------------------------------- 应用 / 回滚


def apply_proposal(proposal: dict[str, Any], root: Path, backup_root: Path) -> dict[str, Any]:
    """应用提案 ✓ —— **先全量校验、再全量备份、最后才写**；写中途失败就地回滚 ⚠️。"""
    edits = proposal.get("edits") or []
    if not edits:
        raise RepairError("方案为空")

    plan: list[tuple[dict[str, Any], Path]] = []
    for edit in edits:
        file = resolve_inside(root, edit["path"])
        if edit.get("isCreate"):
            if file.exists():
                raise RepairError(f"文件已被创建，方案过期：{edit['path']}")
        else:
            if not file.exists():
                raise RepairError(f"文件已不存在，方案过期：{edit['path']}")
            current = _read_text_file(file, edit["path"])
            if current != edit.get("oldContent"):
                raise RepairError(f"文件已被其他操作修改，请重新生成方案：{edit['path']}")
        plan.append((edit, file))

    backup_dir = Path(backup_root) / str(proposal["id"])
    backup_dir.mkdir(parents=True, exist_ok=True)
    for edit, file in plan:
        if edit.get("isCreate"):
            continue
        target = backup_dir / edit["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)

    applied: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    try:
        for edit, file in plan:
            _write_text_file(file, edit["newContent"])
            written.append(edit)
            applied.append({"path": edit["path"], "isCreate": bool(edit.get("isCreate"))})
    except Exception:
        # ⚠️ 源没有这一步；写前校验过 ≠ 写不失败（磁盘满 / 文件被占用）⇒ 就地回滚已写的 ✓
        for edit in written:
            _restore_one(edit, root, backup_dir)
        raise
    return {"proposalId": proposal["id"], "backupDir": str(backup_dir), "applied": applied}


def rollback_proposal(proposal: dict[str, Any], root: Path, backup_root: Path) -> dict[str, Any]:
    """按备份把文件退回 ✓ —— 新建的文件删除，改过的文件从备份复制回去。"""
    backup_dir = Path(backup_root) / str(proposal["id"])
    restored: list[str] = []
    for edit in proposal.get("edits") or []:
        _restore_one(edit, root, backup_dir)
        restored.append(edit["path"])
    return {"proposalId": proposal.get("id"), "restored": restored}


def _restore_one(edit: dict[str, Any], root: Path, backup_dir: Path) -> None:
    file = resolve_inside(root, edit["path"])
    if edit.get("isCreate"):
        if file.exists():
            file.unlink()
        return
    backup_file = backup_dir / edit["path"]
    if not backup_file.exists():
        # ⚠️ 不静默跳过：没有备份就意味着无法保证回到原状，必须让调用方知道 ✗
        raise RepairError(f"缺少回滚备份：{edit['path']}")
    file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_file, file)
