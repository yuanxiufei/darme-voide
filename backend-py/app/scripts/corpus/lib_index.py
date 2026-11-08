"""知识库**增量**全文索引：补「盘上有、检索不到」的那批文件。

为什么需要第三个库
------------------
检索链路上现在有两块库：

* ``curated-screenplay-methods.sqlite`` —— 供应商产物（5403 份），**只读、不改、不重建**；
* ``library-ocr.sqlite`` —— 本仓 sidecar，只补「主库已登记但没抽出正文」的扫描件
  （``ocr_index.py`` 的 ``_pending`` 从主库 ``documents`` 取活）。

于是盘上还有一类文件**永远进不了检索**：主库压根没登记它们的格式。实测最典型的是
``.doc``（Word 97-2003 二进制）——CSV 索引里 **873 份**标着「格式暂不支持全文索引」，
而它们绝大多数是能抽出正文的剧本/剧作法。知识库里"有这本书"与"这本书能被搜到"是两件事，
这个脚本补的就是后者。

写进**本仓自己的** ``library-extra.sqlite``（与 OCR sidecar 同构：``documents`` ＋
fts5 ``chunks`` trigram），检索层按秩合并第三库、每条结果带 ``from="extra"``。
供应商那份仍然一个字不写。

抽文本按扩展名分派
------------------
=================  =========================================================
``.txt/.md/.srt``  直接解码（BOM → UTF-16 → UTF-8 → GB18030 回落）
``.doc``           ``antiword -m UTF-8.txt``（本机 Git/mingw64 自带；缺则整批跳过并明说）
                   文件路径打不开时**拷到 ASCII 临时路径再试一次**（见 ``_doc_text``）
``.docx``          zipfile 读 ``word/document.xml``，按 ``<w:p>`` 分段
``.pdf``           ``pypdfium2`` 抽**文字层**（12/12 实测；扫描件抽不满仍判 quarantine
                   交 ``ocr_index.py``）—— 选型凭据见 ``_pdf_text``
``.html/.htm``     ``HTMLParser`` 剥标签（``<script>/<style>`` 整段丢）
``.pptx``          zipfile 读 ``ppt/slides/slideN.xml`` 的 ``<a:t>``
``.epub/.chm/...`` 明说跳过（无可靠纯文本通道，别装作抽到了）
=================  =========================================================

质量门禁
--------
只判**明确坏**的：可打印率低于 ``MIN_PRINTABLE``（``antiword`` 抽到二进制垃圾、
编码猜错成乱码）⇒ ``status='quarantine'`` 且**一个块都不写**（坏文本进了 fts5，
检索时无法区分「书里就这么写」与「抽错了」）。状态名与 OCR sidecar 同名，
``script_library.search_methods`` 现成的「未完成文档不进检索」判据直接生效，不用改。

用法::

    python -X utf8 -m app.scripts.corpus.lib_index --stats
    python -X utf8 -m app.scripts.corpus.lib_index --all
    python -X utf8 -m app.scripts.corpus.lib_index --all --path-substr 历史戏曲剧本
    python -X utf8 -m app.scripts.corpus.lib_index --all --force --limit 50
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.services import script_library

#: 与 OCR sidecar 同一套切块口径（块 800 字、伪页 1500 字）：两库结果并排显示时，
#: 「命中在第几页」的口径必须一致，否则同一个剧本在两个库里页码对不上。
CHUNK_CHARS = 800
PAGE_CHARS = 1500
MIN_CHARS = 12

#: 可打印率下限。<0.85 意味着每 7 个字里就有 1 个是控制符/替换符 —— 那不是排版问题，
#: 是抽错了（``antiword`` 打在损坏的 .doc 上、或编码猜错）。正常中文剧本实测 ≥0.97。
MIN_PRINTABLE = 0.85

STATUS_EXTRA = "extra"
STATUS_QUARANTINE = "quarantine"  # 与 ocr sidecar 同名 ⇒ 检索层现成排除判据直接生效

TEXT_EXTS = (".txt", ".md", ".srt")
DOC_EXTS = (".doc",)
DOCX_EXTS = (".docx",)
RTF_EXTS = (".rtf",)  # 内置解析器已落地（见 ``_rtf_text``），不再是"无可靠通道"
#: 2026-11-07 起 PDF 归本脚本：抽**文字层**（``pypdfium2``），扫描件抽不满仍判
#: ``quarantine`` 交 ``ocr_index.py``。选型凭据见 ``_pdf_text`` 的对比表。
PDF_EXTS = (".pdf",)
HTML_EXTS = (".html", ".htm")
PPTX_EXTS = (".pptx",)

#: **单一真源**：「本脚本登记了抽文本通道」的全部扩展名。
#: 三处必须都读它，抄第二份就会分裂 —— 2026-11-07 实测过的原样事故：
#: ``_extract`` 已能抽 ``.pdf``，而 ``_candidates`` 的白名单还是手抄的旧四组
#: ⇒ 批量扫库时 25 份（19 份 PDF ＋ 6 份 HTML）被记成「未登记抽文本通道」静默跳过，
#: 只有界面上传那条路能用（两边行为不一致，且报错文案还是「这个格式不支持」）。
SUPPORTED_EXTS = (TEXT_EXTS + DOC_EXTS + DOCX_EXTS + RTF_EXTS
                  + PDF_EXTS + HTML_EXTS + PPTX_EXTS)

#: 明确不归本脚本的扩展名：写在这里是为了**输出里能说明白**，而不是静默漏掉。
HANDLED_ELSEWHERE = {
    ".csv": "登记表/清单，不是语料（收了会让「查询来源」命中一堆目录清单）",
    ".log": "运行日志，不是语料",
    ".epub": "无可靠纯文本通道，未做",
    ".ppt": "旧二进制 PPT（OLE2），无可靠纯文本通道；另存 .pptx 后归本脚本",
    ".chm": "无可靠纯文本通道，未做",
    ".mobi": "无可靠纯文本通道，未做",
    ".azw3": "无可靠纯文本通道，未做",
    ".xls": "表格，非剧本文本",
    ".xlsx": "表格，非剧本文本",
    ".jpg": "图片，属资产不属语料",
    ".png": "图片，属资产不属语料",
}

#: 与 ``services/script_library.EXTRA_DB`` **同一个文件** ✓（默认值手抄两处，故这里必须同步 ✗）。
#:
#: ⚠️ 2026-11-07 修漂移：原先默认写成 ``PROJECT_ROOT/data/ocr/library-extra.sqlite`` ——
#: 那是「派生索引还住在 ``data/``」时代的旧落点；2026-11-07 三只派生库已整体迁到 ``indexes/`` ✓，
#: 本文件当时没跟着改 ✗ ⇒ 这里建的库与后端读的库是**两个不同文件**，
#: 症状是「索引跑完了，检索还是说没有」✓✗（不报错，各写各的）。
EXTRA_DB = Path(os.environ.get("SCRIPT_LIBRARY_EXTRA_DB")
                or (PROJECT_ROOT / "indexes" / "library-extra.sqlite"))

_SKIP_TOP = ("90-隔离与待审", "99-索引与检索规则")
_SKIP_SUB = ("不进索引-原始与待审",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    name TEXT,
    extension TEXT,
    size INTEGER,
    modified_ns INTEGER,
    signature TEXT,
    pages INTEGER,
    status TEXT,
    gated INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    document_id UNINDEXED, path UNINDEXED, name, page UNINDEXED,
    chunk_no UNINDEXED, text, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""

_BREAK = "。！？；…\n"


# ===========================================================================
# 库与范围
# ===========================================================================


def _connect(*, writable: bool = False) -> sqlite3.Connection:
    if writable:
        EXTRA_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(EXTRA_DB.as_posix(), timeout=30)
        conn.executescript(_SCHEMA)
        return conn
    conn = sqlite3.connect(f"file:{EXTRA_DB.as_posix()}?mode=ro", uri=True, timeout=20)
    return conn


def _covered_split() -> tuple[set[str], set[str]]:
    """返回 ``(主库＋OCR 已覆盖, 本库已收录)``，两拨**必须分开**。

    主库 ``documents.path`` 是**建库机器上的旧盘符绝对路径**（形如 ``…\\电影剧本\\…``），
    不能直接比 ——
    走 ``script_library._on_disk`` 拿锚点右边那段，与 sidecar 的相对路径同域。

    合并成一个集合会让 ``--force`` 变成假开关：``_candidates`` 在算候选时就把「本库
    自己刚收的」当已覆盖剔掉，``main`` 里那道 ``--force`` 判断永远遇不到它们。实测
    ``--all --force`` 报「待补 0 份」，看着像"全都补完了"，其实重算了 0 份。
    """
    main_covered: set[str] = set()
    main = script_library.METHODS_DB
    if main.exists():
        try:
            conn = sqlite3.connect(f"file:{main.as_posix()}?mode=ro", uri=True, timeout=20)
            try:
                for (raw,) in conn.execute("SELECT path FROM documents"):
                    _, rel = script_library._on_disk(raw)  # noqa: SLF001 —— 同一套归一逻辑
                    if rel:
                        main_covered.add(rel)
            finally:
                conn.close()
        except sqlite3.Error as exc:  # noqa: BLE001 —— 主库读不了要说，别当成"全都没覆盖"
            print(f"!! 主库 documents 读取失败：{exc}（会重复收录已有正文）", flush=True)
    if Path(script_library.OCR_DB).exists():
        try:
            conn = sqlite3.connect(f"file:{Path(script_library.OCR_DB).as_posix()}?mode=ro",
                                   uri=True, timeout=20)
            try:
                main_covered |= {row[0] for row in conn.execute("SELECT path FROM documents")}
            finally:
                conn.close()
        except sqlite3.Error:
            pass
    extra_covered: set[str] = set()
    if Path(EXTRA_DB).exists():
        try:
            conn = sqlite3.connect(f"file:{Path(EXTRA_DB).as_posix()}?mode=ro", uri=True, timeout=20)
            try:
                extra_covered = {row[0] for row in conn.execute("SELECT path FROM documents")}
            finally:
                conn.close()
        except sqlite3.Error:
            pass
    return main_covered, extra_covered


def _candidates(*, substr: str = "", force: bool = False) -> tuple[list[dict], dict[str, int]]:
    """盘上待补的文件 ＋ 跳过原因的计数（跳过也要能报出来，不许静默）。

    ``force=True`` 时**本库已收录的也重算**（主库/OCR 覆盖的仍跳过：那些正文不归本库管）。
    """
    root = Path(script_library.LIBRARY_ROOT)
    main_covered, extra_covered = _covered_split()
    items: list[dict] = []
    skipped: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if parts[0] in _SKIP_TOP or (len(parts) > 1 and parts[1] in _SKIP_SUB):
            continue
        rel = path.relative_to(root).as_posix()
        if substr and substr not in rel:
            continue
        ext = path.suffix.lower()
        if ext in HANDLED_ELSEWHERE:
            skipped[f"{ext} {HANDLED_ELSEWHERE[ext]}"] = skipped.get(
                f"{ext} {HANDLED_ELSEWHERE[ext]}", 0) + 1
            continue
        if ext not in SUPPORTED_EXTS:
            skipped[f"{ext or '(无扩展名)'} 未登记抽文本通道"] = skipped.get(
                f"{ext or '(无扩展名)'} 未登记抽文本通道", 0) + 1
            continue
        if rel in main_covered:
            skipped[f"{ext} 主库/OCR 已覆盖"] = skipped.get(f"{ext} 主库/OCR 已覆盖", 0) + 1
            continue
        if rel in extra_covered and not force:
            reason = f"{ext} 本库已收录（--force 可重算）"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        stat = path.stat()
        items.append({"path": rel, "local": path, "name": path.name, "extension": ext,
                      "size": stat.st_size, "modifiedNs": stat.st_mtime_ns,
                      "signature": _signature(stat.st_size, stat.st_mtime_ns)})
    items.sort(key=lambda row: row["size"])  # 小的先跑：先看到成果，也先暴露抽文本的坑
    return items, skipped


def _signature(size: int, modified_ns: int) -> str:
    return hashlib.sha1(f"{size}:{modified_ns}".encode()).hexdigest()[:16]


# ===========================================================================
# 抽文本
# ===========================================================================


def _decode(data: bytes) -> tuple[str, str]:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace"), "utf-16"
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace"), "utf-8-replace"


def _antiword() -> str | None:
    # 候选目录只取**标准安装位**（Program Files 系 / ``LOCALAPPDATA``）＋ ``ANTIWORD_HOME``
    # 覆盖自定义安装；不写死任何盘符路径（换台机器就是废字，且那是本仓「扫描路径一律不写死」
    # 的硬口径）。原先那两条本机目录候选把反斜杠写成了两遍（转义错值 ⇒ 路径不成立），
    # 从没命中过 ⇒ 删掉零损失。
    exe = os.environ.get("ANTIWORD_EXE") or shutil.which("antiword")
    if exe and Path(exe).exists():
        return exe
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA"), os.environ.get("ANTIWORD_HOME")):
        if not base:
            continue
        for rel in ("Git/mingw64/bin/antiword.exe", "Git/usr/bin/antiword.exe"):
            cand = Path(base) / rel
            if cand.exists():
                return str(cand)
    return None


def _antiword_env(exe: str) -> dict[str, str]:
    """补 ``HOME`` / ``ANTWORDHOME``：antiword 靠它们找 ``UTF-8.txt`` 映射表。

    本机（Git/mingw64）实测：``HOME`` 没设时它每次先吐一行 ``I can't find the name of
    your HOME directory``，然后**碰运气**回落到编译期目录——找出映射表就正常，找不出就
    按 latin-1 吐问号。路径不猜：显式指到 exe 旁边的 ``share/antiword``，不存在就不设。
    """
    env = dict(os.environ)
    if not env.get("HOME"):
        env["HOME"] = str(Path.home())  # antiword 只认 HOME，不认 USERPROFILE
    if not env.get("ANTWORDHOME"):
        share = Path(exe).resolve().parent.parent / "share" / "antiword"
        if share.is_dir():
            env["ANTWORDHOME"] = str(share)
    return env


def _ascii_tempdir() -> str | None:
    """纯 ASCII 的临时目录（antiword 打不开非 ASCII 路径）。找不到就返回 ``None``。"""
    system_root = os.environ.get("SystemRoot") or os.environ.get("windir")
    system_temp = str(Path(system_root) / "Temp") if system_root else None
    for cand in (tempfile.gettempdir(), system_temp, os.environ.get("TEMP")):
        if not cand:
            continue
        try:
            if not str(cand).isascii():
                continue
            base = Path(cand)
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return str(base)
    return None


def _run_antiword(exe: str, target: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    #: ``-m UTF-8.txt`` 是必须的：不带映射表时 antiword 按 latin-1 吐，中文全成问号。
    return subprocess.run([exe, "-m", "UTF-8.txt", str(target)],
                          capture_output=True, timeout=120, env=env)


def _antiword_text(path: Path) -> str:
    """走 antiword 本体：两道坎都要过，缺一个就是整批文件"有书搜不到"。

    ① **映射表**：``-m UTF-8.txt`` 得让 antiword 找得到（``home/ANTWORDHOME``，见
       ``_antiword_env``），否则中文按 latin-1 出成问号（还是 rc=0，更难发现）。
    ② **路径**：mingw 版 antiword 的文件名转码会在部分非 ASCII 字符上失败。实测
       ``约翰•沁孤《骑马下海的人》.doc`` 的 ``•``(U+2022) 必炸 —— 报
       ``I can't open '<乱码路径>'``，同一份文件拷到 ASCII 路径就正常吐出 22158 字。
       所以**只在第一次失败后**才多花一次 I/O 拷进 ASCII 临时目录重试；两次都失败
       才算真失败，并把两次 stderr 都带上（别把"路径打不开"报成"文件坏了"）。
    """
    exe = _antiword()
    if not exe:
        raise RuntimeError("缺 antiword")
    env = _antiword_env(exe)
    proc = _run_antiword(exe, path, env)
    if proc.returncode == 0:
        return proc.stdout.decode("utf-8", "replace")
    tmpdir = _ascii_tempdir()
    retry: subprocess.CompletedProcess | None = None
    if tmpdir:
        with tempfile.TemporaryDirectory(prefix="aw-", dir=tmpdir) as td:
            copy = Path(td) / "doc.doc"
            shutil.copyfile(path, copy)
            retry = _run_antiword(exe, copy, env)
    if retry is not None and retry.returncode == 0:
        return retry.stdout.decode("utf-8", "replace")
    if retry is None:
        raise RuntimeError(f"antiword rc={proc.returncode} {proc.stderr[:120]!r}"
                           "（文件名含非 ASCII 字符，但找不到纯 ASCII 临时目录可重试）")
    raise RuntimeError(f"antiword rc={proc.returncode} {proc.stderr[:120]!r}"
                       f"（改 ASCII 路径重试仍失败 rc={retry.returncode} "
                       f"{retry.stderr[:120]!r}）")


# ---------------------------------------------------------------------------
# .doc 的内置通路（不依赖任何外部转换器）
#
# 实测本库 873 份 ``.doc`` 里，光靠 antiword 会漏掉一大片，且漏法各不相同：
#
# * **50 份其实是 RTF**（文件头 ``{\rtf1\ansi\ansicpg936``，改了扩展名而已）；
# * **77 份是 WPS 写的 Word**（45 份带 ``Wps`` 流）或结构较老的 Word —— antiword
#   一律回 ``is not a Word Document`` / ``version of Word before Word 6``，
#   但它们的 OLE 里 ``WordDocument`` 流是完好的，正文一个字没少。
#
# 这两种都在 Python 里自己解，不装 LibreOffice / pandoc（本机一个都没有），也不改
# 「装不上就算了吧」的旧口径：能抽的必须抽出来，抽不出的要说出为什么。
# ---------------------------------------------------------------------------

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_RTF_MAGIC = b"{\\rtf"


def _ole_stream(raw: bytes, want: str) -> bytes | None:
    """OLE2 复合文档（``d0cf11e0``）里取一条流；没有就 ``None``。

    只做 ``documents`` 这一层需要的事：FAT/DIFAT → 目录 → 流。mini 流（<4096 字节）
    也要过，因为 WPS 有时把 ``WordDocument`` 塞进 mini fat。
    """
    if not raw.startswith(_OLE_MAGIC) or len(raw) < 516:
        return None
    sect_shift, mini_shift = struct.unpack_from("<HH", raw, 0x1E)
    nfat, dirstart = struct.unpack_from("<II", raw, 0x2C)
    ministart, difstart, ndif = struct.unpack_from("<III", raw, 0x3C)
    if sect_shift not in (9, 12) or mini_shift != 6:
        return None  # 头都读歪了，别拿垃圾字段往下走
    size = 1 << sect_shift
    mini_size = 1 << mini_shift
    #: 上限一律按「这个文件最多能有多少扇区」算：语法上 FAT 可能指回自己，
    #: 也可能给出一串越界但彼此不重复的扇区号。真文件里 chain 只会更短。
    max_sectors = len(raw) // size + 2

    def sector(n: int) -> bytes:
        start = 512 + n * size
        return raw[start:start + size]

    fat_sectors = list(struct.unpack_from("<109I", raw, 0x4C))[:min(nfat, 109)]
    nxt, seen_dif = difstart, set()
    while nxt < 0xFFFFFFF0 and len(seen_dif) < max_sectors and nxt not in seen_dif:
        seen_dif.add(nxt)
        block = sector(nxt)
        if len(block) < size:
            break
        per = size // 4 - 1
        fat_sectors += list(struct.unpack_from(f"<{per}I", block, 0))[:max(0, nfat - len(fat_sectors))]
        nxt = struct.unpack_from("<I", block, size - 4)[0]
    fat: list[int] = []
    for idx in [x for x in fat_sectors if x < max_sectors][:nfat]:
        fat += list(struct.unpack_from(f"<{size // 4}I", sector(idx), 0))

    def chain(start: int) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        cur = start
        while cur < max_sectors and cur not in seen and len(out) < max_sectors:
            seen.add(cur)
            out.append(cur)
            cur = fat[cur] if cur < len(fat) else 0xFFFFFFFE
        return out

    if dirstart >= max_sectors:
        return None
    directory = b"".join(sector(n) for n in chain(dirstart))
    entries: list[tuple[str, int, int, int]] = []
    for i in range(len(directory) // 128):
        slot = directory[i * 128:(i + 1) * 128]
        name_len = struct.unpack_from("<H", slot, 0x40)[0]
        name = slot[:max(0, name_len - 2)].decode("utf-16-le", "replace")
        entries.append((name, slot[0x42],
                        struct.unpack_from("<I", slot, 0x74)[0],
                        struct.unpack_from("<I", slot, 0x78)[0]))
    root = next((e for e in entries if e[1] == 5), None)
    if root is None:
        return None
    target = next((e for e in entries if e[1] == 2 and e[0] == want), None)
    if target is None:
        return None
    name, _, start, count = target
    if count >= 4096 or ministart >= max_sectors:
        return b"".join(sector(n) for n in chain(start))[:count]
    mini = b"".join(sector(n) for n in chain(root[2]))
    out = b""
    seen: set[int] = set()
    cur = start
    while cur not in seen and len(out) < count:
        seen.add(cur)
        out += mini[cur * mini_size:(cur + 1) * mini_size]
        cur = fat[cur] if cur < len(fat) else 0xFFFFFFFE
    return out[:count]


#: FIB 的 ``lid``（语言 id）→ 压缩分片用的 ANSI 代码页。Word 的 8 位压缩分片是按
#: **文档语言**存字节的，猜不出来：cp936 与 cp949 的字节区间几乎完全重叠，两边都能
#: 「成功」解出字来，只是一边是真字、一边是乱码。所以有 ``lid`` 就用 ``lid``。
_LID_CP = {
    0x0804: "cp936", 0x1004: "cp936", 0x0404: "cp950", 0x0C04: "cp950", 0x1404: "cp950",
    0x0411: "cp932", 0x0412: "cp949", 0x0419: "cp1251", 0x0402: "cp1251", 0x0422: "cp1251",
    0x0408: "cp1253", 0x041F: "cp1254", 0x040D: "cp1255", 0x0401: "cp1256",
    0x0425: "cp1257", 0x0427: "cp1257", 0x041E: "cp874",
    0x0415: "cp1250", 0x0405: "cp1250", 0x040E: "cp1250", 0x0418: "cp1250",
    0x041A: "cp1250", 0x041B: "cp1250", 0x0424: "cp1250",
}


def _decode_compressed(data: bytes, codepage: str | None = None) -> str:
    """Word 的「压缩分片」是 8 位字符，码表跟文档 ANSI 代码页走（见 ``_LID_CP``）。

    有 ``lid`` 派生的 ``codepage`` 就照它解；没有（或就是 cp1252）才用高字节占比兜底：
    >30% 高位字节才试 cp936 —— 先前一律 cp1252 会把中文解成乱码，而这种乱码**每格都
    可打印**，可打印率闸门拦不住。
    """
    if codepage and codepage != "cp1252":
        return data.decode(codepage, "replace")
    if data and sum(1 for b in data if b >= 0x80) / len(data) > 0.3:
        try:
            return data.decode("cp936")
        except UnicodeDecodeError:
            pass
    return data.decode("cp1252", "replace")


def _clean_doc_text(text: str) -> str:
    """Word 正文里的控制符 → 可读文本（``\\r`` 段落、``\\x07`` 单元格、``\\x01`` 图片锚点）。"""
    text = text.replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
    text = text.replace("\x07", "\t")
    text = re.sub(r"[\x00-\x08\x0e-\x1f]", "", text)
    return text


def _doc_ole_text(raw: bytes) -> str:
    """Word FIB ＋ 分片表（CLX/Pcdt）直读正文：不经过 antiword，也就不受它挑格式。

    ``nFib>=191`` 的 Word 全走这条：``WordDocument`` 流头是 FIB，``fcClx/lcbClx``
    指向 ``0Table``/``1Table`` 里的分片表；每个分片给一段 CP 区间与 fc，``fc`` 的
    bit30 表示「8 位压缩」还是 UTF-16LE。
    """
    word = _ole_stream(raw, "WordDocument")
    if not word or len(word) < 0x01AA:
        raise RuntimeError("OLE 里没有可用的 WordDocument 流")
    if struct.unpack_from("<H", word, 0)[0] != 0xA5EC:
        raise RuntimeError("WordDocument 流头不是 FIB（wIdent≠0xA5EC）")
    flags = struct.unpack_from("<H", word, 0x0A)[0]
    codepage = _LID_CP.get(struct.unpack_from("<H", word, 0x06)[0])
    table = _ole_stream(raw, "1Table" if flags & 0x0200 else "0Table")
    if not table:
        raise RuntimeError("缺 Table 流（分片表就在里面）")
    fc_clx, lcb_clx = struct.unpack_from("<II", word, 0x01A2)
    clx = table[fc_clx:fc_clx + lcb_clx]
    pos = 0
    while pos < len(clx) and clx[pos] == 1:  # Prc：修订用，跳过
        cb = struct.unpack_from("<H", clx, pos + 1)[0]
        pos += 3 + cb
    if pos >= len(clx) or clx[pos] != 2:
        raise RuntimeError("CLX 里没有 Pcdt（拿不到分片表）")
    lcb = struct.unpack_from("<I", clx, pos + 1)[0]
    plc = clx[pos + 5:pos + 5 + lcb]
    pieces = (len(plc) - 4) // 12
    if pieces <= 0:
        raise RuntimeError("分片表为空")
    cps = struct.unpack_from(f"<{pieces + 1}I", plc, 0)
    parts: list[str] = []
    for k in range(pieces):
        fc = struct.unpack_from("<I", plc, 4 * (pieces + 1) + k * 8 + 2)[0]
        chars = cps[k + 1] - cps[k]
        if chars <= 0:
            continue
        if fc & 0x40000000:
            start = (fc & 0x3FFFFFFF) // 2
            parts.append(_decode_compressed(word[start:start + chars], codepage))
        else:
            start = fc & 0x3FFFFFFF
            parts.append(word[start:start + chars * 2].decode("utf-16-le", "replace"))
    return _clean_doc_text("".join(parts))


#: RTF 里这些 destination 是元数据/图片/编号，不是正文，整组丢掉。
#: 注意**不丢** ``field``（字段外层）——只丢 ``fldinst``（域指令），正文常在 ``fldrslt`` 里。
_RTF_DROP = {
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "objdata", "header",
    "footer", "headerl", "headerr", "headerf", "footerl", "footerr", "footerf", "footnote",
    "fldinst", "listtable", "listoverridetable", "listtext", "rsidtbl", "generator",
    "xmlnstbl", "themedata", "colorschememapping", "datastore", "latentstyles", "shppict",
    "nonshppict", "bkmkstart", "bkmkend", "revtbl", "filetbl", "pgdsctbl", "pntext",
    "pntxta", "pntxtb", "pnseclvl", "wgrffmtfilter", "userprops", "docvar",
}
#: 换行类控制字（段落/软回车/分节/分页）。
_RTF_NEWLINE = {"par", "line", "row", "sect", "page"}
_RTF_TOKEN = re.compile(
    r"\\(?P<word>[a-zA-Z]{1,32})(?P<arg>-?\d{1,10})?[ ]?"
    r"|\\'(?P<hex>[0-9a-fA-F]{2})"
    r"|\\(?P<star>\*)"
    r"|\\(?P<sym>[^a-zA-Z])"
    r"|(?P<open>\{)|(?P<close>\})"
    r"|(?P<plain>[^\\{}\r\n]+)"
    r"|(?P<eol>[\r\n]+)"
)


#: RTF 的 ``\fcharsetN`` → 代码页。这个映射是**必需**的而不是锦上添花：
#: 本库 50 份中文 RTF 里混着韩国剧本，文档头声明 ``\ansicpg936``，正文却用
#: ``\f1``(fcharset129) 排韩文、``\f0``(fcharset134) 排中文 —— 只按文档级码表解，
#: 韩文全成 GBK 乱码（而且**每个字都可打印**，可打印率闸门拦不住）。
_RTF_CHARSET_CP = {
    0: "cp1252", 2: "cp1252", 77: "cp932", 128: "cp932", 129: "cp949", 130: "cp1361",
    134: "cp936", 136: "cp950", 161: "cp1253", 162: "cp1254", 163: "cp1258",
    177: "cp1255", 178: "cp1256", 186: "cp1257", 204: "cp1251", 222: "cp874",
    238: "cp1250", 255: "cp437",
}


def _rtf_charset_codepages(text: str, default: str) -> dict[int, str]:
    """从 ``{\\fonttbl ...}`` 里读 ``\\fN`` → ``\\fcharset M`` 的对应表。

    字体名本身常是 ``\'hh`` 字节（``{\\f0 ... \\fcharset134 \\'cb\\'ce\\'cc\\'e5;}``
    就是「宋体」），所以这里只认括号内、不含嵌套花括号的条目，不做全文猜。
    """
    found = re.search(r"\{\\fonttbl", text)
    if not found:
        return {}
    table: dict[int, str] = {}
    names = re.compile(r"\{\\f(\d+)\b([^{}]*)\}")
    for m in names.finditer(text, found.start()):
        charset = re.search(r"\\fcharset(\d+)", m.group(2))
        if charset:
            table[int(m.group(1))] = _RTF_CHARSET_CP.get(int(charset.group(1)), default)
    return table


def _rtf_text(raw: bytes) -> str:
    """RTF → 正文。

    四条 RTF 里最容易踩的规矩都在这里落地：``\\'hh`` 的码表**先按当前字体的
    ``\\fcharset`` 走**（同文档混排中日韩时必须这样，见 ``_RTF_CHARSET_CP``），没有字体
    信息才退回文档头的 ``\\ansicpgNNN``；``\\uN`` 后面跟着 ``\\ucN`` 声明的 N 个
    **回退字符**必须吃掉，否则正文里会多出一串 ``?``；``\\*`` 起头的组是"可忽略
    destination"，整组不当正文。
    """
    text = raw.decode("latin-1")
    declared = re.search(r"\\ansicpg(\d+)", text)
    default_cp = f"cp{declared.group(1)}" if declared else "cp1252"
    font_cp = _rtf_charset_codepages(text, default_cp)
    codepage = default_cp
    stack: list[tuple[bool, int, int, str]] = []
    drop, uc, skip_left = False, 1, 0
    out: list[str] = []
    pending = bytearray()

    def flush() -> None:
        if pending:
            out.append(pending.decode(codepage, "replace"))
            pending.clear()

    pos = 0
    while pos < len(text):
        m = _RTF_TOKEN.match(text, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        if m.group("open"):
            stack.append((drop, uc, skip_left, codepage))
        elif m.group("close"):
            flush()
            if stack:
                drop, uc, skip_left, codepage = stack.pop()
        elif m.group("star"):
            drop = True
        elif m.group("hex"):
            if drop:
                continue
            if skip_left > 0:  # \uN 的 ANSI 回退字节，吃掉
                skip_left -= 1
                continue
            pending.append(int(m.group("hex"), 16))
        elif m.group("word"):
            flush()
            word, arg = m.group("word"), m.group("arg")
            if word == "uc":
                uc = int(arg or 1)
                continue
            if word == "f" and arg is not None:  # 换字体 ⇒ 换码表（先 flush 再换）
                codepage = font_cp.get(int(arg), default_cp)
                continue
            if drop:
                continue
            if word in _RTF_NEWLINE:
                out.append("\n")
            elif word in ("tab", "cell"):
                out.append("\t")
            elif word == "emdash":
                out.append("—")
            elif word == "endash":
                out.append("–")
            elif word == "u" and arg is not None:
                out.append(chr(int(arg) % 0x10000))
                skip_left = uc
            elif word in _RTF_DROP:
                drop = True
        elif m.group("sym"):
            flush()
            if drop:
                continue
            sym = m.group("sym")
            if sym in "\\{}":
                out.append(sym)
            elif sym == "~":
                out.append(" ")
        elif m.group("plain"):
            flush()
            if drop:
                continue
            chunk = m.group("plain")
            if skip_left:
                chunk = chunk[skip_left:]
                skip_left = max(0, skip_left - len(m.group("plain")))
            if chunk:
                out.append(chunk)
    flush()
    body = _clean_doc_text("".join(out))
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def _doc_text(path: Path) -> tuple[str, str]:
    """``.doc`` → 正文：**先认内容，再认扩展名**。

    扩展名在这一层没有信用 —— 本库实测 873 份 ``.doc`` 里混着 50 份 RTF、77 份
    antiword 打不开的 Word（WPS 写的、或结构偏老）。分派顺序：

    1. ``{\\rtf`` ⇒ 内置 RTF 解（antiword 从来不解 RTF）；
    2. OLE ⇒ 先 antiword（排版还原最好），**失败**再走内置 FIB 分片表；
    3. 都不是 ⇒ 如实报「不是 Word 文档也不是 RTF」。

    antiword 缺失不再等于「.doc 整批跳过」：有内置通路就照抽，缺工具只是少了一条路。
    """
    raw = path.read_bytes()
    if raw.startswith(_RTF_MAGIC):
        return _rtf_text(raw), "rtf"
    if not raw.startswith(_OLE_MAGIC):
        raise RuntimeError(f"既不是 OLE2 也不是 RTF（头 {raw[:8].hex()}）")
    antiword_error = ""
    if _antiword():
        try:
            return _antiword_text(path), "antiword"
        except RuntimeError as exc:  # noqa: BLE001 —— 换内置通路，理由要留着
            antiword_error = f"antiword: {exc}"
    try:
        return _doc_ole_text(raw), "doc-ole"
    except RuntimeError as exc:
        raise RuntimeError(f"{antiword_error}；内置 FIB 解析也失败：{exc}") from exc


def _docx_text(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", "replace")
    out: list[str] = []
    for para in re.split(r"</w:p>", xml):
        runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S)
        if runs:
            out.append(html.unescape("".join(runs)))
        elif "<w:br" in para:
            out.append("")
    return "\n".join(out), "docx-xml"


class _HtmlStripper(HTMLParser):
    """``<script>/<style>`` 整段丢、块级标签变换行 —— 只留人读得到的字。

    不用正则剥标签：``<a href="x>y">`` 与 ``<td>`` 里的 ``<`` 都会让正则错位，
    而 ``HTMLParser`` 自带容错（库里 6 份 HTML 是从网页另存的剧本，闭包不齐）。
    """

    _SKIP = ("script", "style")
    _BLOCK = ("p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4",
              "h5", "h6", "section", "article", "pre", "blockquote", "table")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        #: ⚠️ 纯空白也要留 —— 行内标签两侧的空格是**词的一部分**：
        #: ``<p>INT. HOTEL <i>— DAY</i></p>`` 若把 ``"HOTEL "`` 的尾空格 strip 掉，
        #: 抽出来就是 ``HOTEL— DAY``（英文剧本整篇词粘连，BM25 再也切不出词）。
        #: 这里只管攒，行首行尾的空白在 :func:`_html_text` 里统一收。
        if not self._skip:
            self.parts.append(data)


def _html_text(path: Path) -> tuple[str, str]:
    """``.html/.htm`` → 正文（剥标签）。网页存下来的剧本靠这条进检索。"""
    text, enc = _decode(path.read_bytes())
    stripper = _HtmlStripper()
    stripper.feed(text)
    stripper.close()
    raw = "".join(stripper.parts)
    #: 行内空白压成一个、行首行尾收干净、空行丢掉（块级标签已经各占一行了）。
    lines = [re.sub(r"[ \t\u3000\xa0]+", " ", line).strip() for line in raw.split("\n")]
    return "\n".join(line for line in lines if line), f"html-{enc}"


def _pptx_text(path: Path) -> tuple[str, str]:
    """``.pptx`` → 正文：``ppt/slides/slideN.xml`` 里按 ``<a:t>`` 抽文字。

    与 ``_docx_text`` 同一条路数（zipfile + XML 正则），不引 python-pptx：
    抽文字只需要 ``<a:t>`` 一种标签（属性容错，见下），多引一个库只为读几段字不划算。
    纯图 PPT（讲座海报）抽不出字 ⇒ 如实报错，由 ``_build`` 判 quarantine。
    """
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        names.sort(key=lambda n: int(re.search(r"slide(\d+)\.xml$", n).group(1)))
        out: list[str] = []
        for name in names:
            xml = zf.read(name).decode("utf-8", "replace")
            #: 属性容错：schema 上 ``a:t`` 不带属性，但见过产出方写
            #: ``<a:t xml:space="preserve">`` —— 与 ``_docx_text`` 的 ``<w:t[^>]*>`` 同口径。
            runs = re.findall(r"<a:t(?:\s[^>]*)?>(.*?)</a:t>", xml, re.S)
            if runs:
                out.append("\n".join(html.unescape(run) for run in runs))
    body = "\n\n".join(out)
    if not body.strip():
        raise RuntimeError(f"{len(names)} 页 slide 里没抽出文字（可能是纯图 PPT）")
    return body, "pptx-xml"


#: PDF 两端对齐的副产物：汉字之间被塞进单空格（**只删 CJK 与 CJK 标点两侧的**，
#: 英文单词间空格一个不动）。这不是排版洁癖，是检索失效 —— 实测库内三份 PDF：
#: 《证人席》标题抽成 ``证 人 席``（搜「证人席」命中 0）、《纳德和西敏》269 处
#: （``世 界/电 影``）、川端论文 2070 处（``关于电影 与文学的一 点思索``）。
_CJK_GAP = re.compile(
    r"(?<=[\u4e00-\u9fff])[ \u3000]+(?=[\u4e00-\u9fff\u3000-\u303f\uff01-\uff60])"
    r"|(?<=[\u3000-\u303f\uff01-\uff60])[ \u3000]+(?=[\u4e00-\u9fff])"
)


def _cjk_deglue(text: str) -> str:
    """去掉汉字/中文标点两侧的填充空格。**只给 PDF 通道用**。

    不并进 ``_tidy_text``：那条路是 ``.doc/.docx/.txt`` 共用的，改它会让已入库
    的 873 份 ``.doc`` 摘要口径一起漂（台账按字节/摘要核验，会集体失配）。
    """
    return _CJK_GAP.sub("", text)


def _pdf_text(path: Path) -> tuple[str, str]:
    """``.pdf`` → 正文：只抽**文字层**，OCR 仍归 ``ocr_index.py``。

    为什么是 ``pypdfium2`` 而不是另外两条现成路 —— 2026-11-07 用库内 12 份真实
    剧本/期刊 PDF 实测（同批、同机、``random.seed(20261107)``）：

    ==================  ==========  ==========  ================================
    通道                成功/12     平均耗时    实测问题
    ==================  ==========  ==========  ================================
    ``pdfminer.six``    9/12        0.5–3.8 s   **表格化**：每行套 ``| … |``，剧本
                                              正文被切成单元格（English 剧本
                                              39.8 万字里大半是管道符与填充空格）；
                                              3 份非标 PDF 直接抛 ``No /Root object``
    ``PyMuPDF``         12/12       0.08 s      **AGPL-3.0** —— 进主链路即传染
    ``pypdfium2``       12/12       0.08 s      Apache-2.0 / BSD-3，输出干净行
    ==================  ==========  ==========  ================================

    所以这里只认 pypdfium2（本仓主依赖唯一新增）；``markitdown`` 那套更适合
    Office/HTML，不用在 PDF 上 —— 它的表格化对"按行切块 + BM25/trigram"是负收益。

    扫描件（如 290 字的图册）抽不满 ``MIN_CHARS`` ⇒ ``_build`` 判 quarantine，
    与旧口径一致：库里"有这本书"和"这本书能被搜到"仍是两件事。
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover —— 主依赖缺失属环境问题
        raise RuntimeError("缺 pypdfium2（pip install pypdfium2），PDF 文字层抽不出") from exc
    try:
        doc = pdfium.PdfDocument(path)
    except Exception as exc:  # noqa: BLE001 —— 打包成一句人话，别让调用方读 pdfium 内部
        raise RuntimeError(f"pypdfium2 打不开这份 PDF：{exc}") from exc
    pages: list[str] = []
    try:
        for page in doc:
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
    finally:
        doc.close()
    body = _cjk_deglue("\n\x0c\n".join(pages)).strip()
    if not body:
        raise RuntimeError(f"{len(pages)} 页都没有文字层（扫描件，需走 OCR 线）")
    return body, "pypdfium2"


def _extract(item: dict) -> tuple[str, str]:
    ext = item["extension"]
    path: Path = item["local"]
    if ext in DOC_EXTS:
        return _doc_text(path)
    if ext in DOCX_EXTS:
        return _docx_text(path)
    if ext in RTF_EXTS:
        return _rtf_text(path.read_bytes()), "rtf"
    if ext in PDF_EXTS:
        return _pdf_text(path)
    if ext in HTML_EXTS:
        return _html_text(path)
    if ext in PPTX_EXTS:
        return _pptx_text(path)
    data = path.read_bytes()
    text, enc = _decode(data)
    if ext in (".md", ".csv", ".srt", ".log"):
        text = re.sub(r"\r\n?", "\n", text)
    return text, enc


# ===========================================================================
# 切块
# ===========================================================================


def _printable_ratio(text: str) -> float:
    """可打印率。**空白一律算可打印** —— 这是判据本身的坑，不是排版问题。

    ``str.isprintable()`` 对 ``U+3000``（全角空格）与 ``U+00A0``（不换行空格）返回
    ``False``（它们属 Separator 类），而中文剧本整段靠全角空格缩进：实测《生死场》
    33884 字里 5833 个正是这类空白，按 ``isprintable`` 算得 0.83 ⇒ 判隔离、一块不写，
    可正文一个字都没坏。反过来 ``U+FFFD``（解码替换符）``isprintable`` 为 ``True``，
    它才是真该扣分的 —— 扣分项显式点名，别指望 ``isprintable`` 替我们分。
    """
    if not text:
        return 0.0
    bad = sum(1 for ch in text
              if ch == "\ufffd" or not (ch.isprintable() or ch.isspace()))
    return 1 - bad / len(text)


def _tidy_text(text: str) -> str:
    """入库前的规范化：三件都属"看着不像问题、检索时才露头"。

    ① ``U+00A0``（antiword 把行尾空格吐成不换行空格）与零宽空格都是排版噪声，
       留在正文里会让摘录片段出现看不见的字符；
    ② 控制符（``\\x00-\\x08`` 等）既拉低可打印率，又会让 fts5 的 trigram 断词；
    ③ ``\\r\\n`` 混用会让切块时 ``line.strip()`` 在两条通道上行为不一致。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _pages(text: str) -> list[str]:
    body = re.sub(r"[ \t\u3000]+", " ", text).strip()
    return [body[i:i + PAGE_CHARS] for i in range(0, len(body), PAGE_CHARS)] or [""]


def _chunks(pages: list[str]) -> list[tuple[int, int, str]]:
    """``[(page, chunk_no, text)]``；``page`` 是块**起点**所在伪页（1 起）。"""
    rows: list[tuple[int, int, str]] = []
    buffer = ""
    buffer_page = 1
    for page_no, page in enumerate(pages, start=1):
        for line in page.split("\n"):
            piece = line.strip()
            if not piece:
                continue
            if not buffer:
                buffer_page = page_no
            buffer = f"{buffer}\n{piece}" if buffer else piece
            while len(buffer) >= CHUNK_CHARS:
                cut = CHUNK_CHARS
                window = max(buffer.rfind(ch, CHUNK_CHARS - 200, CHUNK_CHARS) for ch in _BREAK)
                if window > 0:
                    cut = window + 1
                rows.append((buffer_page, len(rows) + 1, buffer[:cut].strip()))
                buffer = buffer[cut:].strip()
                buffer_page = page_no
    if len(buffer) >= MIN_CHARS:
        rows.append((buffer_page, len(rows) + 1, buffer.strip()))
    return rows


def _build(item: dict) -> dict:
    started = time.perf_counter()
    text, engine = _extract(item)
    text = _tidy_text(text or "")
    ratio = _printable_ratio(text)
    status = STATUS_EXTRA if (len(text) >= MIN_CHARS and ratio >= MIN_PRINTABLE) else STATUS_QUARANTINE
    pages = _pages(text) if status == STATUS_EXTRA else []
    return {"text": text, "engine": engine, "status": status, "ratio": ratio,
            "chars": len(text),
            "pages": len(pages), "chunks": _chunks(pages) if status == STATUS_EXTRA else [],
            "seconds": time.perf_counter() - started}


def _save(item: dict, outcome: dict) -> None:
    conn = _connect(writable=True)
    try:
        with conn:
            conn.execute("DELETE FROM chunks WHERE document_id IN "
                         "(SELECT id FROM documents WHERE path = ?)", [item["path"]])
            conn.execute("DELETE FROM documents WHERE path = ?", [item["path"]])
            cursor = conn.execute(
                "INSERT INTO documents (path, name, extension, size, modified_ns, "
                "signature, pages, status, gated) VALUES (?,?,?,?,?,?,?,?,1)",
                (item["path"], item["name"], item["extension"], item["size"],
                 item["modifiedNs"], item["signature"], outcome["pages"], outcome["status"]),
            )
            doc_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO chunks (document_id, path, name, page, chunk_no, text) "
                "VALUES (?,?,?,?,?,?)",
                [(doc_id, item["path"], item["name"], page, no, body)
                 for page, no, body in outcome["chunks"]],
            )
            conn.execute("INSERT INTO meta (k, v) VALUES ('engine', ?) "
                         "ON CONFLICT(k) DO UPDATE SET v = excluded.v", (outcome["engine"],))
            conn.execute("INSERT INTO meta (k, v) VALUES ('updatedAt', ?) "
                         "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                         (time.strftime("%Y-%m-%d %H:%M:%S"),))
    finally:
        conn.close()


# ===========================================================================
# 报表 / 入口
# ===========================================================================


def stats() -> dict:
    available = EXTRA_DB.exists()
    data = {"available": available, "db": EXTRA_DB.as_posix()}
    if not available:
        return data
    conn = _connect()
    try:
        data["documents"] = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        data["chunks"] = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        data["byStatus"] = dict(conn.execute(
            "SELECT status, count(*) FROM documents GROUP BY status").fetchall())
        data["byExtension"] = dict(conn.execute(
            "SELECT extension, count(*) FROM documents GROUP BY extension").fetchall())
    finally:
        conn.close()
    return data


def prune(*, apply: bool = False) -> dict:
    """清掉**口径或盘上情况已经变了**的记录 —— ``--all`` 只增不删，老记录会永远留着。

    三类都属自相矛盾，留着比删掉更坏：

    * 扩展名已登记「不归本库」（``HANDLED_ELSEWHERE``）：早期把 ``.csv`` 当文本收进来过，
      那批是目录清单，检索时会以「来源清单」的名义命中；
    * 路径落在 ``_SKIP_TOP``/``_SKIP_SUB``：判据排除了，库却还收着；
    * 文件已不在盘上：``--stats`` 报的份数比真实候选多，报告本身就是谎话。

    默认只报不删，``apply=True`` 才动库（``documents`` 与 ``chunks`` 一起删）。
    """
    root = Path(script_library.LIBRARY_ROOT)
    removed: list[tuple[str, str]] = []
    if not EXTRA_DB.exists():
        return {"removed": removed, "applied": apply}
    conn = _connect(writable=apply)
    try:
        rows = conn.execute("SELECT id, path, extension FROM documents").fetchall()
        doomed: list[int] = []
        for doc_id, rel, ext in rows:
            parts = Path(rel).parts
            if parts and (parts[0] in _SKIP_TOP
                          or (len(parts) > 1 and parts[1] in _SKIP_SUB)):
                why = "落在判据排除的目录里"
            elif ext in HANDLED_ELSEWHERE:
                why = f"扩展名已登记不归本库：{HANDLED_ELSEWHERE[ext]}"
            elif ext not in SUPPORTED_EXTS:
                why = "扩展名不在本库抽文本通道内"
            elif not (root / rel).exists():
                why = "盘上已找不到该文件"
            else:
                continue
            doomed.append(doc_id)
            removed.append((rel, why))
        if apply:
            for doc_id in doomed:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
    finally:
        conn.close()
    return {"removed": removed, "applied": apply}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    argv = sys.argv
    if "--prune" in argv:
        report = prune(apply="--apply" in argv)
        for rel, why in report["removed"]:
            print(f"        {'删' if report['applied'] else '该删'} {rel}：{why}")
        verb = "已删除" if report["applied"] else "待删除（加 --apply 才动库）"
        print(f"[extra] {verb} {len(report['removed'])} 份")
        return 0
    if "--stats" in argv:
        report = stats()
        if not report["available"]:
            print(f"[extra] 还没建：{report['db']}（跑 --all 生成）")
            return 0
        print(f"[extra] {report['documents']} 份 ｜ {report['chunks']} 块 ｜ {report['db']}")
        for ext, count in sorted(report["byExtension"].items()):
            print(f"          {ext}: {count}")
        for status, count in sorted(report["byStatus"].items()):
            print(f"          status={status}: {count}")
        return 0

    substr = next((a.split("=", 1)[1] for a in argv if a.startswith("--path-substr=")), "")
    limit = next((a.split("=", 1)[1] for a in argv if a.startswith("--limit=")), "")
    limit_n = int(limit) if limit.isdigit() else 0
    force = "--force" in argv

    items, skipped = _candidates(substr=substr, force=force)
    if not force:
        done = stats().get("documents", 0) if EXTRA_DB.exists() else 0
        if done:
            conn = _connect()
            try:
                have = {(row[0], row[1]) for row in conn.execute(
                    "SELECT path, signature FROM documents WHERE status = ?", (STATUS_EXTRA,))}
            finally:
                conn.close()
            items = [it for it in items if (it["path"], it["signature"]) not in have]
    if limit_n:
        items = items[:limit_n]

    print(f"[extra] 待补 {len(items)} 份")
    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"          跳过 {count} 份：{reason}")
    if not items:
        return 0

    ok = quarantined = failed = 0
    chunks = 0
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        try:
            outcome = _build(item)
        except Exception as exc:  # noqa: BLE001 —— 一份坏了不该中断整批，但要逐份报出来
            failed += 1
            print(f"  [{index}/{len(items)}] !! {item['path']}：{type(exc).__name__}: {exc}",
                  flush=True)
            continue
        _save(item, outcome)
        if outcome["status"] == STATUS_EXTRA:
            ok += 1
            chunks += len(outcome["chunks"])
            tail = f"块={len(outcome['chunks'])} 页={outcome['pages']} 引擎={outcome['engine']}"
        else:
            quarantined += 1
            #: 隔离理由要能分辨两种完全不同的情况：抽出来是垃圾（可打印率低），
            #: 还是根本没字可抽（正文是图片/嵌入对象 —— 该走 OCR，不是解析器坏了）。
            why = (f"抽出 {outcome['chars']} 字，正文可能是图片/嵌入对象"
                   if outcome["chars"] < MIN_CHARS
                   else f"可打印率 {outcome['ratio']:.2f}")
            tail = f"隔离（{why}）"
        print(f"  [{index}/{len(items)}] {item['path']} {tail}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"[extra] 完成：入库 {ok} 份 / {chunks} 块，隔离 {quarantined} 份，失败 {failed} 份，"
          f"用时 {elapsed:.1f}s ⇒ {EXTRA_DB.as_posix()}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
