"""知识库上传：**落盘 → 抽正文 → 入补录库 → 记台账**，一条链路，每步失败都能说出是哪一步。

## 为什么不自己再写一套解析

``lib_index.py`` 已经把「一份 ``.doc`` 到底抽不抽得出字」这件事做全了（``antiword`` / 自研
OLE2 / RTF / 码表回退 / 可打印率 / 隔离判据）。上传若另写一套，就会出现**同一个文件用脚本跑
能进库、用界面上传进不去**的分裂 —— 用户没法理解，我们也说不清哪边是对的。所以这里直接复用
``lib_index._build`` 与 ``lib_index._save``（同一个 ``MIN_CHARS``、同一个 ``MIN_PRINTABLE``、
同一个 ``CHUNK_CHARS``，连块号都对得上）。

## 三档结果，不许合并成「失败」

| status | 含义 | 盘上有没有文件 |
|---|---|---|
| ``ingested`` | 正文抽出来了，能搜到 | 有 |
| ``quarantined`` | 落盘了，但抽不出可用正文（扫描件/图片型/损坏） | **有** —— 可以人工处理 |
| ``rejected`` | 连落盘都没做（类型不支持/太大/重名放弃） | 没有 |

前两档都写进 ``uploads`` 台账。把 ``quarantined`` 报成「失败」是最糟的一种：用户在盘上找得到
文件，却被告知没上传成功，只能反复重试。
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from app.scripts.corpus import lib_index

from . import library_overlay, script_library

#: 落盘目录（相对 ``LIBRARY_ROOT``）。单独一层，删起来不会误伤供应商那批。
UPLOAD_DIR = "00-本仓补录"

#: 单文件上限。剧本 20 MB 已经很大（纯文本 20 MB ≈ 600 万字），64 MB 留足扫描版 docx 的余量。
MAX_FILE_SIZE = 64 * 1024 * 1024

#: 一次请求最多几份。前端队列一次拖 20 份是正常操作，再多就该分批了。
MAX_FILES = 20

#: 可接受的扩展名 —— 与 ``lib_index`` 登记的抽文本通道**同一个真源**，多一个都不收
#: （收了也得靠 ``lib_index`` 才抽得出，它不认的这里收进来只是死文件）。
#: 直接读 ``lib_index.SUPPORTED_EXTS``，别再在这里做加法：抄一份的后果是
#: 上传能收、批量扫库却把同一批文件记成「未登记抽文本通道」（2026-11-07 实测）。
ALLOWED_SUFFIXES = tuple(lib_index.SUPPORTED_EXTS)

#: 给前端做提示用（"支持这些类型"），与 ``ALLOWED_SUFFIXES`` 同一真源。
ALLOWED_LABEL = "、".join(sorted({suffix.lstrip(".").upper() for suffix in ALLOWED_SUFFIXES}))

_SAFE = re.compile(r"[^\w\u4e00-\u9fff.\- ]+", re.UNICODE)
_MAX_NAME = 120


def safe_name(filename: str) -> str:
    """把浏览器给的 ``filename`` 变成能落盘的干净名字。

    只留「字/数字/下划线/汉字/点/横线/空格」；**不许有路径分隔符**（``../../evil.doc`` 这类
    穿越在 ``Path`` 上会直接逃出上传目录）。名字过长时截断主干、保住扩展名。
    """
    raw = Path(str(filename or "")).name.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE.sub("-", raw).strip(" .-")
    if not cleaned:
        cleaned = "未命名"
    stem, suffix = Path(cleaned).stem, Path(cleaned).suffix
    if len(cleaned) > _MAX_NAME:
        stem = stem[:max(1, _MAX_NAME - len(suffix))]
        cleaned = f"{stem}{suffix}"
    return cleaned


def _free_path(root: Path, name: str) -> Path:
    """同名不覆盖：``x.doc`` 已存在就落成 ``x-2.doc``、``x-3.doc``…（最多退 50 次）。"""
    target = root / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    for index in range(2, 52):
        candidate = root / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    return root / f"{stem}-{int(time.time())}{suffix}"


def _reject(name: str, reason: str, *, size: int = 0) -> dict[str, Any]:
    return {"ok": False, "status": "rejected", "name": name, "size": int(size),
            "reason": reason, "path": None, "chunks": 0, "pages": 0, "chars": 0}


def ingest(data: bytes, filename: str, *, note: str | None = None) -> dict[str, Any]:
    """收一份文件。**永远返回字典**（不抛），调用方直接把它塞进回执数组。"""
    name = safe_name(filename)
    size = len(data or b"")
    suffix = Path(name).suffix.lower()

    if not size:
        return _reject(name, "空文件（0 字节），没有任何内容可入库。")
    if suffix not in ALLOWED_SUFFIXES:
        return _reject(name, f"不支持的类型「{suffix or '(无扩展名)'}」；"
                             f"本库只收 {ALLOWED_LABEL}。",
                       size=size)
    if size > MAX_FILE_SIZE:
        return _reject(name, f"文件太大（{size / 1024 / 1024:.1f} MB），"
                             f"单份上限 {MAX_FILE_SIZE // 1024 // 1024} MB。", size=size)

    root = Path(script_library.LIBRARY_ROOT) / UPLOAD_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        target = _free_path(root, name)
        target.write_bytes(data)
    except OSError as exc:  # noqa: BLE001
        return _reject(name, f"落盘失败：{exc}", size=size)

    rel = target.relative_to(Path(script_library.LIBRARY_ROOT)).as_posix()
    digest = hashlib.sha256(data).hexdigest()[:32]
    stat = target.stat()
    item = {"path": rel, "local": target, "name": target.name, "extension": suffix,
            "size": stat.st_size, "modifiedNs": stat.st_mtime_ns,
            "signature": lib_index._signature(stat.st_size, stat.st_mtime_ns)}  # noqa: SLF001

    started = time.perf_counter()
    try:
        outcome = lib_index._build(item)  # noqa: SLF001 —— 与命令行同一条解析/判据
        lib_index._save(item, outcome)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 —— 解析器内部异常也要变成一条可读结果
        #: 文件已经在盘上了 ⇒ 记 ``quarantined``（不是 ``rejected``）：用户能在盘上找到它，
        #: 我们也能对着这份文件排查，而不是让人以为白传了一次。
        library_overlay.record_upload(path=rel, name=target.name, size=size,
                                      mime="", sha256=digest, status="quarantined",
                                      reason=f"抽正文时异常：{exc}")
        script_library.reset_cache()
        return {"ok": False, "status": "quarantined", "name": target.name, "path": rel,
                "size": size, "sha256": digest, "chunks": 0, "pages": 0, "chars": 0,
                "reason": f"已落盘，但抽正文失败（{type(exc).__name__}: {exc}）；"
                          "文件在盘上，可换格式重传或走 OCR 线。",
                "seconds": round(time.perf_counter() - started, 3)}

    ingested = outcome["status"] == lib_index.STATUS_EXTRA
    reason = None
    if not ingested:
        #: 一律先说「已落盘」：``quarantined`` 与 ``rejected`` 对用户的差别就在
        #: **盘上有没有这份文件**（前者能拿去人工处理/走 OCR，后者根本没有）。
        if outcome["chars"] < lib_index.MIN_CHARS:
            reason = "已落盘，但正文只有几个字"
            #: PDF 抽不满要指向 OCR 线（那是它唯一的补救路），其余格式只说没内容。
            reason += ("（多半是图片型/扫描件），要走 OCR 那条线才读得出内容。"
                       if suffix == ".pdf" else "，没有可检索内容。")
        else:
            reason = (f"已落盘，但抽出的正文可打印率只有 {outcome['ratio']:.2f}"
                      f"（下限 {lib_index.MIN_PRINTABLE}），像是抽错了/编码不对。")
    library_overlay.record_upload(
        path=rel, name=target.name, size=size, mime="", sha256=digest,
        status="ingested" if ingested else "quarantined", reason=reason,
        documents=1, chunks=len(outcome["chunks"]))
    script_library.reset_cache()
    return {
        "ok": ingested,
        "status": "ingested" if ingested else "quarantined",
        "name": target.name,
        "path": rel,
        "size": size,
        "sha256": digest,
        "chars": outcome["chars"],
        "pages": outcome["pages"],
        "chunks": len(outcome["chunks"]),
        "engine": outcome["engine"],
        "printable": round(outcome["ratio"], 4),
        "reason": reason,
        "note": note,
        "seconds": round(time.perf_counter() - started, 3),
    }


def limits() -> dict[str, Any]:
    """上传的口径（类型/单份上限/一次几份）——**由后端下发**，前端不许再抄一份。

    抄一份的后果不是报错而是「两个说法」：界面说 50 MB 能传、后端 64 MB 才拒，
    用户照着界面选完文件点了确认，然后在第 3 份上收到拒绝，还得自己猜哪边是对的。
    """
    return {
        "accept": list(ALLOWED_SUFFIXES),
        "label": ALLOWED_LABEL,
        "maxBytes": MAX_FILE_SIZE,
        "maxFiles": MAX_FILES,
        "dir": UPLOAD_DIR,
    }


def remove(path: str) -> dict[str, Any]:
    """彻底删掉一份**本仓上传的**文件（索引记录 + 盘上文件 + 台账）。"""
    return script_library.bury_document(path, hard=True, reason="上传件删除")
