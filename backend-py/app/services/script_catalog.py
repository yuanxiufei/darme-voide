"""剧本知识库的**目录层**（两份供应商 CSV → 内存索引，2026-11-04）。

为什么必须有这一层
==================

供应商的 ``curated-screenplay-methods.sqlite`` 里 ``documents`` 表**没有**国家/类型/奖项/权限
这些列（只有 ``id/path/name/extension/size/modified_ns/signature/pages/status``）⇒ 想按
「韩国 · 犯罪 · 戛纳获奖」过滤，SQL 层根本无从下手；这些字段只存在于两份 CSV：

* ``99-索引与检索规则/全球剧本知识库索引.csv``（6452 行，**UTF-8 带 BOM**）
  —— 一级分类 / 国家地区 / 二级分类 / 资料类型 / 权限状态 / 索引状态 / 页数 / 本地相对路径
* ``99-索引与检索规则/电影节剧本标签索引.csv``（**GBK**）
  —— 电影节 / 影片 / 电影节关系 / 本地完整路径

两个「不装样子」的口径
======================

1. **不建 sidecar SQLite**：两份 CSV 合计 6452 行、不到 15 MB，读进内存 ≈0.2 s。建库只会多一个
   **必须重建的派生物**，而「派生物与原始 CSV 不同步」正是本仓最忌的两本账（同
   :mod:`app.services.script_library` 头注释的纪律）。CSV 一改就靠 ``(mtime_ns, size)`` 失效。
2. **编码必须分开读**：主目录是 UTF-8 BOM、电影节标签是 **GBK**。用 ``utf-8`` 硬读 GBK 文件
   **不会抛错**，只会得到一堆乱码 —— 而乱码的后果是「按电影节筛永远 0 条」，是最坏的一种错
   （看起来像"库里没有"）。故 :func:`_read_csv` 逐编码试解，试不通才用 ``errors="replace"``。

与 ``script_library`` 的关系：**单向**（``script_library`` → 本模块）。本模块不 import 它，
避免 services 之间成环；路径锚点由调用方按需传入（默认值与 ``LOCAL_MARKER`` 同词）。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CATALOG_DIR",
    "DIMENSIONS",
    "FESTIVAL_CSV",
    "MARKER",
    "MAIN_CSV",
    "Entry",
    "entries",
    "facets",
    "lookup",
    "normalize",
    "report",
    "select",
]

#: 目录 CSV 所在的子目录（与索引 sqlite 同目录）
CATALOG_DIR = "99-索引与检索规则"
MAIN_CSV = "全球剧本知识库索引.csv"
FESTIVAL_CSV = "电影节剧本标签索引.csv"
#: 路径锚点：它**右边**就是相对库根的路径（默认值与 ``script_library.LOCAL_MARKER`` 同词）
MARKER = "全球热门与获奖影视剧本知识库"

#: 可过滤的维度（顺序即回执顺序）。``docType``/``permission`` 用驼峰是为了直接进 JSON。
DIMENSIONS = ("category", "country", "genre", "docType", "permission", "festival")

#: 维度值的分隔符见 :func:`_split_values`（``|`` / ``,`` / ``，`` / ``、``）


@dataclass(frozen=True)
class Entry:
    """一部作品的目录信息（唯一键是 :attr:`path`，与索引里的 path 同一套归一化）。"""

    path: str
    title: str
    category: str
    country: str
    genre: str
    doc_type: str
    permission: str
    indexed: str
    pages: int
    festivals: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "category": self.category,
            "country": self.country,
            "genre": self.genre,
            "docType": self.doc_type,
            "permission": self.permission,
            "indexed": self.indexed,
            "pages": self.pages,
            "festivals": list(self.festivals),
        }


# ===========================================================================
# 读盘与归一化
# ===========================================================================


def normalize(raw: str, marker: str = MARKER) -> str:
    """任意写法的路径 → **相对库根的 posix 路径**（跨机器唯一可比的那种写法）。

    * 索引里是 ``G:\\电影剧本\\全球热门与获奖影视剧本知识库\\01-…\\x.pdf``
    * CSV 里是 ``01-传统神话志怪原典\\志怪原典\\1801 玄怪录.pdf``（相对）
    * 也见过绝对路径写在 CSV 的 ``本地完整路径`` 列

    三者都归到 ``01-传统神话志怪原典/志怪原典/1801 玄怪录.pdf``。没有锚点且不是相对路径时
    返回 ``""``（**不猜**）。
    """
    text = (raw or "").strip().replace("/", "\\")
    if not text:
        return ""
    index = text.find(marker)
    if index >= 0:
        text = text[index + len(marker):]
    text = text.lstrip("\\")
    if not text:
        return ""
    #: 绝对盘符（``D:\…``）但没有锚点 ⇒ 无法映射，明确失败而不是「截一段凑数」
    if len(text) > 1 and text[1] == ":":
        return ""
    return text.replace("\\", "/")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """逐编码试解（UTF-8 BOM → GBK → UTF-8 → replace）。**读不出就空表**，由调用方报缺。"""
    if not path.exists():
        return []
    raw = path.read_bytes()
    text: str | None = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except csv.Error:  # 个别行坏掉不该让整层不可用
        return []


def _cell(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def _pages(raw: str) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


_CACHE: dict[str, dict[str, Any]] = {}


def _load(root: Path | str, marker: str = MARKER) -> dict[str, Any]:
    """读两份 CSV 并合并；``_CACHE`` 以 ``(mtime_ns, size)`` 判失效。"""
    base = Path(root) / CATALOG_DIR
    main_file, festival_file = base / MAIN_CSV, base / FESTIVAL_CSV
    key = base.as_posix()

    def stamp(item: Path) -> tuple[int, int]:
        try:
            stat = item.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return 0, 0

    sig = (stamp(main_file), stamp(festival_file))
    cached = _CACHE.get(key)
    if cached and cached["sig"] == sig:
        return cached

    festivals: dict[str, list[str]] = {}
    for row in _read_csv(festival_file):
        rel = normalize(_cell(row, "本地完整路径", "本地相对路径"), marker)
        if not rel:
            continue
        name = _cell(row, "电影节")
        relation = _cell(row, "电影节关系")
        label = f"{name}·{relation}" if name and relation else (name or relation)
        if label:
            festivals.setdefault(rel, []).append(label)

    table: dict[str, Entry] = {}
    for row in _read_csv(main_file):
        rel = normalize(_cell(row, "本地相对路径", "本地完整路径"), marker)
        if not rel:
            continue
        entry = Entry(
            path=rel,
            title=_cell(row, "文件名") or Path(rel).name,
            category=_cell(row, "一级分类"),
            country=_cell(row, "国家地区"),
            genre=_cell(row, "二级分类"),
            doc_type=_cell(row, "资料类型"),
            permission=_cell(row, "权限状态"),
            indexed=_cell(row, "索引状态"),
            pages=_pages(_cell(row, "页数")),
            festivals=tuple(dict.fromkeys(festivals.get(rel, ()))),  # 去重且保序
        )
        table[rel] = entry

    #: 只在电影节 CSV 里出现的作品也收进来（两本账合成一本，缺哪列就空着，不编造）
    for rel, labels in festivals.items():
        if rel not in table:
            table[rel] = Entry(path=rel, title=Path(rel).name, category="", country="",
                               genre="", doc_type="", permission="", indexed="",
                               pages=0, festivals=tuple(dict.fromkeys(labels)))

    payload = {
        "sig": sig,
        "table": table,
        "lower": {rel.lower(): rel for rel in table},
        "files": {"main": bool(main_file.exists()), "festival": bool(festival_file.exists())},
        "root": Path(root).as_posix(),
    }
    _CACHE[key] = payload
    return payload


def _available(root: Path | str, marker: str = MARKER) -> bool:
    return bool(_load(root, marker)["table"])


def entries(root: Path | str, marker: str = MARKER) -> dict[str, Entry]:
    """``相对路径 → Entry``（库目录未安装时是空 dict，**不是 None**）。"""
    return dict(_load(root, marker)["table"])


def lookup(root: Path | str, raw_path: str, marker: str = MARKER) -> Entry | None:
    """按任意写法的路径查一部作品（大小写不敏感，NTFS 语义）。"""
    payload = _load(root, marker)
    rel = normalize(raw_path, marker)
    if not rel:
        return None
    table: dict[str, Entry] = payload["table"]
    return table.get(rel) or table.get(payload["lower"].get(rel.lower(), ""))


# ===========================================================================
# 过滤与分面
# ===========================================================================


def _split_values(value: Any) -> list[str]:
    """把 ``"韩国，日本"`` / ``"韩国|日本"`` / ``["韩国"]`` 统一成小写词表。"""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for item in items:
        for part in str(item).replace("|", ",").replace("，", ",").replace("、", ",").split(","):
            part = part.strip().lower()
            if part:
                out.append(part)
    return out


def _hit(field: str, wanted: list[str]) -> bool:
    """字段值命中任一检索词（**子串**语义：``韩国`` 命中 ``韩国``、``美`` 命中 ``美国``）。"""
    low = (field or "").lower()
    return any(item in low for item in wanted)


def select(root: Path | str, *, marker: str = MARKER, **criteria: Any) -> set[str] | None:
    """按维度筛出**允许的相对路径集合**。

    返回 ``None`` 表示「这一维没法用」——目录 CSV 未安装。**不能返回空集**：
    空集会让检索结果变成 0 条，而那会被误读成「库里没有这部」。
    返回 ``set()`` 表示目录在、但**没有任何作品满足条件**（这是真结论，要如实回报）。
    """
    wanted = {name: _split_values(criteria.get(name)) for name in DIMENSIONS}
    if not any(wanted.values()):
        return None
    payload = _load(root, marker)
    table: dict[str, Entry] = payload["table"]
    if not table:
        return None
    allowed: set[str] = set()
    for rel, entry in table.items():
        if wanted["category"] and not _hit(entry.category, wanted["category"]):
            continue
        if wanted["country"] and not _hit(entry.country, wanted["country"]):
            continue
        if wanted["genre"] and not _hit(entry.genre, wanted["genre"]):
            continue
        if wanted["docType"] and not _hit(entry.doc_type, wanted["docType"]):
            continue
        if wanted["permission"] and not _hit(entry.permission, wanted["permission"]):
            continue
        if wanted["festival"]:
            joined = " ".join(entry.festivals).lower()
            if not any(item in joined for item in wanted["festival"]):
                continue
        allowed.add(rel)
    return allowed


def facets(root: Path | str, marker: str = MARKER) -> dict[str, dict[str, int]]:
    """各维度的可选值 + 条数（前端下拉/Agent 自查都靠它，避免瞎猜词）。"""
    table = entries(root, marker)
    out: dict[str, dict[str, int]] = {name: {} for name in DIMENSIONS}
    for entry in table.values():
        pairs = (
            ("category", entry.category),
            ("country", entry.country),
            ("genre", entry.genre),
            ("docType", entry.doc_type),
            ("permission", entry.permission),
        )
        for name, value in pairs:
            if value:
                out[name][value] = out[name].get(value, 0) + 1
        for label in entry.festivals:
            out["festival"][label] = out["festival"].get(label, 0) + 1
    return {name: dict(sorted(values.items(), key=lambda kv: (-kv[1], kv[0])))
            for name, values in out.items()}


def report(root: Path | str, marker: str = MARKER) -> dict[str, Any]:
    """盘点：目录在不在、几部、每维几个值（缺哪个文件要能自证）。"""
    payload = _load(root, marker)
    table: dict[str, Entry] = payload["table"]
    if not table:
        return {
            "available": False,
            "root": payload["root"],
            "files": payload["files"],
            "hint": (
                "目录 CSV 未安装（或路径不对）：应在 "
                f"「{CATALOG_DIR}/{MAIN_CSV}」与「{CATALOG_DIR}/{FESTIVAL_CSV}」；"
                "缺它就**没法按国家/类型/奖项/权限过滤**（检索本身仍可用）。"
            ),
        }
    dimensions = facets(root, marker)
    return {
        "available": True,
        "root": payload["root"],
        "files": payload["files"],
        "entries": len(table),
        "withFestival": sum(1 for item in table.values() if item.festivals),
        "dimensions": {name: len(values) for name, values in dimensions.items()},
    }
