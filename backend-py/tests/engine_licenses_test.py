"""自检：**第三方许可审计**（E0.6 ✓）—— 钉的是「不猜许可」这件事的机器形态。

`THIRD_PARTY_NOTICES.md` 是这份审计的交付物，但它最大的风险不是写漏，而是**写得太顺** ✗✗：
清单里 16 条权重一个 `license` 都没有 ✓，谁要是凭印象给 H3 填个 `Apache-2.0` ✓、
给 SDXL 填个 `MIT` ✓，下游就会当作商用依据 ✓✗ —— 这比留空危害大得多 ✓。

所以本套不验"文档写得好不好看"，只验四件：

1. **条目不多不少**：权重表的 key 集合 == `configs/models.json` 的 key 集合（双向）✓
   —— 少一条＝漏登、多一条＝文档里有盘上不存在的模型 ✓✗；
2. **许可列与清单逐条一致**：清单有 `license` ⇒ 该行逐字等于它 ✓；清单没有 ⇒ 必须是「未声明」✓；
3. **许可列不许出现任何具体许可名**（`Apache` / `MIT` / `BSD` / `GPL` / `OpenRAIL` …词表）——
   与第 2 条互为冗余：即便清单将来加了一个许可，也只能从清单来、不能从人脑来 ✓✗；
4. **依赖版本与实测一致**：§二 那几行的版本 == 本机包元数据 ✓，没装就得写「未安装」✓
   （依赖升级后这里会红 —— 这是**故意的**，提醒刷新审计 ✓）。

⚠️ 全程不碰 torch ✓：许可只读包元数据（`importlib.metadata`）与清单 JSON ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_licenses_test.py
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_PY.parent
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import inventory  # noqa: E402

NOTICES = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
UNKNOWN = "未声明"
NO_FILE = "—（无落点）"

#: 许可**名字**的词表（判据 ③ 用）——不列 SPDX 全表，只列"人最容易顺手编"的那批。
_BANNED = ("apache", "mit", "bsd", "gpl", "mpl", "cc0", "cc-by", "openrail",
           "zlib", "unlicense", "public domain", "artistic")

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _sections() -> dict[str, list[str]]:
    """按 `## ` 标题切节 ✓（返回 标题 → 行列表 ✓，不含标题行 ✓）。"""
    text = NOTICES.read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out[current] = []
        elif current:
            out[current].append(line)
    return out


def _rows(lines: list[str]) -> list[list[str]]:
    """把 Markdown 表格数据行拆成单元格 ✓（跳表头与分隔行 ✓）。"""
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or set("".join(cells)) <= set("-: "):
            continue
        if cells[0] in ("key", "包"):
            continue
        rows.append(cells)
    return rows


def _weight_rows() -> list[list[str]]:
    for title, lines in _sections().items():
        if title.startswith("一、"):
            return _rows(lines)
    raise AssertionError("文档里找不到「一、模型权重」那节（标题改了就取不到 ✓✗）")


def _dep_rows() -> list[list[str]]:
    for title, lines in _sections().items():
        if title.startswith("二、"):
            return _rows(lines)
    raise AssertionError("文档里找不到「二、Python 依赖」那节")


def case_catalog_entries() -> None:
    """判据 ①：条目不多不少（双向）✓。"""
    models = inventory.load_catalog()["models"]
    catalog = {str(item.get("key")) for item in models}
    rows = _weight_rows()
    documented = {row[0] for row in rows}

    check("① 权重表**不是空表** ✓（空表会让下面每条判据都在验空气 ✓✗）", bool(rows), len(rows))
    check(f"① 权重表条目 == 清单条目（{len(catalog)} 条）：多一条＝文档里有盘上不存在的模型 ✓✗、"
          f"少一条＝漏登 ✓✗",
          documented == catalog,
          {"文档多出": sorted(documented - catalog), "文档漏掉": sorted(catalog - documented)})
    check("① 权重表**没有重复行** ✓（同一 key 写两遍会让「逐条一致」与条数判据同时失真 ✓✗）",
          len(documented) == len(rows), len(rows))

    for row in rows:
        if len(row) < 4:
            check(f"① 行 `{row[0]}` 列数够 4（key / filename / category / 许可）✓", False, row)


def case_license_cells() -> None:
    """判据 ②③：许可列**只能来自清单**，清单没有就是「未声明」✓。"""
    catalog = {str(item.get("key")): item for item in inventory.load_catalog()["models"]}
    rows = _weight_rows()
    wrong: list[dict[str, object]] = []

    for row in rows:
        if len(row) < 4:
            continue
        key, cell = row[0], row[3]
        declared = (catalog.get(key) or {}).get("license")
        expected = str(declared) if declared else UNKNOWN
        if cell != expected:
            wrong.append({"key": key, "细胞": cell, "应为": expected})

    check("② 许可列**逐条等于清单**：清单有 ⇒ 逐字照抄 ✓；清单没有 ⇒ 「未声明」✓ "
          "（手写一个 `Apache-2.0` 进来就在这条红 ✓✗）",
          not wrong, wrong)

    # 判据 ③：许可**名字**不许出现在许可列（防"清单没有、文档里有"这条绕过路径 ✓）。
    banned_hit = [{"key": row[0], "细胞": row[3]}
                  for row in rows if len(row) >= 4
                  and any(word in row[3].lower() for word in _BANNED)]
    check("③ 许可列**没有任何具体许可名** ✓✗（「未声明」是一种结论，不是占位符：能写「未声明」，"
          "就不能顺手写个「大概是 MIT」✓）",
          not banned_hit, banned_hit)


def case_no_file_rows() -> None:
    """有落点 / 无落点必须分得清 ✓ —— `manifest_of` 按**文件**出清单，无落点的它根本谈不到 ✓✗。"""
    models = inventory.load_catalog()["models"]
    without = {str(item.get("key")) for item in models if not item.get("filename")}
    rows = {row[0]: row for row in _weight_rows() if len(row) >= 2}

    mismatched = {key: rows[key][1] for key in without
                  if key in rows and rows[key][1] != NO_FILE}
    check(f"有落点/无落点: 清单里 {len(without)} 条**没有 filename** ⇒ 表里必须写「无落点」✓"
          f"（写成「—」或干脆留空，都会让人以为「有文件但没查」✓✗）",
          not mismatched, mismatched)

    claimed = {key: row[1] for key, row in rows.items()
               if key not in without and row[1] in ("", NO_FILE, "—")}
    check("有落点/无落点: 有 `filename` 的条目在表里必须**写出文件名** ✓（否则许可审计无法回溯到那份文件 ✓）",
          not claimed, claimed)


def case_dependencies() -> None:
    """判据 ④：依赖版本 == 本机实测 ✓（升级后这里红，是**故意的** ✓）。"""
    rows = _dep_rows()
    drift: list[dict[str, object]] = []
    uninstalled_claims: list[str] = []

    for row in rows:
        if len(row) < 2:
            continue
        name, version = row[0], row[1]
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            if version != "未安装":
                uninstalled_claims.append(f"{name}={version}")
            continue
        if version == "未安装":
            uninstalled_claims.append(f"{name} 已装 {actual} 却写未安装")
        elif version != actual:
            drift.append({"包": name, "文档": version, "实测": actual})

    check("④ 依赖版本与实测一致 ✓（依赖升级后刷新本文件 ✓ —— 过期的许可审计比没有更坏 ✓✗）",
          not drift, drift)
    check("④ 「装了/没装」不许写反 ✓✗（把已装的写成未安装，会让人以为不用审这一条 ✓）",
          not uninstalled_claims, uninstalled_claims)


def case_scope() -> None:
    """文档要自己说清**边界** ✓：不替上游补许可、无落点的条目无从审、参考副本不算依赖 ✓。"""
    text = NOTICES.read_text(encoding="utf-8")
    for phrase in ("不替上游补许可", "无落点", "参考副本"):
        check(f"边界: 文档里写明了「{phrase}」✓（留空不解释，读的人只会当它漏了 ✓✗）",
              phrase in text, phrase)
    check("边界: 判据在哪也写在文档里 ✓（否则下一个人改文档不知道会撞红什么 ✓）",
          "engine_licenses_test" in text, "")


def main() -> int:
    case_catalog_entries()
    case_license_cells()
    case_no_file_rows()
    case_dependencies()
    case_scope()

    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
