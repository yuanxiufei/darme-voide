#!/usr/bin/env python3
"""语料归一化：各源原始文件 → 统一 JSONL（供检索层消费）。

> 本文件是 ``backend-py/scripts/corpus/normalize.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。
> 唯一的**有意差异**：文件名列表用 ``sorted()``（Node 的 ``readdirSync`` 不保证顺序）⇒
> 产物条数、字段、内容一致，仅**行序**在极端情况下可能不同。

── 设计取舍（重要，勿随手加维度）────────────────────────────────────────
**只抽「能检索的最小字段集」**：id / source / license / lang / title / prompt / category /
tags / mode / duration / aspect。

为什么不做 11 维全标注：1.2 万条 × 11 维是周级重活，而且**没有 ground truth 可验收**
标注质量。而「能不能检索到相关镜头」这件事，用上面这几个字段就足以验证。
⇒ 先证明检索有用，再按需补标注。路线讨论见 ``docs/video-prompt-data-sources.md``。

── 数据来源与许可 ───────────────────────────────────────────────────────
  seedance-prompt-ericgood      CC BY 4.0（可商用，需署名）  结构：数组 JSON
  awesome-seedance-2-5-flaqai   MIT（宽松）                  结构：Markdown（层级不一致）
  seedance2（HF 已下载）         CC BY 4.0                   结构：JSONL

⚠️ 原始正文**不得进仓库**（第三方版权红线）：本脚本产物落在 ``data/prompt-corpus/``（已 gitignore）。

用法::

    python backend-py/scripts/corpus/normalize.py             # 归一化全部已采集的源
    python backend-py/scripts/corpus/normalize.py --source=flaqai

产物
  ``data/prompt-corpus/_normalized/prompts.jsonl``   （每行一条，UTF-8）
  ``data/prompt-corpus/_normalized/report.txt``      （字段填充率与分布，供人工核对）
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "data" / "prompt-corpus"
OUT_DIR = CORPUS / "_normalized"
ONLY = next((arg[9:] for arg in sys.argv if arg.startswith("--source=")), "")

_report: list[str] = []


def say(message: str) -> None:
    _report.append(message)


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
TAG_SPLIT_RE = re.compile(r"[,;，；|]")


def cjk_count(text: object) -> int:
    return len(CJK_RE.findall(str(text)))


def pick(obj: dict, names: tuple[str, ...]):
    """取首个非空：不同源字段名不同，用候选名列表兜住（大小写不敏感，按对象键序）。"""
    for name in names:
        for key in obj:
            if key.lower() == name.lower():
                value = obj[key]
                if value is not None and str(value).strip() != "":
                    return value
    return None


def to_tags(value: object) -> list[str]:
    """把任意 tag 形态（数组 / 逗号串 / 分号串）规整成字符串数组。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in TAG_SPLIT_RE.split(value) if part.strip()]
    return []


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _js_num_str(value: object) -> str:
    """JS 的 ``String(number)`` 口径 —— **整数型浮点不带 ``.0``**。

    ⚠️ 实测坑：``spec.ratio`` 在语料里是 ``1.0`` 这种 JSON 数字，JS 侧 ``String(1.0)`` 得 ``"1"``，
    而 Python ``str(1.0)`` 得 ``"1.0"`` ⇒ 8987 条里有 206 条的 ``aspect`` 会悄悄变样（对拍时抓到）。
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _text(value: object) -> str:
    """``String(x ?? '')`` 语义：None → 空串；数字走 JS 口径（见 ``_js_num_str``）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _js_num_str(value)
    return str(value)


# ── 源 1：Ericgood seedance-prompt（数组 JSON）─────────────────────────────
def parse_ericgood() -> list[dict]:
    path = CORPUS / "seedance-prompt-ericgood" / "raw" / "prompts.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("prompts") or payload.get("data") or payload.get("items") or []
    out: list[dict] = []
    for index, row in enumerate(rows):
        prompt = pick(row, ("prompt", "prompt_zh", "content", "text", "description"))
        if not prompt or not str(prompt).strip():
            continue
        zh_title = _text(row.get("title_zh")).strip()
        en_title = _text(row.get("title")).strip()
        # 实测：该源 prompt 是**英文**（112 条里仅 1 条含中文），但带成套中文元数据
        # （title_zh / description_zh / tag_zh）⇒ prompt 语言记 en，中文侧另存
        # title / summary_zh / tags，保证中文检索能命中。
        raw_id = row.get("id")
        out.append({
            # `r.id ?? i + 1`：只有 null/undefined 才回退（键存在但为 null 也要回退）
            "id": f"ericgood:{_text(raw_id if raw_id is not None else index + 1)}",
            "source": "seedance-prompt-ericgood",
            "license": "CC-BY-4.0",
            "source_url": "https://github.com/Ericgood/seedance-prompt",
            "lang": "zh" if cjk_count(prompt) > 0 else "en",
            "title": zh_title or en_title,
            "title_en": en_title,
            "summary_zh": _text(row.get("description_zh")).strip(),
            "prompt": str(prompt).strip(),
            "category": _text(row.get("category")).strip(),
            "tags": [*to_tags(row.get("tag")), *to_tags(row.get("tag_zh")),
                     _text(row.get("model")).strip()],
            "mode": _text(row.get("mode")).strip(),
            "duration": _text(row.get("duration")).strip(),
            "aspect": _text(row.get("aspect")).strip(),
        })
    for item in out:
        item["tags"] = [tag for tag in item["tags"] if tag]
    return out


# ── 源 2：flaqai awesome-seedance-2-5（Markdown，中文 ### / 英文 ## 混用）──
SCENE_RE = re.compile(r"^(#{2,3})\s*(\d+)\.\s*(.+?)\s*$", re.MULTILINE)
SCENE_LINE_RE = re.compile(r"^(#{2,3})\s*(\d+)\.\s*(.+?)\s*$")
HEAD_RE = re.compile(r"^#{1,2}\s+(.+?)\s*$", re.MULTILINE)
META_RE = re.compile(r"\*\*([^*]+?)\s*[:：]\s*\*\*\s*([^·\n]+)")
FENCE_RE = re.compile(r"```(?:text|txt|markdown)?\s*\n([\s\S]*?)```")


def parse_flaqai() -> list[dict]:
    directory = CORPUS / "awesome-seedance-2-5-flaqai" / "raw" / "prompts"
    if not directory.exists():
        return []
    out: list[dict] = []
    names = sorted(name for name in __import__("os").listdir(directory)
                   if name.endswith(".md") and name != "README.md")
    for name in names:
        text = (directory / name).read_text(encoding="utf-8")
        # ⚠️ 必须**全文**切块：```text 围栏跨行，逐行匹配永远命中不了（Node 首版即踩此坑，产出 0 条）
        marks = [{"index": match.start(), "num": match.group(2), "title": match.group(3)}
                 for match in SCENE_RE.finditer(text)]
        # 场景可能用 ## 或 ###，故分类取「本标题之前最近的 非场景 标题行」
        heads = [{"index": match.start(), "text": match.group(1).strip()}
                 for match in HEAD_RE.finditer(text)
                 if not SCENE_LINE_RE.match(match.group(0))]

        def category_of(index: int) -> str:
            current = ""
            for head in heads:
                if head["index"] < index:
                    current = head["text"]
            return current

        for position, mark in enumerate(marks):
            end = marks[position + 1]["index"] if position + 1 < len(marks) else len(text)
            block = text[mark["index"]:end]
            fence = FENCE_RE.search(block)
            body = fence.group(1).strip() if fence else ""
            if not body:
                continue
            meta: dict[str, str] = {}
            for line in block.split("\n"):
                if "```" in line:
                    break
                for match in META_RE.finditer(line):
                    meta[match.group(1).strip().lower()] = match.group(2).strip()
            out.append({
                "id": f"flaqai:{mark['num']}",
                "source": "awesome-seedance-2-5-flaqai",
                "license": "MIT",
                "source_url": "https://github.com/flaqai/awesome_seedance_2_5",
                "lang": "zh" if cjk_count(body) > 0 else "en",
                "title": mark["title"],
                "summary_zh": "",
                "prompt": body,
                "category": category_of(mark["index"]),
                "tags": [re.sub(r"\.md$", "", name)],
                "mode": meta.get("mode") or meta.get("模式") or "",
                "duration": meta.get("duration") or meta.get("时长") or "",
                "aspect": meta.get("format") or meta.get("画幅") or "",
            })
    return out


# ── 源 3：Seedance 2（本地已下载的 8755 条 JSONL）─────────────────────────
def parse_seedance2() -> list[dict]:
    path = CORPUS / "seedance2" / "metadata.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        zh = (row.get("i18n") or {}).get("zh") or {}
        prompt = zh.get("p") if isinstance(zh.get("p"), str) and zh["p"].strip() else row.get("raw_p")
        if not prompt or not str(prompt).strip():
            continue
        duration = (row.get("spec") or {}).get("duration")
        # ⚠️ spec.duration 有脏值（最小 -3.69e17、最大 887.75）⇒ 只在 0 < d <= 300 时采信。
        #    `Math.round` 是 **half-up**（12.5 → 13），Python `round()` 是银行家舍入（12.5 → 12）✗
        #    ⇒ 用 floor(d + 0.5) 对齐。
        duration_text = (f"{math.floor(duration + 0.5)}s"
                         if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                         and 0 < duration <= 300 else "")
        spec = row.get("spec") or {}
        out.append({
            "id": f"sd2:{index + 1}",
            "source": "seedance2",
            "license": "CC-BY-4.0",
            "source_url": "https://huggingface.co/datasets/GokuScraper/seedance-2-prompts-datasets",
            "lang": "zh" if cjk_count(prompt) > 0 else "en",
            "title": _text(zh.get("t")).strip(),
            "summary_zh": "",
            "prompt": str(prompt).strip(),
            "category": _text(row.get("category")).strip(),
            "tags": [*to_tags(zh.get("tags")),
                     _text((row.get("model_info") or {}).get("name")).strip()],
            "mode": "",
            "duration": duration_text,
            "aspect": _text(spec.get("ratio")) if spec.get("ratio") else "",
        })
    for item in out:
        item["tags"] = [tag for tag in item["tags"] if tag]
    return out


SOURCES = (
    ("ericgood", parse_ericgood),
    ("flaqai", parse_flaqai),
    ("seedance2", parse_seedance2),
)


def main() -> int:
    say(f"# 语料归一化 {_iso_now()}")
    all_rows: list[dict] = []
    for key, parser in SOURCES:
        if ONLY and not key.startswith(ONLY):
            continue
        rows = parser()
        say(f"\n=== {key}  → {len(rows)} 条")
        for field in ("title", "category", "mode", "duration", "aspect", "tags"):
            filled = sum(1 for row in rows
                         if (len(row[field]) if isinstance(row[field], list)
                             else str(row[field] or "").strip()))
            say(f"  fill {field.ljust(10)} {filled}/{len(rows)}")
        zh = sum(1 for row in rows if row["lang"] == "zh")
        say(f"  lang zh/en   {zh}/{len(rows) - zh}")
        lengths = sorted(len(row["prompt"]) for row in rows)
        if lengths:
            say(f"  prompt len min/med/max = {lengths[0]} / "
                f"{lengths[len(lengths) // 2]} / {lengths[-1]}")
        all_rows.extend(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in all_rows),
        encoding="utf-8")
    say(f"\n# 合计 {len(all_rows)} 条 → data/prompt-corpus/_normalized/prompts.jsonl")
    (OUT_DIR / "report.txt").write_text("\n".join(_report), encoding="utf-8")
    print("\n".join(_report))
    print(f"\nTOTAL={len(all_rows)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
