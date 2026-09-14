#!/usr/bin/env python3
"""语料检索（最小闭环验证）—— 归一化后的本地 prompt 库上做「找相似镜头」。

> 本文件是 ``backend-py/scripts/corpus/search.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。

定位：**先证明检索有没有用**，不追求工业级检索。
  8987 条 / 约 2.5 MB 文本，全量读内存 + 命中打分就是毫秒级，**无需 SQLite / 向量库**。
  若实测有效，再决定要不要上 embedding（本地已装 ``nomic-embed-text``）与持久索引。

打分口径（可调，别在没数据前过度设计）
  标题命中 5 ・ tags 命中 3 ・ 分类命中 2 ・ 中文摘要命中 2 ・ 正文命中 1
  中文用 **2-gram** 切词：中文无空格，2-gram 是最省事且够用的近似
  （"雨夜霓虹" → 雨夜/夜霓/霓虹，因此「霓虹雨夜」也能互相命中）。

⚠️ **同口径的 Agent 工具在 Python 侧**（``app/services/agents/tools/script_tools.py`` 一带，
挂给分镜/宫格 Agent，语料缺失时降级）—— **改打分口径必须两处同步**，否则「命令行验证有效」
与「Agent 实际效果」会分叉。（Node 时代的这条约束同样适用，只是对面换成了 Python 实现。）

用法（⚠️ PowerShell 直接传中文参数会乱码 ⇒ 优先用 ``--q-file``）::

    python backend-py/backend-py/scripts/corpus/search.py --q-file=tmp/q.txt --top=5
    python backend-py/backend-py/scripts/corpus/search.py "rainy neon street" --top=5
    CORPUS_QUERY=雨夜霓虹 python backend-py/backend-py/scripts/corpus/search.py   # 环境变量亦可

参数：``--top=N``（默认 5）｜``--lang=zh|en``｜``--source=<id 前缀>``｜``--full``（打印完整 prompt）
"""

from __future__ import annotations

import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JSONL = ROOT / "data" / "prompt-corpus" / "_normalized" / "prompts.jsonl"

LATIN_RE = re.compile(r"[a-z0-9][a-z0-9._-]+")
NON_CJK_RE = re.compile(r"[^\u4e00-\u9fff]+")
WHITESPACE_RE = re.compile(r"\s+")


def _arg(name: str, default: str = "") -> str:
    prefix = f"--{name}="
    return next((arg[len(prefix):] for arg in sys.argv if arg.startswith(prefix)), default)


def _js_len(text: str) -> int:
    """Node 的 ``String.length`` 口径（UTF-16 码元）。"""
    return len(text.encode("utf-16-le")) // 2


def _js_slice(text: str, limit: int) -> str:
    """Node 的 ``String.prototype.slice`` 口径（按 UTF-16 码元切）。"""
    return text.encode("utf-16-le")[: limit * 2].decode("utf-16-le", errors="ignore")


def _js_to_fixed(value: float, digits: int) -> str:
    """``Number.prototype.toFixed`` 口径：**四舍五入（half-up）**，不是 Python 的银行家舍入。

    本项目已被这个差异坑过两次（``round()``/``:.0f`` 把 .5 舍成偶数）⇒ 统一用 Decimal。
    """
    quantized = Decimal(repr(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{quantized:.{digits}f}"


def tokenize(text: str) -> list[str]:
    """查询分词：拉丁词按词边界；CJK 取 2-gram。"""
    out: set[str] = set()
    lowered = str(text).lower()
    out.update(LATIN_RE.findall(lowered))
    for segment in NON_CJK_RE.sub(" ", lowered).split():
        if not segment:
            continue
        if len(segment) == 1:
            out.add(segment)
        for index in range(len(segment) - 1):
            out.add(segment[index:index + 2])
    return list(out)


def main() -> int:
    query = _arg("q") or os.environ.get("CORPUS_QUERY") or (sys.argv[2] if len(sys.argv) > 2 else "")
    q_file = _arg("q-file")
    if q_file and (ROOT / q_file).exists():
        query = (ROOT / q_file).read_text(encoding="utf-8").strip()
    if not query:
        print("缺少查询词。用 --q-file=tmp/q.txt 或 CORPUS_QUERY 环境变量"
              "（PowerShell 直传中文会乱码）", file=sys.stderr)
        return 2

    if not JSONL.exists():
        print("缺少 data/prompt-corpus/_normalized/prompts.jsonl —— "
              "先跑 python backend-py/backend-py/scripts/corpus/normalize.py", file=sys.stderr)
        return 2

    rows = []
    for line in JSONL.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue

    lang = _arg("lang")
    source = _arg("source")
    if lang:
        rows = [row for row in rows if row.get("lang") == lang]
    if source:
        rows = [row for row in rows if str(row.get("source", "")).startswith(source)]

    top = int(_arg("top", "5"))
    full = "--full" in sys.argv
    tokens = tokenize(query)
    scored = []
    for row in rows:
        title = str(row.get("title") or "").lower()
        tags = " ".join(row.get("tags") or []).lower()
        category = str(row.get("category") or "").lower()
        summary = str(row.get("summary_zh") or "").lower()
        body = str(row.get("prompt") or "").lower()
        score = 0
        hits = 0
        for key in tokens:
            got = 0
            if key in title:
                got += 5
            if key in tags:
                got += 3
            if key in category:
                got += 2
            if key in summary:
                got += 2
            if key in body:
                got += 1
            if got:
                hits += 1
            score += got
        if score > 0:
            scored.append((row, score, hits / len(tokens) if tokens else 0.0))

    scored.sort(key=lambda item: -item[1])

    print(f"查询：{query}")
    print(f"词元 {len(tokens)} 个 ｜ 候选 {len(rows)} 条 ｜ 命中 {len(scored)} 条\n")
    for row, score, cover in scored[:top]:
        print(f"[{score} 分 · 覆盖 {_js_to_fixed(cover * 100, 0)}%] "
              f"{row.get('source')} · {row.get('lang')} · {row.get('category') or '-'}")
        extras = (("  ·  " + str(row["mode"])) if row.get("mode") else "")
        extras += ((" · " + str(row["duration"])) if row.get("duration") else "")
        extras += ((" · " + str(row["aspect"])) if row.get("aspect") else "")
        print(f"  {row.get('title') or '(无标题)'}{extras}")
        prompt = str(row.get("prompt") or "")
        if full:
            body = prompt
        else:
            body = WHITESPACE_RE.sub(" ", prompt)
            body = _js_slice(body, 220) + (" …" if _js_len(prompt) > 220 else "")
        print("  " + body.replace("\n", "\n  "))
        if row.get("tags"):
            print("  tags: " + ", ".join(row["tags"][:8]))
        print("")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
