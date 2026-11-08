"""真实引擎冒烟：拿盘上一份**真扫描件**跑前 2 页，验证「渲染 → OCR → 分栏重排 → 拼文本」。

为什么不放进常规测试集：它要加载 rapidocr 并读 10 GB 语料里的真文件（约 15 秒），
而单测必须秒级、可离线复现。但这一步不能省 —— 单测里的框是手搓的 list，真实
rapidocr 给的是 numpy 数组、可能是 RGBA、可能只有两三行，这些差异只有真跑才暴露。

不写库、不改任何记录，只打印。

运行::

    ./.venv/Scripts/python.exe tests/ocr_engine_smoke.py
    ./.venv/Scripts/python.exe tests/ocr_engine_smoke.py 世界电影   # 指定子串，专挑双栏期刊
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

import contextlib  # noqa: E402

from app.scripts.corpus import ocr_index as oi  # noqa: E402
from app.scripts.corpus import ocr_layout as ol  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    #: 只要 PDF：待做队列里还混着 docx／图片，直接按页数挑会挑到 `.docx` 然后开 PDF 时抛
    #: `PdfiumError: Data format error`（2026-11-07 实测就这么栽了一次），而本合同线里
    #: 「扫描件」只有 PDF 这一种 —— 冒烟要验的是 PDF 的渲染 + OCR。
    plan = [entry for entry in oi._pending()
            if (entry.get("extension") or Path(entry["rel"]).suffix).lower() == ".pdf"]
    if needle:
        #: 指定子串时挑**页数最多**的一份：期刊的双栏/三栏版式出现在厚刊里
        plan = [entry for entry in plan if needle in entry["path"] or needle in entry["rel"]]
        if not plan:
            print(f"待做队列里没有匹配「{needle}」的文件")
            return 0
        item = max(plan, key=lambda entry: entry.get("pages") or 0)
    else:
        #: 不给子串时挑页数最少的一份：冒烟只需要"像样的版式"，不需要 500 页的巨著
        if not plan:
            print("待做队列为空，无可抽样（不写库，退出）")
            return 0
        item = min(plan, key=lambda entry: entry.get("pages") or 10 ** 9)
    local = oi._local(item["path"])
    if not local or not Path(local).exists():
        print(f"文件不在了：{item['path']}")
        return 1

    print(f"抽样：{item['rel']}")
    print(f"      盘上 {item.get('pages')} 页 · {Path(local).stat().st_size / 1e6:.1f} MB")
    ocr = oi._new_engine()
    with contextlib.closing(oi._pdf_open(local)) as document:
        #: 直接走 ``oi`` 自己的渲染口（而不是在这儿另写一份 frombuffer）：冒烟要验的正是
        #: 「产品代码那条渲染路」—— 字节序、dpi 换算、通道数都在那里面，另写一遍就白测了。
        for page_no in range(1, min(2, len(document)) + 1):
            page = document[page_no - 1]
            image = oi._pdf_page_image(page, 150)

            lines, elapsed = ocr(image)
            report = ol.analyze(lines or [], image.shape[1])
            text, note = oi._page_text(image, ocr, page_no=page_no)

            replaced = text.count("\ufffd")
            print(f"\n  第 {page_no} 页：{len(lines or [])} 行 · {elapsed * 1000:.0f} ms · "
                  f"{len(text)} 字 · 分栏={report['columns']} 锚点={report['anchors']} "
                  f"重排={'是' if report['order'] else '否'} · 替换字符={replaced}")
            if note:
                print(f"  VLM：{note}")
            print(f"  首 120 字：{text[:120]!r}")

            if not text.strip():
                print("  ✗ 冒烟失败：这页没出字")
                return 1
    print("\n冒烟通过：真实链路（含 numpy 框、分栏判定）无异常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
